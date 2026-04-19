"""
PP bubble tracer — generates Chrome/Perfetto trace JSON for pipeline schedules.

Produces a multi-row Perfetto visualization:
  1. Meta bar: high-level phases (rollout, inference, ref, training)
  2. Per-GPU bars: 1F1B operations (forward, backward, send, recv)
  3. HBM counters: per-GPU memory usage over time
  4. GPU-util counters: NVML utilization / memory util, per-rank

GPU utilization polling is started in begin_step and stopped in end_step
so samples land inside the trace window. Configure via env var
RLPIPE_GPU_UTIL_POLL_MS (default 100, 0 disables).

Usage:
    tracer = PPTracer(pp_rank=rank, pp_size=4, save_dir="/path/to/traces")
    tracer.begin_step(step=42)

    # Meta-level phase timing
    with tracer.phase("rollout"):
        ...
    with tracer.phase("compute_log_prob"):
        ...

    # Detailed PP operations within a phase
    with tracer.trace("train_forward", micro_batch_id=0):
        stage.forward_step(0, ...)

    tracer.record_hbm()  # snapshot current HBM
    tracer.end_step()     # save trace file

Open the JSON in chrome://tracing or https://ui.perfetto.dev
"""

import json
import os
import threading
import time
from typing import Dict, List, Optional

import torch


# Color mapping for different operation types
_CATEGORY_COLORS = {
    # Phases (meta bar)
    "load_rollout": "rail_idle",          # blue-gray
    "rollout": "cq_build_passed",         # green
    "unload_rollout": "rail_idle",        # blue-gray
    "load_inference": "rail_idle",        # blue-gray
    "compute_log_prob": "rail_response",  # blue
    "unload_inference": "rail_idle",      # blue-gray
    "load_ref": "rail_idle",              # blue-gray
    "compute_ref_log_prob": "olive",      # olive
    "unload_ref": "rail_idle",            # blue-gray
    "load_training": "rail_idle",         # blue-gray
    "update_actor": "bad",                # red
    "fused_update_actor": "bad",          # red
    "unload_training": "rail_idle",       # blue-gray
    # PP operations
    "train_forward": "good",              # green
    "train_backward": "terrible",         # dark red
    "infer_forward": "olive",             # olive
    "infer_log_probs": "olive",           # olive (same family as infer)
    "p2p_send": "rail_animation",         # orange
    "p2p_recv": "rail_idle",              # blue-gray
    "loss": "rail_response",              # blue
    "optimizer": "generic_work",          # teal
}


class _GpuUtilPoller:
    """Background thread that polls NVML and appends Perfetto counter events.

    Owned by a PPTracer. Thread-safety: relies on CPython's atomic
    list.append; _ts_us reads _step_start/_time_offset_us which the main
    thread writes only in begin_step *before* the poller starts.

    Device identity assumes CUDA_VISIBLE_DEVICES is identity-mapped — we
    resolve NVML handles by torch.cuda.current_device(). If the env
    remaps devices (rare in worker processes), pass the NVML index
    explicitly via gpu_util_device_index on PPTracer.
    """

    def __init__(self, tracer: "PPTracer", interval_ms: int, device_index: Optional[int] = None):
        self.tracer = tracer
        self.interval_s = interval_ms / 1000.0
        self._device_index = device_index
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._backend: Optional[str] = None   # "nvml" | "smi"
        self._handle = None
        self._pynvml = None

    def _resolve_device(self) -> int:
        if self._device_index is not None:
            return self._device_index
        if torch.cuda.is_available():
            return torch.cuda.current_device()
        return 0

    def _init_nvml(self) -> bool:
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._resolve_device())
            self._pynvml = pynvml
            self._backend = "nvml"
            return True
        except Exception:
            return False

    def _sample_nvml(self) -> Optional[Dict]:
        try:
            u = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            m = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return {
                "utilization_pct": int(u.gpu),
                "mem_util_pct": int(u.memory),
                "mem_used_mb": int(m.used // (1024 * 1024)),
            }
        except Exception:
            return None

    def _sample_smi(self) -> Optional[Dict]:
        import subprocess
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={self._resolve_device()}",
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                timeout=1.0,
            ).decode().strip()
            fields = [f.strip() for f in out.split(",")]
            return {
                "utilization_pct": int(fields[0]),
                "mem_util_pct": int(fields[1]),
                "mem_used_mb": int(fields[2]),
            }
        except Exception:
            return None

    def _loop(self):
        sample_fn = self._sample_nvml if self._backend == "nvml" else self._sample_smi
        while not self._stop.is_set():
            t = time.perf_counter()
            s = sample_fn()
            if s is not None and self.tracer.enabled:
                self.tracer.events.append({
                    "name": "gpu_util",
                    "cat": "gpu_util",
                    "ph": "C",
                    "ts": self.tracer._ts_us(t),
                    "pid": f"6 GPU util GPU {self.tracer.pp_rank}",
                    "tid": "nvml",
                    "args": s,
                })
            self._stop.wait(self.interval_s)

    def start(self):
        if self.interval_s <= 0 or self._thread is not None:
            return
        if not self._init_nvml():
            import shutil
            if shutil.which("nvidia-smi") is None:
                return
            self._backend = "smi"
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gpu_util_poller")
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        if self._backend == "nvml" and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
            self._handle = None


class PPTracer:
    """Collects timestamped events for one PP rank across an entire step."""

    def __init__(
        self,
        pp_rank: int,
        pp_size: int,
        enabled: bool = True,
        save_dir: str = "/tmp/pp_traces",
        gpu_util_poll_ms: Optional[int] = None,
        gpu_util_device_index: Optional[int] = None,
    ):
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.enabled = enabled
        self.save_dir = save_dir
        self.events: List[Dict] = []
        self._step_start: Optional[float] = None
        self._step: int = 0
        self._time_offset_us: float = 0.0

        if gpu_util_poll_ms is None:
            gpu_util_poll_ms = int(os.environ.get("RLPIPE_GPU_UTIL_POLL_MS", "100"))
        self._gpu_util_poller = (
            _GpuUtilPoller(self, gpu_util_poll_ms, gpu_util_device_index)
            if gpu_util_poll_ms > 0
            else None
        )

    def begin_step(self, step: int, time_offset_us: float = 0.0):
        """Start a new step — clears events and records base timestamp."""
        self.events.clear()
        self._step = step
        self._step_start = time.perf_counter()
        self._time_offset_us = time_offset_us
        if self._gpu_util_poller is not None and self.enabled:
            self._gpu_util_poller.start()

    def _ts_us(self, t: float) -> float:
        """Convert absolute time to microseconds relative to step start."""
        if self._step_start is None:
            self._step_start = t
        return self._time_offset_us + (t - self._step_start) * 1e6

    # ── Context managers ──

    class _TraceContext:
        def __init__(self, tracer: "PPTracer", name: str, cat: str, pid: str, tid: str, args: Dict):
            self.tracer = tracer
            self.name = name
            self.cat = cat
            self.pid = pid
            self.tid = tid
            self.args = args
            self.start = 0.0

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *exc):
            if not self.tracer.enabled:
                return
            end = time.perf_counter()
            ts = self.tracer._ts_us(self.start)
            dur = (end - self.start) * 1e6
            self.tracer.events.append({
                "name": self.name,
                "cat": self.cat,
                "ph": "X",
                "ts": ts,
                "dur": dur,
                "pid": self.pid,
                "tid": self.tid,
                "args": self.args,
                "cname": _CATEGORY_COLORS.get(self.cat, "generic_work"),
            })

    def phase(self, phase_name: str, **extra_args) -> _TraceContext:
        """Trace a high-level phase (rollout, inference, training).

        Each rank emits its own phase events on a per-rank row so you can
        see straggler ranks and sync gaps in the merged Perfetto view.
        """
        args = {"pp_rank": self.pp_rank, "step": self._step}
        args.update(extra_args)
        return self._TraceContext(
            self, phase_name, phase_name,
            pid=f"0 Phases GPU {self.pp_rank}",  # sorts first, one row per rank
            tid="phases",
            args=args,
        )

    def trace(self, category: str, micro_batch_id: int = -1, **extra_args) -> _TraceContext:
        """Trace a PP operation (forward, backward, send, recv).

        Shows up on the per-GPU bar in Perfetto.
        """
        name = category
        if micro_batch_id >= 0:
            name = f"{category} mb={micro_batch_id}"
        args = {"pp_rank": self.pp_rank, "micro_batch_id": micro_batch_id}
        args.update(extra_args)
        return self._TraceContext(
            self, name, category,
            pid=f"1 GPU {self.pp_rank}",  # sorts after Phases, before HBM
            tid="ops",
            args=args,
        )

    # ── HBM tracking ──

    def record_hbm(self, label: str = ""):
        """Record current GPU HBM usage as a counter event."""
        if not self.enabled:
            return
        try:
            allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        except Exception:
            return

        ts = self._ts_us(time.perf_counter())
        self.events.append({
            "name": "HBM (GB)",
            "cat": "memory",
            "ph": "C",  # counter event
            "ts": ts,
            "pid": f"5 HBM GPU {self.pp_rank}",  # separate process, sorts after GPU ops
            "tid": "memory",
            "args": {
                "allocated_gb": round(allocated_gb, 2),
                "reserved_gb": round(reserved_gb, 2),
            },
        })

    # ── Save / merge ──

    def end_step(self):
        """Save trace for this step."""
        if self._gpu_util_poller is not None:
            self._gpu_util_poller.stop()
        if not self.enabled or not self.events:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(
            self.save_dir,
            f"step{self._step}_rank{self.pp_rank}.json",
        )
        self.save(path)

    def save(self, path: str):
        """Save trace to a JSON file (Chrome/Perfetto format)."""
        existing_events = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    existing_events = loaded
            except Exception:
                existing_events = []
        with open(path, "w") as f:
            json.dump(existing_events + self.events, f)

    @staticmethod
    def merge_ranks(save_dir: str, step: int, pp_size: int) -> str:
        """Merge per-rank traces into a single file for Perfetto."""
        all_events = []
        for rank in range(pp_size):
            path = os.path.join(save_dir, f"step{step}_rank{rank}.json")
            if os.path.exists(path):
                with open(path) as f:
                    all_events.extend(json.load(f))
        merged_path = os.path.join(save_dir, f"step{step}_merged.json")
        with open(merged_path, "w") as f:
            json.dump(all_events, f)
        return merged_path

    def summary(self) -> Dict[str, float]:
        """Compute per-category total time and bubble fraction."""
        cat_totals: Dict[str, float] = {}
        for event in self.events:
            if event.get("ph") != "X":
                continue
            cat = event["cat"]
            dur_ms = event["dur"] / 1000
            cat_totals[cat] = cat_totals.get(cat, 0) + dur_ms

        total_ms = sum(cat_totals.values())
        compute_cats = {"train_forward", "train_backward", "infer_forward", "infer_log_probs", "loss", "optimizer"}
        comm_cats = {"p2p_send", "p2p_recv"}
        phase_cats = {
            "rollout", "compute_log_prob", "compute_ref_log_prob",
            "update_actor", "fused_update_actor",
            "load_rollout", "unload_rollout",
            "load_inference", "unload_inference",
            "load_ref", "unload_ref",
            "load_training", "unload_training",
        }

        compute_ms = sum(v for k, v in cat_totals.items() if k in compute_cats)
        comm_ms = sum(v for k, v in cat_totals.items() if k in comm_cats)
        # Don't count phase-level timers in bubble calc (they wrap everything)
        op_total_ms = sum(v for k, v in cat_totals.items() if k not in phase_cats)
        bubble_ms = max(0, op_total_ms - compute_ms - comm_ms)

        result = {f"trace/{k}_ms": v for k, v in cat_totals.items() if k not in phase_cats}
        result["trace/total_ms"] = op_total_ms
        result["trace/compute_ms"] = compute_ms
        result["trace/comm_ms"] = comm_ms
        result["trace/bubble_ms"] = bubble_ms
        if op_total_ms > 0:
            result["trace/bubble_fraction"] = bubble_ms / op_total_ms
            result["trace/compute_fraction"] = compute_ms / op_total_ms
        return result
