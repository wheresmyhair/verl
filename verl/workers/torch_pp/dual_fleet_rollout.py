"""Dual-fleet fan-in rollout for verl torch_pp.

Extends HetSGLangRollout. On the TP leader (rank 0 with tp_groups=
[[0,1,2,3]]), launches TWO fleets of SGLang HTTP servers:

  - TP fleet: 1 server, tp_size=4, spans all 4 GPUs.
    This is the "fast-per-token" fleet used when few requests remain.
  - DP fleet: 4 servers, tp_size=1 each, one per GPU.
    This is the "many-parallel-workers" fleet used during bulk rollout.

Both fleets share GPUs. A DualFleetCoordinator manages release/resume so
only one fleet's weights are resident at a time (the other sits in CPU
backup). A DynamicFanInOrchestrator monitors in-flight count during
bulk DP rollout and swaps to TP when enough workers go idle; stragglers
get aborted on DP and re-prefilled on TP.

Non-leader TP ranks have self._engine = None and wait on broadcasts
from the leader — same pattern as HetSGLangRollout for TP-only.

Config flags:
  - actor_rollout_ref.rollout.enable_dual_fleet_fanin: bool (default
    False). When True, this class is used via the rollout registry.
  - actor_rollout_ref.rollout.fanin_idle_threshold: int (default 2).
    Swap DP→TP when this many DP workers have finished their request.
"""
from __future__ import annotations

import logging
import os
import socket
import time
from typing import Any, Dict, List, Optional

import torch.distributed as dist

from verl.utils.device import get_visible_devices_keyword
from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter
from verl.workers.torch_pp.het_sglang_rollout import HetSGLangRollout, _DIST_ENV_VARS, _needs_p2p_workaround
from verl.workers.rollout.sglang_rollout import http_server_engine as _engine_mod

logger = logging.getLogger(__name__)


def _pick_free_port(start: int, end: int = 65535) -> int:
    for p in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"No free port in [{start}, {end})")


class DualFleetFanInRollout(HetSGLangRollout):
    """Dual-fleet fan-in rollout. Requires tp_groups=[[0,1,2,3]] (single
    TP group, rank 0 is leader). Non-leader ranks are passive."""

    def _init_inference_engine(self, trust_remote_code, actor_module, port):
        """Override: launch TP engine (via super) + DP fleet (here only).

        TP engine: one HTTP server with tp_size=N spanning all N GPUs.
        DP fleet: N HTTP servers, each tp_size=1 on its own GPU.

        Only the TP leader launches anything. Other ranks participate in
        the TP engine through its internal scheduler subprocesses — they
        don't get their own Python-side handle.
        """
        # First: let the parent class launch the TP engine into self._engine.
        super()._init_inference_engine(trust_remote_code, actor_module, port)

        if self._tp_rank != 0:
            # Non-leader ranks have no engines. They'll receive broadcasts.
            self._dp_fleet: List[Any] = []
            self._fanin_coord = None
            self._fanin_orch = None
            return

        # TP leader: launch 4 DP engines, one per GPU.
        dp_size = self._tp_size
        logger.info(f"[DualFleet] launching {dp_size} DP HTTP servers…")

        # Save and clear verl's distributed env vars so DP server children
        # don't inherit them (same reason as the TP launch in parent class).
        saved = {}
        for key in _DIST_ENV_VARS:
            val = os.environ.pop(key, None)
            if val is not None:
                saved[key] = val

        # CUDA visibility: DP servers need access to individual GPUs within
        # the TP group's combined device set. _helper_visible_devices is
        # "0,1,2,3" on our setup.
        devices_keyword = get_visible_devices_keyword()
        saved_devices = os.environ.get(devices_keyword, None)
        os.environ[devices_keyword] = self._helper_visible_devices

        # Patch NCCL_P2P_DISABLE for DP servers if needed (same logic as
        # parent class). DP has tp_size=1 so no NCCL but keep for safety.
        _orig_launch = None
        _disable_p2p = self._tp_size > 1 and _needs_p2p_workaround()
        if _disable_p2p:
            _orig_launch = _engine_mod.launch_server
            def _launch_with_p2p_disable(server_args):
                os.environ["NCCL_P2P_DISABLE"] = "1"
                return _orig_launch(server_args)
            _engine_mod.launch_server = _launch_with_p2p_disable

        try:
            load_format = (
                "dummy" if self.config.load_format.startswith("dummy")
                else self.config.load_format
            )
            engine_kwargs = self.config.get("engine_kwargs", {}).get("sglang", {}) or {}
            engine_kwargs = {k: v for k, v in engine_kwargs.items() if v is not None}
            attention_backend = engine_kwargs.pop("attention_backend", None)
            backend = attention_backend if attention_backend is not None else "fa3"
            max_running_requests = self.config.get("max_num_seqs", None)
            helper_timeout = max(float(self.config.server["timeout"]), 1800.0)

            self._dp_fleet = []
            base_port_dp = 41000
            base_nccl_port_dp = 37000
            for i in range(dp_size):
                args = {
                    "model_path": actor_module,
                    "dtype": self.config.dtype,
                    "mem_fraction_static": self.config.gpu_memory_utilization,
                    "enable_memory_saver": True,
                    "enable_weights_cpu_backup": True,
                    "base_gpu_id": i,
                    "gpu_id_step": 1,
                    "tp_size": 1,
                    "node_rank": 0,
                    "load_format": load_format,
                    "dist_init_addr": None,
                    "nnodes": 1,
                    "trust_remote_code": trust_remote_code,
                    "max_running_requests": max_running_requests,
                    "port": _pick_free_port(base_port_dp + i),
                    "nccl_port": _pick_free_port(base_nccl_port_dp + i),
                    "log_level": "info",
                    "mm_attention_backend": backend,
                    "attention_backend": backend,
                    "skip_tokenizer_init": self.config.skip_tokenizer_init,
                    "dist_timeout": 1800,
                    "disable_custom_all_reduce": True,
                    "first_rank_in_node": True,
                    "timeout": helper_timeout,
                    "max_attempts": self.config.server["max_attempts"],
                    "retry_delay": self.config.server["retry_delay"],
                    "max_connections": self.config.server["max_connections"],
                    "max_start_wait_time": self.config.server["max_start_wait_time"],
                }
                args.update(engine_kwargs)
                logger.info(
                    f"[DualFleet] launching DP engine {i}: "
                    f"base_gpu_id={i} port={args['port']} nccl_port={args['nccl_port']}"
                )
                eng = AsyncHttpServerAdapter(**args)
                self._dp_fleet.append(eng)

            logger.info(f"[DualFleet] all {dp_size} DP engines launched")

            # Build coordinator + orchestrator on the leader.
            from sglang.srt.utils.rlpipe_dual_fleet import DualFleetCoordinator
            from sglang.srt.utils.rlpipe_fan_in import DynamicFanInOrchestrator

            self._fanin_coord = DualFleetCoordinator(
                dp_engines=self._dp_fleet,
                tp_engines=[self._engine],
                initial_active="dp",
                auto_pause_inactive=True,  # pause TP at boot so DP is active
            )

            idle_threshold = int(
                os.environ.get(
                    "VERL_RLPIPE_FANIN_IDLE_THRESHOLD",
                    str(max(1, dp_size // 2)),
                )
            )
            self._fanin_orch = DynamicFanInOrchestrator(
                coordinator=self._fanin_coord,
                idle_dp_threshold=idle_threshold,
                swap_back_after=True,
            )
            logger.info(
                f"[DualFleet] coordinator + orchestrator ready "
                f"(idle_threshold={idle_threshold})"
            )

        except Exception as exc:
            raise RuntimeError(
                f"DualFleet DP fleet init failed: rank={self._rank}"
            ) from exc
        finally:
            if saved_devices is None:
                os.environ.pop(devices_keyword, None)
            else:
                os.environ[devices_keyword] = saved_devices
            os.environ.update(saved)
            if _orig_launch is not None:
                _engine_mod.launch_server = _orig_launch

    def __del__(self):
        for eng in getattr(self, "_dp_fleet", []) or []:
            try:
                eng.shutdown()
            except Exception:
                pass
        super().__del__()
