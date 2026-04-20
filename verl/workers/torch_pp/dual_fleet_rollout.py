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
import asyncio
import concurrent.futures

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


class _FanInEngineFacade:
    """async_generate-compatible facade that routes through the
    DynamicFanInOrchestrator.

    Matches the subset of sglang.Engine's async_generate API that verl's
    SGLangRollout actually uses in _batch_level_generate_sequences.
    Accepts a LIST of input_ids (one per batch item), runs orchestrator
    in a thread (since orchestrator is sync; we don't want to block the
    verl event loop). Returns the original engine's output format: a
    list of dicts with `text`, `meta_info`, `output_ids` etc.

    This deliberately side-steps the archived `_fanin_generate`-based
    path in sglang_rollout.py. It is a fresh integration.
    """

    def __init__(self, orchestrator, tokenizer_manager=None):
        from sglang.srt.utils.rlpipe_fan_in import RolloutRequest
        self._orch = orchestrator
        self._RolloutRequest = RolloutRequest
        # tokenizer_manager is an attribute some verl code reads; stub.
        self.tokenizer_manager = tokenizer_manager
        # For any attribute access verl might do that we don't wrap,
        # fall back to the underlying TP engine (most query-only APIs
        # like flush_cache, check_weights can route to TP).
        # Callers that need DP-fleet-specific behavior must go through
        # the orchestrator / coordinator directly.
        self._fallback_engine = None

    def set_fallback(self, engine):
        self._fallback_engine = engine

    def __getattr__(self, name):
        # Only called when normal lookup fails. Forward to fallback.
        if self._fallback_engine is not None:
            return getattr(self._fallback_engine, name)
        raise AttributeError(name)

    async def async_generate(
        self,
        prompt=None,
        sampling_params=None,
        input_ids=None,
        image_data=None,
        return_logprob=False,
        rid=None,
        **_ignored,
    ):
        """Main entry. If input_ids is a list-of-lists (batched), build
        one RolloutRequest per batch item and fan in via orchestrator.
        Return list of engine dicts.
        """
        if input_ids is None:
            # verl's non-batched code path — just one prompt.
            input_ids_list = None
            prompts = [prompt] if prompt is not None else None
        elif isinstance(input_ids[0], list):
            # Batched: list of token-id lists, one per request.
            input_ids_list = list(input_ids)
            prompts = None
        else:
            # Single request as flat list of ids.
            input_ids_list = [list(input_ids)]
            prompts = None

        n = len(input_ids_list) if input_ids_list else len(prompts)
        imgs = image_data if image_data is not None else [None] * n
        requests = [
            self._RolloutRequest(
                input_ids=(input_ids_list[i] if input_ids_list else None),
                prompt=(prompts[i] if prompts else None),
                sampling_params=dict(sampling_params or {}),
                image_data=imgs[i],
                return_logprob=return_logprob,
            )
            for i in range(n)
        ]

        # Run the orchestrator in a thread so we don't block the caller's
        # event loop (orchestrator uses ThreadPool + sync SGLang HTTP calls).
        loop = asyncio.get_event_loop()
        results, tel = await loop.run_in_executor(
            None, self._orch.rollout, requests
        )
        # Each RolloutResult carries the raw engine dict (meta_info,
        # output_ids, output_token_logprobs, etc). Return them in batch
        # order so callers unpacking engine-style lists get the right
        # thing.
        out_dicts = []
        for r in results:
            if r.raw is not None:
                out_dicts.append(r.raw)
            else:
                out_dicts.append({"text": r.text or "", "meta_info": {}})
        import logging
        logging.getLogger(__name__).info(
            f"[dual-fleet] rollout done: n={len(results)} swap={tel.swap_triggered} "
            f"dp={tel.n_finished_on_dp} tp={tel.n_finished_on_tp} "
            f"wall={tel.total_wall_s:.3f}s"
        )
        return out_dicts


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

        DP URLs are broadcast from rank 0 to all TP ranks so each rank
        can push weight updates to its own DP engine (rank i ↔ DP
        engine i, both on physical GPU i). This sidesteps needing a
        cross-GPU IPC handle transfer from rank 0.
        """
        # First: let the parent class launch the TP engine into self._engine.
        super()._init_inference_engine(trust_remote_code, actor_module, port)

        if self._tp_rank != 0:
            # Non-leader ranks have no engines of their own but still
            # receive the DP URL broadcast so they can push weight
            # updates to their own DP engine.
            self._dp_fleet: List[Any] = []
            self._fanin_coord = None
            self._fanin_orch = None
            self._receive_dp_urls_broadcast()
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

            # auto_pause_inactive=False + memory_managed_externally=True:
            # verl's torch_pp_workers.rollout_mode already resumes
            # TP weights+kv_cache before rollout starts. If the
            # coordinator also called release_memory_occupation /
            # resume_memory_occupation on TP during swap, it would
            # hit a KeyError on an already-resident tag. So at rollout
            # time both fleets are resident (no HBM savings from the
            # swap) and the "swap" is a pure routing flip. Proper
            # HBM management is a follow-up.
            self._fanin_coord = DualFleetCoordinator(
                dp_engines=self._dp_fleet,
                tp_engines=[self._engine],
                initial_active="dp",
                auto_pause_inactive=False,
                memory_managed_externally=True,
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

            # Broadcast DP URLs to all TP ranks so each rank has a
            # handle for its own DP engine weight updates.
            self._broadcast_dp_urls()

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

    # ---------- generate override ----------

    def _batch_level_generate_sequences(self, prompts, **kwargs):
        """Route through the fan-in orchestrator on TP leader; non-leader
        ranks fall through to the parent's broadcast-receive path.

        Implementation: temporarily swap self._engine with the facade so
        the parent class's logic (prompt-preprocessing, sampling-params
        construction, result assembly, broadcast) runs unchanged, but the
        engine call is intercepted by the facade and routed through the
        orchestrator.
        """
        if self._tp_rank != 0:
            # Non-leader: no engine, just receive broadcast.
            return super()._batch_level_generate_sequences(prompts, **kwargs)

        if self._fanin_orch is None:
            return super()._batch_level_generate_sequences(prompts, **kwargs)

        saved_engine = self._engine
        facade = _FanInEngineFacade(self._fanin_orch)
        facade.set_fallback(saved_engine)
        self._engine = facade
        try:
            return super()._batch_level_generate_sequences(prompts, **kwargs)
        finally:
            self._engine = saved_engine

    # ---------- DP URL broadcast (called from _init_inference_engine) ----------

    def _broadcast_dp_urls(self):
        """Rank 0: collect URLs of the DP fleet and broadcast to all TP
        ranks. Each rank then stores self._my_dp_url pointing to "its"
        DP engine (rank i ↔ DP engine i, both on physical GPU i)."""
        urls = [
            f"http://{eng.server_args.host}:{eng.server_args.port}"
            for eng in self._dp_fleet
        ]
        obj_list = [urls]
        tp_group = self._device_mesh_cpu["tp"].get_group()
        src = dist.distributed_c10d.get_global_rank(tp_group, 0)
        dist.broadcast_object_list(obj_list, src=src, group=tp_group)
        self._dp_urls: List[str] = obj_list[0]
        self._my_dp_url: Optional[str] = (
            self._dp_urls[self._tp_rank] if self._dp_urls else None
        )

    def _receive_dp_urls_broadcast(self):
        """Non-leader ranks: receive the DP URL list from rank 0."""
        obj_list: List[Any] = [None]
        tp_group = self._device_mesh_cpu["tp"].get_group()
        src = dist.distributed_c10d.get_global_rank(tp_group, 0)
        dist.broadcast_object_list(obj_list, src=src, group=tp_group)
        self._dp_urls = obj_list[0] or []
        self._my_dp_url = (
            self._dp_urls[self._tp_rank] if self._dp_urls else None
        )

    # ---------- update_weights override ----------

    async def update_weights(self, weights, **kwargs):
        """Update weights on BOTH the TP engine and each DP engine.

        Iteration structure: `weights` is a generator from
        torch_pp's `_collect_full_state_dict` that yields each param
        tensor after a cross-rank `dist.broadcast` — so it MUST be
        consumed in lockstep across all TP ranks. We iterate it once,
        bucket-by-bucket, and for each bucket do:

          (a) TP update via the existing `sgl_update_weights` path:
              all ranks gather_object, then rank 0's TP engine does
              `update_weights_from_tensor`.

          (b) DP update: each rank pushes its bucket to its own DP
              engine. Since `_collect_full_state_dict` broadcasts each
              full tensor to every rank's GPU, each rank has a local
              copy to serialize via CUDA IPC — no extra collective
              needed. Rank i ↔ DP engine i, both on physical GPU i.
        """
        from sglang.srt.utils.common import MultiprocessingSerializer
        from sglang.srt.model_executor.model_runner import LocalSerializedTensor
        from sglang.srt.weight_sync.utils import (
            _preprocess_tensor_for_update_weights,
            update_weights as sgl_update_weights,
        )
        from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
        from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets

        bucket_bytes = int(self.config.update_weights_bucket_megabytes) << 20

        for bucket in get_named_tensor_buckets(weights, bucket_bytes):
            # (a) TP update — all ranks participate via gather_object.
            await sgl_update_weights(
                engine=self._engine,
                params_batch=bucket,
                device_mesh_key="infer_tp",
                device_mesh=self.device_mesh,
            )

            # (b) DP update on each rank, in parallel (this rank → its DP engine).
            await self._update_own_dp_engine(bucket)

        if self._tp_rank == 0 and self._engine is not None:
            await self._engine.flush_cache()
            # Flush each DP engine too so stale KV from any earlier
            # prefill doesn't leak into the fresh-weights rollout.
            for eng in self._dp_fleet:
                await eng.flush_cache()

    async def _update_own_dp_engine(self, bucket):
        """Serialize this rank's tensors for its bucket and push to
        this rank's DP engine via HTTP POST /update_weights_from_tensor.

        Safe no-op if this rank has no DP URL (shouldn't happen once
        _broadcast_dp_urls has run, but defensive)."""
        if not getattr(self, "_my_dp_url", None):
            return
        import base64
        import aiohttp
        from sglang.srt.utils.common import MultiprocessingSerializer
        from sglang.srt.model_executor.model_runner import LocalSerializedTensor
        from sglang.srt.weight_sync.utils import _preprocess_tensor_for_update_weights

        named_tensors = [
            (
                name,
                LocalSerializedTensor(values=[
                    MultiprocessingSerializer.serialize(
                        _preprocess_tensor_for_update_weights(t.detach())
                    )
                ]),
            )
            for name, t in bucket
        ]
        blob = MultiprocessingSerializer.serialize(named_tensors)
        body = {
            "serialized_named_tensors": [base64.b64encode(blob).decode("utf-8")],
            "load_format": None,
            "flush_cache": False,
        }
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self._my_dp_url}/update_weights_from_tensor",
                json=body,
            ) as resp:
                if resp.status >= 400:
                    txt = await resp.text()
                    raise RuntimeError(
                        f"DP update_weights_from_tensor failed on rank "
                        f"{self._tp_rank} (url={self._my_dp_url}): "
                        f"status={resp.status} body={txt[:500]}"
                    )

    def __del__(self):
        for eng in getattr(self, "_dp_fleet", []) or []:
            try:
                eng.shutdown()
            except Exception:
                pass
        super().__del__()
