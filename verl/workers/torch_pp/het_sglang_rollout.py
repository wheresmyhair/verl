"""SGLangRollout subclass for heterogeneous TP groups.

Overrides _init_distributed_env to use a pre-built FakeDeviceMesh
instead of the collective init_device_mesh("cpu", ...).

Uses an HTTP-backed SGLang helper process on TP leader ranks so the
parent Ray worker does not need group-wide CUDA visibility.
"""

from __future__ import annotations

import logging
import os

import torch.distributed as dist

from verl.utils.device import get_visible_devices_keyword
from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

logger = logging.getLogger(__name__)

# verl's distributed env vars that SGLang children must NOT inherit
_DIST_ENV_VARS = (
    "MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE",
    "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE",
)


class HetSGLangRollout(SGLangRollout):
    """SGLangRollout that accepts a pre-built CPU device mesh for het TP.

    The TP leader launches a helper server process with group-scoped
    CUDA_VISIBLE_DEVICES. TP followers keep _engine=None and participate
    only in TP-group collectives/broadcasts.
    """

    def __init__(self, config, model_config, device_mesh, device_mesh_cpu, tp_groups=None):
        # Store BEFORE super().__init__ which calls _init_distributed_env
        self._het_device_mesh_cpu = device_mesh_cpu
        self._tp_groups = tp_groups
        super().__init__(config, model_config, device_mesh)

    def _init_distributed_env(self, device_mesh_cpu=None, **kwargs):
        """Use pre-built FakeDeviceMesh, skip collective init_device_mesh."""
        self._device_mesh_cpu = self._het_device_mesh_cpu

        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
        self.tensor_parallel_size = self._device_mesh_cpu["tp"].size()
        self.train_tp = None

        self._rank = dist.get_rank()
        self._tp_rank = self._device_mesh_cpu["tp"].get_local_rank()
        self._tp_size = self._device_mesh_cpu["tp"].size()
        tp_group = self._device_mesh_cpu["tp"].get_group()
        if tp_group is None:
            raise RuntimeError("Het SGLang rollout requires a valid TP process group")

        logger.info(
            f"[Het SGLang] rank={self._rank} tp_rank={self._tp_rank} "
            f"tp_size={self._tp_size} tp_groups={self._tp_groups}"
        )

        # Gather visible devices within the TP group.
        # With RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1, CUDA_VISIBLE_DEVICES
        # may not be set. Each worker's assigned GPU is in LOCAL_RANK instead.
        devices_keyword = get_visible_devices_keyword()
        my_device = os.environ.get(
            devices_keyword,
            os.environ.get("LOCAL_RANK", "0"),
        )
        visible_devices = [None] * self._tp_size
        dist.all_gather_object(
            visible_devices,
            my_device,
            tp_group,
        )
        if any(device is None for device in visible_devices):
            raise RuntimeError(
                f"Failed to gather visible devices for het TP group: rank={self._rank}, "
                f"tp_rank={self._tp_rank}, tp_size={self._tp_size}, visible_devices={visible_devices}"
            )
        self.visible_devices_set = set(",".join(visible_devices).split(","))
        self._helper_visible_devices = ",".join(sorted(self.visible_devices_set, key=int))

    def _init_inference_engine(self, trust_remote_code, actor_module, port):
        """Launch an HTTP-backed SGLang helper on the TP leader only."""
        # Save and clear verl's distributed env vars so SGLang's spawned
        # children don't inherit them and conflict with verl's process group.
        saved = {}
        for key in _DIST_ENV_VARS:
            val = os.environ.pop(key, None)
            if val is not None:
                saved[key] = val

        # Use gloo for SGLang's internal TP process group init.
        # NCCL init_process_group from spawned children hangs on some GPUs
        # (e.g. Blackwell). Gloo works. SGLang's actual tensor ops use
        # custom CUDA allreduce, not the process group backend.
        os.environ["SGLANG_DIST_BACKEND"] = "gloo"

        devices_keyword = get_visible_devices_keyword()
        saved_devices = os.environ.get(devices_keyword, None)

        try:
            if self._tp_rank != 0:
                self._engine = None
                return

            os.environ[devices_keyword] = self._helper_visible_devices
            logger.info(
                "[Het SGLang] starting helper-backed inference engine init with "
                f"rank={self._rank} tp_rank={self._tp_rank} tp_size={self._tp_size} "
                f"visible_devices={self._helper_visible_devices}"
            )

            load_format = "dummy" if self.config.load_format.startswith("dummy") else self.config.load_format
            engine_kwargs = self.config.get("engine_kwargs", {}).get("sglang", {}) or {}
            engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
            attention_backend = engine_kwargs.pop("attention_backend", None)
            backend = attention_backend if attention_backend is not None else "fa3"
            max_running_requests = self.config.get("max_num_seqs", None)

            helper_timeout = max(float(self.config.server["timeout"]), 1800.0)

            args = {
                "model_path": actor_module,
                "dtype": self.config.dtype,
                "mem_fraction_static": self.config.gpu_memory_utilization,
                "enable_memory_saver": True,
                "base_gpu_id": 0,
                "gpu_id_step": 1,
                "tp_size": self._tp_size,
                "node_rank": 0,
                "load_format": load_format,
                "dist_init_addr": None,
                "nnodes": 1,
                "trust_remote_code": trust_remote_code,
                "max_running_requests": max_running_requests,
                "port": 30000 + self._rank,
                "log_level": "info",
                "mm_attention_backend": backend,
                "attention_backend": backend,
                "skip_tokenizer_init": self.config.skip_tokenizer_init,
                "dist_timeout": 1800,
                "first_rank_in_node": True,
                # HTTP helper requests should tolerate long rollout batches.
                # The direct AsyncEngine path does not have this extra client-side
                # timeout, so keep it aligned with the engine dist timeout.
                "timeout": helper_timeout,
                "max_attempts": self.config.server["max_attempts"],
                "retry_delay": self.config.server["retry_delay"],
                "max_connections": self.config.server["max_connections"],
                "max_start_wait_time": self.config.server["max_start_wait_time"],
            }
            args.update(engine_kwargs)
            self._engine = AsyncHttpServerAdapter(**args)
        except Exception as exc:
            raise RuntimeError(
                "Het SGLang inference engine init failed with "
                f"rank={self._rank} tp_rank={self._tp_rank} tp_size={self._tp_size} "
                f"visible_devices={self._helper_visible_devices}"
            ) from exc
        finally:
            # Restore env vars for verl's own use
            if saved_devices is None:
                os.environ.pop(devices_keyword, None)
            else:
                os.environ[devices_keyword] = saved_devices
            os.environ.update(saved)
            os.environ.pop("SGLANG_DIST_BACKEND", None)

    def __del__(self):
        engine = getattr(self, "_engine", None)
        if engine is not None and hasattr(engine, "shutdown"):
            try:
                engine.shutdown()
            except Exception:
                pass
