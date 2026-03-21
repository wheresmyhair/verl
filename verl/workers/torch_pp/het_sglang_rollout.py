"""SGLangRollout subclass for heterogeneous TP groups.

Overrides _init_distributed_env to use a pre-built FakeDeviceMesh
instead of the collective init_device_mesh("cpu", ...).

Overrides _init_inference_engine to clear verl's distributed env vars
before SGLang spawns child processes, preventing NCCL init conflicts.
"""

from __future__ import annotations

import logging
import os

import torch.distributed as dist

from verl.utils.device import get_visible_devices_keyword
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

logger = logging.getLogger(__name__)

# verl's distributed env vars that SGLang children must NOT inherit
_DIST_ENV_VARS = (
    "MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE",
    "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE",
)


class HetSGLangRollout(SGLangRollout):
    """SGLangRollout that accepts a pre-built CPU device mesh for het TP.

    Requires RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 so that each
    worker sees all GPUs and SGLang can properly spawn TP child processes.
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
        os.environ[devices_keyword] = ",".join(
            sorted(self.visible_devices_set, key=int)
        )

    def _init_inference_engine(self, trust_remote_code, actor_module, port):
        """Wrap parent's _init_inference_engine with env var cleanup.

        SGLang spawns child processes via mp.Process(start_method='spawn').
        Children inherit env vars. verl's MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE
        would cause the children's init_process_group to try joining verl's group
        instead of SGLang's internal TCP store. We temporarily clear them.
        """
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

        try:
            logger.info(
                "[Het SGLang] starting inference engine init with "
                f"rank={self._rank} tp_rank={self._tp_rank} tp_size={self._tp_size} "
                f"visible_devices={sorted(self.visible_devices_set, key=int)}"
            )
            super()._init_inference_engine(trust_remote_code, actor_module, port)
        except Exception as exc:
            raise RuntimeError(
                "Het SGLang inference engine init failed with "
                f"rank={self._rank} tp_rank={self._tp_rank} tp_size={self._tp_size} "
                f"visible_devices={sorted(self.visible_devices_set, key=int)}"
            ) from exc
        finally:
            # Restore env vars for verl's own use
            os.environ.update(saved)
            os.environ.pop("SGLANG_DIST_BACKEND", None)
