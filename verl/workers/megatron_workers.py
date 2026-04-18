# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The main entry point to run the PPO algorithm
"""

import datetime
import logging
import os
import time
from typing import Any, Optional

import psutil
import torch
import torch.distributed
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf

try:
    from mindspeed.megatron_adaptor import repatch
except ImportError:
    repatch = None

from megatron.core import parallel_state as mpu

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register, DYNAMIC_INDEX_DISPATCH
from verl.utils import hf_tokenizer
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.distributed import set_numa_affinity
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
    per_tensor_generator,
)
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import get_hf_model_path, load_mcore_dist_weights, load_megatron_gptmodel_weights
from verl.utils.profiler import (
    DistProfiler,
    DistProfilerExtension,
    GPUMemoryLogger,
    ProfilerConfig,
    log_gpu_memory_usage,
    simple_timer,
)
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.utils.ray_utils import get_event_loop
from verl.workers.actor.megatron_actor import MegatronPPOActor
from verl.workers.config import HFModelConfig, McoreCriticConfig, RolloutConfig
from verl.workers.critic.megatron_critic import MegatronPPOCritic
from verl.workers.reward_model.megatron.reward_model import MegatronRewardModel
from verl.workers.rollout import get_rollout_class
from verl.utils.addon import save_log_by_rank

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def set_random_seed(seed):
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if get_torch_device().device_count() > 0:
        from megatron.core import tensor_parallel

        tensor_parallel.model_parallel_cuda_manual_seed(seed)
    # FIXME: torch cumsum not support deterministic (used in vllm sampler),
    # https://github.com/pytorch/pytorch/issues/89492
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


class MegatronWorker(Worker):
    def _init_hf_config_and_tf_config(
        self,
        model_path,
        tokenizer_or_path,
        dtype,
        override_model_config,
        override_transformer_config,
        trust_remote_code=False,
        use_mbridge=False,
    ):
        from transformers import AutoConfig

        from verl.models.mcore import hf_to_mcore_config
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local
        from verl.utils.model import update_model_config

        # Step 1: initialize the tokenizer
        self.local_path = copy_to_local(model_path)
        if tokenizer_or_path is None:
            self.tokenizer = hf_tokenizer(self.local_path, trust_remote_code=trust_remote_code)
            self.processor = hf_processor(self.local_path, trust_remote_code=trust_remote_code)
        elif isinstance(tokenizer_or_path, str):
            self.tokenizer = hf_tokenizer(copy_to_local(tokenizer_or_path), trust_remote_code=trust_remote_code)
            self.processor = hf_processor(copy_to_local(tokenizer_or_path), trust_remote_code=trust_remote_code)
        else:
            self.tokenizer = tokenizer_or_path
            self.processor = tokenizer_or_path

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        # Step 2: get the hf
        hf_config = AutoConfig.from_pretrained(self.local_path, trust_remote_code=trust_remote_code)

        # Step 3: override the hf config
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config.get("model_config", {}))
        self.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)
        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)
        self.architectures = getattr(hf_config, "architectures", None)
        if self.rank == 0:
            print(f"Model config after override: {hf_config}")

        from verl.models.mcore.config_converter import mapping_string_to_attn_backend

        # todo: remove this line after mcore adopt mbridge 0.15, now for compatibility
        override_transformer_config = mapping_string_to_attn_backend(override_transformer_config)

        if use_mbridge:
            from verl.models.mcore.mbridge import AutoBridge

            bridge = AutoBridge.from_config(hf_config)
            bridge.set_extra_args(**override_transformer_config)
            tf_config = bridge.config
            self.bridge = bridge
        else:
            tf_config = hf_to_mcore_config(hf_config, dtype, **override_transformer_config)
            self.bridge = None

        print(f"TF config: {tf_config}")
        self.hf_config = hf_config
        self.tf_config = tf_config


class ActorRolloutRefWorker(MegatronWorker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)
        self.config = config
        if repatch is not None:
            # NPU MindSpeed patch, will be refactored with MindSpeedEngine.
            repatch(self.config.actor.megatron.get("override_transformer_config", {}))

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel strategy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            set_numa_affinity()
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            get_torch_device().set_device(rank)

            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.actor.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.actor.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.actor.megatron.virtual_pipeline_model_parallel_size,
                use_sharp=False,
                context_parallel_size=self.config.actor.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.actor.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.actor.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

        is_collect = (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            and mpu.get_context_parallel_rank() == 0
        )
        self._register_dispatch_collect_info(
            mesh_name="actor", dp_rank=mpu.get_data_parallel_rank(), is_collect=is_collect
        )

        set_random_seed(seed=self.config.actor.megatron.seed)

        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
            # This is for extendability in AsyncRL cases
            omega_profiler_config = config.rollout.get("profiler", {})
        elif self._is_ref:
            omega_profiler_config = config.ref.get("profiler", {})
        else:
            raise ValueError(
                f"Invalid role {self.role}, should be one of "
                "['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']"
            )
        # omega_profiler_config is DictConfig
        # profiler_config is a ProfilerConfig dataclass
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

        # TODO(sgm): Currently, we only support reference model param offload
        # will support other offload later
        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False

        # normalize config
        if self._is_actor and self._is_rollout:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
            if self.config.actor.get("ppo_micro_batch_size", None):
                self.config.actor.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.rollout.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size
                self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size

            self._is_offload_param = self.config.actor.megatron.get("param_offload", False)
            self._is_offload_grad = self.config.actor.megatron.get("grad_offload", False)
            self._is_offload_optimizer = self.config.actor.megatron.get("optimizer_offload", False)
        elif self._is_ref:
            if self.config.ref.get("log_prob_micro_batch_size", None):
                self.config.ref.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size
            else:
                assert self.config.ref.get("log_prob_micro_batch_size_per_gpu", None) is not None, (
                    "Please note that in the ref policy configuration, `log_prob_micro_batch_size_per_gpu` and "
                    "`log_prob_micro_batch_size` should not be None at the same time."
                )
            self._ref_is_offload_param = self.config.ref.megatron.get("param_offload", False)

    def _build_model_optimizer(
        self, model_path, optim_config, override_model_config, override_transformer_config, override_ddp_config=None
    ):
        from verl.utils.megatron.optimizer import (
            get_megatron_optimizer,
            get_megatron_optimizer_param_scheduler,
            init_megatron_optim_config,
        )
        from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module
        from verl.utils.model import get_generation_config, print_model_size

        self._init_hf_config_and_tf_config(
            model_path,
            model_path,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.config.model.get("trust_remote_code", False),
            self.config.actor.megatron.use_mbridge,
        )
        self.generation_config = get_generation_config(self.local_path)

        if self._is_actor or self._is_rollout:
            wrap_config = McoreModuleWrapperConfig(
                is_value_model=False,  # actor is not value model
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                wrap_with_ddp=True,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
            )
            actor_module = make_megatron_module(
                wrap_config=wrap_config,
                tf_config=self.tf_config,
                hf_config=self.hf_config,
                bridge=self.bridge,
                override_model_config=override_model_config,
                override_ddp_config=override_ddp_config,
            )
            print(f"actor_module: {len(actor_module)}")
            if self.config.actor.load_weight:
                if self.config.actor.megatron.use_dist_checkpointing:
                    load_mcore_dist_weights(
                        actor_module, self.config.actor.megatron.dist_checkpointing_path, is_value_model=False
                    )
                else:
                    if self.bridge is not None:
                        local_model_path = get_hf_model_path(self.config)
                        self.bridge.load_weights(actor_module, local_model_path)
                    else:
                        load_megatron_gptmodel_weights(
                            self.config, self.hf_config, actor_module, params_dtype=self.dtype, is_value_model=False
                        )

            if self.rank == 0:
                print_model_size(actor_module[0])
            log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)
        elif self._is_ref:
            wrap_config = McoreModuleWrapperConfig(
                is_value_model=False,  # ref is not value model
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                wrap_with_ddp=False,
                use_distributed_optimizer=self.config.ref.megatron.use_distributed_optimizer,
            )
            ref_module = make_megatron_module(
                wrap_config=wrap_config,
                tf_config=self.tf_config,
                hf_config=self.hf_config,
                bridge=self.bridge,
                override_model_config=override_model_config,
            )
            if self.config.ref.load_weight:  # should align with the actor:
                assert self.config.actor.load_weight == self.config.ref.load_weight
                print("load ref weight start")
                if self.config.ref.megatron.use_dist_checkpointing:
                    load_mcore_dist_weights(
                        ref_module, self.config.ref.megatron.dist_checkpointing_path, is_value_model=False
                    )
                else:
                    if self.bridge is not None:
                        local_model_path = get_hf_model_path(self.config)
                        self.bridge.load_weights(ref_module, local_model_path)
                    else:
                        load_megatron_gptmodel_weights(
                            self.config, self.hf_config, ref_module, params_dtype=self.dtype, is_value_model=False
                        )
            log_gpu_memory_usage("After ref module init", logger=logger)
            return ref_module, self.hf_config

        # TODO: add more optimizer args into config
        if self._is_actor:
            optim_config_megatron = init_megatron_optim_config(optim_config)
            actor_optimizer = get_megatron_optimizer(model=actor_module, config=optim_config_megatron)
            actor_optimizer_scheduler = get_megatron_optimizer_param_scheduler(
                optimizer=actor_optimizer, config=optim_config
            )
        else:
            optim_config = None
            actor_optimizer = None
            actor_optimizer_scheduler = None

        log_gpu_memory_usage("After actor optimizer init", logger=logger)

        return actor_module, actor_optimizer, actor_optimizer_scheduler, self.hf_config, optim_config

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # 1. parse rollout and huggingface model config
        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)

        # 2. build rollout device mesh
        infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
        infer_pp = self.config.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )
        rollout_device_mesh = init_device_mesh(
            get_device_name(), mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
        )

        is_collect = (
            rollout_device_mesh["infer_tp"].get_local_rank() == 0
            and rollout_device_mesh["infer_pp"].get_local_rank() == 0
        )
        self._register_dispatch_collect_info(
            "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
        )

        # 3. init trainer and rollout random states
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        # 4. build rollout model
        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )
        log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=logger)

        from verl.models.mcore import get_mcore_weight_converter

        self.layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }
        self.weight_converter = None
        if not self.bridge:
            self.weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)

        # 5. switch to trainer mode
        # NOTE: It's critical that hybrid engine in trainer mode initially to load checkpoint.
        # For sync mode, we directly switch to trainer mode here.
        # For async mode, we can't call run_until_complete here, so we will switch to trainer mode in AgentLoopManager.
        if rollout_config.mode == "sync" and self._is_actor:
            loop = get_event_loop()
            loop.run_until_complete(self.trainer_mode())

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        from verl.utils.torch_dtypes import PrecisionType

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        if self._is_actor:
            override_transformer_config = OmegaConf.to_container(
                OmegaConf.create(self.config.actor.megatron.get("override_transformer_config", {}))
            )
            override_ddp_config = OmegaConf.to_container(
                OmegaConf.create(self.config.actor.megatron.get("override_ddp_config", {}))
            )
        elif self._is_ref:
            override_transformer_config = OmegaConf.to_container(
                OmegaConf.create(self.config.ref.megatron.get("override_transformer_config", {}))
            )
        else:
            override_transformer_config = {}
        self.param_dtype = torch.bfloat16
        log_gpu_memory_usage("Before init actor model and optimizer", logger=logger)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            optim_config = self.config.actor.optim if self._is_actor else None
            (
                self.actor_module,
                self.actor_optimizer,
                self.actor_optimizer_scheduler,
                self.actor_model_config,
                self.actor_optim_config,
            ) = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=optim_config,
                override_model_config=override_model_config,
                override_transformer_config=override_transformer_config,
                override_ddp_config=override_ddp_config,
            )
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
                log_gpu_memory_usage("After offload actor params and grad during init", logger=logger)
            if self._is_offload_optimizer:
                offload_megatron_optimizer(self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = MegatronPPOActor(
                config=actor_cfg,
                model_config=self.actor_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.actor_module,
                actor_optimizer=self.actor_optimizer,
            )
            log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)

        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
            log_gpu_memory_usage("After rollout init", logger=logger)

        if self._is_ref:
            self.ref_module, self.ref_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=None,
                override_model_config=override_model_config,
                override_transformer_config=override_transformer_config,
            )
            log_gpu_memory_usage("After ref model init", logger=logger)
            self.ref_policy = MegatronPPOActor(
                config=self.config.ref,
                model_config=self.ref_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.ref_module,
                actor_optimizer=None,
            )
            if self._ref_is_offload_param:
                offload_megatron_model_to_cpu(self.ref_module)
                log_gpu_memory_usage("After offload ref params during init", logger=logger)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_mananager = MegatronCheckpointManager(
                config=self.config,
                checkpoint_config=self.config.actor.checkpoint,
                model_config=self.actor_model_config,
                transformer_config=self.tf_config,
                role="actor",
                model=self.actor_module,
                arch=self.architectures[0],
                hf_config=self.hf_config,
                param_dtype=self.param_dtype,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                optimizer=self.actor_optimizer,
                optimizer_scheduler=self.actor_optimizer_scheduler,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
                use_checkpoint_opt_param_scheduler=self.config.actor.optim.use_checkpoint_opt_param_scheduler,
                bridge=self.bridge,
                use_dist_checkpointing=self.config.actor.megatron.use_dist_checkpointing,
            )
        # rlpipe: Step-level PPTracer for Perfetto. Gated by env var
        # RLPIPE_MEGATRON_PP_TRACE=<save_dir>. Reuses the existing torch_pp
        # PPTracer (it's framework-agnostic, just an event collector).
        # Installs Megatron schedule monkey-patches so the 1F1B tF/tB ops
        # show up per micro-batch on a per-rank row, matching the torch_pp
        # trace layout.
        #
        # **Single tracer per process**: multiple worker sub-instances may
        # live in the same Ray worker process (e.g., actor_rollout and ref
        # under one PID). Each would call __init__. We stash the tracer +
        # step counter + backward counter in a process-wide module slot so
        # all instances share the same events list and the forward_step /
        # backward_step monkey-patches always write to the same place.
        self._pp_tracer = None
        self._pp_trace_step = 0
        self._backward_mb_counter = {"mb": 0}
        if os.environ.get("RLPIPE_MEGATRON_PP_TRACE"):
            try:
                import sys as _sys
                _slot_name = "_rlpipe_pp_trace_slot"
                _slot = _sys.modules.get(_slot_name)
                if _slot is None:
                    _slot = type(_sys)(_slot_name)
                    _sys.modules[_slot_name] = _slot
                if not hasattr(_slot, "tracer"):
                    from verl.workers.torch_pp.pp_trace import PPTracer as _PPTracer
                    _save_dir = os.environ["RLPIPE_MEGATRON_PP_TRACE"]
                    _slot.tracer = _PPTracer(
                        pp_rank=mpu.get_pipeline_model_parallel_rank(),
                        pp_size=mpu.get_pipeline_model_parallel_world_size(),
                        enabled=True,
                        save_dir=_save_dir,
                    )
                    _slot.step_counter = 0
                    _slot.backward_counter = {"mb": 0}
                # All worker instances in this process share the same slot.
                self._pp_tracer = _slot.tracer
                self._backward_mb_counter = _slot.backward_counter
                self._pp_trace_slot = _slot
                # Propagate to actor / ref so the fused forward path can
                # record per-chunk iF events.
                if self._is_actor and hasattr(self, "actor"):
                    self.actor._pp_tracer = self._pp_tracer
                if self._is_ref and hasattr(self, "ref_policy"):
                    self.ref_policy._pp_tracer = self._pp_tracer
                self._install_megatron_pp_trace_hooks()
                logger.info(
                    "[rlpipe pp-trace] enabled, save_dir=%s pp_rank=%d role=%s",
                    os.environ["RLPIPE_MEGATRON_PP_TRACE"],
                    self._pp_tracer.pp_rank,
                    self.role,
                )
            except Exception as _pp_err:
                logger.warning(
                    "[rlpipe pp-trace] failed to initialize: %s", _pp_err
                )
                self._pp_tracer = None

        get_torch_device().empty_cache()
        log_gpu_memory_usage("After init_model finish", logger=logger)

    def _install_megatron_pp_trace_hooks(self) -> None:
        """Monkey-patch Megatron's schedule `forward_step`/`backward_step` so
        each micro-batch emits a `trace("train_forward", mb=i)` / `trace(
        "train_backward", mb=i)` event on this rank's PP-ops row.

        The schedule module looks up ``forward_step`` / ``backward_step`` via
        its own module globals each call, so replacing them there is enough
        to intercept every call inside
        ``forward_backward_pipelining_without_interleaving``.

        `forward_step` is called with ``current_microbatch=i`` as a kwarg —
        we pull it directly. `backward_step` has no mb arg, so we use a
        per-rank counter that resets on every `begin_step`.

        **Process-level idempotence**: multiple worker sub-instances can
        live in the same Ray process (e.g., actor_rollout and ref under the
        same PID), each calling this init. Without a guard, the second
        install would wrap the already-wrapped function, producing **2×
        trace events per schedule call**. We detect already-patched
        functions via a sentinel attribute and replace the tracer ref on
        the existing wrapper instead of re-wrapping. The wrapper closes
        over a *process-wide* tracer/counter slot held in module globals.
        """
        try:
            from megatron.core.pipeline_parallel import schedules as _sched
        except ImportError:
            logger.warning("[rlpipe pp-trace] megatron.core.schedules unavailable; hooks not installed")
            return

        import sys as _sys
        # Use a process-wide slot so multiple workers in the same process
        # share a single patched wrapper + a single tracer pointer.
        _mod = _sys.modules.setdefault(
            "_rlpipe_pp_trace_slot",
            type(_sys)("_rlpipe_pp_trace_slot"),
        )
        if not hasattr(_mod, "active_tracer"):
            _mod.active_tracer = None
            _mod.backward_counter = None
        _mod.active_tracer = self._pp_tracer
        _mod.backward_counter = self._backward_mb_counter

        if getattr(_sched.forward_step, "_rlpipe_pp_traced", False):
            # Already patched by a previous worker init in this process.
            # Just leave the existing wrapper in place (its process-wide
            # slot has been updated above).
            return

        _orig_forward_step = _sched.forward_step
        _orig_backward_step = _sched.backward_step

        def _traced_forward_step(*args, **kwargs):
            tracer = _mod.active_tracer
            if tracer is None or not tracer.enabled:
                return _orig_forward_step(*args, **kwargs)
            _mb = kwargs.get("current_microbatch", -1)
            with tracer.trace("train_forward", micro_batch_id=_mb):
                return _orig_forward_step(*args, **kwargs)

        def _traced_backward_step(*args, **kwargs):
            tracer = _mod.active_tracer
            if tracer is None or not tracer.enabled:
                return _orig_backward_step(*args, **kwargs)
            bc = _mod.backward_counter
            _mb = bc["mb"] if bc is not None else -1
            if bc is not None:
                bc["mb"] += 1
            with tracer.trace("train_backward", micro_batch_id=_mb):
                return _orig_backward_step(*args, **kwargs)

        _traced_forward_step._rlpipe_pp_traced = True
        _traced_backward_step._rlpipe_pp_traced = True
        _sched.forward_step = _traced_forward_step
        _sched.backward_step = _traced_backward_step

        # Also patch P2PCommunicator's send/recv methods so the bubbles
        # between tF / tB show up as labeled p2p_send / p2p_recv bars on
        # the same GPU Ops row, matching the torch_pp trace layout.
        try:
            from megatron.core.pipeline_parallel import p2p_communication as _p2p
        except ImportError:
            return

        if getattr(_p2p.P2PCommunicator.send_forward, "_rlpipe_pp_traced", False):
            return  # already patched

        def _wrap_p2p(method_name: str, cat: str):
            _orig = getattr(_p2p.P2PCommunicator, method_name)
            def _wrapper(self, *args, **kwargs):
                tracer = _mod.active_tracer
                if tracer is None or not tracer.enabled:
                    return _orig(self, *args, **kwargs)
                with tracer.trace(cat):
                    return _orig(self, *args, **kwargs)
            _wrapper._rlpipe_pp_traced = True
            _wrapper.__name__ = f"_traced_{method_name}"
            setattr(_p2p.P2PCommunicator, method_name, _wrapper)

        # Forward-direction P2P
        _wrap_p2p("recv_forward", "p2p_recv")
        _wrap_p2p("send_forward", "p2p_send")
        # Backward-direction P2P
        _wrap_p2p("recv_backward", "p2p_recv")
        _wrap_p2p("send_backward", "p2p_send")
        # 1F1B combined ops — these are the bulk of the schedule's P2P
        _wrap_p2p("send_forward_recv_backward", "p2p_send_recv")
        _wrap_p2p("send_backward_recv_forward", "p2p_send_recv")

    def _pp_trace_begin_step_if_active(self) -> None:
        """Bump step counter + start a new tracer step. Safe no-op if
        tracer not configured. **Idempotent** — if a step is already in
        flight (``_step_start is not None``), this is a no-op so multiple
        entry points (generate_sequences, compute_ref_log_prob for
        ref-only workers) don't reset accumulated events.

        Uses the **process-wide slot's step counter** so multiple worker
        sub-instances in the same process advance the counter together
        and share one events list.

        Calls a `torch.distributed.barrier()` so all ranks line up their
        `time.perf_counter()` origins to within barrier completion
        (~sub-ms), giving the merged view roughly synchronized per-rank
        tracks.
        """
        if self._pp_tracer is None:
            return
        if self._pp_tracer._step_start is not None:
            return  # Already in a step — no-op
        # Increment shared counter (falls back to per-instance if no slot).
        _slot = getattr(self, "_pp_trace_slot", None)
        if _slot is not None:
            _slot.step_counter += 1
            _step = _slot.step_counter
        else:
            self._pp_trace_step += 1
            _step = self._pp_trace_step
        try:
            torch.distributed.barrier()
        except Exception:
            pass
        self._pp_tracer.begin_step(step=_step)
        # Reset the per-step backward mb counter (forward uses kwargs).
        self._backward_mb_counter["mb"] = 0

    def _pp_trace_end_step_if_active(self) -> None:
        """Flush accumulated events to ``step{N}_rank{R}.json``, clear the
        events list, and reset the step-start marker so the next
        ``begin_step_if_active`` can start a fresh step. No-op if tracer
        not configured.

        Clearing events after save is important when multiple worker sub-
        instances (e.g., colocated actor + ref) share a single step file:
        save uses an append-on-existing pattern, so without clearing,
        subsequent end_step calls on the same instance would double-write
        events. The sequence for a colocated actor+ref worker is:
          1. generate_sequences → begin_step (T0 set)
          2. events accumulate across compute_log_prob, compute_ref_log_prob
          3. update_actor → end_step → save events, clear, T0 → None
          4. Next step → begin_step starts fresh
        """
        if self._pp_tracer is None:
            return
        self._pp_tracer.end_step()
        self._pp_tracer.events.clear()
        self._pp_tracer._step_start = None

    async def rollout_mode(self):
        """Context switch hybridengine to rollout mode."""
        _timing = {}
        aggressive_empty_cache(force_sync=True)

        with simple_timer("rollout_mode/load_actor_to_gpu", _timing):
            if self._is_offload_param:
                load_megatron_model_to_gpu(self.actor.actor_module, load_grad=False)
                log_gpu_memory_usage("After load actor params during rollout_mode", logger=logger)

        with simple_timer("rollout_mode/export_weights", _timing):
            if self.bridge is not None:
                per_tensor_param = self.bridge.export_weights(self.actor.actor_module)
            else:
                per_tensor_param = per_tensor_generator(
                    self.actor.actor_module,
                    self.actor_model_config,
                    self.weight_converter,
                    self.tf_config,
                    self.layer_name_mapping,
                )

        set_expandable_segments(False)

        with simple_timer("rollout_mode/resume_weights", _timing):
            if self.config.rollout.free_cache_engine:
                await self.rollout.resume(tags=["weights"])
        with simple_timer("rollout_mode/update_weights", _timing):
            await self.rollout.update_weights(per_tensor_param)
        with simple_timer("rollout_mode/offload_actor", _timing):
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor.actor_module)
        aggressive_empty_cache(force_sync=True)
        with simple_timer("rollout_mode/resume_kv_cache", _timing):
            if self.config.rollout.free_cache_engine:
                await self.rollout.resume(tags=["kv_cache"])

        # important: need to manually set the random states of each tp to be identical.
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)

        if torch.distributed.get_rank() == 0:
            print(f"[PROFILING] rollout_mode sub-phases: {_timing}")

    async def trainer_mode(self):
        """Context switch hybridengine to trainer mode."""
        _timing = {}
        with simple_timer("trainer_mode/release_rollout", _timing):
            if self.config.rollout.free_cache_engine:
                log_gpu_memory_usage("Before rollout offload", logger=logger)
                await self.rollout.release()
                log_gpu_memory_usage("After rollout offload", logger=logger)

        with simple_timer("trainer_mode/set_train_and_cleanup", _timing):
            for model in self.actor.actor_module:
                model.train()
            # add empty cache after each compute
            aggressive_empty_cache(force_sync=True)

        # FIXME(@wuxibin): megatron+sglang failed with `expandable_segments:True` in ci,
        # can't reproduce it in dev environment, temporary disable it.
        # https://github.com/volcengine/verl/actions/runs/17382936845/job/49344264323?pr=3285
        if os.environ.get("MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS", "0") == "0":
            set_expandable_segments(True)

        # restore random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        if torch.distributed.get_rank() == 0:
            print(f"[PROFILING] trainer_mode sub-phases: {_timing}")

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="update_actor", logger=logger)
    @DistProfiler.annotate(color="red")
    def update_actor(self, data: DataProto):
        assert self._is_actor
        timing_update = {}
        _tracer = self._pp_tracer
        _ua_outer = _tracer.phase("update_actor") if _tracer else None
        if _ua_outer is not None:
            _ua_outer.__enter__()

        _lt_ctx = _tracer.phase("load_training") if _tracer else None
        if _lt_ctx is not None:
            _lt_ctx.__enter__()
        try:
            with simple_timer("load_training", timing_update):
                if self._is_offload_param:
                    load_megatron_model_to_gpu(self.actor_module)
                    log_gpu_memory_usage("After load actor params and grad during update_actor", logger=logger)
                if self._is_offload_optimizer:
                    load_megatron_optimizer(self.actor_optimizer)
                    log_gpu_memory_usage("After load actor optimizer during update_actor", logger=logger)
        finally:
            if _lt_ctx is not None:
                _lt_ctx.__exit__(None, None, None)
        if _tracer is not None:
            _tracer.record_hbm("update_actor/after_load_training")

        # Reset the backward-step mb counter so `tB mb=i` labels start from 0
        # for this step's 1F1B schedule (the forward path uses the explicit
        # `current_microbatch` kwarg from the schedule, so it's already correct).
        self._backward_mb_counter["mb"] = 0

        micro_batch_size = self.config.actor.ppo_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        dataloader = self.actor.make_minibatch_iterator(data=data)
        with Timer(name="update_policy", logger=None) as timer:
            metrics = self.actor.update_policy(dataloader=dataloader)
        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        metrics["actor/lr"] = get_megatron_last_lr(self.actor_optimizer)
        self.actor_optimizer_scheduler.step(1)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        _ut_ctx = _tracer.phase("unload_training") if _tracer else None
        if _ut_ctx is not None:
            _ut_ctx.__enter__()
        try:
            with simple_timer("unload_training", timing_update):
                if self._is_offload_param:
                    offload_megatron_model_to_cpu(self.actor_module)
                    log_gpu_memory_usage("After offload actor params and grad during update_actor", logger=logger)
                if self._is_offload_optimizer:
                    offload_megatron_optimizer(self.actor_optimizer)
                    log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)
        finally:
            if _ut_ctx is not None:
                _ut_ctx.__exit__(None, None, None)

        aggressive_empty_cache(force_sync=True)
        if _tracer is not None:
            _tracer.record_hbm("after_update_actor")
        if _ua_outer is not None:
            _ua_outer.__exit__(None, None, None)
        # End of RL step — flush this rank's trace file.
        self._pp_trace_end_step_if_active()

        timing_update = reduce_timing(timing_update)
        if torch.distributed.get_rank() == 0:
            print(f"[PROFILING] update_actor timing: {timing_update}")
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="fused_update_actor", logger=logger)
    @DistProfiler.annotate(color="red")
    def fused_update_actor(self, data: DataProto):
        """Fused forward: compute old_log_probs (HF inference) + training
        update in a single load/unload cycle.

        Merges the separate ``compute_log_prob`` and ``update_actor`` phases
        into one method. Savings come from eliminating the double Megatron
        model load (currently loaded once for compute_log_prob without grad,
        then again for update_actor with grad+optimizer).

        Flow:
            1. Load Megatron model + grad + optimizer to GPU
            2. Restore HF inference replica (if CPU-offloaded)
            3. Sync HF replica from current Megatron weights
            4. Run HF inference → old_log_probs
            5. Offload HF replica (free HBM for training backward)
            6. Inject old_log_probs into batch
            7. Megatron forward_backward_func (training)
            8. Optimizer step
            9. Unload everything

        The constraint "all iF before first tB" is trivially satisfied
        because HF inference is fully local (no PP) and finishes in step 4
        before Megatron training begins in step 7.
        """
        assert self._is_actor
        timing_fused = {}
        _tracer = self._pp_tracer

        _fua_outer = _tracer.phase("fused_update_actor") if _tracer else None
        if _fua_outer is not None:
            _fua_outer.__enter__()

        # ------------------------------------------------------------------
        # Phase 1: Load training state (Megatron model + optimizer)
        # ------------------------------------------------------------------
        _lt_ctx = _tracer.phase("load_training") if _tracer else None
        if _lt_ctx is not None:
            _lt_ctx.__enter__()
        try:
            with simple_timer("load_training", timing_fused):
                if self._is_offload_param:
                    load_megatron_model_to_gpu(self.actor_module)
                    log_gpu_memory_usage("After load actor params during fused_update_actor", logger=logger)
                if self._is_offload_optimizer:
                    load_megatron_optimizer(self.actor_optimizer)
                    log_gpu_memory_usage("After load actor optimizer during fused_update_actor", logger=logger)
        finally:
            if _lt_ctx is not None:
                _lt_ctx.__exit__(None, None, None)
        if _tracer is not None:
            _tracer.record_hbm("fused_update_actor/after_load_training")

        # ------------------------------------------------------------------
        # Phase 2: Restore + sync HF inference replica
        # ------------------------------------------------------------------
        _fused_active = (
            self.actor is not None
            and getattr(self.actor, "use_fused_forward_pp", False)
            and getattr(self.actor, "_fused_inference_stage", None) is not None
        )
        _reverse_pp_active = (
            self.actor is not None
            and getattr(self.actor, "_reverse_pp_infer_stage", None) is not None
        )
        _cpu_offload = os.environ.get("RLPIPE_FUSED_FORWARD_CPU_OFFLOAD", "0") == "1"

        if _fused_active and _cpu_offload:
            _rf_ctx = _tracer.phase("restore_fused_replica") if _tracer else None
            if _rf_ctx is not None:
                _rf_ctx.__enter__()
            try:
                with simple_timer("restore_fused_replica", timing_fused):
                    _device_id = torch.cuda.current_device()
                    self.actor._fused_inference_stage.to(f"cuda:{_device_id}")
            finally:
                if _rf_ctx is not None:
                    _rf_ctx.__exit__(None, None, None)

        if _fused_active:
            _sf_ctx = _tracer.phase("sync_fused_replica") if _tracer else None
            if _sf_ctx is not None:
                _sf_ctx.__enter__()
            try:
                with simple_timer("sync_fused_replica", timing_fused):
                    self._sync_fused_replica_from_megatron(
                        self.actor._fused_inference_stage
                    )
            finally:
                if _sf_ctx is not None:
                    _sf_ctx.__exit__(None, None, None)
        # V2: sync reverse-PP inference stage (PP-sharded, permanently on GPU)
        if _reverse_pp_active:
            _sf_ctx = _tracer.phase("sync_fused_replica") if _tracer else None
            if _sf_ctx is not None:
                _sf_ctx.__enter__()
            try:
                with simple_timer("sync_fused_replica", timing_fused):
                    self._sync_fused_replica_from_megatron(
                        self.actor._reverse_pp_infer_stage
                    )
            finally:
                if _sf_ctx is not None:
                    _sf_ctx.__exit__(None, None, None)
        if _tracer is not None:
            _tracer.record_hbm("fused_update_actor/after_sync_replica")

        # Offload HF replica to CPU after sync. The fused schedule loop
        # loads it on-demand for each iF op to avoid OOM (Megatron
        # training state + SGLang residual already use ~73 GB).
        if _fused_active:
            self.actor._fused_inference_stage.to("cpu")
            aggressive_empty_cache(force_sync=True)

        # ------------------------------------------------------------------
        # Phase 3: Fused forward-backward (iF interleaved in PP bubbles)
        # ------------------------------------------------------------------
        # Reset the backward-step mb counter for this step's schedule.
        self._backward_mb_counter["mb"] = 0

        micro_batch_size = self.config.actor.ppo_micro_batch_size_per_gpu
        data.meta_info["temperature"] = self.config.rollout.temperature

        _ffb_ctx = _tracer.phase("fused_forward_backward") if _tracer else None
        if _ffb_ctx is not None:
            _ffb_ctx.__enter__()
        try:
            with Timer(name="fused_forward_backward", logger=None) as timer:
                metrics, old_log_probs = self.actor.fused_forward_backward(
                    data=data,
                    micro_batch_size=micro_batch_size,
                )
        finally:
            if _ffb_ctx is not None:
                _ffb_ctx.__exit__(None, None, None)

        delta_time = timer.last
        global_num_tokens = data.meta_info.get("global_token_num", None)
        if global_num_tokens is not None:
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_num_tokens, delta_time
            )
            metrics["perf/mfu/actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)
        from verl.utils.megatron.optimizer import get_megatron_last_lr
        metrics["actor/lr"] = get_megatron_last_lr(self.actor_optimizer)
        self.actor_optimizer_scheduler.step(1)

        # ------------------------------------------------------------------
        # Phase 4: Build output with old_log_probs + metrics
        # ------------------------------------------------------------------
        # With reverse-PP, only the last-infer rank (rank 0) and last-train
        # rank (rank P-1) populate old_log_probs. Middle ranks return a
        # zero placeholder — the trainer picks the correct one.
        if old_log_probs is None:
            # Middle PP rank: create dummy of correct shape
            resp = data.batch["responses"]
            old_log_probs = torch.zeros(
                resp.shape[0], resp.shape[1], dtype=torch.float32
            )
        output = DataProto.from_dict(
            tensors={"old_log_probs": old_log_probs},
            meta_info={"metrics": metrics},
        )
        output = output.to("cpu")

        # ------------------------------------------------------------------
        # Phase 7: Unload training state
        # ------------------------------------------------------------------
        _ut_ctx = _tracer.phase("unload_training") if _tracer else None
        if _ut_ctx is not None:
            _ut_ctx.__enter__()
        try:
            with simple_timer("unload_training", timing_fused):
                if self._is_offload_param:
                    offload_megatron_model_to_cpu(self.actor_module)
                    log_gpu_memory_usage("After offload actor params during fused_update_actor", logger=logger)
                if self._is_offload_optimizer:
                    offload_megatron_optimizer(self.actor_optimizer)
                    log_gpu_memory_usage("After offload actor optimizer during fused_update_actor", logger=logger)
        finally:
            if _ut_ctx is not None:
                _ut_ctx.__exit__(None, None, None)

        aggressive_empty_cache(force_sync=True)
        if _tracer is not None:
            _tracer.record_hbm("after_fused_update_actor")
        if _fua_outer is not None:
            _fua_outer.__exit__(None, None, None)

        # End of RL step — flush this rank's trace file.
        self._pp_trace_end_step_if_active()

        timing_fused = reduce_timing(timing_fused)
        if torch.distributed.get_rank() == 0:
            print(f"[PROFILING] fused_update_actor timing: {timing_fused}")
        return output

    # @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @register(dispatch_mode=DYNAMIC_INDEX_DISPATCH)
    @GPUMemoryLogger(role="generate_sequences", logger=logger)
    @DistProfiler.annotate(color="red")
    def generate_sequences(self, prompts: DataProto):
        assert self._is_rollout
        # Start of RL step — begin tracer (bumps step counter, barriers, writes base timestamp).
        self._pp_trace_begin_step_if_active()
        prompts = prompts.to(get_device_name())
        save_log_by_rank(f"[Rank {torch.distributed.get_rank()}] {prompts=}")
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)

        timing_generate = {}
        _tracer = self._pp_tracer
        if self._is_actor:  # For rollout only, we do not switch context.
            loop = get_event_loop()
            _lr_ctx = _tracer.phase("load_rollout") if _tracer else None
            if _lr_ctx is not None:
                _lr_ctx.__enter__()
            try:
                with simple_timer("load_rollout", timing_generate):
                    loop.run_until_complete(self.rollout_mode())
                log_gpu_memory_usage("After switch to rollout mode", logger=logger)
            finally:
                if _lr_ctx is not None:
                    _lr_ctx.__exit__(None, None, None)

        _g_ctx = _tracer.phase("rollout") if _tracer else None
        if _g_ctx is not None:
            _g_ctx.__enter__()
        try:
            with simple_timer("generate_sequences", timing_generate):
                start_time = datetime.datetime.now()
                save_log_by_rank(f"[Rank {torch.distributed.get_rank()}] rollout start, {start_time=}")
                output = self.rollout.generate_sequences(prompts=prompts)
                end_time = datetime.datetime.now()
                save_log_by_rank(f"[Rank {torch.distributed.get_rank()}] rollout end, {end_time=}")
                save_log_by_rank(f"[Rank {torch.distributed.get_rank()}] rollout time, {end_time - start_time=}")
        finally:
            if _g_ctx is not None:
                _g_ctx.__exit__(None, None, None)
        if _tracer is not None:
            _tracer.record_hbm("after_rollout")

        if self._is_actor:
            _ur_ctx = _tracer.phase("unload_rollout") if _tracer else None
            if _ur_ctx is not None:
                _ur_ctx.__enter__()
            try:
                with simple_timer("unload_rollout", timing_generate):
                    loop.run_until_complete(self.trainer_mode())
                log_gpu_memory_usage("After switch to trainer mode", logger=logger)
            finally:
                if _ur_ctx is not None:
                    _ur_ctx.__exit__(None, None, None)
            if _tracer is not None:
                _tracer.record_hbm("after_unload_rollout")

        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        if torch.distributed.get_rank() == 0:
            print(f"[PROFILING] generate_sequences timing: {timing_generate}")
        output = output.to("cpu")
        # clear kv cache
        aggressive_empty_cache(force_sync=True)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="compute_ref_log_prob", logger=logger)
    @DistProfiler.annotate(color="olive")
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref
        # Idempotent — bumps counter on ref-only workers, no-op when
        # colocated with actor (already started by generate_sequences).
        self._pp_trace_begin_step_if_active()
        timing_ref = {}
        _tracer = self._pp_tracer
        _ref_outer = _tracer.phase("compute_ref_log_prob") if _tracer else None
        if _ref_outer is not None:
            _ref_outer.__enter__()

        _lr_ctx = _tracer.phase("load_ref") if _tracer else None
        if _lr_ctx is not None:
            _lr_ctx.__enter__()
        try:
            with simple_timer("load_ref", timing_ref):
                if self._ref_is_offload_param:
                    load_megatron_model_to_gpu(self.ref_module, load_grad=False)
                    log_gpu_memory_usage("After load ref params and grad during compute_ref_log_prob", logger=logger)
        finally:
            if _lr_ctx is not None:
                _lr_ctx.__exit__(None, None, None)

        # Same CPU-offload pattern as the actor path: restore the ref's HF
        # replica to GPU on entry (if present), offload on exit. No weight
        # sync needed — ref weights never change. Gated by
        # RLPIPE_FUSED_FORWARD_CPU_OFFLOAD=1 (same env var as actor path).
        _ref_fused_active = (
            getattr(self, "ref_policy", None) is not None
            and getattr(self.ref_policy, "use_fused_forward_pp", False)
            and getattr(self.ref_policy, "_fused_inference_stage", None) is not None
        )
        import os as _os_cpu_off_ref
        _ref_cpu_offload = _os_cpu_off_ref.environ.get("RLPIPE_FUSED_FORWARD_CPU_OFFLOAD", "0") == "1"
        if _ref_fused_active and _ref_cpu_offload:
            _rf_r_ctx = _tracer.phase("ref/restore_fused_replica") if _tracer else None
            if _rf_r_ctx is not None:
                _rf_r_ctx.__enter__()
            try:
                with simple_timer("restore_fused_replica", timing_ref):
                    import torch as _torch
                    _device_id = _torch.cuda.current_device()
                    self.ref_policy._fused_inference_stage.to(f"cuda:{_device_id}")
            finally:
                if _rf_r_ctx is not None:
                    _rf_r_ctx.__exit__(None, None, None)

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        _clp_ref = _tracer.phase("compute_ref_log_prob_inner") if _tracer else None
        if _clp_ref is not None:
            _clp_ref.__enter__()
        try:
            with simple_timer("compute_ref_log_prob", timing_ref):
                output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
        finally:
            if _clp_ref is not None:
                _clp_ref.__exit__(None, None, None)
        output = DataProto.from_dict(tensors={"ref_log_prob": output})
        output = output.to("cpu")
        _ur_ctx = _tracer.phase("unload_ref") if _tracer else None
        if _ur_ctx is not None:
            _ur_ctx.__enter__()
        try:
            with simple_timer("unload_ref", timing_ref):
                if self._ref_is_offload_param:
                    offload_megatron_model_to_cpu(self.ref_module)
                    log_gpu_memory_usage("After offload ref params and grad during compute_ref_log_prob", logger=logger)
        finally:
            if _ur_ctx is not None:
                _ur_ctx.__exit__(None, None, None)

        if _ref_fused_active and _ref_cpu_offload:
            _of_r_ctx = _tracer.phase("ref/offload_fused_replica") if _tracer else None
            if _of_r_ctx is not None:
                _of_r_ctx.__enter__()
            try:
                with simple_timer("offload_fused_replica", timing_ref):
                    self.ref_policy._fused_inference_stage.to("cpu")
            finally:
                if _of_r_ctx is not None:
                    _of_r_ctx.__exit__(None, None, None)

        aggressive_empty_cache(force_sync=True)
        if _ref_outer is not None:
            _ref_outer.__exit__(None, None, None)
        if _tracer is not None:
            _tracer.record_hbm("after_compute_ref_log_prob")

        # Ref-only workers (no actor role) must flush here because there's
        # no downstream update_actor call to do it. Colocated actor+ref
        # workers wait for update_actor's end_step.
        if not self._is_actor:
            self._pp_trace_end_step_if_active()

        timing_ref = reduce_timing(timing_ref)
        if torch.distributed.get_rank() == 0:
            print(f"[PROFILING] compute_ref_log_prob timing: {timing_ref}")
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="compute_log_prob", logger=logger)
    @DistProfiler.annotate(color="blue")
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        timing_log_prob = {}
        _tracer = self._pp_tracer
        # Rename the phase to make the visualization unambiguous: the fused
        # path keeps the same Python method name as the baseline (so verl
        # callers don't change), but the trace label flips between
        # `compute_log_prob_megatron_pp` and `compute_log_prob_fused` so
        # you can tell at a glance which implementation ran. The contents
        # also differ — fused contains `infer_forward` chunks, baseline
        # contains `train_forward` micro-batches via the Megatron PP.
        _is_fused_path = (
            self.actor is not None
            and getattr(self.actor, "use_fused_forward_pp", False)
            and getattr(self.actor, "_fused_inference_stage", None) is not None
        )
        _phase_label = "compute_log_prob_fused" if _is_fused_path else "compute_log_prob_megatron_pp"
        _clp_outer = _tracer.phase(_phase_label) if _tracer else None
        if _clp_outer is not None:
            _clp_outer.__enter__()

        _li_ctx = _tracer.phase("load_inference") if _tracer else None
        if _li_ctx is not None:
            _li_ctx.__enter__()
        try:
            with simple_timer("load_inference", timing_log_prob):
                if self._is_offload_param:
                    load_megatron_model_to_gpu(self.actor_module, load_grad=False)
                    log_gpu_memory_usage("After load actor params and grad during compute_log_prob", logger=logger)
        finally:
            if _li_ctx is not None:
                _li_ctx.__exit__(None, None, None)
        if _tracer is not None:
            _tracer.record_hbm("compute_log_prob/after_load_inference")

        # rlpipe: Fixed in phase 2-lite+sync — the HF replica under the fused
        # forward path is loaded once at init from the initial checkpoint and
        # never updated as the Megatron actor trains. After the first
        # update_actor, returning log_probs from the replica gives stale
        # (initial-weight) results instead of the current actor's log_probs.
        # Run 14 verified the drift is measurable by step 2. We sync the HF
        # replica from the current Megatron state here, reusing verl's
        # `per_tensor_generator` (the same producer feeding the SGLang rollout
        # weight sync in `rollout_mode`). Megatron is already on GPU at this
        # point because `load_inference` above loaded it.
        #
        # Separately, for 8B+ models the HF replica (~16 GB at 8B) competes
        # with `update_actor`'s training state for HBM. We pin it to CPU
        # between compute_log_prob calls and restore-on-demand, so during
        # `update_actor` only Megatron state is resident on GPU.
        _fused_active = (
            self.actor is not None
            and getattr(self.actor, "use_fused_forward_pp", False)
            and getattr(self.actor, "_fused_inference_stage", None) is not None
        )
        # CPU offload gate — set RLPIPE_FUSED_FORWARD_CPU_OFFLOAD=1 for 8B+
        # models where the HF replica (16 GB) competes with `update_actor`
        # training state. At 1.7B the 3.4 GB replica fits alongside training
        # state, so we skip the offload entirely (zero overhead).
        import os as _os_cpu_off
        _cpu_offload = _os_cpu_off.environ.get("RLPIPE_FUSED_FORWARD_CPU_OFFLOAD", "0") == "1"
        if _fused_active and _cpu_offload:
            _rf_ctx = _tracer.phase("restore_fused_replica") if _tracer else None
            if _rf_ctx is not None:
                _rf_ctx.__enter__()
            try:
                with simple_timer("restore_fused_replica", timing_log_prob):
                    import torch as _torch
                    _device_id = _torch.cuda.current_device()
                    self.actor._fused_inference_stage.to(f"cuda:{_device_id}")
            finally:
                if _rf_ctx is not None:
                    _rf_ctx.__exit__(None, None, None)
        if _fused_active:
            _sf_ctx = _tracer.phase("sync_fused_replica") if _tracer else None
            if _sf_ctx is not None:
                _sf_ctx.__enter__()
            try:
                with simple_timer("sync_fused_replica", timing_log_prob):
                    self._sync_fused_replica_from_megatron(
                        self.actor._fused_inference_stage
                    )
            finally:
                if _sf_ctx is not None:
                    _sf_ctx.__exit__(None, None, None)

        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        _clp_inner = _tracer.phase("compute_log_prob_inner") if _tracer else None
        if _clp_inner is not None:
            _clp_inner.__enter__()
        try:
            with simple_timer("compute_log_prob", timing_log_prob):
                output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
        finally:
            if _clp_inner is not None:
                _clp_inner.__exit__(None, None, None)
        output = DataProto.from_dict(
            tensors={"old_log_probs": output, "entropys": entropys},
            meta_info={"temperature": self.config.rollout.temperature},
        )
        output = output.to("cpu")
        # clear kv cache
        _ui_ctx = _tracer.phase("unload_inference") if _tracer else None
        if _ui_ctx is not None:
            _ui_ctx.__enter__()
        try:
            with simple_timer("unload_inference", timing_log_prob):
                if self._is_offload_param:
                    offload_megatron_model_to_cpu(self.actor_module)
                    log_gpu_memory_usage("After offload actor params and grad during compute_log_prob", logger=logger)
        finally:
            if _ui_ctx is not None:
                _ui_ctx.__exit__(None, None, None)

        # Offload the fused HF replica to CPU so update_actor has the full
        # HBM budget (matters at 8B+ where the replica is ~16 GB). gpu_restore
        # happens at the start of the next compute_log_prob call. Gated by
        # RLPIPE_FUSED_FORWARD_CPU_OFFLOAD=1 — at 1.7B we skip this to avoid
        # the ~3 s transfer overhead per step.
        if _fused_active and _cpu_offload:
            _of_ctx = _tracer.phase("offload_fused_replica") if _tracer else None
            if _of_ctx is not None:
                _of_ctx.__enter__()
            try:
                with simple_timer("offload_fused_replica", timing_log_prob):
                    self.actor._fused_inference_stage.to("cpu")
            finally:
                if _of_ctx is not None:
                    _of_ctx.__exit__(None, None, None)

        aggressive_empty_cache(force_sync=True)
        if _tracer is not None:
            _tracer.record_hbm("compute_log_prob/after_unload")
        if _clp_outer is not None:
            _clp_outer.__exit__(None, None, None)
        timing_log_prob = reduce_timing(timing_log_prob)
        if torch.distributed.get_rank() == 0:
            print(f"[PROFILING] compute_log_prob timing: {timing_log_prob}")
        return output

    def _sync_fused_replica_from_megatron(self, fused_stage) -> None:
        """Copy current Megatron actor state into the fused-forward HF replica.

        Reuses the same `per_tensor_generator` that the sharding manager uses
        to stream HF-format weights to SGLang during `rollout_mode`. The
        generator iterates through the per-rank Megatron state, normalizes
        names, broadcasts across PP, all-gathers across TP, and converts
        each parameter to HF layout. Output names match what
        `AutoModelForCausalLM.from_pretrained` loads.

        Runs every `compute_log_prob` call — the HF replica on this rank is
        refreshed to match the current actor state, so `old_log_prob`
        reflects the current policy (not the initial checkpoint). This
        fixes the weight-drift bug documented in
        `project_megatron_port_phase1_mvp.md`.
        """
        import torch
        from verl.utils.megatron_utils import per_tensor_generator

        gen = per_tensor_generator(
            self.actor_module,
            self.actor_model_config,
            self.weight_converter,
            self.tf_config,
            self.layer_name_mapping,
        )
        # V1: MegatronFusedInferenceStage has .model (full HF replica)
        # V2: MegatronReversePPInferenceStage has .stage.model (PP-sharded)
        if hasattr(fused_stage, 'model'):
            hf_model = fused_stage.model
        elif hasattr(fused_stage, 'stage'):
            hf_model = fused_stage.stage.model
        else:
            raise ValueError(f"Unknown fused stage type: {type(fused_stage)}")
        hf_dtype = fused_stage.dtype
        hf_device = fused_stage.device
        hf_state = dict(hf_model.state_dict())  # snapshot of existing tensors
        num_assigned = 0
        for name, tensor in gen:
            if name not in hf_state:
                continue
            target = hf_state[name]
            tensor = tensor.to(device=hf_device, dtype=hf_dtype, non_blocking=True)
            if tensor.shape != target.shape:
                # Shape mismatch usually means the layer name normalization
                # skipped something — skip silently and let load_state_dict
                # report it below.
                continue
            with torch.no_grad():
                target.copy_(tensor)
            num_assigned += 1
        # No explicit `load_state_dict` — we copied in-place above, which
        # keeps the HF module's parameter identity stable (important if any
        # references were cached). Log once at rank 0 for visibility.
        try:
            _rank = torch.distributed.get_rank()
        except Exception:
            _rank = -1
        if _rank == 0:
            total = len(hf_state)
            print(
                f"[rlpipe megatron-fused] rank={_rank} weight-sync: "
                f"{num_assigned}/{total} HF params updated from Megatron actor",
                flush=True,
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, checkpoint_path, hdfs_path=None, del_local_after_load=True):
        # No checkpoint to load, just offload the model and optimizer to CPU
        if checkpoint_path is None:
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
            if self._is_offload_optimizer:
                offload_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After offload actor params and optimizer during load_checkpoint", logger=logger)
            return

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        self.checkpoint_mananager.load_checkpoint(
            local_path=checkpoint_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_pretrained_model(self, checkpoint_path, del_local_after_load=True):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        self.checkpoint_mananager.save_checkpoint(
            local_path=checkpoint_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)


class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        await self.rollout_mode()
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        await self.trainer_mode()
        return True

    # ============================ vLLM related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        return self.rollout.get_zeromq_address()

    # ============================ SGLang related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        ret = await self.rollout.chat_completion(json_request)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> list[int]:
        ret = await self.rollout.generate(prompt_ids, sampling_params, request_id, image_data=image_data)
        return ret


class CriticWorker(MegatronWorker, DistProfilerExtension):
    def __init__(self, config: McoreCriticConfig):
        Worker.__init__(self)

        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        self.config: McoreCriticConfig = config

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel strategy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            set_numa_affinity()
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            get_torch_device().set_device(rank)

            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

        is_collect = (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            and mpu.get_context_parallel_rank() == 0
        )
        self._register_dispatch_collect_info(
            mesh_name="critic", dp_rank=mpu.get_data_parallel_rank(), is_collect=is_collect
        )

        set_random_seed(seed=self.config.megatron.seed)

        # set FSDP offload params
        self._is_offload_param = self.config.megatron.param_offload
        self._is_offload_optimizer = self.config.megatron.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
        if self.config.get("ppo_micro_batch_size", None):
            self.config.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size

        # TODO(sgm): support critic model offload

    def _build_critic_model_optimizer(
        self, model_path, optim_config, override_model_config, override_transformer_config, override_ddp_config
    ):
        from verl.utils.megatron.optimizer import (
            get_megatron_optimizer,
            get_megatron_optimizer_param_scheduler,
            init_megatron_optim_config,
        )
        from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module
        from verl.utils.model import print_model_size

        self._init_hf_config_and_tf_config(
            model_path,
            self.config.model.tokenizer_path,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.config.model.get("trust_remote_code", False),
            self.config.megatron.use_mbridge,
        )

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=True,  # critic is value model
            share_embeddings_and_output_weights=False,
            wrap_with_ddp=True,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
        )
        critic_module = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            bridge=self.bridge,
            override_model_config=override_model_config,
            override_ddp_config=override_ddp_config,
        )
        # note that here critic_module will be a list to be compatible with the construction of interleaved pp (vpp).
        # but here, we do not use pp (vpp) yet. For simplicity, we remove the list
        # critic_module = nn.ModuleList(critic_module)

        if self.config.load_weight:
            t0 = time.time()
            if self.config.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(
                    critic_module, self.config.megatron.dist_checkpointing_path, is_value_model=True
                )
            else:
                if self.bridge is not None:
                    local_model_path = get_hf_model_path(self.config)
                    self.bridge.load_weights(critic_module, local_model_path)
                else:
                    load_megatron_gptmodel_weights(
                        self.config, self.hf_config, critic_module, params_dtype=self.dtype, is_value_model=True
                    )
            t1 = time.time()
            if torch.distributed.get_rank() == 0:
                print(f"critic load_weight time: {t1 - t0}")
        if self.rank == 0:
            print_model_size(critic_module[0])

        # TODO: add more optimizer args into config
        optim_config_megatron = init_megatron_optim_config(optim_config)
        critic_optimizer = get_megatron_optimizer(model=critic_module, config=optim_config_megatron)
        critic_optimizer_scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=critic_optimizer, config=optim_config
        )
        get_torch_device().empty_cache()
        return critic_module, critic_optimizer, critic_optimizer_scheduler, self.hf_config, optim_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # create critic

        from verl.utils.torch_dtypes import PrecisionType

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)
        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        override_transformer_config = OmegaConf.to_container(
            OmegaConf.create(self.config.megatron.get("override_transformer_config", {}))
        )
        override_ddp_config = OmegaConf.to_container(
            OmegaConf.create(self.config.megatron.get("override_ddp_config", {}))
        )
        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        (
            self.critic_module,
            self.critic_optimizer,
            self.critic_optimizer_scheduler,
            self.critic_model_config,
            critic_optimizer_config,
        ) = self._build_critic_model_optimizer(
            model_path=self.config.model.path,
            optim_config=self.config.optim,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
            override_ddp_config=override_ddp_config,
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)

        self.critic = MegatronPPOCritic(
            config=self.config,
            model_config=self.critic_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            critic_module=self.critic_module,
            critic_optimizer=self.critic_optimizer,
            critic_optimizer_config=critic_optimizer_config,
        )
        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_mananager = MegatronCheckpointManager(
            config=self.config,
            checkpoint_config=self.config.checkpoint,
            model_config=self.critic_model_config,
            transformer_config=self.tf_config,
            role="critic",
            model=self.critic_module,
            arch=self.architectures[0],
            hf_config=self.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=False,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            optimizer=self.critic_optimizer,
            optimizer_scheduler=self.critic_optimizer_scheduler,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.config.optim.use_checkpoint_opt_param_scheduler,
            bridge=self.bridge,
            use_dist_checkpointing=self.config.megatron.use_dist_checkpointing,
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="cyan")
    def compute_values(self, data: DataProto):
        micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        data = data.to(get_device_id())
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        values = self.critic.compute_values(data=data)
        output = DataProto.from_dict(tensors={"values": values})
        output = output.to("cpu")
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="pink")
    def update_critic(self, data: DataProto):
        data = data.to(get_device_id())

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.critic_optimizer)

        dataloader = self.critic.make_minibatch_iterator(data)
        with Timer(name="update_critic", logger=None) as timer:
            metrics = self.critic.update_critic(dataloader=dataloader)
        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        metrics["critic/lr"] = get_megatron_last_lr(self.critic_optimizer)
        self.critic_optimizer_scheduler.step(1)

        output = DataProto(batch=None, meta_info={"metrics": metrics})

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)
        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, checkpoint_path, hdfs_path=None, del_local_after_load=True):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        self.checkpoint_mananager.load_checkpoint(
            local_path=checkpoint_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_steps=0, max_ckpt_to_keep=None):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        self.checkpoint_mananager.save_checkpoint(
            local_path=checkpoint_path, hdfs_path=hdfs_path, global_step=global_steps, max_ckpt_to_keep=max_ckpt_to_keep
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)


class RewardModelWorker(MegatronWorker, DistProfilerExtension):
    """
    Note that we only implement the reward model that is subclass of AutoModelForSequenceClassification.
    """

    def __init__(self, config):
        Worker.__init__(self)

        profiler_config = omega_conf_to_dataclass(config.get("profiler", {}), dataclass_type=ProfilerConfig)
        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self,
            DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config),
        )
        self.config = config

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel strategy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            set_numa_affinity()
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            get_torch_device().set_device(rank)

            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

        is_collect = (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            and mpu.get_context_parallel_rank() == 0
        )
        self._register_dispatch_collect_info(
            mesh_name="reward", dp_rank=mpu.get_data_parallel_rank(), is_collect=is_collect
        )

        set_random_seed(seed=self.config.megatron.seed)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_rm_model(self, model_path, tokenizer, override_model_config, override_transformer_config):
        from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module

        self._init_hf_config_and_tf_config(
            model_path,
            tokenizer,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.config.model.get("trust_remote_code", False),
            self.config.megatron.use_mbridge,
        )

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=True,  # reward model is value model
            share_embeddings_and_output_weights=False,
            wrap_with_ddp=False,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
        )
        reward_model = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            bridge=self.bridge,
            override_model_config=override_model_config,
        )

        if self.config.load_weight:
            if self.config.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(reward_model, self.config.megatron.dist_checkpointing_path, is_value_model=True)
            else:
                if self.bridge is not None:
                    local_model_path = get_hf_model_path(self.config)
                    self.bridge.load_weights(reward_model, local_model_path)
                else:
                    load_megatron_gptmodel_weights(
                        self.config, self.hf_config, reward_model, params_dtype=self.dtype, is_value_model=True
                    )

        get_torch_device().empty_cache()
        return reward_model, self.hf_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # create critic

        from verl.utils.torch_dtypes import PrecisionType

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)
        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        override_transformer_config = OmegaConf.to_container(
            OmegaConf.create(self.config.megatron.get("override_transformer_config", {}))
        )

        use_shm = self.config.model.get("use_shm", False)
        sft_tokenizer_local_path = copy_to_local(self.config.model.input_tokenizer, use_shm=use_shm)
        sft_tokenizer = hf_tokenizer(sft_tokenizer_local_path)
        rm_tokenizer_path = self.config.model.get("rm_tokenizer", None)
        rm_tokenizer = None
        if rm_tokenizer_path is not None:
            rm_tokenizer_local_path = copy_to_local(rm_tokenizer_path, use_shm=use_shm)
            rm_tokenizer = hf_tokenizer(
                rm_tokenizer_local_path, trust_remote_code=self.config.model.get("trust_remote_code", False)
            )

        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        reward_model_module, reward_model_config = self._build_rm_model(
            model_path=self.config.model.path,
            tokenizer=rm_tokenizer,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
        )
        # FIXME(sgm): reward model param offload is implemented in MegatronRewardModel
        # should be implemented in workers
        self.rm = MegatronRewardModel(
            config=self.config,
            reward_model_module=reward_model_module,
            model_config=reward_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            sft_tokenizer=sft_tokenizer,
            rm_tokenizer=rm_tokenizer,
        )

    # TODO: reward model use itself tokenizer instead of sft tokenizer
    # the input_ids, responses, attention_mask and position_ids may be different!
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        data.meta_info["micro_batch_size"] = self.config.micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        data = data.to(get_device_id())
        output = self.rm.compute_reward(data)
        output = output.to("cpu")
        return output
