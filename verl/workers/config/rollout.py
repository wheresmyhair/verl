# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.utils.profiler import ProfilerConfig

__all__ = [
    "SamplingConfig",
    "MultiTurnConfig",
    "CustomAsyncServerConfig",
    "AgentLoopConfig",
    "TraceConfig",
    "ServerConfig",
    "RolloutConfig",
]


@dataclass
class SamplingConfig(BaseConfig):
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    do_sample: bool = True
    n: int = 1


@dataclass
class MultiTurnConfig(BaseConfig):
    _mutable_fields = {"max_assistant_turns", "max_user_turns"}

    enable: bool = False
    max_assistant_turns: Optional[int] = None
    tool_config_path: Optional[str] = None
    max_user_turns: Optional[int] = None
    max_parallel_calls: int = 1
    max_tool_response_length: int = 256
    tool_response_truncate_side: str = "middle"
    interaction_config_path: Optional[str] = None
    use_inference_chat_template: bool = False
    tokenization_sanity_check_mode: str = "strict"
    format: str = "hermes"
    num_repeat_rollouts: Optional[int] = None


@dataclass
class CustomAsyncServerConfig(BaseConfig):
    path: Optional[str] = None
    name: Optional[str] = None


@dataclass
class AgentLoopConfig(BaseConfig):
    num_workers: int = 8
    default_agent_loop: str = "single_turn_agent"
    agent_loop_config_path: Optional[str] = None
    custom_async_server: CustomAsyncServerConfig = field(default_factory=CustomAsyncServerConfig)


@dataclass
class TraceConfig(BaseConfig):
    backend: Optional[str] = None
    token2text: bool = False


@dataclass
class ServerConfig(BaseConfig):
    """
    Configuration for SGLang server when running in server mode
    """

    timeout: float = 60.0
    max_attempts: int = 3
    retry_delay: float = 2.0
    max_connections: int = 1000
    max_start_wait_time: float = 300.0


@dataclass
class RolloutConfig(BaseConfig):
    _mutable_fields = {"max_model_len", "load_format", "tensor_model_parallel_size"}

    name: Optional[str] = MISSING
    mode: str = "sync"
    skip_tokenizer_init: bool = True

    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    do_sample: bool = True
    n: int = 1

    # Early termination threshold for multi-turn rollout in sglang.
    # Abort remaining requests when (1 - over_sample_rate) * total_requests are completed.
    over_sample_rate: float = 0.0

    prompt_length: int = 512
    response_length: int = 512

    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    ignore_eos: bool = False
    enforce_eager: bool = True
    cudagraph_capture_sizes: Optional[list] = None
    free_cache_engine: bool = True
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    tensor_model_parallel_size: int = 2
    pipeline_model_parallel_size: int = 1
    max_num_batched_tokens: int = 8192

    # TODO: enable train_kwargs
    # train_sampling_config: SamplingConfig = field(default_factory=SamplingConfig)

    val_kwargs: SamplingConfig = field(default_factory=SamplingConfig)

    max_model_len: Optional[int] = None
    max_num_seqs: int = 1024

    # note that the logprob computation should belong to the actor
    log_prob_micro_batch_size: Optional[int] = None
    log_prob_micro_batch_size_per_gpu: Optional[int] = None
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int = 16384

    disable_log_stats: bool = True

    multi_stage_wake_up: bool = False
    engine_kwargs: dict = field(default_factory=dict)

    calculate_log_probs: bool = False

    agent: AgentLoopConfig = field(default_factory=AgentLoopConfig)

    trace: TraceConfig = field(default_factory=TraceConfig)

    multi_turn: MultiTurnConfig = field(default_factory=MultiTurnConfig)

    # Server configuration for sglang server mode
    server: ServerConfig = field(default_factory=ServerConfig)

    update_weights_bucket_megabytes: int = 512

    # Weight sync mode between training engine and inference engine.
    # - "tensor": stock CUDA-IPC path (verl default; needs container CAP_SYS_PTRACE)
    # - "distributed": NCCL collective broadcast over a TCPStore-rendezvous group
    #   (rlpipe extension for ptrace-restricted docker — see
    #   docs/rlpipe/option_b_distributed_weight_sync_plan.md)
    weight_sync_mode: str = "tensor"
    # Master address/port for the distributed weight-sync TCPStore. Both
    # actor and sglang sides must agree. Only used when weight_sync_mode="distributed".
    weight_sync_master_addr: str = "127.0.0.1"
    weight_sync_master_port: int = 29600

    skip_rollout: bool = False

    skip_dump_dir: str = "/tmp/rollout_dump"

    profiler: Optional[ProfilerConfig] = None

    enable_chunked_prefill: bool = True

    enable_prefix_caching: bool = True

    load_format: str = "dummy"

    layered_summon: bool = False

    layer_name_map: dict = field(default_factory=dict)

    tp_groups: Optional[list] = None  # e.g. [[0,1],[2],[3]] for heterogeneous TP

    # Routing config (used by RolloutRouter in the trainer for het-TP dispatch)
    routing_strategy: str = "round_robin"
    routing_warmup_epochs: int = 1
    routing_prompt_coef: float = 1.0
    routing_response_coef: float = 4.0
    routing_default_response_length: float = 1024.0
    routing_history_estimator: str = "mean"
    routing_ema_alpha: float = 0.5
    routing_response_agg: str = "max"
    routing_group_weights: Optional[list] = None
    routing_random_seed: int = 0

    # Progressive rollout: abort remaining requests when this fraction complete.
    # None or 1.0 = disabled (default, wait for all). 0.9 = return when 90% done.
    progressive_threshold: Optional[float] = None

    # rlpipe dual-fleet fan-in rollout. When True, _build_rollout_heterogeneous
    # picks DualFleetFanInRollout which launches 4 DP HTTP servers + 1 TP
    # HTTP server on the TP leader and orchestrates dynamic DP→TP swap on
    # tail. Requires tp_groups=[[0,1,2,3]] (single TP group).
    enable_dual_fleet_fanin: bool = False

    sglang_engine_mode: str = "local"

    limit_images: Optional[int] = None

    skip_tokenizer_init: bool = False

    def __post_init__(self):
        """Validate the rollout config"""
        if self.expert_parallel_size > 1:
            assert self.expert_parallel_size == (self.tensor_model_parallel_size * self.data_parallel_size), (
                "expert_parallel_size must be equal to tensor_model_parallel_size * data_parallel_size"
            )

        if self.pipeline_model_parallel_size > 1:
            if self.name == "vllm" or self.name == "sglang":
                raise NotImplementedError(
                    f"Current rollout {self.name=} not implemented pipeline_model_parallel_size > 1 yet."
                )

        if self.tp_groups is not None:
            if self.name != "sglang":
                raise ValueError("rollout.tp_groups is currently only supported for rollout.name='sglang'")
            if not isinstance(self.tp_groups, list) or len(self.tp_groups) == 0:
                raise ValueError("rollout.tp_groups must be a non-empty list of rank groups")

            flat_ranks = []
            for idx, group in enumerate(self.tp_groups):
                if not isinstance(group, list) or len(group) == 0:
                    raise ValueError(f"rollout.tp_groups[{idx}] must be a non-empty list of ranks")
                for rank in group:
                    if not isinstance(rank, int):
                        raise ValueError(f"rollout.tp_groups[{idx}] contains non-integer rank {rank!r}")
                    if rank < 0:
                        raise ValueError(f"rollout.tp_groups[{idx}] contains negative rank {rank}")
                    flat_ranks.append(rank)

            if len(set(flat_ranks)) != len(flat_ranks):
                raise ValueError(f"rollout.tp_groups must not contain duplicate ranks: {self.tp_groups!r}")
