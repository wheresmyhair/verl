"""
Torch naive PP worker for verl — follows the exact same patterns as megatron_workers.py.

Each GPU = 1 Ray worker = 1 PP stage for training + 1 independent vLLM instance for rollout.

Training: GPUs cooperate via NCCL P2P send/recv in a 1F1B pipeline schedule.
Rollout: Each GPU runs its own vLLM (DP rollout). Weights gathered from all PP stages.
Fused forward (optional): Each GPU holds TWO stages — training at rank r,
    inference at rank pp_size-1-r. Inference runs in reverse direction during
    training bubbles, computing old_log_probs inline.
"""

import datetime
import logging
import os
import shutil
import time
from typing import Any, Dict, List, Optional

import psutil
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import (
    Dispatch,
    DYNAMIC_INDEX_DISPATCH,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils.device import (
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.distributed import set_numa_affinity
from verl.utils.fs import copy_to_local
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import (
    GPUMemoryLogger,
    log_gpu_memory_usage,
    simple_timer,
)
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.utils.ray_utils import get_event_loop
from verl.workers.rollout import get_rollout_class

from .torch_pp.pipeline_stage import PipelineStage, restore_global_layer_keys
from .torch_pp.inference_stage import InferenceStage
from .torch_pp.schedule import ScheduleOp, build_1f1b_schedule
from .torch_pp.fused_schedule import (
    FusedScheduleOp,
    build_default_fused_schedule,
    parse_schedule,
)
from .torch_pp.loss import (
    compute_grpo_loss,
    compute_grpo_loss_fused,
    entropy_from_logits,
    gather_response_log_probs,
    log_probs_from_logits,
)
from .torch_pp.pp_trace import PPTracer
from .torch_pp.response_length_profiler import ResponseLengthProfiler
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_CHECKPOINT_AUX_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "merges.txt",
    "vocab.json",
    "added_tokens.json",
    "chat_template.jinja",
    "preprocessor_config.json",
]


# ======================================================================
# Offload / load helpers
# ======================================================================

def _validate_heterogeneous_tp_groups(tp_groups: List[List[int]], world_size: int):
    """Validate het TP groups before building rollout process groups."""
    if not tp_groups:
        raise ValueError("rollout.tp_groups must be a non-empty list")

    seen_ranks = []
    for idx, group in enumerate(tp_groups):
        if not group:
            raise ValueError(f"rollout.tp_groups[{idx}] must not be empty")
        for rank in group:
            if rank < 0 or rank >= world_size:
                raise ValueError(
                    f"rollout.tp_groups[{idx}] contains rank {rank}, but world_size is {world_size}"
                )
            seen_ranks.append(rank)

    missing_ranks = sorted(set(range(world_size)) - set(seen_ranks))
    duplicate_ranks = sorted({rank for rank in seen_ranks if seen_ranks.count(rank) > 1})
    if duplicate_ranks:
        raise ValueError(
            f"rollout.tp_groups contains duplicate ranks {duplicate_ranks}: {tp_groups!r}"
        )
    if missing_ranks:
        raise ValueError(
            f"rollout.tp_groups must cover every rollout worker rank exactly once; missing {missing_ranks}"
        )


def _format_het_tp_debug_info(
    my_rank: int,
    my_group_ranks: List[int],
    my_tp_local_rank: int,
    my_tp_size: int,
    my_dp_rank: int,
    is_tp_leader: bool,
):
    """Compact rank/group context for het TP log and error messages."""
    return (
        f"rank={my_rank} group={my_group_ranks} tp_rank={my_tp_local_rank} "
        f"tp_size={my_tp_size} dp_rank={my_dp_rank} is_leader={is_tp_leader}"
    )

def _offload_module(module):
    """Offload module parameters and buffers to CPU."""
    if module is not None:
        module.to("cpu")


def _load_module(module, device):
    """Load module parameters and buffers to device."""
    if module is not None:
        module.to(device)


def _offload_optimizer(optimizer):
    """Offload optimizer states to CPU."""
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to("cpu")


def _load_optimizer(optimizer, device):
    """Load optimizer states to device."""
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


# ======================================================================
# Simplified GRPO loss from pre-computed response log_probs
# ======================================================================

def _compute_grpo_loss_from_resp_lp(
    new_resp_lp: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    ref_log_probs=None,
    clip_ratio: float = 0.2,
    kl_coef: float = 0.001,
    entropy_coef: float = 0.0,
):
    """GRPO loss given pre-computed new response log_probs [micro_B, R]."""
    log_ratio = new_resp_lp - old_log_probs
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

    adv = advantages.unsqueeze(1)
    pg_loss = torch.max(-ratio * adv, -clipped * adv)

    kl_loss = torch.zeros_like(pg_loss)
    if ref_log_probs is not None and kl_coef > 0.0:
        kl_loss = kl_coef * (new_resp_lp - ref_log_probs)

    token_loss = (pg_loss + kl_loss) * response_mask
    num_tokens = response_mask.sum().clamp(min=1)
    loss = token_loss.sum() / num_tokens

    with torch.no_grad():
        valid = response_mask.bool()
        stats = {
            "loss": loss.item(),
            "pg_loss": (pg_loss * response_mask).sum().item() / num_tokens.item(),
            "ratio_mean": ratio[valid].mean().item() if valid.any() else 0.0,
            "ratio_max": ratio[valid].max().item() if valid.any() else 0.0,
            "clip_frac": ((ratio[valid] - 1.0).abs() > clip_ratio).float().mean().item()
            if valid.any() else 0.0,
        }
        if ref_log_probs is not None:
            kl_valid = (new_resp_lp - ref_log_probs)[valid]
            stats["kl_per_token"] = kl_valid.mean().item() if valid.any() else 0.0
    return loss, stats


# ======================================================================
# PP forward helper (shared between compute_log_prob / compute_ref_log_prob)
# ======================================================================

def _pp_forward_log_prob(
    stage: PipelineStage,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    M: int,
    hidden_size: int,
    pp_rank: int,
    device: torch.device,
    calculate_entropy: bool = True,
    pp_pair_groups: dict = None,
    use_fused_loss: bool = True,
    use_remove_padding: bool = False,
    tracer: Optional[PPTracer] = None,
    forward_trace_name: str = "infer_forward",
    recv_trace_name: str = "p2p_recv",
    send_trace_name: str = "p2p_send",
    loss_trace_name: str = "loss",
):
    """
    Run forward-only PP across all micro-batches and collect log probs
    on the last stage. Matches the megatron actor's compute_log_prob pattern.

    When use_fused_loss=True, the last stage returns hidden states and
    applies FusedLinearForPPO to avoid materializing [B, S, V] logits.

    When use_remove_padding=True, padding tokens are removed before forward,
    reducing compute and memory proportional to actual token count.

    Returns (full_log_probs, full_entropy) on last stage, (None, None) elsewhere.
    Both are [B, S-1] shaped (padded back if rmpad was used).
    """
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    B, S = input_ids.shape
    micro_B = B // M

    stage.set_batch_data(input_ids, attention_mask, M, use_remove_padding=use_remove_padding)

    all_log_probs = []
    all_entropys = []

    fused = FusedLinearForPPO(chunk_size=512) if use_fused_loss else None

    with torch.no_grad():
        for mb in range(M):
            if stage.is_first:
                if tracer is not None:
                    with tracer.trace(forward_trace_name, mb):
                        output = stage.forward_step(mb, return_hidden=(use_fused_loss and stage.is_last))
                else:
                    output = stage.forward_step(mb, return_hidden=(use_fused_loss and stage.is_last))
            else:
                # Activation shape: [1, total_nnz, H] if rmpad, else [micro_B, S, H]
                nnz = stage._micro_nnz[mb]
                if use_remove_padding:
                    act_shape = (1, nnz, hidden_size)
                else:
                    act_shape = (micro_B, S, hidden_size)
                hidden_cpu = torch.empty(act_shape, dtype=stage.dtype, device="cpu")
                pair_group = pp_pair_groups[pp_rank - 1]
                if tracer is not None:
                    with tracer.trace(recv_trace_name, mb):
                        dist.recv(hidden_cpu, src=pp_rank - 1, group=pair_group)
                else:
                    dist.recv(hidden_cpu, src=pp_rank - 1, group=pair_group)
                hidden = hidden_cpu.to(device)
                if tracer is not None:
                    with tracer.trace(forward_trace_name, mb):
                        output = stage.forward_step(mb, hidden, return_hidden=(use_fused_loss and stage.is_last))
                else:
                    output = stage.forward_step(mb, hidden, return_hidden=(use_fused_loss and stage.is_last))

            if stage.is_last:
                micro_ids = stage._micro_input_ids[mb]  # [1, nnz] or [micro_B, S]

                def _compute_lp(output, micro_ids, mb):
                    if use_fused_loss:
                        if use_remove_padding and stage._micro_cu_seqlens:
                            # Packed sequences: build shifted labels respecting
                            # sequence boundaries (torch.roll would cross them).
                            cu = stage._micro_cu_seqlens[mb]
                            ids_flat = micro_ids.squeeze(0)  # [nnz]
                            shifted = torch.empty_like(ids_flat)
                            for j in range(len(cu) - 1):
                                s, e = cu[j].item(), cu[j + 1].item()
                                shifted[s:e - 1] = ids_flat[s + 1:e]
                                shifted[e - 1] = 0  # last pos per seq → garbage label, will be dropped
                            shifted_labels = shifted.unsqueeze(0)  # [1, nnz]
                            lp, ent = fused.forward(
                                hidden_states=output,
                                vocab_weights=stage.lm_head_weight,
                                input_ids=shifted_labels,
                                temperature=1.0,
                            )
                            # lp/ent are [1, nnz]. Extract valid positions
                            # (drop last of each seq) → [1, nnz - num_seqs]
                            valid = []
                            valid_ent = []
                            for j in range(len(cu) - 1):
                                s, e = cu[j].item(), cu[j + 1].item()
                                valid.append(lp[0, s:e - 1])
                                if calculate_entropy:
                                    valid_ent.append(ent[0, s:e - 1])
                            all_log_probs.append(torch.cat(valid).unsqueeze(0))  # [1, nnz-nseqs]
                            if calculate_entropy:
                                all_entropys.append(torch.cat(valid_ent).unsqueeze(0))
                        else:
                            rolled_labels = torch.roll(micro_ids, shifts=-1, dims=-1)
                            lp, ent = fused.forward(
                                hidden_states=output,
                                vocab_weights=stage.lm_head_weight,
                                input_ids=rolled_labels,
                                temperature=1.0,
                            )
                            all_log_probs.append(lp[:, :-1])
                            if calculate_entropy:
                                all_entropys.append(ent[:, :-1])
                    else:
                        all_log_probs.append(log_probs_from_logits(output, micro_ids))
                        if calculate_entropy:
                            all_entropys.append(entropy_from_logits(output))

                if tracer is not None:
                    with tracer.trace(loss_trace_name, mb):
                        _compute_lp(output, micro_ids, mb)
                else:
                    _compute_lp(output, micro_ids, mb)
            else:
                out_t = output.detach().contiguous().cpu()
                pair_group = pp_pair_groups[pp_rank]
                if tracer is not None:
                    with tracer.trace(send_trace_name, mb):
                        dist.send(out_t, dst=pp_rank + 1, group=pair_group)
                else:
                    dist.send(out_t, dst=pp_rank + 1, group=pair_group)

    stage.clear_batch_data()

    if stage.is_last:
        if use_remove_padding:
            # Each all_log_probs[i] is [1, nnz_i - num_seqs_i] (valid positions
            # only, last position of each sequence already dropped).
            # Re-pad to [micro_B, S-1] per micro-batch, then cat to [B, S-1].
            from flash_attn.bert_padding import pad_input, unpad_input
            padded_lps = []
            padded_ents = []
            micro_masks = list(attention_mask.chunk(M, dim=0))
            for i, lp_rmpad in enumerate(all_log_probs):
                mb_mask = micro_masks[i]
                mb_B = mb_mask.size(0)
                _, indices, cu_seqlens, *_ = unpad_input(
                    mb_mask.unsqueeze(-1), mb_mask
                )
                lp_flat = lp_rmpad.squeeze(0)  # [nnz - num_seqs]
                # Build shifted indices: for each seq of length L, map L-1
                # log_prob values to positions 0..L-2 in the (S-1)-padded output.
                seq_lens = cu_seqlens.diff().tolist()
                shifted_indices = []
                offset = 0
                for sl in seq_lens:
                    shifted_indices.append(indices[offset:offset + sl - 1])
                    offset += sl
                shifted_idx = torch.cat(shifted_indices) if shifted_indices else indices[:0]
                # Indices are in [mb_B, S] flat space but pad_input targets
                # [mb_B, S-1]. Adjust: new_flat = (old_flat // S) * (S-1) + (old_flat % S)
                adjusted_idx = (shifted_idx // S) * (S - 1) + (shifted_idx % S)
                lp_padded_flat = pad_input(
                    lp_flat.unsqueeze(-1), adjusted_idx, mb_B, S - 1
                ).squeeze(-1)
                padded_lps.append(lp_padded_flat)
                if calculate_entropy and i < len(all_entropys):
                    ent_flat = all_entropys[i].squeeze(0)
                    ent_padded = pad_input(
                        ent_flat.unsqueeze(-1), adjusted_idx, mb_B, S - 1
                    ).squeeze(-1)
                    padded_ents.append(ent_padded)
            full_log_probs = torch.cat(padded_lps, dim=0)
            full_entropy = torch.cat(padded_ents, dim=0) if calculate_entropy else None
        else:
            full_log_probs = torch.cat(all_log_probs, dim=0)
            full_entropy = torch.cat(all_entropys, dim=0) if calculate_entropy else None
        return full_log_probs, full_entropy
    return None, None


def _extract_response_log_probs(
    full_log_probs: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
):
    """
    Extract response-portion log probs from full-sequence shifted log probs.
    Returns [B, R] shaped tensor.
    """
    R = response_mask.size(1)
    prompt_lengths = attention_mask.sum(dim=-1) - response_mask.sum(dim=-1)
    response_start_positions = prompt_lengths.long()
    resp_lp = gather_response_log_probs(full_log_probs, response_start_positions, R)
    return resp_lp * response_mask


# ======================================================================
# Main worker
# ======================================================================

class ActorRolloutRefWorker(Worker):
    """
    Torch naive PP worker following verl's worker pattern.
    Follows the exact same structure as megatron_workers.ActorRolloutRefWorker.

    Dispatch mechanics (identical to megatron):
    - "actor" mesh: all workers dp_rank=0, only last PP stage is_collect=True
      -> all workers get same data, only last stage returns results
    - "rollout" mesh: each worker dp_rank=rank, all is_collect=True
      -> data split across GPUs for DP rollout
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)
        self.config = config

        # ── Initialize torch.distributed (NCCL) ──
        if not torch.distributed.is_initialized():
            set_numa_affinity()
            local_rank = int(os.environ["LOCAL_RANK"])
            # Pin the CUDA device before distributed init. When every worker can
            # see all GPUs (RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1),
            # initializing the process group first can create a stray CUDA
            # context on GPU 0 in non-owner workers.
            get_torch_device().set_device(local_rank)
            torch.distributed.init_process_group(
                backend="gloo",
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        self.pp_rank = self.rank
        self.pp_size = self.world_size
        self.device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")

        # ── Register actor mesh dispatch ──
        # PP with no TP/DP: dp_rank=0 for all, only last stage collects
        is_collect = self.pp_rank == self.pp_size - 1
        self._register_dispatch_collect_info(
            mesh_name="actor", dp_rank=0, is_collect=is_collect,
        )

        # ── Parse role ──
        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        # ── Offload flags ──
        self._is_offload_param = False
        self._is_offload_optimizer = False
        self._fused_forward = False

        if self._is_actor and self._is_rollout:
            self._is_offload_param = self.config.actor.get("param_offload", False)
            self._is_offload_optimizer = self.config.actor.get("optimizer_offload", False)
            self._fused_forward = self.config.actor.get("fused_forward", False)
        if self._is_ref:
            self._ref_is_offload_param = self.config.ref.get("param_offload", False)

        self._num_micro_batches: int = self.config.actor.get("num_micro_batches", 4)
        self._use_remove_padding: bool = self.config.model.get("use_remove_padding", True)

        # ── Will be set in init_model ──
        self.train_stage: Optional[PipelineStage] = None
        self.ref_stage: Optional[PipelineStage] = None
        self.infer_stage: Optional[InferenceStage] = None
        self.optimizer: Optional[AdamW] = None
        self.rollout = None
        self.tokenizer = None
        self.generation_config = None
        self.hidden_size: Optional[int] = None
        self.local_path: Optional[str] = None
        self.dtype = torch.bfloat16

        # ── Profiling tools ──
        self._pp_tracer: Optional[PPTracer] = None
        self._response_profiler: Optional[ResponseLengthProfiler] = None
        self._enable_pp_trace = self.config.get("enable_pp_trace", False)
        self._enable_response_profiling = self.config.get("enable_response_profiling", True)
        self._profiling_save_dir = self.config.get("profiling_save_dir", "/tmp/pp_profiling")
        self._step_counter = 0

    # ==================================================================
    # Build rollout
    # ==================================================================

    def _build_rollout(self, trust_remote_code=False):
        """Build rollout engine — delegates to heterogeneous path if tp_groups is set."""
        tp_groups = self.config.rollout.get("tp_groups", None)
        if tp_groups is not None:
            tp_groups = [list(g) for g in tp_groups]
            return self._build_rollout_heterogeneous(tp_groups, trust_remote_code)

        # ── Homogeneous TP path (existing) ──
        from torch.distributed.device_mesh import init_device_mesh
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import HFModelConfig, RolloutConfig

        rollout_config = omega_conf_to_dataclass(self.config.rollout)
        model_config = omega_conf_to_dataclass(
            self.config.model, dataclass_type=HFModelConfig,
        )

        infer_tp = self.config.rollout.get("tensor_model_parallel_size", 1)
        infer_pp = self.config.rollout.get("pipeline_model_parallel_size", 1)
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"world_size {self.world_size} not divisible by infer_world_size {infer_world_size}"
        )

        rollout_device_mesh = init_device_mesh(
            get_device_name(),
            mesh_shape=(dp, infer_tp, infer_pp),
            mesh_dim_names=["dp", "infer_tp", "infer_pp"],
        )

        is_collect = (
            rollout_device_mesh["infer_tp"].get_local_rank() == 0
            and rollout_device_mesh["infer_pp"].get_local_rank() == 0
        )
        self._register_dispatch_collect_info(
            "rollout",
            dp_rank=rollout_device_mesh["dp"].get_local_rank(),
            is_collect=is_collect,
        )

        # Init trainer and rollout random states
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config,
            model_config=model_config,
            device_mesh=rollout_device_mesh,
        )
        log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=logger)

    def _build_rollout_heterogeneous(self, tp_groups, trust_remote_code=False):
        """Build SGLang rollout with heterogeneous TP groups.

        Args:
            tp_groups: list of lists, e.g. [[0,1],[2],[3]].
                       Each sublist is a TP group of global ranks.
        """
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import HFModelConfig
        from verl.workers.torch_pp.het_device_mesh import FakeDeviceMesh, FakeDeviceMeshDim
        from verl.workers.torch_pp.het_sglang_rollout import HetSGLangRollout

        rollout_config = omega_conf_to_dataclass(self.config.rollout)
        model_config = omega_conf_to_dataclass(
            self.config.model, dataclass_type=HFModelConfig,
        )

        _validate_heterogeneous_tp_groups(tp_groups, self.world_size)

        my_rank = self.pp_rank
        num_dp_groups = len(tp_groups)

        # 1. Find this rank's TP group
        my_group_ranks = None
        my_dp_rank = None
        my_tp_local_rank = None
        for dp_rank, group_ranks in enumerate(tp_groups):
            if my_rank in group_ranks:
                my_group_ranks = group_ranks
                my_dp_rank = dp_rank
                my_tp_local_rank = group_ranks.index(my_rank)
                break
        assert my_group_ranks is not None, (
            f"Rank {my_rank} not found in tp_groups {tp_groups}"
        )

        my_tp_size = len(my_group_ranks)
        is_tp_leader = my_tp_local_rank == 0
        dp_leaders = [group[0] for group in tp_groups]
        debug_info = _format_het_tp_debug_info(
            my_rank, my_group_ranks, my_tp_local_rank, my_tp_size, my_dp_rank, is_tp_leader
        )

        devices_keyword = "CUDA_VISIBLE_DEVICES"
        my_cuda_devices = os.environ.get(devices_keyword, "")
        logger.info(f"[Het TP] {debug_info} CUDA_VISIBLE_DEVICES={my_cuda_devices}")

        # 2. Create process groups (collective — all ranks participate in every call)
        tp_pg_map = {}
        for group_ranks in tp_groups:
            pg = dist.new_group(ranks=group_ranks, backend="gloo")
            for r in group_ranks:
                tp_pg_map[r] = pg
        my_tp_pg = tp_pg_map[my_rank]

        # DP leaders group (for future use, e.g. DP-level allreduce)
        dp_pg = dist.new_group(ranks=dp_leaders, backend="gloo")

        # 3. Build FakeDeviceMesh instances
        tp_dim = FakeDeviceMeshDim(my_group_ranks, my_tp_local_rank, my_tp_pg)
        pp_dim = FakeDeviceMeshDim([my_rank], 0, None)  # trivial, no collectives
        dp_dim = FakeDeviceMeshDim(dp_leaders, my_dp_rank, dp_pg)

        rollout_device_mesh = FakeDeviceMesh(
            dim_map={"infer_tp": tp_dim, "infer_pp": pp_dim, "dp": dp_dim},
            rank=my_rank,
        )
        cpu_device_mesh = FakeDeviceMesh(
            dim_map={"tp": tp_dim, "pp": pp_dim, "dp": dp_dim},
            rank=my_rank,
        )

        # 4. Register dispatch/collect info
        self._register_dispatch_collect_info(
            "rollout", dp_rank=my_dp_rank, is_collect=is_tp_leader,
        )

        # 5. Random states (seeded by dp_rank for reproducibility within DP group)
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().manual_seed(my_dp_rank + 1000)
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        # 6. Override config tp_size for this group
        rollout_config.tensor_model_parallel_size = my_tp_size

        # 7. Build heterogeneous SGLang rollout
        log_gpu_memory_usage("Before building het SGLang rollout", logger=logger)
        try:
            self.rollout = HetSGLangRollout(
                config=rollout_config,
                model_config=model_config,
                device_mesh=rollout_device_mesh,
                device_mesh_cpu=cpu_device_mesh,
                tp_groups=tp_groups,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to build heterogeneous SGLang rollout for {debug_info}") from exc
        log_gpu_memory_usage("After building het SGLang rollout", logger=logger)

    # ==================================================================
    # NCCL P2P warmup
    # ==================================================================

    def _create_pp_pair_groups(self):
        """Create per-pair gloo groups for PP communication.

        Uses gloo backend to avoid NCCL's lazy communicator init deadlocks.
        Tensors are staged through CPU for send/recv, which adds overhead
        but is reliable. NCCL P2P on pair groups deadlocks because NCCL's
        sub-communicator init requires rank 0 to distribute ncclUniqueId
        via the TCPStore, and the sequential P2P pattern prevents this.
        """
        from .torch_pp.comm import get_pp_pair_groups, set_pp_pair_groups

        existing = get_pp_pair_groups()
        if existing is not None:
            self._pp_pair_groups = existing
            return  # Already created (e.g. init_model called twice for actor+ref)

        rank = self.pp_rank
        world_size = self.pp_size

        pair_groups = {}

        # Create a gloo group for each adjacent pair (world-wide collective)
        for i in range(world_size - 1):
            pair_group = dist.new_group(ranks=[i, i + 1], backend="gloo")
            if rank == i or rank == i + 1:
                pair_groups[i] = pair_group

        # Separate inference pair groups — in fused forward, training backward
        # and inference forward both send rank i+1 → i on the same pair group.
        # Gloo matches sends/recvs in FIFO order per direction, so interleaving
        # them causes size mismatches. Separate groups avoid this.
        for i in range(world_size - 1):
            infer_group = dist.new_group(ranks=[i, i + 1], backend="gloo")
            if rank == i or rank == i + 1:
                pair_groups[f"infer_{i}"] = infer_group

        # Fused forward needs rank 0 ↔ rank P-1 for old_log_probs transfer.
        # For P>=3 these ranks aren't adjacent, so create a dedicated group.
        # Always create it (new_group is a collective — all ranks must call it).
        if world_size >= 3:
            olp_group = dist.new_group(ranks=[0, world_size - 1], backend="gloo")
            if rank == 0 or rank == world_size - 1:
                pair_groups["olp"] = olp_group

        set_pp_pair_groups(pair_groups)
        self._pp_pair_groups = pair_groups

        if rank == 0:
            print(f"[PP] Created {world_size - 1} training + {world_size - 1} inference gloo pair groups", flush=True)

    # ==================================================================
    # init_model
    # ==================================================================

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from transformers import AutoConfig
        from verl.utils import hf_tokenizer
        from verl.utils.model import get_generation_config

        if self.config.model.get("external_lib", None) is not None:
            import importlib
            importlib.import_module(self.config.model.external_lib)

        model_path = self.config.model.path
        local_path = copy_to_local(model_path)
        trust_remote_code = self.config.model.get("trust_remote_code", False)

        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.generation_config = get_generation_config(local_path)

        hf_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        self.hidden_size = hf_config.hidden_size
        self.local_path = local_path

        # Initialize profiling tools
        self._pp_tracer = PPTracer(
            pp_rank=self.pp_rank, pp_size=self.pp_size,
            enabled=self._enable_pp_trace,
            save_dir=os.path.join(self._profiling_save_dir, "traces"),
        )
        max_resp_len = self.config.rollout.get("response_length", 7168)
        max_seq_len = self.config.rollout.get("prompt_length", 1024) + max_resp_len
        self._response_profiler = ResponseLengthProfiler(
            max_response_len=max_resp_len, max_seq_len=max_seq_len,
        )

        log_gpu_memory_usage("Before init actor model and optimizer", logger=logger)

        # ── Create PP pair groups (must be before vLLM init) ──
        # new_group is a world-wide collective, so it must be called when
        # all ranks are synchronized (ONE_TO_ALL dispatch in init_model).
        self._create_pp_pair_groups()

        # ── Training stage + optimizer ──
        if self._is_actor or self._is_rollout:
            _enable_gc = self.config.actor.get("gradient_checkpointing", True)
            self.train_stage = PipelineStage.from_pretrained(
                model_path=local_path,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
                device=self.device,
                dtype=self.dtype,
                trust_remote_code=trust_remote_code,
                enable_gradient_checkpointing=_enable_gc,
            )
            self.train_stage.train()
            log_gpu_memory_usage("After training stage init", logger=logger)

        if self._is_actor:
            lr = self.config.actor.optim.lr if hasattr(self.config.actor, "optim") else 1e-6
            wd = self.config.actor.optim.get("weight_decay", 0.0) if hasattr(self.config.actor, "optim") else 0.0
            self.optimizer = AdamW(self.train_stage.parameters(), lr=lr, weight_decay=wd)
            log_gpu_memory_usage("After actor optimizer init", logger=logger)

            if self._is_offload_param:
                _offload_module(self.train_stage)
                log_gpu_memory_usage("After offload actor params during init", logger=logger)
            if self._is_offload_optimizer:
                _offload_optimizer(self.optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        # ── Rollout ──
        if self._is_rollout:
            self._build_rollout(trust_remote_code=trust_remote_code)
            log_gpu_memory_usage("After rollout init", logger=logger)

            # Switch to trainer mode initially (matches megatron)
            if self._is_actor and self.config.rollout.get("mode", "sync") == "sync":
                loop = get_event_loop()
                loop.run_until_complete(self.trainer_mode())

        # ── Reference stage ──
        if self._is_ref and not self._fused_forward:
            self.ref_stage = PipelineStage.from_pretrained(
                model_path=local_path,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
                device=self.device,
                dtype=self.dtype,
                trust_remote_code=trust_remote_code,
            )
            self.ref_stage.eval()
            self.ref_stage.requires_grad_(False)
            log_gpu_memory_usage("After ref model init", logger=logger)
            if self._ref_is_offload_param:
                _offload_module(self.ref_stage)
                log_gpu_memory_usage("After offload ref params during init", logger=logger)

        # ── Inference stage (fused forward) ──
        if self._fused_forward:
            self.infer_stage = InferenceStage(
                model_path=local_path,
                train_pp_rank=self.pp_rank,
                pp_size=self.pp_size,
                device=self.device,
                dtype=self.dtype,
                trust_remote_code=trust_remote_code,
            )
            log_gpu_memory_usage("After inference stage init", logger=logger)

        get_torch_device().empty_cache()
        log_gpu_memory_usage("After init_model finish", logger=logger)

    # ==================================================================
    # Rollout / trainer mode switching
    # ==================================================================

    async def rollout_mode(self):
        """Context switch to rollout mode — matches megatron rollout_mode."""
        _timing = {}
        aggressive_empty_cache(force_sync=True)

        with simple_timer("rollout_mode/load_actor_to_gpu", _timing):
            if self._is_offload_param:
                _load_module(self.train_stage, self.device)
                log_gpu_memory_usage("After load actor params during rollout_mode", logger=logger)

        with simple_timer("rollout_mode/export_weights", _timing):
            per_tensor_param = self._collect_full_state_dict()

        set_expandable_segments(False)

        with simple_timer("rollout_mode/resume_weights", _timing):
            if self.config.rollout.get("free_cache_engine", True):
                await self.rollout.resume(tags=["weights"])
        with simple_timer("rollout_mode/update_weights", _timing):
            try:
                await self.rollout.update_weights(per_tensor_param)
            except Exception as exc:
                raise RuntimeError(
                    f"rollout.update_weights failed on pp_rank={self.pp_rank}, world_size={self.pp_size}"
                ) from exc
        with simple_timer("rollout_mode/offload_actor", _timing):
            if self._is_offload_param:
                _offload_module(self.train_stage)
        aggressive_empty_cache(force_sync=True)
        with simple_timer("rollout_mode/resume_kv_cache", _timing):
            if self.config.rollout.get("free_cache_engine", True):
                await self.rollout.resume(tags=["kv_cache"])

        # Switch random states
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)

        if torch.distributed.get_rank() == 0:
            print(f"[PROFILING] rollout_mode sub-phases: {_timing}")

    async def trainer_mode(self):
        """Context switch to trainer mode — matches megatron trainer_mode."""
        _timing = {}
        with simple_timer("trainer_mode/release_rollout", _timing):
            if self.config.rollout.get("free_cache_engine", True):
                log_gpu_memory_usage("Before rollout offload", logger=logger)
                await self.rollout.release()
                log_gpu_memory_usage("After rollout offload", logger=logger)

        with simple_timer("trainer_mode/set_train_and_cleanup", _timing):
            if self.train_stage is not None:
                self.train_stage.train()
            aggressive_empty_cache(force_sync=True)

        set_expandable_segments(True)

        # Restore random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        if torch.distributed.get_rank() == 0:
            print(f"[PROFILING] trainer_mode sub-phases: {_timing}")

    # ==================================================================
    # Weight collection
    # ==================================================================

    def _collect_full_state_dict(self):
        """
        Generator yielding (name, tensor) pairs of the full merged model.

        Uses per-tensor NCCL broadcast instead of all_gather_object to avoid
        pickle serialization overhead (~30s → ~1-3s for 1.7B model).

        Phase 1: all_gather_object on metadata only (list of param names/shapes).
        Phase 2: For each param, owning stage broadcasts the tensor to all ranks.
        """
        # Phase 1: collect metadata — which rank owns which params.
        # Use get_global_state_dict for name mapping (local→global layer indices),
        # but keep tensors on GPU for NCCL broadcast (no .cpu() copy).
        from verl.workers.torch_pp.pipeline_stage import restore_global_layer_keys

        # Build local state dict with GPU tensors + global key names
        raw_sd = self.train_stage.state_dict()
        prefix = "model."
        local_sd_cpu_keys = {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in raw_sd.items()
        }
        local_sd = restore_global_layer_keys(local_sd_cpu_keys, self.train_stage.start_layer)

        # Determine which params this rank owns:
        #   embed_tokens → first stage only (non-first stages have stale copies)
        #   norm, lm_head → last stage only (non-last stages have Identity, no params)
        #   layer params → the stage whose [start, end) covers them
        my_meta = []
        my_tensors = {}
        for name, tensor in local_sd.items():
            if "embed_tokens" in name and not self.train_stage.is_first:
                continue
            my_meta.append((name, tuple(tensor.shape), str(tensor.dtype)))
            my_tensors[name] = tensor

        all_meta = [None] * self.pp_size
        dist.all_gather_object(all_meta, my_meta)

        # Phase 2: per-tensor NCCL broadcast.
        # Build ordered list: (owning_rank, name, shape, dtype)
        param_list = []
        for rank, rank_meta in enumerate(all_meta):
            for (name, shape, dtype_str) in rank_meta:
                param_list.append((rank, name, shape, dtype_str))

        dtype_map = {
            "torch.bfloat16": torch.bfloat16,
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
        }

        for owning_rank, name, shape, dtype_str in param_list:
            dt = dtype_map.get(dtype_str, torch.bfloat16)
            if owning_rank == self.pp_rank:
                # Ensure contiguous CPU tensor for broadcast
                tensor_cpu = my_tensors[name].detach().cpu().contiguous()
            else:
                tensor_cpu = torch.empty(shape, dtype=dt, device="cpu")
            # Broadcast on CPU (gloo-compatible, no CUDA IPC issues)
            dist.broadcast(tensor_cpu, src=owning_rank)
            # Move to GPU — creates a fresh CUDA allocation (IPC-compatible)
            yield name, tensor_cpu.to(self.device)

    # ==================================================================
    # generate_sequences — matches megatron generate_sequences exactly
    # ==================================================================

    @register(dispatch_mode=DYNAMIC_INDEX_DISPATCH)
    @GPUMemoryLogger(role="generate_sequences", logger=logger)
    def generate_sequences(self, prompts: DataProto):
        assert self._is_rollout
        prompts = prompts.to(get_device_name())
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
            _offload_optimizer(self.optimizer)

        timing_generate = {}

        # begin_step is called here (earliest phase), but step counter
        # is incremented in update_actor where end_step saves the trace.
        # Use a tentative step number; update_actor will finalize it.
        self._pp_tracer.begin_step(self._step_counter + 1)

        if self._is_actor:
            with self._pp_tracer.phase("load_rollout"):
                with simple_timer("load_rollout", timing_generate):
                    loop = get_event_loop()
                    loop.run_until_complete(self.rollout_mode())
            self._pp_tracer.record_hbm("after_load_rollout")
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)

        with self._pp_tracer.phase("rollout"):
            with simple_timer("generate_sequences", timing_generate):
                try:
                    output = self.rollout.generate_sequences(prompts=prompts)
                except Exception as exc:
                    raise RuntimeError(
                        f"rollout.generate_sequences failed on pp_rank={self.pp_rank}, world_size={self.pp_size}"
                    ) from exc
        self._pp_tracer.record_hbm("after_rollout")

        if self._is_actor:
            with self._pp_tracer.phase("unload_rollout"):
                with simple_timer("unload_rollout", timing_generate):
                    loop.run_until_complete(self.trainer_mode())
            self._pp_tracer.record_hbm("after_unload_rollout")
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)

        tp_groups = self.config.rollout.get("tp_groups", None)
        if tp_groups is not None:
            # Heterogeneous TP can intentionally create long per-group rollout skew.
            # Avoid post-generation world collectives here: faster workers may reach
            # the reduction while a slower group is still decoding, which trips the
            # default process-group timeout. Keep timing metadata present so the
            # trainer path remains unchanged.
            timing_generate = {}
        else:
            # Average timing across all ranks (same as megatron)
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

        # ── Response length profiling ──
        # NOTE: Don't put per-worker metrics into output.meta_info — DP workers
        # see different samples, causing conflicts in DataProto.concat().
        # Save to disk only; wandb gets response_length stats from the trainer.
        if self._enable_response_profiling and self._response_profiler is not None:
            self._response_profiler.reset()
            resp_lens = None
            if "response_mask" in output.batch:
                resp_lens = output.batch["response_mask"].sum(dim=-1)
            elif "response_lengths" in output.non_tensor_batch:
                resp_lens = output.non_tensor_batch["response_lengths"]
            elif "attention_mask" in output.batch:
                resp_lens = output.batch["attention_mask"].sum(dim=-1)

            if resp_lens is not None:
                prompt_ids = output.non_tensor_batch.get("index", None)
                prompt_lens = None
                if "attention_mask" in output.batch and "response_mask" in output.batch:
                    prompt_lens = (
                        output.batch["attention_mask"].sum(dim=-1)
                        - output.batch["response_mask"].sum(dim=-1)
                    )
                self._response_profiler.record(resp_lens, prompt_ids, prompt_lens)
                # Save per-step raw data for offline analysis (each DP rank saves its shard)
                self._response_profiler.save_step(
                    os.path.join(self._profiling_save_dir, "response_lengths"),
                    self._step_counter + 1,
                )
                if torch.distributed.get_rank() == 0:
                    resp_metrics = self._response_profiler.compute_metrics()
                    print(f"[PROFILING] response_length: mean={resp_metrics.get('response_length/mean', 0):.0f}, "
                          f"p90={resp_metrics.get('response_length/p90', 0):.0f}, "
                          f"waste={resp_metrics.get('response_length/padding_waste', 0):.1%}")

        output = output.to("cpu")
        aggressive_empty_cache(force_sync=True)
        return output

    # ==================================================================
    # compute_log_prob — matches megatron compute_log_prob
    # ==================================================================

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="compute_log_prob", logger=logger)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        timing_log_prob = {}
        self._pp_tracer.record_hbm("before_compute_log_prob")

        # Pair groups were created in init_model (before vLLM init)
        assert self._pp_pair_groups is not None, "PP pair groups not initialized"

        with self._pp_tracer.phase("load_inference"):
            with simple_timer("load_inference", timing_log_prob):
                if self._is_offload_param:
                    _load_module(self.train_stage, self.device)
                    log_gpu_memory_usage("After load actor params during compute_log_prob", logger=logger)
        self._pp_tracer.record_hbm("after_load_inference")

        self.train_stage.eval()
        data = data.to(get_device_name())

        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        B, S = input_ids.shape

        print(f"[PP rank {self.pp_rank}] compute_log_prob B={B}, S={S}, is_first={self.train_stage.is_first}, is_last={self.train_stage.is_last}", flush=True)

        # Determine micro-batch count
        mbs_per_gpu = self.config.rollout.get("log_prob_micro_batch_size_per_gpu", None)
        if mbs_per_gpu and mbs_per_gpu > 0:
            M = max(1, B // mbs_per_gpu)
        else:
            M = self._num_micro_batches
        while B % M != 0 and M > 1:
            M -= 1

        print(f"[PP rank {self.pp_rank}] compute_log_prob M={M}, about to call _pp_forward_log_prob", flush=True)

        with self._pp_tracer.phase("compute_log_prob"):
            with simple_timer("compute_log_prob", timing_log_prob):
                full_log_probs, full_entropy = _pp_forward_log_prob(
                    stage=self.train_stage,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    M=M,
                    hidden_size=self.hidden_size,
                    pp_rank=self.pp_rank,
                    device=self.device,
                    calculate_entropy=True,
                    pp_pair_groups=self._pp_pair_groups,
                    use_remove_padding=self._use_remove_padding,
                    tracer=self._pp_tracer,
                )
        self._pp_tracer.record_hbm("after_compute_log_prob")

        self.train_stage.train()

        if self.train_stage.is_last:
            response_mask = data.batch["response_mask"]
            old_log_probs = _extract_response_log_probs(full_log_probs, attention_mask, response_mask)
            entropys = _extract_response_log_probs(full_entropy, attention_mask, response_mask)
            output = DataProto.from_dict(
                tensors={"old_log_probs": old_log_probs, "entropys": entropys},
            )
            output = output.to("cpu")
        else:
            output = DataProto(meta_info={})

        with self._pp_tracer.phase("unload_inference"):
            with simple_timer("unload_inference", timing_log_prob):
                if self._is_offload_param:
                    _offload_module(self.train_stage)
                    log_gpu_memory_usage("After offload actor params during compute_log_prob", logger=logger)
        self._pp_tracer.record_hbm("after_unload_inference")

        aggressive_empty_cache(force_sync=True)
        timing_log_prob = reduce_timing(timing_log_prob)
        if torch.distributed.get_rank() == 0:
            print(f"[PROFILING] compute_log_prob timing: {timing_log_prob}")
        output.meta_info["timing"] = {
            f"compute_log_prob/{k}": v for k, v in timing_log_prob.items()
        }
        return output

    # ==================================================================
    # compute_ref_log_prob — matches megatron compute_ref_log_prob
    # ==================================================================

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="compute_ref_log_prob", logger=logger)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref
        timing_ref = {}
        ref_only_trace = self._is_ref and not self._is_actor
        if ref_only_trace:
            self._step_counter += 1
            trace_offset_us = float(data.meta_info.get("trace_time_offset_us", 0.0))
            self._pp_tracer.begin_step(self._step_counter, time_offset_us=trace_offset_us)
        self._pp_tracer.record_hbm("before_compute_ref_log_prob")

        # Pair groups were created in init_model (before vLLM init)
        assert self._pp_pair_groups is not None, "PP pair groups not initialized"

        with self._pp_tracer.phase("load_ref"):
            with simple_timer("load_ref", timing_ref):
                if self._ref_is_offload_param:
                    _load_module(self.ref_stage, self.device)
                    log_gpu_memory_usage("After load ref params during compute_ref_log_prob", logger=logger)
        self._pp_tracer.record_hbm("after_load_ref")

        data = data.to(get_device_name())

        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        B, S = input_ids.shape

        mbs_per_gpu = self.config.ref.get("log_prob_micro_batch_size_per_gpu", None)
        if mbs_per_gpu and mbs_per_gpu > 0:
            M = max(1, B // mbs_per_gpu)
        else:
            M = self._num_micro_batches
        while B % M != 0 and M > 1:
            M -= 1

        with self._pp_tracer.phase("compute_ref_log_prob"):
            with simple_timer("compute_ref_log_prob", timing_ref):
                full_log_probs, _ = _pp_forward_log_prob(
                    stage=self.ref_stage,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    M=M,
                    hidden_size=self.hidden_size,
                    pp_rank=self.pp_rank,
                    device=self.device,
                    calculate_entropy=False,
                    pp_pair_groups=self._pp_pair_groups,
                    use_remove_padding=self._use_remove_padding,
                    tracer=self._pp_tracer,
                )
        self._pp_tracer.record_hbm("after_compute_ref_log_prob")

        if self.ref_stage.is_last:
            response_mask = data.batch["response_mask"]
            ref_log_probs = _extract_response_log_probs(full_log_probs, attention_mask, response_mask)
            output = DataProto.from_dict(tensors={"ref_log_prob": ref_log_probs})
            output = output.to("cpu")
        else:
            output = DataProto(meta_info={})

        with self._pp_tracer.phase("unload_ref"):
            with simple_timer("unload_ref", timing_ref):
                if self._ref_is_offload_param:
                    _offload_module(self.ref_stage)
                    log_gpu_memory_usage("After offload ref params during compute_ref_log_prob", logger=logger)
        self._pp_tracer.record_hbm("after_unload_ref")

        aggressive_empty_cache(force_sync=True)
        timing_ref = reduce_timing(timing_ref)
        if torch.distributed.get_rank() == 0:
            print(f"[PROFILING] compute_ref_log_prob timing: {timing_ref}")
        output.meta_info["timing"] = {
            f"compute_ref_log_prob/{k}": v for k, v in timing_ref.items()
        }
        if ref_only_trace:
            self._pp_tracer.end_step()
        return output

    # ==================================================================
    # update_actor — matches megatron update_actor
    # ==================================================================

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="update_actor", logger=logger)
    def update_actor(self, data: DataProto):
        assert self._is_actor
        timing_update = {}

        # Pair groups were created in init_model (before vLLM init)
        assert self._pp_pair_groups is not None, "PP pair groups not initialized"

        with self._pp_tracer.phase("load_training"):
            with simple_timer("load_training", timing_update):
                if self._is_offload_param:
                    _load_module(self.train_stage, self.device)
                    log_gpu_memory_usage("After load actor params and grad during update_actor", logger=logger)
                if self._is_offload_optimizer:
                    _load_optimizer(self.optimizer, self.device)
                log_gpu_memory_usage("After load actor optimizer during update_actor", logger=logger)

        self.train_stage.train()
        data = data.to(get_device_name())

        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        old_log_probs_full = data.batch["old_log_probs"]
        advantages = data.batch["advantages"]
        response_mask = data.batch["response_mask"]
        ref_log_probs_full = data.batch.get("ref_log_prob", None)

        B, S = input_ids.shape
        R = response_mask.size(1)

        prompt_lengths = attention_mask.sum(dim=-1) - response_mask.sum(dim=-1)
        response_start_positions = prompt_lengths.long()

        M = self._num_micro_batches
        while B % M != 0 and M > 1:
            M -= 1
        micro_B = B // M

        self._step_counter += 1
        self._pp_tracer._step = self._step_counter  # finalize step number
        self._pp_tracer.record_hbm("before_update_actor")

        with self._pp_tracer.phase("update_actor"), simple_timer("update_policy", timing_update):
            self.train_stage.set_batch_data(
                input_ids, attention_mask, M,
                use_remove_padding=self._use_remove_padding,
            )
            self.optimizer.zero_grad()

            # Pre-chunk loss inputs for last stage
            micro_shifted_labels = None
            if self.train_stage.is_last:
                micro_input_ids = list(input_ids.chunk(M, dim=0))
                micro_resp_starts = list(response_start_positions.chunk(M, dim=0))
                micro_old_lp = list(old_log_probs_full.chunk(M, dim=0))
                micro_adv = list(advantages.chunk(M, dim=0))
                micro_resp_mask = list(response_mask.chunk(M, dim=0))
                micro_ref_lp = list(ref_log_probs_full.chunk(M, dim=0)) if ref_log_probs_full is not None else None

                # When rmpad is active, build shifted labels that match
                # compute_log_prob's convention: within each sequence,
                # label[i] = input_ids[i+1]; last non-pad position gets 0.
                # This avoids the torch.roll mismatch where compute_log_prob
                # uses label=0 but torch.roll uses label=pad/EOS at that position.
                if self._use_remove_padding:
                    micro_masks = list(attention_mask.chunk(M, dim=0))
                    micro_shifted_labels = []
                    for i in range(M):
                        ids_mb = micro_input_ids[i]        # [micro_B, S]
                        mask_mb = micro_masks[i]            # [micro_B, S]
                        shifted = torch.roll(ids_mb, shifts=-1, dims=-1)
                        # Fix last non-pad position per sequence: set label to 0
                        seq_lengths = mask_mb.sum(dim=-1)   # [micro_B]
                        # Find the column of the last non-pad token
                        # For right-padded: last_col = first_nonzero + seq_len - 1
                        # General: last non-pad = last index where mask=1
                        for b in range(ids_mb.size(0)):
                            sl = seq_lengths[b].item()
                            if sl > 0:
                                last_col = mask_mb[b].nonzero()[-1].item()
                                shifted[b, last_col] = 0
                        micro_shifted_labels.append(shifted)

            schedule = build_1f1b_schedule(self.pp_rank, self.pp_size, M)

            all_stats = []
            pending_sends = []

            # Keep refs to CPU send buffers to prevent GC before isend completes
            _send_bufs = []

            def _p2p_send(tensor, dst_rank):
                """Non-blocking send via gloo pair group (CPU-staged)."""
                t = tensor.detach().contiguous().cpu()
                _send_bufs.append(t)  # prevent GC
                pair_idx = min(self.pp_rank, dst_rank)
                group = self._pp_pair_groups[pair_idx]
                return dist.isend(t, dst=dst_rank, group=group)

            def _p2p_recv(shape, src_rank, dtype):
                """Recv via gloo pair group (CPU-staged)."""
                buf = torch.empty(shape, dtype=dtype, device="cpu")
                pair_idx = min(self.pp_rank, src_rank)
                group = self._pp_pair_groups[pair_idx]
                dist.recv(buf, src=src_rank, group=group)
                return buf.to(self.device)

            tracer = self._pp_tracer

            for op in schedule:
                mb = op.micro_batch_id

                if op.op == "forward":
                    input_hidden = None
                    if not self.train_stage.is_first:
                        nnz = self.train_stage._micro_nnz[mb]
                        if self._use_remove_padding:
                            act_shape = (1, nnz, self.hidden_size)
                        else:
                            act_shape = (micro_B, S, self.hidden_size)
                        with tracer.trace("p2p_recv", mb):
                            input_hidden = _p2p_recv(act_shape, self.pp_rank - 1, self.train_stage.dtype)
                        input_hidden.requires_grad_(True)

                    with tracer.trace("train_forward", mb):
                        output = self.train_stage.forward_step(
                            mb, input_hidden,
                            return_hidden=self.train_stage.is_last,
                        )

                    if self.train_stage.is_last:
                        with tracer.trace("loss", mb):
                            if (self._use_remove_padding
                                    and self.train_stage._micro_cu_seqlens):
                                # === Packed-format loss path ===
                                # Compute log_probs on packed hidden (same as compute_log_prob)
                                # then re-pad log_probs (NOT hidden_states) to ensure
                                # padding positions get 0 — matching compute_log_prob exactly.
                                from flash_attn.bert_padding import pad_input
                                from verl.utils.experimental.torch_functional import FusedLinearForPPO

                                cu = self.train_stage._micro_cu_seqlens[mb]
                                micro_ids_packed = self.train_stage._micro_input_ids[mb]  # [1, nnz]
                                ids_flat = micro_ids_packed.squeeze(0)  # [nnz]

                                # Build shifted labels (same convention as compute_log_prob)
                                shifted = torch.empty_like(ids_flat)
                                for j in range(len(cu) - 1):
                                    s, e = cu[j].item(), cu[j + 1].item()
                                    shifted[s:e - 1] = ids_flat[s + 1:e]
                                    shifted[e - 1] = 0

                                fused_loss = FusedLinearForPPO(chunk_size=512)
                                lp, ent = fused_loss.forward(
                                    hidden_states=output,  # [1, nnz, H]
                                    vocab_weights=self.train_stage.lm_head_weight,
                                    input_ids=shifted.unsqueeze(0),
                                    temperature=1.0,
                                )  # lp, ent: [1, nnz]

                                # Extract valid positions (drop last per seq)
                                valid_lp_parts = []
                                for j in range(len(cu) - 1):
                                    s, e = cu[j].item(), cu[j + 1].item()
                                    valid_lp_parts.append(lp[0, s:e - 1])
                                valid_lp = torch.cat(valid_lp_parts)  # [nnz - nseqs]

                                # Re-pad to [micro_B, S-1] — same as compute_log_prob
                                _mb_B, _mb_S = self.train_stage._micro_batch_shape
                                unpad_indices = self.train_stage._micro_unpad_indices[mb]
                                seq_lens_list = cu.diff().tolist()
                                shifted_indices = []
                                offset = 0
                                for sl in seq_lens_list:
                                    shifted_indices.append(unpad_indices[offset:offset + sl - 1])
                                    offset += sl
                                shifted_idx = torch.cat(shifted_indices) if shifted_indices else unpad_indices[:0]
                                adjusted_idx = (shifted_idx // _mb_S) * (_mb_S - 1) + (shifted_idx % _mb_S)

                                new_full_lp = pad_input(
                                    valid_lp.unsqueeze(-1), adjusted_idx, _mb_B, _mb_S - 1
                                ).squeeze(-1)  # [micro_B, S-1]

                                # Extract response portion — same as _extract_response_log_probs
                                R = micro_old_lp[mb].size(1)
                                new_resp_lp = gather_response_log_probs(
                                    new_full_lp, micro_resp_starts[mb], R
                                )
                                new_resp_lp = new_resp_lp * micro_resp_mask[mb]

                                # GRPO loss from pre-computed response log_probs
                                loss, stats = _compute_grpo_loss_from_resp_lp(
                                    new_resp_lp=new_resp_lp,
                                    old_log_probs=micro_old_lp[mb],
                                    advantages=micro_adv[mb],
                                    response_mask=micro_resp_mask[mb],
                                    ref_log_probs=micro_ref_lp[mb] if micro_ref_lp else None,
                                    clip_ratio=self.config.actor.clip_ratio,
                                    kl_coef=self.config.actor.get("kl_loss_coef", 0.001),
                                    entropy_coef=self.config.actor.get("entropy_coeff", 0.0),
                                )
                            else:
                                # === Standard padded loss path ===
                                loss, stats = compute_grpo_loss_fused(
                                    hidden_states=output,
                                    lm_head_weight=self.train_stage.lm_head_weight,
                                    input_ids=micro_input_ids[mb],
                                    response_start_positions=micro_resp_starts[mb],
                                    old_log_probs=micro_old_lp[mb],
                                    advantages=micro_adv[mb],
                                    response_mask=micro_resp_mask[mb],
                                    ref_log_probs=micro_ref_lp[mb] if micro_ref_lp else None,
                                    clip_ratio=self.config.actor.clip_ratio,
                                    kl_coef=self.config.actor.get("kl_loss_coef", 0.001),
                                    entropy_coef=self.config.actor.get("entropy_coeff", 0.0),
                                )
                            scaled_loss = loss / M
                            scaled_loss.backward()
                        all_stats.append(stats)
                    else:
                        with tracer.trace("p2p_send", mb):
                            handle = _p2p_send(output.detach(), self.pp_rank + 1)
                        pending_sends.append(handle)

                elif op.op == "backward":
                    if self.train_stage.is_last:
                        with tracer.trace("train_backward", mb):
                            input_grad = self.train_stage.backward_step(mb, grad_output=None)
                        if input_grad is not None:
                            with tracer.trace("p2p_send", mb):
                                handle = _p2p_send(input_grad, self.pp_rank - 1)
                            pending_sends.append(handle)
                    else:
                        nnz = self.train_stage._micro_nnz[mb]
                        if self._use_remove_padding:
                            act_shape = (1, nnz, self.hidden_size)
                        else:
                            act_shape = (micro_B, S, self.hidden_size)
                        with tracer.trace("p2p_recv", mb):
                            grad = _p2p_recv(act_shape, self.pp_rank + 1, self.train_stage.dtype)
                        with tracer.trace("train_backward", mb):
                            input_grad = self.train_stage.backward_step(mb, grad_output=grad)

                        if not self.train_stage.is_first and input_grad is not None:
                            with tracer.trace("p2p_send", mb):
                                handle = _p2p_send(input_grad, self.pp_rank - 1)
                            pending_sends.append(handle)

            for handle in pending_sends:
                handle.wait()

            with tracer.trace("optimizer"):
                grad_clip = self.config.actor.get("grad_clip", 1.0)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.train_stage.parameters(), grad_clip)
                self.optimizer.step()
            self.train_stage.clear_batch_data()

        self._pp_tracer.record_hbm("after_update_actor")

        # Build metrics (matches megatron output format)
        metrics = {}
        if self._enable_pp_trace:
            metrics.update(self._pp_tracer.summary())
        if all_stats:
            for key in all_stats[0]:
                vals = [s[key] for s in all_stats]
                metrics[f"actor/{key}"] = sum(vals) / len(vals)
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        with self._pp_tracer.phase("unload_training"):
            with simple_timer("unload_training", timing_update):
                if self._is_offload_param:
                    _offload_module(self.train_stage)
                    log_gpu_memory_usage("After offload actor params and grad during update_actor", logger=logger)
                if self._is_offload_optimizer:
                    _offload_optimizer(self.optimizer)
                    log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)
        self._pp_tracer.record_hbm("after_unload_training")

        # Save trace AFTER all phases including unload
        self._pp_tracer.end_step()

        aggressive_empty_cache(force_sync=True)
        timing_update = reduce_timing(timing_update)
        if torch.distributed.get_rank() == 0:
            print(f"[PROFILING] update_actor timing: {timing_update}")
        return output

    # ==================================================================
    # fused_update_actor — fused forward path
    # ==================================================================

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="fused_update_actor", logger=logger)
    def fused_update_actor(self, data: DataProto):
        """
        Fused forward: interleave training forward/backward with inference
        forward in pipeline bubbles. Returns old_log_probs + training metrics.
        """
        assert self._is_actor and self._fused_forward
        timing_fused = {}

        # Pair groups were created in init_model (before vLLM init)
        assert self._pp_pair_groups is not None, "PP pair groups not initialized"

        self._step_counter += 1
        self._pp_tracer._step = self._step_counter
        self._pp_tracer.record_hbm("before_fused_update_actor")

        with self._pp_tracer.phase("load_training"), simple_timer("load_training", timing_fused):
            if self._is_offload_param:
                _load_module(self.train_stage, self.device)
                log_gpu_memory_usage("After load actor params during fused_update_actor", logger=logger)
            if self._is_offload_optimizer:
                _load_optimizer(self.optimizer, self.device)
                log_gpu_memory_usage("After load actor optimizer during fused_update_actor", logger=logger)
            if self.infer_stage is not None:
                self.infer_stage.to(self.device)
                log_gpu_memory_usage("After load infer stage during fused_update_actor", logger=logger)

        self.train_stage.train()
        data = data.to(get_device_name())

        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        advantages = data.batch["advantages"]
        response_mask = data.batch["response_mask"]
        ref_log_probs_full = data.batch.get("ref_log_prob", None)

        B, S = input_ids.shape
        R = response_mask.size(1)

        prompt_lengths = attention_mask.sum(dim=-1) - response_mask.sum(dim=-1)
        response_start_positions = prompt_lengths.long()

        M = self._num_micro_batches
        while B % M != 0 and M > 1:
            M -= 1
        micro_B = B // M

        tracer = self._pp_tracer

        with tracer.phase("fused_update_actor"), simple_timer("fused_update_policy", timing_fused):
            # Set batch data on BOTH stages
            self.train_stage.set_batch_data(
                input_ids, attention_mask, M,
                use_remove_padding=self._use_remove_padding,
            )
            self.infer_stage.set_batch_data(
                input_ids, attention_mask, M,
                use_remove_padding=self._use_remove_padding,
            )

            # Pre-chunk loss inputs for last training stage
            is_last_train = self.train_stage.is_last
            if is_last_train:
                micro_input_ids = list(input_ids.chunk(M, dim=0))
                micro_resp_starts = list(response_start_positions.chunk(M, dim=0))
                micro_adv = list(advantages.chunk(M, dim=0))
                micro_resp_mask = list(response_mask.chunk(M, dim=0))
                micro_ref_lp = list(ref_log_probs_full.chunk(M, dim=0)) if ref_log_probs_full is not None else None

            # Pre-chunk for last inference stage
            is_last_infer = self.infer_stage.is_last_infer
            micro_infer_resp_starts = None
            if is_last_infer:
                micro_infer_resp_starts = list(response_start_positions.chunk(M, dim=0))

            self.optimizer.zero_grad()

            fused_schedules = build_default_fused_schedule(self.pp_size, M)
            schedule_ops = parse_schedule(fused_schedules[self.pp_rank])

            all_stats: List[Dict[str, float]] = []
            pending_sends: List = []
            _send_bufs: List = []  # prevent GC before isend completes

            def _get_pair_group(rank_a, rank_b, is_infer=False):
                """Look up the gloo pair group for two ranks."""
                if abs(rank_a - rank_b) == 1:
                    key = f"infer_{min(rank_a, rank_b)}" if is_infer else min(rank_a, rank_b)
                    return self._pp_pair_groups[key]
                # Non-adjacent: old_log_probs group (rank 0 ↔ P-1)
                return self._pp_pair_groups["olp"]

            def _p2p_send(tensor, dst_rank, is_infer=False):
                """Non-blocking send via gloo pair group (CPU-staged)."""
                t = tensor.detach().contiguous().cpu()
                _send_bufs.append(t)
                group = _get_pair_group(self.pp_rank, dst_rank, is_infer=is_infer)
                return dist.isend(t, dst=dst_rank, group=group)

            def _p2p_recv(shape, src_rank, dtype, is_infer=False):
                """Recv via gloo pair group (CPU-staged)."""
                buf = torch.empty(shape, dtype=dtype, device="cpu")
                group = _get_pair_group(self.pp_rank, src_rank, is_infer=is_infer)
                dist.recv(buf, src=src_rank, group=group)
                return buf.to(self.device)
            old_log_probs_stash: Dict[int, torch.Tensor] = {}
            all_old_log_probs = [None] * M
            _infer_offloaded = False  # Track phase transition

            P = self.pp_size

            for op in schedule_ops:
                mb = op.micro_batch_id

                # Phase transition: offload inference weights before first tB
                # The constrained schedule guarantees all iF are done by now.
                if op.op == "train_backward" and not _infer_offloaded:
                    if self.infer_stage is not None and self._is_offload_param:
                        with tracer.trace("offload_infer"):
                            self.infer_stage.to("cpu")
                    _infer_offloaded = True

                if op.op == "train_forward":
                    input_hidden = None
                    if not self.train_stage.is_first:
                        nnz = self.train_stage._micro_nnz[mb]
                        act_shape = (1, nnz, self.hidden_size) if self._use_remove_padding else (micro_B, S, self.hidden_size)
                        with tracer.trace("p2p_recv", mb):
                            input_hidden = _p2p_recv(act_shape, self.pp_rank - 1, self.train_stage.dtype)
                        input_hidden.requires_grad_(True)

                    with tracer.trace("train_forward", mb):
                        output = self.train_stage.forward_step(
                            mb, input_hidden,
                            return_hidden=is_last_train,
                        )

                    if is_last_train:
                        if P == 1:
                            old_lp = old_log_probs_stash.pop(mb)
                        else:
                            with tracer.trace("p2p_recv", mb):
                                olp_shape = (micro_B, R)
                                old_lp = _p2p_recv(olp_shape, 0, self.train_stage.dtype)

                        all_old_log_probs[mb] = old_lp.detach()

                        with tracer.trace("loss", mb):
                            if (self._use_remove_padding
                                    and self.train_stage._micro_cu_seqlens):
                                # === Packed-format loss path ===
                                from flash_attn.bert_padding import pad_input
                                from verl.utils.experimental.torch_functional import FusedLinearForPPO

                                cu = self.train_stage._micro_cu_seqlens[mb]
                                micro_ids_packed = self.train_stage._micro_input_ids[mb]
                                ids_flat = micro_ids_packed.squeeze(0)  # [nnz]

                                shifted = torch.empty_like(ids_flat)
                                for j in range(len(cu) - 1):
                                    s, e = cu[j].item(), cu[j + 1].item()
                                    shifted[s:e - 1] = ids_flat[s + 1:e]
                                    shifted[e - 1] = 0

                                fused_loss = FusedLinearForPPO(chunk_size=512)
                                lp, _ent = fused_loss.forward(
                                    hidden_states=output,
                                    vocab_weights=self.train_stage.lm_head_weight,
                                    input_ids=shifted.unsqueeze(0),
                                    temperature=1.0,
                                )  # [1, nnz]

                                valid_lp_parts = []
                                for j in range(len(cu) - 1):
                                    s, e = cu[j].item(), cu[j + 1].item()
                                    valid_lp_parts.append(lp[0, s:e - 1])
                                valid_lp = torch.cat(valid_lp_parts)

                                _mb_B, _mb_S = self.train_stage._micro_batch_shape
                                unpad_indices = self.train_stage._micro_unpad_indices[mb]
                                seq_lens_list = cu.diff().tolist()
                                shifted_indices = []
                                offset = 0
                                for sl in seq_lens_list:
                                    shifted_indices.append(unpad_indices[offset:offset + sl - 1])
                                    offset += sl
                                shifted_idx = (torch.cat(shifted_indices) if shifted_indices
                                               else unpad_indices[:0])
                                adjusted_idx = ((shifted_idx // _mb_S) * (_mb_S - 1)
                                                + (shifted_idx % _mb_S))

                                new_full_lp = pad_input(
                                    valid_lp.unsqueeze(-1), adjusted_idx, _mb_B, _mb_S - 1
                                ).squeeze(-1)  # [micro_B, S-1]

                                new_resp_lp = gather_response_log_probs(
                                    new_full_lp, micro_resp_starts[mb], R
                                )
                                new_resp_lp = new_resp_lp * micro_resp_mask[mb]

                                loss, stats = _compute_grpo_loss_from_resp_lp(
                                    new_resp_lp=new_resp_lp,
                                    old_log_probs=old_lp,
                                    advantages=micro_adv[mb],
                                    response_mask=micro_resp_mask[mb],
                                    ref_log_probs=micro_ref_lp[mb] if micro_ref_lp else None,
                                    clip_ratio=self.config.actor.clip_ratio,
                                    kl_coef=self.config.actor.get("kl_loss_coef", 0.001),
                                    entropy_coef=self.config.actor.get("entropy_coeff", 0.0),
                                )
                            else:
                                # === Standard padded loss path ===
                                loss, stats = compute_grpo_loss_fused(
                                    hidden_states=output,
                                    lm_head_weight=self.train_stage.lm_head_weight,
                                    input_ids=micro_input_ids[mb],
                                    response_start_positions=micro_resp_starts[mb],
                                    old_log_probs=old_lp,
                                    advantages=micro_adv[mb],
                                    response_mask=micro_resp_mask[mb],
                                    ref_log_probs=micro_ref_lp[mb] if micro_ref_lp else None,
                                    clip_ratio=self.config.actor.clip_ratio,
                                    kl_coef=self.config.actor.get("kl_loss_coef", 0.001),
                                    entropy_coef=self.config.actor.get("entropy_coeff", 0.0),
                                )
                            scaled_loss = loss / M
                            scaled_loss.backward()
                        all_stats.append(stats)
                    else:
                        with tracer.trace("p2p_send", mb):
                            handle = _p2p_send(output.detach(), self.pp_rank + 1)
                        pending_sends.append(handle)

                elif op.op == "train_backward":
                    if is_last_train:
                        with tracer.trace("train_backward", mb):
                            input_grad = self.train_stage.backward_step(mb, grad_output=None)
                        if input_grad is not None:
                            with tracer.trace("p2p_send", mb):
                                handle = _p2p_send(input_grad, self.pp_rank - 1)
                            pending_sends.append(handle)
                    else:
                        nnz = self.train_stage._micro_nnz[mb]
                        act_shape = (1, nnz, self.hidden_size) if self._use_remove_padding else (micro_B, S, self.hidden_size)
                        with tracer.trace("p2p_recv", mb):
                            grad = _p2p_recv(act_shape, self.pp_rank + 1, self.train_stage.dtype)
                        with tracer.trace("train_backward", mb):
                            input_grad = self.train_stage.backward_step(mb, grad_output=grad)

                        if not self.train_stage.is_first and input_grad is not None:
                            with tracer.trace("p2p_send", mb):
                                handle = _p2p_send(input_grad, self.pp_rank - 1)
                            pending_sends.append(handle)

                elif op.op == "infer_forward":
                    input_hidden = None
                    if not self.infer_stage.is_first_infer:
                        nnz = self.infer_stage._micro_nnz[mb]
                        act_shape = (1, nnz, self.hidden_size) if self._use_remove_padding else (micro_B, S, self.hidden_size)
                        with tracer.trace("p2p_recv", mb):
                            input_hidden = _p2p_recv(act_shape, self.pp_rank + 1, self.infer_stage.dtype, is_infer=True)

                    with tracer.trace("infer_forward", mb):
                        output = self.infer_stage.forward_step(
                            mb, input_hidden,
                            return_hidden=is_last_infer,
                        )

                    if is_last_infer:
                        with tracer.trace("infer_log_probs", mb):
                            log_probs = self.infer_stage.compute_log_probs_fused(
                                micro_batch_id=mb,
                                hidden_states=output,
                                response_start_positions=micro_infer_resp_starts[mb],
                                max_resp_len=R,
                            )
                        if P == 1:
                            old_log_probs_stash[mb] = log_probs
                        else:
                            with tracer.trace("p2p_send", mb):
                                handle = _p2p_send(log_probs, P - 1)
                            pending_sends.append(handle)
                    else:
                        with tracer.trace("p2p_send", mb):
                            handle = _p2p_send(output, self.pp_rank - 1, is_infer=True)
                        pending_sends.append(handle)

            for handle in pending_sends:
                handle.wait()

            with tracer.trace("optimizer"):
                grad_clip = self.config.actor.get("grad_clip", 1.0)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.train_stage.parameters(), grad_clip)
                self.optimizer.step()

            self.train_stage.clear_batch_data()
            self.infer_stage.clear_batch_data()

        tracer.record_hbm("after_fused_update_actor")

        # Build metrics
        metrics = {}
        if self._enable_pp_trace:
            metrics.update(tracer.summary())
        if all_stats:
            for key in all_stats[0]:
                vals = [s[key] for s in all_stats]
                metrics[f"actor/{key}"] = sum(vals) / len(vals)
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

        if is_last_train:
            old_log_probs_cat = torch.cat(all_old_log_probs, dim=0)
            output = DataProto.from_dict(
                tensors={"old_log_probs": old_log_probs_cat},
                meta_info={"metrics": metrics},
            )
        else:
            output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        with tracer.phase("unload_training"), simple_timer("unload_training", timing_fused):
            if self._is_offload_param:
                _offload_module(self.train_stage)
                log_gpu_memory_usage("After offload actor params during fused_update_actor", logger=logger)
            if self._is_offload_optimizer:
                _offload_optimizer(self.optimizer)
                log_gpu_memory_usage("After offload actor optimizer during fused_update_actor", logger=logger)
            if self.infer_stage is not None and self._is_offload_param and not _infer_offloaded:
                _offload_module(self.infer_stage)
                log_gpu_memory_usage("After offload infer stage during fused_update_actor", logger=logger)
        tracer.record_hbm("after_unload_fused")

        # Save trace AFTER all phases including unload
        tracer.end_step()

        aggressive_empty_cache(force_sync=True)
        timing_fused = reduce_timing(timing_fused)
        if torch.distributed.get_rank() == 0:
            print(f"[PROFILING] fused_update_actor timing: {timing_fused}")
        return output

    # ==================================================================
    # Checkpoint stubs — matches megatron interface
    # ==================================================================

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, checkpoint_path, hdfs_path=None, del_local_after_load=True):
        if checkpoint_path is None:
            if self._is_offload_param and self.train_stage is not None:
                _offload_module(self.train_stage)
            if self._is_offload_optimizer and self.optimizer is not None:
                _offload_optimizer(self.optimizer)
            log_gpu_memory_usage("After offload during load_checkpoint", logger=logger)
            return

        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(os.getcwd(), checkpoint_path)

        model_path = None
        safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
        torch_path = os.path.join(checkpoint_path, "model.pt")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file

            model_state = load_file(safetensors_path, device="cpu")
            model_path = safetensors_path
        elif os.path.exists(torch_path):
            model_state = torch.load(torch_path, map_location="cpu")
            model_path = torch_path
        else:
            raise FileNotFoundError(f"No checkpoint model file found under {checkpoint_path}")

        if self.train_stage is not None:
            self.train_stage.load_state_dict_from_full(model_state)
        if self.ref_stage is not None:
            self.ref_stage.load_state_dict_from_full(model_state)

        if self.optimizer is not None:
            optim_path = os.path.join(checkpoint_path, f"optimizer_rank{self.pp_rank}.pt")
            if os.path.exists(optim_path):
                optimizer_state = torch.load(optim_path, map_location="cpu")
                self.optimizer.load_state_dict(optimizer_state)

        if self._is_offload_param and self.train_stage is not None:
            _offload_module(self.train_stage)
        if getattr(self, "_ref_is_offload_param", False) and self.ref_stage is not None:
            _offload_module(self.ref_stage)
        if self._is_offload_optimizer and self.optimizer is not None:
            _offload_optimizer(self.optimizer)

        dist.barrier()
        log_gpu_memory_usage(f"After load_checkpoint from {model_path}", logger=logger)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_pretrained_model(self, checkpoint_path, del_local_after_load=True):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        if self.train_stage is None:
            return

        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(os.getcwd(), checkpoint_path)
        os.makedirs(checkpoint_path, exist_ok=True)

        local_sd = self.train_stage.get_global_state_dict()
        gathered = [None] * self.pp_size
        dist.all_gather_object(gathered, local_sd)

        if self.pp_rank == 0:
            merged_state = {}
            for stage_sd in gathered:
                merged_state.update(stage_sd)

            try:
                from safetensors.torch import save_file

                save_file(merged_state, os.path.join(checkpoint_path, "model.safetensors"))
            except Exception:
                torch.save(merged_state, os.path.join(checkpoint_path, "model.pt"))

            if self.local_path is not None and os.path.isdir(self.local_path):
                for filename in _CHECKPOINT_AUX_FILES:
                    src = os.path.join(self.local_path, filename)
                    dst = os.path.join(checkpoint_path, filename)
                    if os.path.exists(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)

        if self.optimizer is not None:
            optimizer_state = self.optimizer.state_dict()
            torch.save(optimizer_state, os.path.join(checkpoint_path, f"optimizer_rank{self.pp_rank}.pt"))

        dist.barrier()
