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
Megatron Actor.
In megatron actor, the differences are:
1. We only make minibatch

Note that our model doesn't have to be `MegatronModule` because we don't share embedding in the last layer
"""

import itertools
import logging
import os
from functools import partial
from typing import Iterable

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from megatron.core.distributed import finalize_model_grads

# from megatron.core.optimizer import DistributedOptimizer
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from omegaconf import OmegaConf
from torch import nn

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits
from verl.utils.megatron_utils import get_model_config
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.profiler.profile import Profiler
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import broadcast_dict_tensor
from verl.workers.actor import BasePPOActor

__all__ = ["MegatronPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MegatronPPOActor(BasePPOActor):
    def __init__(
        self,
        config,
        model_config,
        hf_config,
        tf_config,
        actor_module: nn.ModuleList,
        actor_optimizer: DistributedOptimizer,
    ):
        """MeagtronPPOActor class. This class implements the simple PPO logics when the model is built with Megatron.

        Args:
            config (OmegaConf): the basic config that contains the hyper-parameters of PPO Actor. It must contain

                ``ppo_micro_batch_size_per_gpu``: micro batch size when updating ppo.

                ``ppo_mini_batch_size``: minibatch size when updating ppo using the batch data.

                ``ppo_epochs``: number of epochs to update the actor using the batch data.

                ``shuffle``: whether to shuffle the data after each ppo epoch.

                ``clip_ratio``: clip ratio of the ppo algorithm. See https://arxiv.org/abs/1707.06347.

                ``entropy_coeff``: entropy coefficient of the PPO loss. See https://arxiv.org/abs/1707.06347.
            model_config (OmegaConf): model configuration. It must contains ``model_config.vocab_size`` and
                ``model_config.hidden_size``
            hf_config (PretrainedConfig): huggingface config
            tf_config (TransformerConfig): mcore transformer config
            actor_module (nn.ModuleList): actor module is a ModuleList that contains a list of nn.Module in this
                pp stage.
                each nn.Module in this rank holds a vpp module chunk. See https://arxiv.org/pdf/2104.04473.pdf for
                more details.
                The actor module has some constraints to follow in order to use the updating logics implemented here

                1. It must implement unpad_input before any computation and pad_input after all the computation.
                Remove padding is an
                optimization that removes the padding tokens. See unpad_input and pad_input function in flash-attn
                (https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py).

                2. Each pp stage must return the hidden state with the same shape [total_nnz, 1, hidden_size],
                where total_nnz is the number of valid tokens in this batch. If sequence parallel is enabled, the size
                of the hidden state is [total_nnz // tp, 1, hidden_size].
            actor_optimizer (DistributedOptimizer): currently, we only support DistributedOptimizer in Megatron.
                It implements
                zero1 optimizer that shards the optimizer state across dp ranks.

        >>> from megatron.training import get_model
        >>> from megatron.optimizer import get_megatron_optimizer
        >>> actor_module = get_model(megatron_actor_model_provider, wrap_with_ddp=True)
        >>> actor_module = nn.ModuleList(actor_module)
        >>> actor_optimizer = get_megatron_optimizer(actor_module)
        >>> actor = MegatronPPOActor(config=config,
        >>>                          model_config=actor_model_config,
        >>>                          hf_config=hf_config,
        >>>                          tf_config=tf_config,
        >>>                          actor_module=actor_module,
        >>>                          actor_optimizer=actor_optimizer)
        """
        super().__init__(config)
        self._validate_config(config)
        self.model_config = model_config
        self.hf_config = hf_config
        self.tf_config = tf_config
        self.actor_module = actor_module
        self.actor_optimizer: DistributedOptimizer = actor_optimizer
        self.use_torch_profiler = self.config.profiler.get("tool") == "torch"
        if self.use_torch_profiler:
            self.prof = Profiler(
                self.config.profiler, tool_config=self.config.profiler.get("tool_config", {}).get("torch", {})
            )
        else:
            self.prof = None
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if self.use_fused_kernels:
            from verl.models.mcore.model_forward_fused import patch_fused_forward

            for model in self.actor_module:
                patch_fused_forward(model)

        # rlpipe: Megatron fused-forward Phase 1 MVP.
        # When `use_fused_forward_pp` is enabled, `compute_log_prob` short-circuits
        # the Megatron PP forward_backward_func and instead uses a local HF model
        # replica to compute response log probs. Trades 4× redundant HF forward
        # compute (each rank runs the same HF model) for avoiding PP bubbles in
        # the inference-only phase. Phase 1 assumes the HF weights match the
        # Megatron weights as of the start of this step (i.e., no weight sync
        # yet — see `recompute_old_log_prob` contract). This is correct for the
        # FIRST compute_log_prob call of each step; subsequent log_prob
        # recomputation within the same step (e.g., dual log_prob for
        # diagnostics) would see updated weights and is not supported.
        # Mode switch: full HF replica (V1) or PP-sharded reverse-PP (V2).
        # V2 eliminates the 3.4GB full replica and the per-iF load/offload
        # overhead. Set RLPIPE_FUSED_REVERSE_PP=1 to enable.
        import os as _os_rev
        self.use_fused_reverse_pp = (
            _os_rev.environ.get("RLPIPE_FUSED_REVERSE_PP", "0") == "1"
        )
        self._reverse_pp_infer_stage = None

        self.use_fused_forward_pp = self.config.get("use_fused_forward_pp", False)
        self._fused_inference_stage = None
        if self.use_fused_forward_pp and not self.use_fused_reverse_pp:
            from verl.workers.actor.megatron_fused_inference import (
                MegatronFusedInferenceStage,
            )
            fused_model_path = self.config.get(
                "use_fused_forward_pp_model_path",
                getattr(self.hf_config, "_name_or_path", None),
            )
            if not fused_model_path:
                raise ValueError(
                    "use_fused_forward_pp=True requires either "
                    "actor.use_fused_forward_pp_model_path to be set or "
                    "hf_config._name_or_path to resolve to a valid HF path"
                )
            logger.info(
                "[rlpipe megatron-fused] loading HF inference stage from %s",
                fused_model_path,
            )
            self._fused_inference_stage = MegatronFusedInferenceStage(
                model_path=fused_model_path,
                device=torch.device(f"cuda:{torch.cuda.current_device()}"),
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            # At 8B+ the HF replica (16 GB per rank) competes with
            # `update_actor` for HBM. Set RLPIPE_FUSED_FORWARD_CPU_OFFLOAD=1
            # to park the replica on CPU between compute_log_prob calls.
            # At 1.7B the replica is 3.4 GB and fits alongside training state
            # comfortably, so leave it on GPU to save the ~3 s/step transfer
            # overhead.
            import os as _os
            if _os.environ.get("RLPIPE_FUSED_FORWARD_CPU_OFFLOAD", "0") == "1":
                self._fused_inference_stage.to("cpu")
                logger.info(
                    "[rlpipe megatron-fused] inference stage ready (parked on CPU, 16 GB→host): %s",
                    self._fused_inference_stage,
                )
            else:
                logger.info(
                    "[rlpipe megatron-fused] inference stage ready (resident on GPU): %s",
                    self._fused_inference_stage,
                )

        # V2: PP-sharded reverse-direction inference model. Each rank holds
        # 1/pp_size of the inference params (~850 MB at 1.7B/PP=4 vs 3.4 GB
        # for the full HF replica). iF flows rank P-1 → ... → 0 via gloo
        # PP communication. Eliminates per-iF load/offload overhead.
        if self.use_fused_forward_pp and self.use_fused_reverse_pp:
            from verl.workers.actor.megatron_reverse_pp_inference import (
                MegatronReversePPInferenceStage,
            )
            fused_model_path = self.config.get(
                "use_fused_forward_pp_model_path",
                getattr(self.hf_config, "_name_or_path", None),
            )
            if not fused_model_path:
                raise ValueError(
                    "use_fused_forward_pp=True requires fused_model_path"
                )
            from megatron.core import parallel_state as _mpu
            _pp_rank = _mpu.get_pipeline_model_parallel_rank()
            _pp_size = _mpu.get_pipeline_model_parallel_world_size()
            logger.info(
                "[rlpipe megatron-fused-v2] loading reverse-PP inference stage "
                "(train_pp_rank=%d, pp_size=%d, infer_rank=%d)",
                _pp_rank, _pp_size, _pp_size - 1 - _pp_rank,
            )
            self._reverse_pp_infer_stage = MegatronReversePPInferenceStage(
                model_path=fused_model_path,
                train_pp_rank=_pp_rank,
                pp_size=_pp_size,
                device=torch.device(f"cuda:{torch.cuda.current_device()}"),
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            logger.info(
                "[rlpipe megatron-fused-v2] inference stage ready: %s",
                self._reverse_pp_infer_stage,
            )

        # Gloo P2P pair groups for fused_forward_backward.
        # Created lazily on first call to _init_fused_pp_pair_groups.
        # Gloo is used instead of NCCL because NCCL's send_forward is
        # fully blocking (even with wait_on_reqs=False the shape exchange
        # blocks), causing circular deadlock in the fused schedule.
        # Each pair group connects adjacent PP ranks. "olp" group is for
        # rank 0 ↔ rank P-1 (old_log_probs transfer in fused schedule).
        self._pp_pair_groups = None

        self.optimizer_step_args = OmegaConf.create(
            {
                "skip_grad": None,
                "overlap_dp_param_comm": False,
                "overlap_dp_grad_comm": False,
                "gradient_accumulation_steps": 1,
                "sequence_parallel": self.tf_config.sequence_parallel,
                "DDP_impl": "local",
                "layernorm_allreduce_bucket_threshold": 0,
                "pipeline_model_parallel_split_rank": None,
                "reduce_grads_use_alltoall": False,
            }
        )

        config = get_model_config(self.actor_module[0])
        print(config)
        config.finalize_model_grads_func = finalize_model_grads

    def _validate_config(self, config) -> None:
        """Validate config options not implemented for Megatron backend"""
        assert config.get("ulysses_sequence_parallel_size", 1) == 1
        if config.get("shuffle", False):
            assert config.data_loader_seed is not None, "If shuffle dataloader, seed must be manually set"
        if config.megatron.tensor_model_parallel_size == 1:
            print("[Warining] Because actor tp size == 1, set sp to False")
            config.megatron.sequence_parallel = False
        self.config = config

    @GPUMemoryLogger(role="megatron actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            DataProto: torch.Tensor: the log_prob tensor
        """
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)
        micro_batch_size = data.meta_info.get("micro_batch_size", None)
        max_token_len = data.meta_info.get("max_token_len", None)
        if use_dynamic_bsz:
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            max_token_len = max_token_len * self.config.megatron.context_parallel_size
        else:
            assert micro_batch_size is not None, (
                "micro batch size is needed for forward compute when use_dynamic_bsz is False"
            )

        # rlpipe: Megatron fused-forward Phase 1 MVP short-circuit.
        # Skip the Megatron PP forward_backward_func entirely and compute
        # response log probs via a local HF replica. All ranks compute the
        # same result from the same input (broadcast across pp already
        # implicit via the upstream DataProto.to() + broadcast) so no
        # further comms are needed.
        if self.use_fused_forward_pp and self._fused_inference_stage is not None:
            return self._compute_log_prob_fused_forward(
                data,
                calculate_entropy=calculate_entropy,
                use_dynamic_bsz=use_dynamic_bsz,
                micro_batch_size=micro_batch_size,
                max_token_len=max_token_len,
            )

        # We make recompute_old_log_prob by default here.
        # TODO (zhangchi.usc1992): actually, this function should only return log_prob and this logic should be
        # handled by user outside
        recompute_old_log_prob = self.config.get("recompute_old_log_prob", True)

        entropys = torch.Tensor()
        if recompute_old_log_prob:
            log_probs, entropys = self._compute_log_prob_via_megatron_pp(
                data,
                calculate_entropy=calculate_entropy,
                use_dynamic_bsz=use_dynamic_bsz,
                micro_batch_size=micro_batch_size,
                max_token_len=max_token_len,
            )

        # add empty cache after each compute
        get_torch_device().empty_cache()

        return log_probs, entropys

    @torch.no_grad()
    def _compute_log_prob_via_megatron_pp(
        self,
        data: DataProto,
        *,
        calculate_entropy: bool,
        use_dynamic_bsz: bool,
        micro_batch_size,
        max_token_len,
    ):
        """Megatron PP=4 forward_backward_func path for ``compute_log_prob``.

        Extracted from the inlined body of :meth:`compute_log_prob` so the
        fused-forward compare mode (``RLPIPE_FUSED_FORWARD_COMPARE=1``) can
        call it side-by-side with the HF replica path for numerical
        validation.
        """

        def compute_logprobs_fn(output, data, use_dynamic_bsz=False, indices=None):
            response = data["responses"]
            response_length = response.size(1)
            log_probs = output["log_probs"][:, -response_length - 1 : -1].contiguous()
            return {"log_probs": log_probs}

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        input_ids = batch["input_ids"]
        batch_size = input_ids.size(0)
        response = batch["responses"]
        response_length = response.size(1)
        entropys = torch.Tensor()

        # Parallel profiling hook for A/B visual comparison with the fused
        # path — set RLPIPE_MEGATRON_PP_PROFILE=<dir> to emit a chrome trace
        # of this rank's Megatron PP=4 forward_backward_func call. Guarded
        # by a class-level flag so only the first call is profiled.
        import os as _os_mg_prof
        _mg_profile_dir = _os_mg_prof.environ.get("RLPIPE_MEGATRON_PP_PROFILE", "")
        _mg_already = getattr(self, "_megatron_pp_profiled", False)
        _mg_profile_ctx = None
        if _mg_profile_dir and not _mg_already:
            import torch.profiler as _tp_mg
            _mg_profile_ctx = _tp_mg.profile(
                activities=[
                    _tp_mg.ProfilerActivity.CPU,
                    _tp_mg.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=False,
            )
            _mg_profile_ctx.__enter__()
            self._megatron_pp_profiled = True

        output = self.forward_backward_batch(
            data,
            forward_only=True,
            post_process_fn=compute_logprobs_fn,
            calculate_entropy=calculate_entropy,
            use_dynamic_bsz=use_dynamic_bsz,
            micro_batch_size=micro_batch_size,
            max_token_len=max_token_len,
        )

        if _mg_profile_ctx is not None:
            torch.cuda.synchronize()
            _mg_profile_ctx.__exit__(None, None, None)
            _os_mg_prof.makedirs(_mg_profile_dir, exist_ok=True)
            try:
                _mg_r = torch.distributed.get_rank()
            except Exception:
                _mg_r = 0
            _mg_out = _os_mg_prof.path.join(
                _mg_profile_dir, f"megatron_pp_rank{_mg_r}.json"
            )
            _mg_profile_ctx.export_chrome_trace(_mg_out)
            print(
                f"[rlpipe megatron-pp] rank={_mg_r} chrome trace written: {_mg_out}",
                flush=True,
            )
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            if calculate_entropy:
                log_probs = [o[0]["log_probs"] for o in output["output"]]
            else:
                log_probs = [o["log_probs"] for o in output["output"]]
            log_probs = torch.cat(log_probs, dim=0).to(torch.float32)
            if use_dynamic_bsz:
                indices = output["indices"]
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                log_probs = log_probs[revert_indices]
        else:
            log_probs = torch.empty(
                size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device
            )
        log_probs = log_probs.to(get_device_id())
        torch.distributed.broadcast(
            tensor=log_probs,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
            async_op=False,
        )
        log_probs = log_probs.to("cpu")
        if calculate_entropy:
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                entropys = torch.cat([o[1] for o in output["output"]], dim=0)
                entropys = entropys.to(torch.float32)
                if use_dynamic_bsz:
                    indices = output["indices"]
                    indices = list(itertools.chain.from_iterable(indices))
                    assert len(indices) == entropys.size(0), f"{len(indices)} vs. {entropys.size()}"
                    revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                    entropys = entropys[revert_indices]
            else:
                entropys = torch.empty(
                    size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device
                )
            entropys = entropys.to(get_device_id())
            torch.distributed.broadcast(
                tensor=entropys,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
                async_op=False,
            )
            entropys = entropys.to("cpu")
        return log_probs, entropys

    @torch.no_grad()
    def _compute_log_prob_fused_forward(
        self,
        data: DataProto,
        calculate_entropy: bool = False,
        *,
        use_dynamic_bsz: bool = False,
        micro_batch_size=None,
        max_token_len=None,
    ):
        """rlpipe: Megatron fused-forward Phase 1 short-circuit for
        `compute_log_prob`. Runs HF model forward on this rank (all ranks
        produce the same result from the same input, no comms needed).

        Returns ``(log_probs, entropys)`` matching the non-fused path's
        contract. `entropys` is returned as a zero tensor for now —
        proper entropy computation is deferred to Phase 2 (needs HF
        logits, not just log probs, and incurs extra memory).

        When ``RLPIPE_FUSED_FORWARD_COMPARE=1`` is set, also runs the
        Megatron PP=4 path on the same batch and prints max/mean absolute
        log-prob deltas across real response positions. Used to validate
        numerical parity vs the PP baseline without trusting pg_loss
        alone (which is near-zero in step 1 for any self-consistent
        path, proving nothing).
        """
        stage = self._fused_inference_stage
        device_id = get_device_id()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        input_ids_full = batch["input_ids"].to(device_id)
        attention_mask_full = batch["attention_mask"].to(device_id).long()
        position_ids_full = batch["position_ids"].to(device_id).long()
        responses = batch["responses"]
        response_length = responses.size(1)
        batch_size = input_ids_full.size(0)

        temperature = float(data.meta_info.get("temperature", 1.0))
        # Use a small micro-batch to keep HF forward activation memory
        # bounded; full Qwen3-1.7B forward on 17k tokens × 4 seqs fits
        # comfortably within the HBM headroom left after Megatron's
        # param_offload.
        fused_mbs = int(
            self.config.get("use_fused_forward_pp_micro_batch_size", 4)
        )

        # Phase 2-Lite: shard the batch across PP ranks and all-gather.
        # Phase 1 MVP ran the full batch on every rank (4× redundant compute).
        # Sharding divides the per-rank work by pp_size (~30 s → ~7.5 s at PP=4)
        # and costs only a tiny all-gather (10 MB × pp_size over NVLink).
        # Gated by env var so we can still compare to the Phase 1 baseline.
        import os as _os_shard
        _sharded = _os_shard.environ.get("RLPIPE_FUSED_FORWARD_SHARDED", "0") == "1"
        try:
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            pp_size = mpu.get_pipeline_model_parallel_world_size()
            pp_group = mpu.get_pipeline_model_parallel_group()
        except Exception:
            pp_rank, pp_size, pp_group = 0, 1, None
        if _sharded and pp_size > 1 and batch_size % pp_size == 0:
            shard_size = batch_size // pp_size
            start_idx = pp_rank * shard_size
            end_idx = start_idx + shard_size
        else:
            # Fall back to Phase 1 replicated behavior if:
            # (a) sharding disabled, (b) single PP rank, or (c) batch not
            # evenly divisible (would require padding the all-gather).
            _sharded = False
            start_idx, end_idx = 0, batch_size
            shard_size = batch_size

        import time as _time
        # Optional torch profiler hook — set RLPIPE_FUSED_FORWARD_PROFILE=<dir>
        # to capture a Chrome trace of this rank's fused forward loop and
        # write it to <dir>/fused_forward_rank{R}.json. Output is
        # Perfetto-loadable at ui.perfetto.dev. Only profiles the first call
        # per rank (guarded by a class-level flag).
        _profile_dir = _os_shard.environ.get("RLPIPE_FUSED_FORWARD_PROFILE", "")
        _already_profiled = getattr(self, "_fused_forward_profiled", False)
        _profile_ctx = None
        if _profile_dir and not _already_profiled:
            import torch.profiler as _tp
            _profile_ctx = _tp.profile(
                activities=[
                    _tp.ProfilerActivity.CPU,
                    _tp.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=False,
            )
            _profile_ctx.__enter__()
            self._fused_forward_profiled = True

        _t0 = _time.perf_counter()
        log_probs_chunks = []
        num_chunks = 0
        # Optional PPTracer handle from the worker — fires per chunk on the
        # rank's GPU-ops row in the merged Perfetto view, with `iF mb=i`
        # labels so the fused forward lanes are directly comparable to
        # Megatron's training `tF mb=i` / `tB mb=i` lanes.
        _pp_tracer = getattr(self, "_pp_tracer", None)
        for i in range(start_idx, end_idx, fused_mbs):
            j = min(i + fused_mbs, end_idx)
            # Use the local chunk index (0..num_chunks-1) as the mb id so
            # the trace shows the per-rank shard's own 0-based ordering.
            _mb_id = num_chunks
            _if_ctx = _pp_tracer.trace("infer_forward", micro_batch_id=_mb_id) if _pp_tracer else None
            if _if_ctx is not None:
                _if_ctx.__enter__()
            try:
                chunk_log_probs = stage.forward_log_probs(
                    input_ids=input_ids_full[i:j],
                    attention_mask=attention_mask_full[i:j],
                    position_ids=position_ids_full[i:j],
                    response_length=response_length,
                    temperature=temperature,
                )
            finally:
                if _if_ctx is not None:
                    _if_ctx.__exit__(None, None, None)
            # Keep on GPU in fp32 for the all-gather path; we'll move to CPU
            # after aggregation. Non-sharded path also benefits — skipping the
            # per-chunk D2H transfer inside the loop shaves a few hundred ms.
            log_probs_chunks.append(chunk_log_probs.to(torch.float32))
            num_chunks += 1
        torch.cuda.synchronize()
        _t1 = _time.perf_counter()

        if _profile_ctx is not None:
            _profile_ctx.__exit__(None, None, None)
            import os as _os_prof
            _os_prof.makedirs(_profile_dir, exist_ok=True)
            try:
                _r = torch.distributed.get_rank()
            except Exception:
                _r = 0
            _out = _os_prof.path.join(_profile_dir, f"fused_forward_rank{_r}.json")
            _profile_ctx.export_chrome_trace(_out)
            print(
                f"[rlpipe megatron-fused] rank={_r} chrome trace written: {_out}",
                flush=True,
            )
        local_log_probs = torch.cat(log_probs_chunks, dim=0)  # [shard_size, rl]

        if _sharded:
            # All-gather across PP group. Every rank ends up with the full
            # [batch_size, response_length] tensor. No padding needed here
            # because we guard on `batch_size % pp_size == 0` above.
            gather_list = [
                torch.empty_like(local_log_probs) for _ in range(pp_size)
            ]
            torch.distributed.all_gather(
                gather_list, local_log_probs, group=pp_group
            )
            log_probs = torch.cat(gather_list, dim=0).cpu()
        else:
            log_probs = local_log_probs.cpu()
        try:
            _rank = torch.distributed.get_rank()
        except Exception:
            _rank = -1
        print(
            f"[rlpipe megatron-fused] rank={_rank} compute_log_prob: "
            f"batch_size={batch_size} shard={shard_size} "
            f"seq_len={input_ids_full.shape[1]} num_chunks={num_chunks} "
            f"fused_mbs={fused_mbs} sharded={_sharded} "
            f"total_time={_t1 - _t0:.3f}s per_chunk={(_t1 - _t0) / max(num_chunks, 1) * 1000:.1f}ms",
            flush=True,
        )

        # Compare mode: run Megatron PP path side-by-side and log the
        # max/mean |HF - Megatron| across real response tokens. Active only
        # when env var is set so normal runs don't pay the 2× compute cost.
        import os as _os
        if _os.environ.get("RLPIPE_FUSED_FORWARD_COMPARE", "0") == "1":
            _tc0 = _time.perf_counter()
            mg_log_probs, _mg_entropys = self._compute_log_prob_via_megatron_pp(
                data,
                calculate_entropy=False,
                use_dynamic_bsz=use_dynamic_bsz,
                micro_batch_size=micro_batch_size,
                max_token_len=max_token_len,
            )
            torch.cuda.synchronize()
            _tc1 = _time.perf_counter()
            # Only compare at real response positions. `attention_mask[:, -rl:]`
            # gives 1 for real response tokens, 0 for right-pad.
            response_mask_cpu = (
                attention_mask_full[:, -response_length:].to(torch.bool).cpu()
            )
            diff = (log_probs - mg_log_probs).abs()
            diff_masked = diff[response_mask_cpu]
            if diff_masked.numel() > 0:
                _max = diff_masked.max().item()
                _mean = diff_masked.mean().item()
                _p99 = diff_masked.kthvalue(
                    max(1, int(diff_masked.numel() * 0.99))
                ).values.item()
            else:
                _max = _mean = _p99 = 0.0
            # HF reference magnitude for relative scale (abs value of MG log_probs
            # on same positions). log_probs are negative (or zero), so |mg| tells
            # us the typical magnitude we're comparing against.
            mg_masked_mag = mg_log_probs[response_mask_cpu].abs()
            _mg_mean_abs = (
                mg_masked_mag.mean().item() if mg_masked_mag.numel() > 0 else 0.0
            )
            print(
                f"[rlpipe megatron-fused] rank={_rank} COMPARE vs Megatron PP: "
                f"max|HF - MG|={_max:.6f} "
                f"mean|HF - MG|={_mean:.6f} "
                f"p99|HF - MG|={_p99:.6f} "
                f"mean|MG|={_mg_mean_abs:.3f} "
                f"n_positions={int(diff_masked.numel())} "
                f"mg_time={_tc1 - _tc0:.2f}s",
                flush=True,
            )

            # Histogram of absolute diffs — gives a quick sense of the
            # distribution shape (is 6% mean drift uniform or driven by
            # a long tail of bad positions?).
            _bin_edges = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, float("inf")]
            _bin_labels = [
                "[0, 5e-3)", "[5e-3, 1e-2)", "[1e-2, 2e-2)", "[2e-2, 5e-2)",
                "[5e-2, 0.1)", "[0.1, 0.2)", "[0.2, 0.5)", "[0.5, 1)", "[1, 2)", "[2, inf)",
            ]
            _hist = []
            _total = diff_masked.numel()
            for _i in range(len(_bin_edges) - 1):
                _lo, _hi = _bin_edges[_i], _bin_edges[_i + 1]
                _count = int(
                    ((diff_masked >= _lo) & (diff_masked < _hi)).sum().item()
                )
                _hist.append(f"{_bin_labels[_i]}={_count}({100 * _count / max(_total, 1):.2f}%)")
            print(
                f"[rlpipe megatron-fused] rank={_rank} COMPARE hist: " + " ".join(_hist),
                flush=True,
            )

            # Top-5 worst positions: print the diff + the corresponding
            # predicted token id so we can check if outliers cluster on
            # special tokens (BOS/EOS/pad).
            if diff_masked.numel() > 0 and _rank == 0:
                # Need a [B, rl] view of diffs to recover (seq, pos, token_id).
                diff_full = (log_probs - mg_log_probs).abs()  # [B, rl]
                # Token IDs at each response position: responses tensor.
                _responses_cpu = data.batch["responses"].cpu()  # [B, rl]
                # Mask out pad positions (right pad in the response) so we only
                # rank real-token positions.
                _resp_mask_cpu = (
                    attention_mask_full[:, -response_length:].cpu().to(torch.bool)
                )
                diff_masked_full = diff_full.clone()
                diff_masked_full[~_resp_mask_cpu] = -1.0
                _flat = diff_masked_full.view(-1)
                _topk_vals, _topk_idx = torch.topk(_flat, k=min(5, _flat.numel()))
                _rl = response_length
                _report = []
                for _v, _ii in zip(_topk_vals.tolist(), _topk_idx.tolist()):
                    _seq = _ii // _rl
                    _pos = _ii % _rl
                    _tok = int(_responses_cpu[_seq, _pos].item())
                    _hf = float(log_probs[_seq, _pos].item())
                    _mg = float(mg_log_probs[_seq, _pos].item())
                    _report.append(
                        f"(seq={_seq},pos={_pos},tok={_tok},HF={_hf:.3f},MG={_mg:.3f},|d|={_v:.3f})"
                    )
                print(
                    f"[rlpipe megatron-fused] rank={_rank} COMPARE top5 outliers: "
                    + " ".join(_report),
                    flush=True,
                )

        # Phase 1: entropy not fused. Return a zero tensor of the right
        # shape so downstream code's shape assumptions hold. If a caller
        # asked for entropy, they'll get zeros — caller must decide
        # whether that's acceptable or to fall back to the non-fused path.
        entropys = (
            torch.zeros_like(log_probs)
            if calculate_entropy
            else torch.Tensor()
        )

        get_torch_device().empty_cache()
        return log_probs, entropys

    def _init_fused_pp_pair_groups(self):
        """Create gloo pair groups for fused_forward_backward P2P.

        Gloo is used instead of NCCL because NCCL's `send_forward` in
        Megatron's P2PCommunicator is fully blocking (even with
        `wait_on_reqs=False` the shape exchange blocks via
        `batch_isend_irecv + wait()`). With the fused schedule, ranks are
        in different states simultaneously (rank 3 doing iF while rank 2
        tries to send) — blocking P2P causes circular deadlock.

        Gloo `isend` is truly non-blocking (CPU-staged), so the sender
        returns immediately and the receiver matches via FIFO order when
        it posts `recv`. No deadlock.

        **Critical**: `torch.distributed.new_group` is a collective over
        the ENTIRE world, not just the PP group. All processes must call
        it with the same argument order, even if they're not in the
        group. We create PP adjacent-pair groups + an "olp" group for
        rank 0 ↔ rank P-1 (transfers old_log_probs in the fused schedule:
        the last inference rank (rank 0) computes log_probs and sends
        them to the last training rank (rank P-1) for the loss).
        """
        if self._pp_pair_groups is not None:
            return
        import torch.distributed as dist

        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        pp_group = mpu.get_pipeline_model_parallel_group()

        # Resolve PP-local ranks to global ranks.
        pp_global_ranks = [
            dist.get_global_rank(pp_group, r) for r in range(pp_size)
        ]

        self._pp_pair_groups = {}
        # Training pair groups (forward direction: rank i → rank i+1)
        for i in range(pp_size - 1):
            ranks = [pp_global_ranks[i], pp_global_ranks[i + 1]]
            grp = dist.new_group(ranks=ranks, backend="gloo")
            self._pp_pair_groups[i] = grp
        # Inference pair groups (reverse direction: rank i+1 → rank i).
        # SEPARATE from training groups: gloo preserves FIFO within a group,
        # so if training backward (i+1→i) and inference forward (i+1→i)
        # share one group, they'd collide. Separate groups prevent this.
        for i in range(pp_size - 1):
            ranks = [pp_global_ranks[i], pp_global_ranks[i + 1]]
            grp = dist.new_group(ranks=ranks, backend="gloo")
            self._pp_pair_groups[f"infer_{i}"] = grp
        # old_log_probs group: rank 0 ↔ rank P-1
        olp_ranks = [pp_global_ranks[0], pp_global_ranks[pp_size - 1]]
        if pp_size > 1:
            self._pp_pair_groups["olp"] = dist.new_group(
                ranks=olp_ranks, backend="gloo"
            )

        if pp_rank == 0:
            logger.info(
                "[rlpipe megatron-fused] created %d gloo pair groups for PP=%d",
                len(self._pp_pair_groups), pp_size,
            )

    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        """Make minibatch iterator for updating the actor

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64, where
                ``sequence_length = prompt_length + response_length``

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``responses``: tensor of shape [batch_size, response_length]. torch.int64. Note that
                responses = input_ids[:, -response_length:]

                ``old_log_probs``: tensor of shape [batch_size, response_length]. torch.float32. The log probability
                of responses.

                ``advantages``: tensor of shape [batch_size, response_length]. torch.float32. The advantages of
                responses.
                See PPO paper for details. https://arxiv.org/abs/1707.06347

        Returns:

        """
        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "response_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        self.has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        if self.has_multi_modal_inputs:
            data = data.select(select_keys, ["multi_modal_inputs"])
        else:
            data = data.select(batch_keys=select_keys)
        return data.make_iterator(
            mini_batch_size=self.config.ppo_mini_batch_size,
            epochs=self.config.ppo_epochs,
            seed=self.config.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.shuffle},
        )

    def forward_backward_batch(
        self,
        data: DataProto,
        forward_only=False,
        post_process_fn=None,
        calculate_entropy=False,
        use_dynamic_bsz=False,
        micro_batch_size=None,
        max_token_len=None,
        mini_batch_size=None,
    ):
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
        """
        # broadcast from last pp rank to all other pp ranks
        # TODO: actually, we just need to control the sampling order.
        data.to(get_device_id())
        data.batch = data.batch.contiguous()
        mini_batch = data
        broadcast_dict_tensor(
            mini_batch.batch,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        mini_batch.to("cpu")
        # split into micro-batches
        mini_batch.batch["attention_mask"] = mini_batch.batch["attention_mask"].to(bool)
        self.has_multi_modal_inputs = "multi_modal_inputs" in mini_batch.non_tensor_batch.keys()
        if self.has_multi_modal_inputs:
            mini_batch.batch["multi_modal_inputs"] = mini_batch.non_tensor_batch["multi_modal_inputs"]
            mini_batch.batch["multi_modal_inputs_idx"] = torch.Tensor(
                list(range(len(mini_batch.non_tensor_batch["multi_modal_inputs"])))
            ).to(torch.int64)

        if mini_batch.batch["position_ids"].dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            mini_batch.batch["position_ids"] = mini_batch.batch["position_ids"][
                :, 0
            ]  # mcore patch recompute qwen2vl's pos ids during forward

        indices = None
        temperature = data.meta_info["temperature"]
        if use_dynamic_bsz:
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            if vpp_size is not None and vpp_size > 1:
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                micro_batches, indices = rearrange_micro_batches(
                    batch=mini_batch.batch,
                    num_batches_divided_by=microbatch_group_size_per_vp_stage,
                    max_token_len=max_token_len,
                )
                assert len(micro_batches) % self.tf_config.microbatch_group_size_per_vp_stage == 0, (
                    f"micro_batches {micro_batches} must be divisible by microbatch_group_size_per_vp_stage "
                    f"{microbatch_group_size_per_vp_stage} for megatron backend"
                )
            else:
                micro_batches, indices = rearrange_micro_batches(batch=mini_batch.batch, max_token_len=max_token_len)
            total_seqlen = max_token_len
        else:
            assert micro_batch_size is not None, (
                "micro_batch_size is needed to be passed in when not using dynamic batch size"
            )
            micro_batches = mini_batch.batch.split(micro_batch_size)
            seq_len = micro_batches[0]["input_ids"].shape[1]
            total_seqlen = micro_batch_size * seq_len
        # compute input shapes for pp stages
        n_micro_batch = len(micro_batches)

        forward_backward_func = get_forward_backward_func()

        def loss_func(output, data, meta_info):
            # For memory efficiency
            # We move calculation of entropy to compute_log_probs, forward_only == True
            device = output["log_probs"].device
            metrics = {}
            if forward_only:
                if post_process_fn is None:
                    pass
                    # metrics["logits"] = output
                else:
                    stats = post_process_fn(output, data)
                    metrics.update(stats)
                if not calculate_entropy:
                    return torch.tensor(1.0, device=device), metrics

            responses = data["responses"]
            response_length = responses.size(1)
            response_mask = data["response_mask"].to(bool)
            loss_agg_mode = self.config.loss_agg_mode
            # compute policy loss
            log_prob = output["log_probs"][:, -response_length - 1 : -1].contiguous()
            ret_entropy = None
            stats = {}
            if not forward_only:
                old_log_prob = data["old_log_probs"]
                advantages = data["advantages"]

                entropy_coeff = self.config.entropy_coeff
                loss_agg_mode = self.config.loss_agg_mode

                loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                policy_loss_fn = get_policy_loss_fn(loss_mode)

                # Extract pre-computed rollout importance sampling weights if present
                # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                rollout_is_weights = data.get("rollout_is_weights", None)

                # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
                # are computed centrally in ray_trainer.py for consistency and efficiency.
                # This ensures metrics are computed uniformly across all batches at the trainer level
                # and avoids redundant computation across workers and micro-batches.
                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    response_mask=response_mask,
                    loss_agg_mode=loss_agg_mode,
                    config=self.config,
                    rollout_is_weights=rollout_is_weights,
                )

                stats.update(
                    {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                )
                policy_loss = pg_loss

            if calculate_entropy:
                entropy = output["entropy"][:, -response_length - 1 : -1].contiguous()
                if not forward_only:
                    entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                    entropy_coeff = meta_info["entropy_coeff"]
                    policy_loss = pg_loss - entropy_coeff * entropy_loss
                else:
                    ret_entropy = entropy

            if forward_only:
                policy_loss = torch.tensor(1.0, device=device)
            else:
                if self.config.use_kl_loss:
                    ref_log_prob = data["ref_log_prob"]
                    # compute kl loss
                    kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics["actor/kl_loss"] = kl_loss.detach().item()
                    metrics["actor/kl_coef"] = self.config.kl_loss_coef

                # return loss and stats

            append_to_dict(metrics, stats)
            return policy_loss, [metrics, ret_entropy]

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            batch = batch.to(get_device_id())
            batch = batch.contiguous()

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]

            multi_modal_inputs = {}
            if "multi_modal_inputs" in batch:
                from verl.utils.model import extract_multi_modal_inputs

                indices = batch.get("multi_modal_inputs_idx", None)
                multi_modal_inputs = extract_multi_modal_inputs(batch["multi_modal_inputs"], indices)
            responses = batch["responses"]
            response_length = responses.size(1)
            label = position_ids.clone()
            label[:, -response_length - 1 : -1] = responses
            label_mask = attention_mask.clone()
            label_mask[:, : -response_length - 1] = False
            label_mask[:, -1] = False

            from verl.models.mcore import get_mcore_forward_fn, get_mcore_forward_fused_fn

            if self.use_fused_kernels:
                forward_fn = get_mcore_forward_fused_fn(self.hf_config)
                # return dict of [logits, entropy]
                output = forward_fn(
                    model,
                    input_ids,
                    position_ids,
                    attention_mask,
                    sequence_parallel=self.tf_config.sequence_parallel,
                    multi_modal_inputs=multi_modal_inputs,
                    labels=label,
                    labels_mask=label_mask,
                    temperature=temperature,
                )
            else:
                forward_fn = get_mcore_forward_fn(self.hf_config)

                def logits_processor(logits, label, label_mask):
                    assert logits.shape[:2] == label.shape[:2]
                    assert label.shape == label_mask.shape
                    logits.div_(temperature)
                    ret = {}
                    if calculate_entropy:
                        logits_bak = logits.clone()
                        logger.warning_once(
                            "For memory-efficient computation, enable fused kernels via "
                            "`actor_rollout_ref.model.use_fused_kernels=True`. "
                            "The current `clone()` operation ensures correctness but increases memory usage."
                        )
                        entropy = vocab_parallel_entropy(logits)
                        ret["entropy"] = entropy
                    else:
                        logits_bak = logits
                    log_probs = vocab_parallel_log_probs_from_logits(logits_bak, label)
                    log_probs = log_probs.masked_fill(~label_mask, 0.0)
                    ret["log_probs"] = log_probs
                    return ret

                logits_processor_args = {"label": label, "label_mask": label_mask}
                output = forward_fn(
                    model,
                    input_ids,
                    attention_mask,
                    position_ids,
                    sequence_parallel=self.tf_config.sequence_parallel,
                    multi_modal_inputs=multi_modal_inputs,
                    logits_processor=logits_processor,
                    logits_processor_args=logits_processor_args,
                )

            if forward_only:
                meta_info = None
            else:
                clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                meta_info = {
                    "clip_ratio": self.config.clip_ratio,
                    "entropy_coeff": self.config.entropy_coeff,
                    "clip_ratio_c": clip_ratio_c,
                }
            return output, partial(loss_func, data=batch, meta_info=meta_info)

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # no use when input_shapes was set
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=forward_only,
            )
        else:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # in use for pp = 1
                micro_batch_size=1,  # in use for pp = 1
                forward_only=forward_only,
            )
        # loss_reduces contains the stats returned from loss_func

        if self.has_multi_modal_inputs:
            data.batch.pop("multi_modal_inputs")
            data.batch.pop("multi_modal_inputs_idx")
            data.non_tensor_batch.pop("multi_modal_inputs")

        losses_reduced = {"output": losses_reduced}
        if use_dynamic_bsz:
            losses_reduced["indices"] = indices
        return losses_reduced

    @GPUMemoryLogger(role="megatron actor", logger=logger)
    def update_policy(self, dataloader: Iterable[DataProto]) -> dict:
        """Update the policy with an iterator of DataProto

        Args:
            dataloader (Iterable[DataProto]): an iterator over the DataProto that returns by ``make_minibatch_iterator``
                The keys of each data batch is described in the make_minibatch_iterator.

        Returns:
            Dict: a dictionary containing the statistics. Note that the statistics are only valid in the last pp stage
            and users have to combine the output in each dp rank manually.

        """
        metrics = {}
        if self.use_torch_profiler and self.prof and self.prof.enable:
            self.prof.start()
        for data in dataloader:
            self.actor_optimizer.zero_grad()
            # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
            for chunk in self.actor_module:
                # if use distributed optimizer, zero grad buffer will be handled by optimizer
                chunk.zero_grad_buffer()

            calculate_entropy = self.config.entropy_coeff != 0
            if data.meta_info.get("micro_batch_size", None) is not None:
                micro_batch_size = data.meta_info["micro_batch_size"]
            else:
                micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
            max_token_len = None
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.config.megatron.context_parallel_size
            metric_micro_batch = self.forward_backward_batch(
                data,
                calculate_entropy=calculate_entropy,
                use_dynamic_bsz=self.config.use_dynamic_bsz,
                micro_batch_size=micro_batch_size,
                max_token_len=max_token_len,
                mini_batch_size=self.config.ppo_mini_batch_size,
            )
            metric_micro_batch = metric_micro_batch["output"]
            for metric in metric_micro_batch:
                # Note that o[0] is metrics, o[1] is entropy, o[2] is response_mask
                append_to_dict(metrics, metric[0])  # append the metric from this micro-batch to global metrics.

            update_successful, grad_norm, num_zeros_in_grad = self.actor_optimizer.step()
            data = {"actor/grad_norm": grad_norm}
            append_to_dict(metrics, data)

            if update_successful:
                # allgather already execute in optimizer.step in new megatron
                pass
            else:
                raise NotImplementedError
            if self.use_torch_profiler and self.prof and self.prof.enable:
                self.prof.step()
        # add empty cache after each compute
        if self.use_torch_profiler and self.prof and self.prof.enable:
            self.prof.stop_and_save()
            self.prof.stop_trace()
        get_torch_device().empty_cache()
        return metrics

    # ==================================================================
    # Real fused forward: interleave iF in PP training bubbles
    # ==================================================================

    def fused_forward_backward(
        self,
        data: DataProto,
        micro_batch_size: int,
    ):
        """Fused forward-backward with iF interleaved in PP training bubbles.

        Uses ``build_default_fused_schedule`` to fill warmup bubbles with
        local HF inference (iF). Training forward/backward (tF/tB) use
        Megatron's ``forward_step``/``backward_step`` primitives with
        **gloo P2P** (not NCCL) for non-blocking sends.

        **Why gloo instead of NCCL**: Megatron's P2PCommunicator uses
        blocking NCCL sends (`isend + wait()`). The fused schedule
        requires ranks in different states simultaneously — blocking
        sends cause circular deadlock (confirmed experimentally). Gloo
        `isend` is truly non-blocking (CPU-staged); the sender returns
        immediately, receiver matches via FIFO when it posts `recv`.

        **Current implementation (MVP v1)**:
        - iF: local HF replica forward (no P2P, each rank has full replica)
        - tF/tB: Megatron forward_step/backward_step + gloo pair groups
        - Memory constraint: all iF before first tB per rank

        **V2 (future)**: replace HF replica with Megatron reverse-PP
        inference model (each rank holds 1/pp_size of inference params
        instead of full replica, saving 2-3 GB/rank at 1.7B).

        Returns:
            (dict, Tensor): training metrics + old_log_probs.
        """
        import time as _time
        from functools import partial as _partial
        import torch.distributed as dist

        from megatron.core.pipeline_parallel.schedules import (
            backward_step as mg_backward_step,
            forward_step as mg_forward_step,
        )

        from verl.utils.megatron_utils import get_model_config
        from verl.workers.torch_pp.fused_schedule import (
            build_default_fused_schedule,
            parse_schedule,
        )

        # Ensure gloo pair groups exist. This is a collective over all
        # global ranks, so all Megatron workers must reach this point
        # together (which they do via Ray's dispatch).
        self._init_fused_pp_pair_groups()

        # ── Setup ────────────────────────────────────────────────────
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        pp_group = mpu.get_pipeline_model_parallel_group()
        is_first_stage = mpu.is_pipeline_first_stage(ignore_virtual=True)
        is_last_stage = mpu.is_pipeline_last_stage(ignore_virtual=True)
        pp_global_ranks = [
            dist.get_global_rank(pp_group, r) for r in range(pp_size)
        ]
        self_global_rank = dist.get_global_rank(pp_group, pp_rank)

        config = get_model_config(self.actor_module[0])
        config.finalize_model_grads_func = finalize_model_grads
        model = self.actor_module
        hf_stage = self._fused_inference_stage
        reverse_stage = self._reverse_pp_infer_stage
        assert hf_stage is not None or reverse_stage is not None, \
            "fused_forward_backward requires either HF replica (_fused_inference_stage) " \
            "or reverse-PP inference (_reverse_pp_infer_stage)"
        _pp_tracer = getattr(self, "_pp_tracer", None)
        device = torch.device(f"cuda:{get_device_id()}")

        # ── Gloo P2P helpers ─────────────────────────────────────────
        pair_groups = self._pp_pair_groups
        _send_bufs: list = []  # keep tensor refs alive until cuda sync

        def _pair_key(a: int, b: int, is_infer: bool = False):
            """Adjacent pair: use min(a,b). Non-adjacent (olp): use 'olp'.
            For reverse-PP inference, use infer_{min} groups."""
            if abs(a - b) == 1:
                k = min(a, b)
                return f"infer_{k}" if is_infer else k
            return "olp"

        def _p2p_send(tensor: torch.Tensor, dst_pp_rank: int, is_infer: bool = False):
            """Non-blocking gloo isend with dynamic shape exchange.

            1. Send tensor shape (always 4 int64, padded with -1)
            2. Send tensor data (non-blocking isend, CPU-staged)

            Shape is padded to a fixed 4-element vector so recv always
            knows the exact message size (gloo requires matched sizes).
            Sentinel -1 marks unused trailing dimensions.
            """
            t = tensor.detach().contiguous().cpu()
            _send_bufs.append(t)
            dst_global = pp_global_ranks[dst_pp_rank]
            grp = pair_groups[_pair_key(pp_rank, dst_pp_rank, is_infer)]
            # Pad shape to exactly 4 elements with -1 sentinel
            shape_list = list(t.shape)
            while len(shape_list) < 4:
                shape_list.append(-1)
            shape_t = torch.tensor(shape_list[:4], dtype=torch.int64)
            _send_bufs.append(shape_t)
            h1 = dist.isend(shape_t, dst=dst_global, group=grp)
            h2 = dist.isend(t, dst=dst_global, group=grp)
            return [h1, h2]

        def _p2p_recv(src_pp_rank: int, dtype=torch.bfloat16, is_infer: bool = False):
            """Blocking gloo recv with dynamic shape exchange.

            1. Recv tensor shape (always 4 int64, -1 = unused dim)
            2. Recv tensor data
            Returns tensor on GPU.
            """
            src_global = pp_global_ranks[src_pp_rank]
            grp = pair_groups[_pair_key(pp_rank, src_pp_rank, is_infer)]
            # Recv shape (fixed 4-element vector, -1 = sentinel)
            shape_t = torch.zeros(4, dtype=torch.int64)
            dist.recv(shape_t, src=src_global, group=grp)
            shape = tuple(d for d in shape_t.tolist() if d >= 0)
            if not shape:
                shape = (0,)
            # Recv data
            buf = torch.empty(shape, dtype=dtype, device="cpu")
            dist.recv(buf, src=src_global, group=grp)
            return buf.to(device)

        # ── Data prep ────────────────────────────────────────────────
        temperature = float(data.meta_info.get("temperature", 1.0))
        calculate_entropy = self.config.entropy_coeff != 0
        response_length = data.batch["responses"].size(1)

        # Placeholder old_log_probs so make_minibatch_iterator's select
        # doesn't fail. Replaced per-mb by iF-computed values.
        if "old_log_probs" not in data.batch:
            data.batch["old_log_probs"] = torch.zeros(
                data.batch["responses"].shape, dtype=torch.float32
            )

        dataloader = self.make_minibatch_iterator(data=data)

        all_metrics: dict = {}
        all_old_log_probs: list = []

        # ── Iterate over mini-batches ────────────────────────────────
        for mb_idx, mini_batch in enumerate(dataloader):
            mb_metrics, mb_old_lp = self._fused_forward_backward_mini_batch(
                mini_batch=mini_batch,
                micro_batch_size=micro_batch_size,
                pp_rank=pp_rank,
                pp_size=pp_size,
                pp_group=pp_group,
                pp_global_ranks=pp_global_ranks,
                is_first_stage=is_first_stage,
                is_last_stage=is_last_stage,
                config=config,
                model=model,
                hf_stage=hf_stage,
                reverse_stage=reverse_stage,
                temperature=temperature,
                response_length=response_length,
                calculate_entropy=calculate_entropy,
                device=device,
                _pp_tracer=_pp_tracer,
                _mb_idx=mb_idx,
                _p2p_send=_p2p_send,
                _p2p_recv=_p2p_recv,
                _send_bufs=_send_bufs,
                mg_forward_step=mg_forward_step,
                mg_backward_step=mg_backward_step,
            )
            from verl.utils.py_functional import append_to_dict
            append_to_dict(all_metrics, mb_metrics)
            if mb_old_lp is not None:
                all_old_log_probs.append(mb_old_lp)

        old_log_probs = torch.cat(all_old_log_probs, dim=0) if all_old_log_probs else None
        get_torch_device().empty_cache()
        return all_metrics, old_log_probs

    def _fused_forward_backward_mini_batch(
        self, *, mini_batch, micro_batch_size, pp_rank, pp_size, pp_group,
        pp_global_ranks, is_first_stage, is_last_stage, config, model, hf_stage,
        reverse_stage, temperature, response_length, calculate_entropy, device,
        _pp_tracer, _mb_idx, _p2p_send, _p2p_recv, _send_bufs, mg_forward_step,
        mg_backward_step,
    ):
        """One mini-batch of the fused schedule: iF + tF/tB + optimizer step."""
        import time as _time
        from functools import partial as _partial

        from verl.workers.torch_pp.fused_schedule import (
            build_default_fused_schedule,
            parse_schedule,
        )

        # Broadcast mini-batch from last PP rank (trainer) to all PP ranks
        mini_batch.to(get_device_id())
        mini_batch.batch = mini_batch.batch.contiguous()
        broadcast_dict_tensor(
            mini_batch.batch,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=pp_group,
        )
        mini_batch.to("cpu")
        mini_batch.batch["attention_mask"] = mini_batch.batch["attention_mask"].to(bool)

        micro_batches = mini_batch.batch.split(micro_batch_size)
        M = len(micro_batches)
        seq_len = micro_batches[0]["input_ids"].shape[1]
        hidden_size = getattr(self.tf_config, "hidden_size", None)
        assert hidden_size is not None

        # Activation shape for P2P (non-remove-padding Megatron uses
        # [seq, batch, hidden] layout after the embedding layer).
        act_shape = (seq_len, micro_batch_size, hidden_size)

        # Build fused schedule
        fused_schedules = build_default_fused_schedule(pp_size, M)
        schedule_ops = parse_schedule(fused_schedules[pp_rank])
        if pp_rank == 0 and _mb_idx == 0:
            sched_str = " ".join(repr(op) for op in schedule_ops[:10])
            print(f"[fused-sched] rank={pp_rank} M={M} ops={len(schedule_ops)} "
                  f"first10: {sched_str}", flush=True)

        # Per-mini-batch state
        old_log_probs_dict: dict = {}
        tF_counter = [0]

        def _fused_fwd_step(batch_iter, model_arg):
            batch = next(batch_iter)
            batch = batch.to(get_device_id())
            batch = batch.contiguous()
            mb_i = tF_counter[0]
            tF_counter[0] += 1
            if mb_i in old_log_probs_dict:
                batch["old_log_probs"] = old_log_probs_dict[mb_i].to(get_device_id())
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]
            responses = batch["responses"]
            rl = responses.size(1)
            label = position_ids.clone()
            label[:, -rl - 1 : -1] = responses
            label_mask = attention_mask.clone()
            label_mask[:, : -rl - 1] = False
            label_mask[:, -1] = False
            from verl.models.mcore import get_mcore_forward_fn, get_mcore_forward_fused_fn
            # Use the standard gptmodel_forward (same as baseline).
            # It packs sequences internally via preprocess_packed_seqs.
            # With dynamic shape gloo P2P, the packed activation shape
            # (NNZ, 1, H) is communicated correctly between ranks.
            if self.use_fused_kernels:
                output = get_mcore_forward_fused_fn(self.hf_config)(
                    model_arg, input_ids, position_ids, attention_mask,
                    sequence_parallel=self.tf_config.sequence_parallel,
                    labels=label, labels_mask=label_mask, temperature=temperature,
                )
            else:
                def _lp(logits, label, label_mask):
                    logits.div_(temperature)
                    ret = {}
                    if calculate_entropy:
                        ret["entropy"] = vocab_parallel_entropy(logits.clone())
                    log_probs = vocab_parallel_log_probs_from_logits(logits, label)
                    log_probs = log_probs.masked_fill(~label_mask, 0.0)
                    ret["log_probs"] = log_probs
                    return ret
                output = get_mcore_forward_fn(self.hf_config)(
                    model_arg, input_ids, attention_mask, position_ids,
                    sequence_parallel=self.tf_config.sequence_parallel,
                    logits_processor=_lp,
                    logits_processor_args={"label": label, "label_mask": label_mask},
                )
            meta = {"clip_ratio": self.config.clip_ratio,
                    "entropy_coeff": self.config.entropy_coeff,
                    "clip_ratio_c": self.config.get("clip_ratio_c", 3.0)}
            return output, _partial(_fused_loss, data=batch, meta_info=meta)

        def _fused_loss(output, data, meta_info):
            metrics = {}
            rl = data["responses"].size(1)
            response_mask = data["response_mask"].to(bool)
            log_prob = output["log_probs"][:, -rl - 1 : -1].contiguous()
            old_log_prob = data["old_log_probs"]
            advantages = data["advantages"]
            policy_loss_fn = get_policy_loss_fn(
                self.config.policy_loss.get("loss_mode", "vanilla"))
            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                old_log_prob=old_log_prob, log_prob=log_prob,
                advantages=advantages, response_mask=response_mask,
                loss_agg_mode=self.config.loss_agg_mode, config=self.config,
                rollout_is_weights=data.get("rollout_is_weights", None))
            stats = {"actor/pg_loss": pg_loss.detach().item(),
                     "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                     "actor/ppo_kl": ppo_kl.detach().item(),
                     "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item()}
            policy_loss = pg_loss
            if calculate_entropy:
                ent = output["entropy"][:, -rl - 1 : -1].contiguous()
                ent_loss = agg_loss(loss_mat=ent, loss_mask=response_mask,
                                    loss_agg_mode=self.config.loss_agg_mode)
                policy_loss = pg_loss - self.config.entropy_coeff * ent_loss
            if self.config.use_kl_loss:
                kld = kl_penalty(logprob=log_prob, ref_logprob=data["ref_log_prob"],
                                 kl_penalty=self.config.kl_loss_type)
                kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask,
                                   loss_agg_mode=self.config.loss_agg_mode)
                policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                metrics["actor/kl_loss"] = kl_loss.detach().item()
            from verl.utils.py_functional import append_to_dict
            append_to_dict(metrics, stats)
            return policy_loss, [metrics, None]

        # Zero grads
        self.actor_optimizer.zero_grad()
        for chunk in model:
            chunk.zero_grad_buffer()

        forward_data_store: list = []
        input_tensors: list = []
        output_tensors: list = []
        pending_sends: list = []
        batch_generator = iter(micro_batches)
        _infer_offloaded = False
        _t0 = _time.perf_counter()

        # ── Schedule loop ────────────────────────────────────────────
        for op_idx, op in enumerate(schedule_ops):
            mb = op.micro_batch_id
            _op_t0 = _time.perf_counter()

            # Offload HF replica before first tB (memory constraint)
            if op.op == "train_backward" and not _infer_offloaded:
                import os as _os_off
                if _os_off.environ.get("RLPIPE_FUSED_FORWARD_CPU_OFFLOAD", "0") == "1":
                    hf_stage.to("cpu")
                _infer_offloaded = True

            if op.op == "infer_forward":
                _if_ctx = (_pp_tracer.trace("infer_forward", micro_batch_id=mb)
                           if _pp_tracer else None)
                if _if_ctx: _if_ctx.__enter__()
                try:
                    if reverse_stage is not None:
                        # V2: PP-sharded reverse-direction inference.
                        # iF flows: rank P-1 → P-2 → ... → 0 via infer_* gloo groups.
                        mb_data = micro_batches[mb]
                        attn = mb_data["attention_mask"].to(get_device_id()).long()
                        pos = mb_data["position_ids"].to(get_device_id()).long()
                        if reverse_stage.is_first_infer:
                            # First infer (rank P-1): embedding + first layers
                            ids = mb_data["input_ids"].to(get_device_id())
                            hidden_out = reverse_stage.forward(
                                input_ids=ids, attention_mask=attn, position_ids=pos,
                            )
                            # Send to next infer rank (pp_rank - 1)
                            if pp_size > 1:
                                h = _p2p_send(hidden_out, pp_rank - 1, is_infer=True)
                                if isinstance(h, list): pending_sends.extend(h)
                                else: pending_sends.append(h)
                        elif reverse_stage.is_last_infer:
                            # Last infer (rank 0): recv hidden, run last layers, compute log_probs
                            hidden_in = _p2p_recv(
                                pp_rank + 1, dtype=torch.bfloat16, is_infer=True,
                            )
                            hidden_out = reverse_stage.forward(
                                input_hidden=hidden_in, attention_mask=attn, position_ids=pos,
                                return_hidden=True,
                            )
                            ids = mb_data["input_ids"].to(get_device_id())
                            log_probs = reverse_stage.compute_log_probs_from_hidden(
                                hidden_out, ids, response_length, temperature=temperature,
                            )
                            old_log_probs_dict[mb] = log_probs.cpu()
                            # Send log_probs to last train rank (pp_size - 1) via olp
                            if pp_size > 1:
                                h = _p2p_send(log_probs, pp_size - 1, is_infer=False)
                                if isinstance(h, list): pending_sends.extend(h)
                                else: pending_sends.append(h)
                        else:
                            # Middle infer: recv → forward → send
                            hidden_in = _p2p_recv(
                                pp_rank + 1, dtype=torch.bfloat16, is_infer=True,
                            )
                            hidden_out = reverse_stage.forward(
                                input_hidden=hidden_in, attention_mask=attn, position_ids=pos,
                            )
                            h = _p2p_send(hidden_out, pp_rank - 1, is_infer=True)
                            if isinstance(h, list): pending_sends.extend(h)
                            else: pending_sends.append(h)
                    else:
                        # V1: Full HF replica on-demand GPU load.
                        torch.cuda.empty_cache()
                        hf_stage.to(device)
                        mb_data = micro_batches[mb]
                        old_lp = hf_stage.forward_log_probs(
                            input_ids=mb_data["input_ids"].to(get_device_id()),
                            attention_mask=mb_data["attention_mask"].to(get_device_id()).long(),
                            position_ids=mb_data["position_ids"].to(get_device_id()).long(),
                            response_length=response_length,
                            temperature=temperature,
                        )
                        old_log_probs_dict[mb] = old_lp.to(torch.float32).cpu()
                        hf_stage.to("cpu")
                        torch.cuda.empty_cache()
                    _alloc = torch.cuda.memory_allocated() / (1024**3)
                    print(f"[fused-mem] rank={pp_rank} after iF.{mb}: "
                          f"alloc={_alloc:.1f}G", flush=True)
                finally:
                    if _if_ctx: _if_ctx.__exit__(None, None, None)

            elif op.op == "train_forward":
                _tf_ctx = (_pp_tracer.trace("train_forward", micro_batch_id=mb)
                           if _pp_tracer else None)
                if _tf_ctx: _tf_ctx.__enter__()
                try:
                    # With reverse-PP inference: last train rank recvs
                    # old_log_probs[mb] from last infer rank (rank 0) via olp
                    # group. Needed by loss_func. Skipped if already present
                    # (rank 0 is both last infer and first train for PP=1).
                    if (reverse_stage is not None and is_last_stage
                            and pp_size > 1 and mb not in old_log_probs_dict):
                        lp = _p2p_recv(
                            0, dtype=torch.float32, is_infer=False,
                        )
                        old_log_probs_dict[mb] = lp
                    # Recv activation from previous rank (blocking)
                    input_tensor = None
                    if not is_first_stage:
                        input_tensor = _p2p_recv(
                            pp_rank - 1,
                            dtype=getattr(config, "pipeline_dtype", torch.bfloat16),
                        )
                        input_tensor.requires_grad_(True)
                    # Forward (Megatron forward_step handles set_input_tensor)
                    output_tensor, num_tokens = mg_forward_step(
                        _fused_fwd_step, batch_generator, model[0], M,
                        input_tensor, forward_data_store, config,
                        mpu.get_context_parallel_world_size()
                        if hasattr(mpu, "get_context_parallel_world_size") else 1,
                        current_microbatch=mb, is_last_stage=is_last_stage,
                    )
                    # Non-blocking gloo send to next rank
                    if not is_last_stage:
                        t = (output_tensor if isinstance(output_tensor, torch.Tensor)
                             else output_tensor[0])
                        handle = _p2p_send(t, pp_rank + 1)
                        if isinstance(handle, list):
                            pending_sends.extend(handle)
                        else:
                            pending_sends.append(handle)
                    _alloc_tf = torch.cuda.memory_allocated() / (1024**3)
                    print(f"[fused-mem] rank={pp_rank} after tF.{mb}: "
                          f"alloc={_alloc_tf:.1f}G", flush=True)
                finally:
                    if _tf_ctx: _tf_ctx.__exit__(None, None, None)
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)

            elif op.op == "train_backward":
                _tb_ctx = (_pp_tracer.trace("train_backward", micro_batch_id=mb)
                           if _pp_tracer else None)
                if _tb_ctx: _tb_ctx.__enter__()
                try:
                    # Recv grad from next rank (blocking)
                    output_tensor_grad = None
                    if not is_last_stage:
                        output_tensor_grad = _p2p_recv(
                            pp_rank + 1,
                            dtype=getattr(config, "pipeline_dtype", torch.bfloat16),
                        )
                    in_t = input_tensors.pop(0)
                    out_t = output_tensors.pop(0)
                    input_tensor_grad = mg_backward_step(
                        in_t, out_t, output_tensor_grad, None, config)
                    # Non-blocking gloo send grad to previous rank
                    if not is_first_stage:
                        g = (input_tensor_grad if isinstance(input_tensor_grad, torch.Tensor)
                             else input_tensor_grad[0])
                        handle = _p2p_send(g, pp_rank - 1)
                        if isinstance(handle, list):
                            pending_sends.extend(handle)
                        else:
                            pending_sends.append(handle)
                finally:
                    if _tb_ctx: _tb_ctx.__exit__(None, None, None)

            _op_dt = _time.perf_counter() - _op_t0
            if op_idx < 20 or op_idx == len(schedule_ops) - 1 or op.op == "infer_forward":
                print(f"[fused-sched] rank={pp_rank} mb={_mb_idx} op[{op_idx}]="
                      f"{op.op}.{mb} dt={_op_dt:.3f}s", flush=True)

        # Wait for all async sends to complete before freeing tensors
        for handle in pending_sends:
            handle.wait()

        torch.cuda.synchronize()
        _total = _time.perf_counter() - _t0
        print(f"[fused-sched] rank={pp_rank} mb={_mb_idx} done: "
              f"{len(schedule_ops)} ops in {_total:.2f}s", flush=True)

        # Finalize gradients & optimizer step
        if config.finalize_model_grads_func is not None:
            config.finalize_model_grads_func(model)

        update_successful, grad_norm, num_zeros_in_grad = (
            self.actor_optimizer.step()
        )
        if not update_successful:
            raise RuntimeError("fused_forward_backward: optimizer step failed")

        # Collect metrics
        mb_metrics = {}
        if forward_data_store:
            for item in forward_data_store:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    if isinstance(item[0], dict):
                        from verl.utils.py_functional import append_to_dict
                        append_to_dict(mb_metrics, item[0])
        mb_metrics["actor/grad_norm"] = grad_norm

        # old_log_probs for this mini-batch.
        # With reverse-PP, only rank 0 (last infer) and rank P-1 (last train)
        # have old_log_probs populated. Middle ranks return None — the caller
        # (fused_forward_backward) collects from the ranks that have them.
        if old_log_probs_dict:
            mb_old_lp = torch.cat(
                [old_log_probs_dict[i] for i in range(M)], dim=0
            )
        else:
            mb_old_lp = None

        _send_bufs.clear()
        return mb_metrics, mb_old_lp

