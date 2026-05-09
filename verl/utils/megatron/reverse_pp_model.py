"""Build a Megatron GPTModel with REVERSED pipeline-parallel layer assignment.

For fused-forward (idea 2) we run inference iF flowing rank P-1 → 0
(reverse direction) so that the deepest-warmup-bubble rank (P-1) can
start iFs immediately without waiting for upstream training tFs to
propagate. This requires an inference model whose layer offset on
each PP rank is the REVERSE of the training model's:

  training PP rank r → has layers [r·L/P, (r+1)·L/P)
  inference PP rank r → has layers [(P-1-r)·L/P, (P-r)·L/P)

Approach
--------
Megatron's `GPTModel(pg_collection=...)` and
`TransformerBlock(pg_collection=...)` derive layer offsets and
pre_process/post_process from `get_pg_rank(pg_collection.pp)` — i.e.
the calling rank's INDEX inside the pp ProcessGroup. By giving the
inference model a pp_group whose rank ordering is REVERSED relative
to the training pp_group, the same construction code produces the
reversed layer assignment automatically.

Concretely:
  train pp_group ranks:    [g0, g1, g2, g3]    (in-group rank: 0,1,2,3)
  infer pp_group ranks:    [g3, g2, g1, g0]    (in-group rank: 0,1,2,3)
                                                 (each global rank's
                                                  in-group rank is
                                                  (P-1) - its train rank)

So on world rank g3 (training pp rank 3), the inference pp_group rank
is 0 → infer model gets pre_process=True (embedding lives here).
On world rank g0 (training pp rank 0), inf pp_group rank = 3 →
post_process=True (lm_head lives here).

API
---
    rev_collection = build_reverse_pp_pg_collection(actor_pg_collection)
    infer_model = build_reverse_pp_inference_model(
        actor_model_config, rev_collection,
        share_embeddings_and_output_weights=...,
    )
    # infer_model has reversed-PP layer assignment + reversed
    # pre_process / post_process. Use it for iF compute (eval mode).

Notes
-----
- This module does NOT load weights. Caller must replicate
  weights from the training model (via state_dict copy) since
  inference shares the same parameters as training (just rearranged
  spatially across PP ranks).
- Weight transfer between train_module's PP shard and
  infer_module's PP shard requires gathering the FULL state_dict
  on each rank then slicing to local layers — expensive (~16 GB for
  Qwen3-8B). Done once at init, then iF reads from infer_module.
- `dist.new_group` is collective: every world rank must call this.
  Caller must wrap setup in collective context.
"""
from __future__ import annotations
import logging
from typing import Optional

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


def build_reverse_pp_group(actor_pp_group) -> dist.ProcessGroup:
    """Create a process group spanning the same world ranks as the actor's
    pp group, but reused as the inference pp group.

    NOTE: pytorch's `dist.new_group(ranks=...)` SORTS the ranks before
    assigning in-group rank, so we cannot get "reverse" in-group ranks
    just by passing reversed `ranks=`. Instead we reuse the actor's
    pp group and rely on a monkey-patch context manager (see
    `_reverse_pp_rank_context`) during inference model construction
    to make Megatron's pp_rank lookups return (P-1 - actor_pp_rank).
    """
    return actor_pp_group


def build_reverse_pp_pg_collection(actor_pg_collection):
    """Use actor's pg_collection as-is. Layer-offset reversal is achieved
    via monkey-patching during inference-model construction, not via a
    different pg.
    """
    return actor_pg_collection


import contextlib

@contextlib.contextmanager
def _reverse_pp_rank_context(pp_size: int, actor_pp_rank: int):
    """While active, megatron's pp_rank lookups (in parallel_state and
    in core.utils.get_pg_rank for the actor's pp_group) return
    (pp_size - 1 - actor_pp_rank). Used during inference-model
    construction to flip layer-offset/pre_process/post_process.

    Patches:
      - megatron.core.parallel_state.get_pipeline_model_parallel_rank
      - megatron.core.utils.get_pg_rank
      - megatron.core.transformer.transformer_block.get_pg_rank
        (re-imported binding inside the module)
    """
    import megatron.core.parallel_state as ps
    import megatron.core.utils as mcu
    import megatron.core.transformer.transformer_block as tb

    reversed_rank = pp_size - 1 - actor_pp_rank

    orig_ps_get_rank = ps.get_pipeline_model_parallel_rank
    orig_mcu_get_pg_rank = mcu.get_pg_rank
    orig_tb_get_pg_rank = tb.get_pg_rank

    def patched_ps_get_rank():
        return reversed_rank

    def patched_get_pg_rank(group=None):
        # Return reversed rank only for the actor's pp_group;
        # other groups (tp, dp, ...) keep stock behavior.
        if group is None or not torch.distributed.is_initialized():
            return 0
        try:
            actual = group.rank()
        except Exception:
            actual = 0
        # Heuristic: if group's size matches pp_size, treat as pp_group
        # and reverse. (We can't easily compare ProcessGroup identity
        # since we may have multiple references.)
        try:
            if group.size() == pp_size:
                return reversed_rank
        except Exception:
            pass
        return actual

    ps.get_pipeline_model_parallel_rank = patched_ps_get_rank
    mcu.get_pg_rank = patched_get_pg_rank
    tb.get_pg_rank = patched_get_pg_rank
    try:
        yield reversed_rank
    finally:
        ps.get_pipeline_model_parallel_rank = orig_ps_get_rank
        mcu.get_pg_rank = orig_mcu_get_pg_rank
        tb.get_pg_rank = orig_tb_get_pg_rank


def build_reverse_pp_inference_model(
    actor_tf_config,
    actor_hf_config,
    actor_pg_collection,
    *,
    share_embeddings_and_output_weights: bool = False,
    parallel_output: bool = True,
):
    """Build a GPTModel using `actor_tf_config` but with a
    pg_collection whose pp_group is reversed.

    Returns the GPTModel instance (NOT wrapped in DDP — inference
    only). Weights are NOT loaded; caller must copy from actor.

    Collective: every world rank must call this.
    """
    from megatron.core.models.gpt.gpt_model import GPTModel

    # 1. Same pg_collection as actor (we don't try to literally reverse the
    # process group — pytorch sorts ranks during new_group).
    rev_collection = build_reverse_pp_pg_collection(actor_pg_collection)

    # 2. Determine pre_process / post_process from REVERSED in-group rank.
    # The construction code (under our monkey-patch context) will see
    # the reversed rank, so embedding/lm_head/layer-offset all flip.
    actor_pp_rank = dist.get_rank(group=rev_collection.pp)
    pp_size = dist.get_world_size(group=rev_collection.pp)
    rev_pp_rank = pp_size - 1 - actor_pp_rank
    pre_process = (rev_pp_rank == 0)           # embedding lives at rev rank 0 (= train rank P-1)
    post_process = (rev_pp_rank == pp_size - 1)  # lm_head lives at rev rank P-1 (= train rank 0)

    if dist.get_rank() == 0:
        logger.info(
            "[reverse-PP infer] world ranks in rev_pp_group: %s; this rank rev_pp=%d, pre=%s, post=%s",
            dist.get_process_group_ranks(rev_collection.pp), rev_pp_rank, pre_process, post_process,
        )

    # 3. Use verl's existing model initializer to get the right
    # transformer_layer_spec (handles Qwen2/3 dense + MoE etc.),
    # then build GPTModel ourselves with our reversed pg_collection.
    from verl.models.mcore.registry import get_supported_model, MODEL_INITIALIZER_REGISTRY

    arch = actor_hf_config.architectures[0]
    model_key = get_supported_model(arch)
    initializer_cls = MODEL_INITIALIZER_REGISTRY[model_key]
    initializer = initializer_cls(actor_tf_config, actor_hf_config)
    transformer_layer_spec = initializer.get_transformer_layer_spec()

    rope_scaling_args = initializer.get_rope_scaling_args()
    rotary_base = getattr(actor_hf_config, "rope_theta", 10000)

    # Build under monkey-patch so layer offsets / pre_process / post_process
    # are computed for the REVERSED pp_rank.
    with _reverse_pp_rank_context(pp_size, actor_pp_rank):
        gpt_model = GPTModel(
            config=actor_tf_config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=actor_hf_config.vocab_size,
            max_sequence_length=actor_hf_config.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type="rope",
            rotary_base=rotary_base,
            parallel_output=parallel_output,
            pg_collection=rev_collection,
            **rope_scaling_args,
        )

    return gpt_model, rev_collection
