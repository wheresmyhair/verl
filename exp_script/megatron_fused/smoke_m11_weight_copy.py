"""M11 — verify copy_actor_to_reverse_infer round-trips weights correctly.

Builds an actor GPTModel (forward-PP) and an infer GPTModel (reverse-PP) per
rank, runs `copy_actor_to_reverse_infer`, then checks that for every common
key in infer's state_dict, the copied data on rank r matches the original
actor data on rank (P-1-r) bit-exactly.

This is a fast self-contained smoke (no verl trainer / sglang). Builds the
two models from a tiny synthetic config to keep memory / time low.

Run:
  torchrun --nproc-per-node=4 \
    /home/user/rlpipe/verl/exp_script/megatron_fused/smoke_m11_weight_copy.py
"""
from __future__ import annotations
import os
import sys

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://", device_id=torch.device(f"cuda:{rank}"))

    sys.path.insert(0, "/home/user/rlpipe/verl")

    from megatron.core import parallel_state as mpu
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.transformer_config import TransformerConfig

    # Init megatron parallel state with PP=world_size, TP=1, DP=1
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=world_size,
    )

    pp_group = mpu.get_pipeline_model_parallel_group()
    actor_pp_rank = dist.get_rank(group=pp_group)
    pp_size = world_size

    # Tiny GPT config
    tf_config = TransformerConfig(
        num_layers=8,
        hidden_size=128,
        num_attention_heads=4,
        kv_channels=32,
        num_query_groups=4,
        ffn_hidden_size=256,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        add_qkv_bias=False,
        qk_layernorm=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pp_size,
        sequence_parallel=False,
        variable_seq_lengths=True,
    )
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    layer_spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)

    # Actor model (forward-PP, layer offset = pp_rank * L/P)
    actor = GPTModel(
        config=tf_config,
        transformer_layer_spec=layer_spec,
        vocab_size=128,
        max_sequence_length=64,
        pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
        post_process=mpu.is_pipeline_last_stage(ignore_virtual=True),
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_base=10000,
    ).cuda()

    # Reset parameters with a per-rank seed so each shard has distinct values
    torch.manual_seed(1000 + rank)
    for p in actor.parameters():
        if p.numel() > 0:
            p.data.normal_(mean=0.0, std=0.05)

    # Build infer (reverse-PP)
    from verl.utils.megatron.reverse_pp_model import (
        build_reverse_pp_inference_model,
        copy_actor_to_reverse_infer,
    )

    pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    class FakeHF:
        vocab_size = 128
        max_position_embeddings = 64
        rope_theta = 10000
        architectures = ["Qwen3ForCausalLM"]

    infer = build_reverse_pp_inference_model(
        actor_tf_config=tf_config,
        actor_hf_config=FakeHF(),
        actor_pg_collection=pg_collection,
        share_embeddings_and_output_weights=False,
        parallel_output=True,
    ).cuda()

    if rank == 0:
        print(f"[r{rank}] actor params: {sum(p.numel() for p in actor.parameters())}")
        print(f"[r{rank}] infer params: {sum(p.numel() for p in infer.parameters())}")

    # Snapshot per-rank actor state pre-copy (to compare later: we copy peer's
    # actor state into our infer; verify it matches).
    actor_state_local = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items() if v is not None}

    # Gather everyone's actor state for verification (post-copy oracle)
    all_actor_states = [None] * pp_size
    dist.all_gather_object(all_actor_states, actor_state_local, group=pp_group)

    expected_peer = all_actor_states[pp_size - 1 - actor_pp_rank]

    # Run weight copy
    copy_actor_to_reverse_infer(
        actor_gpt_model=actor,
        infer_gpt_model=infer,
        actor_pp_group=pp_group,
    )

    # Validate: for each key infer has, it should match expected_peer's value
    infer_state = infer.state_dict()
    n_compared = 0
    n_mismatched = 0
    for k, v_infer in infer_state.items():
        if v_infer is None:
            continue
        if k not in expected_peer:
            print(f"[r{rank}] WARN key {k} in infer but missing in expected peer state")
            continue
        v_expected = expected_peer[k].cuda()
        if not torch.equal(v_infer, v_expected):
            n_mismatched += 1
            if n_mismatched < 3:
                print(
                    f"[r{rank}] MISMATCH at key {k}: infer={v_infer.flatten()[:3].tolist()}, "
                    f"expected={v_expected.flatten()[:3].tolist()}"
                )
        n_compared += 1

    print(f"[r{rank}] compared={n_compared}, mismatched={n_mismatched}", flush=True)
    dist.barrier()

    # Aggregate result
    mismatch_t = torch.tensor([n_mismatched], device="cuda")
    dist.all_reduce(mismatch_t)
    if rank == 0:
        total_mismatch = mismatch_t.item()
        if total_mismatch == 0:
            print(f"\n=== M11 PASS: copy_actor_to_reverse_infer is bit-exact across all {pp_size} ranks ===", flush=True)
        else:
            print(f"\n=== M11 FAIL: {total_mismatch} mismatches across ranks ===", flush=True)
            sys.exit(1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
