"""HBM probe: how much GPU memory does a single forward+backward of an
8B model on an 18432-token sample actually cost?

Two experiments, both deterministic and synthetic (no rollout required):

  A) Full HF model on a single GPU — reference. Measures peak bf16 peak
     with FlashAttention + HF's gradient checkpointing toggle.
  B) Single torch_pp PipelineStage on one GPU — isolates one stage's
     forward/backward memory footprint; the real training run shards
     across 4 ranks so per-rank footprint should be ~1/4 of (A)'s
     activation term (weights/grads are already sharded by PP).

Run directly (bypasses pytest and verl's Ray plumbing):

    python tests/torch_pp/test_hbm_probe.py \
        --model /path/to/Qwen3-8B \
        --seq 18432 --batch 1 --stage full
    python tests/torch_pp/test_hbm_probe.py \
        --model /path/to/Qwen3-8B \
        --seq 18432 --batch 1 --stage pp --pp-rank 1 --pp-size 4

The point is to pinpoint whether verl's `update_actor` peak of ~70 GB
is inherent (so we must cut max_resp) or a leak/missing checkpoint
(so we fix the code).
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

import torch

# Make sure verl imports work when running from the tests/ dir.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def reset_peak():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def report(label: str):
    alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
    print(
        f"[{label}] peak alloc={alloc:.2f} GB  "
        f"peak reserved={reserved:.2f} GB"
    )


def probe_full_model(model_path: str, seq: int, batch: int,
                     checkpoint: bool, dtype=torch.bfloat16):
    """Experiment A: full HF model forward+backward on one GPU."""
    from transformers import AutoConfig, AutoModelForCausalLM

    device = torch.device("cuda:0")
    reset_peak()
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device)

    if checkpoint:
        model.gradient_checkpointing_enable()
    model.train()
    report("A: after model load")

    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    attention_mask = torch.ones(batch, seq, device=device, dtype=torch.long)
    report("A: after input alloc")

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
        use_cache=False,
    )
    report("A: after forward")

    out.loss.backward()
    report("A: after backward")

    print(
        f"  config: layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
        f"kv_heads={getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)} "
        f"intermediate={cfg.intermediate_size}"
    )


def probe_pp_stage(model_path: str, seq: int, batch: int,
                   pp_rank: int, pp_size: int, checkpoint: bool,
                   dtype=torch.bfloat16):
    """Experiment B: single torch_pp PipelineStage on one GPU.

    Simulates the middle of a PP pipeline: the stage receives
    hidden-state input (batch × seq × hidden) and produces either
    hidden-state (non-last) or logits (last stage). We drive
    forward+backward directly without dist.send/recv by supplying
    synthetic upstream gradients."""
    from transformers import AutoConfig
    from verl.workers.torch_pp.pipeline_stage import PipelineStage

    device = torch.device("cuda:0")
    reset_peak()

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    stage = PipelineStage.from_pretrained(
        model_path=model_path,
        pp_rank=pp_rank,
        pp_size=pp_size,
        device=device,
        dtype=dtype,
        trust_remote_code=True,
        enable_gradient_checkpointing=checkpoint,
    )
    stage.train()
    report(f"B: after stage {pp_rank}/{pp_size} load")

    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    attention_mask = torch.ones(batch, seq, device=device, dtype=torch.long)

    # First stage consumes input_ids; middle/last stages consume
    # hidden. Simulate the middle case by building a fake hidden tensor
    # that requires grad, so a backward can flow.
    M = 1  # single micro-batch
    stage.set_batch_data(input_ids, attention_mask, M)

    if stage.is_first:
        out = stage.forward_step(0)
    else:
        hidden_in = torch.randn(
            batch, seq, cfg.hidden_size,
            device=device, dtype=dtype, requires_grad=True,
        )
        out = stage.forward_step(0, hidden_in)
    report(f"B: after forward (stage is_first={stage.is_first} "
           f"is_last={stage.is_last})")

    # Build a loss. For the last stage `out` is logits; for others
    # `out` is hidden — in either case, sum() is a valid differentiable
    # scalar that drives a full-shaped backward.
    loss = out.float().sum()
    loss.backward()
    report("B: after backward")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model path (e.g. Qwen3-8B snapshot dir)")
    p.add_argument("--seq", type=int, default=18432)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--stage", choices=["full", "pp"], default="full")
    p.add_argument("--pp-rank", type=int, default=0)
    p.add_argument("--pp-size", type=int, default=4)
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Disable gradient checkpointing")
    args = p.parse_args()

    if args.stage == "full":
        probe_full_model(args.model, args.seq, args.batch,
                         checkpoint=not args.no_checkpoint)
    else:
        probe_pp_stage(args.model, args.seq, args.batch,
                       args.pp_rank, args.pp_size,
                       checkpoint=not args.no_checkpoint)


if __name__ == "__main__":
    main()
