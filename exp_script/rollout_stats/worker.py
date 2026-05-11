"""Rollout-only response-length stats worker.

One worker = one SGLang engine on a contiguous GPU range. Reads a chunk
of prompts (JSONL, one `{"prompt_id": int, "prompt": str}` per line),
generates n samples per prompt with the paper's RLVR config
(max_new_tokens=16384, temperature=1.0, top_p=1.0), writes one
`{"prompt_id", "sample_id", "response_tokens", "finish_reason",
"prompt_tokens"}` JSONL per sampled output.

No reward, no training. Purpose is pure length distribution + CV
analysis across the n samples of the same prompt.

Intended usage: forked in parallel by run.sh with CUDA_VISIBLE_DEVICES
narrowed per worker. Each worker is independent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def load_prompts(path: str):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--prompts", required=True, help="JSONL with prompt_id + prompt")
    p.add_argument("--out", required=True, help="Output JSONL path")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=16384)
    p.add_argument("--max-prompt-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--mem-fraction", type=float, default=0.85,
                   help="SGLang gpu_memory_utilization (mem_fraction_static)")
    p.add_argument("--attention-backend", default="flashinfer")
    p.add_argument(
        "--batch-size", type=int, default=16,
        help="Number of prompts submitted per engine.generate call "
             "(SGLang runs them as a single large batch, internally "
             "continuous-batched)."
    )
    p.add_argument("--flush", type=int, default=50,
                   help="Flush output file every N prompts")
    args = p.parse_args()

    # Import SGLang lazily so arg parsing failures don't pay import cost.
    import sglang as sgl

    prompts = load_prompts(args.prompts)
    print(f"[worker pid={os.getpid()}] loaded {len(prompts)} prompts "
          f"model={args.model_path} tp={args.tp_size} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
          flush=True)

    engine_kwargs = dict(
        model_path=args.model_path,
        dtype=args.dtype,
        tp_size=args.tp_size,
        mem_fraction_static=args.mem_fraction,
        attention_backend=args.attention_backend,
        trust_remote_code=True,
        log_level="warning",
    )
    t_boot = time.time()
    engine = sgl.Engine(**engine_kwargs)
    print(f"[worker pid={os.getpid()}] engine boot {time.time() - t_boot:.1f}s",
          flush=True)

    sp = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "n": args.n_samples,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    of = out_path.open("w", buffering=1)
    seen_prompts = 0

    # Submit in small batches; SGLang continuously batches internally.
    for i in range(0, len(prompts), args.batch_size):
        chunk = prompts[i : i + args.batch_size]
        chunk_prompts = [r["prompt"] for r in chunk]
        t0 = time.time()
        outs = engine.generate(chunk_prompts, sampling_params=[sp] * len(chunk_prompts))
        dt = time.time() - t0

        # SGLang returns a FLAT list of len(prompts) × n dicts when n>1,
        # in the order: prompt0_sample0, prompt0_sample1, ..., prompt1_sample0, ...
        expected = len(chunk) * args.n_samples
        if len(outs) != expected:
            print(f"[worker pid={os.getpid()}] WARN unexpected output count: "
                  f"got {len(outs)} expected {expected}", flush=True)
        for i, rec in enumerate(chunk):
            prompt_id = rec["prompt_id"]
            start = i * args.n_samples
            per_prompt = outs[start : start + args.n_samples]
            for s_idx, o in enumerate(per_prompt):
                mi = o.get("meta_info", {}) if isinstance(o, dict) else {}
                fr = mi.get("finish_reason", {}) or {}
                of.write(json.dumps({
                    "prompt_id": prompt_id,
                    "sample_id": s_idx,
                    "response_tokens": int(mi.get("completion_tokens", 0)),
                    "prompt_tokens": int(mi.get("prompt_tokens", 0)),
                    "finish_type": fr.get("type") if isinstance(fr, dict) else None,
                    "finish_matched": fr.get("matched") if isinstance(fr, dict) else None,
                }) + "\n")
        seen_prompts += len(chunk)
        print(f"[worker pid={os.getpid()}] done {seen_prompts}/{len(prompts)} "
              f"chunk={len(chunk)} dt={dt:.1f}s", flush=True)
        if seen_prompts % args.flush == 0:
            of.flush()

    of.close()
    engine.shutdown()
    print(f"[worker pid={os.getpid()}] finished", flush=True)


if __name__ == "__main__":
    sys.exit(main())
