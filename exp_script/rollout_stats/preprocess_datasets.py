"""Write a prompts JSONL from a supported dataset.

Accepts dataset name; looks up the right HF dataset / local parquet,
extracts the "problem" / "question" text, wraps it in a basic chat
template using the model's tokenizer, and emits one
{"prompt_id": int, "prompt": str, "source": str, "raw_answer": str|None}
per line.

Supported datasets:
  dapo-math-17k   (BytedTsinghua-SIA/DAPO-Math-17k, or local parquet)
  aime-24         (Maxwell-Jia/AIME_2024)
  aime-25         (AIMO/aime_2025)
  math-500        (HuggingFaceH4/MATH-500)
  livecodebench   (livecodebench/code_generation_lite)
  codecontests    (deepmind/code_contests)

Use a "neutral" chat template (just the raw problem in the user turn)
by default so tokenizer-level chat template isn't required for
rollout-only stats. Pass --use-chat-template to apply the model's
tokenizer chat template.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

# Silence HF datasets progress bar for cleaner logs.
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")


def _load(name: str, max_examples: int | None):
    """Return list[dict] with keys {"question", "answer", "source"}."""
    from datasets import load_dataset
    if name == "dapo-math-17k":
        # Prefer local parquet if exists.
        local = "/home/user/data/dapo-math-4k/train.parquet"
        if os.path.exists(local):
            import pyarrow.parquet as pq
            tbl = pq.read_table(local).to_pylist()
            out = []
            for row in tbl:
                # Local DAPO-Math parquet uses 'prompt' as a list of messages.
                prompt_field = row.get("prompt")
                if isinstance(prompt_field, list) and prompt_field:
                    q = prompt_field[0].get("content", "")
                else:
                    q = str(prompt_field or row.get("question") or "")
                rm = row.get("reward_model", {}) or {}
                a = rm.get("ground_truth") if isinstance(rm, dict) else None
                out.append({"question": q, "answer": a, "source": "dapo-math-17k"})
            if max_examples:
                out = out[:max_examples]
            return out
        ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
        return [
            {"question": ex.get("prompt") or ex.get("problem") or ex.get("question"),
             "answer": ex.get("answer") or ex.get("ground_truth"),
             "source": "dapo-math-17k"}
            for ex in (ds.select(range(min(max_examples, len(ds)))) if max_examples else ds)
        ]
    if name == "aime-24":
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        return [
            {"question": ex["Problem"], "answer": str(ex.get("Answer")),
             "source": "aime-24"}
            for ex in (ds.select(range(min(max_examples, len(ds)))) if max_examples else ds)
        ]
    if name == "aime-25":
        ds = load_dataset("opencompass/AIME2025", split="test")
        return [
            {"question": ex.get("question") or ex.get("problem"),
             "answer": str(ex.get("answer")), "source": "aime-25"}
            for ex in (ds.select(range(min(max_examples, len(ds)))) if max_examples else ds)
        ]
    if name == "math-500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [
            {"question": ex["problem"], "answer": ex.get("answer"),
             "source": "math-500"}
            for ex in (ds.select(range(min(max_examples, len(ds)))) if max_examples else ds)
        ]
    if name == "livecodebench":
        # Upstream `code_generation_lite` uses a legacy loading script
        # that datasets>=4.0 rejects. Fetch the latest test JSONL directly
        # from the HF hub via hf_hub_download and parse ourselves. The
        # latest cut is test6.jsonl (updated 2025); fall back to earlier
        # if unavailable.
        from huggingface_hub import hf_hub_download
        candidates = ["test6.jsonl", "test5.jsonl", "test4.jsonl", "test3.jsonl", "test2.jsonl", "test.jsonl"]
        rows = []
        for fn in candidates:
            try:
                path = hf_hub_download(
                    repo_id="livecodebench/code_generation_lite",
                    filename=fn, repo_type="dataset",
                )
                with open(path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        ex = json.loads(line)
                        q = ex.get("question_content") or ex.get("problem")
                        if q:
                            rows.append({
                                "question": q, "answer": None,
                                "source": "livecodebench",
                            })
                break  # first successful file wins
            except Exception:
                continue
        if not rows:
            raise RuntimeError("Could not load livecodebench from hub")
        if max_examples:
            rows = rows[:max_examples]
        return rows
    if name == "codecontests":
        ds = load_dataset("deepmind/code_contests", split="test")
        return [
            {"question": ex["description"], "answer": None,
             "source": "codecontests"}
            for ex in (ds.select(range(min(max_examples, len(ds)))) if max_examples else ds)
        ]
    raise ValueError(f"Unknown dataset: {name}")


def _apply_chat_template(tokenizer, q: str) -> str:
    msgs = [{"role": "user", "content": q}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument(
        "--model-path", default=None,
        help="If set, applies the tokenizer's chat template. "
             "Otherwise raw question text is used."
    )
    args = p.parse_args()

    examples = _load(args.dataset, args.max_examples)

    tokenizer = None
    if args.model_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as f:
        for i, ex in enumerate(examples):
            q = (ex["question"] or "").strip()
            if not q:
                continue
            prompt = _apply_chat_template(tokenizer, q) if tokenizer else q
            f.write(json.dumps({
                "prompt_id": i,
                "prompt": prompt,
                "source": ex["source"],
                "raw_answer": ex.get("answer"),
            }) + "\n")
    print(f"wrote {len(examples)} prompts → {outp}")


if __name__ == "__main__":
    sys.exit(main())
