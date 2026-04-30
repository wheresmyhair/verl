"""Preprocess HuggingFaceH4/MATH-500 → verl parquet (DAPO-compatible
schema).

Produces train.parquet + val.parquet under --out_dir, with the same
fields as the DAPO-Math-17k parquets so the existing
`reward_model.reward_manager=dapo` + `style=rule-lighteval/MATH_v2`
machinery can score this dataset without changes.

Default split: 400 train / 100 val (random shuffle, seed=42).
"""
from __future__ import annotations
import argparse
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq

# Reuse the exact DAPO-Math prompt template so the model is asked
# to put its final answer on a single line after "Answer:" — the
# DAPO reward manager + rule-lighteval/MATH_v2 style depend on this.
PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of "
    "your response should be of the form Answer: $Answer (without quotes) "
    "where $Answer is the answer to the problem.\n\n"
    "{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)


def build_row(ex: dict) -> dict:
    return {
        # Use "math" so verl.utils.reward_score.__init__ dispatches to
        # math_dapo.compute_score (same handler used for math_dapo /
        # aime); MATH-500 answers are MATH_v2-style.
        "data_source": "math",
        "prompt": [
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(problem=ex["problem"]),
            }
        ],
        "ability": "MATH",
        "reward_model": {
            "ground_truth": ex["answer"],
            "style": "rule-lighteval/MATH_v2",
        },
        "extra_info": {
            "index": ex["unique_id"],
            "level": int(ex["level"]),
            "subject": ex["subject"],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_dir",
        default=os.path.expanduser("~/data/math-500"),
    )
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = [build_row(ex) for ex in ds]

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    train_rows = rows[: args.n_train]
    val_rows = rows[args.n_train:]

    os.makedirs(args.out_dir, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(train_rows),
        os.path.join(args.out_dir, "train.parquet"),
    )
    pq.write_table(
        pa.Table.from_pylist(val_rows),
        os.path.join(args.out_dir, "val.parquet"),
    )
    print(
        f"wrote {len(train_rows)} train + {len(val_rows)} val rows "
        f"→ {args.out_dir}/{{train,val}}.parquet"
    )


if __name__ == "__main__":
    main()
