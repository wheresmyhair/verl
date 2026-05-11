"""Build 6 length-profile datasets from the 27-cell rollout data.

Each profile is defined by an acceptance window over (mean, CV) of
the per-prompt 16-sample response-length distribution from
Qwen3-8B (the base model for downstream training experiments).

Outputs per profile:
  - <profile>/prompts_lengths.json  — {prompt_id, src, lengths: [16 ints]}
                                       (simulator-ready, no reward needed)
  - <profile>/train.parquet, val.parquet — verl-format with reward
                                            (math profiles P1-P5 only;
                                            P6 is simulator-only because
                                            it's saturated code/cap-bound
                                            and reward is messy)

Profile design: see report §"中场" (length distribution) — these
profiles span the trajectory of RL training (early-short → typical-mid
→ wide-mixed → late-long → bimodal-transition → saturated worst-case).
"""
from __future__ import annotations
import argparse
import json
import os
import random
import statistics
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROLLOUT_ROOT = Path("/home/user/profiling_rlpipe/rollout_stats")
MODEL = "Qwen3-8B"
DATASETS = ["dapo-math-17k", "math-500", "aime-24"]  # math-only for verl reward
ALL_DATASETS = DATASETS + ["livecodebench", "codecontests"]

PROFILES = {
    "P1_tight_short":  {"mean_lo": 800,   "mean_hi": 3500,  "cv_lo": 0.0,  "cv_hi": 0.25},
    "P2_tight_mid":    {"mean_lo": 4500,  "mean_hi": 7500,  "cv_lo": 0.0,  "cv_hi": 0.20},
    "P3_wide_mid":     {"mean_lo": 4500,  "mean_hi": 8500,  "cv_lo": 0.20, "cv_hi": 1.0},
    "P4_tight_long":   {"mean_lo": 7500,  "mean_hi": 11500, "cv_lo": 0.0,  "cv_hi": 0.20},
    # P5_bimodal is built as 50/50 mix of P1 and P4
    "P6_saturated":    {"mean_lo": 13000, "mean_hi": 99999, "cv_lo": 0.0,  "cv_hi": 0.10},
}

PROMPTS_PER_PROFILE = 200
N_VAL = 50

# ── Source-dataset loaders that return {prompt_id: (chat_prompt_text,
#    ground_truth, style)} — keyed by the same prompt_id our 27-cell
#    rollouts used (= source dataset's row index). ──

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of "
    "your response should be of the form Answer: $Answer (without quotes) "
    "where $Answer is the answer to the problem.\n\n"
    "{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)


def load_dapo_math() -> dict[int, tuple[str, str]]:
    import pyarrow.parquet as pq
    tbl = pq.read_table("/home/user/data/dapo-math-4k/train.parquet")
    out = {}
    rows = tbl.to_pylist()
    for i, row in enumerate(rows[:200]):  # we only used first 200 in 27-cell
        prompt_field = row.get("prompt")
        if isinstance(prompt_field, list) and prompt_field:
            content = prompt_field[0].get("content", "")
        else:
            content = str(prompt_field or "")
        rm = row.get("reward_model", {}) or {}
        gt = rm.get("ground_truth") if isinstance(rm, dict) else None
        out[i] = (content, gt or "")
    return out


def load_math_500() -> dict[int, tuple[str, str]]:
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = {}
    for i, ex in enumerate(ds):
        if i >= 200: break
        out[i] = (PROMPT_TEMPLATE.format(problem=ex["problem"]), str(ex["answer"]))
    return out


def load_aime_24() -> dict[int, tuple[str, str]]:
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    from datasets import load_dataset
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    out = {}
    for i, ex in enumerate(ds):
        out[i] = (PROMPT_TEMPLATE.format(problem=ex["Problem"]), str(ex["Answer"]))
    return out


SOURCE_LOADERS = {
    "dapo-math-17k": load_dapo_math,
    "math-500": load_math_500,
    "aime-24": load_aime_24,
}


def collect_per_prompt_records():
    """Return list of dicts: {src, pid, lengths, mean, cv, p90, max}."""
    records = []
    for ds in ALL_DATASETS:
        fp = ROLLOUT_ROOT / MODEL / ds / "responses.jsonl"
        if not fp.exists():
            continue
        by_pid: dict[int, list[int]] = {}
        with fp.open() as f:
            for line in f:
                r = json.loads(line)
                by_pid.setdefault(r["prompt_id"], []).append(int(r["response_tokens"]))
        for pid, lens in by_pid.items():
            if len(lens) < 2:
                continue
            m = statistics.fmean(lens)
            if m <= 0:
                continue
            sd = statistics.pstdev(lens)
            records.append({
                "src": ds, "pid": pid,
                "lengths": sorted(lens),
                "mean": m, "cv": sd / m,
                "p90": sorted(lens)[int(len(lens) * 0.9)],
                "max": max(lens),
            })
    return records


def filter_profile(records, window):
    return [
        r for r in records
        if window["mean_lo"] <= r["mean"] < window["mean_hi"]
        and window["cv_lo"] <= r["cv"] < window["cv_hi"]
    ]


def sample_with_repeat(pool, n, seed):
    rng = random.Random(seed)
    if len(pool) >= n:
        return rng.sample(pool, n)
    out = list(pool)
    out.extend(rng.choice(pool) for _ in range(n - len(pool)))
    return out


def write_simulator_json(profile_dir: Path, profile_name: str, prompts: list[dict]):
    """Write a JSONL of length-only data for the simulator."""
    out = profile_dir / "prompts_lengths.json"
    with out.open("w") as f:
        json.dump({
            "profile": profile_name,
            "prompts": [
                {"src": p["src"], "pid": p["pid"], "lengths": p["lengths"]}
                for p in prompts
            ],
        }, f, indent=2)
    print(f"  → {out}")


def build_verl_parquet(
    profile_dir: Path, profile_name: str, prompts: list[dict],
    source_lookups: dict, n_val: int, seed: int,
):
    """Write train + val parquet (verl format) by joining each prompt
    record back to its source dataset's text + ground_truth.

    Skip prompts whose source isn't math (LCB / CodeContests) — return
    early if no math prompts available.
    """
    rows = []
    for p in prompts:
        src = p["src"]
        if src not in source_lookups:
            continue
        lookup = source_lookups[src]
        if p["pid"] not in lookup:
            continue
        text, gt = lookup[p["pid"]]
        rows.append({
            "data_source": "math",  # routes to math_dapo.compute_score
            "prompt": [{"role": "user", "content": text}],
            "ability": "MATH",
            "reward_model": {
                "ground_truth": gt,
                "style": "rule-lighteval/MATH_v2",
            },
            "extra_info": {
                "profile": profile_name,
                "src_dataset": src,
                "src_prompt_id": p["pid"],
                "expected_mean_length": p["mean"],
                "expected_cv": p["cv"],
            },
        })

    if not rows:
        print(f"  ! no math prompts → skipping verl parquet for {profile_name}")
        return

    rng = random.Random(seed)
    rng.shuffle(rows)
    val_n = min(n_val, max(1, len(rows) // 10))
    train_rows = rows[val_n:]
    val_rows = rows[:val_n]

    pq.write_table(pa.Table.from_pylist(train_rows), profile_dir / "train.parquet")
    pq.write_table(pa.Table.from_pylist(val_rows), profile_dir / "val.parquet")
    print(f"  → {profile_dir}/{{train,val}}.parquet ({len(train_rows)} train + {len(val_rows)} val)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/home/user/data/length-profiles")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loading 27-cell rollout records...")
    records = collect_per_prompt_records()
    print(f"  total {len(records)} per-prompt records across {len(ALL_DATASETS)} datasets\n")

    print("Loading source-dataset lookups (math only)...")
    source_lookups = {ds: SOURCE_LOADERS[ds]() for ds in DATASETS}
    for ds, lk in source_lookups.items():
        print(f"  {ds}: {len(lk)} prompts")
    print()

    # Build P1-P4 + P6
    profile_data = {}
    for name, w in PROFILES.items():
        pool = filter_profile(records, w)
        sampled = sample_with_repeat(pool, PROMPTS_PER_PROFILE, args.seed)
        profile_data[name] = sampled
        print(f"Profile {name}: pool={len(pool)} → sampled {len(sampled)}")

    # Build P5_bimodal: 50% from P1 + 50% from P4 (no replacement re-sampling
    # within these halves; just a fresh draw from each parent pool)
    p1_pool = filter_profile(records, PROFILES["P1_tight_short"])
    p4_pool = filter_profile(records, PROFILES["P4_tight_long"])
    half = PROMPTS_PER_PROFILE // 2
    bimodal = (sample_with_repeat(p1_pool, half, args.seed + 1) +
               sample_with_repeat(p4_pool, half, args.seed + 2))
    profile_data["P5_bimodal"] = bimodal
    print(f"Profile P5_bimodal: {len(bimodal)} prompts (50/50 P1+P4)\n")

    print("Writing profile artifacts:")
    for name, prompts in profile_data.items():
        profile_dir = out_root / name
        profile_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{name}:")
        # always write simulator JSON
        write_simulator_json(profile_dir, name, prompts)
        # write verl parquet for math-feasible profiles (P1-P5)
        if name != "P6_saturated":
            build_verl_parquet(
                profile_dir, name, prompts,
                source_lookups, n_val=N_VAL, seed=args.seed,
            )

    # Print final stats
    print("\n=== Profile aggregate stats ===")
    print(f"{'profile':<22}{'n':>6}{'mean':>9}{'p90':>9}{'CV(prompt med)':>18}")
    print('-' * 65)
    for name, prompts in profile_data.items():
        all_lens = [v for p in prompts for v in p["lengths"]]
        per_prompt_cv = [p["cv"] for p in prompts]
        all_lens.sort()
        print(f"{name:<22}{len(prompts):>6}"
              f"{statistics.fmean(all_lens):>9.0f}"
              f"{all_lens[int(len(all_lens) * 0.9)]:>9}"
              f"{statistics.median(per_prompt_cv):>18.2f}")


if __name__ == "__main__":
    main()
