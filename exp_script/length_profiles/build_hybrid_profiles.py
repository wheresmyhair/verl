"""Build 6 hybrid length-profile datasets (H1-H5, H7) from REAL prompts.

These complement P1-P5 (real curated math) by sampling extreme/cherry-pick
distributions that real curated math datasets don't naturally produce —
sparse outliers, long-dominant skew, narrow-uniform — but using REAL
prompts so we can do training-validation runs (path 2).

Composition:
  H1 rare-outlier-1/16   188×P1-pool + 12×AIME-long  (~6% outliers, fan-in's design target)
  H2 mid-outlier-4/16    150×P1-pool + 50×AIME-long  (25%, fan-in counter-example)
  H3 extreme-bimodal-cap 100×MATH-500-short + 100×AIME-long  (extreme spread, cap hits)
  H4 long-dominant-25/75 50×P1-pool + 150×P4-pool  (P5 reverse ratio)
  H5 extreme-rare-1/32   194×P1-pool + 6×AIME-long  (~3%, sparser than H1)
  H7 uniform-narrow      MATH-500 mid with CV<0.10  (lower-bound case)

Outputs per profile (same format as P1-P5):
  <profile>/prompts_lengths.json   simulator input
  <profile>/train.parquet, val.parquet  verl training input
"""
from __future__ import annotations
import argparse
import json
import random
import statistics
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from build_profiles import (  # type: ignore
    PROFILES,
    SOURCE_LOADERS,
    build_verl_parquet,
    collect_per_prompt_records,
    filter_profile,
    sample_with_repeat,
    write_simulator_json,
    PROMPTS_PER_PROFILE,
    N_VAL,
)


# Outlier pool: AIME-24 prompts whose mean >= AIME_LONG_MEAN are our
# "natural long outliers" (these are competition math, mean ≈ 12K, p90
# = 16K cap, so they are the realistic source of cap-hitting outliers).
AIME_LONG_MEAN = 10000

# MATH-500 short pool for H3 bimodal short side.
MATH500_SHORT_MEAN = 3500

# H7 uniform-narrow window: mid-mean prompts with low intra-prompt CV.
# Pulls from MATH-500 + dapo-math-17k since MATH-500 alone is sparse here.
H7_MEAN_LO, H7_MEAN_HI = 3500, 8000
H7_CV_MAX = 0.15


def aime_long_pool(records):
    return [r for r in records
            if r["src"] == "aime-24" and r["mean"] >= AIME_LONG_MEAN]


def math500_short_pool(records):
    return [r for r in records
            if r["src"] == "math-500" and r["mean"] < MATH500_SHORT_MEAN]


def h7_pool(records):
    return [r for r in records
            if r["src"] in ("math-500", "dapo-math-17k")
            and H7_MEAN_LO <= r["mean"] < H7_MEAN_HI
            and r["cv"] < H7_CV_MAX]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/home/user/data/length-profiles")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loading 27-cell rollout records...")
    records = collect_per_prompt_records()
    print(f"  total {len(records)} per-prompt records\n")

    # Source pools
    p1_pool = filter_profile(records, PROFILES["P1_tight_short"])
    p4_pool = filter_profile(records, PROFILES["P4_tight_long"])
    aime_pool = aime_long_pool(records)
    m500_short = math500_short_pool(records)
    h7p = h7_pool(records)

    print(f"  pool P1 (mean<3.5K, CV<0.25): {len(p1_pool)}")
    print(f"  pool P4 (mean 7.5-11.5K, CV<0.20): {len(p4_pool)}")
    print(f"  pool AIME-long (mean>=10K): {len(aime_pool)}")
    print(f"  pool MATH-500 short (mean<3.5K): {len(m500_short)}")
    print(f"  pool H7 narrow-mid: {len(h7p)}")
    print()

    s = args.seed
    profile_data = {
        # H1 rare-outlier 1/16: 188 short + 12 long ≈ 6%
        "H1_rare_outlier_1of16": (
            sample_with_repeat(p1_pool, 188, s + 100)
            + sample_with_repeat(aime_pool, 12, s + 101)
        ),
        # H2 mid-outlier 4/16: 150 short + 50 long = 25%
        "H2_mid_outlier_4of16": (
            sample_with_repeat(p1_pool, 150, s + 102)
            + sample_with_repeat(aime_pool, 50, s + 103)
        ),
        # H3 extreme-bimodal w/ cap: 100 MATH-500 short + 100 AIME long
        "H3_extreme_bimodal_cap": (
            sample_with_repeat(m500_short, 100, s + 104)
            + sample_with_repeat(aime_pool, 100, s + 105)
        ),
        # H4 long-dominant 25/75: 50 short + 150 long (P5 reverse)
        "H4_long_dominant_25_75": (
            sample_with_repeat(p1_pool, 50, s + 106)
            + sample_with_repeat(p4_pool, 150, s + 107)
        ),
        # H5 extreme-rare 1/32: 194 short + 6 long ≈ 3%
        "H5_extreme_rare_1of32": (
            sample_with_repeat(p1_pool, 194, s + 108)
            + sample_with_repeat(aime_pool, 6, s + 109)
        ),
        # H7 uniform-narrow: lower bound (filtered MATH-500 mid w/ low CV)
        "H7_uniform_narrow": sample_with_repeat(h7p, PROMPTS_PER_PROFILE, s + 110),
    }

    print("Loading source-dataset lookups (math)...")
    source_lookups = {ds: SOURCE_LOADERS[ds]() for ds in ["dapo-math-17k", "math-500", "aime-24"]}
    for ds, lk in source_lookups.items():
        print(f"  {ds}: {len(lk)} prompts")
    print()

    print("Writing profile artifacts:")
    for name, prompts in profile_data.items():
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        print(f"\n{name}:")
        write_simulator_json(d, name, prompts)
        build_verl_parquet(d, name, prompts, source_lookups, n_val=N_VAL, seed=args.seed)

    # Aggregate stats
    print("\n=== Hybrid profile aggregate stats ===")
    print(f"{'profile':<25}{'n':>5}{'mean':>9}{'p50':>9}{'p90':>9}{'max':>9}"
          f"{'CV(prompt med)':>17}{'cap_hit':>10}")
    print('-' * 92)
    for name, prompts in profile_data.items():
        all_lens = [v for p in prompts for v in p["lengths"]]
        per_prompt_cv = [p["cv"] for p in prompts]
        all_lens.sort()
        n = len(all_lens)
        cap = sum(1 for v in all_lens if v >= 16384) / n
        print(f"{name:<25}{len(prompts):>5}"
              f"{statistics.fmean(all_lens):>9.0f}"
              f"{all_lens[n // 2]:>9}"
              f"{all_lens[int(n * 0.9)]:>9}"
              f"{all_lens[-1]:>9}"
              f"{statistics.median(per_prompt_cv):>17.2f}"
              f"{cap * 100:>9.1f}%")


if __name__ == "__main__":
    main()
