"""Stratified-proportional resize each profile's parquet to a target
batch size. Preserves the per-stratum proportions of every profile
(short vs long, math vs aime, P1-pool vs P4-pool, ...).

Target=512 (DAPO/AReaL): cycles within each stratum to fill the target.
Target=128 (RollPacker P0=128): subsamples each stratum proportionally.

Stratification key: (extra_info.src_dataset, length_bin) where
length_bin ∈ {short(<3500), mid(3500-7500), long(>7500)}.
  - For P1-P5: strata are {dapo, math-500, aime-24} × bins.
  - For H1, H2, H5: strata are {P1-pool sources, aime-24}.
  - For H3: strata are {math-500-short, aime-24-long}.
  - For H4: strata are {P1-pool sources, P4-pool sources}.
  - For H7: strata are {math-500, dapo-math-17k} narrow-mid.

Output: <profile>/train_x{TARGET}.parquet  (alongside original train.parquet)
"""
from __future__ import annotations
import argparse
import random
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROFILES = [
    "P1_tight_short", "P2_tight_mid", "P3_wide_mid",
    "P4_tight_long", "P5_bimodal",
    "H1_rare_outlier_1of16", "H2_mid_outlier_4of16",
    "H3_extreme_bimodal_cap", "H4_long_dominant_25_75",
    "H5_extreme_rare_1of32", "H7_uniform_narrow",
]

SEED = 42


def stratum_key(row):
    """Group rows by (src_dataset, expected_mean_length_bin) so that within
    a profile we preserve both src AND length-bucket proportions.
    Length bin: <3500 (short), 3500-7500 (mid), >7500 (long)."""
    extra = row.get("extra_info") or {}
    src = extra.get("src_dataset", "?")
    mean = extra.get("expected_mean_length", 0)
    if mean < 3500:
        bin_ = "short"
    elif mean < 7500:
        bin_ = "mid"
    else:
        bin_ = "long"
    return (src, bin_)


def replicate_one(in_path: Path, out_path: Path, target: int, seed: int = SEED):
    table = pq.read_table(in_path)
    rows = table.to_pylist()
    n = len(rows)

    # Group by stratum
    by_stratum: dict[tuple, list] = defaultdict(list)
    for r in rows:
        by_stratum[stratum_key(r)].append(r)

    # Stratum target counts (proportional, rounded)
    raw_targets = {k: len(v) * target / n for k, v in by_stratum.items()}
    int_targets = {k: int(round(v)) for k, v in raw_targets.items()}
    # Drop tiny strata that round to 0 in subsample mode (target < n)
    if target < n:
        for k in list(int_targets.keys()):
            if int_targets[k] == 0 and len(by_stratum[k]) > 0:
                # Keep at least 1 if the stratum is meaningful (≥ 2 rows in source)
                if len(by_stratum[k]) >= 2:
                    int_targets[k] = 1
    diff = target - sum(int_targets.values())
    # Adjust the largest stratum to absorb rounding diff
    if diff != 0:
        biggest = max(int_targets, key=lambda k: raw_targets[k])
        int_targets[biggest] += diff

    # Sample within each stratum: subsample if target ≤ pool size, cycle otherwise
    out_rows = []
    rng = random.Random(seed)
    for stratum, t in int_targets.items():
        pool = by_stratum[stratum]
        if not pool or t <= 0:
            continue
        if t <= len(pool):
            # Subsample without replacement (preserves diversity)
            picked = rng.sample(pool, t)
            out_rows.extend(picked)
        else:
            # Cycle to fill (when target > pool size)
            for i in range(t):
                out_rows.append(pool[i % len(pool)])

    # Sanity: should be exactly target after rounding adjust
    assert len(out_rows) == target, f"got {len(out_rows)} != {target} for {in_path}"

    rng.shuffle(out_rows)

    # Print composition diff
    print(f"  {in_path.parent.name}: {n} → {target}")
    for stratum, t in sorted(int_targets.items(), key=lambda kv: -kv[1]):
        pct_orig = len(by_stratum[stratum]) / n * 100
        pct_new = t / target * 100
        print(f"    {str(stratum):<35}  orig={len(by_stratum[stratum]):>3} ({pct_orig:>5.1f}%)  new={t:>3} ({pct_new:>5.1f}%)  Δ={pct_new - pct_orig:+.2f}pp")

    pq.write_table(pa.Table.from_pylist(out_rows), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/user/data/length-profiles")
    ap.add_argument("--target", type=int, default=512,
                    help="Target rows per profile (e.g. 512 DAPO, 128 RollPacker)")
    args = ap.parse_args()
    root = Path(args.root)
    print(f"Resizing {len(PROFILES)} profiles to {args.target} rows each:\n")
    for p in PROFILES:
        in_pq = root / p / "train.parquet"
        out_pq = root / p / f"train_x{args.target}.parquet"
        if not in_pq.exists():
            print(f"  ! missing {in_pq}, skipping")
            continue
        replicate_one(in_pq, out_pq, target=args.target)
        print()


if __name__ == "__main__":
    main()
