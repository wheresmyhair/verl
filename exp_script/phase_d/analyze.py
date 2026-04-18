"""Phase D post-run analyzer.

Aggregates step-level metrics across experiments × seeds and writes:
- profiling_phase_d/summary/timing_table.csv
- profiling_phase_d/summary/convergence.csv
- profiling_phase_d/summary/perfetto_links.md

Usage:
    python3 exp_script/phase_d/analyze.py $HOME/profiling_phase_d/

Assumes each <profiling_root>/<exp_name>/seed_<N>/train.log follows verl's
step-metric format (`training/global_step:<N>` keys in a single line per step).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


# Metrics we care about (key in log, column in CSV)
PRIMARY_METRICS = [
    "timing_s/step",
    "timing_s/gen",
    "timing_s/ref",
    "timing_s/update_actor",
    "timing_s/fused_update_actor",
    "timing_s/old_log_prob",
    "timing_s/generation_timing/max",
    "timing_s/generation_timing/min",
    "perf/throughput",
    "perf/max_memory_allocated_gb",
    "actor/grad_norm",
    "actor/pg_loss",
    "actor/kl_loss",
    "critic/score/mean",
    "response_length/mean",
    "response_length/max",
    "response_length/min",
]

WARMUP_STEPS = 2  # drop first N steps from mean/std

METRIC_RE = re.compile(r"(\w[\w/]*):([0-9.eE+\-]+)")


def parse_step_line(line: str) -> dict[str, float] | None:
    """Parse one 'step:N - metric1:val1 - metric2:val2 - ...' line."""
    if not line.strip().startswith("step:") and "step:" not in line:
        return None
    m = re.search(r"step:(\d+)", line)
    if not m:
        return None
    out: dict[str, float] = {"step": int(m.group(1))}
    for key, val in METRIC_RE.findall(line):
        if key == "step":
            continue
        try:
            out[key] = float(val)
        except ValueError:
            pass
    return out


def load_run(log_path: Path) -> list[dict[str, float]]:
    """Extract step metrics from a train.log."""
    steps: list[dict[str, float]] = []
    if not log_path.exists():
        return steps
    with log_path.open() as f:
        for line in f:
            if "training/global_step:" not in line:
                continue
            parsed = parse_step_line(line)
            if parsed is None:
                continue
            steps.append(parsed)
    return steps


def aggregate(steps: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Compute mean/std per metric across post-warmup steps."""
    if len(steps) <= WARMUP_STEPS:
        return {}
    post_warmup = steps[WARMUP_STEPS:]
    stats: dict[str, dict[str, float]] = {}
    for metric in PRIMARY_METRICS:
        vals = [s[metric] for s in post_warmup if metric in s]
        if not vals:
            continue
        stats[metric] = {
            "mean": mean(vals),
            "std": stdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    return stats


def find_perfetto_traces(exp_dir: Path) -> list[Path]:
    """Find all *.json trace files under an experiment's profiling dir."""
    return sorted(exp_dir.rglob("*.json"))


def main():
    if len(sys.argv) != 2:
        print("Usage: analyze.py <profiling_root>")
        sys.exit(1)
    root = Path(sys.argv[1])
    if not root.exists():
        print(f"Error: {root} does not exist")
        sys.exit(1)

    summary_dir = root / "summary"
    summary_dir.mkdir(exist_ok=True)

    exp_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name != "summary"])

    # --- timing_table.csv: one row per (exp, seed) ---
    timing_rows: list[dict[str, Any]] = []
    trace_links: dict[str, list[Path]] = {}

    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        seed_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
        trace_links[exp_name] = find_perfetto_traces(exp_dir)

        for seed_dir in seed_dirs:
            seed = seed_dir.name.replace("seed_", "")
            log_path = seed_dir / "train.log"
            steps = load_run(log_path)
            if not steps:
                print(f"[warn] no steps parsed from {log_path}")
                continue
            stats = aggregate(steps)
            row = {
                "exp": exp_name,
                "seed": seed,
                "n_steps_logged": len(steps),
                "n_steps_post_warmup": len(steps) - WARMUP_STEPS,
            }
            for metric, s in stats.items():
                row[f"{metric}_mean"] = s["mean"]
                row[f"{metric}_std"] = s["std"]
            timing_rows.append(row)

    if timing_rows:
        # Collect all columns across rows
        all_cols = ["exp", "seed", "n_steps_logged", "n_steps_post_warmup"]
        for row in timing_rows:
            for k in row:
                if k not in all_cols:
                    all_cols.append(k)
        out_csv = summary_dir / "timing_table.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_cols)
            w.writeheader()
            for row in timing_rows:
                w.writerow(row)
        print(f"[ok] wrote {out_csv} ({len(timing_rows)} rows)")

    # --- convergence.csv: one row per (exp, seed, step) ---
    conv_rows: list[dict[str, Any]] = []
    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        for seed_dir in sorted([d for d in exp_dir.iterdir() if d.is_dir()]):
            seed = seed_dir.name.replace("seed_", "")
            log_path = seed_dir / "train.log"
            steps = load_run(log_path)
            for s in steps:
                conv_rows.append({
                    "exp": exp_name,
                    "seed": seed,
                    **s,
                })
    if conv_rows:
        all_cols = sorted({k for row in conv_rows for k in row})
        # Put exp, seed, step first
        lead = [c for c in ("exp", "seed", "step") if c in all_cols]
        rest = [c for c in all_cols if c not in lead]
        cols = lead + rest
        out_csv = summary_dir / "convergence.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in conv_rows:
                w.writerow(row)
        print(f"[ok] wrote {out_csv} ({len(conv_rows)} rows)")

    # --- perfetto_links.md ---
    out_md = summary_dir / "perfetto_links.md"
    with out_md.open("w") as f:
        f.write("# Phase D Perfetto traces\n\n")
        f.write("Open each .json in https://ui.perfetto.dev/ to inspect.\n\n")
        for exp_name, traces in trace_links.items():
            if not traces:
                continue
            f.write(f"## {exp_name} ({len(traces)} traces)\n\n")
            for t in traces:
                rel = t.relative_to(root)
                f.write(f"- `{rel}`\n")
            f.write("\n")
    print(f"[ok] wrote {out_md}")

    # --- brief textual summary ---
    print("\n=== Phase D summary ===")
    print(f"Experiments found: {len(exp_dirs)}")
    print(f"Runs with metrics: {len(timing_rows)}")
    print(f"Steps logged total: {sum(r['n_steps_logged'] for r in timing_rows)}")

    # Print mean timing_s/step per experiment (aggregate across seeds)
    print("\nmean ± std of timing_s/step (post-warmup):")
    by_exp: dict[str, list[float]] = defaultdict(list)
    for row in timing_rows:
        key = "timing_s/step_mean"
        if key in row:
            by_exp[row["exp"]].append(row[key])
    for exp, vals in sorted(by_exp.items()):
        if vals:
            m = mean(vals)
            s = stdev(vals) if len(vals) > 1 else 0.0
            print(f"  {exp:<35s} {m:>8.2f} ± {s:>6.2f}  (n={len(vals)} seeds)")


if __name__ == "__main__":
    main()
