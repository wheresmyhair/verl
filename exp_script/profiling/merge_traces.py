#!/usr/bin/env python3
"""Merge per-rank PP traces into single Perfetto-viewable files.

Usage:
    python merge_traces.py [traces_dir] [pp_size]
    python merge_traces.py                          # defaults: ./traces, pp_size=4
    python merge_traces.py /home/user/profiling/traces 4
    python merge_traces.py --step 3                 # merge only step 3

Open the merged JSON at https://ui.perfetto.dev
"""

import argparse
import glob
import json
import os
import sys


def merge_step(traces_dir: str, step: int, pp_size: int) -> str:
    all_events = []
    found = 0
    for rank in range(pp_size):
        path = os.path.join(traces_dir, f"step{step}_rank{rank}.json")
        if os.path.exists(path):
            with open(path) as f:
                all_events.extend(json.load(f))
            found += 1

    if found == 0:
        return ""

    # Sort by pid (Phases first, then GPU 0-N, then HBM) then by timestamp
    all_events.sort(key=lambda e: (e.get("pid", ""), e.get("ts", 0)))

    merged_path = os.path.join(traces_dir, f"step{step}_merged.json")
    with open(merged_path, "w") as f:
        json.dump(all_events, f)
    return merged_path


def find_steps(traces_dir: str) -> list[int]:
    """Find all step numbers that have trace files."""
    steps = set()
    for f in glob.glob(os.path.join(traces_dir, "step*_rank0.json")):
        basename = os.path.basename(f)
        # step3_rank0.json -> 3
        step_str = basename.split("_")[0].replace("step", "")
        try:
            steps.add(int(step_str))
        except ValueError:
            pass
    return sorted(steps)


def main():
    parser = argparse.ArgumentParser(description="Merge PP traces for Perfetto")
    parser.add_argument("traces_dir", nargs="?", default="./traces",
                        help="Directory containing step*_rank*.json files")
    parser.add_argument("pp_size", nargs="?", type=int, default=4,
                        help="Number of PP ranks")
    parser.add_argument("--step", type=int, default=None,
                        help="Merge only this step (default: all)")
    args = parser.parse_args()

    if not os.path.isdir(args.traces_dir):
        print(f"Directory not found: {args.traces_dir}")
        sys.exit(1)

    if args.step is not None:
        steps = [args.step]
    else:
        steps = find_steps(args.traces_dir)

    if not steps:
        print(f"No trace files found in {args.traces_dir}")
        sys.exit(1)

    for step in steps:
        path = merge_step(args.traces_dir, step, args.pp_size)
        if path:
            print(f"  step {step}: {path}")
        else:
            print(f"  step {step}: no rank files found")

    print(f"\nOpen merged files at https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
