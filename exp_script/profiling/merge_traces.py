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


def _extract_rank(label: str) -> int | None:
    """Parse trailing GPU rank from labels like '0 Phases GPU 3'."""
    if not isinstance(label, str) or "GPU " not in label:
        return None
    try:
        return int(label.rsplit("GPU ", 1)[1])
    except ValueError:
        return None


def _normalize_tracks(events: list[dict], pp_size: int) -> list[dict]:
    """Group phase/op/HBM rows into stable Perfetto processes with ordered threads.

    This keeps G0 at the top and G{pp_size-1} at the bottom, while collapsing
    the per-rank phase rows under one shared process for easier PP bubble analysis.
    """
    normalized = []
    for event in events:
        event = dict(event)
        pid = event.get("pid")
        rank = _extract_rank(pid)
        if rank is None:
            normalized.append(event)
            continue

        if str(pid).startswith("0 Phases GPU "):
            event["pid"] = "0 Phases"
            event["tid"] = f"GPU {rank}"
        elif str(pid).startswith("1 GPU "):
            event["pid"] = "1 GPU Ops"
            event["tid"] = f"GPU {rank}"
        elif str(pid).startswith("5 HBM GPU "):
            event["pid"] = "5 HBM"
            event["tid"] = f"GPU {rank}"
        normalized.append(event)

    metadata = []
    process_sort = {
        "-1 Trainer": -1,
        "0 Phases": 0,
        "1 GPU Ops": 1,
        "5 HBM": 2,
    }
    for pid, sort_index in process_sort.items():
        metadata.append({"ph": "M", "name": "process_name", "pid": pid, "tid": 0, "args": {"name": pid}})
        metadata.append({"ph": "M", "name": "process_sort_index", "pid": pid, "tid": 0, "args": {"sort_index": sort_index}})
        if pid == "-1 Trainer":
            metadata.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": "trainer", "args": {"name": "trainer"}})
            metadata.append({"ph": "M", "name": "thread_sort_index", "pid": pid, "tid": "trainer", "args": {"sort_index": 0}})
            continue
        for rank in range(pp_size):
            tid = f"GPU {rank}"
            metadata.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid, "args": {"name": tid}})
            metadata.append({"ph": "M", "name": "thread_sort_index", "pid": pid, "tid": tid, "args": {"sort_index": rank}})

    return metadata + normalized


def _event_sort_key(event: dict):
    process_order = {
        "-1 Trainer": -1,
        "0 Phases": 0,
        "1 GPU Ops": 1,
        "5 HBM": 2,
    }
    pid = event.get("pid", "")
    tid = event.get("tid", "")
    rank = _extract_rank(str(tid)) if isinstance(tid, str) else None
    rank = rank if rank is not None else 0
    return (
        event.get("ph") != "M",
        process_order.get(pid, 99),
        rank,
        event.get("ts", 0),
        event.get("name", ""),
    )


def merge_step(traces_dir: str, step: int, pp_size: int) -> str:
    all_events = []
    found = 0
    trainer_path = os.path.join(traces_dir, f"step{step}_trainer.json")
    if os.path.exists(trainer_path):
        with open(trainer_path) as f:
            all_events.extend(json.load(f))
        found += 1
    for rank in range(pp_size):
        path = os.path.join(traces_dir, f"step{step}_rank{rank}.json")
        if os.path.exists(path):
            with open(path) as f:
                all_events.extend(json.load(f))
            found += 1

    if found == 0:
        return ""

    all_events = _normalize_tracks(all_events, pp_size)
    all_events.sort(key=_event_sort_key)

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
