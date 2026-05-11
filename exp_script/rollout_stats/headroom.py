"""Theoretical optimization headroom in rollout long-tail.

For each cell + each (n_engines, max_running, batch_size) config:

  ideal_makespan   = sum(L_i) / (n_engines * aggregate_tps_per_engine)
                     i.e. the wall time of a perfectly work-conserving
                     schedule with no idle GPU. Any rollout optimization
                     (fan-in, packing, async, predictor-based scheduling,
                     ...) is bounded below by this number.

  longest_floor    = max(L_i) / aggregate_tps_per_engine
                     additional floor: with single-GPU per-request
                     bandwidth, no scheme finishes faster than the
                     longest request decoded alone. (TP-fan-in could
                     beat this if memory is fully pooled, by tp_speedup;
                     so we report both DP-floor and TP-pooled-floor.)

  vanilla_makespan = continuous-batching DP scheduler over the actual
                     length distribution.

  headroom         = 1 - ideal_makespan / vanilla_makespan
                     fraction of vanilla wall time that is "bubble"
                     vs the optimal schedule.

We sweep every (cell, batch_size) and produce a CSV.

Key insight this answers: regardless of which optimization scheme we
build, what is the most wall-clock saving the long-tail of THIS
workload can in principle yield?
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from simulate import simulate_vanilla


def compute_headroom(
    lengths: list[int],
    n_engines: int,
    max_running: int,
    aggregate_tps_per_engine: float,
    tp_speedup_pooled: float,
) -> dict:
    sum_L = sum(lengths)
    max_L = max(lengths)
    ideal_dp = sum_L / (n_engines * aggregate_tps_per_engine)
    ideal_tp = sum_L / (n_engines * aggregate_tps_per_engine)  # same aggregate
    floor_dp = max_L / aggregate_tps_per_engine
    floor_tp = max_L / (aggregate_tps_per_engine * tp_speedup_pooled)
    lower_bound_dp = max(ideal_dp, floor_dp)
    lower_bound_tp = max(ideal_tp, floor_tp)

    sim = simulate_vanilla(
        request_lengths=list(lengths),
        n_engines=n_engines,
        aggregate_tps_per_engine=aggregate_tps_per_engine,
        max_running_per_engine=max_running,
    )
    vanilla = sim["makespan_s"]
    return {
        "n_requests": len(lengths),
        "sum_tokens": sum_L,
        "max_tokens": max_L,
        "vanilla_makespan_s": vanilla,
        "vanilla_bubble_ratio": sim["bubble_ratio"],
        "vanilla_tail_10pct_frac": (
            (vanilla - sorted(sim["finished_at"])[max(0, int(len(sim["finished_at"]) * 0.9) - 1)]) / vanilla
            if sim["finished_at"] else None
        ),
        "lower_bound_dp_s": lower_bound_dp,
        "lower_bound_tp_pooled_s": lower_bound_tp,
        "headroom_dp_frac": 1.0 - lower_bound_dp / vanilla,
        "headroom_tp_pooled_frac": 1.0 - lower_bound_tp / vanilla,
        "max_over_mean": max_L / (sum_L / len(lengths)),
    }


def load_lengths(path: Path) -> list[int]:
    out = []
    with path.open() as f:
        for line in f:
            out.append(int(json.loads(line)["response_tokens"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/user/profiling_rlpipe/rollout_stats")
    ap.add_argument(
        "--batch-sizes", default="128,512,full",
        help="comma list of request batch sizes; 'full' = use entire cell"
    )
    ap.add_argument("--n-engines", type=int, default=4)
    ap.add_argument("--max-running", type=int, default=32)
    ap.add_argument("--tp-speedup-pooled", type=float, default=3.0)
    ap.add_argument("--aggregate-tps-per-engine", type=float, default=4000.0)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    out_csv = Path(args.out_csv or root / "headroom_table.csv")

    sizes = []
    for s in args.batch_sizes.split(","):
        s = s.strip()
        sizes.append(None if s == "full" else int(s))

    rows = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        for ds_dir in sorted(model_dir.iterdir()):
            resp = ds_dir / "responses.jsonl"
            if not resp.exists():
                continue
            full = load_lengths(resp)
            for bs in sizes:
                lengths = full if bs is None else full[:bs]
                if len(lengths) < 2:
                    continue
                r = compute_headroom(
                    lengths=lengths,
                    n_engines=args.n_engines,
                    max_running=args.max_running,
                    aggregate_tps_per_engine=args.aggregate_tps_per_engine,
                    tp_speedup_pooled=args.tp_speedup_pooled,
                )
                r["model"] = model_dir.name
                r["dataset"] = ds_dir.name
                r["batch_size"] = "full" if bs is None else bs
                rows.append(r)

    cols = [
        "model", "dataset", "batch_size", "n_requests", "sum_tokens", "max_tokens",
        "vanilla_makespan_s", "vanilla_bubble_ratio", "vanilla_tail_10pct_frac",
        "lower_bound_dp_s", "lower_bound_tp_pooled_s",
        "headroom_dp_frac", "headroom_tp_pooled_frac", "max_over_mean",
    ]
    with out_csv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Pretty print: dataset → headroom across batch sizes for a few key models.
    hdr = (
        f"{'model':<26}{'dataset':<14}{'batch':>7}"
        f"{'wall':>8}{'LB_dp':>8}{'LB_tp':>8}"
        f"{'hr_dp':>9}{'hr_tp':>9}{'tail10%':>9}{'max/mean':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tail = r["vanilla_tail_10pct_frac"] or 0
        print(
            f"{r['model']:<26}{r['dataset']:<14}{str(r['batch_size']):>7}"
            f"{r['vanilla_makespan_s']:>8.1f}"
            f"{r['lower_bound_dp_s']:>8.1f}"
            f"{r['lower_bound_tp_pooled_s']:>8.1f}"
            f"{r['headroom_dp_frac'] * 100:>8.1f}%"
            f"{r['headroom_tp_pooled_frac'] * 100:>8.1f}%"
            f"{tail * 100:>8.1f}%"
            f"{r['max_over_mean']:>10.2f}"
        )
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
