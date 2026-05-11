"""Compute the long-tail metrics from each related-work paper on our
27-cell length data, then check whether their published claims
reproduce on our workload.

Per cell, output:
  RollPacker:  max/median          (claim: 25-32× on DAPO-Math-17k)
  RLHFuse:     p999/p50            (claim: >10× on LMSYS-Chat-1M)
  SortedRL-A:  fraction <= 3K tok  (claim: 80% on DeepSeek-R1-Distill-Llama-8B/4K-budget)
  SortedRL-B:  hit-limit ratio     (claim: 5% on DeepSeek-R1-Distill-Llama-8B/4K-budget)
  Seer:        last 10% time frac  (claim: ≈50% on Moonlight/Qwen2-VL-72B/Kimi-K2)
  RhymeRL:     earliest-engine idle/total (claim: 76% on DeepSeek-R1-distill-Qwen)

For metrics that depend on schedule (Seer's tail-time, RhymeRL's idle),
re-use simulate_vanilla under realistic settings.

Output: paper_metric_match.csv
"""
from __future__ import annotations
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from simulate import simulate_vanilla, tail_concentration


ROOT = Path("/home/user/profiling_rlpipe/rollout_stats")


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def compute_paper_metrics(
    lengths: list[int],
    n_engines: int = 4,
    max_running: int = 32,
    aggregate_tps: float = 4000.0,
    short_threshold: int = 3000,
    cap: int = 16384,
) -> dict:
    n = len(lengths)
    sorted_L = sorted(lengths)
    median = sorted_L[n // 2]
    p50 = percentile(lengths, 0.50)
    p99 = percentile(lengths, 0.99)
    p999 = percentile(lengths, 0.999)
    max_L = max(lengths)

    rollpacker_max_over_median = max_L / median if median > 0 else None
    rlhfuse_p999_over_p50 = p999 / p50 if p50 > 0 else None
    p99_over_p50 = p99 / p50 if p50 > 0 else None

    sortedrl_short_frac = sum(1 for x in lengths if x <= short_threshold) / n
    sortedrl_hit_cap_frac = sum(1 for x in lengths if x >= cap) / n

    # Schedule-dependent metrics: re-simulate vanilla.
    sim = simulate_vanilla(
        request_lengths=list(lengths),
        n_engines=n_engines,
        aggregate_tps_per_engine=aggregate_tps,
        max_running_per_engine=max_running,
    )
    seer_last10_time = tail_concentration(
        sim["finished_at"], sim["makespan_s"], 0.10
    )
    seer_last5_time = tail_concentration(
        sim["finished_at"], sim["makespan_s"], 0.05
    )
    rhymerl_max_engine_idle = sim["idle_fraction_per_engine_max"]

    return {
        "n_requests": n,
        "median": median,
        "p50": p50,
        "p99": p99,
        "p999": p999,
        "max": max_L,
        "rollpacker_max_over_median": rollpacker_max_over_median,
        "rlhfuse_p999_over_p50": rlhfuse_p999_over_p50,
        "p99_over_p50": p99_over_p50,
        "sortedrl_short_frac": sortedrl_short_frac,
        "sortedrl_hit_cap_frac": sortedrl_hit_cap_frac,
        "seer_last10_time_frac": seer_last10_time,
        "seer_last5_time_frac": seer_last5_time,
        "rhymerl_max_engine_idle": rhymerl_max_engine_idle,
        "vanilla_makespan_s": sim["makespan_s"],
    }


def main():
    rows = []
    for model_dir in sorted(ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        for ds_dir in sorted(model_dir.iterdir()):
            resp = ds_dir / "responses.jsonl"
            if not resp.exists():
                continue
            lengths = []
            with resp.open() as f:
                for line in f:
                    lengths.append(int(json.loads(line)["response_tokens"]))
            # Use the realistic small-batch RL operating point (matches
            # paper conditions where fan-in / sorted-RL are pitched).
            for bs in [128, 512]:
                if len(lengths) < bs:
                    continue
                m = compute_paper_metrics(
                    lengths=lengths[:bs],
                    n_engines=4, max_running=32,
                    aggregate_tps=4000.0,
                )
                m["model"] = model_dir.name
                m["dataset"] = ds_dir.name
                m["batch_size"] = bs
                rows.append(m)

    cols = [
        "model", "dataset", "batch_size", "n_requests",
        "median", "p50", "p99", "p999", "max",
        "rollpacker_max_over_median",
        "rlhfuse_p999_over_p50",
        "p99_over_p50",
        "sortedrl_short_frac",
        "sortedrl_hit_cap_frac",
        "seer_last10_time_frac",
        "seer_last5_time_frac",
        "rhymerl_max_engine_idle",
        "vanilla_makespan_s",
    ]
    out = ROOT / "paper_metric_match.csv"
    with out.open("w") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out} ({len(rows)} rows)")

    # Headline table at batch=128 only.
    print()
    hdr = (
        f"{'model':<26}{'dataset':<14}"
        f"{'max/med':>9}{'p999/50':>9}{'<3K':>7}{'hit':>7}"
        f"{'last10%t':>10}{'maxIdle':>9}"
    )
    print("Batch = 128 reqs (matches RL operating point)")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r["batch_size"] != 128:
            continue
        print(
            f"{r['model']:<26}{r['dataset']:<14}"
            f"{r['rollpacker_max_over_median']:>9.2f}"
            f"{r['rlhfuse_p999_over_p50']:>9.2f}"
            f"{r['sortedrl_short_frac'] * 100:>6.0f}%"
            f"{r['sortedrl_hit_cap_frac'] * 100:>6.0f}%"
            f"{r['seer_last10_time_frac'] * 100:>9.1f}%"
            f"{r['rhymerl_max_engine_idle'] * 100:>8.1f}%"
        )


if __name__ == "__main__":
    main()
