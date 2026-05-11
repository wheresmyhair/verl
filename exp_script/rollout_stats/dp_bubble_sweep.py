"""DP-worker-level bubble sweep.

Question this answers: in a synchronous RL rollout where total samples
are distributed across W DP workers, how much bubble does each worker
incur waiting for the slowest one to finish? How does this depend on
(W, samples_per_worker, length distribution, per-worker continuous-
batching cap, routing policy)?

Setup per simulation:
  - Total samples T = (W × samples_per_worker) drawn (with replacement)
    from the cell's empirical length distribution.
  - W rollout workers; each receives T/W samples assigned by `routing`.
  - Each worker runs its assigned samples through continuous batching:
    aggregate throughput = `agg_tps_per_worker`, divided across the
    currently running set up to `max_running_per_worker`.
  - Worker w finish time T_w = continuous-batching makespan of its
    queue.
  - DP synchronization: rollout phase wall = max(T_w).
  - Per-worker bubble_w = (max - T_w) / max.
  - Mean bubble = 1 - mean(T_w) / max(T_w).

Routings tested:
  - random:      shuffle then split round-robin.
  - sorted_asc:  sort all samples by length ascending, deal round-robin
                 (long ones go to "later-deal" workers — naive bad case).
  - sorted_desc: longest first, round-robin (LPT-like, balances).
  - lpt_greedy:  classic LPT — assign longest unassigned to worker with
                 smallest current load.
  - oracle:      lpt_greedy is approximation; oracle = lpt_greedy here.

Output: CSV with one row per (cell, W, samples_per_worker, routing).
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import statistics
from pathlib import Path


def _continuous_batching_makespan(
    request_lengths: list[int],
    aggregate_tps: float,
    max_running: int,
) -> float:
    """Makespan of one worker's queue under continuous batching."""
    queue = list(request_lengths)
    active: list[float] = []
    while queue and len(active) < max_running:
        active.append(float(queue.pop(0)))
    t = 0.0
    while active:
        # Per-active rate = aggregate_tps / len(active).
        rate = aggregate_tps / len(active)
        # Time until the shortest active request completes.
        dt = min(active) / rate
        t += dt
        # Advance all by `rate * dt = min(active)` tokens.
        advance = rate * dt
        active = [x - advance for x in active if x - advance > 1e-6]
        # Refill from queue.
        while queue and len(active) < max_running:
            active.append(float(queue.pop(0)))
    return t


def _route(
    samples: list[int], W: int, routing: str,
    seed: int = 0, n_per_prompt: int = 1,
) -> list[list[int]]:
    """Distribute samples to W workers. Returns list of W lists.

    `samples` is assumed to be in the order produced by verl's
    `gen_batch.repeat(n, interleave=True)`: indices [0..n-1] are the n
    samples of prompt 0, [n..2n-1] are the n samples of prompt 1, etc.
    `n_per_prompt` tells the router this layout.
    """
    rng = random.Random(seed)
    if routing == "random":
        # Sample-level shuffle + round-robin (sanity-check baseline,
        # NOT what verl defaults to).
        idx = list(range(len(samples)))
        rng.shuffle(idx)
        bins = [[] for _ in range(W)]
        for i, j in enumerate(idx):
            bins[i % W].append(samples[j])
        return bins
    if routing == "verl_default":
        # verl's default: round_robin on the interleaved batch.
        # worker w gets indices [w, w+W, w+2W, ...] from the
        # interleaved layout. Same-prompt samples (consecutive in the
        # batch) get spread across workers in a stride pattern.
        bins = [[] for _ in range(W)]
        for i, v in enumerate(samples):
            bins[i % W].append(v)
        return bins
    if routing == "prompt_grouped":
        # Worst-case for imbalance: all n samples of one prompt go to
        # the SAME worker. Round-robin assignment of prompts to workers.
        # If a prompt is hard, that worker gets all its long samples.
        n_prompts = len(samples) // n_per_prompt
        bins = [[] for _ in range(W)]
        for p in range(n_prompts):
            w = p % W
            for s in range(n_per_prompt):
                bins[w].append(samples[p * n_per_prompt + s])
        return bins
    if routing == "sorted_asc":
        ordered = sorted(samples)
        bins = [[] for _ in range(W)]
        for i, v in enumerate(ordered):
            bins[i % W].append(v)
        return bins
    if routing == "sorted_desc":
        ordered = sorted(samples, reverse=True)
        bins = [[] for _ in range(W)]
        for i, v in enumerate(ordered):
            bins[i % W].append(v)
        return bins
    if routing == "lpt_greedy":
        bins = [[] for _ in range(W)]
        loads = [0] * W
        for v in sorted(samples, reverse=True):
            i = loads.index(min(loads))
            bins[i].append(v)
            loads[i] += v
        return bins
    raise ValueError(f"Unknown routing: {routing}")


def simulate_dp(
    cell_lengths_by_prompt: list[list[int]],
    W: int,
    samples_per_worker: int,
    aggregate_tps_per_worker: float,
    max_running_per_worker: int,
    routing: str,
    n_per_prompt: int = 16,
    seed: int = 0,
) -> dict:
    """One simulation. Build a batch of B prompts × n samples each
    (matching verl's `gen_batch.repeat(n, interleave=True)` layout),
    route to W workers, run each worker's queue, report per-worker
    times + bubble.

    `cell_lengths_by_prompt`: list of per-prompt lists of n length
    values (n=16 in our data). We sample B = ceil(W·S / n) prompts.
    """
    rng = random.Random(seed)
    T = W * samples_per_worker
    B = (T + n_per_prompt - 1) // n_per_prompt
    # Sample B prompts with replacement to get exactly T samples.
    sampled_prompts = [rng.choice(cell_lengths_by_prompt) for _ in range(B)]
    flat = []
    for prompt_samples in sampled_prompts:
        flat.extend(prompt_samples[:n_per_prompt])
    flat = flat[:T]
    bins = _route(flat, W, routing, seed=seed, n_per_prompt=n_per_prompt)
    worker_times = [
        _continuous_batching_makespan(
            request_lengths=b,
            aggregate_tps=aggregate_tps_per_worker,
            max_running=max_running_per_worker,
        )
        for b in bins
    ]
    t_max = max(worker_times)
    t_mean = statistics.fmean(worker_times)
    t_min = min(worker_times)
    # RhymeRL §3.1 definition: "earliest-finishing GPU remains idle for
    # ~76% of total rollout duration". Idle duration of the earliest
    # worker = t_max - t_min; total rollout = t_max; so the bubble is
    # (t_max - t_min) / t_max. This is the primary metric.
    earliest_idle_frac = (t_max - t_min) / t_max if t_max > 0 else 0.0
    # Auxiliary: mean-based bubble (averaged idle of all workers).
    mean_idle_frac = 1.0 - (t_mean / t_max) if t_max > 0 else 0.0
    return {
        "W": W,
        "samples_per_worker": samples_per_worker,
        "total_samples": T,
        "routing": routing,
        "wall_max_s": t_max,
        "wall_mean_s": t_mean,
        "wall_min_s": t_min,
        # primary metric (RhymeRL definition)
        "dp_bubble": earliest_idle_frac,
        # auxiliary
        "mean_idle_bubble": mean_idle_frac,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/user/profiling_rlpipe/rollout_stats")
    ap.add_argument(
        "--workers", default="1,2,4,8,16,32,64",
        help="comma list of DP worker counts to sweep"
    )
    ap.add_argument(
        "--samples-per-worker", default="1,2,4,8,16,32,64,128",
        help="comma list"
    )
    ap.add_argument(
        "--routings", default="random,lpt_greedy",
        help="comma list of routing policies"
    )
    ap.add_argument("--max-running-per-worker", type=int, default=32)
    ap.add_argument("--agg-tps-per-worker", type=float, default=4000.0)
    ap.add_argument("--seeds", type=int, default=3,
                    help="number of seeds per config (averaged)")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    out_csv = Path(args.out_csv or root / "dp_bubble_sweep.csv")

    Ws = [int(x) for x in args.workers.split(",")]
    Ss = [int(x) for x in args.samples_per_worker.split(",")]
    routings = args.routings.split(",")

    cells = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        for ds_dir in sorted(model_dir.iterdir()):
            resp = ds_dir / "responses.jsonl"
            if not resp.exists():
                continue
            # Group by prompt_id so we can sample whole prompts (verl's
            # gen_batch.repeat(n, interleave=True) layout).
            by_prompt: dict[int, list[int]] = {}
            with resp.open() as f:
                for line in f:
                    rec = json.loads(line)
                    by_prompt.setdefault(rec["prompt_id"], []).append(
                        int(rec["response_tokens"])
                    )
            prompts = [v for _, v in sorted(by_prompt.items())]
            cells.append((model_dir.name, ds_dir.name, prompts))

    rows = []
    for model, dataset, prompts in cells:
        # Detect n samples per prompt from the data (typically 16; could
        # be smaller for cells with truncated / partial runs).
        n_per_prompt = max(1, min(len(p) for p in prompts))
        for W in Ws:
            for S in Ss:
                for routing in routings:
                    bubbles = []
                    walls_max = []
                    for s in range(args.seeds):
                        r = simulate_dp(
                            cell_lengths_by_prompt=prompts, W=W, samples_per_worker=S,
                            aggregate_tps_per_worker=args.agg_tps_per_worker,
                            max_running_per_worker=args.max_running_per_worker,
                            routing=routing, n_per_prompt=n_per_prompt, seed=s,
                        )
                        bubbles.append(r["dp_bubble"])
                        walls_max.append(r["wall_max_s"])
                    rows.append({
                        "model": model,
                        "dataset": dataset,
                        "W": W,
                        "samples_per_worker": S,
                        "total_samples": W * S,
                        "routing": routing,
                        "dp_bubble_mean": statistics.fmean(bubbles),
                        "dp_bubble_std": statistics.pstdev(bubbles) if len(bubbles) > 1 else 0.0,
                        "wall_max_mean_s": statistics.fmean(walls_max),
                    })

    cols = list(rows[0].keys())
    with out_csv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_csv} ({len(rows)} rows)")

    # Print summary heatmap for one representative cell.
    rep = ("Qwen3-8B", "dapo-math-17k")
    print(f"\nDP bubble heatmap for {rep[0]} / {rep[1]} (random routing):")
    hdr = f"{'W \\ S':<8}" + "".join(f"{S:>8}" for S in Ss)
    print(hdr)
    print("-" * len(hdr))
    for W in Ws:
        line = f"W={W:<6}"
        for S in Ss:
            for r in rows:
                if (r["model"], r["dataset"], r["W"], r["samples_per_worker"],
                    r["routing"]) == (rep[0], rep[1], W, S, "random"):
                    line += f"{r['dp_bubble_mean'] * 100:>7.1f}%"
                    break
        print(line)


if __name__ == "__main__":
    main()
