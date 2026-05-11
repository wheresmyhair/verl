"""Discrete-event rollout simulator over real response-length data.

Reads a `responses.jsonl` produced by worker.py (one record per
generated sample, with `response_tokens`), simulates a synchronous
RL rollout phase under different scheduling strategies, and reports
the metrics from the related-work papers we benchmark against:

  - **Bubble Ratio** (SortedRL eq. 1): Σ(Q - rₖ)·Δtₖ / (T·Q),
    where Q is global concurrency cap, rₖ is running-request count
    at time slot k, T is makespan.
  - **Tail concentration** (Seer Sec 4.2.2): wall-clock fraction
    consumed by the last X% of requests to complete.
  - **GPU idle fraction** (RhymeRL Sec 3.1): per-engine fraction of
    rollout duration spent with no active request, plus the
    "earliest-finishing engine idle" peak.
  - **Makespan** and per-strategy speedup ratio.

Model of computation (first-order, faithful to RL synchronous
rollout):
  - All B·n requests are queued at t=0 (synchronous batch submission).
  - D engines, each with `max_running` continuous-batching slots.
  - Per-request decode rate = aggregate_throughput_per_engine /
    running_count_on_that_engine. This is the standard approximation:
    aggregate throughput is roughly constant across moderate batch
    sizes (memory-bandwidth-bound decode), so adding running requests
    proportionally slows each one.

Strategies:
  - `vanilla`: D engines, `max_running` each, FIFO queue, fixed for
    the whole rollout.
  - `fanin_threshold`: when number of in-flight requests drops below
    `fanin_threshold` (e.g. 4 = swap when only 4 stragglers remain),
    pause D engines, pay `swap_cost` seconds, swap to 1 TP-D engine
    with aggregate throughput multiplied by `tp_speedup`.
  - `fanin_oracle`: knows exact response lengths, picks the optimal
    swap time to minimize makespan (used as theoretical ceiling).

The simulator does NOT model:
  - prefill latency (assume it's small relative to long-tail decode)
  - inter-engine cross-talk
  - KV pool eviction / page failures (assume fits)

Use: this simulator produces the apples-to-apples bubble/tail/idle
numbers that line up with how SortedRL, Seer, RollPacker, RhymeRL
frame the rollout-phase long-tail problem.
"""
from __future__ import annotations
import argparse
import heapq
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ─────────────────────── Engine / scheduling model ───────────────────────


@dataclass
class EngineState:
    """Single rollout engine. Tracks concurrent active requests."""
    aggregate_tps: float            # tokens/sec total over all running
    max_running: int                # continuous-batching cap
    active: list = field(default_factory=list)  # remaining tokens per slot
    cum_idle_time: float = 0.0      # time engine spent with active=[]
    last_event_time: float = 0.0


def _advance_engine(eng: EngineState, dt: float):
    """Drain `dt` seconds across the engine; remove any slot whose
    remaining tokens drop to ≤0."""
    if not eng.active:
        eng.cum_idle_time += dt
        return []
    rate = eng.aggregate_tps / len(eng.active)
    advance = rate * dt
    finished_idx = []
    for i in range(len(eng.active)):
        eng.active[i] -= advance
    new_active = []
    for i, r in enumerate(eng.active):
        if r <= 1e-6:
            finished_idx.append(i)
        else:
            new_active.append(r)
    eng.active = new_active
    return finished_idx


def _next_finish_time(eng: EngineState) -> float:
    if not eng.active:
        return math.inf
    rate = eng.aggregate_tps / len(eng.active)
    return min(eng.active) / rate


# ─────────────────────── Strategies ───────────────────────


def simulate_vanilla(
    request_lengths: List[int],
    n_engines: int,
    aggregate_tps_per_engine: float,
    max_running_per_engine: int,
    sample_dt: float = 0.05,
) -> dict:
    """Static D engines, FIFO queue, continuous batching."""
    engines = [
        EngineState(
            aggregate_tps=aggregate_tps_per_engine,
            max_running=max_running_per_engine,
        )
        for _ in range(n_engines)
    ]
    queue: List[int] = list(request_lengths)
    finished_at: List[float] = []
    Q = n_engines * max_running_per_engine

    # Initial fill: round-robin to engines.
    rr = 0
    while queue and any(len(e.active) < e.max_running for e in engines):
        if len(engines[rr].active) < engines[rr].max_running:
            engines[rr].active.append(queue.pop(0))
        rr = (rr + 1) % n_engines
        if all(len(e.active) >= e.max_running for e in engines):
            break

    t = 0.0
    bubble_num = 0.0  # Σ(Q - rₖ)·Δtₖ accumulator
    samples = []     # (t, running_count) for SortedRL ratio

    while queue or any(e.active for e in engines):
        # Time to next request completion across all engines.
        dt_finish = min(_next_finish_time(e) for e in engines)
        dt = min(dt_finish, sample_dt) if dt_finish == math.inf else dt_finish
        if dt == math.inf:
            break

        running_total = sum(len(e.active) for e in engines)
        bubble_num += (Q - running_total) * dt
        samples.append((t, running_total))

        # Advance each engine.
        for e in engines:
            finished = _advance_engine(e, dt)
            for _ in finished:
                finished_at.append(t + dt)

        t += dt

        # Refill: round-robin assign queue items to engines with slack.
        rr = 0
        attempts = 0
        while queue and attempts < n_engines:
            if len(engines[rr].active) < engines[rr].max_running:
                engines[rr].active.append(queue.pop(0))
                attempts = 0
            else:
                attempts += 1
            rr = (rr + 1) % n_engines

    makespan = t
    bubble_ratio = bubble_num / max(makespan * Q, 1e-9)
    idle_fractions = [e.cum_idle_time / max(makespan, 1e-9) for e in engines]
    return {
        "strategy": "vanilla",
        "makespan_s": makespan,
        "bubble_ratio": bubble_ratio,
        "idle_fraction_per_engine_max": max(idle_fractions),
        "idle_fraction_per_engine_mean": statistics.fmean(idle_fractions),
        "finished_at": finished_at,
    }


def simulate_fanin_threshold(
    request_lengths: List[int],
    n_engines: int,
    aggregate_tps_per_engine: float,
    max_running_per_engine: int,
    fanin_threshold: int,
    swap_cost_s: float,
    tp_speedup: float,
    sample_dt: float = 0.05,
) -> dict:
    """Like vanilla, but at the moment running_total drops below
    fanin_threshold AND the queue is empty (only stragglers left),
    pause everything, pay swap_cost_s, then run remaining stragglers
    on a single TP-fused engine with aggregate_tps × tp_speedup."""
    # First, run as vanilla until trigger fires.
    engines = [
        EngineState(
            aggregate_tps=aggregate_tps_per_engine,
            max_running=max_running_per_engine,
        )
        for _ in range(n_engines)
    ]
    queue: List[int] = list(request_lengths)
    finished_at: List[float] = []
    Q = n_engines * max_running_per_engine

    rr = 0
    while queue and any(len(e.active) < e.max_running for e in engines):
        if len(engines[rr].active) < engines[rr].max_running:
            engines[rr].active.append(queue.pop(0))
        rr = (rr + 1) % n_engines
        if all(len(e.active) >= e.max_running for e in engines):
            break

    t = 0.0
    bubble_num = 0.0
    swap_triggered = False
    swap_t = None

    while queue or any(e.active for e in engines):
        running_total = sum(len(e.active) for e in engines)
        # Trigger fan-in: queue empty + few stragglers.
        if (not queue and not swap_triggered and
                running_total > 0 and running_total <= fanin_threshold):
            swap_triggered = True
            swap_t = t
            t += swap_cost_s
            bubble_num += (Q - running_total) * swap_cost_s
            # Collapse all stragglers onto a single fused engine.
            stragglers = []
            for e in engines:
                stragglers.extend(e.active)
                e.active = []
                e.cum_idle_time += swap_cost_s
            fused = EngineState(
                aggregate_tps=aggregate_tps_per_engine * tp_speedup,
                max_running=max_running_per_engine * n_engines,
                active=stragglers,
            )
            # For the rest of simulation we only have `fused`.
            engines = [fused]
            Q = fused.max_running

        dt_finish = min(_next_finish_time(e) for e in engines)
        dt = dt_finish
        if dt == math.inf:
            break

        running_total = sum(len(e.active) for e in engines)
        bubble_num += (Q - running_total) * dt

        for e in engines:
            finished = _advance_engine(e, dt)
            for _ in finished:
                finished_at.append(t + dt)
        t += dt

        rr = 0
        attempts = 0
        while queue and attempts < len(engines):
            if len(engines[rr].active) < engines[rr].max_running:
                engines[rr].active.append(queue.pop(0))
                attempts = 0
            else:
                attempts += 1
            rr = (rr + 1) % len(engines)

    makespan = t
    bubble_ratio = bubble_num / max(makespan * Q, 1e-9)
    return {
        "strategy": f"fanin_thr={fanin_threshold}",
        "makespan_s": makespan,
        "bubble_ratio": bubble_ratio,
        "swap_triggered": swap_triggered,
        "swap_time_s": swap_t,
        "finished_at": finished_at,
    }


# ─────────────────────── Metric computers ───────────────────────


def tail_concentration(finished_at: List[float], makespan: float, frac: float = 0.10):
    """Fraction of total time consumed by the last `frac` of requests
    to complete (Seer's metric)."""
    if not finished_at:
        return None
    sorted_t = sorted(finished_at)
    n = len(sorted_t)
    k = max(1, int(n * (1.0 - frac)))
    last_x_start = sorted_t[k - 1]  # time when "last X%" started
    last_x_duration = makespan - last_x_start
    return last_x_duration / max(makespan, 1e-9)


# ─────────────────────── CLI driver ───────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True,
                    help="responses.jsonl from worker.py")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--n-engines", type=int, default=4)
    ap.add_argument("--max-running-per-engine", type=int, default=64)
    ap.add_argument("--aggregate-tps-per-engine", type=float, default=4000.0,
                    help="aggregate decode tokens/sec/engine over the "
                         "running batch (rough first-order constant)")
    ap.add_argument("--swap-cost-s", type=float, default=2.0)
    ap.add_argument("--tp-speedup", type=float, default=2.5,
                    help="aggregate throughput multiplier when collapsing "
                         "n_engines DP into 1 TP-N (memory-bandwidth-bound "
                         "decode → ~n_engines×, but routing/launch overhead "
                         "shaves it; 2-3× is typical for n=4)")
    ap.add_argument("--fanin-thresholds", default="2,4,8,16",
                    help="comma list of straggler counts to trigger fan-in")
    args = ap.parse_args()

    rows = []
    with open(args.responses) as f:
        for line in f:
            r = json.loads(line)
            rows.append(int(r["response_tokens"]))

    out = {
        "responses": args.responses,
        "n_requests": len(rows),
        "n_engines": args.n_engines,
        "max_running_per_engine": args.max_running_per_engine,
        "aggregate_tps_per_engine": args.aggregate_tps_per_engine,
        "swap_cost_s": args.swap_cost_s,
        "tp_speedup": args.tp_speedup,
        "strategies": {},
    }

    res = simulate_vanilla(
        request_lengths=rows,
        n_engines=args.n_engines,
        aggregate_tps_per_engine=args.aggregate_tps_per_engine,
        max_running_per_engine=args.max_running_per_engine,
    )
    res["tail_10pct_time_fraction"] = tail_concentration(
        res["finished_at"], res["makespan_s"], 0.10
    )
    res["tail_5pct_time_fraction"] = tail_concentration(
        res["finished_at"], res["makespan_s"], 0.05
    )
    out["strategies"]["vanilla"] = {
        k: v for k, v in res.items() if k != "finished_at"
    }
    baseline_makespan = res["makespan_s"]

    for thr_s in args.fanin_thresholds.split(","):
        thr = int(thr_s)
        res2 = simulate_fanin_threshold(
            request_lengths=rows,
            n_engines=args.n_engines,
            aggregate_tps_per_engine=args.aggregate_tps_per_engine,
            max_running_per_engine=args.max_running_per_engine,
            fanin_threshold=thr,
            swap_cost_s=args.swap_cost_s,
            tp_speedup=args.tp_speedup,
        )
        res2["tail_10pct_time_fraction"] = tail_concentration(
            res2["finished_at"], res2["makespan_s"], 0.10
        )
        res2["speedup_vs_vanilla"] = baseline_makespan / max(res2["makespan_s"], 1e-9)
        out["strategies"][f"fanin_thr={thr}"] = {
            k: v for k, v in res2.items() if k != "finished_at"
        }

    Path(args.out_json).write_text(json.dumps(out, indent=2))
    # Print one-line summary.
    v = out["strategies"]["vanilla"]
    print(f"vanilla: makespan={v['makespan_s']:.1f}s  "
          f"bubble={v['bubble_ratio']:.3f}  "
          f"tail-10%-frac={v['tail_10pct_time_fraction']:.3f}  "
          f"max-engine-idle={v['idle_fraction_per_engine_max']:.3f}")
    for k, s in out["strategies"].items():
        if k == "vanilla":
            continue
        print(f"{k}: makespan={s['makespan_s']:.1f}s  "
              f"bubble={s['bubble_ratio']:.3f}  "
              f"speedup={s['speedup_vs_vanilla']:.3f}×  "
              f"swap@{s.get('swap_time_s')}")


if __name__ == "__main__":
    main()
