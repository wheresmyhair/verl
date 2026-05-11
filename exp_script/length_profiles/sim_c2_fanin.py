"""C2 fan-in analytical sim on top of C1 sim_largeW_z data.

Model
-----
W=32 cluster, S=32 samples per worker. C1 measured per-cluster_worker wall T_w.
At time t, N_active(t) = |{w: T_w > t}|. Fan-in triggers at the earliest t
where N_active ≤ idle_threshold τ:
  T_trigger = sort(T_w)_desc[τ-1] = (W-τ)-th smallest wall.

After trigger, all workers reshape into TP=W. Remaining work at T_trigger:
  remaining_token_time = Σ_{alive}(T_w − T_trigger)   [in DP=1 token-time units]
TP=W decodes this with throughput tp_speedup × DP=1:
  tail_wall = remaining_token_time / tp_speedup
  C2_wall   = T_trigger + tail_wall

Metrics
-------
- C2 wall reduction vs C1: (C1_max − C2_wall) / C1_max
- C2 bubble (after fan-in, all workers finish together): nominally 0
- Speedup: C1_max / C2_wall

Inputs
------
- C1 sim_summary.json per Z profile (gives 32 cluster_worker walls)
- idle_threshold ∈ {1, 2, 4, 8}
- tp_speedup ∈ {4, 8, 16, 24, 32}  — TP=W throughput multiplier vs DP=1

Output: c2_fanin_sweep.csv
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

GRID = Path("/home/user/profiling_rlpipe/length_profile_grid")
PROFILES_DIR = Path("/home/user/data/length-profiles")
W = 32
THRESHOLDS = [1, 2, 4, 8, 16, 24, 31]  # 31 = trigger at T_min (most aggressive)
TP_SPEEDUPS = [4, 8, 16, 24, 32]


def simulate_fanin(walls: list[float], tau: int, tp_speedup: float) -> dict:
    walls_sorted = sorted(walls)
    n = len(walls_sorted)
    assert tau < n, f"τ={tau} must be < W={n}"
    # T_trigger = wall when only τ workers still busy = (n-τ-1)-indexed in ascending sort
    # i.e., the τ workers with largest walls are still busy at T_trigger.
    T_trigger = walls_sorted[n - tau - 1]
    alive = walls_sorted[n - tau:]  # the τ slowest workers
    remaining_token_time = sum(w - T_trigger for w in alive)
    tail_wall = remaining_token_time / tp_speedup
    c2_wall = T_trigger + tail_wall
    c1_max = walls_sorted[-1]
    c1_min = walls_sorted[0]
    c1_bubble = (c1_max - c1_min) / c1_max if c1_max > 0 else 0
    wall_reduction = (c1_max - c2_wall) / c1_max if c1_max > 0 else 0
    speedup = c1_max / c2_wall if c2_wall > 0 else float("inf")
    return {
        "T_trigger_s": T_trigger,
        "tail_wall_s": tail_wall,
        "c2_wall_s": c2_wall,
        "c1_max_s": c1_max,
        "c1_min_s": c1_min,
        "c1_bubble": c1_bubble,
        "wall_reduction": wall_reduction,
        "speedup": speedup,
    }


def main():
    rows = []
    for d in sorted(GRID.iterdir()):
        if not d.name.startswith("Z"):
            continue
        p = d / f"sim_W{W}_S32/seed42/sim_summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        walls = s["aggregate"]["cluster_worker_walls_s"]
        spec = json.loads((PROFILES_DIR / d.name / "length_spec.json").read_text())
        sm = spec["stats"]["sigma_mu"]
        mu = spec["stats"]["mu"]
        for tau in THRESHOLDS:
            for tp in TP_SPEEDUPS:
                r = simulate_fanin(walls, tau, tp)
                r.update({
                    "profile": d.name,
                    "sigma_mu": round(sm, 4),
                    "mu": round(mu, 0),
                    "idle_threshold": tau,
                    "tp_speedup": tp,
                })
                rows.append(r)

    # Sort: profile (by σ/μ), then tau, then tp
    rows.sort(key=lambda r: (r["sigma_mu"], r["idle_threshold"], r["tp_speedup"]))

    out = GRID / "c2_fanin_sweep.csv"
    fields = ["profile", "sigma_mu", "mu", "idle_threshold", "tp_speedup",
              "c1_max_s", "c1_min_s", "c1_bubble",
              "T_trigger_s", "tail_wall_s", "c2_wall_s",
              "wall_reduction", "speedup"]
    with out.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            r_out = {k: r[k] for k in fields}
            for k in ["c1_max_s", "c1_min_s", "T_trigger_s", "tail_wall_s", "c2_wall_s"]:
                r_out[k] = round(r_out[k], 1)
            for k in ["c1_bubble", "wall_reduction"]:
                r_out[k] = round(r_out[k], 4)
            r_out["speedup"] = round(r_out["speedup"], 3)
            w.writerow(r_out)
    print(f"saved {len(rows)} rows → {out}")

    # Pretty-print summary at tp_speedup=16, tau=2 (nominal fan-in config)
    print("\n=== C2 fan-in @ idle_threshold=2, tp_speedup=16 (TP=32 @ 50% eff) ===")
    print(f"{'profile':<32}{'σ/μ':>7}{'C1_bubble':>10}{'C1_max':>9}{'C2_wall':>9}{'reduce':>9}{'speedup':>9}")
    for r in rows:
        if r["idle_threshold"] != 2 or r["tp_speedup"] != 16:
            continue
        print(f"{r['profile']:<32}{r['sigma_mu']:>7.2f}"
              f"{r['c1_bubble']*100:>9.1f}%"
              f"{r['c1_max_s']:>8.0f}s"
              f"{r['c2_wall_s']:>8.0f}s"
              f"{r['wall_reduction']*100:>8.1f}%"
              f"{r['speedup']:>8.2f}x")


if __name__ == "__main__":
    main()
