"""Run simulate.py across all 27 cells; produce a single comparison
table: bubble ratio, tail-10%, fan-in speedup vs vanilla.
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path("/home/user/profiling_rlpipe/rollout_stats")
SIM = "/home/user/rlpipe/verl/exp_script/rollout_stats/simulate.py"


def run_sim(responses_path: Path, out_path: Path,
            n_engines: int, max_running: int, agg_tps: float,
            swap_cost: float, tp_speedup: float,
            thresholds: str, batch_size: int | None) -> dict:
    """Run sim; if batch_size set, truncate inputs to first N requests."""
    if batch_size is not None:
        truncated = out_path.parent / "responses_subset.jsonl"
        truncated.parent.mkdir(parents=True, exist_ok=True)
        with responses_path.open() as f, truncated.open("w") as g:
            for i, line in enumerate(f):
                if i >= batch_size:
                    break
                g.write(line)
        responses_path = truncated
    cmd = [
        sys.executable, SIM,
        "--responses", str(responses_path),
        "--out-json", str(out_path),
        "--n-engines", str(n_engines),
        "--max-running-per-engine", str(max_running),
        "--aggregate-tps-per-engine", str(agg_tps),
        "--swap-cost-s", str(swap_cost),
        "--tp-speedup", str(tp_speedup),
        "--fanin-thresholds", thresholds,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return json.loads(out_path.read_text())


def main():
    rows = []
    for model_dir in sorted(ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        for ds_dir in sorted(model_dir.iterdir()):
            resp = ds_dir / "responses.jsonl"
            if not resp.exists():
                continue
            sim_out = ds_dir / "sim.json"
            try:
                # Real-RL config: 128 reqs (batch=8 × n=16), 4 engines,
                # 32 max_running each (KV-budget-ish). Truncate inputs.
                d = run_sim(
                    resp, sim_out,
                    n_engines=4, max_running=32,
                    agg_tps=4000.0,
                    swap_cost=2.0, tp_speedup=3.0,
                    thresholds="2,4,8,16",
                    batch_size=None,
                )
            except subprocess.CalledProcessError as e:
                print(f"FAIL {model_dir.name}/{ds_dir.name}: {e.stderr.decode()[:200]}")
                continue

            v = d["strategies"]["vanilla"]
            best_strat, best_speedup = "vanilla", 1.0
            for k, s in d["strategies"].items():
                if k == "vanilla":
                    continue
                sp = s.get("speedup_vs_vanilla", 1.0)
                if sp > best_speedup:
                    best_speedup = sp
                    best_strat = k

            rows.append({
                "model": model_dir.name,
                "dataset": ds_dir.name,
                "vanilla_makespan_s": v["makespan_s"],
                "vanilla_bubble_ratio": v["bubble_ratio"],
                "vanilla_tail_10pct_frac": v["tail_10pct_time_fraction"],
                "vanilla_max_engine_idle": v["idle_fraction_per_engine_max"],
                "best_fanin_strategy": best_strat,
                "best_fanin_speedup": best_speedup,
            })

    # Print table.
    hdr = (
        f"{'model':<26}{'dataset':<15}"
        f"{'wall':>7}{'bubble':>9}{'tail10%':>9}{'idle_max':>10}"
        f"{'best_fanin':>14}{'speedup':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['model']:<26}{r['dataset']:<15}"
            f"{r['vanilla_makespan_s']:>7.1f}"
            f"{r['vanilla_bubble_ratio']:>9.3f}"
            f"{r['vanilla_tail_10pct_frac']:>9.3f}"
            f"{r['vanilla_max_engine_idle']:>10.3f}"
            f"{r['best_fanin_strategy']:>14}"
            f"{r['best_fanin_speedup']:>8.3f}×"
        )
    out_csv = ROOT / "sim_summary.csv"
    import csv
    with out_csv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
