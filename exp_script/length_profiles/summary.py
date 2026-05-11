"""Scan completed cells in length_profile_grid/ and emit a progress table.

Idempotent: re-running rebuilds from on-disk state. Invoked after each
cell finishes by run_grid.sh (and can be run anytime manually).

Outputs:
  /home/user/profiling_rlpipe/length_profile_grid/summary.csv   one row per cell
  /home/user/profiling_rlpipe/length_profile_grid/SUMMARY.md    human-readable
"""
from __future__ import annotations
import csv
import datetime as dt
import json
from pathlib import Path

ROOT = Path("/home/user/profiling_rlpipe/length_profile_grid")
PROFILES = [
    "P1_tight_short", "P2_tight_mid", "P3_wide_mid",
    "P4_tight_long", "P5_bimodal",
    "H1_rare_outlier_1of16", "H2_mid_outlier_4of16",
    "H3_extreme_bimodal_cap", "H4_long_dominant_25_75",
    "H5_extreme_rare_1of32", "H7_uniform_narrow",
]
SWEEP_PROFILES = ["H1_rare_outlier_1of16", "H2_mid_outlier_4of16", "P5_bimodal"]


def parse_cell(json_path: Path) -> dict | None:
    if not json_path.exists():
        return None
    try:
        d = json.loads(json_path.read_text())
    except Exception:
        return None
    t = d.get("timing", {})
    lens = d.get("response_lengths", []) or []
    n = len(lens)
    cap = sum(1 for x in lens if x >= 16384) / n if n else 0.0
    mn = t.get("generation_timing/min", 0.0)
    mx = t.get("generation_timing/max", 0.0)
    bubble = (mx - mn) / mx if mx > 0 else 0.0
    fanin = d.get("fanin_metrics", {}) or {}
    out = {
        "n_resp": n,
        "T_gen": t.get("gen", 0.0),
        "generate_sequences": t.get("generate_sequences", 0.0),
        "min_worker": mn,
        "max_worker": mx,
        "bubble": bubble,
        "cap_hit": cap,
        "mean_len": (sum(lens) / n) if n else 0.0,
        "load_rollout": t.get("load_rollout", 0.0),
        # fanin telemetry (C2 only, empty otherwise)
        "fanin_fired": bool(fanin.get("fanin/fanin_fired", False) or fanin.get("fanin_fired", False)),
        "fanin_t_first_dp_done_s": fanin.get("fanin/t_first_dp_done_s", fanin.get("t_first_dp_done_s", 0.0)),
        "fanin_t_last_dp_done_s": fanin.get("fanin/t_last_dp_done_s", fanin.get("t_last_dp_done_s", 0.0)),
        "fanin_n_dp_completed": fanin.get("fanin/n_dp_completed_at_swap", fanin.get("n_dp_completed_at_swap", 0)),
    }
    return out


def collect():
    rows = []
    expected = []
    for prof in PROFILES:
        for cond in ("C1", "C2"):
            tag = "default"
            cell_dir = ROOT / prof / cond / "seed42"
            json_path = cell_dir / "rollout_only.json"
            expected.append((prof, cond, "default", str(json_path)))
            r = parse_cell(json_path)
            row = {"profile": prof, "cond": cond, "thr": tag, "seed": 42}
            if r is None:
                row["status"] = "pending"
            else:
                row["status"] = "done"
                row.update(r)
            rows.append(row)
    # Threshold sweep cells (C2 only; threshold ∈ {1, 3})
    for prof in SWEEP_PROFILES:
        for thr in (1, 3):
            tag = f"t{thr}"
            cell_dir = ROOT / prof / f"C2_{tag}" / "seed42"
            json_path = cell_dir / "rollout_only.json"
            r = parse_cell(json_path)
            row = {"profile": prof, "cond": "C2", "thr": tag, "seed": 42}
            if r is None:
                row["status"] = "pending"
            else:
                row["status"] = "done"
                row.update(r)
            rows.append(row)
    return rows


def emit_csv(rows, out: Path):
    fields = ["profile", "cond", "thr", "seed", "status", "n_resp",
              "T_gen", "generate_sequences", "min_worker", "max_worker",
              "bubble", "cap_hit", "mean_len", "load_rollout",
              "fanin_fired", "fanin_t_first_dp_done_s",
              "fanin_t_last_dp_done_s", "fanin_n_dp_completed"]
    with out.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def fmt_speedup_table(rows):
    """Build the C1 vs C2 speedup table at threshold=default."""
    by_key = {(r["profile"], r["cond"], r["thr"]): r for r in rows}
    lines = []
    lines.append("| profile | C1 T^gen | C2 T^gen | speedup | C1 bubble | C2 bubble | bubble Δ | C2 fired | cap_hit |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|:-:|---:|")
    for prof in PROFILES:
        c1 = by_key.get((prof, "C1", "default"), {})
        c2 = by_key.get((prof, "C2", "default"), {})

        def f_t(r):
            if r.get("status") != "done":
                return "—"
            t = r["T_gen"]
            return f"{t:.0f}s ({t/60:.1f}m)"

        def f_b(r):
            if r.get("status") != "done":
                return "—"
            return f"{r['bubble']*100:.1f}%"

        def f_cap(r):
            if r.get("status") != "done":
                return "—"
            return f"{r['cap_hit']*100:.1f}%"

        if c1.get("status") == "done" and c2.get("status") == "done":
            sp = c1["T_gen"] / c2["T_gen"] if c2["T_gen"] > 0 else 0
            speedup = f"**{sp:.2f}×**"
            db = (c1["bubble"] - c2["bubble"]) * 100
            d_bubble = f"**{db:+.1f}pp**"
        else:
            speedup = "—"
            d_bubble = "—"

        fired = "✓" if c2.get("fanin_fired") else ("✗" if c2.get("status") == "done" else "—")
        cap_str = f_cap(c1) if c1.get("status") == "done" else f_cap(c2)

        lines.append(
            f"| {prof} | {f_t(c1)} | {f_t(c2)} | {speedup} "
            f"| {f_b(c1)} | {f_b(c2)} | {d_bubble} | {fired} | {cap_str} |"
        )
    return "\n".join(lines)


def fmt_sweep_table(rows):
    by_key = {(r["profile"], r["cond"], r["thr"]): r for r in rows}
    lines = []
    lines.append("\n## Threshold sweep (C2 only, seed=42)\n")
    lines.append("Compares C2 default (idle_threshold=2) against threshold ∈ {1, 3} on H1, H2, P5.\n")
    lines.append("| profile | thr=1 | thr=2 (default) | thr=3 |")
    lines.append("|---|---:|---:|---:|")
    for prof in SWEEP_PROFILES:
        cells = []
        for tag in ("t1", "default", "t3"):
            r = by_key.get((prof, "C2", tag), {})
            if r.get("status") != "done":
                cells.append("—")
            else:
                t = r["T_gen"]
                b = r["bubble"]
                cells.append(f"{t:.0f}s ({t/60:.1f}m) bubble={b*100:.1f}%")
        lines.append(f"| {prof} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def emit_md(rows, out: Path):
    n_done = sum(1 for r in rows if r.get("status") == "done")
    n_total = len(rows)
    pct = n_done / n_total * 100 if n_total else 0
    eta_text = ""
    done_times = [r["T_gen"] for r in rows if r.get("status") == "done" and r.get("T_gen")]
    if done_times and n_done < n_total:
        avg = sum(done_times) / len(done_times)
        # Add ~3 min startup overhead per cell
        per_cell = avg + 180
        remaining = (n_total - n_done) * per_cell
        eta_text = f"  ETA: **{remaining/3600:.1f}h** (avg {avg/60:.1f}m gen + ~3m setup)"

    out.write_text(f"""# Length-profile grid progress

Last updated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Progress: {n_done}/{n_total} = {pct:.0f}%**{eta_text}

## Main grid (C1 baseline vs C2 fan-in, default threshold=2, seed=42)

Config: BATCH=128 × N=8 = 1024 generations/step (RollPacker scale,
arXiv 2509.21009), MAX_RESP=16384, GMU=0.6 matched (lowered from 0.7
after P5 bimodal verl-post-gen OOM), rollout-only via
VERL_ROLLOUT_ONLY=1. Hardware: 4×A100-80G.

{fmt_speedup_table(rows)}

{fmt_sweep_table(rows)}

## Conditions

- **C1 baseline** = `idea1_baseline.sh`, SGLang DP=4 only, no dual-fleet.
- **C2 fan-in** = `idea1_fanin.sh`, SGLang DP=4 + TP=4 dual-fleet, fan-in
  fires when `idle_threshold` workers go idle.

## Bubble definition

RhymeRL §3.1: `bubble = (max_worker_makespan - min_worker_makespan) / max`.

Source: `verl/trainer/ppo/ray_trainer.py` exposes `generation_timing/min`
and `/max` to wandb metrics; we read these from each cell's
`rollout_only.json`.

## Files

CSV: `summary.csv`  (programmatic)
This MD: `SUMMARY.md`  (human-readable)
""")


def main():
    rows = collect()
    ROOT.mkdir(parents=True, exist_ok=True)
    emit_csv(rows, ROOT / "summary.csv")
    emit_md(rows, ROOT / "SUMMARY.md")
    n_done = sum(1 for r in rows if r.get("status") == "done")
    print(f"summary updated: {n_done}/{len(rows)} done — {ROOT}/SUMMARY.md")


if __name__ == "__main__":
    main()
