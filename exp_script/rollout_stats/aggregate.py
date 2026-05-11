"""Aggregate all cells under /home/user/profiling_rlpipe/rollout_stats/
into a single summary table (CSV + JSON) for paper figures.

Per-cell columns: model, dataset, n_prompts, n_samples,
  resp_min, resp_p10, resp_p50, resp_p90, resp_p99, resp_max, resp_mean,
  overall_cv, clip_ratio, finish_stop_count,
  per_prompt_cv_p50, per_prompt_cv_p90, per_prompt_cv_max,
  tail_p90_over_p50, tail_max_over_median,
  spearman_history_vs_rest, history_k, num_prompts_used_for_predictability,
  per_prompt_bubble_ratio.

Also writes a tall-form CSV with one row per (model, dataset, percentile)
to make plotting in seaborn/matplotlib trivial.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


COLUMNS = [
    "model", "dataset", "n_prompts", "n_samples",
    "resp_min", "resp_p10", "resp_p50", "resp_p90", "resp_p99",
    "resp_max", "resp_mean", "resp_std", "overall_cv",
    "clip_count", "clip_ratio", "finish_stop_count",
    "per_prompt_cv_p50", "per_prompt_cv_p90", "per_prompt_cv_max",
    "tail_p90_over_p50", "tail_p99_over_p50", "tail_max_over_median",
    "spearman_history_vs_rest", "history_k",
    "num_prompts_used_for_predictability",
    "per_prompt_bubble_ratio",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/user/profiling_rlpipe/rollout_stats")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-tall-csv", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    out_csv = Path(args.out_csv or root / "summary_table.csv")
    out_tall = Path(args.out_tall_csv or root / "summary_long.csv")
    out_json = Path(args.out_json or root / "summary_table.json")

    rows = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        for ds_dir in sorted(model_dir.iterdir()):
            sj = ds_dir / "summary.json"
            if not sj.exists():
                continue
            d = json.loads(sj.read_text())
            ld = d.get("length_distribution", {})
            cv = d.get("per_prompt_cv", {})
            tail = d.get("tail_severity_ratios", {})
            pred = d.get("predictability", {})
            fr = d.get("finish_reason_breakdown", {})
            n_total = ld.get("n", 0)
            clip_count = int(fr.get("length", 0))
            stop_count = int(fr.get("stop", 0))
            row = {
                "model": model_dir.name,
                "dataset": ds_dir.name,
                "n_prompts": d.get("num_prompts", 0),
                "n_samples": n_total,
                "resp_min": ld.get("min"),
                "resp_p10": ld.get("p10"),
                "resp_p50": ld.get("p50"),
                "resp_p90": ld.get("p90"),
                "resp_p99": ld.get("p99"),
                "resp_max": ld.get("max"),
                "resp_mean": ld.get("mean"),
                "resp_std": ld.get("std"),
                "overall_cv": ld.get("cv"),
                "clip_count": clip_count,
                "clip_ratio": clip_count / n_total if n_total else None,
                "finish_stop_count": stop_count,
                "per_prompt_cv_p50": cv.get("p50"),
                "per_prompt_cv_p90": cv.get("p90"),
                "per_prompt_cv_max": cv.get("max"),
                "tail_p90_over_p50": tail.get("p90/p50"),
                "tail_p99_over_p50": tail.get("p99/p50"),
                "tail_max_over_median": tail.get("max/median"),
                "spearman_history_vs_rest": pred.get(
                    "spearman_rho_hist_vs_rest_median"
                ),
                "history_k": pred.get("history_k"),
                "num_prompts_used_for_predictability": pred.get(
                    "num_prompts_used"
                ),
                "per_prompt_bubble_ratio": d.get("per_prompt_bubble_ratio"),
            }
            rows.append(row)

    with out_csv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    out_json.write_text(json.dumps(rows, indent=2))

    # Tall form: one row per (model, dataset, metric, value) for easy
    # plotting. Only the percentile-style columns.
    tall_metrics = [
        ("p10", "resp_p10"), ("p50", "resp_p50"), ("p90", "resp_p90"),
        ("p99", "resp_p99"), ("max", "resp_max"), ("mean", "resp_mean"),
    ]
    with out_tall.open("w") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "metric", "value"])
        for r in rows:
            for label, key in tall_metrics:
                v = r.get(key)
                if v is not None:
                    w.writerow([r["model"], r["dataset"], label, v])

    print(f"wrote {len(rows)} rows → {out_csv} / {out_tall} / {out_json}")


if __name__ == "__main__":
    main()
