"""Analyze a merged responses.jsonl:
  - overall length distribution (min / p10 / p50 / p90 / p99 / max / mean / std)
  - finish_reason breakdown (eos vs length cap)
  - per-prompt CV across n samples: mean / p50 / p90 / p99
  - predictability probe: split each prompt's n samples into history
    (first k) vs rest; report rank correlation between the history
    median and the held-out median across prompts
  - bubble ratio estimate: assume a batch's wall time = max_i len_i and
    compute (sum of idle time once worker i finished) / (max_i × N)

Writes a JSON summary to stdout; optionally dumps a per-prompt CSV.
"""
from __future__ import annotations
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def describe(xs):
    if not xs:
        return {}
    mean = statistics.fmean(xs)
    sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return {
        "n": len(xs),
        "min": min(xs),
        "p10": percentile(xs, 0.10),
        "p50": percentile(xs, 0.50),
        "p90": percentile(xs, 0.90),
        "p99": percentile(xs, 0.99),
        "max": max(xs),
        "mean": mean,
        "std": sd,
        "cv": sd / mean if mean > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--per-prompt-csv", default=None)
    ap.add_argument("--history-k", type=int, default=4,
                    help="Number of samples used as 'history' for the "
                         "predictability probe (default: 4 of 16).")
    args = ap.parse_args()

    path = Path(args.responses)
    per_prompt = defaultdict(list)
    finish_counts = defaultdict(int)
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            per_prompt[row["prompt_id"]].append(row)
            finish_counts[row.get("finish_type") or "unknown"] += 1

    all_lens = []
    prompt_cvs = []
    prompt_stats = []
    # Predictability: history-median vs rest-median, per prompt.
    hist_medians, rest_medians = [], []
    for pid, rows in per_prompt.items():
        rows = sorted(rows, key=lambda r: r.get("sample_id", 0))
        lens = [r["response_tokens"] for r in rows]
        all_lens.extend(lens)
        d = describe(lens)
        d["prompt_id"] = pid
        prompt_stats.append(d)
        prompt_cvs.append(d["cv"])
        k = min(args.history_k, max(1, len(lens) - 1))
        hist = lens[:k]
        rest = lens[k:]
        if hist and rest:
            hist_medians.append(statistics.median(hist))
            rest_medians.append(statistics.median(rest))

    summary = {
        "responses_path": str(path),
        "num_prompts": len(per_prompt),
        "num_samples": len(all_lens),
        "samples_per_prompt": (
            len(all_lens) / len(per_prompt) if per_prompt else 0
        ),
        "length_distribution": describe(all_lens),
        "finish_reason_breakdown": dict(finish_counts),
        "per_prompt_cv": describe(prompt_cvs),
        "tail_severity_ratios": {
            "p90/p50": (
                percentile(all_lens, 0.90) / percentile(all_lens, 0.50)
                if all_lens and percentile(all_lens, 0.50) else None
            ),
            "p99/p50": (
                percentile(all_lens, 0.99) / percentile(all_lens, 0.50)
                if all_lens and percentile(all_lens, 0.50) else None
            ),
            "max/median": (
                max(all_lens) / statistics.median(all_lens)
                if all_lens else None
            ),
        },
    }

    # Spearman rank correlation between history-median and rest-median.
    if len(hist_medians) >= 3:
        def rank(xs):
            idx = sorted(range(len(xs)), key=lambda i: xs[i])
            r = [0] * len(xs)
            for rank_pos, i in enumerate(idx):
                r[i] = rank_pos
            return r
        rh = rank(hist_medians)
        rr = rank(rest_medians)
        n = len(rh)
        mean_r = (n - 1) / 2
        num = sum((rh[i] - mean_r) * (rr[i] - mean_r) for i in range(n))
        den = (sum((rh[i] - mean_r) ** 2 for i in range(n)) *
               sum((rr[i] - mean_r) ** 2 for i in range(n))) ** 0.5
        spearman = num / den if den > 0 else 0.0
        summary["predictability"] = {
            "history_k": args.history_k,
            "num_prompts_used": n,
            "spearman_rho_hist_vs_rest_median": spearman,
        }

    # Bubble ratio: for each prompt, worker idle time once finished =
    # (max_len - len_i). Treating n-samples as n concurrent workers.
    bubble_sum = 0
    bubble_denom = 0
    for rows in per_prompt.values():
        lens = [r["response_tokens"] for r in rows]
        if not lens:
            continue
        mx = max(lens)
        bubble_sum += sum(mx - v for v in lens)
        bubble_denom += mx * len(lens)
    if bubble_denom > 0:
        summary["per_prompt_bubble_ratio"] = bubble_sum / bubble_denom

    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if args.per_prompt_csv:
        with open(args.per_prompt_csv, "w") as f:
            f.write("prompt_id,n,min,p10,p50,p90,p99,max,mean,std,cv\n")
            for d in prompt_stats:
                f.write(
                    f"{d['prompt_id']},{d['n']},{d['min']},{d['p10']:.1f},"
                    f"{d['p50']:.1f},{d['p90']:.1f},{d['p99']:.1f},"
                    f"{d['max']},{d['mean']:.1f},{d['std']:.1f},{d['cv']:.3f}\n"
                )


if __name__ == "__main__":
    main()
