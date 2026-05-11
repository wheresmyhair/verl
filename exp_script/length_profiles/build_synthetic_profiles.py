"""Synthetic length-profile suite covering extreme / cherry-picked cases.

Each profile generates 200 "prompts" each with 16 synthetic length values
from a controlled distribution. No real model run required — these go
straight to the simulator. The point is to map out the BUBBLE LANDSCAPE
under regimes that real datasets don't trivially produce:

  outlier_1_of_32   — 31 prompts × 16 samples ~2K + 1 prompt × 16 samples ~16K
  outlier_1_of_16   — 15 × 2K + 1 × 16K  (idle-threshold trigger range)
  outlier_4_of_32   — 28 × 2K + 4 × 16K
  bimodal_50_50     — 100 × 1K + 100 × 16K (most extreme bimodal)
  bimodal_75_25     — 150 × 2K + 50 × 14K
  pareto_alpha_1_5  — heavy Pareto tail (lots of outliers)
  pareto_alpha_3_0  — light Pareto tail
  uniform_narrow    — every prompt + sample = 6K (no variance, headroom 0)
  high_within_cv    — each prompt's 16 samples drawn iid from N(8K, 4K),
                       saturated at 16K — extreme intra-prompt variance
  step_4K_12K       — 50% all-16-samples=4K + 50% all-16-samples=12K
  inverse_bimodal   — 75% long (12K) + 25% short (2K)
  near_cap          — most ~14K, low variance, partially clipped to 16K cap

These names map to the simulator output columns of profile_sim_sweep.csv
so we can compare with the real-data profiles P1–P6.
"""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path

CAP = 16384
N_PROMPTS = 200
N_SAMPLES_PER_PROMPT = 16


def mk_outlier(n_outliers: int, n_total: int, base_mean: int, outlier_mean: int,
               base_cv: float, outlier_cv: float, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    prompts = []
    for i in range(n_total):
        if i < n_outliers:
            mu, cv = outlier_mean, outlier_cv
        else:
            mu, cv = base_mean, base_cv
        sigma = mu * cv
        samples = [max(50, min(CAP, int(rng.gauss(mu, sigma)))) for _ in range(N_SAMPLES_PER_PROMPT)]
        prompts.append(samples)
    return prompts


def mk_bimodal(frac_short: float, short_mean: int, long_mean: int, cv: float,
               n_total: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    n_short = int(n_total * frac_short)
    return (mk_outlier(0, n_short, short_mean, 0, cv, 0, seed) +
            mk_outlier(n_total - n_short, n_total - n_short,
                      0, long_mean, 0, cv, seed + 1)[:n_total - n_short])


def mk_pareto(alpha: float, scale: int, n_total: int, seed: int,
              cv_intra: float = 0.10) -> list[list[int]]:
    rng = random.Random(seed)
    prompts = []
    for _ in range(n_total):
        # Per-prompt mean drawn from a Pareto with shape alpha
        u = rng.random()
        per_prompt_mean = scale * ((1.0 / (1.0 - u)) ** (1.0 / alpha))
        per_prompt_mean = max(500, min(CAP, per_prompt_mean))
        sigma = per_prompt_mean * cv_intra
        samples = [max(50, min(CAP, int(rng.gauss(per_prompt_mean, sigma))))
                   for _ in range(N_SAMPLES_PER_PROMPT)]
        prompts.append(samples)
    return prompts


def mk_uniform(mean: int, cv: float, n_total: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    prompts = []
    sigma = mean * cv
    for _ in range(n_total):
        samples = [max(50, min(CAP, int(rng.gauss(mean, sigma))))
                   for _ in range(N_SAMPLES_PER_PROMPT)]
        prompts.append(samples)
    return prompts


def mk_high_within_cv(per_prompt_mean: int, intra_cv: float, n_total: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    prompts = []
    sigma = per_prompt_mean * intra_cv
    for _ in range(n_total):
        samples = [max(50, min(CAP, int(rng.gauss(per_prompt_mean, sigma))))
                   for _ in range(N_SAMPLES_PER_PROMPT)]
        prompts.append(samples)
    return prompts


def mk_step(n_low: int, low_val: int, high_val: int, n_total: int) -> list[list[int]]:
    prompts = []
    for i in range(n_total):
        v = low_val if i < n_low else high_val
        prompts.append([v] * N_SAMPLES_PER_PROMPT)
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/home/user/data/length-profiles")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    builders = {
        "S1_outlier_1of32":   lambda s: mk_outlier(1, 32, 2000, 16000, 0.10, 0.05, s) * (N_PROMPTS // 32 + 1),
        "S2_outlier_1of16":   lambda s: mk_outlier(1, 16, 2000, 16000, 0.10, 0.05, s) * (N_PROMPTS // 16 + 1),
        "S3_outlier_4of32":   lambda s: mk_outlier(4, 32, 2000, 16000, 0.10, 0.05, s) * (N_PROMPTS // 32 + 1),
        "S4_bimodal_50_50":   lambda s: mk_bimodal(0.5, 1000, 16000, 0.05, N_PROMPTS, s),
        "S5_bimodal_75_25":   lambda s: mk_bimodal(0.75, 2000, 14000, 0.10, N_PROMPTS, s),
        "S6_pareto_a1_5":     lambda s: mk_pareto(1.5, 1500, N_PROMPTS, s),
        "S7_pareto_a3_0":     lambda s: mk_pareto(3.0, 3000, N_PROMPTS, s),
        "S8_uniform_narrow":  lambda s: mk_uniform(6000, 0.05, N_PROMPTS, s),
        "S9_high_within_cv":  lambda s: mk_high_within_cv(8000, 0.50, N_PROMPTS, s),
        "S10_step_4K_12K":    lambda s: mk_step(N_PROMPTS // 2, 4000, 12000, N_PROMPTS),
        "S11_inverse_bimodal":lambda s: mk_bimodal(0.25, 2000, 12000, 0.10, N_PROMPTS, s),
        "S12_near_cap":       lambda s: mk_uniform(14000, 0.10, N_PROMPTS, s),
    }

    summary = []
    for name, builder in builders.items():
        prompts_lens = builder(args.seed)[:N_PROMPTS]
        # Ensure each prompt has exactly N_SAMPLES_PER_PROMPT lengths.
        prompts_lens = [p[:N_SAMPLES_PER_PROMPT] for p in prompts_lens
                        if len(p) >= N_SAMPLES_PER_PROMPT]
        # Top up if some got dropped:
        while len(prompts_lens) < N_PROMPTS:
            prompts_lens.append(prompts_lens[len(prompts_lens) % max(1, len(prompts_lens))])

        prof_dir = out_root / name
        prof_dir.mkdir(parents=True, exist_ok=True)
        out = prof_dir / "prompts_lengths.json"
        out.write_text(json.dumps({
            "profile": name,
            "prompts": [{"src": "synthetic", "pid": i, "lengths": sorted(L)}
                        for i, L in enumerate(prompts_lens)],
        }, indent=2))
        all_lens = [v for p in prompts_lens for v in p]
        all_lens.sort()
        n = len(all_lens)
        per_prompt_means = [sum(p) / len(p) for p in prompts_lens]
        per_prompt_cvs = []
        for p in prompts_lens:
            m = sum(p) / len(p)
            if m > 0:
                var = sum((x - m) ** 2 for x in p) / len(p)
                per_prompt_cvs.append((var ** 0.5) / m)
        summary.append({
            "name": name, "n_total_samples": n,
            "mean": sum(all_lens) / n,
            "p50": all_lens[n // 2],
            "p90": all_lens[int(n * 0.9)],
            "max": max(all_lens),
            "intra_cv_med": sorted(per_prompt_cvs)[len(per_prompt_cvs) // 2] if per_prompt_cvs else 0,
            "inter_cv": (
                ((sum((m - sum(per_prompt_means) / len(per_prompt_means)) ** 2 for m in per_prompt_means)
                  / len(per_prompt_means)) ** 0.5) /
                (sum(per_prompt_means) / len(per_prompt_means))
                if per_prompt_means else 0
            ),
        })

    print(f"{'name':<22}{'n':>6}{'mean':>9}{'p50':>9}{'p90':>9}{'max':>9}{'intra_cv':>11}{'inter_cv':>11}")
    print('-' * 85)
    for s in summary:
        print(f"{s['name']:<22}{s['n_total_samples']:>6}{s['mean']:>9.0f}{s['p50']:>9}{s['p90']:>9}"
              f"{s['max']:>9}{s['intra_cv_med']:>10.2f}{s['inter_cv']:>10.2f}")

    print(f"\nwrote 12 synthetic profiles to {out_root}/S*")


if __name__ == "__main__":
    main()
