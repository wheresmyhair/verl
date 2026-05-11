"""Synthetic length-distribution profiles using ignore_eos for exact length control.

Decouples length distribution from model behavior: each (prompt_id, sample_id)
gets a max_new_tokens drawn iid from the profile's distribution. At runtime,
sim_largeW_z.py sends each request with that exact max_new_tokens and
ignore_eos=True, so SGLang generates exactly that many tokens.

This isolates the σ/μ-bubble relationship from real-model length variance,
giving a clean controlled sweep that complements the (archived) 11 real-data
profiles.

Distribution types:
  - bimodal:        {L_s, L_l, p_long}
  - uniform:        U[L_min, L_max]
  - trimodal:       three points with weights
  - trunc_normal:   N(mu, sigma) clipped to [1, 16384]
  - mixed_uniform:  weighted mixture of uniform components

Output: <profile>/length_spec.json
"""
from __future__ import annotations
import json
import math
import random
from pathlib import Path

PROFILES_ROOT = Path("/home/user/data/length-profiles")
N_PROMPTS = 128
N_SAMPLES = 8
SEED = 42
L_CAP = 16384

# (name, type, params)
# Theoretical σ/μ in comments — actual drawn σ/μ recorded in length_spec.json stats.
PROFILES = [
    # ============================================================
    # Original Z1-Z6 (kept as-is)
    # ============================================================
    ("Z1_const_4K",                "bimodal", {"L_s": 4000, "L_l": 4000, "p_long": 0.5}),    # σ/μ = 0
    ("Z2_bimodal_3K_5K_50pct",     "bimodal", {"L_s": 3000, "L_l": 5000, "p_long": 0.5}),    # σ/μ ≈ 0.25
    ("Z3_bimodal_2K_6K_50pct",     "bimodal", {"L_s": 2000, "L_l": 6000, "p_long": 0.5}),    # σ/μ ≈ 0.50
    ("Z4_bimodal_200_8K_50pct",    "bimodal", {"L_s": 200,  "L_l": 8000, "p_long": 0.5}),    # σ/μ ≈ 0.95
    ("Z5_bimodal_200_16K_10pct",   "bimodal", {"L_s": 200,  "L_l": 16000, "p_long": 0.10}),  # σ/μ ≈ 2.66
    ("Z6_bimodal_200_16K_5pct",    "bimodal", {"L_s": 200,  "L_l": 16000, "p_long": 0.05}),  # σ/μ ≈ 3.48
    # ============================================================
    # Group A — density fill (μ=4K balanced bimodal, σ/μ ∈ {0.375, 0.625, 0.875, ~1.2})
    # ============================================================
    ("Z7_bimodal_2.5K_5.5K_50pct", "bimodal", {"L_s": 2500, "L_l": 5500, "p_long": 0.5}),    # σ/μ ≈ 0.375
    ("Z8_bimodal_1.5K_6.5K_50pct", "bimodal", {"L_s": 1500, "L_l": 6500, "p_long": 0.5}),    # σ/μ ≈ 0.625
    ("Z9_bimodal_500_7.5K_50pct",  "bimodal", {"L_s": 500,  "L_l": 7500, "p_long": 0.5}),    # σ/μ ≈ 0.875
    ("Z10_bimodal_200_8K_30pct",   "bimodal", {"L_s": 200,  "L_l": 8000, "p_long": 0.30}),   # σ/μ ≈ 1.19
    # ============================================================
    # Group B — rare-long density (L_s=200, L_l=16K, vary p) — σ/μ ∈ [1.5, 3]
    # ============================================================
    ("Z11_bimodal_200_16K_30pct",  "bimodal", {"L_s": 200, "L_l": 16000, "p_long": 0.30}),   # σ/μ ≈ 1.47
    ("Z12_bimodal_200_16K_20pct",  "bimodal", {"L_s": 200, "L_l": 16000, "p_long": 0.20}),   # σ/μ ≈ 1.88
    ("Z13_bimodal_200_16K_15pct",  "bimodal", {"L_s": 200, "L_l": 16000, "p_long": 0.15}),   # σ/μ ≈ 2.20
    ("Z14_bimodal_200_16K_8pct",   "bimodal", {"L_s": 200, "L_l": 16000, "p_long": 0.08}),   # σ/μ ≈ 2.95
    # ============================================================
    # Group C — shape variety at matched σ/μ (test "σ/μ alone determines bubble")
    # ============================================================
    ("Z15_uniform_2K_6K",          "uniform",      {"L_min": 2000, "L_max": 6000}),               # σ/μ ≈ 0.29 (vs Z2 bimodal 0.25)
    ("Z16_uniform_500_8K",         "uniform",      {"L_min": 500,  "L_max": 8000}),               # σ/μ ≈ 0.51 (vs Z3 bimodal 0.50)
    ("Z17_trimodal_500_4K_12K",    "trimodal",     {"Ls": [500, 4000, 12000], "ps": [1/3, 1/3, 1/3]}),  # σ/μ ≈ 0.875 (vs Z9)
    ("Z18_truncnormal_4K_2K",      "trunc_normal", {"mu": 4000, "sigma": 2000, "L_min": 1, "L_max": L_CAP}),  # σ/μ ≈ 0.50 (vs Z3)
    # ============================================================
    # Group D — production-realistic tails (Seer-style mixed uniforms)
    # ============================================================
    ("Z19_pareto_80short_20long",  "mixed_uniform", {"components": [
        {"L_min": 200, "L_max": 1000,  "weight": 0.80},
        {"L_min": 8000, "L_max": 16000, "weight": 0.20},
    ]}),                                                                                        # σ/μ ≈ 1.62
    ("Z20_pareto_90short_10long",  "mixed_uniform", {"components": [
        {"L_min": 200, "L_max": 1000,   "weight": 0.90},
        {"L_min": 12000, "L_max": 16000, "weight": 0.10},
    ]}),                                                                                        # σ/μ ≈ 2.07
    ("Z21_pareto_95short_5long",   "mixed_uniform", {"components": [
        {"L_min": 200, "L_max": 1000,   "weight": 0.95},
        {"L_min": 12000, "L_max": 16000, "weight": 0.05},
    ]}),                                                                                        # σ/μ ≈ 2.32
    # ============================================================
    # Group E — μ scaling at fixed σ/μ (test absolute-length system effects)
    # ============================================================
    ("Z22_bimodal_9K_15K_50pct",   "bimodal", {"L_s": 9000, "L_l": 15000, "p_long": 0.5}),     # μ=12K, σ/μ=0.25 (vs Z2 μ=4K)
    ("Z23_bimodal_5K_15K_50pct",   "bimodal", {"L_s": 5000, "L_l": 15000, "p_long": 0.5}),     # μ=10K, σ/μ=0.50 (vs Z3 μ=4K)
    ("Z24_bimodal_500_1.5K_50pct", "bimodal", {"L_s": 500,  "L_l": 1500,  "p_long": 0.5}),     # μ=1K, σ/μ=0.50 (vs Z3 μ=4K)
]


def sample_bimodal(rng: random.Random, L_s: int, L_l: int, p_long: float) -> int:
    return L_l if rng.random() < p_long else L_s


def sample_uniform(rng: random.Random, L_min: int, L_max: int) -> int:
    return rng.randint(L_min, L_max)


def sample_trimodal(rng: random.Random, Ls: list[int], ps: list[float]) -> int:
    r = rng.random()
    cum = 0.0
    for L, p in zip(Ls, ps):
        cum += p
        if r < cum:
            return int(L)
    return int(Ls[-1])


def sample_trunc_normal(rng: random.Random, mu: float, sigma: float,
                        L_min: int, L_max: int) -> int:
    # Rejection-sample to keep mean/std faithful (cheap because reject rate < 1% at our params)
    while True:
        x = rng.gauss(mu, sigma)
        if L_min <= x <= L_max:
            return max(1, int(round(x)))


def sample_mixed_uniform(rng: random.Random, components: list[dict]) -> int:
    r = rng.random()
    cum = 0.0
    for c in components:
        cum += c["weight"]
        if r < cum:
            return rng.randint(c["L_min"], c["L_max"])
    c = components[-1]
    return rng.randint(c["L_min"], c["L_max"])


SAMPLERS = {
    "bimodal":       lambda rng, p: sample_bimodal(rng, **p),
    "uniform":       lambda rng, p: sample_uniform(rng, **p),
    "trimodal":      lambda rng, p: sample_trimodal(rng, **p),
    "trunc_normal":  lambda rng, p: sample_trunc_normal(rng, **p),
    "mixed_uniform": lambda rng, p: sample_mixed_uniform(rng, **p),
}


def generate_lengths(dist_type: str, params: dict, seed: int) -> list[dict]:
    rng = random.Random(seed)
    sampler = SAMPLERS[dist_type]
    out = []
    for prompt_id in range(N_PROMPTS):
        for sample_id in range(N_SAMPLES):
            L = sampler(rng, params)
            L = max(1, min(L_CAP, int(L)))
            out.append({
                "prompt_id": prompt_id,
                "sample_id": sample_id,
                "max_new_tokens": L,
            })
    return out


def stats(lengths: list[dict]) -> dict:
    vals = [x["max_new_tokens"] for x in lengths]
    n = len(vals)
    mu = sum(vals) / n
    var = sum((x - mu) ** 2 for x in vals) / n
    sigma = math.sqrt(var)
    return {
        "mu": mu, "sigma": sigma,
        "sigma_mu": sigma / mu if mu > 0 else 0.0,
        "min": min(vals), "max": max(vals),
    }


def main():
    print(f"{'profile':<34}{'type':>15}{'μ':>9}{'σ':>9}{'σ/μ':>7}{'pred bubble W=32':>20}")
    print('-' * 95)
    rows = []
    for name, dist_type, params in PROFILES:
        out_dir = PROFILES_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        lengths = generate_lengths(dist_type, params, seed=SEED)
        s = stats(lengths)
        spec = {
            "profile": name,
            "type": dist_type,
            "params": params,
            "n_prompts": N_PROMPTS, "n_samples": N_SAMPLES,
            "seed": SEED,
            "stats": s,
            "lengths": lengths,
        }
        (out_dir / "length_spec.json").write_text(json.dumps(spec, indent=2))
        pred_bubble = 2 * math.sqrt(2 * math.log(32)) * s["sigma_mu"] / math.sqrt(32)
        pred_bubble = min(pred_bubble, 1.0)
        rows.append((name, dist_type, s["mu"], s["sigma"], s["sigma_mu"], pred_bubble))

    rows.sort(key=lambda r: r[4])
    for name, dist_type, mu, sigma, sm, pred in rows:
        print(f"{name:<34}{dist_type:>15}{mu:>9.0f}{sigma:>9.0f}{sm:>7.2f}{pred*100:>19.1f}%")


if __name__ == "__main__":
    main()
