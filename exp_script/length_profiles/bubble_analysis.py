"""Bubble model analysis:
  1. Theoretical bubble vs W curves at fixed S, varying σ/μ
  2. Sim: 11 cells × W ∈ {4,8,16,32} at BATCH=128, N=8 (= 1024 gen).

Outputs:
  bubble_vs_W.png       theoretical curves
  bubble_W_sweep.csv    sim sweep at our scale
"""
from __future__ import annotations
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROFILES_ROOT = Path("/home/user/data/length-profiles")
PROFILES = [
    "P1_tight_short", "P2_tight_mid", "P3_wide_mid", "P4_tight_long",
    "P5_bimodal",
    "H1_rare_outlier_1of16", "H2_mid_outlier_4of16",
    "H3_extreme_bimodal_cap", "H4_long_dominant_25_75",
    "H5_extreme_rare_1of32", "H7_uniform_narrow",
]

# ── Theoretical bubble curves ────────────────────────────────────────
def theoretical_bubble(W: int, S: int, sigma_mu: float) -> float:
    """bubble = 2·√(2 ln W)·(σ/μ)/√S, saturated at 1.0."""
    R = sigma_mu / math.sqrt(S)
    raw = 2 * math.sqrt(2 * math.log(W)) * R
    return min(raw, 1.0)


def plot_theoretical_curves():
    Ws = [4, 8, 16, 32, 64, 128, 256, 512]
    sigma_mu_values = [0.20, 0.50, 1.0, 1.5, 3.0]
    S = 256  # our config

    plt.figure(figsize=(8, 5))
    for sm in sigma_mu_values:
        bubbles = [theoretical_bubble(W, S, sm) * 100 for W in Ws]
        plt.plot(Ws, bubbles, marker="o", label=f"σ/μ = {sm}")
    plt.xscale("log", base=2)
    plt.xlabel("W (DP workers)")
    plt.ylabel("Bubble %")
    plt.title(f"Theoretical bubble vs W  (S={S}, formula: 2·√(2 ln W)·(σ/μ)/√S)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.axhline(76, color="red", linestyle="--", linewidth=1, alpha=0.5,
                label=None)
    plt.text(4, 78, "RhymeRL claim 76%", fontsize=9, color="red", alpha=0.7)
    plt.tight_layout()
    out = "/tmp/bubble_vs_W.png"
    plt.savefig(out, dpi=140)
    print(f"saved {out}")
    plt.close()

    # Also plot at fixed σ/μ = 0.5, varying S
    plt.figure(figsize=(8, 5))
    sm = 0.5
    for S_ in [16, 32, 64, 128, 256, 512, 1024]:
        bubbles = [theoretical_bubble(W, S_, sm) * 100 for W in Ws]
        plt.plot(Ws, bubbles, marker="o", label=f"S = {S_}")
    plt.xscale("log", base=2)
    plt.xlabel("W (DP workers)")
    plt.ylabel("Bubble %")
    plt.title(f"Theoretical bubble vs W  (σ/μ={sm}, varying S)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = "/tmp/bubble_vs_W_varyS.png"
    plt.savefig(out, dpi=140)
    print(f"saved {out}")
    plt.close()


# ── Empirical σ/μ from per-prompt length distributions ──────────────
def compute_profile_stats(profile: str) -> dict:
    path = PROFILES_ROOT / profile / "prompts_lengths.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    all_lens = [v for p in data["prompts"] for v in p["lengths"]]
    mean = statistics.fmean(all_lens)
    std = statistics.pstdev(all_lens)
    return {
        "n_total": len(all_lens),
        "mean": mean,
        "std": std,
        "sigma_mu": std / mean if mean > 0 else 0.0,
        "p50": sorted(all_lens)[len(all_lens) // 2],
        "p90": sorted(all_lens)[int(len(all_lens) * 0.9)],
        "max": max(all_lens),
    }


# ── Sim sweep at our scale: load existing CSV  ─────────────────────
def load_sim_csv():
    rows = list(csv.DictReader(open(PROFILES_ROOT / "profile_sim_sweep.csv")))
    return rows


def sim_get(rows, prof: str, W: int, S: int, routing: str = "verl_default") -> float | None:
    for r in rows:
        if r["profile"] == prof and int(r["W"]) == W and int(r["S"]) == S and r["routing"] == routing:
            return float(r["dp_bubble_mean"])
    return None


def main():
    print("=" * 80)
    print("Q1: Theoretical bubble vs W curves")
    print("=" * 80)
    plot_theoretical_curves()
    print()

    print("=" * 80)
    print("Profile stats: empirical σ/μ from prompts_lengths.json")
    print("=" * 80)
    print(f"{'profile':<26}{'n':>7}{'mean':>9}{'std':>9}{'σ/μ':>9}{'p50':>7}{'p90':>7}{'max':>7}")
    profile_stats = {}
    for p in PROFILES:
        s = compute_profile_stats(p)
        profile_stats[p] = s
        if s:
            print(f"{p:<26}{s['n_total']:>7}{s['mean']:>9.0f}{s['std']:>9.0f}"
                  f"{s['sigma_mu']:>9.2f}{s['p50']:>7}{s['p90']:>7}{s['max']:>7}")
    print()

    print("=" * 80)
    print("Q3 sim: bubble vs W at our scale (BATCH=128 × N=8 = 1024 gen)")
    print("       W ∈ {4,8,16,32} ⇒ S = 1024/W ∈ {256,128,64,32}")
    print("=" * 80)
    rows = load_sim_csv()
    out_csv = PROFILES_ROOT / "bubble_W_sweep_at_our_scale.csv"
    csv_rows = []

    print(f"{'profile':<26}{'σ/μ':>7}{'W=4':>10}{'W=8':>10}{'W=16':>10}{'W=32':>10}")
    print(f"{'':<26}{'':>7}{'(S=256)':>10}{'(S=128)':>10}{'(S=64)':>10}{'(S=32)':>10}")
    for p in PROFILES:
        sm = profile_stats.get(p, {}).get("sigma_mu", 0.0)
        b4 = sim_get(rows, p, 4, 256)
        b8 = sim_get(rows, p, 8, 128)
        b16 = sim_get(rows, p, 16, 64)
        b32 = sim_get(rows, p, 32, 32)
        f = lambda x: f"{x*100:.1f}%" if x is not None else "—"
        print(f"{p:<26}{sm:>7.2f}{f(b4):>10}{f(b8):>10}{f(b16):>10}{f(b32):>10}")
        csv_rows.append({"profile": p, "sigma_mu": sm,
                         "W4_S256": b4, "W8_S128": b8, "W16_S64": b16, "W32_S32": b32})

    # write csv
    with out_csv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}")

    # Also plot per-profile curves
    plt.figure(figsize=(10, 6))
    Ws = [4, 8, 16, 32]
    Ss = [256, 128, 64, 32]
    for r in csv_rows:
        ys = []
        for w_, s_ in zip(Ws, Ss):
            v = sim_get(rows, r["profile"], w_, s_)
            ys.append(v * 100 if v is not None else None)
        plt.plot(Ws, ys, marker="o", label=f"{r['profile']} (σ/μ={r['sigma_mu']:.2f})")

    # Overlay theoretical curve at typical σ/μ (median across profiles)
    sm_med = statistics.median([r["sigma_mu"] for r in csv_rows])
    th = [theoretical_bubble(w, s, sm_med) * 100 for w, s in zip(Ws, Ss)]
    plt.plot(Ws, th, "k--", linewidth=2,
             label=f"theory @ σ/μ={sm_med:.2f}")

    plt.xscale("log", base=2)
    plt.xlabel("W (DP workers); S = 1024/W")
    plt.ylabel("Bubble % (sim, verl_default)")
    plt.title("Sim bubble vs W at fixed total_gen=1024 (RollPacker scale)\n"
              "11 profiles + theoretical reference")
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = "/tmp/bubble_W_sweep_our_scale.png"
    plt.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    sys.exit(main())
