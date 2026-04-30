"""Run the DP-worker bubble simulator on each length profile.

For each profile × (W, S, routing), report:
  earliest-GPU idle bubble (RhymeRL §3.1 definition):
    bubble = (max - min) / max  across W workers' makespans

Output: profile_sim_sweep.csv with one row per (profile, W, S, routing)
and a printed comparison table.
"""
from __future__ import annotations
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "rollout_stats"))
from dp_bubble_sweep import simulate_dp


PROFILES_ROOT = Path("/home/user/data/length-profiles")
PROFILE_NAMES = [
    "P1_tight_short", "P2_tight_mid", "P3_wide_mid",
    "P4_tight_long", "P5_bimodal", "P6_saturated",
]
W_VALUES = [4, 8, 16, 32]
S_VALUES = [4, 8, 16, 32, 64]
ROUTINGS = ["random", "verl_default", "prompt_grouped", "lpt_greedy"]
SEEDS = 3


def main():
    rows = []
    for prof in PROFILE_NAMES:
        path = PROFILES_ROOT / prof / "prompts_lengths.json"
        data = json.loads(path.read_text())
        prompts_lens = [p["lengths"] for p in data["prompts"]]
        n_per_prompt = min(len(p) for p in prompts_lens)
        print(f"\n{prof}: {len(prompts_lens)} prompts, n_per_prompt={n_per_prompt}")

        for W in W_VALUES:
            for S in S_VALUES:
                for routing in ROUTINGS:
                    bubbles = []
                    walls = []
                    for seed in range(SEEDS):
                        r = simulate_dp(
                            cell_lengths_by_prompt=prompts_lens,
                            W=W, samples_per_worker=S,
                            aggregate_tps_per_worker=4000.0,
                            max_running_per_worker=32,
                            routing=routing,
                            n_per_prompt=n_per_prompt,
                            seed=seed,
                        )
                        bubbles.append(r["dp_bubble"])
                        walls.append(r["wall_max_s"])
                    rows.append({
                        "profile": prof,
                        "W": W, "S": S, "routing": routing,
                        "dp_bubble_mean": statistics.fmean(bubbles),
                        "dp_bubble_std": statistics.pstdev(bubbles),
                        "wall_max_s_mean": statistics.fmean(walls),
                    })
        print("  done")

    out_csv = PROFILES_ROOT / "profile_sim_sweep.csv"
    with out_csv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_csv} ({len(rows)} rows)\n")

    # Print headline table: per profile, verl_default vs lpt_greedy at typical
    # RL configs (W=8, S=8) and at extreme (W=32, S=4 — RhymeRL config).
    def get(prof, W, S, routing):
        for r in rows:
            if (r['profile'], r['W'], r['S'], r['routing']) == (prof, W, S, routing):
                return r['dp_bubble_mean']
        return None

    print("=" * 88)
    print("Profile × routing bubble (RhymeRL definition: earliest-GPU idle)")
    print("=" * 88)
    print(f"{'profile':<22}{'(W,S)':<10}{'random':>9}{'verl_def':>11}"
          f"{'prompt_grp':>12}{'LPT':>9}{'verl→LPT (pp)':>17}")
    print('-' * 88)
    for prof in PROFILE_NAMES:
        for W, S in [(8, 8), (16, 8), (32, 8)]:
            line = f"{prof:<22}({W},{S}){' ' * (10 - len(f'({W},{S})'))}"
            r_v = []
            for routing in ROUTINGS:
                v = get(prof, W, S, routing)
                if v is None:
                    line += f"{'-':>9 if routing != 'verl_default' else 11}"
                    continue
                r_v.append((routing, v))
                if routing == "random":      line += f"{v * 100:>8.1f}%"
                elif routing == "verl_default": line += f"{v * 100:>10.1f}%"
                elif routing == "prompt_grouped": line += f"{v * 100:>11.1f}%"
                elif routing == "lpt_greedy":  line += f"{v * 100:>8.1f}%"
            verl_v = next((v for k, v in r_v if k == "verl_default"), None)
            lpt_v = next((v for k, v in r_v if k == "lpt_greedy"), None)
            if verl_v is not None and lpt_v is not None:
                line += f"{(verl_v - lpt_v) * 100:>16.1f}"
            print(line)
        print()


if __name__ == "__main__":
    main()
