"""C2 oracle-informed fan-in: classify reqs upfront by max_new_tokens,
fast → DP fleet (batched per-engine), tail → TP fleet (batched).

Why oracle here: ignore_eos workloads have exact known lengths via the
length_spec.json. The "predictor" in production is response-length
histogram from prior steps; with synthetic ignore_eos, we have a
perfect predictor — so oracle-informed is the appropriate baseline.

Difference from `sim_largeW_z_c2.py` (dynamic):
  - Pre-classify per request (max_new_tokens vs profile threshold)
  - DP phase: 4 engines each batch all their fast reqs (round-robin)
  - Swap (parallel=False, sequential D2H/H2D, ~1-2s)
  - TP phase: all tails batched on TP=4
  - Avoids dynamic orchestrator's misfire on balanced workloads

Per-profile threshold (from length_spec.json type + params):
  - bimodal:       (L_s + L_l) / 2  (constant: L+1, no tails)
  - uniform:       (L_min + L_max) / 2
  - trimodal:      midpoint of top-2 modes
  - trunc_normal:  μ
  - mixed_uniform: midpoint between upper-mass min and lower-mass max
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, "/home/user/rlpipe/sglang-fork/python")

PROFILES_ROOT = "/home/user/data/length-profiles"
GRID_ROOT = "/home/user/profiling_rlpipe/length_profile_grid"
MODEL_PATH = (
    "/home/user/.cache/huggingface/hub/models--Qwen--Qwen3-8B/"
    "snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)
PROMPT_SOURCE_PARQUET = Path(PROFILES_ROOT) / "P1_tight_short" / "train_x128.parquet"

N_PROMPTS = 128
N_SAMPLES = 8
TOTAL_GEN = N_PROMPTS * N_SAMPLES
SIM_W = 32
SIM_S = TOTAL_GEN // SIM_W
N_MICRO_BATCHES = SIM_W // 4
SUB_WORKERS = 4
MB_GENS = SIM_S * SUB_WORKERS

MAX_PROMPT = 2048
MAX_RESP = 16384
GMU = 0.55


def load_prompt_source() -> list[str]:
    table = pq.read_table(PROMPT_SOURCE_PARQUET)
    rows = table.to_pylist()
    return [r["prompt"][0]["content"] for r in rows]


def load_length_spec(profile: str) -> dict:
    return json.loads((Path(PROFILES_ROOT) / profile / "length_spec.json").read_text())


def threshold_for_profile(spec: dict) -> int:
    t = spec.get("type", "bimodal")
    p = spec.get("params", {})
    if t == "bimodal":
        L_s, L_l = p["L_s"], p["L_l"]
        if L_s == L_l:
            return L_l + 1  # constant, no tails (∞ threshold not needed; +1 is enough)
        return (L_s + L_l) // 2
    elif t == "uniform":
        return (p["L_min"] + p["L_max"]) // 2
    elif t == "trimodal":
        Ls = sorted(p["Ls"])
        return (Ls[-2] + Ls[-1]) // 2
    elif t == "trunc_normal":
        return int(p["mu"])
    elif t == "mixed_uniform":
        comps = p["components"]
        sorted_comps = sorted(comps, key=lambda c: c["L_min"])
        return (sorted_comps[0]["L_max"] + sorted_comps[-1]["L_min"]) // 2
    else:
        return int(spec["stats"]["mu"])


def build_micro_batch(prompts: list[str], spec_lengths: list[dict], mb_id: int):
    idx = {(e["prompt_id"], e["sample_id"]): e["max_new_tokens"] for e in spec_lengths}
    out = []
    for s in range(N_SAMPLES):
        for o in range(4):
            for w in range(SUB_WORKERS):
                prompt_id = 4 * mb_id + w + 32 * o
                sample_id = s
                cluster_w = 4 * mb_id + w
                pos = s * 16 + o * 4 + w
                out.append({
                    "mb_pos": pos,
                    "sub_worker": w,
                    "cluster_worker": cluster_w,
                    "prompt_id": prompt_id,
                    "sample_id": sample_id,
                    "prompt": prompts[prompt_id],
                    "sp": {
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "max_new_tokens": idx[(prompt_id, sample_id)],
                        "ignore_eos": True,
                    },
                })
    out.sort(key=lambda x: x["mb_pos"])
    assert len(out) == MB_GENS
    return out


def boot_dual_fleet(seed: int):
    import sglang as sgl
    from sglang.srt.utils.rlpipe_dual_fleet import DualFleetCoordinator

    common = dict(
        model_path=MODEL_PATH,
        enable_memory_saver=True,
        enable_weights_cpu_backup=True,
        mem_fraction_static=GMU,
        random_seed=seed,
        log_level="warning",
        attention_backend="flashinfer",
        context_length=MAX_PROMPT + MAX_RESP + 32,
        disable_cuda_graph=False,
    )

    print(f"[boot] tp engine (mem_fraction={GMU})…", flush=True)
    tp_engine = sgl.Engine(
        tp_size=4, base_gpu_id=0, port=40000, nccl_port=29600, **common,
    )
    time.sleep(1)
    tp_engine.release_memory_occupation(tags=["weights", "kv_cache"])
    time.sleep(0.5)

    print(f"[boot] 4 dp engines (mem_fraction={GMU})…", flush=True)
    dp_engines = []
    for i in range(SUB_WORKERS):
        eng = sgl.Engine(
            tp_size=1, base_gpu_id=i, port=30000 + i, nccl_port=29500 + i,
            **common,
        )
        dp_engines.append(eng)
        time.sleep(0.5)

    coord = DualFleetCoordinator(
        dp_engines=dp_engines, tp_engines=[tp_engine],
        initial_active="dp", auto_pause_inactive=False,
    )
    for tag in ("weights", "kv_cache"):
        coord.assume_state(tp_engine, tag, "paused")
    time.sleep(1)

    print("[warmup]", flush=True)
    sp_warm = {"temperature": 0.0, "max_new_tokens": 16}
    for eng in dp_engines:
        eng.generate("Hello world", sp_warm)
    coord.swap_topology("tp", parallel=False)
    tp_engine.generate("Hello world", sp_warm)
    coord.swap_topology("dp", parallel=False)
    return coord, dp_engines, tp_engine


def run_micro_batch_oracle(coord, dp_engines, tp_engine, mb_input, threshold):
    """Oracle fan-in for one mb. Returns (mb_wall, phase_telemetry, response_lengths)."""
    response_lengths = [0] * len(mb_input)

    # Classify
    fast_items = [it for it in mb_input if it["sp"]["max_new_tokens"] < threshold]
    tail_items = [it for it in mb_input if it["sp"]["max_new_tokens"] >= threshold]

    t0 = time.perf_counter()
    # ---------- Phase 1: DP fast phase ----------
    # Distribute fast across DP engines round-robin (preserve sub_worker mapping
    # for response-length accounting).
    fast_chunks = [[] for _ in range(SUB_WORKERS)]
    for it in fast_items:
        fast_chunks[it["sub_worker"]].append(it)

    def run_dp_chunk(idx):
        eng = dp_engines[idx]
        items = fast_chunks[idx]
        if not items:
            return 0.0
        prompts = [it["prompt"] for it in items]
        sps = [it["sp"] for it in items]
        ts = time.perf_counter()
        outs = eng.generate(prompts, sps)
        outs = outs if isinstance(outs, list) else [outs]
        for it, o in zip(items, outs):
            meta = o.get("meta_info", {}) if isinstance(o, dict) else {}
            response_lengths[it["mb_pos"]] = meta.get("completion_tokens", 0)
        return time.perf_counter() - ts

    if any(fast_chunks):
        with ThreadPoolExecutor(max_workers=SUB_WORKERS) as pool:
            futs = [pool.submit(run_dp_chunk, k) for k in range(SUB_WORKERS)]
            dp_walls = [f.result() for f in futs]
    else:
        dp_walls = [0.0] * SUB_WORKERS
    t_dp_done = time.perf_counter() - t0

    # ---------- Phase 2: swap to TP and run tails ----------
    t_tp_phase = 0.0
    t_swap = 0.0
    if tail_items:
        ts_swap = time.perf_counter()
        coord.swap_topology("tp", parallel=False)
        t_swap = time.perf_counter() - ts_swap

        prompts = [it["prompt"] for it in tail_items]
        sps = [it["sp"] for it in tail_items]
        ts_tp = time.perf_counter()
        outs = tp_engine.generate(prompts, sps)
        outs = outs if isinstance(outs, list) else [outs]
        for it, o in zip(tail_items, outs):
            meta = o.get("meta_info", {}) if isinstance(o, dict) else {}
            response_lengths[it["mb_pos"]] = meta.get("completion_tokens", 0)
        t_tp_phase = time.perf_counter() - ts_tp

        # swap back to DP for next mb
        coord.swap_topology("dp", parallel=False)

    mb_wall = time.perf_counter() - t0
    tel = {
        "n_fast": len(fast_items), "n_tail": len(tail_items),
        "threshold": threshold,
        "dp_walls_s": dp_walls, "t_dp_phase_done_s": t_dp_done,
        "t_swap_s": t_swap, "t_tp_phase_s": t_tp_phase,
        "swap_triggered": bool(tail_items),
    }
    return mb_wall, tel, response_lengths


def run_profile_oracle(coord, dp_engines, tp_engine, prompts, profile, seed):
    out_dir = Path(GRID_ROOT) / profile / f"sim_W{SIM_W}_S{SIM_S}_C2_oracle" / f"seed{seed}"
    summary_path = out_dir / "sim_summary.json"
    if summary_path.exists():
        print(f"\n=== SKIP {profile} (C2 oracle) — sim_summary.json exists ===", flush=True)
        return None
    print(f"\n=== {profile} (C2 oracle, seed={seed}) ===", flush=True)
    spec = load_length_spec(profile)
    threshold = threshold_for_profile(spec)
    n_tail_total = sum(1 for L in spec["lengths"] if L["max_new_tokens"] >= threshold)
    print(f"  spec: μ={spec['stats']['mu']:.0f}, σ/μ={spec['stats']['sigma_mu']:.2f}, "
          f"threshold={threshold}, n_tail/total={n_tail_total}/{len(spec['lengths'])} "
          f"({100*n_tail_total/len(spec['lengths']):.1f}%)", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    micro_batches = []
    mb_walls = [0.0] * N_MICRO_BATCHES
    all_lens = []

    t_start = time.perf_counter()
    for m in range(N_MICRO_BATCHES):
        mb_input = build_micro_batch(prompts, spec["lengths"], m)
        mb_wall, tel, lens = run_micro_batch_oracle(
            coord, dp_engines, tp_engine, mb_input, threshold,
        )
        mb_walls[m] = mb_wall

        per_cluster_lens = {4 * m + w: [] for w in range(SUB_WORKERS)}
        for it, L in zip(mb_input, lens):
            per_cluster_lens[it["cluster_worker"]].append(L)

        micro_batches.append({
            "mb_id": m,
            "cluster_workers": [4 * m + w for w in range(SUB_WORKERS)],
            "mb_wall_s": mb_wall,
            "per_cluster_response_lengths": per_cluster_lens,
            "fanin_telemetry": tel,
        })
        all_lens.extend(lens)

        print(f"  mb{m}: wall={mb_wall:.1f}s "
              f"(dp={tel['t_dp_phase_done_s']:.1f}s + swap={tel['t_swap_s']:.1f}s + "
              f"tp={tel['t_tp_phase_s']:.1f}s, fast/tail={tel['n_fast']}/{tel['n_tail']})",
              flush=True)

    total_wall = time.perf_counter() - t_start

    cluster_walls = []
    for m in range(N_MICRO_BATCHES):
        cluster_walls.extend([mb_walls[m]] * SUB_WORKERS)

    mn = min(cluster_walls); mx = max(cluster_walls)
    mean = sum(cluster_walls) / len(cluster_walls)
    var = sum((x - mean) ** 2 for x in cluster_walls) / len(cluster_walls)
    std = var ** 0.5
    bubble = (mx - mn) / mx if mx > 0 else 0.0

    summary = {
        "profile": profile, "seed": seed, "synthetic": True,
        "spec": spec.get("params"), "spec_stats": spec["stats"],
        "threshold": threshold, "n_tail_per_step": n_tail_total,
        "sim_config": {
            "W": SIM_W, "S": SIM_S, "n_micro_batches": N_MICRO_BATCHES,
            "sub_workers_per_mb": SUB_WORKERS,
            "n_prompts": N_PROMPTS, "n_samples": N_SAMPLES, "total_gen": TOTAL_GEN,
            "max_prompt": MAX_PROMPT, "max_resp": MAX_RESP, "gmu": GMU,
            "ignore_eos": True, "scheme": "C2_oracle_fanin",
        },
        "micro_batches": micro_batches,
        "aggregate": {
            "cluster_worker_walls_s": cluster_walls,
            "mb_walls_s": mb_walls,
            "min_s": mn, "max_s": mx, "mean_s": mean, "std_s": std,
            "bubble_max_min_div_max": bubble,
            "total_wall_s": total_wall,
        },
        "response_lengths": all_lens,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n  ── {profile} C2 oracle aggregate ──")
    print(f"  mb walls: min={mn:.1f}s max={mx:.1f}s mean={mean:.1f}s")
    print(f"  bubble = {bubble*100:.1f}% (across 8 mb walls)")
    print(f"  total wall = {total_wall:.0f}s ({total_wall/60:.1f}m)")
    print(f"  → {summary_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="+", default=[
        "Z1_const_4K", "Z2_bimodal_3K_5K_50pct", "Z22_bimodal_9K_15K_50pct",
        "Z15_uniform_2K_6K", "Z7_bimodal_2.5K_5.5K_50pct", "Z18_truncnormal_4K_2K",
        "Z3_bimodal_2K_6K_50pct", "Z23_bimodal_5K_15K_50pct", "Z24_bimodal_500_1.5K_50pct",
        "Z16_uniform_500_8K", "Z8_bimodal_1.5K_6.5K_50pct", "Z17_trimodal_500_4K_12K",
        "Z9_bimodal_500_7.5K_50pct", "Z4_bimodal_200_8K_50pct",
        "Z10_bimodal_200_8K_30pct", "Z11_bimodal_200_16K_30pct",
        "Z19_pareto_80short_20long", "Z12_bimodal_200_16K_20pct",
        "Z20_pareto_90short_10long", "Z13_bimodal_200_16K_15pct",
        "Z21_pareto_95short_5long", "Z5_bimodal_200_16K_10pct",
        "Z14_bimodal_200_16K_8pct", "Z6_bimodal_200_16K_5pct",
    ])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    todo = [p for p in args.profiles
            if not (Path(GRID_ROOT) / p / f"sim_W{SIM_W}_S{SIM_S}_C2_oracle"
                    / f"seed{args.seed}" / "sim_summary.json").exists()]
    if not todo:
        print("All C2 oracle profiles done; nothing to run.")
        return 0
    print(f"To run: {todo}")
    print(f"To skip: {[p for p in args.profiles if p not in todo]}\n")

    print("[load prompt source]")
    prompts = load_prompt_source()

    coord, dp_engines, tp_engine = boot_dual_fleet(args.seed)
    try:
        results = []
        for prof in args.profiles:
            r = run_profile_oracle(coord, dp_engines, tp_engine, prompts, prof, args.seed)
            if r is not None:
                results.append(r)
        if results:
            print("\n========= SUMMARY (Z C2 oracle) =========")
            print(f"{'profile':<36}{'σ/μ':>7}{'thr':>8}{'tail%':>8}{'bubble':>10}{'wall(min)':>12}")
            for r in results:
                agg = r["aggregate"]
                sm = r["spec_stats"]["sigma_mu"]
                tail_pct = 100 * r["n_tail_per_step"] / TOTAL_GEN
                print(f"{r['profile']:<36}{sm:>7.2f}{r['threshold']:>8d}{tail_pct:>7.1f}%"
                      f"{agg['bubble_max_min_div_max']*100:>9.1f}%"
                      f"{agg['total_wall_s']/60:>11.1f}")
    finally:
        print("\n[shutdown]")
        try:
            coord.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
