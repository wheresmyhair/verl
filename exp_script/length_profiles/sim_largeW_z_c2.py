"""C2 fan-in measurement: same setup as sim_largeW_z but each micro-batch
runs through DynamicFanInOrchestrator (DP=4 → TP=4 swap on idle_threshold).

Mirrors sim_largeW_z.py:
  - 24 Z profiles, ignore_eos with per-(prompt, sample) max_new_tokens
  - 8 sequential micro-batches × 4 sub-workers (W=4 simulating W=32)
  - cluster_worker (4m+w) handles 4 prompts × 8 samples
  - Output: <profile>/sim_W32_S32_C2_thr{τ}/seed42/sim_summary.json

Difference from C1:
  - 4 DP engines + 1 TP engine (dual-fleet via sglang fork)
  - Each mb's 128 reqs go through orch.rollout() instead of pure ThreadPoolExecutor
  - When idle_threshold DP engines complete, swap to TP and re-prefill stragglers
  - All 4 sub_workers in an mb finish at the same time (orchestrator waits for all)
    ⇒ cluster_worker wall = mb's total_wall_s for each w in that mb's 4 cluster_workers
  - C2 bubble = (max_mb_wall − min_mb_wall) / max_mb_wall (across 8 micro-batches)

This measures W=4 mb-level fan-in. The W=32 cluster-level fan-in story
(all 32 cluster_workers cooperating in TP=32) is not directly reproducible
on 4 GPUs and would require analytical extrapolation.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
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
TOTAL_GEN = N_PROMPTS * N_SAMPLES  # 1024
SIM_W = 32
SIM_S = TOTAL_GEN // SIM_W  # 32
N_MICRO_BATCHES = SIM_W // 4  # 8
SUB_WORKERS = 4
MB_GENS = SIM_S * SUB_WORKERS  # 128

MAX_PROMPT = 2048
MAX_RESP = 16384
GMU = 0.55  # slightly less than C1 0.6, dual-fleet has CUDA-context overhead


def load_prompt_source() -> list[str]:
    table = pq.read_table(PROMPT_SOURCE_PARQUET)
    rows = table.to_pylist()
    if len(rows) != N_PROMPTS:
        raise ValueError(f"prompt source has {len(rows)} rows, expected {N_PROMPTS}")
    return [r["prompt"][0]["content"] for r in rows]


def load_length_spec(profile: str) -> dict:
    return json.loads((Path(PROFILES_ROOT) / profile / "length_spec.json").read_text())


def build_micro_batch(prompts: list[str], spec_lengths: list[dict], mb_id: int):
    """Returns list of 128 dicts in mb_pos order. position p ⇒ sub_worker p%4."""
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
    """Boot 1 TP engine first (releases on boot), then 4 DP engines.
    Returns (coord, dp_engines, tp_engine)."""
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
    # TP engine was released above — sync state.
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


def run_micro_batch_c2(orch, mb_input: list[dict]):
    """Run mb via DynamicFanInOrchestrator. Returns (mb_wall, telemetry, response_lengths_by_pos)."""
    from sglang.srt.utils.rlpipe_fan_in import RolloutRequest

    reqs = [RolloutRequest(prompt=it["prompt"], sampling_params=it["sp"]) for it in mb_input]
    t0 = time.perf_counter()
    results, tel = orch.rollout(reqs)
    mb_wall = time.perf_counter() - t0

    # results indexed by request order = mb_pos order (we built mb_input sorted by pos).
    response_lengths = [0] * len(mb_input)
    for r in results:
        # r.raw is the engine output dict; meta_info has completion_tokens
        meta = (r.raw or {}).get("meta_info", {}) if isinstance(r.raw, dict) else {}
        response_lengths[r.index] = meta.get("completion_tokens", 0)
    return mb_wall, tel, response_lengths


def run_profile_c2(coord, orch, prompts, profile: str, seed: int, idle_threshold: int):
    out_dir = Path(GRID_ROOT) / profile / f"sim_W{SIM_W}_S{SIM_S}_C2_thr{idle_threshold}" / f"seed{seed}"
    summary_path = out_dir / "sim_summary.json"
    if summary_path.exists():
        print(f"\n=== SKIP {profile} (C2, τ={idle_threshold}) — sim_summary.json exists ===", flush=True)
        return None
    print(f"\n=== {profile} (C2, τ={idle_threshold}, seed={seed}) ===", flush=True)
    spec = load_length_spec(profile)
    print(f"  spec: μ={spec['stats']['mu']:.0f}, σ/μ={spec['stats']['sigma_mu']:.2f}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    micro_batches = []
    mb_walls = [0.0] * N_MICRO_BATCHES
    all_lens = []

    t_start = time.perf_counter()
    for m in range(N_MICRO_BATCHES):
        mb_input = build_micro_batch(prompts, spec["lengths"], m)
        mb_wall, tel, lens = run_micro_batch_c2(orch, mb_input)
        mb_walls[m] = mb_wall

        per_cluster_lens = {4 * m + w: [] for w in range(SUB_WORKERS)}
        for it, L in zip(mb_input, lens):
            per_cluster_lens[it["cluster_worker"]].append(L)

        micro_batches.append({
            "mb_id": m,
            "cluster_workers": [4 * m + w for w in range(SUB_WORKERS)],
            "mb_wall_s": mb_wall,
            "per_cluster_response_lengths": per_cluster_lens,
            "fanin_telemetry": {
                "swap_triggered": tel.swap_triggered,
                "n_finished_on_dp": tel.n_finished_on_dp,
                "n_finished_on_tp": tel.n_finished_on_tp,
                "t_first_dp_done_s": tel.t_first_dp_done_s,
                "t_last_dp_done_s": tel.t_last_dp_done_s,
                "t_swap_decision_s": tel.t_swap_decision_s,
                "t_swap_done_s": tel.t_swap_done_s,
                "t_tail_phase_done_s": tel.t_tail_phase_done_s,
                "total_wall_s": tel.total_wall_s,
            },
        })
        all_lens.extend(lens)

        sw = "swap" if tel.swap_triggered else "no-swap"
        dp_done = tel.n_finished_on_dp
        tp_done = tel.n_finished_on_tp
        print(f"  mb{m}: wall={mb_wall:.1f}s ({sw}, DP={dp_done} TP={tp_done})", flush=True)

    total_wall = time.perf_counter() - t_start

    # cluster_worker walls: each mb's 4 cluster_workers all share the mb's wall
    cluster_walls = []
    for m in range(N_MICRO_BATCHES):
        cluster_walls.extend([mb_walls[m]] * SUB_WORKERS)

    mn = min(cluster_walls)
    mx = max(cluster_walls)
    mean = sum(cluster_walls) / len(cluster_walls)
    var = sum((x - mean) ** 2 for x in cluster_walls) / len(cluster_walls)
    std = var ** 0.5
    bubble = (mx - mn) / mx if mx > 0 else 0.0

    summary = {
        "profile": profile, "seed": seed, "synthetic": True,
        "spec": spec.get("params"),
        "spec_stats": spec["stats"],
        "sim_config": {
            "W": SIM_W, "S": SIM_S,
            "n_micro_batches": N_MICRO_BATCHES,
            "sub_workers_per_mb": SUB_WORKERS,
            "n_prompts": N_PROMPTS, "n_samples": N_SAMPLES,
            "total_gen": TOTAL_GEN,
            "max_prompt": MAX_PROMPT, "max_resp": MAX_RESP, "gmu": GMU,
            "ignore_eos": True,
            "scheme": "C2_dynamic_fanin",
            "idle_dp_threshold": idle_threshold,
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

    print(f"\n  ── {profile} C2 aggregate ──")
    print(f"  mb walls: min={mn:.1f}s max={mx:.1f}s mean={mean:.1f}s std={std:.1f}s")
    print(f"  bubble = {bubble*100:.1f}% (across 8 mb walls)")
    print(f"  total wall = {total_wall:.0f}s ({total_wall/60:.1f}m)")
    print(f"  → {summary_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="+", default=[
        # σ/μ-ascending, same order as C1 sim_largeW_z
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
    ap.add_argument("--idle-threshold", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    todo = [p for p in args.profiles
            if not (Path(GRID_ROOT) / p / f"sim_W{SIM_W}_S{SIM_S}_C2_thr{args.idle_threshold}"
                    / f"seed{args.seed}" / "sim_summary.json").exists()]
    if not todo:
        print(f"All C2 profiles done at τ={args.idle_threshold}; nothing to run.")
        return 0
    print(f"To run: {todo}")
    print(f"To skip: {[p for p in args.profiles if p not in todo]}\n")

    print("[load prompt source from P1 parquet]")
    prompts = load_prompt_source()
    print(f"  loaded {len(prompts)} prompts")

    coord, dp_engines, tp_engine = boot_dual_fleet(args.seed)

    from sglang.srt.utils.rlpipe_fan_in import DynamicFanInOrchestrator
    orch = DynamicFanInOrchestrator(
        coord, idle_dp_threshold=args.idle_threshold, swap_back_after=True,
    )

    try:
        results = []
        for prof in args.profiles:
            r = run_profile_c2(coord, orch, prompts, prof, args.seed, args.idle_threshold)
            if r is not None:
                results.append(r)
        if results:
            print("\n========= SUMMARY (Z C2 fan-in) =========")
            print(f"{'profile':<36}{'σ/μ':>7}{'bubble':>10}{'wall(min)':>12}")
            for r in results:
                agg = r["aggregate"]
                sm = r["spec_stats"]["sigma_mu"]
                print(f"{r['profile']:<36}{sm:>7.2f}"
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
