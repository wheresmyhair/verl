"""sim_largeW for synthetic Z profiles using ignore_eos.

Adapted from sim_largeW.py: instead of loading train_x128.parquet and
generating until natural EOS, this loads length_spec.json (built by
build_synthetic_length_profiles.py) and forces each sample to exactly
max_new_tokens via ignore_eos=True. Lengths are pre-determined; the
SGLang generation produces garbage tokens (we don't care about content).

Same dispatch logic as sim_largeW.py:
  - W=32 sim via 8 sequential micro-batches × 4 sub-workers
  - cluster_worker (4m+w) handles prompts {4m+w, 4m+w+32, 4m+w+64, 4m+w+96}
    × all 8 samples each
  - Per-sample max_new_tokens drawn iid from profile's distribution

Output: <profile>/sim_W32_S32/seed42/sim_summary.json (same format as
sim_largeW.py).
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
# Source for prompt content (any text works; we use P1's prompts, content
# doesn't matter because ignore_eos forces fixed length)
PROMPT_SOURCE_PARQUET = Path(PROFILES_ROOT) / "P1_tight_short" / "train_x128.parquet"

N_PROMPTS = 128
N_SAMPLES = 8
TOTAL_GEN = N_PROMPTS * N_SAMPLES  # 1024
SIM_W = 32
SIM_S = TOTAL_GEN // SIM_W  # 32
N_MICRO_BATCHES = SIM_W // 4
SUB_WORKERS = 4
MB_GENS = SIM_S * SUB_WORKERS  # 128

MAX_PROMPT = 2048
MAX_RESP = 16384
GMU = 0.6


def load_prompt_source() -> list[str]:
    table = pq.read_table(PROMPT_SOURCE_PARQUET)
    rows = table.to_pylist()
    if len(rows) != N_PROMPTS:
        raise ValueError(f"prompt source has {len(rows)} rows, expected {N_PROMPTS}")
    return [r["prompt"][0]["content"] for r in rows]


def load_length_spec(profile: str) -> dict:
    spec_path = Path(PROFILES_ROOT) / profile / "length_spec.json"
    return json.loads(spec_path.read_text())


def build_micro_batch(prompts: list[str], spec_lengths: list[dict], mb_id: int) -> list[dict]:
    """Same dispatch logic as sim_largeW.py: micro-batch m hosts cluster
    workers {4m..4m+3}, each handling 4 prompts × 8 samples = 32 gens.

    Position p in mb maps to (sub_worker w, prompt_id, sample_id) such that
    cluster_worker = 4m+w and W=4 round-robin idx%4 routes to w.

    spec_lengths is the full 1024-entry list from length_spec.json. We
    look up the (prompt_id, sample_id) → max_new_tokens.
    """
    # Index for fast lookup
    idx = {(e["prompt_id"], e["sample_id"]): e["max_new_tokens"] for e in spec_lengths}

    out = []
    for s in range(N_SAMPLES):
        for o in range(4):
            for w in range(SUB_WORKERS):
                prompt_id = 4 * mb_id + w + 32 * o
                sample_id = s
                cluster_w = 4 * mb_id + w
                pos = s * 16 + o * 4 + w
                max_new = idx[(prompt_id, sample_id)]
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
                        "max_new_tokens": max_new,
                        "ignore_eos": True,
                    },
                })
    out.sort(key=lambda x: x["mb_pos"])
    assert len(out) == MB_GENS
    return out


def boot_dp_fleet(seed: int):
    import sglang as sgl
    print(f"[boot] 4 dp engines (mem_fraction={GMU})…", flush=True)
    common = dict(
        model_path=MODEL_PATH,
        enable_memory_saver=True,
        mem_fraction_static=GMU,
        random_seed=seed,
        log_level="warning",
        attention_backend="flashinfer",
        context_length=MAX_PROMPT + MAX_RESP + 32,
    )
    engines = []
    for i in range(SUB_WORKERS):
        eng = sgl.Engine(
            tp_size=1, base_gpu_id=i, port=30000 + i, nccl_port=29500 + i,
            **common,
        )
        engines.append(eng)
        time.sleep(0.5)
    print("[warmup]", flush=True)
    for eng in engines:
        eng.generate("Hello world",
                     {"temperature": 0.0, "max_new_tokens": 16})
    return engines


def run_micro_batch(engines, mb_input: list[dict]):
    response_lengths = [0] * len(mb_input)
    chunks = [[] for _ in range(SUB_WORKERS)]
    for it in mb_input:
        chunks[it["sub_worker"]].append(it)

    engine_walls = [0.0] * SUB_WORKERS

    def run_chunk(idx: int):
        eng = engines[idx]
        items = chunks[idx]
        prompts = [it["prompt"] for it in items]
        sps = [it["sp"] for it in items]
        t0 = time.perf_counter()
        outs = eng.generate(prompts, sps)
        wall = time.perf_counter() - t0
        outs = outs if isinstance(outs, list) else [outs]
        for it, o in zip(items, outs):
            meta = o.get("meta_info", {}) if isinstance(o, dict) else {}
            response_lengths[it["mb_pos"]] = meta.get("completion_tokens", 0)
        return wall

    with ThreadPoolExecutor(max_workers=SUB_WORKERS) as pool:
        futs = {k: pool.submit(run_chunk, k) for k in range(SUB_WORKERS)}
        for k, f in futs.items():
            engine_walls[k] = f.result()
    return engine_walls, response_lengths


def run_profile(engines, prompts: list[str], profile: str, seed: int):
    out_dir = Path(GRID_ROOT) / profile / f"sim_W{SIM_W}_S{SIM_S}" / f"seed{seed}"
    summary_path = out_dir / "sim_summary.json"
    if summary_path.exists():
        print(f"\n=== SKIP {profile} (seed={seed}) — sim_summary.json exists ===", flush=True)
        return None

    print(f"\n=== {profile} (seed={seed}) ===", flush=True)
    spec = load_length_spec(profile)
    print(f"  spec: μ={spec['stats']['mu']:.0f}, σ={spec['stats']['sigma']:.0f}, "
          f"σ/μ={spec['stats']['sigma_mu']:.2f}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    micro_batches = []
    cluster_walls = [0.0] * SIM_W
    all_lens = []

    t_start = time.perf_counter()
    for m in range(N_MICRO_BATCHES):
        mb_input = build_micro_batch(prompts, spec["lengths"], m)
        t_mb_start = time.perf_counter()
        eng_walls, lens = run_micro_batch(engines, mb_input)
        mb_wall = time.perf_counter() - t_mb_start

        for sw in range(SUB_WORKERS):
            cluster_walls[4 * m + sw] = eng_walls[sw]

        per_cluster_lens = {4 * m + w: [] for w in range(SUB_WORKERS)}
        for it, L in zip(mb_input, lens):
            per_cluster_lens[it["cluster_worker"]].append(L)

        micro_batches.append({
            "mb_id": m,
            "cluster_workers": [4 * m + w for w in range(SUB_WORKERS)],
            "sub_worker_walls_s": eng_walls,
            "mb_wall_s": mb_wall,
            "per_cluster_response_lengths": per_cluster_lens,
        })
        all_lens.extend(lens)

        print(f"  mb{m}: cluster_workers={[4*m+w for w in range(SUB_WORKERS)]}  "
              f"sub_walls={[f'{w:.1f}s' for w in eng_walls]}  "
              f"mb_wall={mb_wall:.1f}s", flush=True)

    total_wall = time.perf_counter() - t_start

    mn = min(cluster_walls)
    mx = max(cluster_walls)
    mean = sum(cluster_walls) / len(cluster_walls)
    var = sum((x - mean) ** 2 for x in cluster_walls) / len(cluster_walls)
    std = var ** 0.5
    bubble = (mx - mn) / mx if mx > 0 else 0.0

    summary = {
        "profile": profile,
        "seed": seed,
        "synthetic": True,
        "spec": spec["spec"] if "spec" in spec else spec.get("params"),
        "spec_stats": spec["stats"],
        "sim_config": {
            "W": SIM_W, "S": SIM_S,
            "n_micro_batches": N_MICRO_BATCHES,
            "sub_workers_per_mb": SUB_WORKERS,
            "n_prompts": N_PROMPTS, "n_samples": N_SAMPLES,
            "total_gen": TOTAL_GEN,
            "max_prompt": MAX_PROMPT, "max_resp": MAX_RESP, "gmu": GMU,
            "ignore_eos": True,
        },
        "micro_batches": micro_batches,
        "aggregate": {
            "cluster_worker_walls_s": cluster_walls,
            "min_s": mn, "max_s": mx, "mean_s": mean, "std_s": std,
            "bubble_max_min_div_max": bubble,
            "total_wall_s": total_wall,
        },
        "response_lengths": all_lens,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n  ── {profile} aggregate ──")
    print(f"  32 cluster workers: min={mn:.1f}s max={mx:.1f}s mean={mean:.1f}s std={std:.1f}s")
    print(f"  bubble = (max-min)/max = {bubble*100:.1f}%")
    print(f"  total wall (8 micro-batches) = {total_wall:.0f}s ({total_wall/60:.1f}m)")
    print(f"  → {summary_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="+", default=[
        # σ/μ-ascending order. Z1-Z6 have completed runs and will SKIP.
        "Z1_const_4K",                  # 0.00
        "Z2_bimodal_3K_5K_50pct",       # 0.25
        "Z22_bimodal_9K_15K_50pct",     # 0.25 (μ-scale)
        "Z15_uniform_2K_6K",            # 0.29 (uniform)
        "Z7_bimodal_2.5K_5.5K_50pct",   # 0.38
        "Z18_truncnormal_4K_2K",        # 0.46 (trunc normal)
        "Z3_bimodal_2K_6K_50pct",       # 0.51
        "Z23_bimodal_5K_15K_50pct",     # 0.51 (μ-scale)
        "Z24_bimodal_500_1.5K_50pct",   # 0.51 (μ-scale small)
        "Z16_uniform_500_8K",           # 0.51 (uniform)
        "Z8_bimodal_1.5K_6.5K_50pct",   # 0.64
        "Z17_trimodal_500_4K_12K",      # 0.87 (trimodal)
        "Z9_bimodal_500_7.5K_50pct",    # 0.90
        "Z4_bimodal_200_8K_50pct",      # 0.99
        "Z10_bimodal_200_8K_30pct",     # 1.42
        "Z11_bimodal_200_16K_30pct",    # 1.48
        "Z19_pareto_80short_20long",    # 1.73 (mixed uniform)
        "Z12_bimodal_200_16K_20pct",    # 2.00
        "Z20_pareto_90short_10long",    # 2.10 (mixed uniform)
        "Z13_bimodal_200_16K_15pct",    # 2.30
        "Z21_pareto_95short_5long",     # 2.32 (mixed uniform)
        "Z5_bimodal_200_16K_10pct",     # 2.78
        "Z14_bimodal_200_16K_8pct",     # 3.11
        "Z6_bimodal_200_16K_5pct",      # 3.70
    ])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    todo = [p for p in args.profiles
            if not (Path(GRID_ROOT) / p / f"sim_W{SIM_W}_S{SIM_S}" / f"seed{args.seed}" / "sim_summary.json").exists()]
    if not todo:
        print("All profiles already done; nothing to run.")
        return 0
    print(f"To run: {todo}")
    print(f"To skip: {[p for p in args.profiles if p not in todo]}\n")

    print("[load prompt source from P1 parquet]")
    prompts = load_prompt_source()
    print(f"  loaded {len(prompts)} prompts (content irrelevant; ignore_eos enforces length)")

    engines = boot_dp_fleet(args.seed)
    try:
        results = []
        for prof in args.profiles:
            r = run_profile(engines, prompts, prof, args.seed)
            if r is not None:
                results.append(r)
        if results:
            print("\n========= SUMMARY (Z synthetic) =========")
            print(f"{'profile':<36}{'σ/μ':>7}{'bubble':>10}{'wall(min)':>12}")
            for r in results:
                agg = r["aggregate"]
                sm = r["spec_stats"]["sigma_mu"]
                print(f"{r['profile']:<36}{sm:>7.2f}"
                      f"{agg['bubble_max_min_div_max']*100:>9.1f}%"
                      f"{agg['total_wall_s']/60:>11.1f}")
    finally:
        print("\n[shutdown]")
        for eng in engines:
            try:
                eng.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
