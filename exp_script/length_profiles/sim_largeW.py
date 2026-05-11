"""Simulate larger DP-worker count (W=32) using sequential micro-batches
on our 4-GPU hardware (W=4).

Method
------
Total dataset = 128 prompts × 8 samples = 1024 generations (RollPacker
P0=128, R0=8 standard).

Imagine a W=32 cluster receiving the 1024 gens with round-robin dispatch:
  cluster_worker_k handles 32 gens, specifically the 4 prompts
  {k, k+32, k+64, k+96} × 8 samples each.

  Why 4 prompts: with interleave order (cluster_idx = sample_idx*128 +
  prompt_idx), 128 = 4*32 ⇒ all 8 samples of a prompt go to the same
  cluster worker. So each cluster worker handles 4 unique prompts × 8 = 32 gens.

We physically host only W=4 at a time. Run 8 sequential micro-batches:
  micro-batch m hosts cluster workers {4m, 4m+1, 4m+2, 4m+3}.
  Each micro-batch input = 4 cluster workers' 32-gen shares = 128 gens.

Within a micro-batch, the 128 gens are arranged so that W=4 round-robin
(idx % 4) places cluster worker (4m+w)'s 32 gens onto sub-worker w.
The order is `[for s in 0..7][for o in 0..3][for w in 0..3]` of
(prompt 4m+w+32o, sample s); position p = 16s + 4o + w; cluster_worker_id
= (s*128 + 4m+w+32o) % 32 = (4m+w) % 32 = 4m+w. ✓

Bubble = (max - min) / max across 32 cluster-worker makespans.

Cells: H3 (highest σ/μ ≈ 0.65) and P3 (lowest σ/μ ≈ 0.24).
Cost per profile ≈ 8 × per-cell wall ≈ 25-50 min.
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

N_PROMPTS = 128
N_SAMPLES = 8
TOTAL_GEN = N_PROMPTS * N_SAMPLES  # 1024
SIM_W = 32
SIM_S = TOTAL_GEN // SIM_W  # 32
N_MICRO_BATCHES = SIM_W // 4  # 8
SUB_WORKERS = 4
MB_GENS = SIM_S * SUB_WORKERS  # 128 per micro-batch

MAX_RESP = 16384
MAX_PROMPT = 2048
GMU = 0.6


def load_prompts(profile: str) -> list[str]:
    parquet = Path(PROFILES_ROOT) / profile / "train_x128.parquet"
    table = pq.read_table(parquet)
    rows = table.to_pylist()
    if len(rows) != N_PROMPTS:
        raise ValueError(f"{profile}: expected {N_PROMPTS} prompts, got {len(rows)}")
    return [r["prompt"][0]["content"] for r in rows]


def build_micro_batch(prompts: list[str], mb_id: int) -> list[dict]:
    """Construct micro-batch m's 128 gens (cluster workers 4m..4m+3).

    Position p in mb maps to (sub_worker w, prompt_id, sample_id) such that
    cluster_worker_id = 4m+w and W=4 round-robin idx%4 routes it to w.
    """
    sp = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": MAX_RESP}
    out = []
    for s in range(N_SAMPLES):
        for o in range(4):
            for w in range(SUB_WORKERS):
                prompt_id = 4 * mb_id + w + 32 * o
                cluster_w = 4 * mb_id + w
                pos = s * 16 + o * 4 + w
                out.append({
                    "mb_pos": pos,
                    "sub_worker": w,
                    "cluster_worker": cluster_w,
                    "prompt_id": prompt_id,
                    "sample_id": s,
                    "prompt": prompts[prompt_id],
                    "sp": dict(sp),
                })
    out.sort(key=lambda x: x["mb_pos"])
    assert len(out) == MB_GENS
    # Sanity: every sub-worker handles exactly 32 gens
    counts = {w: 0 for w in range(SUB_WORKERS)}
    for it in out:
        counts[it["sub_worker"]] += 1
    assert all(c == SIM_S for c in counts.values()), counts
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
    # Warmup each engine
    print("[warmup]", flush=True)
    for eng in engines:
        eng.generate("Hello world", {"temperature": 0.0, "max_new_tokens": 16})
    return engines


def run_micro_batch(engines, mb_input: list[dict]):
    """Each W=4 sub-worker handles its 32-gen chunk via batched
    SGLang generate(). Returns (engine_walls, response_lengths_indexed_by_pos)."""
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


def run_profile(engines, profile: str, seed: int):
    out_dir = Path(GRID_ROOT) / profile / f"sim_W{SIM_W}_S{SIM_S}" / f"seed{seed}"
    summary_path = out_dir / "sim_summary.json"
    if summary_path.exists():
        print(f"\n=== SKIP {profile} (seed={seed}) — sim_summary.json exists ===", flush=True)
        return None
    print(f"\n=== {profile} (seed={seed}) ===", flush=True)
    prompts = load_prompts(profile)
    out_dir.mkdir(parents=True, exist_ok=True)

    micro_batches = []
    cluster_walls = [0.0] * SIM_W
    all_lens = []

    t_start = time.perf_counter()
    for m in range(N_MICRO_BATCHES):
        mb_input = build_micro_batch(prompts, m)
        t_mb_start = time.perf_counter()
        eng_walls, lens = run_micro_batch(engines, mb_input)
        mb_wall = time.perf_counter() - t_mb_start

        # Map sub-worker walls → cluster worker walls (sub_w=0 → cluster_w=4m)
        for sw in range(SUB_WORKERS):
            cluster_walls[4 * m + sw] = eng_walls[sw]

        # Collect lens by cluster worker for accounting
        per_cluster_lens = {4 * m + w: [] for w in range(SUB_WORKERS)}
        for it, L in zip(mb_input, lens):
            per_cluster_lens[it["cluster_worker"]].append(L)

        mb_record = {
            "mb_id": m,
            "cluster_workers": [4 * m + w for w in range(SUB_WORKERS)],
            "sub_worker_walls_s": eng_walls,  # idx = sub_worker = cluster_worker mod 4
            "mb_wall_s": mb_wall,
            "per_cluster_response_lengths": per_cluster_lens,
        }
        micro_batches.append(mb_record)
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
        "sim_config": {
            "W": SIM_W, "S": SIM_S,
            "n_micro_batches": N_MICRO_BATCHES,
            "sub_workers_per_mb": SUB_WORKERS,
            "n_prompts": N_PROMPTS, "n_samples": N_SAMPLES,
            "total_gen": TOTAL_GEN, "max_resp": MAX_RESP, "gmu": GMU,
        },
        "micro_batches": micro_batches,
        "aggregate": {
            "cluster_worker_walls_s": cluster_walls,  # 32 entries
            "min_s": mn, "max_s": mx, "mean_s": mean, "std_s": std,
            "bubble_max_min_div_max": bubble,
            "total_wall_s": total_wall,
        },
        "response_lengths": all_lens,
    }
    out_path = out_dir / "sim_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n  ── {profile} aggregate ──")
    print(f"  32 cluster workers: min={mn:.1f}s max={mx:.1f}s mean={mean:.1f}s std={std:.1f}s")
    print(f"  bubble = (max-min)/max = {bubble*100:.1f}%")
    print(f"  total wall (all 8 micro-batches) = {total_wall:.0f}s ({total_wall/60:.1f}m)")
    print(f"  → {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="+", default=["H3_extreme_bimodal_cap", "P3_wide_mid"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Check if any profile actually needs running before booting fleet
    todo = [p for p in args.profiles
            if not (Path(GRID_ROOT) / p / f"sim_W{SIM_W}_S{SIM_S}" / f"seed{args.seed}" / "sim_summary.json").exists()]
    if not todo:
        print("All profiles already done; nothing to run.")
        return 0
    print(f"To run: {todo}")
    print(f"To skip: {[p for p in args.profiles if p not in todo]}\n")

    engines = boot_dp_fleet(args.seed)
    try:
        results = []
        for prof in args.profiles:
            r = run_profile(engines, prof, args.seed)
            if r is not None:
                results.append(r)
        if results:
            print("\n========= SUMMARY (this run) =========")
            print(f"{'profile':<26}{'min(s)':>10}{'max(s)':>10}{'mean':>10}{'std':>10}{'bubble':>10}")
            for r in results:
                agg = r["aggregate"]
                print(f"{r['profile']:<26}{agg['min_s']:>10.1f}{agg['max_s']:>10.1f}"
                      f"{agg['mean_s']:>10.1f}{agg['std_s']:>10.1f}{agg['bubble_max_min_div_max']*100:>9.1f}%")
    finally:
        print("\n[shutdown]")
        for eng in engines:
            try:
                eng.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
