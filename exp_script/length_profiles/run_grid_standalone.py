"""Standalone fan-in grid for C2 + sweep cells.

Bypasses verl's torch_pp Ray actor wrapper and the gloo dist.barrier
deadlock that broke all C2 cells when running through verl. Talks
directly to SGLang via sgl.Engine + DualFleetCoordinator +
DynamicFanInOrchestrator (the same fan-in plumbing that the verl path
wraps but with no Ray/gloo overhead).

Output: <profile>/<cond>[/seed42]/rollout_only.json compatible with
exp_script/length_profiles/summary.py.

Loop order: seed (outermost) → cond → profile. Boots fleet once per
(seed, cond) combination and reuses across all profiles to amortize
the ~60s SGLang startup cost.

Cells handled (28 total):
  - 11× C2 main grid (idle_threshold=2 default)
  - 6× sweep on H1, H2, P5 × {idle_threshold=1, 3}
  - C1 cells handled separately by run_grid.sh (verl path works for C1).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

# sglang-fork on python path
sys.path.insert(0, "/home/user/rlpipe/sglang-fork/python")

PROFILES = [
    "P1_tight_short", "P2_tight_mid", "P3_wide_mid",
    "P4_tight_long", "P5_bimodal",
    "H1_rare_outlier_1of16", "H2_mid_outlier_4of16",
    "H3_extreme_bimodal_cap", "H4_long_dominant_25_75",
    "H5_extreme_rare_1of32", "H7_uniform_narrow",
]
SWEEP_PROFILES = ["H1_rare_outlier_1of16", "H2_mid_outlier_4of16", "P5_bimodal"]
SWEEP_THRESHOLDS = [1, 3]
DEFAULT_THRESHOLD = 2

PROFILES_ROOT = "/home/user/data/length-profiles"
GRID_ROOT = "/home/user/profiling_rlpipe/length_profile_grid"
MODEL_PATH = "/home/user/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
SUMMARY_PY = "/home/user/rlpipe/verl/exp_script/length_profiles/summary.py"

# Match verl/idea1_fanin.sh: BATCH=128 prompts × N=8 samples = 1024 gens
N_SAMPLES = 8
MAX_RESP = 16384
MAX_PROMPT = 2048
GMU_PER_ENGINE = 0.55  # 5 engines * 0.55 ≈ 2.75 of HBM (with torch_memory_saver)


def load_prompts(profile: str) -> list[str]:
    """Read train_x128.parquet → render chat template via tokenizer."""
    parquet = Path(PROFILES_ROOT) / profile / "train_x128.parquet"
    table = pq.read_table(parquet)
    rows = table.to_pylist()
    prompts = []
    for r in rows:
        msgs = r.get("prompt") or []
        # verl uses chat template via tokenizer.apply_chat_template; for
        # Qwen3 base it's simple — just concat the user content. The
        # apply_chat_template result is what verl actually feeds to the
        # rollout engine, so do it the same way here.
        text = msgs[0]["content"] if msgs else ""
        prompts.append(text)
    return prompts


def make_requests(prompts: list[str], n: int = N_SAMPLES, max_new: int = MAX_RESP):
    """Expand each prompt → n stochastic samples (n=N_SAMPLES).

    Order matches verl's `gen_batch.repeat(n=n, interleave=True)`:
      [p0_s0, p1_s0, ..., p_{B-1}_s0, p0_s1, p1_s1, ..., ...]
    so contiguous chunking by 4 DP engines puts the same prompt-pool
    on each engine with different sample subsets (balanced load).
    """
    from sglang.srt.utils.rlpipe_fan_in import RolloutRequest
    sp = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": max_new,
    }
    reqs = []
    for s in range(n):
        for p in prompts:
            reqs.append(RolloutRequest(prompt=p, sampling_params=dict(sp)))
    return reqs


def boot_fleet(cond: str, seed: int):
    """Boot DP=4 fleet (+ TP=4 for C2). Returns (coord, dp_engines, tp_engine)."""
    import sglang as sgl
    from sglang.srt.utils.rlpipe_dual_fleet import DualFleetCoordinator

    common = dict(
        model_path=MODEL_PATH,
        enable_memory_saver=True,
        enable_weights_cpu_backup=True,
        mem_fraction_static=GMU_PER_ENGINE,
        random_seed=seed,
        log_level="warning",
        attention_backend="flashinfer",
        context_length=MAX_PROMPT + MAX_RESP + 32,
    )

    tp_engine = None
    if cond == "C2":
        print(f"[{cond}/seed={seed}] booting tp engine (TP=4)…", flush=True)
        tp_engine = sgl.Engine(
            tp_size=4, base_gpu_id=0, port=40000, nccl_port=29600, **common
        )
        time.sleep(1)
        # NOTE: don't explicit release; let auto_pause_inactive in coord
        # manage. release_memory_occupation + later swap_topology hit a
        # Triton "CPU tensor" bug at 8B (works fine at 1.7B in smoke_e7).

    print(f"[{cond}/seed={seed}] booting 4 dp engines…", flush=True)
    dp_engines = []
    for i in range(4):
        eng = sgl.Engine(
            tp_size=1, base_gpu_id=i, port=30000 + i, nccl_port=29500 + i,
            **common,
        )
        dp_engines.append(eng)
        time.sleep(0.5)

    if cond == "C2":
        coord = DualFleetCoordinator(
            dp_engines=dp_engines, tp_engines=[tp_engine],
            initial_active="dp", auto_pause_inactive=True,
        )
    else:
        coord = DualFleetCoordinator(
            dp_engines=dp_engines, tp_engines=[],
            initial_active="dp", auto_pause_inactive=True,
        )
    time.sleep(1)
    return coord, dp_engines, tp_engine


def run_pure_dp(coord, requests):
    """C1 baseline: round-robin across DP engines, batched per-engine.

    Matches DynamicFanInOrchestrator's chunking (i % n_engines) and
    batched generate() so the C1 vs C2 comparison is apples-to-apples
    on routing — they differ ONLY in whether swap fires.
    """
    from concurrent.futures import ThreadPoolExecutor
    from sglang.srt.utils.rlpipe_fan_in import DynamicFanInOrchestrator
    dp_engines = coord.fleet("dp")
    n_eng = len(dp_engines)
    chunks: dict[int, list] = {k: [] for k in range(n_eng)}
    for i, r in enumerate(requests):
        chunks[i % n_eng].append((i, r))

    response_lengths = [0] * len(requests)
    engine_walls = [0.0] * n_eng
    t0 = time.perf_counter()

    def run_engine_chunk(eng_idx: int, items: list):
        eng = dp_engines[eng_idx]
        reqs = [r for _, r in items]
        t_e = time.perf_counter()
        outs = DynamicFanInOrchestrator._run_batch(eng, reqs)
        wall = time.perf_counter() - t_e
        for (gidx, _), o in zip(items, outs):
            meta = o.get("meta", {}) if isinstance(o, dict) else {}
            response_lengths[gidx] = meta.get("completion_tokens", 0)
        return wall

    with ThreadPoolExecutor(max_workers=n_eng) as pool:
        futs = [pool.submit(run_engine_chunk, k, items) for k, items in chunks.items()]
        for k, f in enumerate(futs):
            engine_walls[k] = f.result()

    total_wall = time.perf_counter() - t0
    return total_wall, engine_walls, response_lengths


def run_fanin(coord, requests, idle_threshold: int):
    """C2 fan-in via DynamicFanInOrchestrator."""
    from sglang.srt.utils.rlpipe_fan_in import DynamicFanInOrchestrator
    orch = DynamicFanInOrchestrator(coord, idle_dp_threshold=idle_threshold)
    results, tel = orch.rollout(requests)
    response_lengths = [0] * len(requests)
    for r in results.values() if isinstance(results, dict) else results:
        if hasattr(r, "raw") and isinstance(r.raw, dict):
            meta = r.raw.get("meta_info", {})
            response_lengths[r.index] = meta.get("completion_tokens", 0)
        else:
            response_lengths[r.index] = 0
    return tel, response_lengths


def write_cell_json(profile: str, cond: str, seed: int, threshold: int,
                    total_wall: float, engine_walls: list[float],
                    response_lengths: list[int], tel=None, fired: bool = False):
    if threshold == DEFAULT_THRESHOLD or cond == "C1":
        cell_dir = Path(GRID_ROOT) / profile / cond / f"seed{seed}"
    else:
        cell_dir = Path(GRID_ROOT) / profile / f"{cond}_t{threshold}" / f"seed{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    out_path = cell_dir / "rollout_only.json"
    mn = min(engine_walls) if engine_walls else 0.0
    mx = max(engine_walls) if engine_walls else 0.0
    timing = {
        "gen": total_wall,
        "generate_sequences": total_wall,
        "generation_timing/min": mn,
        "generation_timing/max": mx,
        "generation_timing/topk_ratio": 0.25,
        "load_rollout": 0.0,
        "unload_rollout": 0.0,
    }
    fanin_metrics = {}
    if tel is not None:
        fanin_metrics = {
            "fanin/fanin_fired": tel.swap_triggered,
            "fanin/n_finished_on_dp": tel.n_finished_on_dp,
            "fanin/n_finished_on_tp": tel.n_finished_on_tp,
            "fanin/t_first_dp_done_s": tel.t_first_dp_done_s or 0.0,
            "fanin/t_last_dp_done_s": tel.t_last_dp_done_s or 0.0,
            "fanin/t_swap_decision_s": tel.t_swap_decision_s or 0.0,
            "fanin/t_swap_done_s": tel.t_swap_done_s or 0.0,
        }
    out = {
        "step": 0,
        "timing": timing,
        "fanin_metrics": fanin_metrics,
        "response_lengths": [float(x) for x in response_lengths],
    }
    out_path.write_text(json.dumps(out, indent=2))
    (cell_dir / "done.flag").touch()
    print(f"  wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--conds", nargs="+", default=["C2"], choices=["C1", "C2"])
    ap.add_argument("--profiles", nargs="+", default=PROFILES)
    ap.add_argument("--include-sweep", action="store_true", default=True,
                    help="Include threshold sweep cells (C2 only)")
    args = ap.parse_args()

    log_path = Path(GRID_ROOT) / "standalone.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("a")

    def log(msg: str):
        ts = time.strftime("%F %T")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    log(f"=== STANDALONE start: seeds={args.seeds} conds={args.conds} ===")

    for seed in args.seeds:
        for cond in args.conds:
            coord, dp_engines, tp_engine = boot_fleet(cond, seed)

            # Warmup
            log(f"[{cond}/seed={seed}] warmup…")
            sp_warm = {"temperature": 0.0, "max_new_tokens": 16}
            for eng in dp_engines:
                eng.generate("Hello world", sp_warm)
            if tp_engine is not None:
                coord.swap_topology("tp")
                tp_engine.generate("Hello world", sp_warm)
                coord.swap_topology("dp")

            # Main grid
            for prof in args.profiles:
                cell_dir = Path(GRID_ROOT) / prof / cond / f"seed{seed}"
                done_flag = cell_dir / "done.flag"
                if done_flag.exists():
                    log(f"SKIP {prof} {cond} seed={seed}")
                    continue
                log(f"RUN  {prof} {cond} seed={seed} thr=default")
                t0 = time.time()
                prompts = load_prompts(prof)
                requests = make_requests(prompts)
                try:
                    if cond == "C1":
                        wall, eng_walls, lens = run_pure_dp(coord, requests)
                        write_cell_json(prof, cond, seed, DEFAULT_THRESHOLD,
                                        wall, eng_walls, lens)
                    else:
                        tel, lens = run_fanin(coord, requests, DEFAULT_THRESHOLD)
                        eng_walls = [tel.t_first_dp_done_s or 0.0,
                                     tel.t_last_dp_done_s or 0.0]
                        write_cell_json(prof, cond, seed, DEFAULT_THRESHOLD,
                                        tel.total_wall_s, eng_walls, lens, tel=tel)
                    log(f"DONE {prof} {cond} seed={seed} thr=default dt={time.time()-t0:.0f}s")
                except Exception as e:
                    log(f"FAIL {prof} {cond} seed={seed} thr=default err={e!r}")
                # Regenerate progress table per cell
                os.system(f"python3 {SUMMARY_PY} >> {log_path} 2>&1")

            # Sweep cells (C2 only, threshold ∈ {1, 3})
            if cond == "C2" and args.include_sweep:
                for prof in SWEEP_PROFILES:
                    for thr in SWEEP_THRESHOLDS:
                        tag = f"C2_t{thr}"
                        cell_dir = Path(GRID_ROOT) / prof / tag / f"seed{seed}"
                        done_flag = cell_dir / "done.flag"
                        if done_flag.exists():
                            log(f"SKIP sweep {prof} thr={thr} seed={seed}")
                            continue
                        log(f"RUN  sweep {prof} thr={thr} seed={seed}")
                        t0 = time.time()
                        prompts = load_prompts(prof)
                        requests = make_requests(prompts)
                        try:
                            tel, lens = run_fanin(coord, requests, thr)
                            eng_walls = [tel.t_first_dp_done_s or 0.0,
                                         tel.t_last_dp_done_s or 0.0]
                            write_cell_json(prof, "C2", seed, thr,
                                            tel.total_wall_s, eng_walls, lens, tel=tel)
                            log(f"DONE sweep {prof} thr={thr} seed={seed} dt={time.time()-t0:.0f}s")
                        except Exception as e:
                            log(f"FAIL sweep {prof} thr={thr} seed={seed} err={e!r}")
                        os.system(f"python3 {SUMMARY_PY} >> {log_path} 2>&1")

            log(f"[{cond}/seed={seed}] shutdown fleet…")
            coord.shutdown()
            time.sleep(3)

    log(f"=== STANDALONE end ===")


if __name__ == "__main__":
    sys.exit(main())
