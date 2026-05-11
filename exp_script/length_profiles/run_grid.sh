#!/bin/bash
# 11-cell × 2-cond × 1-seed real-rollout grid for fan-in vs baseline.
# Plus threshold-sweep on 3 representative cells.
#
# CELLS  P1-P5 (real curated math) + H1-H5,H7 (real-prompt hybrids).
# COND   C1 = idea1_baseline.sh   (SGLang DP=4, no dual-fleet)
#        C2 = idea1_fanin.sh      (SGLang DP=4 + TP=4 dual-fleet, idle_threshold=2)
# SEEDS  42 (single — multi-seed can be added later for error bars).
#
# Scale: RollPacker (arXiv 2509.21009) — P0=128 prompts × R0=8 samples
# = 1024 generations/step. Switched from DAPO (512×16=8192) after
# verl pipeline OOMs at end-of-gen position_ids materialization
# (1024×16384=16M elements vs DAPO's 134M).
#
# ROLLOUT-ONLY: VERL_ROLLOUT_ONLY=1 makes ray_trainer.py exit after gen.
#
# DATA: train_x128.parquet — stratified-proportional resized 200→128
# (replicate_parquets.py --target 128). Composition preserved within ±0.76pp.
#
# Per-engine GMU=0.6 in BOTH conditions. NOTE: tried 0.7 first but P5
# (bimodal 50/50) OOM'd at gen-end verl post-processing — outlier-
# concentrated profiles create a memory hotspot on rank 0 even at
# RollPacker scale. 0.6 leaves ~10 GB headroom per GPU.
#
# Per cell: writes <cell>/rollout_only.json with {timing, fanin_metrics,
# response_lengths}. timing.gen is T^gen.
#
# Wall budget: per-worker work = 128 prompts × 16 samples = 2048 gens.
# At ~4K tps with avg ~6K tokens (max=16K), per cell ≈ 50 min ×
# 28 cells ≈ 23-25 hr.
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/home/user/data/length-profiles
OUT_ROOT=/home/user/profiling_rlpipe/length_profile_grid
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/master.log"

PROFILES=(
    P1_tight_short P2_tight_mid P3_wide_mid P4_tight_long P5_bimodal
    H1_rare_outlier_1of16 H2_mid_outlier_4of16 H3_extreme_bimodal_cap
    H4_long_dominant_25_75 H5_extreme_rare_1of32 H7_uniform_narrow
)
SEEDS=(42)

# Threshold sweep: only on 3 representative cells, only seed=42, only
# C2 (threshold doesn't apply to baseline). threshold=2 is already in
# the main grid.
SWEEP_PROFILES=(H1_rare_outlier_1of16 H2_mid_outlier_4of16 P5_bimodal)
SWEEP_THRESHOLDS=(1 3)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_one() {
    local prof=$1 cond=$2 seed=$3 threshold=$4
    local tag="t${threshold}"
    local cell_dir
    if [[ -z "$threshold" || "$threshold" == "2" ]]; then
        # Main grid: no threshold suffix
        cell_dir="$OUT_ROOT/$prof/$cond/seed${seed}"
    else
        cell_dir="$OUT_ROOT/$prof/${cond}_${tag}/seed${seed}"
    fi
    if [[ -f "$cell_dir/done.flag" ]]; then
        log "SKIP $prof $cond seed=$seed thr=${threshold:-default}"
        return 0
    fi
    mkdir -p "$cell_dir"

    local script
    case $cond in
        C1) script="idea1_baseline.sh" ;;
        C2) script="idea1_fanin.sh"    ;;
        *)  log "BAD cond=$cond"; return 1 ;;
    esac

    log "RUN  $prof $cond seed=$seed thr=${threshold:-default}"
    local t0=$SECONDS
    local thr_env=""
    if [[ "$cond" == "C2" && -n "$threshold" ]]; then
        thr_env="VERL_RLPIPE_FANIN_IDLE_THRESHOLD=$threshold"
    fi
    env $thr_env \
        VERL_ROLLOUT_ONLY=1 \
        VERL_ROLLOUT_ONLY_OUT="$cell_dir/rollout_only.json" \
        SEED=$seed \
        TRAIN_FILES="['$ROOT/$prof/train_x128.parquet']" \
        VAL_FILES="['$ROOT/$prof/val.parquet']" \
        MODEL_PATH="/home/user/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218" \
        BATCH=128 N_SAMPLES=8 TOTAL_STEPS=1 \
        MAX_RESP=16384 MAX_PROMPT=2048 \
        MINI_BATCH=128 NUM_MICRO_BATCHES=128 \
        GPU_MEMORY_UTILIZATION=0.6 \
        FUSED_FORWARD=False \
        PROFILING_ROOT="$cell_dir" \
        bash "$SCRIPT_DIR/../$script" > "$cell_dir/run.log" 2>&1
    local rc=$?
    local dt=$((SECONDS - t0))
    # rollout-only mode exits via sys.exit(0) → rc=0; failure = no JSON
    if [[ ! -f "$cell_dir/rollout_only.json" ]]; then
        log "FAIL $prof $cond seed=$seed thr=${threshold:-default} rc=$rc dt=${dt}s (no json)"
        return 1
    fi
    log "DONE $prof $cond seed=$seed thr=${threshold:-default} dt=${dt}s"
    touch "$cell_dir/done.flag"
    # Incremental progress table after every cell.
    python3 "$SCRIPT_DIR/summary.py" >> "$LOG" 2>&1 || true
    return 0
}

# Loop order: seed (outermost) → cond → profile.
# Rationale: complete one full pass (seed=42, all conds, all profiles)
# before adding error-bar seeds. C2 + sweep handled by standalone
# (run_grid_standalone.py) that bypasses verl's torch_pp gloo barrier
# bug; this shell only orchestrates C1.
log "=== C1 grid start: ${#PROFILES[@]} profiles × ${#SEEDS[@]} seeds (C2 via standalone) ==="

for seed in "${SEEDS[@]}"; do
    for cond in C1; do
        for prof in "${PROFILES[@]}"; do
            run_one "$prof" "$cond" "$seed" "" || true
        done
    done
done

log "=== C1 grid end ==="
