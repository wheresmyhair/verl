#!/bin/bash
# Validate the DP-bubble simulator on each math profile (P1-P5).
#
# Strategy: run 1 real training step per profile with the dual-fleet
# fan-in disabled (idle_threshold = W so swap never fires), then read
# the fan-in orchestrator's t_first_dp_done_s / t_last_dp_done_s
# telemetry to compute the real DP-worker bubble:
#     real_bubble = (t_last - t_first) / t_last
# Compare to simulator prediction at (W=4, S=16, verl_default).
#
# 5 profiles × ~10 min/step ≈ 50 min wall.
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/home/user/data/length-profiles
OUT_ROOT=/home/user/profiling_rlpipe/length_profile_validation
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/master.log"

PROFILES=(P1_tight_short P2_tight_mid P3_wide_mid P4_tight_long P5_bimodal)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== validation start: ${#PROFILES[@]} profiles, 1 step each, idle_threshold=4 (no swap) ==="

for prof in "${PROFILES[@]}"; do
    cell_dir="$OUT_ROOT/$prof"
    if [[ -f "$cell_dir/done.flag" ]]; then
        log "SKIP $prof (already done)"
        continue
    fi
    mkdir -p "$cell_dir"
    log "RUN  $prof"
    T0=$SECONDS
    # Note: profile parquets use tokenizer-formatted user prompts (chat
    # template applied at preprocess time). Override TRAIN/VAL_FILES.
    VERL_RLPIPE_FANIN_IDLE_THRESHOLD=4 \
    TRAIN_FILES="['$ROOT/$prof/train.parquet']" \
    VAL_FILES="['$ROOT/$prof/val.parquet']" \
    MODEL_PATH="/home/user/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218" \
    BATCH=16 N_SAMPLES=16 TOTAL_STEPS=1 \
    MAX_RESP=12288 MAX_PROMPT=2048 \
    MINI_BATCH=16 NUM_MICRO_BATCHES=256 \
    GPU_MEMORY_UTILIZATION=0.55 \
    FUSED_FORWARD=False \
    PROFILING_ROOT="$cell_dir" \
        bash "$SCRIPT_DIR/../idea1_fanin.sh" > "$cell_dir/run.log" 2>&1
    RC=$?
    DT=$((SECONDS - T0))
    if [[ $RC -ne 0 ]]; then
        log "FAIL $prof rc=$RC dt=${DT}s"
        continue
    fi
    log "DONE $prof dt=${DT}s"
    touch "$cell_dir/done.flag"
done

log "=== validation end ==="
