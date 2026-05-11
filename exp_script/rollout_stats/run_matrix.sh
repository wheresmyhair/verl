#!/bin/bash
# Master orchestrator: 6 models × 5 datasets = 30 cells sequentially.
# Each cell writes to a separate OUT_DIR; master log records progress
# so a crashed cell can be resumed by editing the SKIP list.
set -o pipefail

OUT_ROOT=${OUT_ROOT:-/home/user/profiling_rlpipe/rollout_stats}
mkdir -p "$OUT_ROOT"
MASTER_LOG="$OUT_ROOT/master.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Shared generation config (matches RLVR rollout).
export N_SAMPLES=${N_SAMPLES:-16}
export MAX_RESP=${MAX_RESP:-16384}
export MAX_PROMPT=${MAX_PROMPT:-2048}
export BATCH_SIZE=${BATCH_SIZE:-4}
export USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE:-1}
export MAX_EXAMPLES=${MAX_EXAMPLES:-200}
export NGPU=${NGPU:-4}

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"
}

HUB=/home/user/.cache/huggingface/hub
declare -a CELLS=(
    # name | model_path | tp_size | mem_fraction
    "Qwen3-1.7B|$HUB/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e|1|0.85"
    "Qwen3-8B|$HUB/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218|1|0.85"
    "Qwen3-14B|$HUB/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18|1|0.85"
    "Qwen3-30B-A3B|$HUB/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39|2|0.85"
    "Qwen3-32B|$HUB/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137|2|0.85"
    "DS-R1-Distill-Llama-70B|$HUB/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/b1c0b44b4369b597ad119a196caf79a9c40e141e|2|0.92"
)

DATASETS=(dapo-math-17k aime-24 math-500 livecodebench codecontests)

log "== matrix start: ${#CELLS[@]} models × ${#DATASETS[@]} datasets = $((${#CELLS[@]} * ${#DATASETS[@]})) cells"
log "config: N_SAMPLES=$N_SAMPLES MAX_RESP=$MAX_RESP MAX_PROMPT=$MAX_PROMPT MAX_EXAMPLES=$MAX_EXAMPLES"

FAILS=0
TOTAL=0
for cell in "${CELLS[@]}"; do
    IFS='|' read -r NAME MODEL_PATH TP_SIZE MEM_FRAC <<< "$cell"
    for DS in "${DATASETS[@]}"; do
        TOTAL=$((TOTAL + 1))
        CELL_DIR="$OUT_ROOT/$NAME/$DS"
        DONE_MARK="$CELL_DIR/summary.json"
        if [[ -f "$DONE_MARK" ]]; then
            log "SKIP $NAME/$DS (summary.json exists)"
            continue
        fi
        mkdir -p "$CELL_DIR"
        log "RUN  $NAME/$DS  tp=$TP_SIZE mem=$MEM_FRAC  dir=$CELL_DIR"
        T0=$SECONDS
        MODEL_PATH="$MODEL_PATH" \
        DATASET="$DS" \
        OUT_DIR="$CELL_DIR" \
        TP_SIZE="$TP_SIZE" \
        MEM_FRACTION="$MEM_FRAC" \
            bash "$SCRIPT_DIR/run.sh" > "$CELL_DIR/run.log" 2>&1
        RC=$?
        DT=$((SECONDS - T0))
        if [[ $RC -ne 0 ]]; then
            log "FAIL $NAME/$DS  rc=$RC  dt=${DT}s  (see $CELL_DIR/run.log)"
            FAILS=$((FAILS + 1))
            continue
        fi
        # Post-process stats.
        python3 "$SCRIPT_DIR/analyze.py" \
            --responses "$CELL_DIR/responses.jsonl" \
            --out-json "$CELL_DIR/summary.json" \
            --per-prompt-csv "$CELL_DIR/per_prompt.csv" \
            > "$CELL_DIR/analyze.log" 2>&1 || true
        log "DONE $NAME/$DS  dt=${DT}s"
    done
done

log "== matrix end: $((TOTAL - FAILS))/$TOTAL ok, $FAILS failed"
exit $FAILS
