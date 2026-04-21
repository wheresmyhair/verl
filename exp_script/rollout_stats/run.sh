#!/bin/bash
# Rollout-only response-length stats orchestrator.
#
# Launches N DP worker processes across 4 GPUs, splits the prompts
# JSONL by round-robin (so same-difficulty prompts are evenly spread),
# each worker holds one SGLang engine (tp=TP_SIZE over a contiguous
# GPU range) and writes its own output JSONL. Merges into a single
# file at the end.
#
# Required args (as env vars):
#   MODEL_PATH    HF model dir
#   DATASET       dataset name (dapo-math-17k / aime-24 / math-500 / ...)
#   OUT_DIR       output directory (auto-created)
#   TP_SIZE       1 | 2 | 4  (2 only for 70B on 4 GPUs)
#
# Optional:
#   MAX_EXAMPLES  cap examples (default: all)
#   N_SAMPLES     samples per prompt (default 16)
#   MAX_RESP      max_new_tokens (default 16384)
#   MAX_PROMPT    max prompt tokens — only used at preprocess time (default 2048)
#   BATCH_SIZE    prompts per engine.generate call (default 16)
#   MEM_FRACTION  mem_fraction_static (default 0.85)
#   NGPU          total GPUs available (default 4)
#   USE_CHAT_TEMPLATE  set to 1 to apply tokenizer chat template
set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH}"
: "${DATASET:?set DATASET}"
: "${OUT_DIR:?set OUT_DIR}"
: "${TP_SIZE:=1}"
: "${N_SAMPLES:=16}"
: "${MAX_RESP:=16384}"
: "${MAX_PROMPT:=2048}"
: "${BATCH_SIZE:=16}"
: "${MEM_FRACTION:=0.85}"
: "${NGPU:=4}"
: "${MAX_EXAMPLES:=}"
: "${USE_CHAT_TEMPLATE:=0}"

if (( NGPU % TP_SIZE != 0 )); then
    echo "NGPU ($NGPU) must be divisible by TP_SIZE ($TP_SIZE)" >&2
    exit 1
fi
DP_SIZE=$(( NGPU / TP_SIZE ))

mkdir -p "$OUT_DIR"
PROMPTS_FILE="$OUT_DIR/prompts.jsonl"

# ─── 1. Build prompts JSONL ──────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPROC_ARGS=(--dataset "$DATASET" --out "$PROMPTS_FILE")
[[ -n "$MAX_EXAMPLES" ]] && PREPROC_ARGS+=(--max-examples "$MAX_EXAMPLES")
[[ "$USE_CHAT_TEMPLATE" == "1" ]] && PREPROC_ARGS+=(--model-path "$MODEL_PATH")

python3 "$SCRIPT_DIR/preprocess_datasets.py" "${PREPROC_ARGS[@]}"

TOTAL=$(wc -l < "$PROMPTS_FILE")
echo "[run.sh] $TOTAL prompts; dp=$DP_SIZE tp=$TP_SIZE n=$N_SAMPLES"

# ─── 2. Shard prompts round-robin into DP_SIZE chunks ────────────────
for ((w=0; w<DP_SIZE; w++)); do
    awk -v w="$w" -v d="$DP_SIZE" 'NR % d == w' "$PROMPTS_FILE" \
        > "$OUT_DIR/prompts_w${w}.jsonl"
done

# ─── 3. Launch workers in parallel ───────────────────────────────────
PIDS=()
for ((w=0; w<DP_SIZE; w++)); do
    START_GPU=$((w * TP_SIZE))
    END_GPU=$((START_GPU + TP_SIZE - 1))
    GPUS=$(seq -s, "$START_GPU" "$END_GPU")
    LOG="$OUT_DIR/worker_${w}.log"
    echo "[run.sh] worker $w → GPUs $GPUS  log=$LOG"
    CUDA_VISIBLE_DEVICES="$GPUS" \
    python3 "$SCRIPT_DIR/worker.py" \
        --model-path "$MODEL_PATH" \
        --prompts "$OUT_DIR/prompts_w${w}.jsonl" \
        --out "$OUT_DIR/out_w${w}.jsonl" \
        --tp-size "$TP_SIZE" \
        --n-samples "$N_SAMPLES" \
        --max-new-tokens "$MAX_RESP" \
        --max-prompt-tokens "$MAX_PROMPT" \
        --batch-size "$BATCH_SIZE" \
        --mem-fraction "$MEM_FRACTION" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

# ─── 4. Wait for all, propagate failures ─────────────────────────────
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "[run.sh] worker pid=$pid failed" >&2
        FAIL=1
    fi
done

# ─── 5. Merge outputs ────────────────────────────────────────────────
cat "$OUT_DIR"/out_w*.jsonl > "$OUT_DIR/responses.jsonl"
echo "[run.sh] merged $(wc -l < "$OUT_DIR/responses.jsonl") samples → $OUT_DIR/responses.jsonl"

exit $FAIL
