#!/bin/bash
# Phase D runner: iterate over 6 experiments × 3 seeds.
#
# Usage:
#   cd /home/user/rlpipe/verl
#   bash exp_script/phase_d/run_all.sh                # full: 6 exp × 3 seeds × 10 steps
#   SMOKE=1 bash exp_script/phase_d/run_all.sh        # smoke: 6 exp × 1 seed × 1 step
#   ONLY_MEGATRON=1 bash exp_script/phase_d/run_all.sh  # skip torch_pp scripts
#   SKIP_FANIN=1 bash exp_script/phase_d/run_all.sh     # skip exp 3 / exp 4
#   SEEDS="42 123" bash exp_script/phase_d/run_all.sh   # custom seed list
#
# Between runs: 10s cooldown + verify GPUs clear.
#
# All profiling output under $HOME/profiling_phase_d/

set -e  # fail fast — if any one experiment crashes, user should see it immediately
set -o pipefail  # so errors propagate through "| tee"

SMOKE=${SMOKE:-0}
ONLY_MEGATRON=${ONLY_MEGATRON:-0}
SKIP_FANIN=${SKIP_FANIN:-0}
SEEDS=${SEEDS:-"42 123 2024"}

if [ "$SMOKE" = "1" ]; then
    export TOTAL_STEPS=1
    SEEDS="42"
    # Smaller batch + shorter responses to validate scripts quickly.
    # Real measurement runs use the script's default (128 × n=16 × 16K).
    export SMOKE_BATCH=${SMOKE_BATCH:-16}
    export SMOKE_MINI_BATCH=${SMOKE_MINI_BATCH:-4}
    export SMOKE_N=${SMOKE_N:-2}
    export SMOKE_MAX_RESP=${SMOKE_MAX_RESP:-2048}
    echo "[run_all] SMOKE mode: TOTAL_STEPS=1, SEEDS=42, BATCH=$SMOKE_BATCH, MINI_BATCH=$SMOKE_MINI_BATCH, N=$SMOKE_N, MAX_RESP=$SMOKE_MAX_RESP"
else
    export TOTAL_STEPS=${TOTAL_STEPS:-10}
fi

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd /home/user/rlpipe/verl

# Build experiment list
EXPS=()
EXPS+=("1_baseline_megatron_default.sh")
if [ "$ONLY_MEGATRON" != "1" ]; then
    EXPS+=("2_baseline_torchpp_default.sh")
    EXPS+=("3_exp_torchpp_fused.sh")
fi
EXPS+=("4_exp_megatron_fused.sh")
if [ "$SKIP_FANIN" != "1" ]; then
    EXPS+=("5_exp_megatron_fanin.sh")
    EXPS+=("6_exp_megatron_fused_fanin.sh")
fi

TOTAL_RUNS=$((${#EXPS[@]} * $(echo $SEEDS | wc -w)))
echo "[run_all] total runs: $TOTAL_RUNS (${#EXPS[@]} exps × $(echo $SEEDS | wc -w) seeds)"
echo "[run_all] TOTAL_STEPS=$TOTAL_STEPS per run"
echo "[run_all] experiments: ${EXPS[*]}"
echo "[run_all] seeds: $SEEDS"

RUN_IDX=0
for exp in "${EXPS[@]}"; do
    for seed in $SEEDS; do
        RUN_IDX=$((RUN_IDX + 1))
        echo ""
        echo "============================================="
        echo "[run_all] RUN $RUN_IDX/$TOTAL_RUNS: $exp seed=$seed"
        echo "[run_all] started at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "============================================="

        # Wait for GPUs to clear (prior run cleanup)
        for i in 1 2 3 4 5; do
            free_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
            if [ "$free_mem" -lt 500 ]; then
                break
            fi
            echo "[run_all] waiting for GPUs to clear... (max used=$free_mem MB)"
            sleep 10
        done

        SEED=$seed bash $SCRIPT_DIR/$exp
        exit_code=$?

        if [ $exit_code -ne 0 ]; then
            echo "[run_all] ERROR: $exp seed=$seed exited with code $exit_code"
            echo "[run_all] stopping run_all. Fix the issue and re-run."
            exit 1
        fi

        echo "[run_all] completed: $exp seed=$seed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        # cooldown between runs
        sleep 10
    done
done

echo ""
echo "============================================="
echo "[run_all] ALL $TOTAL_RUNS RUNS COMPLETED at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[run_all] profiling output: $HOME/profiling_phase_d/"
echo "============================================="
echo "[run_all] next: python3 $SCRIPT_DIR/analyze.py $HOME/profiling_phase_d/"
