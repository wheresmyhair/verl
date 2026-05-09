#!/bin/bash
# M-A: Megatron baseline smoke test bed.
# 1-step GRPO at PP=2, TP=1, Qwen3-0.6B for fastest iteration.
# Goal: establish that verl+megatron training works in our env, before
#       layering fused-forward (gloo) and fanin on top.
set -o pipefail
set -x
rm -f log_rank_*.txt

# CUDA env for megatron overlap
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Local model cache (Qwen3-0.6B for fast smoke)
MODEL_PATH="${MODEL_PATH:-/home/user/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"

source "$(dirname "$0")/../common.sh"

# Override common.sh defaults for SMALL smoke
BATCH=4
N_SAMPLES=2
MAX_RESP=512
MAX_PROMPT=1024
MINI_BATCH=4
INFERENCE_BATCH_SIZE=1
TOTAL_STEPS=1

EXP_NAME=idea2_megatron_baseline
PROFILING_DIR=$PROFILING_ROOT/$EXP_NAME/seed_${SEED}
rm -rf "$PROFILING_DIR"
mkdir -p "$PROFILING_DIR"

ENGINE=sglang
ROLLOUT_TP=1
GPU_MEMORY_UTILIZATION=0.55
PP_SIZE=${PP_SIZE:-2}
TP_SIZE=${TP_SIZE:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}

python3 -m verl.trainer.main_ppo \
    --config-path=/home/user/rlpipe/verl/verl/trainer/config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.prompt_key=prompt \
    data.train_batch_size=$BATCH \
    data.max_prompt_length=$MAX_PROMPT \
    data.max_response_length=$MAX_RESP \
    data.shuffle=True \
    data.seed=$SEED \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.weight_sync_mode=${WEIGHT_SYNC_MODE:-tensor} \
    actor_rollout_ref.rollout.n=$N_SAMPLES \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.ref.megatron.param_offload=True \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=rlpipe \
    trainer.experiment_name=${EXP_NAME}_seed${SEED} \
    trainer.resume_mode=disable \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=9999 \
    trainer.val_before_train=False \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.total_epochs=1 2>&1 | stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' | tee $PROFILING_DIR/train.log
