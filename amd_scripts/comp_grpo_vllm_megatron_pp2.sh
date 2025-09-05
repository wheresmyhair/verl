# Discliamer: the model used in the script is only for academic purpose.
set -x

export HIP_VISIBLE_DEVICES=4,5
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

# setup environment
GPUS_PER_NODE=2
ROLLOUT_TP=1
TRAIN_TP=1
TRAIN_PP=2
ENGINE=vllm
INFERENCE_BATCH_SIZE=40
GPU_MEMORY_UTILIZATION=0.4

YOUR_PROJECT_NAME=amd-megatron-verl-grpo-qwen-gsm8k-report
YOUR_RUN_NAME=ROLLOUTTP$ROLLOUT_TP-TRAINTP$TRAIN_TP-TRAINPP$TRAIN_PP

# setup data
python3 examples/data_preprocess/gsm8k.py --local_dir $HOME/data/gsm8k-$YOUR_RUN_NAME

gsm8k_train_path=$HOME/data/gsm8k-$YOUR_RUN_NAME/train.parquet
gsm8k_test_path=$HOME/data/gsm8k-$YOUR_RUN_NAME/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

# setup model
MODEL_PATH="/root/hf_cache/hub/models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c"

python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer' \
	algorithm.adv_estimator=grpo \
	data.train_files=$train_files \
	data.val_files=$test_files \
	data.train_batch_size=1024 \
	data.max_prompt_length=1024 \
	data.max_response_length=1024 \
	actor_rollout_ref.model.path=$MODEL_PATH \
	actor_rollout_ref.actor.optim.lr=1e-6 \
	actor_rollout_ref.actor.ppo_mini_batch_size=256 \
	actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
	actor_rollout_ref.actor.use_kl_loss=True \
	actor_rollout_ref.actor.kl_loss_coef=0.001 \
	actor_rollout_ref.actor.kl_loss_type=low_var_kl \
	actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TRAIN_TP \
	actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$TRAIN_PP \
	actor_rollout_ref.actor.megatron.param_offload=True \
	actor_rollout_ref.actor.megatron.grad_offload=True \
	actor_rollout_ref.actor.megatron.optimizer_offload=True \
	actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
	actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
	actor_rollout_ref.rollout.name=$ENGINE \
	actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
	actor_rollout_ref.rollout.n=5 \
	actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
	actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TRAIN_TP \
	actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$TRAIN_PP \
	algorithm.kl_ctrl.kl_coef=0.001 \
	trainer.critic_warmup=0 \
	trainer.logger=['console','wandb'] \
	trainer.project_name=$YOUR_PROJECT_NAME \
	trainer.experiment_name=$YOUR_RUN_NAME \
	trainer.n_gpus_per_node=$GPUS_PER_NODE \
	trainer.nnodes=1 \
	trainer.save_freq=-1 \
	trainer.test_freq=9999 \
	trainer.total_epochs=50 \
	trainer.total_training_steps=30