# Discliamer: the model used in the script is only for academic purpose.
set -x

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export HF_HOME=/home/johnny/hf_cache

python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

GPUS_PER_NODE=8
ENGINE=vllm
YOUR_PROJECT_NAME=amd-megatron-verl-grpo-qwen-gsm8k

MODEL_PATH="Qwen/Qwen2-7B-Instruct" #  You can use: deepseek-ai/deepseek-llm-7b-chat
TP_VALUE=2 #If deepseek, set TP_VALUE=2
INFERENCE_BATCH_SIZE=40 #If deepseek, set INFERENCE_BATCH_SIZE=110
GPU_MEMORY_UTILIZATION=0.6 #If deepseek, set GPU_MEMORY_UTILIZATION=0.6

YOUR_RUN_NAME=$ENGINE-TP$TP_VALUE-BSZ$INFERENCE_BATCH_SIZE-GMEM$GPU_MEMORY_UTILIZATION

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo --config-path=/root/verl/amd_scripts/config \
    --config-name='ppo_megatron_example.yaml' \
	algorithm.adv_estimator=grpo \
	data.train_files=$train_files \
	data.val_files=$test_files \
	data.train_batch_size=1024 \
	data.max_prompt_length=512 \
	data.max_response_length=1024 \
	actor_rollout_ref.model.path=$MODEL_PATH \
	actor_rollout_ref.actor.optim.lr=1e-6 \
	actor_rollout_ref.model.use_remove_padding=True \
	actor_rollout_ref.actor.ppo_mini_batch_size=256 \
	actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=80 \
	actor_rollout_ref.actor.use_kl_loss=True \
	actor_rollout_ref.actor.kl_loss_coef=0.001 \
	actor_rollout_ref.actor.kl_loss_type=low_var_kl \
	actor_rollout_ref.model.enable_gradient_checkpointing=True \
	actor_rollout_ref.actor.fsdp_config.param_offload=False \
	actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
	actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
	actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_VALUE \
	actor_rollout_ref.rollout.name=$ENGINE \
	actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
	actor_rollout_ref.rollout.n=5 \
	actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
	actor_rollout_ref.ref.fsdp_config.param_offload=True \
	algorithm.kl_ctrl.kl_coef=0.001 \
	trainer.critic_warmup=0 \
	trainer.logger=['console','wandb'] \
	trainer.project_name=$YOUR_PROJECT_NAME \
	trainer.experiment_name=$YOUR_RUN_NAME \
	trainer.n_gpus_per_node=$GPUS_PER_NODE \
	trainer.nnodes=1 \
	trainer.save_freq=-1 \
	trainer.test_freq=20 \
	trainer.total_epochs=50