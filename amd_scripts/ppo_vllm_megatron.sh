# Discliamer: the model used in the script is only for academic purpose.
set -x

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1

# Data preparation scripts are available in ``examples/data_preprocess``.
# Example usage:
#
#   python3 examples/data_preprocess/math_dataset.py --local_dir ~/data/math
#   python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

GPUS_PER_NODE=8
ENGINE=vllm
YOUR_PROJECT_NAME=amd-verl
YOUR_RUN_NAME=ppo-test


# PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=gae \
#     data.train_files="$train_files" \
#     data.val_files="$test_files" \
#     data.train_batch_size=1024 \
#     data.max_prompt_length=1024 \
#     data.max_response_length=512 \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     data.return_raw_chat=True \
#     actor_rollout_ref.model.path="Qwen/Qwen2-7B-Instruct" \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=256 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
#     actor_rollout_ref.actor.use_kl_loss=False \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#     actor_rollout_ref.rollout.name=$ENGINE \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
#     actor_rollout_ref.rollout.disable_log_stats=False \
#     critic.optim.lr=1e-5 \
#     critic.model.use_remove_padding=True \
#     critic.optim.lr_warmup_steps_ratio=0.05 \
#     critic.model.path="sfairXC/FsfairX-LLaMA3-RM-v0.1" \
#     critic.model.enable_gradient_checkpointing=True \
#     critic.ppo_micro_batch_size_per_gpu=32 \
#     reward_model.enable=True \
#     reward_model.model.path="sfairXC/FsfairX-LLaMA3-RM-v0.1" \
#     reward_model.model.use_remove_padding=True \
#     reward_model.micro_batch_size_per_gpu=32 \
#     algorithm.use_kl_in_reward=False \
#     trainer.critic_warmup=0 \
#     trainer.logger=['console','wandb'] \
#     trainer.project_name=$YOUR_PROJECT_NAME \
#     trainer.val_before_train=False \
#     trainer.experiment_name=$YOUR_RUN_NAME \
#     trainer.n_gpus_per_node=$GPUS_PER_NODE \
#     trainer.nnodes=1 \
#     trainer.save_freq=20 \
#     trainer.test_freq=5 \
#     trainer.total_epochs=3 $@

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo --config-path=/root/verl/verl/trainer/config \
    --config-name='ppo_megatron_trainer.yaml' \
    data.train_files=$gsm8k_train_path \
    data.val_files=$gsm8k_test_path \
    data.train_batch_size=256 \
    data.val_batch_size=1312 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path="Qwen/Qwen2-7B-Instruct" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.optim.lr=1e-5 \
    critic.model.path="Qwen/Qwen2-7B-Instruct" \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=console \
    trainer.project_name=$YOUR_PROJECT_NAME \
    trainer.logger=['console','wandb'] \
    trainer.experiment_name=$YOUR_RUN_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 #2>&1 | tee verl_demo.log