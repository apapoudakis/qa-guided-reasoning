#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-http://127.0.0.1:5001/judge}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

qa_experiment_path="${QA_EXPERIMENT_PATH:-experiments/bookworm/description/qwen_3_8b/hier/grpo}"
qa_data_path="${QA_DATA_PATH:-data/processed/bookworm_descr/grpo_data/hier_16k}"
mkdir -p "$qa_experiment_path"

uv run --locked --group rl python -m openrlhf.cli.train_ppo_ray \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 4 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 4 \
  --pretrain Qwen/Qwen3-8B \
  --vllm_num_engines 4 \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.4 \
  --save_path "$qa_experiment_path/qa-ppo-rlhf-final" \
  --ckpt_path "$qa_experiment_path/qa-ppo-rlhf-ckpt" \
  --save_hf_ckpt \
  --save_steps 20 \
  --eval_steps 20 \
  --train_batch_size 64 \
  --micro_train_batch_size 2 \
  --micro_rollout_batch_size 4 \
  --rollout_batch_size 64 \
  --n_samples_per_prompt 8 \
  --max_epochs 1 \
  --max_samples 100000 \
  --prompt_max_len 17684 \
  --generate_max_len 2048 \
  --zero_stage 3 \
  --bf16 \
  --gradient_checkpointing \
  --actor_learning_rate 5e-7 \
  --init_kl_coef 0.01 \
  --prompt_data "json@$qa_data_path" \
  --input_key input \
  --label_key ground_truth \
  --eval_dataset "json@$qa_data_path" \
  --eval_split validation \
  --eval_n_samples_per_prompt 1 \
  --colocate_all_models \
  --vllm_enable_sleep \
  --deepspeed_enable_sleep \
  --remote_rm_url "$PWD/src/models/rl/reward_func.py" \
  --use_kl_loss \
  --enable_prefix_caching \
  --kl_estimator k3 \
  --advantage_estimator group_norm \
  "$@"
