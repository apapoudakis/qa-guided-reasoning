#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

uv run --locked --group rl python -m uvicorn src.models.rl.reward_server:app \
  --host "${JUDGE_HOST:-127.0.0.1}" \
  --port "${JUDGE_PORT:-5001}"
