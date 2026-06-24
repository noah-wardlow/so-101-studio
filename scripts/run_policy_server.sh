#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-davidlinjiahao/lerobot_so101_base_sim_pickplace}"
DATASET_ID="${DATASET_ID:-davidlinjiahao/lerobot_batch_001}"
POLICY_SPACE="${POLICY_SPACE:-sim-radians-act}"
POLICY_DEVICE="${POLICY_DEVICE:-auto}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-8776}"

cmd=(
  python scripts/lerobot_policy_server.py
  --model-id "${MODEL_ID}"
  --dataset-id "${DATASET_ID}"
  --policy-space "${POLICY_SPACE}"
  --device "${POLICY_DEVICE}"
  --host "${POLICY_HOST}"
  --port "${POLICY_PORT}"
)

if [[ -n "${LEROBOT_ROOT:-}" && -f "${LEROBOT_ROOT}/pyproject.toml" ]]; then
  exec uv run --project "${LEROBOT_ROOT}" "${cmd[@]}"
fi

exec uv run --group policy "${cmd[@]}"
