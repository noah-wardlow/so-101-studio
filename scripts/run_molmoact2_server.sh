#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_path="${MOLMOACT2_VENV:-"$repo_root/.venv-molmoact2"}"
python_bin="${PYTHON:-python3}"

if [[ ! -x "$venv_path/bin/python" ]]; then
  "$python_bin" -m venv "$venv_path"
fi

source "$venv_path/bin/activate"
export SO101_SIM_TO_POLICY_SIGN="${SO101_SIM_TO_POLICY_SIGN:-"1,-1,1,-1,1"}"
export SO101_SIM_ACTION_BIAS_RAD="${SO101_SIM_ACTION_BIAS_RAD:-"-0.15,0.30,0.20,-0.10,0.10,0"}"
export MOLMO_SEED="${MOLMO_SEED:-"1005"}"

python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.5.1" \
  "torchvision==0.20.1"
python -m pip install \
  "transformers==5.12.1" \
  "tokenizers==0.23.0rc0" \
  "huggingface_hub>=1.20.1" \
  "httpx" \
  "requests" \
  "typer" \
  "rich" \
  "pillow" \
  "numpy" \
  "accelerate" \
  "safetensors" \
  "einops" \
  "timm" \
  "sentencepiece" \
  "protobuf"

exec python "$repo_root/scripts/molmoact2_so101_server.py" "$@"
