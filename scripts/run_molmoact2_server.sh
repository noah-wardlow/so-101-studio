#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_path="${MOLMOACT2_VENV:-"$repo_root/.venv-molmoact2"}"
python_bin="${PYTHON:-python3}"

if [[ ! -x "$venv_path/bin/python" ]]; then
  "$python_bin" -m venv "$venv_path"
fi

source "$venv_path/bin/activate"
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.5.1" \
  "torchvision==0.20.1"
python -m pip install \
  "transformers==5.12.1" \
  "tokenizers==0.23.0rc0" \
  "huggingface_hub>=1.20.1" \
  "httpx" \
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
