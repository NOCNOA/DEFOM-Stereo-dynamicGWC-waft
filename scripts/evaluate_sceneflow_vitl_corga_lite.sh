#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/u2025140689/anaconda3/envs/defomstereo2/bin/python}"
CHECKPOINT="${CHECKPOINT:-checkpoints/defomstereo_vitl_corga_lite_s2_bf16/checkpoint_latest.pth}"
SCALE_ITERS="${SCALE_ITERS:-2}"

if [[ ! -s "${CHECKPOINT}" ]]; then
  echo "Checkpoint does not exist or is empty: ${CHECKPOINT}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" evaluate_stereo.py \
  --restore_ckpt "${CHECKPOINT}" \
  --model_arch cor_ga_lite \
  --datasets things \
  --dinov2_encoder vitl \
  --corr_levels 3 \
  --corr_radius 4 \
  --hypothesis_scales 24 48 96 \
  --max_delta_disp 8 \
  --n_downsample 2 \
  --context_norm batch \
  --n_gru_layers 3 \
  --hidden_dims 128 \
  --valid_iters 32 \
  --scale_iters "${SCALE_ITERS}" \
  --max_disp 768 \
  --mixed_precision
