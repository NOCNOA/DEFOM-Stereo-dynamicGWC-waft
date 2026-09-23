#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CHECKPOINT="${CHECKPOINT:-checkpoints/defomstereo_vitl_twostage_768/defomstereo_vitl_twostage_768_200000.pth}"
PYTHON_BIN="${PYTHON_BIN:-/home/u2025140689/anaconda3/envs/defomstereo2/bin/python}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

"${PYTHON_BIN}" evaluate_stereo.py \
  --model_arch cor_ga_fast3 \
  --restore_ckpt "${CHECKPOINT}" \
  --datasets things \
  --max_disp 768 \
  --valid_iters 32 \
  --scale_iters 6 \
  --corr_levels 3 \
  --corr_radius 4 \
  --dinov2_encoder vitl \
  --hypothesis_scales 24 48 96 \
  --mixed_precision \
  --amp_dtype bf16
