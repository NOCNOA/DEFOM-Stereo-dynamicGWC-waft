#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CHECKPOINT="${CHECKPOINT:-checkpoints/defomstereo_vitl_corga_fast_bininit_10k/checkpoint_latest.pth}"
PYTHON_BIN="${PYTHON_BIN:-/home/u2025140689/anaconda3/envs/defomstereo2/bin/python}"
SCALE_ITERS="${SCALE_ITERS:-6}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

"${PYTHON_BIN}" evaluate_stereo.py \
  --model_arch cor_ga_fast_bininit \
  --restore_ckpt "${CHECKPOINT}" \
  --datasets things \
  --max_disp 768 \
  --valid_iters 32 \
  --scale_iters "${SCALE_ITERS}" \
  --num_init_bins 48 \
  --init_max_offset 48 \
  --init_corr_scale 10 \
  --corr_levels 3 \
  --corr_radius 4 \
  --dinov2_encoder vitl \
  --mixed_precision \
  --amp_dtype bf16
