#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
if [[ "${GPU_ID}" == "1" || ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a single GPU index other than 1 (repository policy)." >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
CHECKPOINT="${CHECKPOINT:-checkpoints/defomstereo_vitl_corga_fast2_two_branch_scratch/checkpoint_latest.pth}"
PYTHON_BIN="${PYTHON_BIN:-/home/u2025140689/anaconda3/envs/defomstereo2/bin/python}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

"${PYTHON_BIN}" evaluate_stereo.py \
  --model_arch cor_ga_fast2_two_branch \
  --restore_ckpt "${CHECKPOINT}" \
  --datasets things \
  --max_disp 768 \
  --valid_iters 32 \
  --scale_iters 6 \
  --corr_levels 3 \
  --corr_radius 4 \
  --dinov2_encoder vitl \
  --hypothesis_scales 24 48 \
  --mixed_precision \
  --amp_dtype bf16
