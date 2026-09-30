#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/u2025140689/anaconda3/envs/defomstereo2/bin/python}"
CHECKPOINT="${CHECKPOINT:-checkpoints/defomstereo_corga_igevpp_encoder_coarse_corr_only/checkpoint_latest.pth}"
if [[ ! -s "${CHECKPOINT}" ]]; then
    echo "Checkpoint does not exist or is empty: ${CHECKPOINT}" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" evaluate_stereo.py \
    --restore_ckpt "${CHECKPOINT}" \
    --model_arch cor_ga_stereo_igevpp_encoder_coarse_corr_only \
    --datasets things \
    --corr_levels 3 \
    --corr_radius 4 \
    --n_downsample 2 \
    --context_norm batch \
    --n_gru_layers 3 \
    --hidden_dims 128 \
    --valid_iters 32 \
    --scale_iters 6 \
    --max_delta_disp 8 \
    --max_disp 768 \
    --amp_dtype bf16 \
    --mixed_precision
