#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:-checkpoints/defomstereo_vitl_corga_bins_48_avg_fp32stable/defomstereo_vitl_corga_bins_48_avg_fp32stable_200000.pth}"

if [[ ! -s "${CHECKPOINT}" ]]; then
    echo "Checkpoint does not exist or is empty: ${CHECKPOINT}" >&2
    exit 1
fi

echo "Evaluating SceneFlow/FlyingThings with checkpoint=${CHECKPOINT} gpu=${GPU_ID}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" evaluate_stereo.py \
    --restore_ckpt "${CHECKPOINT}" \
    --model_arch cor_ga_bins \
    --datasets things \
    --dinov2_encoder vitl \
    --corr_levels 3 \
    --corr_radius 4 \
    --num_init_bins 48 \
    --init_max_offset 48 \
    --init_corr_scale 10 \
    --max_delta_disp 8 \
    --n_downsample 2 \
    --context_norm batch \
    --n_gru_layers 3 \
    --hidden_dims 128 \
    --valid_iters 32 \
    --scale_iters 8 \
    --max_disp 768 \
    --mixed_precision
