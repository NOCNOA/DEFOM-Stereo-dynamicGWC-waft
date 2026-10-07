#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
GPU_ID="${GPU_ID:-0}"
if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]] || (( 10#${GPU_ID} == 1 )); then
    echo "Select one GPU index other than GPU 1." >&2
    exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-/home/u2025140689/anaconda3/envs/defomstereo2/bin/python}"
CHECKPOINT="${CHECKPOINT:-checkpoints/defomstereo_vitl_corga_fast2_match_c24_t28_g4_r18/checkpoint_latest.pth}"
if [[ ! -s "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" evaluate_stereo.py \
    --restore_ckpt "${CHECKPOINT}" --model_arch cor_ga_fast2_match --datasets things \
    --dinov2_encoder vitl --n_downsample 2 --hidden_dims 128 \
    --context_norm batch --n_gru_layers 3 --corr_levels 3 --corr_radius 4 \
    --hypothesis_scales 24 48 96 --max_delta_disp 8 --max_disp 768 \
    --match_feature_dim 24 --match_groups 4 --match_token_dim 28 --match_aggregation_layers 1 \
    --match_right_compression conv --valid_iters 32 --scale_iters 6 \
    --mixed_precision --amp_dtype bf16 "$@"
