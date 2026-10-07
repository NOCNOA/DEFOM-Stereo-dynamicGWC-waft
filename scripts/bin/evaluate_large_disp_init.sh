#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
GPU_ID="${GPU_ID:-0}"
if [[ ",${GPU_ID}," == *",1,"* ]]; then
    echo "GPU 1 is prohibited by AGENTS.md." >&2
    exit 1
fi
PYTHON_BIN="${PYTHON_BIN:-/home/u2025140689/anaconda3/envs/defomstereo2/bin/python}"
CHECKPOINT="${CHECKPOINT:-checkpoints/defomstereo_vitl_corga_bins_48_avg_ft20k_control/checkpoint_latest.pth}"
OUTPUT="${OUTPUT:-diagnostics/large_disp_init_100}"
PRECISION_ARGS=()
if [[ "${MIXED_PRECISION:-1}" == 1 ]]; then
    PRECISION_ARGS=(--mixed_precision)
fi
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" scripts/bin/evaluate_large_disp_init.py \
    --restore_ckpt "${CHECKPOINT}" --output "${OUTPUT}" \
    --max_samples "${MAX_SAMPLES:-100}" --valid_iters "${VALID_ITERS:-12}" \
    "${PRECISION_ARGS[@]}" "$@"
