#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

export GPU_ID=3
export GLOBAL_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=2
export NUM_STEPS=30000
export VAL_FREQ=20000
export RESUME_LR_SCALE=0.25
export DIAGNOSTIC_GRAD_FREQ=100
export RESUME_CKPT_OVERRIDE=checkpoints/defomstereo_corga_igevpp_encoder_coarse_corr_only/checkpoint_full_step_000020000.pth
export EXPERIMENT_NAME=defomstereo_corga_igevpp_encoder_coarse_corr_only_gpu3_lrquarter_from20k
export ARCHIVE_FULL_CKPT_FREQ=5000

exec bash "${SCRIPT_DIR}/train_sceneflow_corga_igevpp_encoder_coarse_corr_only.sh" resume
