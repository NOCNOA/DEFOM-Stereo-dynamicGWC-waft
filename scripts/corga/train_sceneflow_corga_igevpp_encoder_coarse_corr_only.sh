#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
export MODEL_ARCH=cor_ga_stereo_igevpp_encoder_coarse_corr_only
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-defomstereo_corga_igevpp_encoder_coarse_corr_only}"
export ARCHIVE_FULL_CKPT_FREQ="${ARCHIVE_FULL_CKPT_FREQ:-5000}"
exec bash "${SCRIPT_DIR}/train_sceneflow_corga_igevpp_encoder.sh" "$@"
