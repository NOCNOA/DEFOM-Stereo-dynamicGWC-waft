#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

export GPU_ID=2,3
export GLOBAL_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=1
export NUM_STEPS=25000
export VAL_FREQ=50000
export RESUME_LR_SCALE=1.0
export RESUME_BATCH_SKIP=0
export DIAGNOSTIC_GRAD_FREQ=100
export ARCHIVE_FULL_CKPT_FREQ=1000
export SAVE_LATEST_CKPT_FREQ=1000
export FREEZE_STEREO_ENCODER=1
export FREEZE_COARSE_GRU_MOTION=1
export FREEZE_REFINE_UPDATE_PROJECTION=0
export RESUME_CKPT_OVERRIDE=checkpoints/defomstereo_corga_igevpp_encoder_coarse_corr_only/checkpoint_full_step_000020000.pth

if [[ ! -s "${RESUME_CKPT_OVERRIDE}" ]]; then
    echo "Missing 20k full checkpoint: ${RESUME_CKPT_OVERRIDE}" >&2
    exit 1
fi

for condition in projection update; do
    export FREEZE_REFINE_PROJECTION=0
    export FREEZE_REFINE_UPDATE=0
    if [[ "${condition}" == projection ]]; then
        export FREEZE_REFINE_PROJECTION=1
    else
        export FREEZE_REFINE_UPDATE=1
    fi
    export EXPERIMENT_NAME="defomstereo_corga_igevpp_encoder_coarse_corr_only_freeze_encoder_coarse_${condition}_only_20to25_full1k"
    if [[ -e "checkpoints/${EXPERIMENT_NAME}/checkpoint_latest.pth" ]]; then
        echo "Refusing to overwrite existing experiment: ${EXPERIMENT_NAME}" >&2
        exit 1
    fi
    echo "Continuing ${condition}-only refine freeze from 20k to 25k on GPUs ${GPU_ID}"
    bash "${SCRIPT_DIR}/train_sceneflow_corga_igevpp_encoder_coarse_corr_only.sh" resume
    echo "Completed ${condition}-only refine freeze at 25k"
done
