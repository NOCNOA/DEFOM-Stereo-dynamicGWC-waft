#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

SOURCE_CKPT=checkpoints/defomstereo_corga_igevpp_encoder_coarse_corr_only/checkpoint_full_step_000020000.pth
if [[ ! -s "${SOURCE_CKPT}" ]]; then
    echo "Missing 20k full checkpoint: ${SOURCE_CKPT}" >&2
    exit 1
fi

export GPU_ID=0,2
export GLOBAL_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=1
export NUM_STEPS=25000
export VAL_FREQ=50000
export RESUME_LR_SCALE=1.0
export DIAGNOSTIC_GRAD_FREQ=100
export ARCHIVE_FULL_CKPT_FREQ=5000
export RESUME_CKPT_OVERRIDE="${SOURCE_CKPT}"

for condition in baseline frozen; do
    export EXPERIMENT_NAME="defomstereo_corga_igevpp_encoder_coarse_corr_only_ab_${condition}"
    if [[ -e "checkpoints/${EXPERIMENT_NAME}/checkpoint_latest.pth" ]]; then
        echo "Refusing to overwrite existing experiment: ${EXPERIMENT_NAME}" >&2
        exit 1
    fi
    if [[ "${condition}" == frozen ]]; then
        export FREEZE_STEREO_ENCODER=1
    else
        export FREEZE_STEREO_ENCODER=0
    fi
    echo "Starting ${condition} from 20k on GPUs ${GPU_ID} at original LR" 
    bash "${SCRIPT_DIR}/train_sceneflow_corga_igevpp_encoder_coarse_corr_only.sh" resume
    echo "Completed ${condition}"
done
