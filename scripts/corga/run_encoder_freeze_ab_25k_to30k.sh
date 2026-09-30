#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

export GPU_ID=2,3
export GLOBAL_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=1
export NUM_STEPS=30000
export VAL_FREQ=50000
export RESUME_LR_SCALE=1.0
export RESUME_BATCH_SKIP=5000
export DIAGNOSTIC_GRAD_FREQ=100
export ARCHIVE_FULL_CKPT_FREQ=5000

for condition in baseline frozen; do
    source_experiment="defomstereo_corga_igevpp_encoder_coarse_corr_only_ab_${condition}"
    export RESUME_CKPT_OVERRIDE="checkpoints/${source_experiment}/checkpoint_full_step_000025000.pth"
    export EXPERIMENT_NAME="${source_experiment}_25to30_gpu23"
    if [[ ! -s "${RESUME_CKPT_OVERRIDE}" ]]; then
        echo "Missing 25k full checkpoint: ${RESUME_CKPT_OVERRIDE}" >&2
        exit 1
    fi
    if [[ -e "checkpoints/${EXPERIMENT_NAME}/checkpoint_latest.pth" ]]; then
        echo "Refusing to overwrite existing experiment: ${EXPERIMENT_NAME}" >&2
        exit 1
    fi
    if [[ "${condition}" == frozen ]]; then
        export FREEZE_STEREO_ENCODER=1
    else
        export FREEZE_STEREO_ENCODER=0
    fi
    echo "Continuing ${condition} from 25k to 30k on GPUs ${GPU_ID}"
    bash "${SCRIPT_DIR}/train_sceneflow_corga_igevpp_encoder_coarse_corr_only.sh" resume
    echo "Completed ${condition} at 30k"
done
