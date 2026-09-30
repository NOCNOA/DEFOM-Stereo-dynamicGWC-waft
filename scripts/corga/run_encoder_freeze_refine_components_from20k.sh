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
export FREEZE_REFINE_UPDATE=0
export FREEZE_REFINE_PROJECTION=0
export RESUME_CKPT_OVERRIDE=checkpoints/defomstereo_corga_igevpp_encoder_coarse_corr_only/checkpoint_full_step_000020000.pth

if [[ ! -s "${RESUME_CKPT_OVERRIDE}" ]]; then
    echo "Missing 20k full checkpoint: ${RESUME_CKPT_OVERRIDE}" >&2
    exit 1
fi

for component in gru_layers motion_encoder; do
    export FREEZE_REFINE_COMPONENT="${component}"
    export EXPERIMENT_NAME="defomstereo_corga_igevpp_encoder_coarse_corr_only_freeze_encoder_coarse_refine_${component}_20to25_full1k"
    if [[ -e "checkpoints/${EXPERIMENT_NAME}/checkpoint_latest.pth" ]]; then
        echo "Refusing to overwrite existing experiment: ${EXPERIMENT_NAME}" >&2
        exit 1
    fi
    echo "Continuing refine ${component} freeze from 20k to 25k on GPUs ${GPU_ID}"
    bash "${SCRIPT_DIR}/train_sceneflow_corga_igevpp_encoder_coarse_corr_only.sh" resume
    echo "Completed refine ${component} freeze at 25k"
done

eval_args=(--output diagnostics/encoder_freeze_refine_components_fixed128
           --num-images 128)
checkpoint_names=(source20)
for component in gru_layers motion_encoder; do
    experiment="defomstereo_corga_igevpp_encoder_coarse_corr_only_freeze_encoder_coarse_refine_${component}_20to25_full1k"
    for step in 21000 22000 23000 24000 25000; do
        path="checkpoints/${experiment}/checkpoint_full_step_$(printf '%09d' "${step}").pth"
        if [[ ! -s "${path}" ]]; then
            echo "Missing experiment checkpoint: ${path}" >&2
            exit 1
        fi
        eval_args+=(--extra-checkpoint "${component}_${step}=${path}")
        checkpoint_names+=("${component}_${step}")
    done
done

CUDA_VISIBLE_DEVICES=3 /home/u2025140689/anaconda3/envs/defomstereo2/bin/python \
    "${SCRIPT_DIR}/evaluate_encoder_freeze_ab_fixed.py" "${eval_args[@]}" \
    --checkpoints "${checkpoint_names[@]}"
