#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

export GLOBAL_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=2
export NUM_STEPS=200000
export STOP_AFTER_STEP=25000
export VAL_FREQ=50000
export SAVE_CKPT_FREQ=5000
export SAVE_LATEST_CKPT_FREQ=1000
export ARCHIVE_FULL_CKPT_FREQ=1000
export ARCHIVE_FULL_CKPT_START_STEP=20000
export PAIRED_ENCODER_INIT=1
export DINOV2_ENCODER=vitl
export FREEZE_STEREO_ENCODER=0
export FREEZE_COARSE_GRU_MOTION=0
export FREEZE_REFINE_UPDATE_PROJECTION=0
export FREEZE_REFINE_UPDATE=0
export FREEZE_REFINE_PROJECTION=0
export FREEZE_REFINE_COMPONENT=

for specification in igevpp:2 dinov2:3; do
    encoder="${specification%%:*}"
    gpu="${specification##*:}"
    experiment="defomstereo_corga_encoder_ab_${encoder}_gpu${gpu}_fresh25k"
    if [[ -e "checkpoints/${experiment}/checkpoint_latest.pth" ]]; then
        echo "Refusing to overwrite existing experiment: ${experiment}" >&2
        exit 1
    fi
done

MATCH_ENCODER=igevpp GPU_ID=2 \
    EXPERIMENT_NAME=defomstereo_corga_encoder_ab_igevpp_gpu2_fresh25k \
    bash "${SCRIPT_DIR}/train_sceneflow_corga_igevpp_encoder_coarse_corr_only.sh" fresh \
    > nohups/corga/match_encoder_ab_igevpp_gpu2.out 2>&1 &
igevpp_pid=$!
MATCH_ENCODER=dinov2 GPU_ID=3 \
    EXPERIMENT_NAME=defomstereo_corga_encoder_ab_dinov2_gpu3_fresh25k \
    bash "${SCRIPT_DIR}/train_sceneflow_corga_igevpp_encoder_coarse_corr_only.sh" fresh \
    > nohups/corga/match_encoder_ab_dinov2_gpu3.out 2>&1 &
dinov2_pid=$!
echo "Started IGEV++ on GPU 2 (PID ${igevpp_pid}) and DINOv2 on GPU 3 (PID ${dinov2_pid})"

status=0
if ! wait "${igevpp_pid}"; then
    echo "IGEV++ arm failed; see nohups/corga/match_encoder_ab_igevpp_gpu2.out" >&2
    status=1
fi
if ! wait "${dinov2_pid}"; then
    echo "DINOv2 arm failed; see nohups/corga/match_encoder_ab_dinov2_gpu3.out" >&2
    status=1
fi
if (( status != 0 )); then
    exit "${status}"
fi

for specification in igevpp:2 dinov2:3; do
    encoder="${specification%%:*}"
    gpu="${specification##*:}"
    experiment="defomstereo_corga_encoder_ab_${encoder}_gpu${gpu}_fresh25k"
    eval_args=(--output "diagnostics/match_encoder_ab_fixed128/${encoder}"
               --num-images 128 --match-encoder "${encoder}")
    checkpoint_names=()
    for step in 20000 21000 22000 23000 24000 25000; do
        path="checkpoints/${experiment}/checkpoint_full_step_$(printf '%09d' "${step}").pth"
        if [[ ! -s "${path}" ]]; then
            echo "Missing A/B checkpoint: ${path}" >&2
            exit 1
        fi
        eval_args+=(--extra-checkpoint "step_${step}=${path}")
        checkpoint_names+=("step_${step}")
    done
    CUDA_VISIBLE_DEVICES=3 /home/u2025140689/anaconda3/envs/defomstereo2/bin/python \
        "${SCRIPT_DIR}/evaluate_encoder_freeze_ab_fixed.py" "${eval_args[@]}" \
        --checkpoints "${checkpoint_names[@]}"
done
