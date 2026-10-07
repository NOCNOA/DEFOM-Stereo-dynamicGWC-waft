#!/usr/bin/env bash
set -euo pipefail

# Disable the update limit in both stages; retain the final disparity bounds.
MODE="${1:-fresh}"
if (( $# > 0 )); then shift; fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-defomstereo_vitl_corga_fast2_no_delta_clamp_ft20k}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
LATEST_CKPT="${CHECKPOINT_DIR}/checkpoint_latest.pth"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-checkpoints/defomstereo_vitl_corga_fast2_scratch/defomstereo_vitl_corga_fast2_scratch_200000.pth}"
GPU_ID="${GPU_ID:-0}"
if [[ "${GPU_ID}" == "1" || ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "GPU_ID must be a single GPU index other than 1 (repository policy)." >&2
    exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-/home/u2025140689/anaconda3/envs/defomstereo2/bin/python}"

case "${MODE}" in
    fresh)
        if [[ -e "${LATEST_CKPT}" ]]; then
            echo "Fine-tuning checkpoint already exists: ${LATEST_CKPT}" >&2
            echo "Use resume to continue, or set a new EXPERIMENT_NAME." >&2
            exit 1
        fi
        if [[ ! -s "${PRETRAINED_CKPT}" ]]; then
            echo "Pretrained checkpoint not found: ${PRETRAINED_CKPT}" >&2
            exit 1
        fi
        # Load only weights, resetting the optimizer, scheduler and step count.
        RESUME_ARGS=(--resume_ckpt "${PRETRAINED_CKPT}" --strict_resume --no_resume_optimizer)
        ;;
    resume)
        if [[ ! -s "${LATEST_CKPT}" ]]; then
            echo "Fine-tuning checkpoint not found: ${LATEST_CKPT}" >&2
            exit 1
        fi
        RESUME_ARGS=(--resume_ckpt "${LATEST_CKPT}" --strict_resume)
        ;;
    *)
        echo "Usage: $0 [fresh|resume] [extra train_stereo.py arguments]" >&2
        exit 2
        ;;
esac

mkdir -p "${CHECKPOINT_DIR}"
TRAIN_COMMAND=(
    "${PYTHON_BIN}" train_stereo.py
    --model_arch cor_ga_fast2 --dinov2_encoder vitl
    --launcher none --gpu_ids 0 --name "${EXPERIMENT_NAME}"
    --batch_size "${BATCH_SIZE:-2}" --num_workers "${NUM_WORKERS:-4}"
    --gradient_accumulation_steps 1
    --train_datasets sceneflow --train_folds 1
    --num_steps "${NUM_STEPS:-20000}"
    --save_ckpt_freq 1000 --save_latest_ckpt_freq 500
    --val_freq "${VAL_FREQ:-5000}" --valid_iters 32
    --n_downsample 2 --train_iters "${TRAIN_ITERS:-18}" --scale_iters 6
    --hypothesis_scales 24 48 96 --coarse_loss_weight 0.2
    --loss_gamma 0.9 --smooth_l1_beta 1.0 --max_delta_disp 8.0
    --train_max_disp 768 --eval_max_disp 768
    --corr_levels 3 --corr_radius 4 --context_norm batch
    --n_gru_layers 3 --hidden_dims 128 --image_size 256 768
    --wdecay 0.00001 --seed 1234565
    --amp_dtype bf16 --mixed_precision
    --lr "${LEARNING_RATE:-0.00002}"
    "${RESUME_ARGS[@]}" "$@"
    --disable_delta_clamp
)
{
    echo "mode=${MODE} experiment=${EXPERIMENT_NAME} gpu=${GPU_ID}"
    printf '%q ' "${TRAIN_COMMAND[@]}"
    printf '\n'
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${TRAIN_COMMAND[@]}"
} 2>&1 | tee -a "${CHECKPOINT_DIR}/train.log"
