#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-fresh}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-defomstereo_vitl_corga_no_mono_igev_200k_low_lr_20k}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
LATEST_CKPT="${CHECKPOINT_DIR}/checkpoint_latest.pth"
SOURCE_CKPT="checkpoints/defomstereo_vitl_corga_no_mono_igev/defomstereo_vitl_corga_no_mono_igev_200000.pth"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/u2025140689/anaconda3/envs/defomstereo2/bin/python}"
COARSE_LOSS_WEIGHT="${COARSE_LOSS_WEIGHT:-0.2}"
BIN_LOSS_MODE="${BIN_LOSS_MODE:-soft_ce}"
BIN_DISP_SMOOTH_L1_WEIGHT="${BIN_DISP_SMOOTH_L1_WEIGHT:-0.2}"

if [[ "${BIN_LOSS_MODE}" != "soft_ce" && "${BIN_LOSS_MODE}" != "smooth_l1" ]]; then
    echo "BIN_LOSS_MODE must be soft_ce or smooth_l1." >&2
    exit 2
fi

case "${MODE}" in
    fresh)
        if [[ -e "${LATEST_CKPT}" ]]; then
            echo "Refusing to start fresh: ${LATEST_CKPT} already exists." >&2
            exit 1
        fi
        if [[ ! -s "${SOURCE_CKPT}" ]]; then
            echo "Source checkpoint not found: ${SOURCE_CKPT}" >&2
            exit 1
        fi
        RESUME_ARGS=(--resume_ckpt "${SOURCE_CKPT}" --strict_resume --no_resume_optimizer)
        ;;
    resume)
        if [[ ! -s "${LATEST_CKPT}" ]]; then
            echo "Cannot resume: ${LATEST_CKPT} does not exist or is empty." >&2
            exit 1
        fi
        RESUME_ARGS=(--resume_ckpt "${LATEST_CKPT}" --strict_resume)
        ;;
    *)
        echo "Usage: $0 [fresh|resume]" >&2
        exit 2
        ;;
esac

mkdir -p "${CHECKPOINT_DIR}"
echo "mode=${MODE} experiment=${EXPERIMENT_NAME} gpu=${GPU_ID} coarse_loss_weight=${COARSE_LOSS_WEIGHT} bin_loss_mode=${BIN_LOSS_MODE} bin_disp_smooth_l1_weight=${BIN_DISP_SMOOTH_L1_WEIGHT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_stereo.py \
    --model_arch cor_ga_no_mono_igev \
    --dinov2_encoder vitl \
    --launcher none \
    --gpu_ids 0 \
    --name "${EXPERIMENT_NAME}" \
    --batch_size 2 \
    --num_workers 4 \
    --train_datasets sceneflow \
    --train_folds 1 \
    --num_steps 20000 \
    --save_ckpt_freq 5000 \
    --save_latest_ckpt_freq 5000 \
    --n_downsample 2 \
    --train_iters 18 \
    --scale_iters 6 \
    --coarse_loss_weight "${COARSE_LOSS_WEIGHT}" \
    --bin_loss_mode "${BIN_LOSS_MODE}" \
    --bin_disp_smooth_l1_weight "${BIN_DISP_SMOOTH_L1_WEIGHT}" \
    --num_init_bins 48 \
    --init_max_offset 48 \
    --init_corr_scale 10 \
    --init_right_compression avg \
    --loss_gamma 0.9 \
    --smooth_l1_beta 1.0 \
    --max_delta_disp 8.0 \
    --train_max_disp 768 \
    --eval_max_disp 768 \
    --corr_levels 3 \
    --corr_radius 4 \
    --context_norm batch \
    --n_gru_layers 3 \
    --hidden_dims 128 \
    --image_size 256 768 \
    --wdecay 0.00001 \
    --seed 1234565 \
    --val_freq 10000 \
    --valid_iters 32 \
    --amp_dtype bf16 \
    --mixed_precision \
    --lr 0.00001 \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee -a "${CHECKPOINT_DIR}/train.log"
