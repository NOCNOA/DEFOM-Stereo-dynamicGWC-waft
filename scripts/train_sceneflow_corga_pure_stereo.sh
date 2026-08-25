#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-fresh}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-defomstereo_pure_stereo_corga_s48_96_128}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/${EXPERIMENT_NAME}}"
LATEST_CKPT="${CHECKPOINT_DIR}/checkpoint_latest.pth"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "${MODE}" in
    fresh)
        if [[ -e "${LATEST_CKPT}" ]]; then
            echo "Refusing to start fresh: ${LATEST_CKPT} already exists." >&2
            echo "Run '$0 resume' to continue this experiment." >&2
            exit 1
        fi
        RESUME_ARGS=()
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
echo "mode=${MODE} experiment=${EXPERIMENT_NAME} gpu=${GPU_ID}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_stereo.py \
--model_arch cor_ga_no_mono_head \
--launcher none \
--gpu_ids 0 \
--name "${EXPERIMENT_NAME}" \
--batch_size 2 \
--num_workers 2 \
--train_datasets sceneflow \
--train_folds 1 \
--num_steps 200000 \
--save_ckpt_freq 20000 \
--save_latest_ckpt_freq 1000 \
--n_downsample 2 \
--train_iters 18 \
--scale_iters 6 \
--hypothesis_scales 48 96 128 \
--coarse_loss_weight 0.2 \
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
"${RESUME_ARGS[@]}" \
--mixed_precision \
--debug_nan \
--lr 0.00020 \
2>&1 | tee -a "${CHECKPOINT_DIR}/train.log"
