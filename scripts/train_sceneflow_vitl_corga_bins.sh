#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-fresh}"
EXPERIMENT_NAME="defomstereo_vitl_corga_bins_no_rope"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
LATEST_CKPT="${CHECKPOINT_DIR}/checkpoint_latest.pth"

case "${MODE}" in
    fresh)
        if [[ -e "${LATEST_CKPT}" ]]; then
            echo "Refusing to start fresh: ${LATEST_CKPT} already exists." >&2
            echo "Run with 'resume' to continue this fixed experiment." >&2
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
echo "mode=${MODE} experiment=${EXPERIMENT_NAME} gpu=0"
CUDA_VISIBLE_DEVICES=0 python train_stereo.py \
--model_arch cor_ga_bins2 \
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
--num_init_bins 48 \
--init_max_offset 48 \
--init_corr_scale 10 \
--loss_gamma 0.9 \
--init_target_temperature 1.0 \
--gate_loss_weight 0.1 \
--gate_target_temperature 4.0 \
--smooth_l1_beta 1.0 \
--d0_loss_weight 1.0 \
--max_delta_disp 8.0 \
--train_max_disp 768 \
--eval_max_disp 768 \
--corr_levels 3 \
--corr_radius 4 \
--dinov2_encoder vitl \
--init_right_compression avg \
--refine_right_compression avg \
--context_norm batch \
--n_gru_layers 3 \
--hidden_dims 128 \
--image_size 256 768 \
--wdecay 0.00001 \
--seed 1234565 \
--val_freq 10000 \
--valid_iters 32 \
--amp_dtype fp16 \
"${RESUME_ARGS[@]}" \
--mixed_precision \
--debug_nan \
--amp_init_scale 64 \
--lr 0.00020 \
2>&1 | tee -a "${CHECKPOINT_DIR}/train.log"
