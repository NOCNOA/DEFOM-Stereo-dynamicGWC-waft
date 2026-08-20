#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-edge}"
MODE="${2:-fresh}"

BASE_CKPT="${BASE_CKPT:-checkpoints/defomstereo_vitl_corga_bins_48_avg_fp32stable/checkpoint_latest.pth}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/u2025140689/anaconda3/envs/defomstereo2/bin/python}"
FINETUNE_STEPS="${FINETUNE_STEPS:-20000}"
FINETUNE_LR="${FINETUNE_LR:-0.00002}"

case "${VARIANT}" in
    control)
        DEFAULT_NAME="defomstereo_vitl_corga_bins_48_avg_ft20k_control"
        EDGE_ARGS=()
        ;;
    edge)
        DEFAULT_NAME="defomstereo_vitl_corga_bins_48_avg_ft20k_gru_edge"
        EDGE_ARGS=(
            --use_gru_edge_loss
            --edge_mode topk
            --edge_topk "${EDGE_TOPK:-0.10}"
            --edge_dilation "${EDGE_DILATION:-5}"
            --gru_edge_lambda "${GRU_EDGE_LAMBDA:-1.0}"
        )
        ;;
    edge_grad)
        DEFAULT_NAME="defomstereo_vitl_corga_bins_48_avg_ft20k_gru_edge_grad"
        EDGE_ARGS=(
            --use_gru_edge_loss
            --edge_mode topk
            --edge_topk "${EDGE_TOPK:-0.10}"
            --edge_dilation "${EDGE_DILATION:-5}"
            --gru_edge_lambda "${GRU_EDGE_LAMBDA:-1.0}"
            --use_gru_disp_grad_loss
            --gru_grad_lambda "${GRU_GRAD_LAMBDA:-0.02}"
        )
        ;;
    *)
        echo "Usage: $0 [control|edge|edge_grad] [fresh|resume]" >&2
        exit 2
        ;;
esac

EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DEFAULT_NAME}}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
LATEST_CKPT="${CHECKPOINT_DIR}/checkpoint_latest.pth"

case "${MODE}" in
    fresh)
        if [[ ! -s "${BASE_CKPT}" ]]; then
            echo "Base checkpoint is missing or empty: ${BASE_CKPT}" >&2
            exit 1
        fi
        if [[ -e "${LATEST_CKPT}" ]]; then
            echo "Refusing to overwrite existing run: ${LATEST_CKPT}" >&2
            echo "Use a different EXPERIMENT_NAME or select resume." >&2
            exit 1
        fi
        RESUME_ARGS=(--resume_ckpt "${BASE_CKPT}" --strict_resume --no_resume_optimizer)
        ;;
    resume)
        if [[ ! -s "${LATEST_CKPT}" ]]; then
            echo "Cannot resume: ${LATEST_CKPT} is missing or empty." >&2
            exit 1
        fi
        RESUME_ARGS=(--resume_ckpt "${LATEST_CKPT}" --strict_resume)
        ;;
    *)
        echo "Usage: $0 [control|edge|edge_grad] [fresh|resume]" >&2
        exit 2
        ;;
esac

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
mkdir -p "${CHECKPOINT_DIR}"
echo "variant=${VARIANT} mode=${MODE} experiment=${EXPERIMENT_NAME}"
echo "checkpoint=${RESUME_ARGS[1]} steps=${FINETUNE_STEPS} lr=${FINETUNE_LR}"

"${PYTHON_BIN}" train_stereo.py \
    --model_arch cor_ga_bins \
    --launcher none \
    --gpu_ids 0 \
    --name "${EXPERIMENT_NAME}" \
    --batch_size 2 \
    --num_workers 2 \
    --train_datasets sceneflow \
    --train_folds 1 \
    --num_steps "${FINETUNE_STEPS}" \
    --save_ckpt_freq 5000 \
    --save_latest_ckpt_freq 1000 \
    --val_freq 5000 \
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
    "${RESUME_ARGS[@]}" \
    "${EDGE_ARGS[@]}" \
    --mixed_precision \
    --amp_dtype fp16 \
    --debug_nan \
    --amp_init_scale 64 \
    --lr "${FINETUNE_LR}" \
    2>&1 | tee -a "${CHECKPOINT_DIR}/train.log"
