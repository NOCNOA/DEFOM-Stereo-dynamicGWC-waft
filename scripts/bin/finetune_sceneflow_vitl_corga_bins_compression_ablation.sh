#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-p0}"
MODE="${2:-fresh}"

BASE_CKPT="${BASE_CKPT:-checkpoints/defomstereo_vitl_corga_bins_48_avg_ft20k_gru_edge/checkpoint_latest.pth}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/u2025140689/anaconda3/envs/defomstereo2/bin/python}"
FINETUNE_STEPS="${FINETUNE_STEPS:-10000}"
FINETUNE_LR="${FINETUNE_LR:-0.00002}"
SEED="${SEED:-1234565}"

case "${VARIANT}" in
    p0)
        INIT_COMPRESSION="avg"
        REFINE_COMPRESSION="avg"
        ;;
    p1)
        INIT_COMPRESSION="conv"
        REFINE_COMPRESSION="avg"
        ;;
    p2)
        INIT_COMPRESSION="avg"
        REFINE_COMPRESSION="conv"
        ;;
    p3)
        INIT_COMPRESSION="conv"
        REFINE_COMPRESSION="conv"
        ;;
    *)
        echo "Usage: $0 [p0|p1|p2|p3] [fresh|resume]" >&2
        exit 2
        ;;
esac

DEFAULT_NAME="defomstereo_vitl_corga_bins_compression_${VARIANT}_ft${FINETUNE_STEPS}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DEFAULT_NAME}}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
LATEST_CKPT="${CHECKPOINT_DIR}/checkpoint_latest.pth"

case "${MODE}" in
    fresh)
        if [[ ! -s "${BASE_CKPT}" ]]; then
            echo "Base checkpoint is missing or empty: ${BASE_CKPT}" >&2
            echo "Set BASE_CKPT to the common avg/avg edge-loss checkpoint." >&2
            exit 1
        fi
        if [[ -e "${LATEST_CKPT}" ]]; then
            echo "Refusing to overwrite existing run: ${LATEST_CKPT}" >&2
            echo "Use a different EXPERIMENT_NAME or select resume." >&2
            exit 1
        fi
        # P0 has exactly the old avg/avg architecture. Conv variants add new
        # compressor parameters, so their common avg checkpoint is loaded
        # non-strictly while all optimizer/scheduler state is reset.
        if [[ "${VARIANT}" == "p0" ]]; then
            RESUME_ARGS=(--resume_ckpt "${BASE_CKPT}" --strict_resume --no_resume_optimizer)
        else
            RESUME_ARGS=(--resume_ckpt "${BASE_CKPT}" --no_resume_optimizer)
        fi
        ;;
    resume)
        if [[ ! -s "${LATEST_CKPT}" ]]; then
            echo "Cannot resume: ${LATEST_CKPT} is missing or empty." >&2
            exit 1
        fi
        RESUME_ARGS=(--resume_ckpt "${LATEST_CKPT}" --strict_resume)
        ;;
    *)
        echo "Usage: $0 [p0|p1|p2|p3] [fresh|resume]" >&2
        exit 2
        ;;
esac

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
mkdir -p "${CHECKPOINT_DIR}"

echo "variant=${VARIANT} mode=${MODE} experiment=${EXPERIMENT_NAME}"
echo "init_compression=${INIT_COMPRESSION} refine_compression=${REFINE_COMPRESSION}"
echo "checkpoint=${RESUME_ARGS[1]} steps=${FINETUNE_STEPS} lr=${FINETUNE_LR} seed=${SEED}"

"${PYTHON_BIN}" train_stereo.py \
    --model_arch cor_ga_bins \
    --init_right_compression "${INIT_COMPRESSION}" \
    --refine_right_compression "${REFINE_COMPRESSION}" \
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
    --use_gru_edge_loss \
    --edge_mode topk \
    --edge_topk "${EDGE_TOPK:-0.10}" \
    --edge_dilation "${EDGE_DILATION:-5}" \
    --gru_edge_lambda "${GRU_EDGE_LAMBDA:-1.0}" \
    --mixed_precision \
    --amp_dtype fp16 \
    --debug_nan \
    --amp_init_scale 64 \
    --lr "${FINETUNE_LR}" \
    --seed "${SEED}" \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee -a "${CHECKPOINT_DIR}/train.log"
