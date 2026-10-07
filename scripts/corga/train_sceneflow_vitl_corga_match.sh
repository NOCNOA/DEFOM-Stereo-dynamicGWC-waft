#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
MODE="${1:-fresh}"
if (( $# )); then shift; fi
GPU_ID="${GPU_ID:-0}"
if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]] || (( 10#${GPU_ID} == 1 )); then
    echo "Select one GPU index other than GPU 1." >&2
    exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-/home/u2025140689/anaconda3/envs/defomstereo2/bin/python}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-defomstereo_vitl_corga_match_g4_r12}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
LATEST_CKPT="${CHECKPOINT_DIR}/checkpoint_latest.pth"
case "${MODE}" in
    fresh)
        if [[ -e "${LATEST_CKPT}" ]]; then
            echo "Checkpoint already exists; use resume or a new EXPERIMENT_NAME." >&2
            exit 1
        fi
        RESUME_ARGS=()
        ;;
    resume)
        if [[ ! -s "${LATEST_CKPT}" ]]; then
            echo "Missing checkpoint: ${LATEST_CKPT}" >&2
            exit 1
        fi
        RESUME_ARGS=(--resume_ckpt "${LATEST_CKPT}" --strict_resume)
        ;;
    *) echo "Usage: $0 [fresh|resume] [training arguments...]" >&2; exit 2 ;;
esac
mkdir -p "${CHECKPOINT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_stereo.py \
    --model_arch cor_ga_match --launcher none --gpu_ids 0 \
    --name "${EXPERIMENT_NAME}" --dinov2_encoder vitl \
    --batch_size "${BATCH_SIZE:-2}" --num_workers 4 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-3}" \
    --train_datasets sceneflow --train_folds 1 --num_steps 200000 \
    --save_ckpt_freq 20000 --save_latest_ckpt_freq 5000 \
    --n_downsample 2 --train_iters 12 --scale_iters 6 \
    --hypothesis_scales 24 48 96 --coarse_loss_weight 0.2 \
    --match_groups 4 --match_token_dim 32 --match_aggregation_layers 1 \
    --match_right_compression conv --corr_levels 3 --corr_radius 4 \
    --context_norm batch --n_gru_layers 3 --hidden_dims 128 \
    --loss_gamma 0.9 --smooth_l1_beta 1 --max_delta_disp 8 \
    --train_max_disp 768 --eval_max_disp 768 --image_size 256 768 \
    --wdecay 0.00001 --seed 1234565 --val_freq 20000 --valid_iters 32 \
    --mixed_precision --amp_dtype bf16 --debug_nan --lr 0.00020 \
    "${RESUME_ARGS[@]}" "$@" \
    2>&1 | tee -a "${CHECKPOINT_DIR}/train.log"
