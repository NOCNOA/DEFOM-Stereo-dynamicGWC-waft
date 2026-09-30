#!/usr/bin/env bash
set -euo pipefail

# Fine-tune CorGA-fast3 for 20k SceneFlow steps from the original CorGA 200k weights.
# If this experiment already has a latest checkpoint, resume that fine-tuning run.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-defomstereo_vitl_corga_fast3_from_corga200k_stable}"
RESUME="${RESUME:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.00002}"
COARSE_LOSS_WEIGHT="${COARSE_LOSS_WEIGHT:-0.2}"
NUM_STEPS="${NUM_STEPS:-20000}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-checkpoints/defomstereo_vitl_twostage_768/defomstereo_vitl_twostage_768_200000.pth}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
PYTHON_BIN="${PYTHON_BIN:-/home/u2025140689/anaconda3/envs/defomstereo2/bin/python}"
mkdir -p "${CHECKPOINT_DIR}"

RESUME_ARGS=()
LATEST_CHECKPOINT="${CHECKPOINT_DIR}/checkpoint_latest.pth"
if [[ "${RESUME}" == "1" && -f "${LATEST_CHECKPOINT}" ]]; then
  RESUME_ARGS=(--resume_ckpt "${LATEST_CHECKPOINT}" --strict_resume)
else
  if [[ ! -f "${PRETRAINED_CKPT}" ]]; then
    echo "Pretrained checkpoint not found: ${PRETRAINED_CKPT}" >&2
    exit 1
  fi
  RESUME_ARGS=(--resume_ckpt "${PRETRAINED_CKPT}" --strict_resume --no_resume_optimizer)
fi

"${PYTHON_BIN}" train_stereo.py \
  --model_arch cor_ga_fast3 \
  --launcher none \
  --gpu_ids 0 \
  --name "${EXPERIMENT_NAME}" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --num_workers 2 \
  --train_datasets sceneflow \
  --train_folds 1 \
  --num_steps "${NUM_STEPS}" \
  --save_ckpt_freq 1000 \
  --save_latest_ckpt_freq 500 \
  --val_freq 50000 \
  --n_downsample 2 \
  --train_iters 18 \
  --scale_iters 6 \
  --hypothesis_scales 24 48 96 \
  --coarse_loss_weight "${COARSE_LOSS_WEIGHT}" \
  --loss_gamma 0.9 \
  --smooth_l1_beta 1.0 \
  --train_max_disp 768 \
  --eval_max_disp 768 \
  --corr_levels 3 \
  --corr_radius 4 \
  --dinov2_encoder vitl \
  --mixed_precision \
  --amp_dtype bf16 \
  --debug_nan \
  --skip_nonfinite_gradients \
  --large_grad_warn_threshold 1000000 \
  --lr "${LEARNING_RATE}" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "${CHECKPOINT_DIR}/train.log"
