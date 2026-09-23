#!/usr/bin/env bash
set -euo pipefail

# CorGA-Lite keeps all three hypotheses, gates, coarse hidden-state fusion and
# the original three-level refinement. Only the coarse path is shortened to
# one recurrent level and (by default) two iterations per hypothesis.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-defomstereo_vitl_corga_lite_s2_bf16}"
RESUME="${RESUME:-1}"
SCALE_ITERS="${SCALE_ITERS:-2}"
TRAIN_ITERS="${TRAIN_ITERS:-18}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.00020}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
PYTHON_BIN="${PYTHON_BIN:-/home/u2025140689/anaconda3/envs/defomstereo2/bin/python}"
mkdir -p "${CHECKPOINT_DIR}"

RESUME_ARGS=()
LATEST_CHECKPOINT="${CHECKPOINT_DIR}/checkpoint_latest.pth"
if [[ "${RESUME}" == "1" && -f "${LATEST_CHECKPOINT}" ]]; then
  RESUME_ARGS=(--resume_ckpt "${LATEST_CHECKPOINT}" --strict_resume)
fi

"${PYTHON_BIN}" train_stereo.py \
  --model_arch cor_ga_lite \
  --launcher none \
  --gpu_ids 0 \
  --name "${EXPERIMENT_NAME}" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --num_workers 2 \
  --train_datasets sceneflow \
  --train_folds 1 \
  --num_steps 200000 \
  --save_ckpt_freq 20000 \
  --save_latest_ckpt_freq 1000 \
  --val_freq 10000 \
  --n_downsample 2 \
  --train_iters "${TRAIN_ITERS}" \
  --scale_iters "${SCALE_ITERS}" \
  --hypothesis_scales 24 48 96 \
  --coarse_loss_weight 0.2 \
  --loss_gamma 0.9 \
  --smooth_l1_beta 1.0 \
  --train_max_disp 768 \
  --eval_max_disp 768 \
  --corr_levels 3 \
  --corr_radius 4 \
  --dinov2_encoder vitl \
  --context_norm batch \
  --n_gru_layers 3 \
  --hidden_dims 128 \
  --image_size 256 768 \
  --wdecay 0.00001 \
  --seed 1234565 \
  --valid_iters 32 \
  --mixed_precision \
  --amp_dtype bf16 \
  --debug_nan \
  --skip_nonfinite_gradients \
  --lr "${LEARNING_RATE}" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "${CHECKPOINT_DIR}/train.log"
