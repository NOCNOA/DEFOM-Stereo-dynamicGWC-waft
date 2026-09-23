#!/usr/bin/env bash
set -euo pipefail

# Train CorGA with one GRU layer per coarse disparity branch. Refinement keeps
# all three GRU layers, while its 1/8 and 1/16 states come from the context
# encoder. This experiment uses its own checkpoint directory.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-defomstereo_vitl_corga_coarse1_bf16}"
RESUME="${RESUME:-1}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
PYTHON_BIN="${PYTHON_BIN:-/home/u2025140689/anaconda3/envs/defomstereo2/bin/python}"
mkdir -p "${CHECKPOINT_DIR}"

RESUME_ARGS=()
LATEST_CHECKPOINT="${CHECKPOINT_DIR}/checkpoint_latest.pth"
if [[ "${RESUME}" == "1" && -f "${LATEST_CHECKPOINT}" ]]; then
  RESUME_ARGS=(--resume_ckpt "${LATEST_CHECKPOINT}" --strict_resume)
fi

"${PYTHON_BIN}" train_stereo.py \
  --model_arch cor_ga_coarse1 \
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
  --hypothesis_scales 24 48 96 \
  --coarse_loss_weight 0.2 \
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
  --lr 0.00020 \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "${CHECKPOINT_DIR}/train.log"
