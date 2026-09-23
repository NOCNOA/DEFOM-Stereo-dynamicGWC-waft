#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-defomstereo_vitl_corga_fast_binmono_corrstats_5k}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LEARNING_RATE="${LEARNING_RATE:-0.000002}"
NUM_STEPS="${NUM_STEPS:-5000}"
SCALE_ITERS="${SCALE_ITERS:-6}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-checkpoints/defomstereo_vitl_corga_fast_binmono_joint_more20k/20000_epoch_defomstereo_vitl_corga_fast_binmono_joint_more20k.pth.gz}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
PYTHON_BIN="${PYTHON_BIN:-/home/u2025140689/anaconda3/envs/defomstereo2/bin/python}"
mkdir -p "${CHECKPOINT_DIR}"

if [[ ! -f "${PRETRAINED_CKPT}" ]]; then
  echo "Pretrained checkpoint not found: ${PRETRAINED_CKPT}" >&2
  exit 1
fi

"${PYTHON_BIN}" train_stereo.py \
  --model_arch cor_ga_fast_binmono_corrstats \
  --launcher none \
  --gpu_ids 0 \
  --name "${EXPERIMENT_NAME}" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps 1 \
  --num_workers 2 \
  --train_datasets sceneflow \
  --train_folds 1 \
  --num_steps "${NUM_STEPS}" \
  --save_ckpt_freq 5000 \
  --save_latest_ckpt_freq 500 \
  --val_freq 5000 \
  --n_downsample 2 \
  --train_iters 18 \
  --valid_iters 32 \
  --scale_iters "${SCALE_ITERS}" \
  --coarse_loss_weight 0.2 \
  --num_init_bins 48 \
  --init_max_offset 48 \
  --init_corr_scale 10 \
  --init_right_compression avg \
  --binmono_conf_hidden 32 \
  --binmono_init_stereo_weight 0.05 \
  --corr_stats_hidden 32 \
  --corr_stats_temperature 1.0 \
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
  --resume_ckpt "${PRETRAINED_CKPT}" \
  --no_resume_optimizer \
  2>&1 | tee -a "${CHECKPOINT_DIR}/train.log"
