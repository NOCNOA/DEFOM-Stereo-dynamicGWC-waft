#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
CHECKPOINT_DIR=checkpoints/defomstereo_vitl_corga_bins_48_avg
LATEST_CKPT=${CHECKPOINT_DIR}/checkpoint_latest.pth
if [[ -s "${LATEST_CKPT}" ]]; then
    RESUME_ARGS=(--resume_ckpt "${LATEST_CKPT}" --strict_resume)
else
    RESUME_ARGS=()
fi

mkdir -p "${CHECKPOINT_DIR}"
python train_stereo.py \
--model_arch cor_ga_bins \
--launcher none \
--gpu_ids 0 \
--name defomstereo_vitl_corga_bins_48_avg \
--batch_size 1 \
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
"${RESUME_ARGS[@]}" \
--mixed_precision \
--amp_init_scale 64 \
--lr 0.00020 \
2>&1 | tee -a "${CHECKPOINT_DIR}/train.log"
