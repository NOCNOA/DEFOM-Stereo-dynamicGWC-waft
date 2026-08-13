#!/usr/bin/env bash

# trained on 2 x 24GB 3090/4090 GPUs
# export CUDA_VISIBLE_DEVICES=0,2,3
# CHECKPOINT_DIR=checkpoints/defomstereo_vits_patchga_6batch && \
# mkdir -p ${CHECKPOINT_DIR} && \
# python -m torch.distributed.launch --nproc_per_node=3 --master_port=9990 train_stereo.py \
# --distributed \
# --launcher pytorch \
# --gpu_ids 0 1 2   \
# --name defomstereo_vits_patchga_6batch \
# --batch_size 6  \
# --num_workers 6  \
# --train_datasets sceneflow \
# --train_folds 1 \
# --num_steps 300000 \
# --mixed_precision \
# --n_downsample 2 \
# --train_iters 18 \
# --scale_iters 8 \
# --idepth_scale 0.5 \
# --corr_levels 2 \
# --corr_radius 4 \
# --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
# --scale_corr_radius 2 \
# --dinov2_encoder vits \
# --lr 0.00030 \
#--resume_ckpt checkpoints/defomstereo_vits_patchga_6batch/checkpoint_latest.pth \
# 2>&1 | tee -a ${CHECKPOINT_DIR}/train.log


# CHECKPOINT_DIR=checkpoints/defomstereo_vits_PRU && \
# mkdir -p ${CHECKPOINT_DIR} && \
# python -m torch.distributed.launch --nproc_per_node=1 --master_port=9990 train_stereo.py \
# --distributed \
# --launcher pytorch \
# --gpu_ids 0   \
# --name defomstereo_vits_PRU \
# --batch_size 2  \
# --num_workers 2  \
# --train_datasets sceneflow \
# --train_folds 1 \
# --num_steps 250000 \
# --mixed_precision \
# --n_downsample 2 \
# --train_iters 18 \
# --scale_iters 0 \
# --idepth_scale 0.5 \
# --corr_levels 2 \
# --corr_radius 4 \
# --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
# --scale_corr_radius 2 \
# --dinov2_encoder vits \
# --lr 0.00025 \
# --resume_ckpt checkpoints/defomstereo_vits_PRU/checkpoint_latest.pth \
# 2>&1 | tee -a ${CHECKPOINT_DIR}/train.log

# CHECKPOINT_DIR=checkpoints/defomstereo_vits_onlygwc && \
# mkdir -p ${CHECKPOINT_DIR} && \
# python -m torch.distributed.launch --nproc_per_node=1 --master_port=9990 train_stereo.py \
# --distributed \
# --launcher pytorch \
# --gpu_ids 0   \
# --name defomstereo_vits_onlygwc \
# --batch_size 2  \
# --num_workers 2  \
# --train_datasets sceneflow \
# --train_folds 1 \
# --num_steps 250000 \
# --mixed_precision \
# --n_downsample 2 \
# --train_iters 18 \
# --scale_iters 8 \
# --idepth_scale 0.5 \
# --corr_levels 2 \
# --corr_radius 4 \
# --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
# --scale_corr_radius 2 \
# --dinov2_encoder vits \
# --lr 0.00025 \
# 2>&1 | tee -a ${CHECKPOINT_DIR}/train.log

# trained on 2 x 24GB 3090/4090 GPUs
# export CUDA_VISIBLE_DEVICES=0,2,3
# CHECKPOINT_DIR=checkpoints/defomstereo_vits_patchga_6batch_NormalGru && \
# mkdir -p ${CHECKPOINT_DIR} && \
# python -m torch.distributed.launch --nproc_per_node=3 --master_port=9990 train_stereo.py \
# --distributed \
# --launcher pytorch \
# --gpu_ids 0 1 2   \
# --name defomstereo_vits_patchga_6batch_NormalGru \
# --batch_size 6  \
# --num_workers 6  \
# --train_datasets sceneflow \
# --train_folds 1 \
# --num_steps 200000 \
# --mixed_precision \
# --n_downsample 2 \
# --train_iters 18 \
# --scale_iters 8 \
# --idepth_scale 0.5 \
# --corr_levels 2 \
# --corr_radius 4 \
# --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
# --scale_corr_radius 2 \
# --dinov2_encoder vits \
# --lr 0.00020 \
# 2>&1 | tee -a ${CHECKPOINT_DIR}/train.log


export CUDA_VISIBLE_DEVICES=0
CHECKPOINT_DIR=checkpoints/defomstereo_vitl_twostage_768 && \
mkdir -p ${CHECKPOINT_DIR} && \
python train_stereo.py \
--launcher none \
--gpu_ids 0    \
--name defomstereo_vitl_twostage_768 \
--batch_size 2  \
--num_workers 2  \
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
--train_max_disp 768 \
--eval_max_disp 768 \
--corr_levels 3 \
--resume_ckpt ${CHECKPOINT_DIR}/checkpoint_latest.pth \
--mixed_precision \
--debug_nan \
--corr_radius 4 \
--dinov2_encoder vitl \
--lr 0.00020 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log
