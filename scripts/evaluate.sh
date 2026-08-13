#!/usr/bin/env bash

# evalutate on scene flow

export CUDA_VISIBLE_DEVICES=1
python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vitl_twostage_768/defomstereo_vitl_twostage_768_160000.pth \
--scale_iters 8 \
--valid_iters 40 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets things \
--max_disp 192 \
--dinov2_encoder vitl

#对于kitti12 目前看128disp \ 32iter 视差会有最大的效果  
python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vits_cor_ga2/defomstereo_vits_cor_ga2_250000.pth \
--scale_iters 8 \
--valid_iters 32 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets kitti12 \
--max_disp 96 \
--dinov2_encoder vitl

#对于kitti15 目前看128disp+ 32iter 视差会有最大的效果  
python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vits_cor_ga2/defomstereo_vits_cor_ga2_250000.pth \
--scale_iters 8 \
--valid_iters 32 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets kitti15 \
--max_disp 96 \
--dinov2_encoder vitl

#disp 120  iter 80
python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vits_cor_ga_SR/defomstereo_vits_cor_ga_SR_250000.pth \
--scale_iters 8 \
--valid_iters 32 \
--idepth_scale 0.5 \
--corr_levels 3 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets middlebury_2021 \
--max_disp 48 \
--dinov2_encoder vitl

#disp 150  iter 120
python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vits_cor_ga_onlyFullCorr/defomstereo_vits_cor_ga_onlyFullCorr_250000.pth  \
--scale_iters 8 \
--valid_iters 32 \
--idepth_scale 0.5 \
--corr_levels 3 \
--corr_radius 4 \
--scale_corr_radius 2 \
--datasets middlebury_2021 \
--max_disp 768 \
--dinov2_encoder vitl 

#disp 15  iter 120
python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vits_cor_ga2/defomstereo_vits_cor_ga2_250000.pth  \
--scale_iters 8 \
--valid_iters 32 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets eth3d \
--max_disp 16 \
--dinov2_encoder vitl

