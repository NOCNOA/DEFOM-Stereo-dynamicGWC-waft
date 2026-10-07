#!/usr/bin/env bash
set -euo pipefail

# Paired continuation from one bin checkpoint. GPU 1 is deliberately excluded.
arm="${1:?Usage: $0 baseline|wavelet GPU_ID}"
gpu_id="${2:?Usage: $0 baseline|wavelet GPU_ID}"
case "$arm" in
  baseline) model_arch=cor_ga_bins ;;
  wavelet) model_arch=cor_ga_bins_wavelet ;;
  *) echo "Unknown arm: $arm" >&2; exit 2 ;;
esac
if [[ "$gpu_id" != 0 && "$gpu_id" != 2 && "$gpu_id" != 3 ]]; then
  echo "Use GPU 0, 2, or 3; GPU 1 is reserved." >&2
  exit 2
fi

cd "$(dirname "$0")/../.."
source_ckpt=checkpoints/defomstereo_vitl_corga_bins_48_avg_ft20k_control/20000_epoch_defomstereo_vitl_corga_bins_48_avg_ft20k_control.pth.gz
run_name="bins_wavelet_pair_20261005_${arm}_2k_b1a2"
if [[ ! -s "$source_ckpt" ]]; then
  echo "Missing source checkpoint: $source_ckpt" >&2
  exit 1
fi
if [[ -e "checkpoints/$run_name/checkpoint_latest.pth" ]]; then
  echo "Refusing to overwrite an existing run: $run_name" >&2
  exit 1
fi

python_bin=/home/u2025140689/anaconda3/envs/defomstereo/bin/python
export LD_LIBRARY_PATH=/home/u2025140689/anaconda3/envs/defomstereo/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES="$gpu_id"
"$python_bin" train_stereo.py \
  --model_arch "$model_arch" --launcher none --gpu_ids 0 \
  --name "$run_name" --resume_ckpt "$source_ckpt" --no_resume_optimizer \
  --batch_size 1 --gradient_accumulation_steps 2 --num_workers 2 \
  --train_datasets sceneflow --train_folds 1 \
  --num_steps 2000 --save_latest_ckpt_freq 1000 --save_ckpt_freq 2000 \
  --val_freq 2000 --valid_iters 32 --train_iters 18 \
  --n_downsample 2 --num_init_bins 48 --init_max_offset 48 \
  --init_corr_scale 10 --loss_gamma 0.9 --init_target_temperature 1.0 \
  --gate_loss_weight 0.1 --gate_target_temperature 4.0 \
  --smooth_l1_beta 1.0 --d0_loss_weight 1.0 --max_delta_disp 8.0 \
  --train_max_disp 768 --eval_max_disp 768 --corr_levels 3 --corr_radius 4 \
  --dinov2_encoder vitl --init_right_compression avg \
  --refine_right_compression avg --context_norm batch --n_gru_layers 3 \
  --hidden_dims 128 --image_size 256 768 --wdecay 0.00001 --seed 1234565 \
  --mixed_precision --amp_dtype fp16 --debug_nan --amp_init_scale 64 \
  --lr 0.00002
