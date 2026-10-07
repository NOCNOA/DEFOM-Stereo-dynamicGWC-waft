#!/usr/bin/env bash
set -euo pipefail

# Wait for two independently launched training processes, then evaluate once.
baseline_pid="${1:?Usage: $0 BASELINE_PID WAVELET_PID}"
wavelet_pid="${2:?Usage: $0 BASELINE_PID WAVELET_PID}"
cd "$(dirname "$0")/../.."
run_root=/var/tmp/codex-home-u2025140689/Documents/Codex/2026-10-04/gru-cost-volume-costvolume
results="$run_root/outputs/wavelet_pair_20261005"
logs="$run_root/work"
source_ckpt=checkpoints/defomstereo_vitl_corga_bins_48_avg_ft20k_control/20000_epoch_defomstereo_vitl_corga_bins_48_avg_ft20k_control.pth.gz
baseline_ckpt=checkpoints/bins_wavelet_pair_20261005_baseline_2k_b1a2/checkpoint_latest.pth
wavelet_ckpt=checkpoints/bins_wavelet_pair_20261005_wavelet_2k_b1a2/checkpoint_latest.pth
baseline_log="$logs/train_baseline_b1a2.log"
wavelet_log="$logs/train_wavelet_b1a2.log"
python_bin=/home/u2025140689/anaconda3/envs/defomstereo/bin/python

mkdir -p "$results"
printf 'training\n' > "$results/status.txt"
tail --pid="$baseline_pid" --pid="$wavelet_pid" -f /dev/null

for arm in baseline wavelet; do
  log="$logs/train_${arm}_b1a2.log"
  if ! rg -q 'FINISHED TRAINING' "$log"; then
    printf 'failed: %s training did not finish\n' "$arm" > "$results/status.txt"
    exit 1
  fi
done
for checkpoint in "$source_ckpt" "$baseline_ckpt" "$wavelet_ckpt"; do
  if [[ ! -s "$checkpoint" ]]; then
    printf 'failed: missing checkpoint %s\n' "$checkpoint" > "$results/status.txt"
    exit 1
  fi
done

printf 'evaluating\n' > "$results/status.txt"
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/home/u2025140689/anaconda3/envs/defomstereo/lib:${LD_LIBRARY_PATH:-}
manifest="$results/large_disp_manifest.json"
for arm in source baseline wavelet; do
  case "$arm" in
    source) checkpoint="$source_ckpt"; model_arch=cor_ga_bins ;;
    baseline) checkpoint="$baseline_ckpt"; model_arch=cor_ga_bins ;;
    wavelet) checkpoint="$wavelet_ckpt"; model_arch=cor_ga_bins_wavelet ;;
  esac
  "$python_bin" scripts/bin/evaluate_large_disp_init.py \
    --restore_ckpt "$checkpoint" --model_arch "$model_arch" \
    --output "$results/$arm" --sample_manifest "$manifest" \
    --max_samples 0 --modes full --valid_iters 32 --mixed_precision \
    > "$logs/eval_${arm}_b1a2.log" 2>&1
done
"$python_bin" scripts/bin/summarize_bins_wavelet_pair.py \
  --results "$results" --baseline_log "$baseline_log" \
  --wavelet_log "$wavelet_log" > "$logs/summary_b1a2.log" 2>&1
printf 'complete\n' > "$results/status.txt"
