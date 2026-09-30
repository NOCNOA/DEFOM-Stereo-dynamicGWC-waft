#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?usage: evaluate_unreal_full_200k.sh GPU_ID}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${REPO_ROOT}/diagnostics/unreal_full_200k"
MANIFEST="${MANIFEST:-${OUT_DIR}/manifest_all.txt}"
test -s "${MANIFEST}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MIXED_RESULT_JSONL="${OUT_DIR}/results_gpu${GPU_ID}.jsonl"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

/home/u2025140689/anaconda3/envs/defomstereo2/bin/python evaluate_stereo.py \
  --model_arch cor_ga_fast \
  --restore_ckpt checkpoints/defomstereo_vitl_twostage_768/defomstereo_vitl_twostage_768_200000.pth \
  --datasets mixed \
  --mixed_names unrealstereo4k \
  --mixed_root unrealstereo4k=/home/share/zhaobofan/UnrealStereo4K \
  --unreal_scene all \
  --mixed_manifest "unrealstereo4k=${MANIFEST}" \
  --mixed_max_samples 8200 \
  --max_disp 768 \
  --valid_iters 32 \
  --scale_iters 6 \
  --corr_levels 3 \
  --corr_radius 4 \
  --dinov2_encoder vitl \
  --hypothesis_scales 24 48 96 \
  --mixed_precision \
  --amp_dtype bf16
