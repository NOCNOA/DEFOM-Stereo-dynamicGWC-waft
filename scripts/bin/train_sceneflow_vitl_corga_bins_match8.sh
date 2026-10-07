#!/usr/bin/env bash
# Fresh width-only match8 training. No stereo checkpoint or optimizer resume.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYTHON_BIN="${PYTHON_BIN:-python}"
MATCH8_GPU="${MATCH8_GPU:-0}"
MATCH8_NAME="${MATCH8_NAME:-defomstereo_vitl_bins_match8_width_scratch}"
MATCH8_BATCH="${MATCH8_BATCH:-2}"
MATCH8_STEPS="${MATCH8_STEPS:-200000}"
MATCH8_LR="${MATCH8_LR:-0.00020}"
CHECKPOINT_DIR="checkpoints/${MATCH8_NAME}"

if [[ ! "${MATCH8_GPU}" =~ ^[0-9]+$ ]] || (( 10#${MATCH8_GPU} == 1 )); then
    echo "Select one GPU index other than GPU 1." >&2
    exit 1
fi
# Hold this lock for the whole run, including the logging pipeline. The
# occupancy check also detects jobs started outside this launcher.
exec 9>"${TMPDIR:-/tmp}/defom-match8-${UID}-gpu-${MATCH8_GPU}.lock"
if ! flock -n 9; then
    echo "Another match8 launcher already owns GPU ${MATCH8_GPU}." >&2
    exit 1
fi
GPU_PROCESSES="$(nvidia-smi -i "${MATCH8_GPU}" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)"
if [[ -n "${GPU_PROCESSES//[[:space:]]/}" ]]; then
    echo "GPU ${MATCH8_GPU} already has compute processes (PID, MiB):" >&2
    echo "${GPU_PROCESSES}" >&2
    exit 1
fi
if [[ -e "${CHECKPOINT_DIR}" ]]; then
    echo "Experiment directory already exists: ${CHECKPOINT_DIR}. Choose a new MATCH8_NAME." >&2
    exit 1
fi
for weight in checkpoints/depth_anything_v2_vitl.pth checkpoints/pytorch_model.bin; do
    if [[ ! -s "${weight}" ]]; then
        echo "Missing encoder pretraining file: ${weight}" >&2
        exit 1
    fi
done
mkdir -p "${CHECKPOINT_DIR}"
echo "model=cor_ga_bins_match8 gpu=${MATCH8_GPU} batch=${MATCH8_BATCH} steps=${MATCH8_STEPS} lr=${MATCH8_LR} name=${MATCH8_NAME}"

CUDA_VISIBLE_DEVICES="${MATCH8_GPU}" "${PYTHON_BIN}" -u train_stereo.py \
    --model_arch cor_ga_bins_match8 \
    --launcher none --gpu_ids 0 --name "${MATCH8_NAME}" \
    --batch_size "${MATCH8_BATCH}" --gradient_accumulation_steps 1 --num_workers 2 \
    --train_datasets sceneflow --train_folds 1 \
    --num_steps "${MATCH8_STEPS}" --lr "${MATCH8_LR}" \
    --save_ckpt_freq 20000 --save_latest_ckpt_freq 1000 \
    --val_freq 10000 --valid_iters 32 --train_iters 18 \
    --dinov2_encoder vitl --n_downsample 2 \
    --num_init_bins 48 --init_max_offset 48 --init_corr_scale 10 \
    --init_right_compression avg --refine_right_compression avg \
    --corr_levels 3 --corr_radius 4 \
    --context_norm batch --n_gru_layers 3 --hidden_dims 128 \
    --image_size 256 768 --train_max_disp 768 --eval_max_disp 768 \
    --loss_gamma 0.9 --init_target_temperature 1 \
    --gate_loss_weight 0.1 --gate_target_temperature 4 \
    --smooth_l1_beta 1 --d0_loss_weight 1 \
    --wdecay 0.00001 --seed 1234565 \
    --mixed_precision --amp_dtype fp16 --amp_init_scale 64 \
    --disable_activation_checkpoint \
    2>&1 | tee "${CHECKPOINT_DIR}/train.log"
