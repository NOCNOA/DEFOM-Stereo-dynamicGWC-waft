#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-fresh}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-defomstereo_corga_igevpp_encoder}"
MODEL_ARCH="${MODEL_ARCH:-cor_ga_stereo_igevpp_encoder}"
ARCHIVE_FULL_CKPT_FREQ="${ARCHIVE_FULL_CKPT_FREQ:-0}"
ARCHIVE_FULL_CKPT_START_STEP="${ARCHIVE_FULL_CKPT_START_STEP:-0}"
SAVE_LATEST_CKPT_FREQ="${SAVE_LATEST_CKPT_FREQ:-5000}"
SAVE_CKPT_FREQ="${SAVE_CKPT_FREQ:-50000}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/${EXPERIMENT_NAME}}"
LATEST_CKPT="${CHECKPOINT_DIR}/checkpoint_latest.pth"
GPU_ID="${GPU_ID:-0,2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_STEPS="${NUM_STEPS:-200000}"
STOP_AFTER_STEP="${STOP_AFTER_STEP:-0}"
VAL_FREQ="${VAL_FREQ:-20000}"
RESUME_LR_SCALE="${RESUME_LR_SCALE:-1.0}"
RESUME_BATCH_SKIP="${RESUME_BATCH_SKIP:-0}"
DIAGNOSTIC_GRAD_FREQ="${DIAGNOSTIC_GRAD_FREQ:-0}"
FREEZE_STEREO_ENCODER="${FREEZE_STEREO_ENCODER:-0}"
FREEZE_COARSE_GRU_MOTION="${FREEZE_COARSE_GRU_MOTION:-0}"
FREEZE_REFINE_UPDATE_PROJECTION="${FREEZE_REFINE_UPDATE_PROJECTION:-0}"
FREEZE_REFINE_UPDATE="${FREEZE_REFINE_UPDATE:-0}"
FREEZE_REFINE_PROJECTION="${FREEZE_REFINE_PROJECTION:-0}"
FREEZE_REFINE_COMPONENT="${FREEZE_REFINE_COMPONENT:-}"
MATCH_ENCODER="${MATCH_ENCODER:-igevpp}"
DINOV2_ENCODER="${DINOV2_ENCODER:-vits}"
PAIRED_ENCODER_INIT="${PAIRED_ENCODER_INIT:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/u2025140689/anaconda3/envs/defomstereo2/bin/python}"
DEBUG_NAN="${DEBUG_NAN:-0}"

if [[ ! "${GPU_ID}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "GPU_ID must be comma-separated GPU indices, got '${GPU_ID}'." >&2
    exit 2
fi
IFS=',' read -r -a GPU_IDS <<< "${GPU_ID}"
GPU_COUNT="${#GPU_IDS[@]}"
if [[ ! "${GLOBAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || (( GLOBAL_BATCH_SIZE % GPU_COUNT != 0 )); then
    echo "GLOBAL_BATCH_SIZE must be a positive multiple of the ${GPU_COUNT} visible GPUs, got '${GLOBAL_BATCH_SIZE}'." >&2
    exit 2
fi
if (( GPU_COUNT > 1 )); then
    LAUNCHER="pytorch"
    TRAIN_LAUNCH=("${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1 "--nproc_per_node=${GPU_COUNT}" train_stereo.py)
else
    LAUNCHER="none"
    TRAIN_LAUNCH=("${PYTHON_BIN}" train_stereo.py)
fi

case "${DEBUG_NAN}" in
    0) DEBUG_ARGS=() ;;
    1) DEBUG_ARGS=(--debug_nan) ;;
    *)
        echo "DEBUG_NAN must be 0 or 1, got '${DEBUG_NAN}'." >&2
        exit 2
        ;;
esac

case "${FREEZE_STEREO_ENCODER}" in
    0) FREEZE_ARGS=() ;;
    1) FREEZE_ARGS=(--freeze_stereo_encoder) ;;
    *)
        echo "FREEZE_STEREO_ENCODER must be 0 or 1, got '${FREEZE_STEREO_ENCODER}'." >&2
        exit 2
        ;;
esac

case "${FREEZE_COARSE_GRU_MOTION}" in
    0) ;;
    1) FREEZE_ARGS+=(--freeze_coarse_gru_motion) ;;
    *)
        echo "FREEZE_COARSE_GRU_MOTION must be 0 or 1, got '${FREEZE_COARSE_GRU_MOTION}'." >&2
        exit 2
        ;;
esac

case "${FREEZE_REFINE_UPDATE_PROJECTION}" in
    0) ;;
    1) FREEZE_ARGS+=(--freeze_refine_update_projection) ;;
    *)
        echo "FREEZE_REFINE_UPDATE_PROJECTION must be 0 or 1, got '${FREEZE_REFINE_UPDATE_PROJECTION}'." >&2
        exit 2
        ;;
esac

case "${FREEZE_REFINE_UPDATE}" in
    0) ;;
    1) FREEZE_ARGS+=(--freeze_refine_update) ;;
    *)
        echo "FREEZE_REFINE_UPDATE must be 0 or 1, got '${FREEZE_REFINE_UPDATE}'." >&2
        exit 2
        ;;
esac

case "${FREEZE_REFINE_PROJECTION}" in
    0) ;;
    1) FREEZE_ARGS+=(--freeze_refine_projection) ;;
    *)
        echo "FREEZE_REFINE_PROJECTION must be 0 or 1, got '${FREEZE_REFINE_PROJECTION}'." >&2
        exit 2
        ;;
esac

case "${FREEZE_REFINE_COMPONENT}" in
    '') ;;
    motion_encoder|gru_layers|context_encoders|disp_head|mask)
        FREEZE_ARGS+=(--freeze_refine_component "${FREEZE_REFINE_COMPONENT}") ;;
    *)
        echo "Invalid FREEZE_REFINE_COMPONENT: '${FREEZE_REFINE_COMPONENT}'." >&2
        exit 2 ;;
esac

case "${MATCH_ENCODER}" in
    igevpp|dinov2) ;;
    *) echo "Invalid MATCH_ENCODER: '${MATCH_ENCODER}'." >&2; exit 2 ;;
esac
case "${PAIRED_ENCODER_INIT}" in
    0) ;;
    1) FREEZE_ARGS+=(--paired_encoder_init) ;;
    *) echo "PAIRED_ENCODER_INIT must be 0 or 1." >&2; exit 2 ;;
esac

case "${MODE}" in
    fresh)
        if [[ -e "${LATEST_CKPT}" ]]; then
            echo "Refusing to start fresh: ${LATEST_CKPT} already exists." >&2
            exit 1
        fi
        RESUME_ARGS=()
        ;;
    resume)
        RESUME_CKPT="${RESUME_CKPT_OVERRIDE:-${LATEST_CKPT}}"
        if [[ ! -s "${RESUME_CKPT}" ]]; then
            echo "Cannot resume: ${RESUME_CKPT} does not exist or is empty." >&2
            exit 1
        fi
        RESUME_ARGS=(--resume_ckpt "${RESUME_CKPT}" --strict_resume)
        ;;
    *)
        echo "Usage: $0 [fresh|resume]" >&2
        exit 2
        ;;
esac

mkdir -p "${CHECKPOINT_DIR}"
echo "mode=${MODE} experiment=${EXPERIMENT_NAME} model=${MODEL_ARCH} gpu=${GPU_ID} launcher=${LAUNCHER} global_batch=${GLOBAL_BATCH_SIZE}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${TRAIN_LAUNCH[@]}" \
--model_arch "${MODEL_ARCH}" \
--match_encoder "${MATCH_ENCODER}" \
--dinov2_encoder "${DINOV2_ENCODER}" \
--launcher "${LAUNCHER}" \
--name "${EXPERIMENT_NAME}" \
--batch_size "${GLOBAL_BATCH_SIZE}" \
--gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
--num_workers 4 \
--train_datasets sceneflow \
--train_folds 1 \
--num_steps "${NUM_STEPS}" \
--stop_after_step "${STOP_AFTER_STEP}" \
--save_ckpt_freq "${SAVE_CKPT_FREQ}" \
--save_latest_ckpt_freq "${SAVE_LATEST_CKPT_FREQ}" \
--archive_full_ckpt_freq "${ARCHIVE_FULL_CKPT_FREQ}" \
--archive_full_ckpt_start_step "${ARCHIVE_FULL_CKPT_START_STEP}" \
--n_downsample 2 \
--train_iters 18 \
--scale_iters 6 \
--coarse_loss_weight 0.2 \
--bin_disp_smooth_l1_weight 0.5 \
--num_init_bins 48 \
--init_max_offset 48 \
--init_corr_scale 10 \
--init_right_compression avg \
--loss_gamma 0.9 \
--smooth_l1_beta 1.0 \
--max_delta_disp 8.0 \
--train_max_disp 768 \
--eval_max_disp 768 \
--corr_levels 3 \
--corr_radius 4 \
--context_norm batch \
--n_gru_layers 3 \
--hidden_dims 128 \
--image_size 256 768 \
--wdecay 0.00001 \
--seed 1234565 \
--val_freq "${VAL_FREQ}" \
--valid_iters 32 \
--amp_dtype bf16 \
--resume_lr_scale "${RESUME_LR_SCALE}" \
--resume_batch_skip "${RESUME_BATCH_SKIP}" \
--diagnostic_grad_freq "${DIAGNOSTIC_GRAD_FREQ}" \
"${RESUME_ARGS[@]}" \
"${FREEZE_ARGS[@]}" \
"${DEBUG_ARGS[@]}" \
--mixed_precision \
--lr 0.00020 \
2>&1 | tee -a "${CHECKPOINT_DIR}/train.log"
