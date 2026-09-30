#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN=/home/u2025140689/anaconda3/envs/defomstereo2/bin/python
NAME=defomstereo_vitl_corga_no_mono_igev_c32r16_ft2k
OUT="checkpoints/${NAME}"
SOURCE=checkpoints/defomstereo_vitl_corga_no_mono_igev/defomstereo_vitl_corga_no_mono_igev_200000.pth
FINISHED="${OUT}/${NAME}_  2000.pth"
RESULTS=diagnostics/large_delta_eval
mkdir -p "$OUT" "$RESULTS"

if [[ ! -s "$SOURCE" ]]; then
    echo "Missing source checkpoint: $SOURCE" >&2
    exit 1
fi

if [[ ! -s "$FINISHED" ]]; then
    if [[ -e "${OUT}/checkpoint_latest.pth" ]]; then
        echo "Existing partial run in $OUT; inspect before restarting" >&2
        exit 1
    fi
    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" train_stereo.py \
        --model_arch cor_ga_no_mono_igev \
        --dinov2_encoder vitl \
        --launcher none --gpu_ids 0 \
        --name "$NAME" \
        --batch_size 2 --num_workers 4 \
        --train_datasets sceneflow --train_folds 1 \
        --num_steps 2000 --save_ckpt_freq 1000 --save_latest_ckpt_freq 1000 \
        --n_downsample 2 --train_iters 18 --scale_iters 6 \
        --coarse_loss_weight 0.2 --bin_loss_mode soft_ce \
        --bin_disp_smooth_l1_weight 0.2 \
        --num_init_bins 48 --init_max_offset 48 --init_corr_scale 10 \
        --init_right_compression avg \
        --loss_gamma 0.9 --smooth_l1_beta 1.0 \
        --max_coarse_delta_disp 32 --max_delta_disp 16 \
        --train_max_disp 768 --eval_max_disp 768 \
        --corr_levels 3 --corr_radius 4 --context_norm batch \
        --n_gru_layers 3 --hidden_dims 128 --image_size 256 768 \
        --wdecay 0.00001 --seed 1234565 \
        --val_freq 10000 --valid_iters 32 \
        --amp_dtype bf16 --mixed_precision --lr 0.00001 \
        --resume_ckpt "$SOURCE" --strict_resume --no_resume_optimizer \
        > "${OUT}/train.log" 2>&1
fi

if [[ ! -s "$FINISHED" ]]; then
    echo "Training finished without $FINISHED" >&2
    exit 1
fi

pids=()
for GPU in 0 2 3; do
    STEM="${RESULTS}/no_mono_igev_c32r16_ft2k_sceneflow_gpu${GPU}"
    CUDA_VISIBLE_DEVICES="$GPU" MIXED_RESULT_JSONL="${ROOT}/${STEM}.jsonl" \
        "$PYTHON_BIN" evaluate_stereo.py \
        --datasets mixed --mixed_names sceneflow \
        --mixed_root sceneflow=/home/share/yijiayi/sceneflow \
        --mixed_manifest "sceneflow=${ROOT}/${RESULTS}/no_mono_igev_sceneflow_gpu${GPU}_manifest.txt" \
        --mixed_max_samples 4370 --max_disp 768 \
        --valid_iters 32 --scale_iters 6 \
        --corr_levels 3 --corr_radius 4 \
        --dinov2_encoder vitl --model_arch cor_ga_no_mono_igev \
        --num_init_bins 48 --init_max_offset 48 --init_corr_scale 10 \
        --mixed_precision --amp_dtype bf16 \
        --max_coarse_delta_disp 32 --max_delta_disp 16 \
        --restore_ckpt "$FINISHED" \
        > "${STEM}.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
if (( failed )); then
    echo "One or more SceneFlow evaluation shards failed" >&2
    exit 1
fi

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

from scripts.corga.eval_no_mono_igev_clamp import BUCKETS, score

root = Path('diagnostics/large_delta_eval')
rows = {}
for gpu in (0, 2, 3):
    manifest = root / f'no_mono_igev_sceneflow_gpu{gpu}_manifest.txt'
    expected = set(manifest.read_text().splitlines())
    path = root / f'no_mono_igev_c32r16_ft2k_sceneflow_gpu{gpu}.jsonl'
    shard = [json.loads(line) for line in path.read_text().splitlines() if line]
    actual = {row['left'] for row in shard}
    if len(shard) != len(expected) or actual != expected:
        raise ValueError(f'Incomplete or duplicate shard {gpu}: {len(shard)} / {len(expected)}')
    rows.update((row['left'], row) for row in shard)
if len(rows) != 4370:
    raise ValueError(f'Expected 4370 results; got {len(rows)}')

reference = json.loads((root / 'no_mono_igev_200k_sceneflow_clamp_summary.json').read_text())
expected_pixels = reference['runs']['coarse32_refine16']['overall']['pixels']
overall = score(rows)
if overall['pixels'] != expected_pixels:
    raise ValueError(f'Valid-pixel mismatch: {overall["pixels"]} vs {expected_pixels}')
summary = {
    'checkpoint': 'checkpoints/defomstereo_vitl_corga_no_mono_igev_c32r16_ft2k/defomstereo_vitl_corga_no_mono_igev_c32r16_ft2k_  2000.pth',
    'training': {'source_step': 200000, 'finetune_steps': 2000, 'batch_size': 2,
                 'max_coarse_delta_disp': 32, 'max_delta_disp': 16, 'lr': 1e-5},
    'images': len(rows),
    'overall': overall,
    'buckets': {bucket: score(rows, bucket) for bucket in BUCKETS},
    'reference_original_cap8': reference['baseline_cap8_train_log'],
    'reference_zero_shot_32_16': reference['runs']['coarse32_refine16']['overall'],
}
path = root / 'no_mono_igev_c32r16_ft2k_sceneflow_summary.json'
path.write_text(json.dumps(summary, indent=2) + '\n')
print(path)
print(f'EPE {overall["image_mean_epe"]:.6f}, Bad3 {overall["bad3_percent"]:.4f}%')
PY
