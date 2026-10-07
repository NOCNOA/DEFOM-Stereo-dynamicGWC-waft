#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/u2025140689/anaconda3/envs/defomstereo2/bin/python
OUTPUT=checkpoints/corga_sparse_hr_candidate_gate_pilot1k
DIAG=diagnostics/sparse_hr_diagnosis

CUDA_VISIBLE_DEVICES=2 "$PYTHON" scripts/corga/train_sparse_hr_candidate_gate.py \
  --steps 1000 --batch-size 2 --n-query 2048 --lr 1e-4 \
  --gate-loss-weight 0.5 --save-every 250 --output "$OUTPUT" \
  2>&1 | tee nohups/corga_sparse_hr_candidate_gate_pilot1k.out

for step in 250 500 1000; do
  CUDA_VISIBLE_DEVICES=2 "$PYTHON" scripts/corga/eval_sparse_hr_refiner.py \
    --refiner "$OUTPUT/refiner_step${step}.pth" \
    --mode predicted_boundary --band-radius 2 --samples 100 \
    --output "$DIAG/candidate_gate_step${step}_predicted_radius2_100.json" \
    > "nohups/corga_sparse_hr_candidate_gate_eval_step${step}.out" 2>&1
done
