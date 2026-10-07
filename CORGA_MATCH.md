# CorGA Matching Token model

`--model_arch cor_ga_match` selects `core/corga/defom_cor_ga_match.py`.
The model inherits directly from `nn.Module`; CorGA is a code/design reference,
not a superclass. It retains CorGA's monocular scales (24/48/96 by default),
three coarse PRU branches, hypothesis/hidden-state fusion, full-correlation
lookup and recursive refinement. Both coarse and refinement feature lookup
use the new `core/matching.py` module.

Each sampled candidate has a token made from:

- Four group-wise cosine similarities (eight channels per group in match,
  six per group in fast2-match).
- The signed difference of channel-normalized left/right features (32 channels
  in match by default; 24 channels in fast2-match).
- Optional weighted `log2(ratio)/2` scale encoding, disabled by default.

For 32-channel image features, the 36-channel evidence is projected to 32 channels.
With 24-channel image features, it is 28 channels (4 cosine + 24 difference).
Fast2-match now keeps all 28 channels as candidate tokens without a token
projection. A projection is used only when evidence and token widths differ.
Physical disparity
offset and validity are not input channels. Coordinate validity and nonnegative
candidate disparity are still used internally to mask invalid candidates.
Refinement first samples nine candidates at each right-feature scale. Each
scale's geometrically invalid or negative-disparity candidates contribute zero.
The sampled right features are multiplied by the learned `w_s` and summed
across scales, keeping the original weights without renormalizing at boundaries.
Cosine similarity and signed difference are then computed against this mixture,
and matching encoding runs once. A mixed candidate is valid if at least one
valid scale with nonzero weight contributes. One residual pair of kernel-3
candidate Conv1d layers aggregates the ordered candidates; invalid tokens are
masked before and inside aggregation. The window is flattened and reduced to
the token width, then concatenated with the left feature before spatial CNN fusion
and existing RoPE/Galerkin attention. With radius 4, the fused window follows
`[B,9,36,H,W] -> [B,9,32,H,W] -> [B,288,H,W] -> [B,32,H,W]`.
For the fast2-match defaults, the fused window instead follows
`[B,9,28,H,W] -> [B,252,H,W] -> [B,28,H,W]`.
The reduced map plus the left feature give `[B,64,H,W]` with match defaults,
or `[B,52,H,W]` with fast2-match defaults (`24 + 28`). Spatial fusion is now
one kernel-3 convolution plus ReLU, directly mapping either input to 32
channels. The candidate axis has already been reduced.
Coarse lookup uses the quarter-width right feature and shares its matching
encoder across the three coarse disparity branches.

Matching statistics, full correlation, context/recurrent updates and gate
fusion are computed in FP32. Matching projections and attention use the
configured FP16/BF16 autocast. `w_s` is predicted once from the coarse output
and reused during refinement, as in CorGA. Sampler validity describes geometry,
not non-occlusion. The shared sampler's default tensor return remains compatible;
`return_valid=True` additionally returns its coordinate mask.

Train a new experiment (GPU 1 is excluded by both scripts):

```bash
bash scripts/corga/train_sceneflow_vitl_corga_match.sh fresh
bash scripts/corga/train_sceneflow_vitl_corga_match.sh resume
bash scripts/corga/evaluate_sceneflow_vitl_corga_match.sh
```

The default experiment is `defomstereo_vitl_corga_match_g4_r12`, with 12
refinement updates during training and 6 coarse updates per branch. ViT-L,
the 256x768 crop, BF16, learning rate 0.0002, weight decay 0.00001, seed
1234565 and the 200000-step schedule follow the CorGA reference script.
On GPU 0, microbatch size 2 and three gradient-accumulation steps preserve
the reference's effective batch size of 6. Validation uses 32 refinement
updates every 20000 steps; complete training state is saved every 5000 steps.
Training checkpoints sampling/fusion/matching encoding and recurrent updates to avoid
retaining every iteration's intermediate activations. Recomputed output and
gradient equivalence are covered by CPU tests.

Override `GPU_ID`, `PYTHON_BIN`, `EXPERIMENT_NAME`, `BATCH_SIZE` or
`GRADIENT_ACCUMULATION_STEPS` as needed.
Evaluation accepts a `CHECKPOINT` environment override. Extra command-line
arguments go last, so ablations can override defaults:

```bash
EXPERIMENT_NAME=defomstereo_vitl_corga_match_g8 bash scripts/corga/train_sceneflow_vitl_corga_match.sh fresh --match_groups 8
EXPERIMENT_NAME=defomstereo_vitl_corga_match_noagg bash scripts/corga/train_sceneflow_vitl_corga_match.sh fresh --match_aggregation_layers 0
```

Training and evaluation must use the same `match_feature_dim`, `match_groups`, `match_token_dim`,
`match_aggregation_layers`, `match_scale_encoding`, compression and radius.
Original CorGA checkpoints do not strictly load into this architecture. For
a weight-only warm start, use `--resume_ckpt PATH --no_resume_optimizer` without
`--strict_resume`; the matching encoders are newly initialized and old feature
fusion keys are unused. Optimizer state from CorGA must not be resumed across
this architecture change.

CPU checks cover coordinate geometry, a known stereo shift, invalid candidates,
candidate isolation, gradient flow and configuration ablations. The integration
test uses the real ViT-S encoder with random weights, BF16 and CUDA/xFormers.
It checks three coarse branches, two refinement iterations, backward and the
inference/zero-iteration return contracts. It requires explicit GPU selection
and is skipped when running the CPU suite:

```bash
CUDA_VISIBLE_DEVICES='' /home/u2025140689/anaconda3/envs/defomstereo2/bin/python -m unittest discover -s tests -p 'test_*match*.py' -v
CUDA_VISIBLE_DEVICES=0 /home/u2025140689/anaconda3/envs/defomstereo2/bin/python -m unittest discover -s tests -p test_corga_match.py -v
```

These checks establish implementation correctness, not an EPE improvement.
Compare against CorGA with the same data, training budget and refinement counts.

## Fast2 matching variant

`--model_arch cor_ga_fast2_match` selects
`core/corga/defom_cor_ga_fast2_match.py`. It reuses the matching architecture and
retains the group-wise cosine/difference evidence, candidate aggregation,
negative-disparity masking and sample-then-weight-and-sum-before-matching
fusion. The differences from match are in execution:

The fast2-match default left/right `low_channel` projection is now `64 -> 24`;
match defaults to `64 -> 32`. `--match_feature_dim` overrides this image-feature
width in either model. Fast2-match candidate tokens and the reduced window map
default to 28 channels; matching output and PRU warp inputs remain 32 channels.
RoPE remains four heads of eight channels,
and the recurrent hidden width is unaffected. Four cosine groups split the
24 image-feature channels into six channels per group.

- Feature and correlation sampling use horizontal linear interpolation with
  cached coordinate rows. Geometric validity has the same definition as match.
- All three coarse hypotheses share a batched correlation/matching lookup per
  iteration, while their PRU weights and states remain independent.
- RoPE positions and static PRU context/prompt features are prepared once per
  forward, with their autograd graphs retained.
- Inference skips coarse masks/predictions and computes the refinement mask only
  for the final update. Zero refinement iterations return the fused initial view.
- Matching and PRU checkpointing remain enabled in training; cached static
  tensors are explicit checkpoint inputs.

PRU updates and their cached context projections use the selected AMP dtype,
along with the backbone and stereo feature head. The context encoder, hypothesis
fusion, matching statistics and FullCorr stay FP32. Recurrent state inputs and
disparity accumulation remain FP32; PRU convolutions use FP16/BF16 autocast.
Consequently AMP output is not expected to be identical to match.
With AMP disabled, CPU tests compare model predictions and gradients against
match, using the real stereo heads/PRUs and a stub for the CUDA/xFormers backbone.

For identical configuration, including explicit `match_feature_dim`, state-dict
keys and shapes match `cor_ga_match` (also specify the same `match_token_dim`),
and weights load strictly in either direction. Checkpoint optimizer parameter
ordering is also preserved; for a new comparison experiment, prefer a weight-only
warm start so its optimizer and step schedule start fresh. Plain `cor_ga_fast2`
weights do not strictly load because they lack matching encoders.

Older match/fast2-match checkpoints with 38-channel evidence have a different
first token-projection weight shape (`[32,38,1,1]` rather than `[32,36,1,1]`
with default settings), so they cannot be resumed strictly by the current model.
The previous encode-then-concatenate refinement architecture also had a
128-channel spatial-fusion input; it is now 64 channels. Its checkpoint weights
and optimizer state cannot be resumed strictly either.

```bash
bash scripts/corga/train_sceneflow_vitl_corga_fast2_match.sh fresh
bash scripts/corga/train_sceneflow_vitl_corga_fast2_match.sh resume
bash scripts/corga/evaluate_sceneflow_vitl_corga_fast2_match.sh
```

The scripts use a separate `defomstereo_vitl_corga_fast2_match_c24_t28_g4_r18` experiment.
Training follows the single-GPU CorGA baseline: ViT-L, batch size 2, one
gradient-accumulation step, 18 refinement updates, 6 coarse updates, 256x768
crops, BF16, learning rate 0.0002, weight decay 0.00001 and 200000 optimizer
steps. The other matching options and environment overrides remain the same
as the match scripts. They explicitly select GPU 0 by default and reject GPU 1.
For match weights trained with the current single-convolution fusion and default
32-channel image features/tokens, evaluate with `--match_feature_dim 32
--match_token_dim 32` and a matching `CHECKPOINT`. Older two-convolution fusion
weights cannot be loaded strictly, even with those dimension overrides.

Run the CPU checks without allocating a GPU:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/u2025140689/anaconda3/envs/defomstereo2/bin/python \
  -m unittest discover -s tests -p 'test_*match*.py' -v
```

The horizontal encoder and model tests check strict weight loading, sample values
and validity, gradients with/without checkpointing, one/three PRU layers, batched
coarse predictions, final/zero-iteration outputs and inference mask skipping.
The new CPU tests temporarily use native convolution kernels because oneDNN
backward stalls on this host; this does not change model execution settings.
Batching retains three copies of coarse feature inputs, so it can increase the
temporary working set; checkpointing reduces retained activations rather than
guaranteeing lower peak memory. Speed, memory and EPE require measurements with
the intended crop, checkpoint and hardware.

The real ViT-S BF16 integration test covers forward/backward and inference on an
explicitly selected GPU:

```bash
CUDA_VISIBLE_DEVICES=0 /home/u2025140689/anaconda3/envs/defomstereo2/bin/python \
  -m unittest discover -s tests -p test_corga_fast2_match.py -k FastMatchingCudaTests -v
```

Image-feature width changes affect the low-channel projection, right compressor,
first token projection and spatial-fusion input. A 32-channel image-feature
checkpoint does not strictly load with `match_feature_dim=24`, although RoPE,
PRU input and hidden dimensions still match. Use a new experiment for 24 channels.
Older checkpoints with two spatial-fusion convolutions are also incompatible
with the current direct fusion, including the earlier 24-feature/32-token model.
Use a new experiment for the 28-token/direct-fusion architecture.

Training accepts `--disable_activation_checkpoint` to disable both matching and
PRU recomputation in either matching model. Defaults remain enabled; use this
flag for the no-checkpoint measurements described below.

The profiling results below were collected before removing the offset/validity
input channels, moving scale fusion ahead of matching or enabling PRU AMP, with
38-channel evidence and separate per-scale matching. They have not been
remeasured for the current encoder.

The full default training configuration was also checked on 2026-10-03 with
two real SceneFlow optimizer steps on GPU 0 (24GB RTX 4090), using ViT-L's
pretrained depth weights, batch size 2, 256x768 crops, 18 refinement updates,
6 coarse updates and BF16. The 200000-step LR schedule was retained via
`--stop_after_step 2`. Forward, backward and optimizer updates completed;
PyTorch peak allocated/reserved memory was 9.722/10.430 GiB. This is a startup
and memory check, not evidence of full-run convergence or validation accuracy.

The same configuration was profiled against fast2 in separate processes on the
same idle GPU. Both successful runs completed two optimizer steps, and CUDA was
synchronized when recording per-step peaks:

| Architecture / training execution | Peak allocated GiB | Peak reserved GiB | Result |
| --- | ---: | ---: | --- |
| `cor_ga_fast2` (existing implementation, no checkpointing) | 19.098 | 19.313 | Two steps completed |
| `cor_ga_fast2_match` (default PRU and matching checkpointing) | 9.722 | 10.430 | Two steps completed |
| `cor_ga_fast2_match` (both checkpoint paths disabled) | 22.933 at failure | 23.037 at failure | CUDA OOM during first refinement forward |

The last row is memory reached before failure, not a measurement of the full
required peak. Checkpointing recomputes matching-scale and PRU activations on
backward instead of retaining them from every iteration. This accounts for the
low default footprint despite additional matching computations. Fast2's PRUs
also use AMP while fast2-match's PRUs are FP32, so this compares each variant's
current execution policy under the same training hyperparameters. These memory
checks do not establish a training-speed advantage; recomputation adds work.
Logs are retained in `checkpoints/defomstereo_memory_compare_*_20261003/`.

Reducing only `train_iters` to 14 under the same configuration completed two
steps with default checkpointing: peak allocated/reserved memory was
9.336/9.986 GiB. Disabling both checkpoint paths still produced CUDA OOM during
the first refinement forward on the 24GB GPU, reaching 22.933/23.037 GiB before
failure. The 14-update comparison is recorded in
`checkpoints/defomstereo_memory_compare_fast2_match_r14_cp_20261003/comparison.json`.

With only `train_iters` reduced to 12, two steps with default checkpointing
completed at 9.141/9.830 GiB peak allocated/reserved memory. Disabling both
checkpoint paths still caused CUDA OOM during the first refinement forward,
reaching 22.933/23.037 GiB before failure. Results are recorded in
`checkpoints/defomstereo_memory_compare_fast2_match_r12_cp_20261003/comparison.json`.

An isolated profiling override restored fast2-style BF16 AMP for context,
cached context projections, PRUs and hypothesis fusion at 12 refinement updates.
Matching statistics, FullCorr and disparity accumulation stayed FP32. Two
checkpointed training steps completed at 8.512/8.941 GiB peak allocated/reserved
memory. Disabling both checkpoint paths still caused CUDA OOM during the first
refinement forward, reaching 22.929/23.047 GiB before failure. This is an
experimental override; the repository model still defaults to FP32 PRUs.
The report and profiling override are retained in
`checkpoints/defomstereo_memory_compare_fast2_match_r12_amp_cp_20261003/`.

An additional short timing comparison used 12 refinement updates, 6 coarse
updates, ViT-L, batch size 2, 256x768 and BF16 on the same GPU. Diagnostic NaN
checks were disabled for all cases. Two steps warmed up kernels; the following
ten steps measured forward, losses, backward, clipping and optimizer/scheduler
updates, excluding data loading and input transfer:

| Execution | Mean compute seconds/step | Peak allocated GiB | Peak reserved GiB |
| --- | ---: | ---: | ---: |
| Original fast2 | 0.717 | 16.587 | 16.809 |
| Experimental AMP match, matching + PRU checkpointing | 1.340 | 8.520 | 8.941 |
| Experimental AMP match, matching checkpointing only | 1.088 | 15.489 | 15.867 |

Skipping PRU recomputation reduced step time by about 19% compared with
checkpointing both paths, while remaining within the 24GB GPU in this test.
Its step time was still about 52% higher than original fast2. These are ten-step
compute timings, not end-to-end training-throughput or convergence results.
The modes are temporary profiling overrides; the model defaults have not been
changed. The timings and profiler are retained in
`checkpoints/defomstereo_memory_compare_r12_amp_match_cp_speed_20261003/`.


## Current fused-before-matching PRU AMP check

The current fast2-match implementation applies the selected AMP dtype to coarse
and refinement PRUs, including cached context projections. The context encoder,
hypothesis fusion, FullCorr and matching statistics remain FP32. Checkpointing
is still enabled by default; it was disabled only in the profiling process.

On 2026-10-03, GPU 0 (24GB RTX 4090) was tested with real SceneFlow training,
pretrained ViT-L depth weights, batch 2, 256x768 crops, six coarse updates and
14 refinement updates. Both matching and PRU checkpointing were disabled,
with BF16 PRU AMP. The first forward ran out of memory during horizontal
right-feature sampling. No optimizer step completed. Allocated/reserved peaks
at failure were 22.930/23.029 GiB; these are incomplete peaks and do not show
the full memory requirement. See
`checkpoints/defomstereo_memory_compare_sum_before_match_r14_b2_pru_amp_no_cp_20261003/comparison.json`.


## Left/right projection 24, matching output 32

The 2026-10-04 test changes only the image-feature projection to `64 -> 24`.
Evidence is `4 + 24 = 28` channels, projected to 32-channel candidate tokens.
The reduced window map stays 32 channels. Concatenating it with the 24-channel
left feature gives 56 channels, followed by `56 -> 64 -> 32` spatial fusion.
RoPE remains four heads and PRU warp input stays 32, with hidden width 128.

With ViT-L, batch 2, 256x768 crops, six coarse and 14 refinement updates,
BF16 PRU AMP, and both matching/PRU checkpointing disabled, two real SceneFlow
optimizer steps completed on GPU 0 (24GB RTX 4090). Losses and gradient norms
were finite. Peak allocated/reserved memory was 22.853/23.006 GiB. This leaves
little headroom and verifies only the two-step run, not full training stability.
Profiling disables checkpointing; model defaults still enable it. Details are in
`checkpoints/defomstereo_memory_compare_lr24_warp32_r14_b2_pru_amp_no_cp_20261004/comparison.json`.


## Raw 28-channel evidence and direct 52-to-32 fusion

The current fast2-match defaults are 24-channel left/right features and
28-channel candidate tokens. The 4 cosine channels plus 24 difference channels
are used directly, with no learned token projection. Candidate Conv1d retains
28 channels; the nine-candidate window is reduced `252 -> 28`. Concatenating
it with the 24-channel left feature gives 52 channels, mapped by one kernel-3
convolution and ReLU directly to 32. RoPE and PRU input remain 32 channels.
The shared match encoder also now uses direct spatial fusion.

On 2026-10-04, ViT-L/B2/256x768/six coarse updates/BF16 PRU was checked on
GPU 0 with all activation checkpointing disabled:

| Refine updates | Result | Allocated / reserved GiB |
|---|---|---|
| 14 | Two real SceneFlow optimizer steps, finite losses/gradients | 21.387 / 21.699 |
| 18 | OOM during the first forward; no optimizer steps | 22.930 / 23.035 at failure |

The 14-update peak allocation is 1.466 GiB lower than the previous
24-feature/32-token/two-convolution fusion run. Successful runs cover only two
steps, not full-run stability or accuracy. OOM peaks do not show the full memory
requirement. See
`checkpoints/defomstereo_memory_compare_lr24_t28_fuse52_r14_b2_no_cp_20261004/comparison.json`.
