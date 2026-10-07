"""Measure multi-range initialization and controlled inference interventions.

Run from any directory. GPU selection must be explicit in the environment.
The range interventions reuse trained weights; they are not retrained ablations.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.bin.defom_cor_ga_bins import DEFOMStereo as BinStereo
from core.bin.defom_cor_ga_bins_wavelet import DEFOMStereo as WaveletBinStereo
from core.stereo_datasets import SceneFlowDatasets
from core.utils.utils import InputPadder, upflow


def accumulate(stats, mode, source, error, masks, gt, edges):
    valid_pixels = masks.get('valid_gt', next(iter(masks.values())))
    if not torch.isfinite(error[valid_pixels]).all():
        raise FloatingPointError(f'Non-finite prediction: {mode}/{source}')
    bands = [('all', torch.ones_like(gt, dtype=torch.bool))]
    bands += [(f'{lo:g}-{hi:g}', (gt >= lo) & (gt < hi))
              for lo, hi in zip(edges[:-1], edges[1:])]
    for scope, mask in masks.items():
        for band, band_mask in bands:
            values = error[mask & band_mask]
            key = (mode, source, scope, band)
            row = stats.setdefault(key, np.zeros(5, dtype=np.float64))
            row += [values.numel(), values.double().sum().item(),
                    (values > 3).sum().item(), (values > 16).sum().item(),
                    (values > 32).sum().item()]


def write_outputs(output, stats, metadata):
    rows = []
    for (mode, source, scope, band), sums in stats.items():
        n = int(sums[0])
        rows.append(dict(mode=mode, source=source, scope=scope, gt_band=band,
                         pixels=n, epe=float(sums[1] / n) if n else None,
                         **{f'bad{t}_pct': float(100 * sums[i] / n) if n else None
                            for i, t in enumerate((3, 16, 32), 2)}))
    with (output / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'results.json').write_text(json.dumps(
        dict(metadata=metadata, metrics=rows), indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in metadata['modes']:
        points = [(int(r['source'][5:]), r['epe']) for r in rows
                  if r['mode'] == mode and r['source'].startswith('iter_')
                  and r['scope'] == 'matchable' and r['gt_band'] == '384-752'
                  and r['pixels']]
        if points:
            x, y = zip(*sorted(points))
            ax.plot(x, y, marker='o', label=mode)
    ax.set(xlabel='Refinement iteration (0 = initialization)',
           ylabel='Pixel-weighted EPE (original image pixels)',
           title='GT disparity 384–752 px; in-image right correspondence')
    ax.grid(alpha=.3)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'convergence.png', dpi=180)
    plt.close(fig)
    lines = ['# Large disparity initialization evaluation', '',
             f"Checkpoint: `{metadata['checkpoint']}` (step {metadata['checkpoint_step']}).",
             f"Samples: {len(metadata['samples'])}, seed: {metadata['seed']}.",
             '', 'All errors use original-image pixels and pixel-weighted aggregation.',
             'Range interventions mask only hypothesis fusion logits; scale weights remain learned.',
             'These are inference interventions on one checkpoint, not retrained ablations.',
             'Oracle selects the smallest individual-branch error per pixel using GT. A mixture can sometimes outperform every individual branch.',
             'Dataset validity excludes GT >=768 px. Matchable means x-GT lies inside the original right image; it does not establish non-occlusion.',
             'covered_4x additionally requires GT within 4x bin support and compressed sampling coordinates inside the unpadded right feature domain.',
             '', '| Mode | Prediction | GT band | Pixels | EPE | >3 px % | >16 px % | >32 px % |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    if 'selection' in metadata:
        lines.insert(5, 'This subset was enriched using GT disparity; it is not a random TEST sample.')
    for r in rows:
        if r['scope'] != 'matchable' or not r['pixels']:
            continue
        if r['source'].startswith('iter_') and r['source'] not in ('iter_0', f"iter_{metadata['valid_iters']}"):
            continue
        lines.append(f"| {r['mode']} | {r['source']} | {r['gt_band']} | {r['pixels']} | "
                     f"{r['epe']:.3f} | {r['bad3_pct']:.2f} | {r['bad16_pct']:.2f} | {r['bad32_pct']:.2f} |")
    lines += ['', '![Convergence](convergence.png)', '',
              'Full metrics for all validity scopes and recorded iterations: `metrics.csv`.',
              'The sample paths, architecture, candidate support, and run settings are recorded in `results.json`.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--restore_ckpt', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--root', default='/home/share/yijiayi/sceneflow/')
    p.add_argument('--max_samples', type=int, default=100, help='0 evaluates the full TEST split')
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--sample_manifest', help='JSON containing sample_paths selected by the GT screening script')
    p.add_argument('--valid_iters', type=int, default=12)
    p.add_argument('--model_arch', choices=['cor_ga_bins', 'cor_ga_bins_wavelet'],
                   default='cor_ga_bins')
    p.add_argument('--modes', nargs='+', choices=['full', 'range1', 'range12'], default=['full', 'range1', 'range12'])
    p.add_argument('--dinov2_encoder', default='vitl')
    p.add_argument('--init_right_compression', choices=['avg', 'conv'], default='avg')
    p.add_argument('--refine_right_compression', choices=['avg', 'conv'], default='avg')
    p.add_argument('--num_init_bins', type=int, default=48)
    p.add_argument('--init_max_offset', type=float, default=48.)
    p.add_argument('--init_corr_scale', type=float, default=10.)
    p.add_argument('--max_delta_disp', type=float, default=8.)
    p.add_argument('--corr_levels', type=int, default=3)
    p.add_argument('--corr_radius', type=int, default=4)
    p.add_argument('--mixed_precision', action='store_true')
    p.set_defaults(hidden_dims=[128], context_norm='batch', n_gru_layers=3, n_downsample=2)
    args = p.parse_args()
    selected = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not selected or '1' in selected.split(','):
        p.error('Explicitly select a GPU other than GPU 1 with CUDA_VISIBLE_DEVICES.')
    if args.valid_iters < 0 or args.max_samples < 0:
        p.error('valid_iters and max_samples must be nonnegative')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = SceneFlowDatasets(root=args.root, dstype='frames_finalpass', things_test=True)
    if not len(dataset):
        raise RuntimeError('No TEST samples found')
    indices = np.random.default_rng(args.seed).permutation(len(dataset))
    if args.max_samples:
        indices = indices[:args.max_samples]
    if args.sample_manifest:
        manifest = json.loads(Path(args.sample_manifest).read_text())
        lookup = {pair[0]: i for i, pair in enumerate(dataset.image_list)}
        indices = [lookup[path] for path in manifest['sample_paths']]
        if not indices:
            raise RuntimeError('Sample manifest is empty')
    checkpoint = torch.load(args.restore_ckpt, map_location='cpu', mmap=True, weights_only=False)
    model_class = BinStereo if args.model_arch == 'cor_ga_bins' else WaveletBinStereo
    model = model_class(args)
    model.load_state_dict(checkpoint.get('model', checkpoint), strict=True)
    step = checkpoint.get('step')
    del checkpoint
    model.cuda().eval()
    factor = 2 ** args.n_downsample
    support = (model.bin_initializer.bin_disparities[:, -1] * factor).cpu().tolist()
    recorded = sorted(set(i for i in [0, 1, 2, 4, 8, 12, args.valid_iters] if i <= args.valid_iters))
    metadata = dict(checkpoint=str(Path(args.restore_ckpt).resolve()), checkpoint_step=step,
                    seed=args.seed, valid_iters=args.valid_iters, modes=args.modes,
                    architecture=vars(args), bin_max_original_px=support,
                    thresholds_px=[3, 16, 32], samples=[], total_dataset_samples=len(dataset))
    if args.sample_manifest:
        metadata['selection'] = manifest
    stats = {}
    for index in indices:
        data = dataset[int(index)]
        metadata['samples'].append(data['imageL_file'])
        gt = data['disp'].squeeze(0).float()
        valid = (data['valid'].squeeze(0).bool() & torch.isfinite(gt)
                 & (gt >= 0) & (gt < 768))
        h, w = gt.shape
        x = torch.arange(w).view(1, w).expand(h, w).float()
        matchable = valid & (x - gt >= 0) & (x - gt <= w - 1)
        # For r=4, x_right_compressed=(x/4 - (r-1)/2 - gt/4)/r.
        # Exclude padding when defining support; padding must not manufacture evidence.
        sample_x = (x / factor - 1.5 - gt / factor) / 4
        covered = matchable & (gt <= support[-1]) & (sample_x >= 0) & (sample_x <= w // factor // 4 - 1)
        masks = dict(valid_gt=valid, matchable=matchable, covered_4x=covered)
        padder = InputPadder(data['img1'][None].shape, divis_by=32)
        left, right = padder.pad(data['img1'][None].cuda(), data['img2'][None].cuda())
        def full(low):
            return padder.unpad(upflow(low.float(), factor=factor)).cpu().squeeze(0)
        for mode in args.modes:
            def intervene(module, inputs, logits):
                if mode == 'full':
                    return logits
                result = logits.clone()
                result[:, (1 if mode == 'range1' else 2):3] = -1e4
                return result
            hook = model.initial_weight_head.register_forward_hook(intervene)
            try:
                predictions, aux = model(left, right, iters=args.valid_iters, return_aux=True)
            finally:
                hook.remove()
            for iteration in recorded:
                pred = padder.unpad(predictions[iteration]).cpu().squeeze()
                try:
                    accumulate(stats, mode, f'iter_{iteration}', (pred - gt).abs(), masks, gt, [0, 192, 384, 752, float('inf')])
                except FloatingPointError:
                    (output / 'failure.json').write_text(json.dumps(dict(
                        image=data['imageL_file'], mode=mode, iteration=iteration,
                        mixed_precision=args.mixed_precision), indent=2))
                    raise
            if mode == 'full':
                branches = full(aux['init_disps'])
                errors = (branches - gt[None]).abs()
                for i, label in enumerate(['branch_1x', 'branch_2x', 'branch_4x']):
                    accumulate(stats, mode, label, errors[i], masks, gt, [0, 192, 384, 752, float('inf')])
                accumulate(stats, mode, 'oracle', errors.amin(0), masks, gt, [0, 192, 384, 752, float('inf')])
                accumulate(stats, mode, 'uniform_init', (branches.mean(0) - gt).abs(), masks, gt, [0, 192, 384, 752, float('inf')])
            del predictions, aux
    write_outputs(output, stats, metadata)
    print(f'Finished {len(indices)} samples. Report: {output / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
