"""Separate actual Top8 coverage from recurrent use on fixed Unreal crops."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.bin.defom_cor_ga_bins import DEFOMStereo
from core.mixed_validation import UnrealStereo4K
from core.utils.utils import InputPadder


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0', '2', '3'):
        p.error('Select GPU 0, 2 or 3 explicitly')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(19)
    common = dict(dinov2_encoder='vitl', hidden_dims=[128], context_norm='batch',
        n_gru_layers=3, n_downsample=2, mixed_precision=False,
        num_init_bins=48, init_max_offset=48., init_corr_scale=10.,
        corr_levels=3, corr_radius=4, train_max_disp=768.,
        init_right_compression='avg', refine_right_compression='avg',
        disable_activation_checkpoint=True)
    checkpoints = {
        'baseline': ROOT/'checkpoints/bins_top8_pair_20261005_rerun2_baseline_2k_b1a2/checkpoint_latest.pth',
        'guided': ROOT/'checkpoints/bins_top8_pair_20261005_rerun2_guided_2k_b1a2/checkpoint_latest.pth',
        'reference': ROOT/'checkpoints/defomstereo_vitl_corga_bins_48_avg_ft20k_control/checkpoint_latest.pth'}
    models = {}
    for name, path in checkpoints.items():
        model = DEFOMStereo(SimpleNamespace(**common, enable_top8_guidance=name == 'guided'))
        checkpoint = torch.load(str(path), map_location='cpu', mmap=True, weights_only=False)
        model.load_state_dict(checkpoint.get('model', checkpoint), strict=True)
        del checkpoint
        models[name] = model.cuda().eval()
    # Read the exact evidence consumed by the trained adapter, including masks.
    captured = {}
    def capture(module, inputs):
        if not captured:
            captured.update(disparities=inputs[0]['disparities'].detach() * 4,
                            valid=inputs[0]['valid'].detach())
    adapter = models['guided'].top8_guidance
    adapter.register_forward_pre_hook(capture)
    dataset = UnrealStereo4K('/home/share/zhaobofan/UnrealStereo4K', scene='all')
    lookup = {pair[0]: i for i, pair in enumerate(dataset.image_list)}
    crops = [json.loads(line) for line in (args.evaluation_root/'baseline/per_image.jsonl').read_text().splitlines()]
    guided_crops = {r['image']: r['crop_y'] for r in map(json.loads,
        (args.evaluation_root/'guided/per_image.jsonl').read_text().splitlines())}
    if any(guided_crops.get(r['image']) != r['crop_y'] for r in crops):
        raise ValueError('Input comparison crops differ')
    totals = {}
    coverage_totals = {}
    iterations = (0, 8, 16, 32)
    with (args.output/'per_image.jsonl').open('w') as stream:
        for number, item in enumerate(crops, 1):
            data = dataset[lookup[item['image']]]
            y0, y1 = item['crop_y']
            for key in ('img1', 'img2', 'disp', 'valid'):
                data[key] = data[key][..., y0:y1, :]
            gt = data['disp'][0].cuda().float()
            valid = data['valid'][0].cuda().bool() & torch.isfinite(gt) & (gt > 0) & (gt < 768)
            right_path = Path(data['disp_file']).parent.parent/'Disp1'/Path(data['disp_file']).name
            right_gt = torch.from_numpy(np.load(right_path)[y0:y1].copy()).cuda()
            h, w = gt.shape
            if h % 32 or w % 32:
                raise ValueError('This diagnostic requires the existing unpadded strips')
            yy, xx = torch.meshgrid(torch.arange(h, device=gt.device), torch.arange(w, device=gt.device), indexing='ij')
            xr = xx-gt
            grid = torch.stack((2*xr/(w-1)-1, 2*yy/(h-1)-1), -1)[None].float()
            rd = F.grid_sample(right_gt[None, None], grid, align_corners=True)[0, 0]
            consistent = valid & (xr >= 0) & (xr <= w-1) & torch.isfinite(rd) & (rd > 0) & ((rd-gt).abs() <= 1)
            low_gt = gt[::4, ::4]
            low_consistent = consistent[::4, ::4]
            left, right = data['img1'][None].cuda(), data['img2'][None].cuda()
            initial = models['reference'](left, right, iters=0)[0][0, 0, ::4, ::4]
            initial_error = (initial-low_gt).abs()
            errors = {}
            captured.clear()
            for name in ('baseline', 'guided', 'guided_off'):
                model = models['baseline'] if name == 'baseline' else models['guided']
                if name == 'guided_off':
                    del model.top8_guidance
                try:
                    predictions = model(left, right, iters=32)
                    errors[name] = {it: (predictions[it][0, 0, ::4, ::4]-low_gt).abs() for it in iterations}
                    if any(not torch.isfinite(e[low_consistent]).all() for e in errors[name].values()):
                        raise FloatingPointError(f'{name}: {item["image"]}')
                    del predictions
                finally:
                    if name == 'guided_off':
                        model.top8_guidance = adapter
            if not captured:
                raise RuntimeError('No evidence consumed')
            peak_errors = (captured['disparities'][0]-low_gt).abs().masked_fill(~captured['valid'][0], torch.inf)
            best_error = peak_errors.amin(0)
            hit = best_error <= 8
            scopes = dict(lr_consistent=low_consistent,
                large_lr_consistent=low_consistent & (low_gt >= 384) & (low_gt < 752),
                initial_bad16_lr_consistent=low_consistent & (initial_error > 16),
                large_initial_bad16_lr_consistent=low_consistent & (low_gt >= 384) & (low_gt < 752) & (initial_error > 16),
                initial_good4_lr_consistent=low_consistent & (initial_error <= 4))
            image_coverage = []
            image_metrics = []
            for scope, mask in scopes.items():
                n = int(mask.sum())
                counts = np.array([n, *(int(((best_error <= t) & mask).sum()) for t in (4, 8, 16))], dtype=np.float64)
                coverage_totals[scope] = coverage_totals.get(scope, np.zeros(4)) + counts
                image_coverage.append(dict(scope=scope, queries=n,
                    recall8_pct=100*counts[2]/n if n else None))
                for group, membership in [('all', mask), ('hit8', mask & hit), ('miss8', mask & ~hit)]:
                    for it in iterations:
                        b = errors['baseline'][it][membership]
                        g = errors['guided'][it][membership]
                        off = errors['guided_off'][it][membership]
                        count = b.numel()
                        sums = np.array([count, b.double().sum().item(), g.double().sum().item(), off.double().sum().item(),
                            int((b <= 8).sum()), int((g <= 8).sum()), int((off <= 8).sum()),
                            int((g < b).sum()), int((g < off).sum()), int((g > off+4).sum())], dtype=np.float64)
                        key = (scope, group, it)
                        totals[key] = totals.get(key, np.zeros(10)) + sums
                        image_metrics.append(dict(scope=scope, group=group, iteration=it, queries=count,
                            epe=[float(v/count) for v in sums[1:4]] if count else [None]*3))
            stream.write(json.dumps(dict(image=item['image'], crop_y=item['crop_y'], coverage=image_coverage, metrics=image_metrics))+'\n')
            stream.flush()
            if number == 1 or number % 10 == 0 or number == len(crops):
                print(f'Completed {number}/{len(crops)} pairs', flush=True)
            del left, right, errors, initial, peak_errors
            captured.clear()
    coverage = [dict(scope=s, queries=int(v[0]), **{f'recall{t}_pct':100*v[i]/v[0] if v[0] else None
                for i, t in enumerate((4, 8, 16), 1)}) for s, v in coverage_totals.items()]
    metrics = []
    for (scope, group, it), v in totals.items():
        n = v[0]
        metrics.append(dict(scope=scope, group=group, iteration=it, queries=int(n),
            epe=[float(x/n) for x in v[1:4]] if n else [None]*3,
            recall8_pct=[float(100*x/n) for x in v[4:7]] if n else [None]*3,
            guided_better_than_baseline_pct=100*v[7]/n if n else None,
            guided_better_than_off_pct=100*v[8]/n if n else None,
            guided_worse_than_off_by4_pct=100*v[9]/n if n else None))
    payload = dict(checkpoints={k:str(v) for k,v in checkpoints.items()},samples=[r['image'] for r in crops],
        models_in_metric_order=['baseline', 'guided', 'guided_off'],mixed_precision=False,
        activation_recomputation=False,delta_clamp=False,grid='native 1/4 samples; errors in original pixels',
        hit_definition='actual trained adapter Top8 candidates, min abs disparity error <=8 original pixels',
        off_note='same guided weights, adapter bypassed only during inference; a distribution-shift sensitivity check',
        coverage=coverage,metrics=metrics)
    (args.output/'results.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
    lines = ['# Top8 候选覆盖与使用诊断', '',
        f'固定 {len(crops)} 对图像，沿用此前 GT 选取的横条；FP32，无权重重计算。',
        '原生 1/4 网格统计，误差为原图像素；不能与此前整分辨率 EPE 直接混用。', '',
        '| 区域 | 查询数 | Top8 ≤4 px % | ≤8 px % | ≤16 px % |',
        '|---|---:|---:|---:|---:|']
    fmt = lambda value: '—' if value is None else f'{value:.4f}'
    for row in coverage:
        lines.append(f"| {row['scope']} | {row['queries']} | {fmt(row['recall4_pct'])} | {fmt(row['recall8_pct'])} | {fmt(row['recall16_pct'])} |")
    lines += ['', '| 更新 / 区域 / 候选组 | 查询数 | baseline EPE | 引导版 EPE | 同权重关闭引导 EPE |',
                  '|---|---:|---:|---:|---:|']
    for row in metrics:
        if row['iteration'] in (8,32) and row['scope'] in ('large_lr_consistent','initial_bad16_lr_consistent'):
            b,g,off = row['epe']
            lines.append(f"| {row['iteration']} / {row['scope']} / {row['group']} | {row['queries']} | {fmt(b)} | {fmt(g)} | {fmt(off)} |")
    lines += ['', 'hit8/miss8 根据引导版实际 Top8 候选划分，三种推理使用相同像素。',
        '训练后的 baseline 和引导版有不同的模型权重；同权重关闭引导用于敏感性分析，存在输入分布变化。',
        '候选命中不代表最终更新成功；结果只描述这批按 GT 筛选的大视差横条。']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Finished: {args.output}',flush=True)


if __name__=='__main__':
    main()
