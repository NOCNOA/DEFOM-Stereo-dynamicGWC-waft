"""Diagnose bin candidate coverage, ranking and fusion without training.

All candidate comparisons use the native feature lattice and original-image
pixel units. Full-resolution d0 metrics are recorded separately. No activation
checkpointing, gradient computation, or model modification is used.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.bin.bin_peak_metrics import candidate_errors, nearest_gt_bin_rank


THRESHOLDS = (2, 4, 8, 16, 32)
EXCESS_THRESHOLDS = (4, 8, 16)
BANDS = ((0, 48), (48, 96), (96, 188), (188, 376),
         (376, 384), (384, 752), (752, 768))


def native_geometry(gt, valid, padding, shape, factor=4):
    """Nearest GT samples at x=factor*j-pad_left, y=factor*i-pad_top."""
    height, width = gt.shape
    left, right, top, bottom = padding
    if (height + top + bottom, width + left + right) != (
            shape[0] * factor, shape[1] * factor):
        raise ValueError('Unexpected feature lattice / image padding relationship')
    finite_valid = valid.bool() & torch.isfinite(gt) & (gt >= 0) & (gt < 768)
    clean = torch.where(finite_valid, gt, torch.zeros_like(gt))
    low_gt = F.pad(clean[None, None], padding)[0, 0, ::factor, ::factor]
    low_valid = F.pad(finite_valid[None, None].float(), padding)[0, 0, ::factor, ::factor].bool()
    x = torch.arange(shape[1], device=gt.device).float()[None].expand(shape) * factor - left
    in_right = low_valid & (x - low_gt >= 0) & (x - low_gt <= width - 1)
    # A fixed physical threshold avoids labeling arbitrary flat pixels as edges.
    edge = torch.zeros_like(finite_valid)
    horizontal = finite_valid[:, 1:] & finite_valid[:, :-1] & ((clean[:, 1:] - clean[:, :-1]).abs() > 3)
    vertical = finite_valid[1:] & finite_valid[:-1] & ((clean[1:] - clean[:-1]).abs() > 3)
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    edge[1:] |= vertical
    edge[:-1] |= vertical
    edge = F.max_pool2d(edge[None, None].float(), 5, stride=1, padding=2)[0, 0].bool() & finite_valid
    low_edge = F.pad(edge[None, None].float(), padding)[0, 0, ::factor, ::factor].bool()
    return low_gt, low_valid, in_right, low_edge, x


class Metrics:
    def __init__(self):
        self.sums = {}

    def add(self, cohorts, errors, scopes, gt):
        names = list(errors)
        stacked = np.stack([errors[name].detach().cpu().numpy() for name in names])
        gt = gt.detach().cpu().numpy()
        reference = stacked[names.index('all_bin_oracle')]
        bands = [('all', np.ones_like(gt, dtype=bool))]
        bands += [(f'{lo}-{hi}', (gt >= lo) & (gt < hi)) for lo, hi in BANDS]
        for scope, tensor in scopes.items():
            base = tensor.detach().cpu().numpy()
            for band, band_mask in bands:
                mask = base & band_mask
                count = int(mask.sum())
                if not count:
                    continue
                selected = stacked[:, mask]
                finite = np.isfinite(selected)
                ref = reference[mask]
                ref_finite = np.isfinite(ref)
                excess = selected[:, ref_finite] - ref[None, ref_finite]
                values = [np.full(len(names), count), finite.sum(1),
                          np.where(finite, selected, 0).sum(1, dtype=np.float64)]
                values += [(selected <= threshold).sum(1) for threshold in THRESHOLDS]
                values += [np.full(len(names), int(ref_finite.sum()))]
                values += [(excess <= threshold).sum(1) for threshold in EXCESS_THRESHOLDS]
                matrix = np.stack(values, axis=1).astype(np.float64)
                for cohort in cohorts:
                    for i, name in enumerate(names):
                        key = (cohort, scope, band, name)
                        if key not in self.sums:
                            self.sums[key] = np.zeros(matrix.shape[1], dtype=np.float64)
                        self.sums[key] += matrix[i]

    def rows(self):
        result = []
        for (cohort, scope, band, method), sums in sorted(self.sums.items()):
            n, finite_n, total = sums[:3]
            row = dict(cohort=cohort, scope=scope, gt_band=band, method=method,
                       queries=int(n), finite_queries=int(finite_n),
                       missing_candidate_pct=100 * (n - finite_n) / n,
                       epe=total / finite_n if finite_n else None)
            for i, threshold in enumerate(THRESHOLDS):
                row[f'recall_{threshold}px_pct'] = 100 * sums[3 + i] / n
            ref_n = sums[3 + len(THRESHOLDS)]
            row['reference_queries'] = int(ref_n)
            for i, threshold in enumerate(EXCESS_THRESHOLDS):
                row[f'within_reference_plus_{threshold}px_pct'] = (
                    100 * sums[4 + len(THRESHOLDS) + i] / ref_n if ref_n else None)
            result.append(row)
        return result


def write_report(output, metadata, metrics, ancillary):
    rows = metrics.rows()
    (output / 'results.json').write_text(json.dumps(
        dict(metadata=metadata, ancillary=ancillary, metrics=rows), indent=2, allow_nan=False))
    with (output / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Bin 多峰候选诊断', '',
             f"权重：`{metadata['checkpoint']}`；记录步数：{metadata['checkpoint_step']}。",
             '本实验只做初始化诊断；没有训练、没有激活重计算。', '',
             '## 统计口径', '',
             '- 候选指标在原生 1/4 网格上计算，误差单位为原图像素；没有插值候选索引。',
             '- 有效候选同时满足模型 init_valid 和原图右对应位置在界内；GT 界内不等于非遮挡。',
             '- 每个分支提取真正的局部峰，再按物理视差距离做 NMS；不足 K 个峰不补假候选。',
             f"- NMS 最小间隔 {metadata['min_separation_px']} px；分支 bin 间隔 {metadata['bin_steps_px']} px。",
             '- topK_oracle 是每分支 K 个峰的联合 GT 选择结果，只用于诊断，不能作为可部署精度。',
             '- all_bin_oracle 是离散候选的量化/覆盖参考；连续视差回归可以优于它。',
             '- EPE 对存在候选的查询汇总；召回率把缺失候选计为失败。',
             '- reference+8 表示候选误差不超过最近离散 bin 误差加 8 px，排除粗间隔的部分影响。',
             '- raw_top4_oracle 不做峰去重，用于观察 NMS 是否丢掉邻近候选。',
             '- branch_oracle 在存在有效候选的分支均值中选最优；与历史 GT-range-filtered oracle 口径不同。',
             '- full-resolution d0 单独报告，不能与原生网格的 oracle EPE 直接比较。', '']
    lines += ['Unreal 固定样本中央横条诊断；不代表整图或全测试集结果。每张图的分组指标存入 per_image.jsonl。', '']
    show = ['fused', 'branch_oracle', 'top1_oracle', 'top2_oracle', 'top4_oracle', 'top8_oracle',
            'branch1_top8_oracle', 'branch2_top8_oracle',
            'branch3_top1_oracle', 'branch3_top2_oracle', 'branch3_top4_oracle', 'branch3_top8_oracle',
            'raw_top4_oracle', 'all_bin_oracle']
    for cohort in metadata['cohorts']:
        lines += [f'## {cohort}', '',
                  f"样本数：{len(metadata['cohorts'][cohort])}。完整分辨率 d0 EPE：{ancillary[cohort]['full_d0_epe']:.4f} px。", '']
        for scope, band in [('gt_in_right', 'all'), ('covered_in_right', 'all'),
                            ('covered_in_right', '384-752'), ('init_bad16_in_right', 'all'),
                            ('boundary_in_right', 'all'), ('covered_lr_consistent', '384-752'),
                            ('covered_bad16_lr_consistent', 'all'), ('lr_consistent', 'all'),
                            ('branch1_covered_lr_consistent', 'all'),
                            ('branch2_covered_lr_consistent', 'all'),
                            ('branch3_covered_lr_consistent', 'all')]:
            selected = {r['method']: r for r in rows if r['cohort'] == cohort
                        and r['scope'] == scope and r['gt_band'] == band}
            if not selected:
                continue
            lines += [f'### {scope} / GT {band}', '',
                      '| 方法 | 查询数 | EPE | ≤4px % | ≤8px % | reference+8 % | 缺失候选 % |',
                      '|---|---:|---:|---:|---:|---:|---:|']
            for method in show:
                r = selected[method]
                def fmt(value):
                    return '—' if value is None else f'{value:.3f}'
                lines.append(f"| {method} | {r['queries']} | {fmt(r['epe'])} | "
                             f"{fmt(r['recall_4px_pct'])} | {fmt(r['recall_8px_pct'])} | "
                             f"{fmt(r['within_reference_plus_8px_pct'])} | {fmt(r['missing_candidate_pct'])} |")
            lines.append('')
    lines += ['## 完整输出', '',
              '- `metrics.csv`：按视差段、GT 边界、右图界内/界外和初始误差分组。',
              '- `results.json`：样本路径、checkpoint、精度配置、候选排名和覆盖统计。',
              '- `per_image.jsonl`：逐图指标与 cohort，便于定位失败图像。',
              '- 诊断显存包含模型和候选分析张量，不是模型独立前向或训练显存。',
              '- 推理统计包含特征提取和公共 FullCorr 构建，不是初始化头独立性能或训练显存测试。', '']
    (output / 'report.md').write_text('\n'.join(lines))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--root', default='/home/share/yijiayi/sceneflow')
    p.add_argument('--max-samples', type=int, default=400)
    p.add_argument('--seed', type=int, default=20260815)
    p.add_argument('--enriched-manifest')
    p.add_argument('--sample-manifest', required=True)
    p.add_argument('--crop-height', type=int, default=512)
    p.add_argument('--lr-threshold', type=float, default=1.)
    p.add_argument('--min-separation-px', type=float, default=8.)
    p.add_argument('--mixed-precision', action='store_true')
    p.add_argument('--dinov2-encoder', default='vitl')
    p.add_argument('--init-right-compression', choices=['avg', 'conv'], default='avg')
    p.add_argument('--refine-right-compression', choices=['avg', 'conv'], default='avg')
    args = p.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(visible) != 1 or visible[0] not in ('0', '2', '3'):
        p.error('Explicitly select physical GPU 0, 2 or 3; GPU 1 is reserved.')
    if args.max_samples <= 0 or args.min_separation_px < 0:
        p.error('max-samples must be positive and separation nonnegative')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'per_image.jsonl').exists():
        raise FileExistsError('Use a new output directory; existing diagnostics are preserved')
    from core.bin.defom_cor_ga_bins import DEFOMStereo
    from core.mixed_validation import UnrealStereo4K
    from core.utils.utils import InputPadder
    from types import SimpleNamespace

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model_args = SimpleNamespace(
        dinov2_encoder=args.dinov2_encoder, hidden_dims=[128], context_norm='batch',
        n_gru_layers=3, n_downsample=2, mixed_precision=args.mixed_precision,
        num_init_bins=48, init_max_offset=48., init_corr_scale=10.,
        max_delta_disp=8., corr_levels=3, corr_radius=4, train_max_disp=768.,
        init_right_compression=args.init_right_compression,
        refine_right_compression=args.refine_right_compression,
        disable_activation_checkpoint=True)
    dataset = UnrealStereo4K(args.root, scene='all')
    manifest = json.loads(Path(args.sample_manifest).read_text())
    lookup = {str(Path(pair[0]).resolve()): i for i, pair in enumerate(dataset.image_list)}
    indices = [lookup[str(Path(path).resolve())] for path in manifest['sample_paths']]
    cohorts = {f'unreal_fixed_{len(indices)}': indices}
    memberships = {}
    for name, group in cohorts.items():
        for index in group:
            memberships.setdefault(index, []).append(name)
    checkpoint = torch.load(str(args.checkpoint), map_location='cpu', mmap=True, weights_only=False)
    model = DEFOMStereo(model_args)
    model.load_state_dict(checkpoint.get('model', checkpoint), strict=True)
    checkpoint_step = checkpoint.get('step')
    del checkpoint
    model.cuda().eval()
    bins = model.bin_initializer.bin_disparities.float() * 4
    metadata = dict(
        checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_step=checkpoint_step,
        checkpoint_size_bytes=Path(args.checkpoint).stat().st_size,
        checkpoint_mtime=Path(args.checkpoint).stat().st_mtime,
        model_args=vars(model_args), seed=args.seed, physical_gpu=visible[0],
        device=torch.cuda.get_device_name(), torch_version=torch.__version__,
        activation_recomputation=False, refinement_iterations=0,
        min_separation_px=args.min_separation_px, peak_counts=[1, 2, 4, 8],
        bin_steps_px=(bins[:, 1] - bins[:, 0]).cpu().tolist(),
        bin_maxima_px=bins[:, -1].cpu().tolist(),
        cohorts={name: [dataset.image_list[i][0] for i in group] for name, group in cohorts.items()},
        cohort_note='Fixed readable pilot samples from systematic scene audit, not random or full test-set evaluation.',
        dataset='UnrealStereo4K', crop_height=args.crop_height, crop_rule='central horizontal strip, full width, no resize',
        lr_threshold_px=args.lr_threshold, lr_note='GT consistency is a visibility proxy; not a definitive occlusion mask',
        grid='nearest GT on native lattice; original pixels; x=4*j-pad_left',
        candidate_domain='model-valid AND original right correspondence center in bounds',
        boundary_definition='GT adjacent jump >3px, dilated by 5x5, valid neighbors only')
    (output / 'config.json').write_text(json.dumps(metadata, indent=2))
    metrics = Metrics()
    ancillary = {name: dict(images=0, full_count=0, full_error_sum=0., low_valid=0,
                            in_right=0, covered=0, rank_count=0, nearest_bin_rank_sum=0.,
                            nearest_bin_top4=0, selected_peaks_sum=0.) for name in cohorts}
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with (output / 'per_image.jsonl').open('w') as file:
        for number, (index, groups) in enumerate(memberships.items(), 1):
            data = dataset[index]
            full_height = data['disp'].shape[-2]
            y0 = max(0, (full_height - args.crop_height) // 2)
            y1 = min(full_height, y0 + args.crop_height)
            for key in ('img1', 'img2', 'disp', 'valid'):
                data[key] = data[key][..., y0:y1, :]
            right_path = str(Path(data['disp_file']).parent.parent / 'Disp1' / Path(data['disp_file']).name)
            right_gt = torch.from_numpy(np.load(right_path, allow_pickle=False)[y0:y1].copy()).cuda().float()
            gt = data['disp'][0].cuda().float()
            valid = data['valid'][0].cuda().bool() & torch.isfinite(gt) & (gt >= 0) & (gt < 768)
            padder = InputPadder(data['img1'][None].shape, divis_by=32)
            left, right = padder.pad(data['img1'][None].cuda(), data['img2'][None].cuda())
            predictions, aux = model(left, right, iters=0, return_aux=True)
            logits = aux['init_logits'][0].float()
            branch_disps = aux['init_disps'][0].float() * 4
            fused = (aux['w_h'][0].float() * branch_disps).sum(0).clamp(0, 768)
            low_gt, low_valid, in_right, boundary, x = native_geometry(
                gt, valid, padder._pad, logits.shape[-2:])
            x_right = x[None, None] - bins[:, :, None, None]
            candidate_valid = aux['init_valid'][0] & (x_right >= 0) & (x_right <= gt.shape[1] - 1)
            if not torch.isfinite(logits[candidate_valid]).all():
                raise FloatingPointError('Nonfinite valid bin logits')
            diagnosis = candidate_errors(logits, candidate_valid, bins, branch_disps,
                                         fused, low_gt, peak_counts=(1, 2, 4, 8),
                                         min_separation_px=args.min_separation_px)
            errors = diagnosis['errors']
            # Diagnose every branch independently of the union.
            for branch in range(3):
                peak_disps = bins[branch][diagnosis['peak_indices'][branch]]
                peak_errors = (peak_disps - low_gt).abs().masked_fill(
                    ~diagnosis['peak_valid'][branch], torch.inf)
                for count in (1, 2, 4, 8):
                    errors[f'branch{branch + 1}_top{count}_oracle'] = peak_errors[:count].amin(0)
            raw_scores, raw_indices = logits.masked_fill(~candidate_valid, -torch.inf).topk(4, dim=1)
            raw_bins = bins[:, :, None, None].expand_as(logits).gather(1, raw_indices)
            errors['raw_top4_oracle'] = (raw_bins - low_gt).abs().masked_fill(
                ~torch.isfinite(raw_scores), torch.inf).flatten(0, 1).amin(0)
            for branch in range(3):
                errors[f'branch_{branch + 1}'] = (branch_disps[branch] - low_gt).abs().masked_fill(
                    ~candidate_valid[branch].any(0), torch.inf)
            available_bins = bins[:, :, None, None].expand_as(logits)
            minimum = available_bins.masked_fill(~candidate_valid, torch.inf).amin(1)
            maximum = available_bins.masked_fill(~candidate_valid, -torch.inf).amax(1)
            branch_covered = (low_gt >= minimum) & (low_gt <= maximum) & candidate_valid.any(1)
            covered = branch_covered.any(0) & in_right
            scopes = dict(valid_gt=low_valid, gt_in_right=in_right,
                          gt_outside_right=low_valid & ~in_right,
                          covered_in_right=covered, uncovered_in_right=in_right & ~covered,
                          boundary_in_right=in_right & boundary,
                          interior_in_right=in_right & ~boundary,
                          init_good4_in_right=in_right & (errors['fused'] <= 4),
                          init_bad16_in_right=in_right & (errors['fused'] > 16))
            # Coordinates are unchanged horizontally by the strip crop.
            lr_x = x - low_gt
            lr_y = torch.arange(low_gt.shape[0], device=gt.device)[:, None].expand_as(low_gt).float() * 4 - padder._pad[2]
            sample_grid = torch.stack((2 * lr_x / (gt.shape[1] - 1) - 1,
                                       2 * lr_y / (gt.shape[0] - 1) - 1), -1)
            sampled_right = F.grid_sample(right_gt[None, None], sample_grid[None], align_corners=True)[0, 0]
            consistent = in_right & torch.isfinite(sampled_right) & (sampled_right > 0) & ((sampled_right-low_gt).abs() <= args.lr_threshold)
            scopes.update(lr_consistent=consistent, lr_inconsistent=in_right & ~consistent,
                          covered_lr_consistent=covered & consistent,
                          covered_bad16_lr_consistent=covered & consistent & (errors['fused'] > 16),
                          interior_lr_consistent=consistent & ~boundary,
                          boundary_lr_consistent=consistent & boundary)
            for branch in range(3):
                scopes[f'branch{branch + 1}_covered_lr_consistent'] = consistent & branch_covered[branch]
            metrics.add(groups, errors, scopes, low_gt)
            image_metrics = Metrics()
            image_metrics.add(groups, errors, scopes, low_gt)
            rank = nearest_gt_bin_rank(logits, candidate_valid, bins, low_gt)
            rank_mask = branch_covered & in_right[None]
            full_pred = padder.unpad(predictions[0])[0, 0]
            full_error = (full_pred - gt).abs()[valid]
            if not torch.isfinite(full_error).all():
                raise FloatingPointError('Nonfinite full-resolution d0')
            row = dict(index=index, image=data['imageL_file'], cohorts=groups,
                       full_count=full_error.numel(), full_error_sum=full_error.double().sum().item(),
                       low_valid=int(low_valid.sum()), in_right=int(in_right.sum()),
                       covered=int(covered.sum()), rank_count=int(rank_mask.sum()),
                       nearest_bin_rank_sum=rank[rank_mask].double().sum().item(),
                       nearest_bin_top4=int((rank[rank_mask] <= 4).sum()),
                       selected_peaks_sum=diagnosis['peak_valid'][:, :, in_right].sum().item())
            row['crop_y'] = [y0, y1]
            row['metrics'] = image_metrics.rows()
            row['native_in_right'] = {}
            for name, error in errors.items():
                values = error[in_right]
                finite = torch.isfinite(values)
                row['native_in_right'][name] = dict(
                    epe=values[finite].double().mean().item() if finite.any() else None,
                    recall8_pct=100 * (values <= 8).float().mean().item() if values.numel() else None)
            file.write(json.dumps(row, allow_nan=False) + '\n')
            file.flush()
            for cohort in groups:
                ancillary[cohort]['images'] += 1
                for key in ancillary[cohort]:
                    if key != 'images':
                        ancillary[cohort][key] += row[key]
            del predictions, aux, logits, diagnosis, errors, left, right, available_bins, right_gt, sampled_right, image_metrics
            if number % 50 == 0 or number == len(memberships):
                print(f'Completed {number}/{len(memberships)} unique images', flush=True)
    metadata['elapsed_seconds'] = time.monotonic() - start
    metadata['peak_inference_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
    for values in ancillary.values():
        values['full_d0_epe'] = values['full_error_sum'] / max(values['full_count'], 1)
        values['in_right_coverage_pct'] = 100 * values['covered'] / max(values['in_right'], 1)
        values['mean_nearest_bin_rank'] = values['nearest_bin_rank_sum'] / max(values['rank_count'], 1)
        values['nearest_bin_top4_pct'] = 100 * values['nearest_bin_top4'] / max(values['rank_count'], 1)
        values['mean_selected_peaks_total'] = values['selected_peaks_sum'] / max(values['in_right'], 1)
    write_report(output, metadata, metrics, ancillary)
    print(f'Finished. Report: {output / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
