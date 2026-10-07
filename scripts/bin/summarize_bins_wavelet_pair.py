"""Summarize a matched bin / wavelet fine-tuning experiment."""

import argparse
import csv
import json
import re
from pathlib import Path


def find_metric(folder, scope, band):
    with (folder / 'metrics.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    matches = [row for row in rows if row['mode'] == 'full'
               and row['source'] == 'iter_32' and row['scope'] == scope
               and row['gt_band'] == band]
    if len(matches) != 1 or not matches[0]['epe']:
        raise ValueError(f'Missing metric: {folder}, {scope}, {band}')
    row = matches[0]
    return int(row['pixels']), float(row['epe']), float(row['bad3_pct'])


def full_validation(log_path):
    text = log_path.read_text(errors='replace')
    matches = re.findall(r'Validation FlyingThings: EPE ([\d.]+), Out3\.0 ([\d.]+)', text)
    if not matches:
        raise ValueError(f'No completed full validation in {log_path}')
    return tuple(float(value) for value in matches[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--baseline_log', type=Path, required=True)
    parser.add_argument('--wavelet_log', type=Path, required=True)
    args = parser.parse_args()
    arms = ('source', 'baseline', 'wavelet')
    samples = [json.loads((args.results / arm / 'results.json').read_text())
               ['metadata']['samples'] for arm in arms]
    if samples[0] != samples[1] or samples[0] != samples[2]:
        raise ValueError('The three evaluations used different samples')
    full = {'baseline': full_validation(args.baseline_log),
            'wavelet': full_validation(args.wavelet_log)}
    metrics = {}
    for arm in arms:
        folder = args.results / arm
        metrics[arm] = {
            'subset_all': find_metric(folder, 'valid_gt', 'all'),
            'large_matchable': find_metric(folder, 'matchable', '384-752'),
        }
    def cell(value):
        return f'{value:.4f}'
    lines = [
        '# Bin 与小波引导：配对微调结果', '',
        '两条训练均从同一 20k 步 bin checkpoint 出发，使用相同的 2000 个优化步、'
        '有效 batch 2、随机种子和训练配置。小波分支只增加引导模块。',
        f'全量 FlyingThings TEST 为训练结束时的验证；大视差诊断使用同一份 GT 筛选的 {len(samples[0])} 对图像。',
        '',
        '| 模型 | 全量 TEST EPE | 大视差子集全部有效像素 EPE | 384–752 px 可匹配像素 EPE | 该区域 >3 px (%) |',
        '|---|---:|---:|---:|---:|',
    ]
    for arm, label in zip(arms, ('原始 20k 权重', '原模型续训 2k', '小波模型续训 2k')):
        all_pixels, all_epe, _ = metrics[arm]['subset_all']
        large_pixels, large_epe, large_bad3 = metrics[arm]['large_matchable']
        full_epe = cell(full[arm][0]) if arm in full else '—'
        lines.append(f'| {label} | {full_epe} | {cell(all_epe)} | {cell(large_epe)} | {cell(large_bad3)} |')
    base_large = metrics['baseline']['large_matchable'][1]
    wave_large = metrics['wavelet']['large_matchable'][1]
    lines += [
        '',
        f'小波相对等步数原模型的大视差 EPE 差值：**{wave_large - base_large:+.4f} px**（负值表示改善）。',
        f'该区域共 {metrics["baseline"]["large_matchable"][0]:,} 个有效且几何上可匹配的像素。',
        '',
        '大视差子集按 GT 筛选，不能代表随机 TEST 分布；配对微调仅 2000 步，结果是初步消融。',
        '原模型的训练验证和大视差诊断分别使用不同汇总方式，表中两列 EPE 不应直接比较。',
        '',
        '[固定样本清单](large_disp_manifest.json) · '
        '[原始权重完整诊断](source/report.md) · '
        '[原模型完整诊断](baseline/report.md) · '
        '[小波模型完整诊断](wavelet/report.md)',
    ]
    (args.results / 'comparison.md').write_text('\n'.join(lines) + '\n')
    (args.results / 'comparison.json').write_text(json.dumps({
        'full_validation': full, 'metrics': metrics, 'sample_count': len(samples[0]),
        'large_epe_delta_wavelet_minus_baseline': wave_large - base_large,
    }, indent=2) + '\n')
    print(args.results / 'comparison.md')


if __name__ == '__main__':
    main()
