"""Run the saved bin/Top8 checkpoints on a common manifest and summarize."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'checkpoints/defomstereo_vitl_corga_bins_48_avg_ft20k_control/checkpoint_latest.pth'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'evaluation_status.txt').exists():
        raise FileExistsError('Evaluation already started; use another directory')
    (args.output/'evaluation_status.txt').write_text('running\n')
    original_manifest = json.loads(Path(args.manifest).read_text())
    usable = []
    skipped = []
    for left in original_manifest['sample_paths']:
        try:
            for path in (left, left.replace('/Image0/', '/Image1/')):
                with Image.open(path) as image:
                    if image.size != (3840, 2160):
                        raise ValueError('Unexpected image size')
                    image.verify()
            usable.append(left)
        except (OSError, ValueError, EOFError) as exc:
            skipped.append(dict(left=left, reason=str(exc)))
    if not usable:
        raise RuntimeError('No readable pairs')
    manifest_path = args.output/'evaluation_manifest.json'
    manifest_path.write_text(json.dumps(dict(sample_paths=usable,
        original_manifest=args.manifest, skipped_unreadable=skipped,
        selection=original_manifest['selection']), indent=2))
    print(f'Preflight: {len(usable)} readable pairs; {len(skipped)} rejected', flush=True)

    def evaluate(arm, gpu):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['LD_LIBRARY_PATH'] = '/home/u2025140689/anaconda3/envs/defomstereo/lib:' + env.get('LD_LIBRARY_PATH', '')
        ckpt = ROOT/f'checkpoints/bins_top8_pair_20261005_rerun2_{arm}_2k_b1a2/checkpoint_latest.pth'
        cmd = [sys.executable, '-u', 'scripts/bin/evaluate_top8_unreal.py',
               '--checkpoint', str(ckpt), '--reference-checkpoint', str(SOURCE),
               '--manifest', str(manifest_path), '--output', str(args.output/arm), '--crop-mode', 'large']
        if arm == 'guided':
            cmd.append('--guided')
        (args.output/f'{arm}_command.json').write_text(json.dumps(cmd, indent=2))
        with (args.output/f'{arm}.log').open('w') as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(evaluate, arm, gpu) for arm, gpu in [('baseline', 2), ('guided', 3)]]
            for future in futures:
                future.result()
        results = {a: json.loads((args.output/a/'results.json').read_text()) for a in ('baseline', 'guided')}
        images = {a: {r['image']: r for r in map(json.loads, (args.output/a/'per_image.jsonl').read_text().splitlines())}
                  for a in results}
        if results['baseline']['samples'] != results['guided']['samples']:
            raise ValueError('Sample mismatch')
        if [(m['iteration'], m['scope'], m['pixels']) for m in results['baseline']['metrics']] != [(m['iteration'], m['scope'], m['pixels']) for m in results['guided']['metrics']]:
            raise ValueError('Pixel masks differ')
        for path in images['baseline']:
            if images['baseline'][path]['crop_y'] != images['guided'][path]['crop_y']:
                raise ValueError('Crop mismatch')
        comparisons = []
        for iteration in (8, 16, 32):
            for scope in ('lr_consistent', 'large_lr_consistent', 'initial_bad16_lr_consistent', 'initial_good4_lr_consistent'):
                per_image = []
                scene_values = {}
                for path, item in images['baseline'].items():
                    b = next(m for m in item['metrics'] if m['iteration'] == iteration and m['scope'] == scope)
                    g = next(m for m in images['guided'][path]['metrics'] if m['iteration'] == iteration and m['scope'] == scope)
                    if b['pixels'] != g['pixels']:
                        raise ValueError('Per-image masks differ')
                    if not b['pixels']:
                        continue
                    scene = Path(path).parent.parent.name
                    values = (b['epe'], g['epe'])
                    per_image.append(values)
                    scene_values.setdefault(scene, []).append(values)
                pixel = {a: next(m for m in r['metrics'] if m['iteration'] == iteration and m['scope'] == scope)
                         for a, r in results.items()}
                means = lambda values: [sum(v[j] for v in values)/len(values) for j in (0, 1)] if values else [None, None]
                scene_means = {s: means(v) for s, v in scene_values.items()}
                comparisons.append(dict(iteration=iteration, scope=scope,
                    pixels=pixel['baseline']['pixels'], contributing_images=len(per_image),
                    improved_images=sum(g < b for b, g in per_image),
                    pixel_epe=[pixel[a]['epe'] for a in ('baseline', 'guided')],
                    image_mean_epe=means(per_image), scene_mean_epe=means(list(scene_means.values())),
                    per_scene=scene_means))
        (args.output/'comparison.json').write_text(json.dumps(comparisons, indent=2))
        lines = ['# Unreal 大视差配对评估', '',
                 f"固定 {len(images['baseline'])} 对图像；GT 选取大视差最多的 512 像素高横条，原生宽度、FP32、32 次更新。", '',
                 '图像按 GT 筛选，因此这是大视差专项评估。两组共用原始 20k 初始化误差掩码。', '',
                 '| 更新次数 / 区域 | 像素数 | 有效图像 / 改善图像 | baseline EPE | Top8 EPE |',
                 '|---|---:|---:|---:|---:|']
        for row in comparisons:
            b, g = row['pixel_epe']
            lines.append(f"| {row['iteration']} / {row['scope']} | {row['pixels']} | {row['contributing_images']} / {row['improved_images']} | {b:.4f} | {g:.4f} |" if b is not None else f"| {row['iteration']} / {row['scope']} | 0 | 0 / 0 | — | — |")
        lines += ['', '逐图、逐场景平均和各场景指标见 comparison.json；全部样本均保留，不按预测误差删图。']
        (args.output/'comparison.md').write_text('\n'.join(lines)+'\n')
        (args.output/'evaluation_status.txt').write_text('complete\n')
        print(args.output/'comparison.md', flush=True)
    except Exception as exc:
        (args.output/'evaluation_status.txt').write_text(f'failed: {exc}\n')
        raise


if __name__ == '__main__':
    main()
