"""Train sample-wise Top8 guidance and compare on the saved 96-image protocol."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
WORK = Path('/var/tmp/codex-home-u2025140689/Documents/Codex/2026-10-04/gru-cost-volume-costvolume')
OUT = WORK/'outputs/top8_samples_20261006'
PREVIOUS = WORK/'outputs/unreal_large_balanced_100/evaluation_v1'
NAME = 'bins_top8_samples_20261006_2k_b1a2'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    if (ROOT/'checkpoints'/NAME).exists():
        raise FileExistsError(NAME)
    status = OUT/'status.txt'
    status.write_text('training\n')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '3'
    env['LD_LIBRARY_PATH'] = '/home/u2025140689/anaconda3/envs/defomstereo/lib:' + env.get('LD_LIBRARY_PATH', '')
    command = json.loads((WORK/'outputs/top8_pair_20261005_rerun2/guided_command.json').read_text())
    command[command.index('--name')+1] = NAME
    # The previous FP16 built-in validation failed despite finite FP32 inference.
    # This pilot uses the existing FP32 evaluation after saving all 2000 steps.
    command[command.index('--val_freq')+1] = '2001'
    command += ['--top8_guidance_mode', 'samples']
    files = ['core/bin/defom_cor_ga_bins.py', 'core/bin/top8_sample_guidance.py',
             'core/bin/top8_guidance.py', 'core/bin/peak_selection.py',
             'core/update_pru.py', 'train_stereo.py', 'scripts/bin/evaluate_top8_unreal.py']
    snapshot = OUT/'source'
    for filename in files:
        target = snapshot/filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/filename).read_bytes())
    config = dict(training_command=command, gpu=3, steps=2000, seed=1234565,
                  global_token_dim=32, query_dim=8, local_queries=27, activation_recomputation=False,
                  delta_clamp=False, baseline='reuse matched baseline from rerun2',
                  previous_results=str(PREVIOUS),
                  code_sha256={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    (OUT/'config.json').write_text(json.dumps(config, indent=2))
    try:
        with (OUT/'train.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if 'FINISHED TRAINING' not in (OUT/'train.log').read_text(errors='replace'):
            raise RuntimeError('Training did not finish')
        status.write_text('evaluating_96_unreal_fp32\n')
        evaluation = [sys.executable, '-u', 'scripts/bin/evaluate_top8_unreal.py',
                      '--checkpoint', str(ROOT/'checkpoints'/NAME/'checkpoint_latest.pth'),
                      '--reference-checkpoint', command[command.index('--resume_ckpt')+1],
                      '--manifest', str(PREVIOUS/'evaluation_manifest.json'),
                      '--output', str(OUT/'samples_unreal'), '--guided',
                      '--top8-guidance-mode', 'samples', '--crop-mode', 'large']
        (OUT/'evaluation_command.json').write_text(json.dumps(evaluation, indent=2))
        with (OUT/'eval.log').open('w') as log:
            subprocess.run(evaluation, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        folders = dict(baseline=PREVIOUS/'baseline', fused=PREVIOUS/'guided', samples=OUT/'samples_unreal')
        results = {a: json.loads((f/'results.json').read_text()) for a, f in folders.items()}
        image_results = {a: {r['image']: r for r in map(json.loads, (f/'per_image.jsonl').read_text().splitlines())}
                         for a, f in folders.items()}
        baseline = results['baseline']
        for arm, result in results.items():
            if result['samples'] != baseline['samples']:
                raise ValueError(f'{arm}: samples differ')
            if [(m['iteration'], m['scope'], m['pixels']) for m in result['metrics']] != [
                    (m['iteration'], m['scope'], m['pixels']) for m in baseline['metrics']]:
                raise ValueError(f'{arm}: masks differ')
            for path, row in image_results['baseline'].items():
                other = image_results[arm][path]
                if row['crop_y'] != other['crop_y'] or [m['pixels'] for m in row['metrics']] != [m['pixels'] for m in other['metrics']]:
                    raise ValueError(f'{arm}: per-image crop/mask differs')
        comparison = []
        lines = ['# 逐采样位置 Top8 引导', '',
                 '同一 20k 起点、2000 步、有效 batch 2、无 Δd 限幅、无重计算。',
                 '复用已完成的匹配 baseline 和旧融合引导结果；固定 96 对 Unreal，FP32，GT 固定横条。', '',
                 '| 迭代 / 区域 | baseline | 旧融合引导 | 逐采样引导 |', '|---|---:|---:|---:|']
        for iteration in (8, 16, 32):
            for scope in ('lr_consistent', 'large_lr_consistent', 'initial_bad16_lr_consistent', 'initial_good4_lr_consistent'):
                rows = {a: next(m for m in r['metrics'] if m['iteration'] == iteration and m['scope'] == scope)
                        for a, r in results.items()}
                values = {a: row['epe'] for a, row in rows.items()}
                comparisons = []
                scenes = {}
                for path, record in image_results['baseline'].items():
                    per_arm = {a: next(m for m in rs[path]['metrics'] if m['iteration'] == iteration and m['scope'] == scope)
                               for a, rs in image_results.items()}
                    if not per_arm['baseline']['pixels']:
                        continue
                    pair = [per_arm[a]['epe'] for a in folders]
                    comparisons.append(pair)
                    scenes.setdefault(Path(path).parent.parent.name, []).append(pair)
                mean = lambda rows: [sum(r[j] for r in rows)/len(rows) for j in range(3)] if rows else [None]*3
                scene_means = {s: mean(r) for s, r in scenes.items()}
                comparison.append(dict(iteration=iteration, scope=scope, pixels=rows['baseline']['pixels'],
                    pixel_epe=values, image_mean_epe=mean(comparisons), per_scene=scene_means,
                    scene_mean_epe=mean(list(scene_means.values())),
                    contributing_images=len(comparisons), improved_vs_baseline=sum(r[2] < r[0] for r in comparisons),
                    improved_vs_fused=sum(r[2] < r[1] for r in comparisons)))
                lines.append(f'| {iteration} / {scope} | ' + ' | '.join('—' if v is None else f'{v:.4f}' for v in values.values()) + ' |')
        (OUT/'comparison.json').write_text(json.dumps(comparison, indent=2))
        (OUT/'comparison.md').write_text('\n'.join(lines)+'\n')
        status.write_text('complete\n')
    except Exception as exc:
        status.write_text(f'failed: {exc}\n')
        raise


if __name__ == '__main__':
    main()
