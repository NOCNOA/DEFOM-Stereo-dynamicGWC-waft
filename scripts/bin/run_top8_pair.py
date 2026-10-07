"""Train the matched no-clamp baseline/Top8 pair, then evaluate automatically."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TAG = os.environ.get('TOP8_RUN_TAG', '20261005')
OUT = Path('/var/tmp/codex-home-u2025140689/Documents/Codex/2026-10-04/gru-cost-volume-costvolume/outputs')/f'top8_pair_{TAG}'
SOURCE = ROOT/'checkpoints/defomstereo_vitl_corga_bins_48_avg_ft20k_control/checkpoint_latest.pth'
MANIFEST = OUT.parent/'unreal_dataset_audit/pilot_manifest.json'
PYTHON = '/home/u2025140689/anaconda3/envs/defomstereo/bin/python'


def run_arm(arm,gpu):
    name=f'bins_top8_pair_{TAG}_{arm}_2k_b1a2'
    folder=ROOT/'checkpoints'/name
    if folder.exists():
        raise FileExistsError(folder)
    env=os.environ.copy()
    env['CUDA_VISIBLE_DEVICES']=str(gpu)
    env['LD_LIBRARY_PATH']='/home/u2025140689/anaconda3/envs/defomstereo/lib:'+env.get('LD_LIBRARY_PATH','')
    command=[PYTHON,'-u','train_stereo.py','--model_arch','cor_ga_bins','--launcher','none','--gpu_ids','0',
        '--name',name,'--resume_ckpt',str(SOURCE),'--no_resume_optimizer',
        '--batch_size','1','--gradient_accumulation_steps','2','--num_workers','2',
        '--train_datasets','sceneflow','--train_folds','1','--num_steps','2000',
        '--save_latest_ckpt_freq','1000','--save_ckpt_freq','2000','--val_freq','2000',
        '--valid_iters','32','--train_iters','18','--n_downsample','2','--num_init_bins','48',
        '--init_max_offset','48','--init_corr_scale','10','--loss_gamma','0.9',
        '--init_target_temperature','1.0','--gate_loss_weight','0.1','--gate_target_temperature','4',
        '--smooth_l1_beta','1','--d0_loss_weight','1','--train_max_disp','768','--eval_max_disp','768',
        '--corr_levels','3','--corr_radius','4','--dinov2_encoder','vitl',
        '--init_right_compression','avg','--refine_right_compression','avg',
        '--context_norm','batch','--n_gru_layers','3','--hidden_dims','128',
        '--image_size','256','768','--wdecay','0.00001','--seed','1234565',
        '--mixed_precision','--amp_dtype','fp16','--debug_nan','--amp_init_scale','64',
        '--disable_activation_checkpoint','--lr','0.00002']
    command+=['--enable_top8_guidance'] if arm=='guided' else ['--strict_resume']
    (OUT/f'{arm}_command.json').write_text(json.dumps(command,indent=2))
    (OUT/f'{arm}_status.txt').write_text('training\n')
    with (OUT/f'{arm}_train.log').open('w') as stream:
        subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    log=(OUT/f'{arm}_train.log').read_text(errors='replace')
    if 'FINISHED TRAINING' not in log:
        raise RuntimeError(f'{arm} did not finish training')
    (OUT/f'{arm}_status.txt').write_text('evaluating_unreal\n')
    evaluation=[PYTHON,'-u','scripts/bin/evaluate_top8_unreal.py','--checkpoint',str(folder/'checkpoint_latest.pth'),
                '--reference-checkpoint',str(SOURCE),'--manifest',str(MANIFEST),'--output',str(OUT/f'{arm}_unreal')]
    if arm=='guided':evaluation.append('--guided')
    with (OUT/f'{arm}_eval.log').open('w') as stream:
        subprocess.run(evaluation,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    (OUT/f'{arm}_status.txt').write_text('complete\n')


def summarize():
    results={arm:json.loads((OUT/f'{arm}_unreal/results.json').read_text()) for arm in ('baseline','guided')}
    if results['baseline']['samples']!=results['guided']['samples']:
        raise ValueError('Sample mismatch')
    if [(r['iteration'],r['scope'],r['pixels']) for r in results['baseline']['metrics']] != [(r['iteration'],r['scope'],r['pixels']) for r in results['guided']['metrics']]:
        raise ValueError('Evaluation mask/count mismatch')
    lines=['# Top8 引导配对微调','',
           '同一 20k 权重、2000 优化步、有效 batch 2；两组均无 Δd 限幅、无权重重计算。',
           'Unreal 为固定 10 对中央横条诊断，不能代表全测试集。初始误差分组按同一原始 20k 权重确定。','',
           '| 评估区域 / 指标 | baseline | Top8 |','|---|---:|---:|']
    for scope in ('lr_consistent','large_lr_consistent','initial_bad16_lr_consistent','initial_good4_lr_consistent'):
        for metric in ('epe','recall8_pct','bad4_pct'):
            values=[]
            for arm in ('baseline','guided'):
                row=next(r for r in results[arm]['metrics'] if r['iteration']==32 and r['scope']==scope)
                values.append('—' if row[metric] is None else f'{row[metric]:.4f}')
            lines.append(f'| {scope} / {metric} | {values[0]} | {values[1]} |')
    validation={}
    for arm in ('baseline','guided'):
        matches=re.findall(r'Validation FlyingThings: EPE ([\d.]+), Out3\.0 ([\d.]+)',(OUT/f'{arm}_train.log').read_text(errors='replace'))
        validation[arm]=matches[-1] if matches else None
    lines+=['',f'全量 FlyingThings TEST（EPE, Out3）：{validation}']
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n')
    (OUT/'comparison.json').write_text(json.dumps(dict(unreal=results,sceneflow_validation=validation),indent=2))


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    files=['core/bin/defom_cor_ga_bins.py','core/bin/top8_guidance.py','core/bin/peak_selection.py','train_stereo.py']
    (OUT/'config.json').write_text(json.dumps(dict(source=str(SOURCE),source_size=SOURCE.stat().st_size,
        source_mtime=SOURCE.stat().st_mtime,manifest=str(MANIFEST),steps=2000,seed=1234565,
        activation_recomputation=False,delta_clamp=False,
        code_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}),indent=2))
    (OUT/'status.txt').write_text('running\n')
    errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures={executor.submit(run_arm,arm,gpu):arm for arm,gpu in [('baseline',2),('guided',3)]}
        for future in concurrent.futures.as_completed(futures):
            arm=futures[future]
            try:future.result()
            except Exception as exc:
                errors.append(f'{arm}: {exc}')
                (OUT/f'{arm}_status.txt').write_text(f'failed: {exc}\n')
    if errors:
        (OUT/'status.txt').write_text('\n'.join(errors)+'\n')
        raise RuntimeError('; '.join(errors))
    summarize()
    (OUT/'status.txt').write_text('complete\n')
    print(OUT/'comparison.md',flush=True)


if __name__=='__main__':
    main()
