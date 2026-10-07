"""Spatial/native token pilot from the original common match8 warm start."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
WORK=Path('/var/tmp/codex-home-u2025140689/Documents/Codex/2026-10-04/gru-cost-volume-costvolume')
PREVIOUS=WORK/'outputs/bin_match8_20261006/pilot_2k'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('native','dense'),default='native')
    args=parser.parse_args()
    out=WORK/f'outputs/bin_match8_spatial_20261006/{args.mode}_pilot_2k'
    out.mkdir(parents=True,exist_ok=False)
    status=out/'status.txt';status.write_text('preparing\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    env['LD_LIBRARY_PATH']='/home/u2025140689/anaconda3/envs/defomstereo/lib:'+env.get('LD_LIBRARY_PATH','')
    try:
        old=json.loads((PREVIOUS/'config.json').read_text())
        unchanged=['core/bin/defom_cor_ga_bins.py','core/bin/defom_cor_ga_bins_match8.py',
                   'core/submodules.py','core/update_pru.py','utils/utils.py']
        for filename in unchanged:
            if hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()!=old['source_sha256'][filename]:
                raise ValueError(f'Control implementation changed: {filename}')
        name=f'bins_match8_spatial_{args.mode}_20261006_2k_b2'
        if (ROOT/'checkpoints'/name).exists():raise FileExistsError(name)
        # Exact same fresh matching weights, data seed, optimizer and schedule;
        # not the previously trained (and regressed) match8 checkpoint.
        command=list(old['commands']['match8'])
        command[0]=sys.executable
        command[command.index('--name')+1]=name
        command[command.index('--model_arch')+1]='cor_ga_bins_match8_spatial'
        command+=['--match_token_grid',args.mode]
        files=unchanged+['core/warp.py','core/bin/defom_cor_ga_bins_match8_spatial.py',
                        'train_stereo.py','evaluate_stereo.py',
                        'scripts/bin/evaluate_bin_match8_subset.py','scripts/bin/run_bin_match8_spatial.py']
        hashes={}
        for filename in files:
            content=(ROOT/filename).read_bytes()
            target=out/'source'/filename;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(content);hashes[filename]=hashlib.sha256(content).hexdigest()
        config=dict(command=command,gpu=3,seed=1234565,steps=2000,batch=2,accumulation=1,
                    token_grid=args.mode,initialization_source='width-only, unchanged',
                    refinement_source='2D average pools at ratios 1/2/4',
                    matching_encoding='right-left(32) + group cosine(4), Conv36->8 + GN4 + GELU',
                    fusion='same 248->64->32, gates applied on quarter grid after token upsampling',
                    activation_recomputation=False,dynamic_gate=False,delta_clamp=False,
                    prior_controls=str(PREVIOUS),source_sha256=hashes,
                    source_sampler_change='Optional height ratio; previous default width sampler verified equivalent',
                    warm_start=command[command.index('--resume_ckpt')+1],
                    note='Two-thousand-step learning pilot; new encoder/fuser initialization is common to the previous controls')
        (out/'config.json').write_text(json.dumps(config,indent=2))
        status.write_text('training\n')
        with (out/'train.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if 'FINISHED TRAINING' not in (out/'train.log').read_text(errors='replace'):
            raise RuntimeError('Training completion marker missing')
        status.write_text('evaluating_fixed_100_test_images_fp32\n')
        evaluation=[sys.executable,'-u','scripts/bin/evaluate_bin_match8_subset.py',
                    '--arch','spatial','--match_token_grid',args.mode,
                    '--checkpoint',str(ROOT/'checkpoints'/name/'checkpoint_latest.pth'),
                    '--output',str(out/'evaluation'),'--samples','100']
        (out/'evaluation_command.json').write_text(json.dumps(evaluation,indent=2))
        with (out/'evaluation.log').open('w') as log:
            subprocess.run(evaluation,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        folders=dict(baseline=PREVIOUS/'baseline_evaluation',match8=PREVIOUS/'match8_evaluation',spatial=out/'evaluation')
        results={arm:json.loads((folder/'results.json').read_text()) for arm,folder in folders.items()}
        manifests={arm:json.loads((folder/'manifest.json').read_text()) for arm,folder in folders.items()}
        records={arm:{r['left']:r for r in map(json.loads,(folder/'per_image.jsonl').read_text().splitlines())}
                 for arm,folder in folders.items()}
        for arm in folders:
            if manifests[arm]!=manifests['baseline']:raise ValueError('Different evaluation images')
            for path,base in records['baseline'].items():
                shape=lambda row:[(m['iteration'],m['scope'],m['pixels']) for m in row['metrics']]
                if shape(base)!=shape(records[arm][path]):raise ValueError('Different per-image masks')
        comparison=[]
        for i,base in enumerate(results['baseline']['metrics']):
            metric={arm:result['metrics'][i] for arm,result in results.items()}
            for arm,row in metric.items():
                if (row['iteration'],row['scope'],row['pixels'])!=(base['iteration'],base['scope'],base['pixels']):
                    raise ValueError('Aggregate masks differ')
            pick=lambda row:next(m for m in row['metrics'] if m['iteration']==base['iteration'] and m['scope']==base['scope'])
            wins={}
            for control in ('baseline','match8'):
                eligible=[path for path,record in records[control].items() if pick(record)['pixels']]
                wins[control]=dict(contributing_images=len(eligible),improved_images=sum(
                    pick(records['spatial'][path])['epe']<pick(records[control][path])['epe'] for path in eligible))
            comparison.append(dict(iteration=base['iteration'],scope=base['scope'],pixels=base['pixels'],metrics=metric,wins=wins))
        (out/'comparison.json').write_text(json.dumps(comparison,indent=2))
        status.write_text('complete\n')
    except Exception as error:
        status.write_text(f'failed: {error}\n')
        raise


if __name__=='__main__':main()
