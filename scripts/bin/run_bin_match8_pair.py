"""Run the first static-scale match8 pilot with a matched base-bin control."""
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
WORK = Path('/var/tmp/codex-home-u2025140689/Documents/Codex/2026-10-04/gru-cost-volume-costvolume')
OUT = WORK/'outputs/bin_match8_20261006/pilot_2k'
SOURCE = ROOT/'checkpoints/defomstereo_vitl_corga_bins_48_avg_ft20k_control/checkpoint_latest.pth'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    status = OUT/'status.txt'
    status.write_text('preparing\n')
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/home/u2025140689/anaconda3/envs/defomstereo/lib:' + env.get('LD_LIBRARY_PATH','')
    env['CUDA_VISIBLE_DEVICES'] = '2'
    env['OMP_NUM_THREADS'] = '4'
    env['MKL_NUM_THREADS'] = '4'
    converted = OUT/'match8_warm_start.pth'
    arms = dict(baseline=dict(gpu='2',architecture='cor_ga_bins',checkpoint=SOURCE),
                match8=dict(gpu='3',architecture='cor_ga_bins_match8',checkpoint=converted))
    processes = {}
    logs = {}
    try:
        with (OUT/'prepare.log').open('w') as log:
            subprocess.run([sys.executable,'scripts/bin/prepare_bin_match8_checkpoint.py',
                            '--source',str(SOURCE),'--output',str(converted)],
                           cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        files = ['core/bin/defom_cor_ga_bins.py','core/bin/defom_cor_ga_bins_match8.py',
                 'core/submodules.py','core/warp.py','core/update_pru.py',
                 'train_stereo.py','evaluate_stereo.py','utils/utils.py',
                 'scripts/bin/prepare_bin_match8_checkpoint.py',
                 'scripts/bin/evaluate_bin_match8_subset.py','scripts/bin/run_bin_match8_pair.py']
        hashes = {}
        for filename in files:
            target = OUT/'source'/filename
            target.parent.mkdir(parents=True,exist_ok=True)
            content = (ROOT/filename).read_bytes()
            target.write_bytes(content)
            hashes[filename] = hashlib.sha256(content).hexdigest()
        commands = {}
        for arm,setup in arms.items():
            name = f'bins_match8_{arm}_20261006_2k_b2'
            if (ROOT/'checkpoints'/name).exists():
                raise FileExistsError(name)
            setup['name'] = name
            command = [sys.executable,'-u','train_stereo.py','--model_arch',setup['architecture'],
                '--name',name,'--resume_ckpt',str(setup['checkpoint']),'--strict_resume','--no_resume_optimizer',
                '--launcher','none','--gpu_ids','0','--batch_size','2','--gradient_accumulation_steps','1',
                '--num_workers','2','--train_datasets','sceneflow','--train_folds','1',
                '--num_steps','2000','--save_ckpt_freq','2000','--save_latest_ckpt_freq','1000',
                '--val_freq','2001','--valid_iters','32','--train_iters','18',
                '--n_downsample','2','--num_init_bins','48','--init_max_offset','48','--init_corr_scale','10',
                '--corr_levels','3','--corr_radius','4','--dinov2_encoder','vitl',
                '--init_right_compression','avg','--refine_right_compression','avg',
                '--context_norm','batch','--n_gru_layers','3','--hidden_dims','128',
                '--image_size','256','768','--train_max_disp','768','--eval_max_disp','768',
                '--loss_gamma','0.9','--init_target_temperature','1','--gate_loss_weight','0.1',
                '--gate_target_temperature','4','--smooth_l1_beta','1','--d0_loss_weight','1',
                '--wdecay','0.00001','--seed','1234565','--mixed_precision','--amp_dtype','fp16',
                '--amp_init_scale','64','--lr','0.00002','--disable_activation_checkpoint']
            commands[arm] = command
        config = dict(source=str(SOURCE),commands=commands,gpus={'baseline':2,'match8':3},
                      batch=2,accumulation=1,steps=2000,seed=1234565,learning_rate=2e-5,
                      evaluation_samples=100,source_sha256=hashes,
                      changed='36->8 per-sample encoding, preserve three scales, 248->64->32 fusion',
                      static_scale_gate=True,activation_recomputation=False,delta_clamp=False,
                      note='New encoder/fusion entrance starts fresh. 2k is a learning pilot, not a convergence verdict. Synthetic profile used 21.92 GiB peak allocated on RTX 3090 batch2.')
        (OUT/'config.json').write_text(json.dumps(config,indent=2))
        status.write_text('training_both_arms\n')
        for arm,command in commands.items():
            arm_env = dict(env,CUDA_VISIBLE_DEVICES=arms[arm]['gpu'])
            logs[arm] = (OUT/f'{arm}_train.log').open('w')
            processes[arm] = subprocess.Popen(command,cwd=ROOT,env=arm_env,stdout=logs[arm],stderr=subprocess.STDOUT)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(process.wait):arm for arm,process in processes.items()}
            for future in as_completed(futures):
                arm = futures[future]
                code = future.result()
                logs[arm].close()
                if code:
                    for process in processes.values():
                        if process.poll() is None:
                            process.terminate()
                    raise RuntimeError(f'{arm} training failed with exit code {code}')
        status.write_text('evaluating_fixed_100_test_images_fp32\n')
        for arm,setup in arms.items():
            eval_env = dict(env,CUDA_VISIBLE_DEVICES=setup['gpu'])
            command = [sys.executable,'-u','scripts/bin/evaluate_bin_match8_subset.py',
                       '--arch',arm,'--checkpoint',str(ROOT/'checkpoints'/setup['name']/'checkpoint_latest.pth'),
                       '--output',str(OUT/f'{arm}_evaluation'),'--samples','100']
            with (OUT/f'{arm}_evaluation.log').open('w') as log:
                subprocess.run(command,cwd=ROOT,env=eval_env,stdout=log,stderr=subprocess.STDOUT,check=True)
        folders = {arm:OUT/f'{arm}_evaluation' for arm in arms}
        manifests = {arm:json.loads((folder/'manifest.json').read_text()) for arm,folder in folders.items()}
        if manifests['baseline'] != manifests['match8']:
            raise RuntimeError('Evaluation manifests differ')
        results = {arm:json.loads((folder/'results.json').read_text()) for arm,folder in folders.items()}
        comparison = []
        for base,new in zip(results['baseline']['metrics'],results['match8']['metrics']):
            if (base['iteration'],base['scope'],base['pixels']) != (new['iteration'],new['scope'],new['pixels']):
                raise RuntimeError('Evaluation masks differ')
            comparison.append(dict(iteration=base['iteration'],scope=base['scope'],pixels=base['pixels'],
                                   baseline=base,match8=new,
                                   image_epe_difference=new['image_mean_epe']-base['image_mean_epe'] if base['image_mean_epe'] is not None else None))
        (OUT/'comparison.json').write_text(json.dumps(comparison,indent=2))
        status.write_text('complete\n')
    except Exception as error:
        status.write_text(f'failed: {error}\n')
        # If one arm fails during launch, stop only the children of this pilot.
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for log in logs.values():
            log.close()


if __name__ == '__main__':
    main()
