"""Extend the completed 2k width-match8/control pilot by another 8k steps."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import signal

ROOT = Path(__file__).resolve().parents[2]
WORK = Path('/var/tmp/codex-home-u2025140689/Documents/Codex/2026-10-04/gru-cost-volume-costvolume')
OLD = WORK / 'outputs/bin_match8_20261006/pilot_2k'
OUT = WORK / 'outputs/bin_match8_width_cumulative10k_20261007'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', default='', help='New output suffix for retrying an interrupted launch')
    args = parser.parse_args()
    if args.attempt and not args.attempt.replace('_', '').isalnum():
        raise ValueError('attempt must contain only letters, digits, and underscores')
    suffix = '_' + args.attempt if args.attempt else ''
    out = OUT.with_name(OUT.name + suffix)
    # The rest of the driver uses a single output directory for this attempt.
    global_run_output = out
    previous = json.loads((OLD / 'config.json').read_text())
    # No matching-model change between the first phase and this extension.
    unchanged = ['core/bin/defom_cor_ga_bins.py', 'core/bin/defom_cor_ga_bins_match8.py',
                 'core/submodules.py', 'core/update_pru.py', 'utils/utils.py']
    for filename in unchanged:
        if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != previous['source_sha256'][filename]:
            raise RuntimeError(f'Control implementation changed: {filename}')
    setups = {}
    locks = []
    for arm, gpu in [('baseline', 2), ('match8', 3)]:
        lock = open(f'/tmp/defom-match8-{os.getuid()}-gpu-{gpu}.lock', 'w')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
        occupants = subprocess.check_output([
            'nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid,used_memory',
            '--format=csv,noheader,nounits'], text=True).strip()
        if occupants:
            raise RuntimeError(f'GPU {gpu} is occupied: {occupants}')
        name = f'bins_match8_{arm}_20261007_add8k_b2{suffix}'
        checkpoint = ROOT / f'checkpoints/bins_match8_{arm}_20261006_2k_b2/checkpoint_latest.pth'
        if not checkpoint.is_file() or (ROOT / 'checkpoints' / name).exists():
            raise RuntimeError(f'Missing input or existing output for {arm}')
        command = list(previous['commands'][arm])
        command[0] = sys.executable
        changes = {'--name': name, '--resume_ckpt': str(checkpoint), '--num_steps': '8000',
                   '--save_ckpt_freq': '2000', '--val_freq': '8001'}
        for key, value in changes.items():
            command[command.index(key) + 1] = value
        command += ['--resume_batch_skip', '2000']
        setups[arm] = dict(gpu=gpu, name=name, source=str(checkpoint), command=command)
    global_run_output.mkdir(parents=True, exist_ok=False)
    status = global_run_output / 'status.txt'
    status.write_text('preparing\n')
    files = unchanged + ['core/warp.py', 'train_stereo.py', 'evaluate_stereo.py',
                         'scripts/bin/evaluate_bin_match8_subset.py',
                         'scripts/bin/continue_bin_match8_pair_10k.py']
    hashes = {}
    for filename in files:
        content = (ROOT / filename).read_bytes()
        target = global_run_output / 'source' / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[filename] = hashlib.sha256(content).hexdigest()
    (global_run_output / 'config.json').write_text(json.dumps(dict(
        arms=setups, prior_steps=2000, additional_steps=8000, cumulative_steps=10000,
        optimizer='Both arms reset AdamW and OneCycleLR; weights retained from 2k checkpoints',
        learning_rate=2e-5, batch=2, accumulation=1, train_iters=18, seed=1234565,
        replay_and_skip_batches=2000, token_channels=8, static_scale_gate=True,
        source_geometry='Right-width compression only; quarter-grid queries',
        recomputation=False, source_sha256=hashes,
        evaluation='Same fixed100 full-resolution SceneFlow TEST images, FP32, updates0/8/16/32',
        note='This is a 2k+8k staged fine-tuning experiment, not a fresh10k or scratch-training run.'
    ), indent=2))
    processes, logs = {}, {}
    def interrupted(signum, frame):
        raise RuntimeError(f'Driver interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    env['LD_LIBRARY_PATH'] = '/home/u2025140689/anaconda3/envs/defomstereo/lib:' + env.get('LD_LIBRARY_PATH', '')
    try:
        import torch
        for arm, setup in setups.items():
            checkpoint = torch.load(setup['source'], map_location='cpu', mmap=True, weights_only=False)
            if checkpoint.get('step') != 2000:
                raise RuntimeError(f'{arm} source is not the completed2k checkpoint')
            del checkpoint
        status.write_text('training_both_arms_additional8000_steps\n')
        for arm, setup in setups.items():
            arm_env = dict(env, CUDA_VISIBLE_DEVICES=str(setup['gpu']))
            logs[arm] = (global_run_output / f'{arm}_train.log').open('w')
            processes[arm] = subprocess.Popen(setup['command'], cwd=ROOT, env=arm_env,
                                             stdout=logs[arm], stderr=subprocess.STDOUT)
        (global_run_output / 'processes.json').write_text(json.dumps(dict(
            driver_pid=os.getpid(), children={arm: process.pid for arm, process in processes.items()}), indent=2))
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(process.wait): arm for arm, process in processes.items()}
            for future in as_completed(futures):
                arm = futures[future]
                code = future.result()
                logs[arm].close()
                if code:
                    for process in processes.values():
                        if process.poll() is None:
                            process.terminate()
                    raise RuntimeError(f'{arm} training failed: exit{code}')
                if 'FINISHED TRAINING' not in (global_run_output / f'{arm}_train.log').read_text(errors='replace'):
                    raise RuntimeError(f'{arm} training completion marker missing')
        status.write_text('evaluating_fixed100_test_images_fp32\n')
        for arm, setup in setups.items():
            command = [sys.executable, '-u', 'scripts/bin/evaluate_bin_match8_subset.py',
                       '--arch', arm, '--checkpoint', str(ROOT / 'checkpoints' / setup['name'] / 'checkpoint_latest.pth'),
                       '--output', str(global_run_output / f'{arm}_evaluation'), '--samples', '100']
            (global_run_output / f'{arm}_evaluation_command.json').write_text(json.dumps(command, indent=2))
            with (global_run_output / f'{arm}_evaluation.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=str(setup['gpu'])),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        folders = {arm: global_run_output / f'{arm}_evaluation' for arm in setups}
        manifests = {arm: json.loads((folder / 'manifest.json').read_text()) for arm, folder in folders.items()}
        old_manifest = json.loads((OLD / 'baseline_evaluation/manifest.json').read_text())
        if any(manifest != old_manifest for manifest in manifests.values()):
            raise RuntimeError('Evaluation image manifests differ from2k pilot')
        records = {arm: {r['left']: r for r in map(json.loads, (folder / 'per_image.jsonl').read_text().splitlines())}
                   for arm, folder in folders.items()}
        for path, base in records['baseline'].items():
            keys = lambda r: [(m['iteration'], m['scope'], m['pixels']) for m in r['metrics']]
            if keys(base) != keys(records['match8'][path]):
                raise RuntimeError(f'Per-image masks differ: {path}')
        results = {arm: json.loads((folder / 'results.json').read_text()) for arm, folder in folders.items()}
        old_results = {arm: json.loads((OLD / f'{arm}_evaluation/results.json').read_text()) for arm in setups}
        comparison = []
        for i, base in enumerate(results['baseline']['metrics']):
            new = results['match8']['metrics'][i]
            if (base['iteration'], base['scope'], base['pixels']) != (new['iteration'], new['scope'], new['pixels']):
                raise RuntimeError('Aggregate masks differ')
            pick = lambda row: next(m for m in row['metrics'] if m['iteration'] == base['iteration'] and m['scope'] == base['scope'])
            eligible = [path for path, row in records['baseline'].items() if pick(row)['pixels']]
            comparison.append(dict(iteration=base['iteration'], scope=base['scope'],
                baseline=base, match8=new, prior2k={arm: r['metrics'][i] for arm, r in old_results.items()},
                contributing_images=len(eligible), improved_images=sum(
                    pick(records['match8'][path])['epe'] < pick(records['baseline'][path])['epe'] for path in eligible)))
        (global_run_output / 'comparison.json').write_text(json.dumps(comparison, indent=2))
        status.write_text('complete\n')
    except Exception as error:
        status.write_text(f'failed: {error}\n')
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for log in logs.values():
            log.close()


if __name__ == '__main__':
    main()
