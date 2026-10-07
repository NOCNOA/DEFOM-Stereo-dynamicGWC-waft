"""Fixed SceneFlow TEST subset and iteration curves for bin/match8 ablation."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts/bin'))
import numpy as np
import torch
from core import stereo_datasets
from core.bin.defom_cor_ga_bins import DEFOMStereo as Baseline
from core.bin.defom_cor_ga_bins_match8 import DEFOMStereo as Match8
from core.bin.defom_cor_ga_bins_match8_spatial import DEFOMStereo as SpatialMatch8
from core.utils.utils import InputPadder
from prepare_bin_match8_checkpoint import model_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch', choices=('baseline','match8','spatial'), required=True)
    parser.add_argument('--match_token_grid', choices=('native','dense'), default='native')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=100)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    config = model_config(mixed_precision=False)
    config.match_token_grid = args.match_token_grid
    model_class = {'baseline':Baseline, 'match8':Match8, 'spatial':SpatialMatch8}[args.arch]
    model = model_class(config)
    weights = torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict(weights.get('model',weights),strict=True)
    del weights
    model.cuda().eval()
    dataset = stereo_datasets.SceneFlowDatasets(dstype='frames_finalpass',things_test=True)
    indices = np.linspace(0,len(dataset)-1,min(args.samples,len(dataset)),dtype=int).tolist()
    aggregates = {(it,scope):dict(image_epes=[],sum_abs=0.,bad3=0,pixels=0)
                  for it in (0,8,16,32) for scope in ('all_valid','large384')}
    manifest = []
    with torch.inference_mode(),(out/'per_image.jsonl').open('w') as stream:
        for index in indices:
            sample = dataset[index]
            path = sample['imageL_file']
            manifest.append(dict(index=index,left=path))
            left,right = sample['img1'][None].cuda(),sample['img2'][None].cuda()
            padder = InputPadder(left.shape,divis_by=32)
            left,right = padder.pad(left,right)
            torch.cuda.synchronize()
            start = time.perf_counter()
            predictions = model(left,right,iters=32,test_mode=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            assert len(predictions) == 33
            gt = sample['disp'].float()
            valid = (sample['valid'] >= .5) & torch.isfinite(gt) & (gt >= 0) & (gt < 768)
            records = []
            for iteration in (0,8,16,32):
                prediction = padder.unpad(predictions[iteration]).cpu().squeeze(0)
                assert prediction.shape == gt.shape and torch.isfinite(prediction).all()
                error = (prediction-gt).abs()
                for scope in ('all_valid','large384'):
                    mask = valid if scope == 'all_valid' else valid & (gt >= 384)
                    pixels = int(mask.sum())
                    absolute = float(error[mask].double().sum())
                    bad3 = int((error[mask] > 3).sum())
                    metric = dict(iteration=iteration,scope=scope,pixels=pixels,
                                  epe=absolute/pixels if pixels else None,bad3=bad3)
                    records.append(metric)
                    total = aggregates[iteration,scope]
                    total['pixels'] += pixels
                    total['sum_abs'] += absolute
                    total['bad3'] += bad3
                    if pixels:
                        total['image_epes'].append(absolute/pixels)
            stream.write(json.dumps(dict(index=index,left=path,seconds=elapsed,metrics=records))+'\n')
            stream.flush()
            del predictions,left,right,error
    metrics = []
    for (iteration,scope), total in aggregates.items():
        pixels = total['pixels']
        metrics.append(dict(iteration=iteration,scope=scope,pixels=pixels,
                            image_mean_epe=float(np.mean(total['image_epes'])) if total['image_epes'] else None,
                            pixel_epe=total['sum_abs']/pixels if pixels else None,
                            bad3_percent=100*total['bad3']/pixels if pixels else None))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    report = dict(arch=args.arch,match_token_grid=args.match_token_grid if args.arch == 'spatial' else 'dense',
                  checkpoint=args.checkpoint,samples=len(indices),metrics=metrics,
                  precision='FP32',input='full image padded to 32; no resize/crop',
                  selection='equally spaced TEST dataset indices, without error filtering',
                  note='100-image pilot; not the full SceneFlow evaluation. Large disparity scope is GT [384,768).')
    (out/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
