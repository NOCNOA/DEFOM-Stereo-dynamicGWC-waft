"""Evaluate trained bin/Top8 models on the fixed Unreal pilot strips."""
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
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--reference-checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--guided', action='store_true')
    p.add_argument('--top8-guidance-mode', choices=['fused', 'samples'], default='fused')
    p.add_argument('--crop-mode', choices=['center', 'large'], default='center')
    args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0', '2', '3'):
        p.error('Select GPU 0, 2 or 3 explicitly')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    config = SimpleNamespace(
        dinov2_encoder='vitl', hidden_dims=[128], context_norm='batch',
        n_gru_layers=3, n_downsample=2, mixed_precision=False,
        num_init_bins=48, init_max_offset=48., init_corr_scale=10.,
        corr_levels=3, corr_radius=4, train_max_disp=768.,
        init_right_compression='avg', refine_right_compression='avg',
        disable_activation_checkpoint=True, enable_top8_guidance=args.guided,
        top8_guidance_mode=args.top8_guidance_mode)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    model = DEFOMStereo(config)
    model.load_state_dict(checkpoint.get('model', checkpoint), strict=True)
    del checkpoint
    model.cuda().eval()
    reference_config = SimpleNamespace(**vars(config))
    reference_config.enable_top8_guidance = False
    reference = DEFOMStereo(reference_config)
    source = torch.load(args.reference_checkpoint, map_location='cpu', mmap=True, weights_only=False)
    reference.load_state_dict(source.get('model', source), strict=True)
    del source
    reference.cuda().eval()
    dataset = UnrealStereo4K('/home/share/zhaobofan/UnrealStereo4K', scene='all')
    lookup = {pair[0]: i for i, pair in enumerate(dataset.image_list)}
    paths = json.loads(Path(args.manifest).read_text())['sample_paths']
    totals = {}
    with (output/'per_image.jsonl').open('w') as stream:
        for path in paths:
            data = dataset[lookup[path]]
            y0 = (data['disp'].shape[-2] - 512)//2
            if args.crop_mode == 'large':
                # Fix a GT-selected strip independently of either model.
                low_gt = data['disp'][0, ::4, ::4]
                x = torch.arange(low_gt.shape[1])[None] * 4
                large = torch.isfinite(low_gt) & (low_gt >= 384) & (low_gt < 752) & (x-low_gt >= 0)
                counts = large.sum(1)
                prefix = torch.cat((torch.zeros(1, dtype=counts.dtype), counts.cumsum(0)))
                window_counts = prefix[128:] - prefix[:-128]
                y0 = int(window_counts.argmax()) * 4
            for key in ('img1', 'img2', 'disp', 'valid'):
                data[key] = data[key][..., y0:y0+512, :]
            gt = data['disp'][0].cuda().float()
            valid = data['valid'][0].cuda().bool() & torch.isfinite(gt) & (gt>0) & (gt<768)
            right_path = Path(data['disp_file']).parent.parent/'Disp1'/Path(data['disp_file']).name
            right_gt = torch.from_numpy(np.load(right_path)[y0:y0+512].copy()).cuda()
            h,w = gt.shape
            yy,xx = torch.meshgrid(torch.arange(h,device=gt.device), torch.arange(w,device=gt.device), indexing='ij')
            xr = xx-gt
            grid = torch.stack((2*xr/(w-1)-1, 2*yy/(h-1)-1), -1)[None].float()
            rd = F.grid_sample(right_gt[None,None],grid,align_corners=True)[0,0]
            consistent = valid & (xr>=0) & (xr<=w-1) & torch.isfinite(rd) & (rd>0) & ((rd-gt).abs()<=1)
            padder = InputPadder(data['img1'][None].shape,divis_by=32)
            left,right = padder.pad(data['img1'][None].cuda(),data['img2'][None].cuda())
            predictions = model(left,right,iters=32)
            initial = padder.unpad(reference(left,right,iters=0)[0])[0,0]
            initial_error = (initial-gt).abs()
            masks = dict(valid_gt=valid,lr_consistent=consistent,
                         large_lr_consistent=consistent & (gt>=384) & (gt<752),
                         initial_good4_lr_consistent=consistent & (initial_error<=4),
                         initial_bad16_lr_consistent=consistent & (initial_error>16))
            image_rows=[]
            for iteration in (0,1,4,8,16,32):
                error=(padder.unpad(predictions[iteration])[0,0]-gt).abs()
                if not torch.isfinite(error[valid]).all():
                    raise FloatingPointError(path)
                for scope,mask in masks.items():
                    v=error[mask];n=v.numel()
                    sums=np.array([n,v.double().sum().item(),(v>3).sum().item(),(v<=8).sum().item(),(v>4).sum().item()],dtype=np.float64)
                    key=(iteration,scope)
                    totals[key]=totals.get(key,np.zeros(5))+sums
                    image_rows.append(dict(iteration=iteration,scope=scope,pixels=n,
                                           epe=v.double().mean().item() if n else None))
            stream.write(json.dumps(dict(image=path,crop_y=[y0,y0+512],metrics=image_rows))+'\n');stream.flush()
            del predictions,left,right
    rows=[]
    for (iteration,scope),(n,total,bad3,good8,bad4) in totals.items():
        rows.append(dict(iteration=iteration,scope=scope,pixels=int(n),
                         epe=total/n if n else None,bad3_pct=100*bad3/n if n else None,
                         recall8_pct=100*good8/n if n else None,bad4_pct=100*bad4/n if n else None))
    (output/'results.json').write_text(json.dumps(dict(checkpoint=args.checkpoint,guided=args.guided,
        top8_guidance_mode=args.top8_guidance_mode,
        reference_checkpoint=args.reference_checkpoint,initial_scopes='fixed source checkpoint d0, identical across arms',
        samples=paths,crop=f'{args.crop_mode} 512 x full width; no resize',delta_clamp=False,
        activation_recomputation=False,mixed_precision=False,metrics=rows),indent=2))
    print(f'Finished {len(paths)} pairs: {output}',flush=True)


if __name__=='__main__':
    main()
