"""Check an existing bin checkpoint with the zero-initialized Top8 adapter."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.bin.defom_cor_ga_bins import DEFOMStereo


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--top8-guidance-mode', choices=['fused', 'samples'], default='fused')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0', '2', '3'):
        parser.error('Select GPU 0, 2 or 3 explicitly')
    torch.manual_seed(19)
    torch.set_num_threads(4)
    config = SimpleNamespace(
        dinov2_encoder='vitl', hidden_dims=[128], context_norm='batch',
        n_gru_layers=3, n_downsample=2, mixed_precision=False,
        num_init_bins=48, init_max_offset=48., init_corr_scale=10.,
        max_delta_disp=8., corr_levels=3, corr_radius=4, train_max_disp=768.,
        init_right_compression='avg', refine_right_compression='avg',
        disable_activation_checkpoint=True, enable_top8_guidance=True,
        top8_guidance_mode=args.top8_guidance_mode)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    model = DEFOMStereo(config)
    result = model.load_state_dict(checkpoint.get('model', checkpoint), strict=False)
    allowed = ('top8_guidance.', 'refine_update_block.motion_encoder.global_convg1.')
    if result.unexpected_keys or any(not key.startswith(allowed) for key in result.missing_keys):
        raise RuntimeError(str(result))
    del checkpoint
    model.cuda().eval()
    left = torch.rand(1, 3, 128, 512, device='cuda') * 255
    right = torch.roll(left, shifts=-16, dims=-1)
    adapter = model.top8_guidance
    del model.top8_guidance
    baseline = model(left, right, iters=2, test_mode=True)
    model.top8_guidance = adapter
    guided = model(left, right, iters=2, test_mode=True)
    torch.testing.assert_close(guided, baseline, atol=1e-5, rtol=1e-5)
    if not torch.isfinite(guided).all():
        raise RuntimeError('Nonfinite prediction')
    output = dict(shape=list(guided.shape), max_difference=(guided-baseline).abs().max().item(),
                  checkpoint=args.checkpoint, new_parameters=sum(p.numel() for p in adapter.parameters()),
                  missing_keys=result.missing_keys, activation_recomputation=False,
                  note='Synthetic FP32 two-iteration integration check, not an accuracy evaluation')
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
