"""Convert base-bin weights into a strictly loadable match8 warm start."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from core.bin.defom_cor_ga_bins_match8 import DEFOMStereo


def model_config(mixed_precision=False):
    return SimpleNamespace(dinov2_encoder='vitl', hidden_dims=[128], context_norm='batch',
        n_gru_layers=3, n_downsample=2, mixed_precision=mixed_precision,
        num_init_bins=48, init_max_offset=48., init_corr_scale=10.,
        corr_levels=3, corr_radius=4, train_max_disp=768.,
        init_right_compression='avg', refine_right_compression='avg',
        disable_activation_checkpoint=True, enable_top8_guidance=False, debug_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=1234565)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    checkpoint = torch.load(args.source, map_location='cpu', mmap=True, weights_only=False)
    model = DEFOMStereo(model_config())
    report = model.initialize_from_bin(checkpoint.get('model', checkpoint))
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(model=model.state_dict(), architecture='cor_ga_bins_match8',
                   model_config=vars(model.args), source_checkpoint=str(Path(args.source).resolve()),
                   conversion=report, seed=args.seed)
    # Deliberately omit optimizer/scheduler/step from the old architecture.
    temporary = output.with_name(output.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(output)
    output.with_suffix('.json').write_text(json.dumps({k:v for k,v in payload.items() if k != 'model'}, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
