"""Bidirectional checkpoint-module swaps for coarse-stage localization."""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "corga"))
from evaluate_encoder_freeze_ab_fixed import (  # noqa: E402
    CHECKPOINTS, DEFOMStereo, SceneFlowDatasets, evaluate_one, model_args, summarize,
)


GROUPS = {
    "control": lambda key: False,
    "coarse_gru": lambda key: key.startswith("coarse_update_blocks.") and ".gru_layers." in key,
    "coarse_motion": lambda key: key.startswith("coarse_update_blocks.") and ".motion_encoder." in key,
    "coarse_context_gates": lambda key: key.startswith("coarse_update_blocks.") and ".context_encoders." in key,
    "coarse_disp_head": lambda key: key.startswith("coarse_update_blocks.") and ".disp_head." in key,
    "coarse_mask": lambda key: key.startswith("coarse_update_blocks.") and ".mask." in key,
    "coarse_gru_context": lambda key: key.startswith("coarse_update_blocks.") and (
        ".gru_layers." in key or ".context_encoders." in key),
    "coarse_gru_motion": lambda key: key.startswith("coarse_update_blocks.") and (
        ".gru_layers." in key or ".motion_encoder." in key),
    "coarse_blocks": lambda key: key.startswith("coarse_update_blocks."),
    "context": lambda key: key.startswith("cnet."),
    "coarse_blocks_context": lambda key: key.startswith(("coarse_update_blocks.", "cnet.")),
    "fusion": lambda key: key.startswith("hypothesis_weight_head."),
    "refine_projection": lambda key: key.startswith("refine_state_projections."),
    "refine_update": lambda key: key.startswith("refine_update_block."),
    "refine_motion": lambda key: key.startswith("refine_update_block.motion_encoder."),
    "refine_gru": lambda key: key.startswith("refine_update_block.gru_layers."),
    "refine_context_gates": lambda key: key.startswith("refine_update_block.context_encoders."),
    "refine_disp_head": lambda key: key.startswith("refine_update_block.disp_head."),
    "refine_mask": lambda key: key.startswith("refine_update_block.mask."),
}


def checkpoint_state(path):
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    except pickle.UnpicklingError:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    return checkpoint.get("model", checkpoint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--donor", required=True)
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--donor-path", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-images", type=int, default=128)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    args = parser.parse_args()
    if args.base == args.donor or args.num_images < 1:
        parser.error("Base and donor must differ, and --num-images must be positive")
    base_path = args.base_path or CHECKPOINTS.get(args.base)
    donor_path = args.donor_path or CHECKPOINTS.get(args.donor)
    if base_path is None or donor_path is None:
        parser.error("Unknown checkpoint label; supply --base-path and/or --donor-path")
    if not base_path.is_file() or not donor_path.is_file():
        parser.error(f"Missing checkpoint: {base_path} or {donor_path}")
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = SceneFlowDatasets(dstype="frames_finalpass", things_test=True)
    if args.num_images > len(dataset):
        parser.error(f"Requested {args.num_images} images; dataset has {len(dataset)}")
    indices = np.linspace(0, len(dataset) - 1, args.num_images, dtype=int).tolist()
    (args.output / "manifest.json").write_text(json.dumps([
        {"index": index, "image": dataset.image_list[index][0]}
        for index in indices
    ], indent=2))
    base = checkpoint_state(base_path)
    donor = checkpoint_state(donor_path)
    if base.keys() != donor.keys():
        raise ValueError("Checkpoint model keys differ")
    torch.backends.cudnn.benchmark = True
    model = DEFOMStereo(model_args()).cuda().eval()
    model.freeze_bn()
    with (args.output / "comparison.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["base", "donor", "group", "replaced_keys", "final_epe",
                         "d0_epe", "coarse0_epe", "coarse1_epe", "coarse2_epe"])
        for group in args.groups:
            selector = GROUPS[group]
            keys = [key for key in base if selector(key)]
            if group != "control" and not keys:
                raise ValueError(f"No model keys matched group {group}")
            hybrid = dict(base)
            hybrid.update({key: donor[key] for key in keys})
            model.load_state_dict(hybrid, strict=True)
            model.eval()
            model.freeze_bn()
            print(f"Evaluating {args.base} + {args.donor}:{group}, {len(keys)} replaced keys", flush=True)
            rows = []
            for sequence, index in enumerate(indices, start=1):
                row = {"index": index, "image": dataset.image_list[index][0]}
                row.update(evaluate_one(model, dataset[index], diagnostics=False))
                rows.append(row)
                if sequence % 32 == 0 or sequence == len(indices):
                    print(f"{group}: {sequence}/{len(indices)}; EPE "
                          f"{np.mean([r['refine'][-1] for r in rows]):.4f}", flush=True)
            summary = summarize(rows)
            (args.output / f"{group}_per_image.json").write_text(json.dumps(rows))
            (args.output / f"{group}_summary.json").write_text(json.dumps(summary, indent=2))
            writer.writerow([args.base, args.donor, group, len(keys),
                             summary["refine"]["mean_image_epe"][-1],
                             summary["refine"]["mean_image_epe"][0],
                             *[summary[f"coarse{i}"]["mean_image_epe"][-1] for i in range(3)]])
            file.flush()
    print("FINISHED", flush=True)


if __name__ == "__main__":
    main()
