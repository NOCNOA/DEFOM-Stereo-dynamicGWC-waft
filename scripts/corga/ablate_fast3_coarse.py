"""Measure fast3 coarse-to-refine contributions on fixed FlyingThings images."""

import argparse
import gc
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.corga.defom_cor_ga_fast3 import DEFOMStereo
from core.stereo_datasets import SceneFlowDatasets
from core.utils.utils import InputPadder


VARIANTS = {
    "baseline": None,
    "uniform_scale_weights": {"uniform_scale_weights": True},
    "context_hidden": {"use_context_hidden": True},
    "initial_disparities": {"use_initial_disparities": True},
    "middle_branch_only": {"active_branches": [1]},
}


def model_args():
    return SimpleNamespace(
        dinov2_encoder="vitl", mixed_precision=True, amp_dtype="bf16",
        n_downsample=2, corr_levels=3, corr_radius=4,
        hypothesis_scales=[24.0, 48.0, 96.0], hidden_dims=[128],
        context_norm="batch", n_gru_layers=3, max_delta_disp=8.0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "checkpoints/defomstereo_vitl_twostage_768/defomstereo_vitl_twostage_768_200000.pth",
    )
    parser.add_argument("--num-images", type=int, default=64)
    args = parser.parse_args()
    if args.num_images <= 0:
        parser.error("--num-images must be positive")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    model = DEFOMStereo(model_args())
    try:
        checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True, mmap=True)
    except pickle.UnpicklingError:
        checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    del checkpoint
    gc.collect()
    model.cuda().eval()

    dataset = SceneFlowDatasets(dstype="frames_finalpass", things_test=True)
    indices = np.linspace(0, len(dataset) - 1, min(args.num_images, len(dataset)), dtype=int)
    scores = {name: {"epes": [], "bad": 0, "count": 0} for name in VARIANTS}
    with torch.inference_mode():
        for number, index in enumerate(indices, 1):
            sample = dataset[int(index)]
            gt = sample["disp"]
            valid = (sample["valid"] >= 0.5) & (gt.abs() < 768)
            left = sample["img1"][None].cuda()
            right = sample["img2"][None].cuda()
            padder = InputPadder(left.shape, divis_by=32)
            left, right = padder.pad(left, right)
            for name, ablation in VARIANTS.items():
                prediction = model(
                    left, right, iters=32, scale_iters=6,
                    test_mode=True, coarse_ablation=ablation,
                )
                prediction = padder.unpad(prediction).cpu().squeeze(0)
                error = (prediction - gt).abs()[valid]
                scores[name]["epes"].append(error.mean().item())
                scores[name]["bad"] += int((error > 3).sum().item())
                scores[name]["count"] += int(valid.sum().item())
            if number % 16 == 0 or number == len(indices):
                print(f"Processed {number}/{len(indices)} images", flush=True)

    for name, score in scores.items():
        print(
            f"{name}: image-mean EPE={np.mean(score['epes']):.6f}, "
            f"pixel Bad3={100 * score['bad'] / score['count']:.6f}%"
        )


if __name__ == "__main__":
    main()
