"""Compare CorGA and fast3 on identical FlyingThings images and weights."""

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

from core.corga.defom_cor_ga import DEFOMStereo as CorGA
from core.corga.defom_cor_ga_fast import DEFOMStereo as Fast
from core.corga.defom_cor_ga_fast3 import DEFOMStereo as Fast3
from core.stereo_datasets import SceneFlowDatasets
from core.utils.utils import InputPadder


def model_args():
    return SimpleNamespace(
        dinov2_encoder="vitl", mixed_precision=True, amp_dtype="bf16",
        n_downsample=2, corr_levels=3, corr_radius=4,
        hypothesis_scales=[24.0, 48.0, 96.0], hidden_dims=[128],
        context_norm="batch", n_gru_layers=3, max_delta_disp=8.0,
    )


def load_state(path):
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    except pickle.UnpicklingError:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    return checkpoint.get("model", checkpoint)


def evaluate(model, dataset, indices):
    results = {}
    model.eval()
    with torch.inference_mode():
        for number, index in enumerate(indices, 1):
            sample = dataset[int(index)]
            left = sample["img1"][None].cuda()
            right = sample["img2"][None].cuda()
            padder = InputPadder(left.shape, divis_by=32)
            left, right = padder.pad(left, right)
            prediction = model(left, right, iters=32, scale_iters=6, test_mode=True)
            prediction = padder.unpad(prediction).cpu().squeeze(0)
            gt = sample["disp"]
            valid = (sample["valid"] >= 0.5) & (gt.abs() < 768)
            error = (prediction - gt).abs()[valid]
            results[int(index)] = {
                "prediction": prediction,
                "epe": error.mean().item(),
                "bad3_count": int((error > 3).sum().item()),
                "valid_count": int(valid.sum().item()),
            }
            if number % 16 == 0 or number == len(indices):
                print(f"{model.__class__.__name__}: {number}/{len(indices)}", flush=True)
    return results


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
    dataset = SceneFlowDatasets(dstype="frames_finalpass", things_test=True)
    indices = np.linspace(0, len(dataset) - 1, min(args.num_images, len(dataset)), dtype=int)
    state = load_state(args.checkpoint)
    outputs = {}
    for name, cls in (("cor_ga", CorGA), ("cor_ga_fast", Fast), ("cor_ga_fast3", Fast3)):
        model = cls(model_args())
        model.load_state_dict(state, strict=True)
        model.cuda()
        outputs[name] = evaluate(model, dataset, indices)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    for name, rows in outputs.items():
        epe = np.mean([row["epe"] for row in rows.values()])
        bad = sum(row["bad3_count"] for row in rows.values())
        count = sum(row["valid_count"] for row in rows.values())
        print(f"{name}: image-mean EPE={epe:.6f}, pixel Bad3={100 * bad / count:.6f}%")
    for left_name, right_name in (("cor_ga", "cor_ga_fast"), ("cor_ga_fast", "cor_ga_fast3")):
        differences = [
            (outputs[left_name][int(index)]["prediction"]
             - outputs[right_name][int(index)]["prediction"]).abs().mean().item()
            for index in indices
        ]
        print(
            f"Paired mean absolute prediction difference "
            f"{left_name} vs {right_name}: {np.mean(differences):.6f}"
        )


if __name__ == "__main__":
    main()
