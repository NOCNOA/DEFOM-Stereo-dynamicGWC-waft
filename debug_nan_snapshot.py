import argparse

import torch

from core.defom_cor_ga import DEFOMStereo


def tensor_stats(name, tensor):
    finite = torch.isfinite(tensor)
    finite_values = tensor[finite]
    print(f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} nan={torch.isnan(tensor).sum().item()} inf={torch.isinf(tensor).sum().item()} finite_min={finite_values.min().item() if finite_values.numel() else 'none'} finite_max={finite_values.max().item() if finite_values.numel() else 'none'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="checkpoints/defomstereo_vitl_twostage_768/nan_batch_snapshot_step_31302.pth")
    parser.add_argument("--model", default="checkpoints/defomstereo_vitl_twostage_768/nan_model_snapshot_step_31302.pth")
    parser.add_argument("--trace-shared", action="store_true")
    args = parser.parse_args()

    model_args = argparse.Namespace(dinov2_encoder="vitl", corr_levels=3, corr_radius=4, hypothesis_scales=[24.0, 48.0, 96.0], hidden_dims=[128], context_norm="batch", n_gru_layers=3, n_downsample=2, mixed_precision=True)
    checkpoint = torch.load(args.model, map_location="cpu")
    state_dict = checkpoint["model"]
    bad_parameters = [(name, int((~torch.isfinite(value)).sum())) for name, value in state_dict.items() if torch.is_tensor(value) and not torch.isfinite(value).all()]
    print(f"snapshot_step={checkpoint.get('step')} nonfinite_state_tensors={bad_parameters}")

    device = torch.device("cuda")
    model = DEFOMStereo(model_args).to(device)
    model.load_state_dict(state_dict, strict=True)
    if args.trace_shared:
        def trace_tensor(name):
            def hook(module, inputs, output):
                tensor_stats(name + ".input", inputs[0])
                tensor_stats(name + ".output", output)
            return hook

        def trace_defom(module, inputs, output):
            for output_index, tensor in enumerate(output):
                tensor_stats(f"defomencoder.output{output_index}", tensor)

        model.defomencoder.register_forward_hook(trace_defom)
        model.fnet.convd.conv.register_forward_hook(trace_tensor("fnet.convd.conv"))
        model.cnet.convd.conv.register_forward_hook(trace_tensor("cnet.convd.conv"))
        model.cnet.convd.norm1.register_forward_hook(trace_tensor("cnet.convd.norm1"))
        model.cnet.layer3.register_forward_hook(trace_tensor("cnet.layer3"))
        model.cnet.layer4[0].conv1.register_forward_hook(trace_tensor("cnet.layer4.0.conv1"))
    del checkpoint, state_dict
    batch = torch.load(args.batch, map_location="cpu")
    print(f"batch_keys={list(batch.keys())}")
    image1 = batch["image1"].to(device)
    image2 = batch["image2"].to(device)
    tensor_stats("image1", image1)
    tensor_stats("image2", image2)

    model.train()
    with torch.no_grad():
        predictions, aux = model(image1, image2, iters=18, scale_iters=6, return_aux=True)

    for branch_index, branch in enumerate(aux["coarse_predictions"]):
        for iteration, prediction in enumerate(branch):
            tensor_stats(f"coarse.branch{branch_index}.prediction{iteration}", prediction)
    tensor_stats("hypothesis_weights", aux["w_h"])
    tensor_stats("scale_weights", aux["w_s"])
    for iteration, prediction in enumerate(predictions):
        tensor_stats(f"refine.prediction{iteration}", prediction)


if __name__ == "__main__":
    main()
