"""Grouped-convolution execution for independent CorGA coarse PRUs."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from core.submodules import normalize_disparity


def _parallel_conv2d(convs, inputs):
    """Apply equal-shaped independent Conv2d modules in one grouped conv."""
    if not convs or len(convs) != len(inputs):
        raise ValueError("convs and inputs must be non-empty and equally sized")
    reference = convs[0]
    if reference.groups != 1 or reference.padding_mode != "zeros":
        raise ValueError("parallel PRU supports ordinary zero-padded Conv2d only")
    signature = (
        reference.in_channels,
        reference.out_channels,
        reference.kernel_size,
        reference.stride,
        reference.padding,
        reference.dilation,
        reference.groups,
        reference.bias is not None,
    )
    for conv in convs[1:]:
        current = (
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
        )
        if current != signature:
            raise ValueError(f"parallel Conv2d signatures differ: {signature} and {current}")
    shape = inputs[0].shape
    if any(value.shape[0] != shape[0] or value.shape[-2:] != shape[-2:] for value in inputs):
        raise ValueError("parallel Conv2d inputs must have equal batch/spatial shapes")

    joined = torch.cat(inputs, dim=1)
    weight = torch.cat([conv.weight for conv in convs], dim=0)
    bias = (
        torch.cat([conv.bias for conv in convs], dim=0)
        if reference.bias is not None
        else None
    )
    output = F.conv2d(
        joined,
        weight,
        bias,
        stride=reference.stride,
        padding=reference.padding,
        dilation=reference.dilation,
        groups=len(convs),
    )
    return list(output.split(reference.out_channels, dim=1))


def _parallel_basic_conv(modules, inputs):
    outputs = _parallel_conv2d([module.conv for module in modules], inputs)
    if modules[0].use_bn:
        outputs = [module.bn(value) for module, value in zip(modules, outputs)]
    if modules[0].relu:
        outputs = [F.leaky_relu(value) for value in outputs]
    return outputs


def _parallel_residual(units, inputs):
    outputs = [unit.activation(value) for unit, value in zip(units, inputs)]
    outputs = _parallel_conv2d([unit.conv1 for unit in units], outputs)
    if units[0].bn:
        outputs = [unit.bn1(value) for unit, value in zip(units, outputs)]
    outputs = [unit.activation(value) for unit, value in zip(units, outputs)]
    outputs = _parallel_conv2d([unit.conv2 for unit in units], outputs)
    if units[0].bn:
        outputs = [unit.bn2(value) for unit, value in zip(units, outputs)]
    return [value + skip for value, skip in zip(outputs, inputs)]


def _parallel_basic_then_conv(sequences, inputs):
    outputs = _parallel_basic_conv([sequence[0] for sequence in sequences], inputs)
    return _parallel_conv2d([sequence[1] for sequence in sequences], outputs)


def _parallel_pool(inputs):
    batch = inputs[0].shape[0]
    output = F.avg_pool2d(torch.cat(inputs, dim=0), 3, stride=2, padding=1)
    return list(output.split(batch, dim=0))


def _parallel_interp(inputs, destinations):
    batch = inputs[0].shape[0]
    size = destinations[0].shape[-2:]
    if any(value.shape[-2:] != size for value in destinations):
        raise ValueError("parallel interpolation destinations must have equal shapes")
    output = F.interpolate(
        torch.cat(inputs, dim=0), size, mode="bilinear", align_corners=True
    )
    return list(output.split(batch, dim=0))


def _parallel_structure(blocks, disps, static_features):
    batch = disps[0].shape[0]
    normalized, _, _ = normalize_disparity(torch.cat(disps, dim=0))
    normalized = list(normalized.split(batch, dim=0))
    residuals = [
        static[1] - norm_disp
        for static, norm_disp in zip(static_features, normalized)
    ]
    encoders = [block.structure_encoder for block in blocks]
    values = _parallel_conv2d([encoder.convd1 for encoder in encoders], residuals)
    values = [F.relu(value, inplace=False) for value in values]
    values = _parallel_conv2d([encoder.convd2 for encoder in encoders], values)
    values = [F.relu(value, inplace=False) for value in values]
    values = [
        torch.cat([static[0], value], dim=1)
        for static, value in zip(static_features, values)
    ]
    values = _parallel_conv2d([encoder.conv for encoder in encoders], values)
    return [F.relu(value, inplace=False) for value in values]


def _parallel_motion(blocks, correlations, disps, warp_features):
    encoders = [block.motion_encoder for block in blocks]
    correlation = _parallel_conv2d(
        [encoder.convc1 for encoder in encoders], correlations
    )
    correlation = [F.relu(value, inplace=False) for value in correlation]
    correlation = _parallel_conv2d(
        [encoder.convc2 for encoder in encoders], correlation
    )
    correlation = [F.relu(value, inplace=False) for value in correlation]

    disparity = _parallel_conv2d(
        [encoder.convd1 for encoder in encoders], disps
    )
    disparity = [F.relu(value, inplace=False) for value in disparity]
    disparity = _parallel_conv2d(
        [encoder.convd2 for encoder in encoders], disparity
    )
    disparity = [F.relu(value, inplace=False) for value in disparity]

    warp = _parallel_conv2d(
        [encoder.convg1 for encoder in encoders], warp_features
    )
    warp = [F.relu(value, inplace=False) for value in warp]
    warp = _parallel_conv2d([encoder.convg2 for encoder in encoders], warp)
    warp = [F.relu(value, inplace=False) for value in warp]

    joined = [
        torch.cat([corr, disp_feature, warp_feature], dim=1)
        for corr, disp_feature, warp_feature in zip(correlation, disparity, warp)
    ]
    output = _parallel_conv2d([encoder.conv for encoder in encoders], joined)
    output = [F.relu(value, inplace=False) for value in output]
    return [
        torch.cat([value, disp], dim=1)
        for value, disp in zip(output, disps)
    ]


def _parallel_update_gate(blocks, level, inputs):
    sequences = [block.update[level] for block in blocks]
    output = _parallel_basic_conv(
        [sequence[0] for sequence in sequences], inputs
    )
    output = _parallel_conv2d(
        [sequence[1] for sequence in sequences], output
    )
    return [torch.sigmoid(value) for value in output]


def _parallel_pru(
    blocks, level, hidden, residual=None, structure=None, motion=None
):
    units = [block.stereo_pru[level] for block in blocks]
    output = hidden
    if residual is not None:
        residual_units = [unit.resConfUnit1 for unit in units]
        values = _parallel_residual(residual_units, residual)
        output = [value + skip for value, skip in zip(output, values)]

    output = _parallel_residual(
        [unit.resConfUnit2 for unit in units], output
    )
    if structure is not None:
        values = _parallel_basic_then_conv(
            [unit.resConfUnitStructure for unit in units], structure
        )
        output = [value + extra for value, extra in zip(output, values)]
    if motion is not None:
        values = _parallel_basic_then_conv(
            [unit.resConfUnitMotion for unit in units], motion
        )
        output = [value + extra for value, extra in zip(output, values)]
    return _parallel_conv2d([unit.out_conv for unit in units], output)


def parallel_coarse_update(
    blocks,
    nets,
    correlations,
    disps,
    contexts,
    warp_features,
    mono_disps,
    static_features,
    need_mask,
):
    """Execute independent equal-shaped MultiPromptUpdateBlocks in parallel."""
    branch_count = len(blocks)
    if not (
        branch_count
        == len(nets)
        == len(correlations)
        == len(disps)
        == len(warp_features)
        == len(static_features)
    ):
        raise ValueError("parallel coarse PRU input counts differ")
    if branch_count == 1:
        block = blocks[0]
        return [block(
            nets[0], correlations[0], disps[0], contexts[0],
            warp_features[0], mono_disps[0],
            static_features=static_features[0], need_mask=need_mask,
        )]

    del contexts, mono_disps
    num_layers = blocks[0].num_layers
    if any(block.num_layers != num_layers for block in blocks):
        raise ValueError("parallel coarse PRUs must have equal layer counts")
    structure = _parallel_structure(blocks, disps, static_features)
    motion = _parallel_motion(blocks, correlations, disps, warp_features)

    for level in reversed(range(num_layers)):
        hidden = [net[level] for net in nets]
        if num_layers == 1:
            gate_input = [
                torch.cat([value, struct, mot], dim=1)
                for value, struct, mot in zip(hidden, structure, motion)
            ]
            gate = _parallel_update_gate(blocks, level, gate_input)
            proposal = _parallel_pru(
                blocks, level, hidden, structure=structure, motion=motion
            )
        elif level == num_layers - 1:
            pooled = _parallel_pool([net[level - 1] for net in nets])
            gate_input = [
                torch.cat([value, lower], dim=1)
                for value, lower in zip(hidden, pooled)
            ]
            gate = _parallel_update_gate(blocks, level, gate_input)
            proposal = _parallel_pru(blocks, level, hidden)
        elif level == 0:
            interpolated = _parallel_interp(
                [net[level + 1] for net in nets], hidden
            )
            gate_input = [
                torch.cat([value, struct, mot], dim=1)
                for value, struct, mot in zip(hidden, structure, motion)
            ]
            gate = _parallel_update_gate(blocks, level, gate_input)
            proposal = _parallel_pru(
                blocks,
                level,
                hidden,
                residual=interpolated,
                structure=structure,
                motion=motion,
            )
        else:
            pooled = _parallel_pool([net[level - 1] for net in nets])
            interpolated = _parallel_interp(
                [net[level + 1] for net in nets], hidden
            )
            gate_input = [
                torch.cat([value, lower], dim=1)
                for value, lower in zip(hidden, pooled)
            ]
            gate = _parallel_update_gate(blocks, level, gate_input)
            proposal = _parallel_pru(
                blocks, level, hidden, residual=interpolated
            )
        for branch in range(branch_count):
            nets[branch][level] = (
                (1.0 - gate[branch]) * hidden[branch]
                + gate[branch] * proposal[branch]
            )

    head1 = _parallel_conv2d(
        [block.disp_head.conv1 for block in blocks],
        [net[0] for net in nets],
    )
    head1 = [F.relu(value, inplace=False) for value in head1]
    deltas = _parallel_conv2d(
        [block.disp_head.conv2 for block in blocks], head1
    )
    if need_mask:
        mask1 = _parallel_conv2d(
            [block.mask[0] for block in blocks],
            [net[0] for net in nets],
        )
        mask1 = [F.relu(value, inplace=False) for value in mask1]
        masks = _parallel_conv2d(
            [block.mask[2] for block in blocks], mask1
        )
        masks = [0.25 * value for value in masks]
    else:
        masks = [None] * branch_count
    return [
        (net, delta, mask)
        for net, delta, mask in zip(nets, deltas, masks)
    ]
