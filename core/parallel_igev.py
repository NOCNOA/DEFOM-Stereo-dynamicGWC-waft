"""Grouped execution for independent CorGA coarse IGEV update blocks."""

import torch
import torch.nn.functional as F

from core.parallel_pru import (
    _parallel_conv2d,
    _parallel_interp,
    _parallel_motion,
    _parallel_pool,
)


def _parallel_gru(blocks, level, hidden, features, context_gates):
    cells = [block.gru_layers[level] for block in blocks]
    hidden_features = [
        torch.cat((state, inputs), dim=1)
        for state, inputs in zip(hidden, features)
    ]
    update = _parallel_conv2d([cell.convz for cell in cells], hidden_features)
    reset = _parallel_conv2d([cell.convr for cell in cells], hidden_features)
    update = [
        torch.sigmoid(value + gates[level][0])
        for value, gates in zip(update, context_gates)
    ]
    reset = [
        torch.sigmoid(value + gates[level][1])
        for value, gates in zip(reset, context_gates)
    ]
    candidate_inputs = [
        torch.cat((gate * state, inputs), dim=1)
        for gate, state, inputs in zip(reset, hidden, features)
    ]
    candidate = _parallel_conv2d(
        [cell.convq for cell in cells], candidate_inputs
    )
    candidate = [
        torch.tanh(value + gates[level][2])
        for value, gates in zip(candidate, context_gates)
    ]
    return [
        (1.0 - gate) * state + gate * proposal
        for gate, state, proposal in zip(update, hidden, candidate)
    ]


def parallel_igev_coarse_update(
    blocks,
    nets,
    correlations,
    disps,
    contexts,
    warp_features,
    anchor_disps,
    static_features,
    need_mask,
):
    """Run equal-shaped, independently parameterized IGEV branches together."""
    branch_count = len(blocks)
    if not (
        branch_count
        == len(nets)
        == len(correlations)
        == len(disps)
        == len(contexts)
        == len(warp_features)
        == len(anchor_disps)
        == len(static_features)
    ):
        raise ValueError("parallel IGEV coarse input counts differ")
    # PyTorch 2.1's CPU oneDNN grouped-convolution backward can mis-route
    # gradients when the grouped weight is assembled with torch.cat. Training
    # uses CUDA; keep CPU diagnostics on the exact sequential implementation.
    if branch_count == 1 or not disps[0].is_cuda:
        return [
            block(
                net, correlation, disp, context, warp, anchor,
                static_features=gates, need_mask=need_mask,
            )
            for block, net, correlation, disp, context, warp, anchor, gates in zip(
                blocks, nets, correlations, disps, contexts, warp_features,
                anchor_disps, static_features,
            )
        ]

    num_layers = blocks[0].num_layers
    if any(block.num_layers != num_layers for block in blocks):
        raise ValueError("parallel IGEV blocks must have equal layer counts")
    if any(len(gates) != num_layers for gates in static_features):
        raise ValueError("parallel IGEV context levels differ")

    motion = _parallel_motion(blocks, correlations, disps, warp_features)
    for level in reversed(range(num_layers)):
        hidden = [net[level] for net in nets]
        finer = (
            _parallel_pool([net[level - 1] for net in nets])
            if level > 0 else None
        )
        coarser = (
            _parallel_interp([net[level + 1] for net in nets], hidden)
            if level + 1 < num_layers else None
        )
        features = []
        for branch in range(branch_count):
            inputs = []
            if level == 0:
                inputs.append(motion[branch])
            if finer is not None:
                inputs.append(finer[branch])
            if coarser is not None:
                inputs.append(coarser[branch])
            features.append(torch.cat(inputs, dim=1))
        updated = _parallel_gru(
            blocks, level, hidden, features, static_features
        )
        for branch, value in enumerate(updated):
            nets[branch][level] = value

    head = _parallel_conv2d(
        [block.disp_head.conv1 for block in blocks],
        [net[0] for net in nets],
    )
    head = [F.relu(value, inplace=False) for value in head]
    deltas = _parallel_conv2d(
        [block.disp_head.conv2 for block in blocks], head
    )
    if need_mask:
        mask_features = _parallel_conv2d(
            [block.mask[0] for block in blocks], [net[0] for net in nets]
        )
        mask_features = [F.relu(value, inplace=False) for value in mask_features]
        masks = _parallel_conv2d(
            [block.mask[2] for block in blocks], mask_features
        )
        masks = [0.25 * value for value in masks]
    else:
        masks = [None] * branch_count
    return list(zip(nets, deltas, masks))
