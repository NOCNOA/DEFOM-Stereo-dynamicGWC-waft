"""Pointwise recurrent stereo refinement at arbitrary full-resolution queries.

All disparity values and query coordinates at this interface use original-image
pixels. The 1/2-resolution feature maps are sampled on demand; no HR cost
volume, dense hidden map, or query interaction is constructed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_map(feature, xy, image_size, mode="bilinear"):
    """Sample [B,C,h,w] at [B,N,2] or [B,N,K,2] original-pixel centers."""
    if xy.ndim not in (3, 4) or xy.shape[-1] != 2:
        raise ValueError("xy must be [B,N,2] or [B,N,K,2]")
    if feature.shape[0] != xy.shape[0]:
        raise ValueError("batch size mismatch")
    image_h, image_w = image_size
    feat_h, feat_w = feature.shape[-2:]
    flat = xy.reshape(xy.shape[0], -1, 2).float()
    # align_corners=True maps original pixel centers consistently to all scales.
    x = (flat[..., 0] + .5) * feat_w / image_w - .5
    y = (flat[..., 1] + .5) * feat_h / image_h - .5
    grid = torch.stack((2 * x / max(feat_w - 1, 1) - 1,
                        2 * y / max(feat_h - 1, 1) - 1), dim=-1)
    sampled = F.grid_sample(feature.float(), grid.unsqueeze(2), mode=mode,
                            padding_mode="border", align_corners=True)
    values = sampled.squeeze(-1).transpose(1, 2)
    return values.reshape(*xy.shape[:-1], feature.shape[1])


def sample_patch(feature, xy, image_size, radius, scale):
    offsets = torch.arange(-radius, radius + 1, device=xy.device, dtype=xy.dtype) * scale
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    patch_xy = xy.unsqueeze(-2) + torch.stack((ox, oy), dim=-1).reshape(1, 1, -1, 2)
    return sample_map(feature, patch_xy, image_size)


def disparity_boundary(gt, valid, threshold=2.0):
    """Mark both sides of a valid GT disparity jump in original pixels."""
    if gt.shape != valid.shape or gt.ndim != 4 or gt.shape[1] != 1:
        raise ValueError("gt and valid must be [B,1,H,W]")
    edge = torch.zeros_like(valid, dtype=torch.bool)
    jump_x = valid[..., 1:] & valid[..., :-1] & ((gt[..., 1:] - gt[..., :-1]).abs() > threshold)
    jump_y = valid[..., 1:, :] & valid[..., :-1, :] & ((gt[..., 1:, :] - gt[..., :-1, :]).abs() > threshold)
    edge[..., 1:] |= jump_x
    edge[..., :-1] |= jump_x
    edge[..., 1:, :] |= jump_y
    edge[..., :-1, :] |= jump_y
    return edge


def dilate(mask, radius):
    if radius == 0:
        return mask
    return F.max_pool2d(mask.float(), 2 * radius + 1, stride=1, padding=radius) > 0


def sample_training_queries(gt, valid, n=4096, threshold=2.0,
                            band_radius=4, neighborhood_radius=8,
                            query_mode="mixed"):
    """Sample the GT boundary band alone or a band/ring/random mixture."""
    if n < 4:
        raise ValueError("n must be at least 4")
    if query_mode not in ("mixed", "boundary_only"):
        raise ValueError(f"Unknown query mode: {query_mode}")
    valid = valid.bool() & torch.isfinite(gt) & (gt > 0)
    edge = disparity_boundary(gt, valid, threshold)
    band = dilate(edge, band_radius) & valid
    ring = dilate(edge, neighborhood_radius) & valid & ~band
    counts = (n // 2, n // 4, n - n // 2 - n // 4)
    output = []
    for batch in range(gt.shape[0]):
        pools = (band[batch, 0], ring[batch, 0], valid[batch, 0])
        valid_xy = torch.nonzero(pools[-1], as_tuple=False)
        if valid_xy.numel() == 0:
            raise ValueError("An image has no valid disparity queries")
        if query_mode == "boundary_only":
            positions = torch.nonzero(pools[0], as_tuple=False)
            if positions.numel() == 0:
                raise ValueError("An image has no valid GT boundary queries")
            indices = torch.randint(len(positions), (n,), device=gt.device)
            yx = positions[indices]
            output.append(yx[:, [1, 0]].float())
            continue
        chosen = []
        for pool, count in zip(pools, counts):
            positions = torch.nonzero(pool, as_tuple=False)
            if positions.numel() == 0:
                positions = valid_xy
            indices = torch.randint(len(positions), (count,), device=gt.device)
            chosen.append(positions[indices])
        yx = torch.cat(chosen, dim=0)
        yx = yx[torch.randperm(n, device=gt.device)]
        output.append(yx[:, [1, 0]].float())
    return torch.stack(output)


def dense_query_coordinates(mask):
    """Return one [1,N,2] tensor per image for a possibly irregular mask."""
    return [torch.nonzero(mask[b, 0], as_tuple=False)[:, [1, 0]].float().unsqueeze(0)
            for b in range(mask.shape[0])]


class ShallowStereoFeature(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, image):
        return self.net(image.float() / 127.5 - 1.0)


class CandidateScorer(nn.Module):
    """Shared score for each local disparity candidate and every query."""

    def __init__(self, feature_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * feature_channels + 2, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, tokens):
        return self.net(tokens).squeeze(-1)


def candidate_tokens(left, right, offsets, inside):
    """Build the exact token layout used by the frozen candidate probe."""
    left = left.expand_as(right)
    offset = (offsets / offsets.abs().max()).view(1, 1, -1, 1).expand(
        *right.shape[:-1], 1)
    return torch.cat((left * right, (left - right).abs(), left - right,
                      offset, inside.unsqueeze(-1).float()), dim=-1)


class SparseHRRefiner(nn.Module):
    def __init__(self, coarse_hidden_channels=128, coarse_context_channels=128,
                 feature_channels=16, hidden_channels=64, context_channels=64,
                 iterations=4, delta_radius=2.0, search_radius=4,
                 groups=4, max_disp=768.0, gate_init_bias=-2.0,
                 direction_classes=False, candidate_guided=False,
                 context_scoring=False):
        super().__init__()
        if feature_channels % groups:
            raise ValueError("feature_channels must be divisible by groups")
        self.iterations = iterations
        self.delta_radius = float(delta_radius)
        self.max_disp = float(max_disp)
        self.groups = groups
        self.direction_classes = bool(direction_classes)
        self.candidate_guided = bool(candidate_guided)
        self.context_scoring = bool(context_scoring)
        if self.context_scoring and not self.candidate_guided:
            raise ValueError("Context scoring requires candidate guidance")
        self.feature = ShallowStereoFeature(feature_channels)
        self.hidden_transfer = nn.Sequential(
            nn.Linear(9 * coarse_hidden_channels, hidden_channels), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))
        half_context = context_channels // 2
        self.coarse_context_encoder = nn.Sequential(
            nn.Linear(25 * coarse_context_channels, half_context), nn.ReLU(inplace=True))
        self.fine_context_encoder = nn.Sequential(
            nn.Linear(25 * feature_channels, half_context), nn.ReLU(inplace=True))
        self.context_fusion = nn.Sequential(
            nn.Linear(2 * half_context, context_channels), nn.ReLU(inplace=True))
        self.difference_projection = nn.Linear(feature_channels, feature_channels)
        offsets = torch.arange(-search_radius, search_radius + 1, dtype=torch.float32)
        self.register_buffer("offsets", offsets)
        candidate_width = groups + feature_channels + 2
        self.motion_encoder = nn.Sequential(
            nn.Linear(len(offsets) * candidate_width, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True))
        self.disparity_embedding = nn.Sequential(nn.Linear(2, 16), nn.ReLU(inplace=True))
        self.gru = nn.GRUCell(64 + context_channels + 16 + 1, hidden_channels)
        self.delta_head = nn.Linear(hidden_channels, 1)
        self.gate_head = nn.Linear(hidden_channels, 1)
        if self.candidate_guided:
            self.candidate_scorer = CandidateScorer(feature_channels)
            self.register_buffer("candidate_mean", torch.zeros(1, 1, 3 * feature_channels + 2))
            self.register_buffer("candidate_std", torch.ones(1, 1, 3 * feature_channels + 2))
            self.candidate_gate = nn.Sequential(
                nn.Linear(hidden_channels + len(offsets) + 3, 64), nn.ReLU(),
                nn.Linear(64, 1))
            nn.init.constant_(self.candidate_gate[-1].bias, -2.)
            if self.context_scoring:
                self.context_candidate_scorer = nn.Sequential(
                    nn.Linear(3 * feature_channels + 2 + context_channels, 64),
                    nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
                # Preserve the pretrained scores at initialization.
                nn.init.zeros_(self.context_candidate_scorer[-1].weight)
                nn.init.zeros_(self.context_candidate_scorer[-1].bias)
        if self.direction_classes:
            # Classes: decrease disparity, stay, increase disparity.
            self.direction_head = nn.Linear(hidden_channels, 3)
            nn.init.normal_(self.direction_head.weight, std=1e-3)
            nn.init.zeros_(self.direction_head.bias)
            with torch.no_grad():
                self.direction_head.bias[1] = 1.
        nn.init.normal_(self.delta_head.weight, std=1e-3)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.constant_(self.gate_head.bias, float(gate_init_bias))

    def load_candidate_probe(self, path):
        if not self.candidate_guided:
            raise ValueError("Enable candidate_guided before loading a probe")
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.candidate_scorer.load_state_dict(state["probe"], strict=True)
        self.candidate_mean.copy_(state["feature_mean"])
        self.candidate_std.copy_(state["feature_std"])

    def _motion(self, left_q, right_feature, xy, disparity, image_size, context=None):
        image_h, image_w = image_size
        candidates = disparity + self.offsets.view(1, 1, -1)
        right_xy = xy.unsqueeze(-2).expand(-1, -1, len(self.offsets), -1).clone()
        right_xy[..., 0] -= candidates
        right = sample_map(right_feature, right_xy, image_size)
        left = left_q.unsqueeze(-2).expand_as(right)
        batch, n, k, channels = right.shape
        corr = (left.reshape(batch, n, k, self.groups, -1)
                * right.reshape(batch, n, k, self.groups, -1)).mean(dim=-1)
        difference = self.difference_projection(left - right)
        offset = (self.offsets / max(float(self.offsets.abs().max()), 1.0))
        offset = offset.view(1, 1, k, 1).expand(batch, n, -1, -1)
        inside = ((right_xy[..., 0] >= 0) & (right_xy[..., 0] <= image_w - 1)
                  & (right_xy[..., 1] >= 0) & (right_xy[..., 1] <= image_h - 1))
        token = torch.cat((corr, difference, offset, inside.unsqueeze(-1).float()), dim=-1)
        motion = self.motion_encoder(token.flatten(start_dim=2))
        if not self.candidate_guided:
            return motion
        probe_token = candidate_tokens(left_q.unsqueeze(-2), right, self.offsets, inside)
        normalized = (probe_token - self.candidate_mean) / self.candidate_std
        score = self.candidate_scorer(normalized)
        if self.context_scoring:
            local_context = context.unsqueeze(-2).expand(-1, -1, k, -1)
            score = score + self.context_candidate_scorer(
                torch.cat((normalized, local_context), dim=-1)).squeeze(-1)
        return motion, score

    def forward(self, image_left, image_right, coarse_full_disp,
                disp_lr_px, hidden_lr, context_lr, query_xy, iterations=None,
                feature_left=None, feature_right=None,
                return_direction_logits=False, return_candidate_info=False):
        """Return predictions [d0,...,dT], each [B,N,1] in original pixels."""
        image_size = image_left.shape[-2:]
        if image_right.shape != image_left.shape or query_xy.shape[-1] != 2:
            raise ValueError("Image or query shape mismatch")
        if coarse_full_disp.shape[-2:] != image_size:
            raise ValueError("coarse_full_disp must be at original resolution")
        if return_direction_logits and not self.direction_classes:
            raise ValueError("Direction logits require direction_classes=True")
        if return_candidate_info and not self.candidate_guided:
            raise ValueError("Candidate info requires candidate_guided=True")
        if query_xy.shape[1] == 0:
            empty = query_xy.new_empty(*query_xy.shape[:2], 1)
            predictions = [empty] * ((iterations or self.iterations) + 1)
            if return_direction_logits:
                logits = [query_xy.new_empty(*query_xy.shape[:2], 3)] * (len(predictions) - 1)
                return predictions, logits
            if return_candidate_info:
                scores = query_xy.new_empty(*query_xy.shape[:2], len(self.offsets))
                info = [(empty, empty, scores)] * (len(predictions) - 1)
                return predictions, info
            return predictions
        if feature_left is None:
            feature_left = self.feature(image_left)
        if feature_right is None:
            feature_right = self.feature(image_right)
        # Sample both low-resolution geometry and the model's convex-upsampled output.
        # The static correction preserves the original dense prediction at step 0.
        low_base = sample_map(disp_lr_px, query_xy, image_size)
        full_base = sample_map(coarse_full_disp, query_xy, image_size)
        upsample_correction = full_base - low_base
        base = low_base + upsample_correction
        hidden_patch = sample_patch(hidden_lr, query_xy, image_size, 1, 4)
        hidden = self.hidden_transfer(hidden_patch.flatten(start_dim=2))
        coarse_patch = sample_patch(context_lr, query_xy, image_size, 2, 4)
        fine_patch = sample_patch(feature_left, query_xy, image_size, 2, 2)
        coarse_context = self.coarse_context_encoder(coarse_patch.flatten(start_dim=2))
        fine_context = self.fine_context_encoder(fine_patch.flatten(start_dim=2))
        context = self.context_fusion(torch.cat((coarse_context, fine_context), dim=-1))
        left_q = sample_map(feature_left, query_xy, image_size)
        residual = torch.zeros_like(base)
        previous_delta = torch.zeros_like(base)
        predictions = [base]
        direction_logits = []
        candidate_info = []
        for _ in range(self.iterations if iterations is None else iterations):
            current = base + residual
            lookup = self._motion(left_q, feature_right, query_xy, current, image_size, context)
            motion, candidate_scores = lookup if self.candidate_guided else (lookup, None)
            disp_embed = self.disparity_embedding(torch.cat(
                (current / self.max_disp, residual / max(self.delta_radius, 1.0)), dim=-1))
            gru_input = torch.cat((motion, disp_embed, context, previous_delta), dim=-1)
            hidden = self.gru(gru_input.reshape(-1, gru_input.shape[-1]),
                              hidden.reshape(-1, hidden.shape[-1])).reshape_as(hidden)
            if self.candidate_guided:
                probabilities = candidate_scores.softmax(dim=-1)
                selected = candidate_scores.argmax(dim=-1)
                proposal = self.offsets[selected].unsqueeze(-1).clamp(
                    -self.delta_radius, self.delta_radius)
                gate_input = torch.cat((hidden, probabilities,
                                        proposal / max(self.delta_radius, 1.),
                                        current / self.max_disp,
                                        previous_delta / max(self.delta_radius, 1.)), dim=-1)
                gate_logits = self.candidate_gate(gate_input)
                gated_delta = torch.sigmoid(gate_logits) * proposal
                candidate_info.append((proposal, gate_logits, candidate_scores))
            else:
                delta = self.delta_radius * torch.tanh(self.delta_head(hidden))
                if self.direction_classes:
                    logits = self.direction_head(hidden)
                    direction_logits.append(logits)
                    probabilities = logits.softmax(dim=-1)
                    signed_confidence = probabilities[..., 2:3] - probabilities[..., 0:1]
                    delta = delta.abs() * signed_confidence
                gated_delta = torch.sigmoid(self.gate_head(hidden)) * delta
            residual = (residual + gated_delta).clamp(-self.max_disp, self.max_disp)
            previous_delta = gated_delta
            predictions.append((base + residual).clamp(0, self.max_disp))
        if return_direction_logits:
            return predictions, direction_logits
        if return_candidate_info:
            return predictions, candidate_info
        return predictions


def update_quality_totals(base, refined, gt, mask, update_threshold=1e-3):
    """Raw pixel sums for pooling update coverage, accuracy and error gain."""
    initial_error = (base - gt).abs()[mask]
    final_error = (refined - gt).abs()[mask]
    update = (refined - base)[mask]
    desired = (gt - base)[mask]
    updated = update.abs() > update_threshold
    gain = initial_error - final_error
    return {"pixels": int(mask.sum()), "updated_pixels": int(updated.sum()),
            "improved_updated_pixels": int((updated & (gain > 1e-6)).sum()),
            "worsened_updated_pixels": int((updated & (gain < -1e-6)).sum()),
            "correct_direction_updated_pixels": int((updated & (update * desired > 0)).sum()),
            "initial_error_sum": float(initial_error.sum()),
            "final_error_sum": float(final_error.sum()),
            "error_reduction_sum": float(gain.clamp_min(0).sum()),
            "error_increase_sum": float((-gain).clamp_min(0).sum())}


def summarize_update_quality(value):
    pixels, updated = value["pixels"], value["updated_pixels"]
    gain = value["error_reduction_sum"] - value["error_increase_sum"]
    return {**value,
            "update_coverage_percent": 100 * updated / pixels if pixels else None,
            "update_improvement_percent": 100 * value["improved_updated_pixels"] / updated if updated else None,
            "update_worsening_percent": 100 * value["worsened_updated_pixels"] / updated if updated else None,
            "direction_accuracy_updated_percent": 100 * value["correct_direction_updated_pixels"] / updated if updated else None,
            "initial_epe": value["initial_error_sum"] / pixels if pixels else None,
            "final_epe": value["final_error_sum"] / pixels if pixels else None,
            "net_error_reduction_sum": gain,
            "net_epe_reduction": gain / pixels if pixels else None}


def scatter_queries(coarse_full_disp, query_xy, refined_disp):
    """Write integer query predictions into a cloned dense coarse prediction."""
    batch, _, height, width = coarse_full_disp.shape
    if query_xy.shape != (batch, refined_disp.shape[1], 2):
        raise ValueError("query and refined disparity shapes differ")
    x, y = query_xy[..., 0], query_xy[..., 1]
    if not torch.all(x == x.round()) or not torch.all(y == y.round()):
        raise ValueError("scatter queries must be integer pixel centers")
    if not torch.all((x >= 0) & (x < width) & (y >= 0) & (y < height)):
        raise ValueError("scatter query outside the image")
    index = (y.long() * width + x.long())
    output = coarse_full_disp.clone().flatten(start_dim=2)
    output.scatter_(2, index.unsqueeze(1), refined_disp.transpose(1, 2))
    return output.reshape_as(coarse_full_disp)


def sparse_sequence_loss(predictions, gt, valid, query_xy, gamma=.9,
                         hard_error_weight=0.):
    """Sequence L1 at integer GT queries, with final iteration weight one."""
    gt_q = sample_map(gt, query_xy, gt.shape[-2:], mode="nearest")
    valid_q = sample_map(valid.float(), query_xy, gt.shape[-2:], mode="nearest") > .5
    valid_q &= torch.isfinite(gt_q) & (gt_q > 0)
    if not valid_q.any():
        raise ValueError("No valid supervised queries")
    if hard_error_weight < 0:
        raise ValueError("hard_error_weight must be non-negative")
    initial_error = (predictions[0].detach() - gt_q).abs()
    correctable = (initial_error >= 1.) & (initial_error <= 8.)
    weights = 1. + hard_error_weight * correctable.float()
    total = gt_q.new_zeros(())
    for index, prediction in enumerate(predictions[1:], start=1):
        weight = gamma ** (len(predictions) - 1 - index)
        error = (prediction[valid_q] - gt_q[valid_q]).abs()
        total = total + weight * (error * weights[valid_q]).sum() / weights[valid_q].sum()
    return total


def sparse_direction_loss(direction_logits, predictions, gt, valid, query_xy,
                          gamma=.9, nonzero_weight=2.):
    """Supervise the next update sign from the detached current GT residual."""
    if len(direction_logits) != len(predictions) - 1:
        raise ValueError("Expected one direction logit tensor per HR update")
    gt_q = sample_map(gt, query_xy, gt.shape[-2:], mode="nearest")
    valid_q = sample_map(valid.float(), query_xy, gt.shape[-2:], mode="nearest") > .5
    valid_q &= torch.isfinite(gt_q) & (gt_q > 0)
    if not valid_q.any():
        raise ValueError("No valid supervised direction queries")
    total = gt_q.new_zeros(())
    for index, logits in enumerate(direction_logits):
        residual = gt_q - predictions[index].detach()
        target = torch.ones_like(residual, dtype=torch.long)
        target[residual <= -1.] = 0
        target[residual >= 1.] = 2
        labels = target[valid_q]
        loss = F.cross_entropy(logits[valid_q.squeeze(-1)], labels,
                               reduction="none")
        weights = torch.where(labels == 1, 1., nonzero_weight)
        total = total + gamma ** (len(direction_logits) - 1 - index) * (
            loss * weights).sum() / weights.sum()
    return total


def candidate_selection_loss(candidate_info, predictions, gt, valid, query_xy,
                             offsets, gamma=.9):
    """Nearest-offset CE, balanced between easy and reachable hard queries.

    Excludes GT correspondences outside the right image and residuals beyond
    the search range. This is a geometric mask, not an occlusion mask.
    """
    gt_q = sample_map(gt, query_xy, gt.shape[-2:], mode="nearest")
    valid_q = sample_map(valid.float(), query_xy, gt.shape[-2:], mode="nearest") > .5
    right_x = query_xy[..., 0:1] - gt_q
    valid_q &= torch.isfinite(gt_q) & (gt_q > 0)
    valid_q &= (right_x >= 0) & (right_x <= gt.shape[-1] - 1)
    total = candidate_info[0][2].sum() * 0.
    for index, (_, _, scores) in enumerate(candidate_info):
        residual = gt_q - predictions[index].detach()
        reachable = valid_q & (residual >= offsets.min()) & (residual <= offsets.max())
        labels = (residual - offsets.view(1, 1, -1)).abs().argmin(dim=-1)
        losses = F.cross_entropy(scores.transpose(1, 2), labels, reduction="none")
        parts = []
        for group in (reachable & (residual.abs() <= 1.),
                      reachable & (residual.abs() > 1.)):
            if group.any():
                parts.append(losses[group.squeeze(-1)].mean())
        if parts:
            total = total + gamma ** (len(candidate_info) - 1 - index) * torch.stack(parts).mean()
    return total


def candidate_gate_loss(candidate_info, predictions, gt, valid, query_xy,
                        gamma=.9, positive_weight=2., target_mode="binary"):
    """Train abstention from the GT gain of each frozen candidate proposal."""
    if len(candidate_info) != len(predictions) - 1:
        raise ValueError("Expected one candidate proposal per HR update")
    gt_q = sample_map(gt, query_xy, gt.shape[-2:], mode="nearest")
    valid_q = sample_map(valid.float(), query_xy, gt.shape[-2:], mode="nearest") > .5
    valid_q &= torch.isfinite(gt_q) & (gt_q > 0)
    if not valid_q.any():
        raise ValueError("No valid gate supervision queries")
    total = gt_q.new_zeros(())
    for index, information in enumerate(candidate_info):
        proposal, gate_logits = information[:2]
        current = predictions[index].detach()
        change = proposal.detach()
        if target_mode == "ratio":
            mask = valid_q & (change.abs() > 1e-6)
            safe_change = torch.where(change.abs() > 1e-6, change, torch.ones_like(change))
            target = ((gt_q - current) / safe_change).clamp(0., 1.)
            if mask.any():
                total = total + gamma ** (len(candidate_info) - 1 - index) * F.smooth_l1_loss(
                    torch.sigmoid(gate_logits)[mask], target[mask])
            else:
                total = total + gate_logits.sum() * 0.
            continue
        if target_mode != "binary":
            raise ValueError(f"Unknown gate target mode: {target_mode}")
        gain = (current - gt_q).abs() - (current + change - gt_q).abs()
        target = (gain > .05).float()
        weights = torch.where(target > .5, positive_weight, 1.)
        bce = F.binary_cross_entropy_with_logits(gate_logits[valid_q],
                                                  target[valid_q], reduction="none")
        total = total + gamma ** (len(candidate_info) - 1 - index) * (
            bce * weights[valid_q]).sum() / weights[valid_q].sum()
    return total
