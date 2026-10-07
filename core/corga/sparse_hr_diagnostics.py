"""GT interventions for diagnosing a single sparse HR update."""
import torch


def optimal_gate(residual, proposal):
    nonzero = proposal.abs() > 1e-6
    safe = torch.where(nonzero, proposal, torch.ones_like(proposal))
    return torch.where(nonzero, (residual / safe).clamp(0., 1.), torch.zeros_like(proposal))


def optimal_candidate(residual, offsets, delta_radius):
    """Choose the best attainable gated candidate; break ties by GT proximity.

    Optimize each candidate's gate before selecting. Nearest-integer selection
    alone would miss correctable subpixel residuals when it selects offset zero.
    """
    proposals = offsets.clamp(-delta_radius, delta_radius).view(1, 1, -1)
    gates = optimal_gate(residual, proposals)
    error = (residual - proposals * gates).abs()
    best = error.amin(dim=-1, keepdim=True)
    distance = (residual - offsets.view(1, 1, -1)).abs()
    distance = torch.where(error <= best + 1e-6, distance, torch.full_like(distance, float("inf")))
    index = distance.argmin(dim=-1, keepdim=True)
    proposal = proposals.expand(*residual.shape[:2], -1).gather(-1, index)
    return proposal, optimal_gate(residual, proposal)
