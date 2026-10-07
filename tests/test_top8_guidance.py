"""Coordinate, gradient, and baseline-preservation checks for global evidence."""
import unittest

import torch

from core.bin.top8_guidance import CoarseTop8Guidance


class Top8GuidanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        self.module = CoarseTop8Guidance(feature_dim=2, token_dim=4)
        self.left = torch.randn(1, 2, 2, 64, requires_grad=True)
        self.right = torch.arange(16.).view(1, 1, 1, 16).expand(1, 2, 2, 16).clone().requires_grad_()
        self.bins = torch.arange(48.) * 4
        logits = torch.zeros(1, 48, 2, 64)
        logits[:, 4] = 10
        logits[:, 8] = 8
        self.logits = logits.requires_grad_()
        x = torch.arange(64.)[None, None, None]
        xr = (x-self.bins[None, :, None, None]-1.5)/4
        self.valid = ((xr >= 0) & (xr <= 15)).expand_as(logits)

    def build(self, valid=None):
        return self.module.build(self.left, self.right, self.logits,
                                 self.valid if valid is None else valid, self.bins)

    def test_pooling_center_and_disparity_units_match_initializer(self):
        seen = []
        handle = self.module.token_encoder[0].register_forward_pre_hook(
            lambda module, inputs: seen.append(inputs[0].detach()))
        evidence = self.build()
        handle.remove()
        self.assertEqual(evidence['disparities'][0, 0, 0, 40].item(), 16)
        # Right feature channel encodes its x coordinate: (40-16-1.5)/4.
        torch.testing.assert_close(seen[0][0, 2, 0, 40], torch.tensor(5.625))
        self.assertLess(evidence['valid'][0, :, 0, 40].sum().item(), 8)

    def test_zero_initialized_residual_preserves_local_features(self):
        local = torch.randn_like(self.left)
        output = self.module(self.build(), self.left, local, torch.zeros(1, 1, 2, 64))
        torch.testing.assert_close(output, local, rtol=0, atol=0)
        output.sum().backward()
        self.assertGreater(self.module.residual[-1].weight.grad.abs().sum().item(), 0)

    def test_nonzero_head_backpropagates_to_matching_features_and_scores(self):
        torch.nn.init.normal_(self.module.residual[-1].weight, std=.1)
        evidence = self.build()
        disp = torch.full((1, 1, 2, 64), 12.)
        output = self.module(evidence, self.left, torch.randn_like(self.left), disp)
        output.square().sum().backward()
        for tensor in (self.left, self.right, self.logits):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        self.assertFalse(evidence['disparities'].requires_grad)
        self.assertGreater(self.module.relative_position.weight.grad.abs().sum().item(), 0)

    def test_no_valid_candidates_cannot_modify_local_features(self):
        torch.nn.init.normal_(self.module.residual[-1].weight)
        torch.nn.init.constant_(self.module.residual[-1].bias, 3.)
        local = torch.randn_like(self.left)
        evidence = self.build(torch.zeros_like(self.valid))
        output = self.module(evidence, self.left, local, torch.zeros(1, 1, 2, 64))
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output, local, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
