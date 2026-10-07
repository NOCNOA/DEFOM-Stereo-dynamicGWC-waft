"""Haar evidence and local-sampling interaction checks; CPU only."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from core.bin.defom_cor_ga_bins import DEFOMStereo as BinStereo
from core.bin.defom_cor_ga_bins_wavelet import DEFOMStereo as WaveletBinStereo
from core.bin.wavelet_guidance import WaveletBinGuidance
from core.submodules import lookup_right_features


class WaveletBinGuidanceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(1)

    def test_full_detail_set_reconstructs_all_bins(self):
        guidance = WaveletBinGuidance(num_bins=16, num_details=14, num_peaks=2)
        logits = torch.randn(1, 3, 16, 2, 3)
        valid = torch.ones_like(logits, dtype=torch.bool)
        valid[:, 1, :, :, 0] = False
        evidence = guidance.compress(logits, valid)
        indices = torch.arange(16).view(1, 1, 16, 1, 1).expand(1, 3, 16, 2, 3)
        reconstructed = guidance._reconstruct_bins(evidence, indices)
        expected = torch.softmax(logits, dim=2) * valid.float()
        expected = expected / expected.sum(dim=2, keepdim=True).clamp_min(1e-6)
        torch.testing.assert_close(reconstructed, expected, atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.equal(reconstructed[:, 1, :, :, 0],
                                    torch.zeros_like(reconstructed[:, 1, :, :, 0])))

    def test_query_uses_actual_disparity_of_each_scale_and_masks_edges(self):
        guidance = WaveletBinGuidance(num_bins=16, bin_width=1)
        for width in (48, 64):
            logits = torch.full((1, 3, 16, 1, width), -15.0)
            # At d_t=8 and warp offset=-1 the sampled disparities are 9, 10, 12.
            for scale, bin_index in enumerate((9, 5, 3)):
                logits[:, scale, bin_index] = 15.0
            evidence = guidance.compress(logits, torch.ones_like(logits, dtype=torch.bool))
            disp = torch.full((1, 1, 1, width), 8.0)
            prior = guidance.query(evidence, disp, torch.tensor([-1., 0., 1.]),
                                   (width, width // 2, width // 4))
            self.assertEqual(tuple(prior.shape), (1, 3, 3, 1, width))
            self.assertTrue(torch.equal(prior[0, :, :, 0, 40].argmax(dim=1),
                                        torch.zeros(3, dtype=torch.long)))
            self.assertTrue(torch.all(prior[0, :, 0, 0, 40] > 0.99))
            self.assertTrue(torch.equal(prior[0, :, :, 0, 0],
                                        torch.zeros_like(prior[0, :, :, 0, 0])))

    def test_zero_initialization_and_gradient_to_bin_evidence(self):
        guidance = WaveletBinGuidance(num_bins=16, bin_width=1, feature_dim=8).configure(3)
        logits = torch.randn(1, 3, 16, 2, 48, requires_grad=True)
        evidence = guidance.compress(logits, torch.ones_like(logits, dtype=torch.bool))
        left = torch.randn(1, 8, 2, 48)
        sampled = torch.randn(1, 3, 3, 8, 2, 48, requires_grad=True)
        disp = torch.full((1, 1, 2, 48), 8.0)
        scale_weights = torch.softmax(torch.randn(1, 3, 2, 48), dim=1)
        args = (evidence, left, sampled, disp, torch.tensor([-1., 0., 1.]),
                scale_weights, (48, 24, 12))
        guided, residual = guidance.condition_samples(*args)
        torch.testing.assert_close(guided, sampled)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

        with torch.no_grad():
            guidance.conditioner[-1].weight.normal_(std=0.01)
        guided, residual = guidance.condition_samples(*args)
        (guided.square().mean() + residual.square().mean()).backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(sampled.grad).all())

    def test_unguided_lookup_path_is_identical(self):
        left = torch.randn(1, 2, 2, 16)
        right = torch.randn_like(left)
        pyramid = (right, right[..., ::2], right[..., ::4])
        disp = torch.full((1, 1, 2, 16), 4.0)
        scale_weights = torch.softmax(torch.randn(1, 3, 2, 16), dim=1)
        options = dict(warp_offsets=torch.tensor([-1., 0., 1.]),
                       feature_fuse=nn.Identity(), scale_weights=scale_weights)
        baseline = lookup_right_features(left, pyramid, disp, **options)
        identity = lookup_right_features(
            left, pyramid, disp, sample_conditioner=lambda _, samples: (samples, None),
            **options)
        torch.testing.assert_close(baseline, identity, rtol=0, atol=0)

    def test_bin_model_uses_guidance_each_iteration(self):
        class FeatureStub(nn.Module):
            def __init__(self, *_args, **_kwargs):
                super().__init__()
                self.out_dim = 64

            def forward(self, left, right, _sizes):
                left_low = F.avg_pool2d(left, 4).mean(1, keepdim=True)
                right_low = F.avg_pool2d(right, 4).mean(1, keepdim=True)
                return (left_low.repeat(1, 64, 1, 1),
                        right_low.repeat(1, 64, 1, 1), left_low.sigmoid() * .2)

        args = SimpleNamespace(
            dinov2_encoder='vits', hidden_dims=[32], context_norm='batch',
            n_gru_layers=3, n_downsample=2, corr_levels=2, corr_radius=2,
            mixed_precision=False, max_delta_disp=8., num_init_bins=16,
            init_max_offset=16.,
        )
        with patch('core.bin.defom_cor_ga_bins.DefomEncoder', FeatureStub), \
                patch.object(BinStereo, '_freeze_structurally_unused_parameters'):
            model = WaveletBinStereo(args).eval()
            baseline = BinStereo(args).eval()
        loaded = model.load_state_dict(baseline.state_dict(), strict=False)
        self.assertTrue(all(key.startswith('wavelet_guidance.') for key in loaded.missing_keys))
        self.assertFalse(loaded.unexpected_keys)
        left, right = torch.rand(1, 3, 32, 64) * 255, torch.rand(1, 3, 32, 64) * 255
        with patch.object(model.wavelet_guidance, 'condition_samples',
                          wraps=model.wavelet_guidance.condition_samples) as condition:
            predictions, aux = model(left, right, iters=2, return_aux=True)
        self.assertEqual(condition.call_count, 2)
        self.assertEqual(len(predictions), 3)
        self.assertEqual(tuple(predictions[-1].shape), (1, 1, 32, 64))
        self.assertEqual(tuple(aux['init_logits'].shape), (1, 3, 16, 8, 16))
        with torch.no_grad():
            baseline_predictions = baseline(left, right, iters=2)
        for guided, original in zip(predictions, baseline_predictions):
            torch.testing.assert_close(guided, original, atol=1e-5, rtol=1e-4)
        predictions[-1].square().mean().backward()
        gradient = model.wavelet_guidance.conditioner[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
