"""Check the two-hypothesis ablation with real stereo heads and PRUs on CPU."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from core.corga.defom_cor_ga_fast2 import DEFOMStereo as Fast2
from core.corga.defom_cor_ga_fast2_two_branch import DEFOMStereo


class FeatureStub(nn.Module):
    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self.out_dim = 64

    def forward(self, left, right, _sizes):
        left = F.avg_pool2d(left, 4).mean(1, keepdim=True)
        right = F.avg_pool2d(right, 4).mean(1, keepdim=True)
        return left.repeat(1, 64, 1, 1), right.repeat(1, 64, 1, 1), left.sigmoid() * .2


class TwoBranchTests(unittest.TestCase):
    @patch('core.corga.defom_cor_ga_fast2_two_branch.DefomEncoder', FeatureStub)
    def test_full_corr_sampling_backward_and_inference(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        previous = torch.backends.mkldnn.enabled
        torch.backends.mkldnn.enabled = False
        self.addCleanup(setattr, torch.backends.mkldnn, 'enabled', previous)
        args = SimpleNamespace(
            dinov2_encoder='vits', hidden_dims=[32], context_norm='batch',
            n_gru_layers=3, n_downsample=2, corr_levels=3, corr_radius=2,
            hypothesis_scales=[24., 48.], mixed_precision=False, max_delta_disp=8.,
        )
        model = DEFOMStereo(args)
        model.freeze_bn()
        self.assertEqual(len(model.coarse_update_blocks), 2)
        self.assertFalse(hasattr(model, 'coarse_right_fuse'))
        self.assertFalse(any('compress_4' in n for n, _ in model.named_parameters()))
        for block in model.coarse_update_blocks:
            self.assertEqual(block.num_layers, 3)
            self.assertTrue(block.motion_encoder.use_corr)
            self.assertFalse(block.motion_encoder.use_warp_features)
            self.assertFalse(hasattr(block.motion_encoder, 'convg1'))
        # The complete FullCorr pyramid and sampled values match original fast2.
        left, right = torch.randn(2, 64, 8, 16), torch.randn(2, 64, 8, 16)
        actual = model._build_full_corr(left, right)
        expected = Fast2._build_full_corr(model, left, right)
        disp = torch.rand(2, 1, 8, 16) * 8
        self.assertEqual(len(actual.volume_pyramid), 3)
        torch.testing.assert_close(actual(disp), expected(disp))
        torch.testing.assert_close(actual.sample_many([disp, disp * 2]), expected.sample_many([disp, disp * 2]))
        left, right = torch.rand(2, 3, 32, 64) * 255, torch.rand(2, 3, 32, 64) * 255
        with patch.object(model, '_lookup_right_features', wraps=model._lookup_right_features) as lookup:
            predictions, aux = model(left, right, iters=2, scale_iters=2, return_aux=True)
            self.assertEqual(lookup.call_count, 2)  # Only refinement samples right features.
            for call in lookup.call_args_list:
                self.assertEqual(call.kwargs['compression_ratios'], (1, 2))
                self.assertEqual(len(call.args[1]), 2)
        self.assertEqual(len(aux['coarse_predictions']), 2)
        self.assertTrue(all(len(branch) == 3 for branch in aux['coarse_predictions']))
        self.assertEqual(aux['w_h'].shape, (2, 2, 8, 16))
        self.assertEqual(aux['w_s'].shape, (2, 2, 8, 16))
        all_predictions = predictions + [p for branch in aux['coarse_predictions'] for p in branch]
        sum(p.square().mean() for p in all_predictions).backward()
        for name, param in model.named_parameters():
            if name.startswith(('fnet.', 'cnet.')):
                continue  # Existing encoder modules contain unused normalization parameters.
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        model.eval()
        with torch.no_grad():
            training_output = model(left, right, iters=2, scale_iters=2)[-1]
            inference_output = model(left, right, iters=2, scale_iters=2, test_mode=True)
            torch.testing.assert_close(inference_output, training_output)
            initial_output = model(left, right, iters=0, scale_iters=0, test_mode=True)
            self.assertEqual(initial_output.shape, (2, 1, 32, 64))
            self.assertTrue(torch.isfinite(initial_output).all())


if __name__ == '__main__':
    unittest.main()
