import unittest

import torch

from core.bin.top8_sample_guidance import SampleTop8Guidance
from core.update_pru import MotionEncoder


class SampleGuidanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)
        self.head = SampleTop8Guidance(feature_dim=2, token_dim=4, query_dim=4, samples_per_scale=3)
        self.left = torch.randn(1, 2, 2, 16, requires_grad=True)
        self.samples = torch.randn(1, 3, 3, 2, 2, 16, requires_grad=True)
        self.evidence = dict(tokens=torch.randn(1, 8, 4, 2, 16, requires_grad=True),
                             disparities=torch.arange(8.)[None, :, None, None].expand(1, 8, 2, 16),
                             valid=torch.ones(1, 8, 2, 16, dtype=torch.bool))

    def forward_head(self):
        return self.head.condition_samples(self.evidence, self.left, self.samples,
                    torch.ones(1, 1, 2, 16), torch.tensor([-1., 0., 1.]), (16, 8, 4))

    def test_queries_remain_distinct_and_receive_gradients(self):
        seen = []
        hook = self.head.sample_ffn[0].register_forward_pre_hook(
            lambda module, inputs: seen.append(inputs[0].detach()))
        feature = self.forward_head()
        hook.remove()
        # Every scale/offset has its own global readout, before spatial fusion.
        self.assertEqual(seen[0].shape, (9, 8, 2, 16))
        self.assertFalse(torch.allclose(seen[0][0, 4:, :, 8], seen[0][1, 4:, :, 8]))
        feature.square().sum().backward()
        for tensor in (self.left, self.samples, self.evidence['tokens']):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_invalid_global_evidence_is_zero(self):
        self.evidence['valid'].zero_()
        feature = self.forward_head()
        self.assertTrue(torch.isfinite(feature).all())
        torch.testing.assert_close(feature, torch.zeros_like(feature), rtol=0, atol=0)

    def test_sample_disparity_uses_scale_and_right_offset_sign(self):
        seen = []
        hook = self.head.sample_query[0].register_forward_pre_hook(
            lambda module, inputs: seen.append(inputs[0].detach()))
        self.forward_head()
        hook.remove()
        # At d=1, offset=-1 gives d_sample=2 at ratio 1, and 5 at ratio 4.
        torch.testing.assert_close(seen[0][0, 7, 0, 8], torch.tensor(2./192))
        torch.testing.assert_close(seen[0][6, 7, 0, 8], torch.tensor(5./192))
        torch.testing.assert_close(seen[0][8, 7, 0, 8], torch.tensor(-3./192))

    def test_motion_projection_preserves_baseline_and_starts_learning(self):
        motion = MotionEncoder(hidden_dim=16, corplane=3, warp_feat_dim=2)
        motion.global_convg1 = torch.nn.Conv2d(2, 8, 1)
        torch.nn.init.zeros_(motion.global_convg1.weight)
        torch.nn.init.zeros_(motion.global_convg1.bias)
        corr, disp, local = torch.randn(1, 3, 2, 16), torch.ones(1, 1, 2, 16), self.left.detach()
        baseline = motion(corr, disp, local)
        guided = motion(corr, disp, local, global_warp_feat=self.forward_head())
        torch.testing.assert_close(baseline, guided, rtol=0, atol=0)
        guided.square().sum().backward()
        self.assertGreater(motion.global_convg1.weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
