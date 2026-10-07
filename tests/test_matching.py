"""Geometric, matching and gradient checks; no dataset or GPU required."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from core.matching import CandidateResidualBlock, MatchingTokenEncoder
from core.warp import sample_right_feature_pyramid


class MatchingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_sampler_coordinates_validity_and_legacy_return(self):
        # A compressed ramp stores the original-coordinate receptive-field
        # centers. Linear interpolation should recover x_left - d + ratio*k.
        for ratio in (1, 2, 4):
            centers = torch.arange(16 // ratio).float() * ratio + (ratio - 1) / 2
            right = centers.view(1, 1, 1, -1)
            disp = torch.full((1, 1, 1, 16), 2.)
            offsets = torch.tensor([-1., 0., 1.])
            sampled, valid = sample_right_feature_pyramid(
                (right,), disp, offsets, (ratio,), compact=True, return_valid=True)
            expected = torch.arange(16).view(1, 1, 1, 1, 16) - 2 + ratio * offsets.view(1, 3, 1, 1, 1)
            expected_valid = (expected >= centers[0]) & (expected <= centers[-1])
            self.assertTrue(torch.equal(valid, expected_valid))
            torch.testing.assert_close(sampled[valid], expected.float()[valid])
            legacy = sample_right_feature_pyramid((right,), disp, offsets, (ratio,))
            self.assertIsInstance(legacy, torch.Tensor)
            torch.testing.assert_close(sampled, legacy)

    def test_known_stereo_shift_without_offset_or_mask_channels(self):
        right = torch.randn(1, 32, 1, 16)
        # Ground-truth disparity is 4, current estimate is 3. The exact match
        # is k=-1: positive disparity correction +1, not -1. Candidate order
        # still identifies the correction without a separate offset channel.
        left = F.pad(right[..., :-4], (4, 0))
        disp = torch.full((1, 1, 1, 16), 3.)
        encoder = MatchingTokenEncoder(compression_ratios=(1,))
        sampled, valid = sample_right_feature_pyramid(
            (right,), disp, encoder.offsets, (1,), return_valid=True)
        evidence, _ = encoder._build_evidence(left, sampled, disp, valid, 1)
        scores = evidence[0, :, :4, 0, 8].mean(dim=1)
        self.assertEqual(scores.argmax().item(), 3)
        torch.testing.assert_close(evidence[0, 3, :4, 0, 8], torch.ones(4))
        torch.testing.assert_close(evidence[0, 3, 4:36, 0, 8], torch.zeros(32))
        self.assertEqual(tuple(evidence.shape), (1, 9, 36, 1, 16))

    def test_all_invalid_and_negative_disparity_candidates(self):
        left, right = torch.randn(1, 32, 2, 16), torch.randn(1, 32, 2, 16)
        encoder = MatchingTokenEncoder(compression_ratios=(1,))
        disp = torch.full((1, 1, 2, 16), 100.)
        feature = encoder._encode_scale(left, right, disp, 1)
        self.assertTrue(torch.equal(feature, torch.zeros_like(feature)))
        disp.zero_()
        sampled, valid = sample_right_feature_pyramid(
            (right,), disp, encoder.offsets, (1,), return_valid=True)
        _, valid = encoder._build_evidence(left, sampled, disp, valid, 1)
        self.assertFalse(valid[:, 5:].any())  # k>0 implies negative disparity.

    def test_candidate_mask_and_independent_pixel_sequences(self):
        block = CandidateResidualBlock(4)
        tokens = torch.randn(2, 4, 9)
        mask = torch.ones(2, 1, 9)
        mask[:, :, 4] = 0
        original = block(tokens * mask, mask)
        self.assertTrue(torch.equal(original[:, :, 4], torch.zeros(2, 4)))
        changed = tokens.clone()
        changed[1] += 100
        torch.testing.assert_close(original[0], block(changed * mask, mask)[0])

    def test_multiscale_gradients_and_ablation_options(self):
        for groups, layers, scale_encoding in ((4, 1, False), (8, 0, True)):
            encoder = MatchingTokenEncoder(groups=groups, aggregation_layers=layers,
                                           scale_encoding=scale_encoding)
            left = torch.randn(1, 32, 3, 16, requires_grad=True)
            right = torch.randn(1, 32, 3, 16, requires_grad=True)
            logits = torch.randn(1, 3, 3, 16, requires_grad=True)
            pyramid = (right, F.avg_pool2d(right, (1, 2)), F.avg_pool2d(right, (1, 4)))
            output = encoder(left, pyramid, torch.full((1, 1, 3, 16), 2.), logits.softmax(1))
            self.assertEqual(tuple(output.shape), (1, 32, 3, 16))
            output.square().mean().backward()
            for tensor in (left, right, logits):
                self.assertTrue(torch.isfinite(tensor.grad).all())
                self.assertGreater(tensor.grad.abs().sum().item(), 0)
            for parameter in encoder.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_zero_weight_removes_a_scale(self):
        encoder = MatchingTokenEncoder()
        left = torch.randn(1, 32, 2, 16)
        right = torch.randn_like(left)
        pyramid = (right, F.avg_pool2d(right, (1, 2)), F.avg_pool2d(right, (1, 4)))
        weights = torch.zeros(1, 3, 2, 16)
        weights[:, 0] = 1
        disp = torch.ones(1, 1, 2, 16)
        first = encoder(left, pyramid, disp, weights)
        changed = (right, pyramid[1] * -100, pyramid[2] * 100)
        torch.testing.assert_close(first, encoder(left, changed, disp, weights))

    def test_right_features_are_summed_before_matching_once(self):
        encoder = MatchingTokenEncoder(checkpoint_scales=False).eval()
        vectors = torch.randn(3, 32)
        weights = torch.tensor([.25, .25, .5])
        # The left feature exactly matches the mixture, but not any individual
        # scale. Its evidence must be perfect only after the right-side sum.
        mixture = (vectors * weights[:, None]).sum(0)
        left = mixture.view(1, 32, 1, 1).expand(1, 32, 2, 40)
        pyramid = tuple(vector.view(1, 32, 1, 1).expand(1, 32, 2, 40 // ratio)
                        for vector, ratio in zip(vectors, (1, 2, 4)))
        weights = weights.view(1, 3, 1, 1).expand(1, 3, 2, 40)
        evidence_inputs = []
        handle = encoder.token_projection.register_forward_pre_hook(
            lambda _module, inputs: evidence_inputs.append(inputs[0].detach()))
        try:
            with torch.no_grad(), patch.object(encoder, '_encode_candidates',
                                              wraps=encoder._encode_candidates) as encode:
                output = encoder(left, pyramid, torch.full((1, 1, 2, 40), 8.), weights)
                self.assertEqual(encode.call_count, 1)
        finally:
            handle.remove()
        self.assertEqual(len(evidence_inputs), 1)
        torch.testing.assert_close(evidence_inputs[0][4, :4, 0, 20], torch.ones(4))
        torch.testing.assert_close(evidence_inputs[0][4, 4:, 0, 20], torch.zeros(32),
                                   atol=1e-6, rtol=0)
        self.assertEqual(tuple(output.shape), (1, 32, 2, 40))

    def test_one_hot_scale_matches_single_scale_including_boundaries(self):
        encoder = MatchingTokenEncoder(checkpoint_scales=False, scale_encoding=True).eval()
        left, right = torch.randn(1, 32, 2, 16), torch.randn(1, 32, 2, 16)
        pyramid = tuple(F.avg_pool2d(right, (1, ratio)) for ratio in (1, 2, 4))
        disp = torch.zeros(1, 1, 2, 16)
        with torch.no_grad():
            for index, ratio in enumerate((1, 2, 4)):
                single = MatchingTokenEncoder(compression_ratios=(ratio,),
                                              checkpoint_scales=False, scale_encoding=True).eval()
                single.load_state_dict(encoder.state_dict(), strict=True)
                weights = torch.zeros(1, 3, 2, 16)
                weights[:, index] = 1
                torch.testing.assert_close(encoder(left, pyramid, disp, weights),
                                           single(left, (pyramid[index],), disp))
            # No candidates contribute when every scale is out of bounds.
            feature = encoder._encode_pyramid(left, disp + 100, weights, *pyramid)
            self.assertTrue(torch.equal(feature, torch.zeros_like(feature)))

    def test_checkpoint_preserves_outputs_and_gradients(self):
        reference = MatchingTokenEncoder(checkpoint_scales=False)
        recomputed = MatchingTokenEncoder(checkpoint_scales=True)
        recomputed.load_state_dict(reference.state_dict())
        inputs = [torch.randn(1, 32, 2, 16), torch.randn(1, 32, 2, 16),
                  torch.randn(1, 3, 2, 16)]
        results = []
        for encoder in (reference, recomputed):
            left, right, logits = [x.clone().requires_grad_() for x in inputs]
            pyramid = (right, F.avg_pool2d(right, (1, 2)), F.avg_pool2d(right, (1, 4)))
            output = encoder(left, pyramid, torch.full((1, 1, 2, 16), 2.), logits.softmax(1))
            output.square().mean().backward()
            results.append((output.detach(), [x.grad for x in (left, right, logits)],
                            [p.grad for p in encoder.parameters()]))
        torch.testing.assert_close(results[0][0], results[1][0])
        for expected, actual in zip(results[0][1] + results[0][2],
                                    results[1][1] + results[1][2]):
            torch.testing.assert_close(expected, actual)


if __name__ == '__main__':
    unittest.main()
