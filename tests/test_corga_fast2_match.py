"""CPU equivalence checks for horizontal matching, cached PRUs and model outputs."""

import copy
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.corga.defom_cor_ga_match import DEFOMStereo as MatchStereo
from core.corga.defom_cor_ga_fast2_match import DEFOMStereo as FastMatchStereo
from core.horizontal_sampler import HorizontalCoordinateCache, sample_right_feature_pyramid_horizontal
from core.matching import MatchingTokenEncoder, HorizontalMatchingTokenEncoder
from core.update_pru import MultiPromptUpdateBlock
from core.warp import sample_right_feature_pyramid


class FeatureStub(nn.Module):
    """Replace only the CUDA/xFormers backbone; stereo heads and PRUs stay real."""

    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self.out_dim = 64
        self.depth_anything = SimpleNamespace(depth_feat=SimpleNamespace(
            scratch=SimpleNamespace(refinenet4=SimpleNamespace(resConfUnit1=nn.Identity()))))

    def forward(self, left, right, _sizes):
        left = F.avg_pool2d(left, 4).mean(1, keepdim=True)
        right = F.avg_pool2d(right, 4).mean(1, keepdim=True)
        return left.repeat(1, 64, 1, 1), right.repeat(1, 64, 1, 1), left.sigmoid() * .2


class FastMatchingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        # oneDNN convolution backward stalls on this host's CPU configuration.
        # Use native CPU kernels for correctness checks and restore the backend.
        previous = torch.backends.mkldnn.enabled
        torch.backends.mkldnn.enabled = False
        self.addCleanup(setattr, torch.backends.mkldnn, 'enabled', previous)

    def test_sampler_values_validity_and_gradients(self):
        cache = HorizontalCoordinateCache()
        # Include fractional/out-of-bounds coordinates and a one-column scale.
        for width in (4, 20):
            for ratio in (1, 2, 4):
                right = torch.randn(2, 3, 3, width // ratio)
                disp = torch.rand(2, 1, 3, width) * 4 - .5
                offsets = torch.tensor([-2., 0., 1.])
                results = []
                for horizontal in (False, True):
                    source = right.clone().requires_grad_()
                    disparity = disp.clone().requires_grad_()
                    if horizontal:
                        sampled, valid = sample_right_feature_pyramid_horizontal(
                            (source,), disparity, offsets, cache, (ratio,), return_valid=True)
                    else:
                        sampled, valid = sample_right_feature_pyramid(
                            (source,), disparity, offsets, (ratio,), compact=True, return_valid=True)
                    # Geometrically invalid candidates are excluded by matching.
                    loss = (sampled * valid).square().mean()
                    loss.backward()
                    results.append((sampled.detach() * valid, valid, source.grad, disparity.grad))
                self.assertTrue(torch.equal(results[0][1], results[1][1]))
                for index in (0, 2, 3):
                    torch.testing.assert_close(results[0][index], results[1][index], atol=2e-6, rtol=2e-5)
        legacy = sample_right_feature_pyramid_horizontal((right,), disp, offsets, cache, (ratio,))
        self.assertIsInstance(legacy, torch.Tensor)

    def test_encoder_checkpoint_and_multiscale_equivalence(self):
        for ratios in ((4,), (1, 2, 4)):
            for recompute in (False, True):
                reference = MatchingTokenEncoder(compression_ratios=ratios, scale_encoding=True,
                                                 checkpoint_scales=False)
                fast = HorizontalMatchingTokenEncoder(compression_ratios=ratios, scale_encoding=True,
                                                       checkpoint_scales=recompute)
                fast.load_state_dict(reference.state_dict(), strict=True)
                tensors = [torch.randn(2, 32, 3, 20), torch.randn(2, 32, 3, 20),
                           torch.randn(2, len(ratios), 3, 20), torch.rand(2, 1, 3, 20) * 3]
                results = []
                for encoder in (reference, fast):
                    left, right, logits, disp = [x.clone().requires_grad_() for x in tensors]
                    pyramid = tuple(F.avg_pool2d(right, (1, r)) for r in ratios)
                    out = encoder(left, pyramid, disp, logits.softmax(1))
                    out.square().mean().backward()
                    results.append((out.detach(), [x.grad for x in (left, right, logits, disp)],
                                    [p.grad for p in encoder.parameters()]))
                torch.testing.assert_close(results[0][0], results[1][0], atol=2e-6, rtol=2e-5)
                for expected, actual in zip(results[0][1] + results[0][2], results[1][1] + results[1][2]):
                    torch.testing.assert_close(expected, actual, atol=2e-6, rtol=2e-4)

    def test_cached_update_outputs_and_gradients(self):
        for layers in (1, 3):
            args = SimpleNamespace(n_gru_layers=layers, corr_levels=2, corr_radius=2)
            reference = MultiPromptUpdateBlock(args, 32, 32, 1, 32)
            tensors = [torch.randn(2, 32, 8 // (2 ** i), 16 // (2 ** i)) for i in range(layers)]
            tensors += [torch.randn(2, 10, 8, 16), torch.rand(2, 1, 8, 16),
                        torch.randn(2, 32, 8, 16), torch.randn(2, 32, 8, 16),
                        torch.rand(2, 1, 8, 16)]
            results = []
            for cached, recompute in ((False, False), (True, False), (True, True)):
                block = copy.deepcopy(reference)
                inputs = [x.clone().requires_grad_() for x in tensors]
                if cached:
                    owner = SimpleNamespace(checkpoint_updates=recompute, training=True)
                    static = block.prepare_static(inputs[layers + 2], inputs[-1])
                    outputs = FastMatchStereo._run_cached_update(
                        owner, block, inputs[:layers], *inputs[layers:], static)
                else:
                    outputs = block(inputs[:layers], *inputs[layers:])
                flattened = outputs[0] + list(outputs[1:])
                sum(x.square().mean() for x in flattened).backward()
                results.append(([x.detach() for x in flattened], [x.grad for x in inputs],
                                [p.grad for p in block.parameters()]))
            for actual in results[1:]:
                for expected, obtained in zip(sum(results[0], []), sum(actual, [])):
                    if expected is None:
                        self.assertIsNone(obtained)
                    else:
                        torch.testing.assert_close(expected, obtained, atol=2e-6, rtol=2e-4)

    @patch('core.corga.defom_cor_ga_match.DefomEncoder', FeatureStub)
    def test_low_channel_24_without_checkpoint_backward_and_inference(self):
        args = SimpleNamespace(
            dinov2_encoder='vits', hidden_dims=[32], context_norm='batch',
            n_gru_layers=3, n_downsample=2, corr_levels=2, corr_radius=2,
            hypothesis_scales=[1., 2., 4.], mixed_precision=False,
            max_delta_disp=8., debug_nan=True, match_feature_dim=None,
            disable_activation_checkpoint=True,
        )
        fast = FastMatchStereo(args)
        self.assertEqual(fast.low_channel.out_channels, 24)
        self.assertEqual(fast.refine_match_encoder.token_dim, 28)
        self.assertIsInstance(fast.refine_match_encoder.token_projection, nn.Identity)
        self.assertEqual(fast.refine_match_encoder.window_projection[0].in_channels, 140)
        self.assertEqual(fast.refine_match_encoder.window_projection[0].out_channels, 28)
        self.assertEqual(fast.refine_match_encoder.spatial_fuse[0].in_channels, 52)
        self.assertEqual(fast.refine_match_encoder.spatial_fuse[0].out_channels, 32)
        self.assertEqual(len(fast.refine_match_encoder.spatial_fuse), 2)
        self.assertEqual(fast.refine_rope.heads, 4)
        self.assertEqual(fast.refine_update_block.motion_encoder.convg1.in_channels, 32)
        reference_args = copy.copy(args)
        reference_args.match_feature_dim = 24
        reference_args.match_token_dim = 28
        reference = MatchStereo(reference_args)
        reference.load_state_dict(fast.state_dict(), strict=True)
        left, right = torch.rand(2, 3, 32, 64) * 255, torch.rand(2, 3, 32, 64) * 255
        results = []
        for model in (reference, fast):
            self.assertFalse(model.checkpoint_updates)
            self.assertFalse(model.coarse_match_encoder.checkpoint_scales)
            self.assertFalse(model.refine_match_encoder.checkpoint_scales)
            model.freeze_bn()
            predictions, aux = model(left, right, iters=2, scale_iters=1, return_aux=True)
            all_predictions = predictions + [p for branch in aux['coarse_predictions'] for p in branch]
            sum(p.square().mean() for p in all_predictions).backward()
            for block in (model.low_channel, model.coarse_match_encoder,
                          model.refine_match_encoder, model.refine_update_block):
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in block.parameters()))
            results.append([p.detach() for p in all_predictions])
            model.eval()
        for expected, actual in zip(*results):
            torch.testing.assert_close(expected, actual, atol=2e-5, rtol=2e-4)
        with torch.no_grad():
            torch.testing.assert_close(
                fast(left, right, iters=2, scale_iters=1, test_mode=True),
                reference(left, right, iters=2, scale_iters=1, test_mode=True),
                atol=2e-5, rtol=2e-4)

    @patch('core.corga.defom_cor_ga_match.DefomEncoder', FeatureStub)
    def test_model_weight_compatibility_outputs_backward_and_inference(self):
        for layers in (1, 3):
            args = SimpleNamespace(
                dinov2_encoder='vits', hidden_dims=[32], context_norm='batch',
                match_feature_dim=32,
                match_token_dim=32,
                n_gru_layers=layers, n_downsample=2, corr_levels=2, corr_radius=2,
                hypothesis_scales=[1., 2., 4.], mixed_precision=False,
                max_delta_disp=8., debug_nan=True,
            )
            reference, fast = MatchStereo(args), FastMatchStereo(args)
            fast.load_state_dict(reference.state_dict(), strict=True)
            reference.load_state_dict(fast.state_dict(), strict=True)
            left, right = torch.rand(2, 3, 32, 64) * 255, torch.rand(2, 3, 32, 64) * 255
            results = []
            for model in (reference, fast):
                model.train()
                model.freeze_bn()
                predictions, aux = model(left, right, iters=2, scale_iters=2, return_aux=True)
                self.assertEqual(len(predictions), 3)
                self.assertTrue(all(p.shape == (2, 1, 32, 64) for p in predictions))
                self.assertTrue(all(len(branch) == 3 for branch in aux['coarse_predictions']))
                all_predictions = predictions + [p for branch in aux['coarse_predictions'] for p in branch]
                sum(p.square().mean() for p in all_predictions).backward()
                results.append(([p.detach() for p in all_predictions] + [aux['w_h'], aux['w_s']],
                                {n: p.grad for n, p in model.named_parameters()}))
            for expected, actual in zip(results[0][0], results[1][0]):
                torch.testing.assert_close(expected, actual, atol=2e-5, rtol=2e-4)
            for name, expected in results[0][1].items():
                actual = results[1][1][name]
                if expected is None:
                    self.assertIsNone(actual, name)
                else:
                    self.assertIsNotNone(actual, name)
                    self.assertTrue(torch.isfinite(actual).all(), name)
                    torch.testing.assert_close(expected, actual, atol=2e-4, rtol=2e-3)
            reference.eval()
            fast.eval()
            # Match final predictions while ensuring intermediate masks are skipped.
            with torch.no_grad(), patch.object(fast.coarse_rope, 'prepare_positions',
                                               wraps=fast.coarse_rope.prepare_positions) as positions:
                with patch.object(fast.refine_update_block.mask, 'forward',
                                  wraps=fast.refine_update_block.mask.forward) as mask:
                    with patch.object(fast.coarse_update_blocks[0].mask, 'forward',
                                      wraps=fast.coarse_update_blocks[0].mask.forward) as coarse_mask:
                        actual = fast(left, right, iters=3, scale_iters=2, test_mode=True)
                        expected = reference(left, right, iters=3, scale_iters=2, test_mode=True)
                        torch.testing.assert_close(expected, actual, atol=2e-5, rtol=2e-4)
                        self.assertEqual(mask.call_count, 1)
                        self.assertEqual(coarse_mask.call_count, 0)
                        self.assertEqual(positions.call_count, 1)
            with torch.no_grad():
                for iters, scale_iters in ((0, 0), (0, 2), (2, 0)):
                    expected = reference(left, right, iters=iters, scale_iters=scale_iters, test_mode=True)
                    actual = fast(left, right, iters=iters, scale_iters=scale_iters, test_mode=True)
                    torch.testing.assert_close(expected, actual, atol=2e-5, rtol=2e-4)


selected_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
safe_gpu_selection = all(g.isdecimal() and int(g) != 1 for g in selected_gpus)


@unittest.skipUnless(safe_gpu_selection and torch.cuda.is_available(),
                     'Explicitly select a CUDA GPU other than GPU 1 for the integration test')
class FastMatchingCudaTests(unittest.TestCase):
    def test_real_encoder_bf16_backward_and_inference(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        args = SimpleNamespace(
            dinov2_encoder='vits', hidden_dims=[32], context_norm='batch',
            n_gru_layers=3, n_downsample=2, corr_levels=2, corr_radius=4,
            hypothesis_scales=[4., 8., 16.], mixed_precision=True, amp_dtype='bf16',
            max_delta_disp=8., debug_nan=True,
        )
        model = FastMatchStereo(args).cuda()
        self.assertEqual(model.low_channel.out_channels, 24)
        self.assertIsInstance(model.refine_match_encoder.token_projection, nn.Identity)
        self.assertEqual(model.refine_match_encoder.token_dim, 28)
        self.assertEqual(model.refine_match_encoder.window_projection[0].in_channels, 252)
        self.assertEqual(model.refine_match_encoder.spatial_fuse[0].in_channels, 52)
        self.assertEqual(model.refine_rope.heads, 4)
        self.assertEqual(model.refine_update_block.motion_encoder.convg1.in_channels, 32)
        model.train()
        model.freeze_bn()
        observed_dtypes = {}
        layers = {'hypothesis_fusion': model.hypothesis_weight_head[0]}
        layers.update({f'coarse_pru_{index}': block.motion_encoder.convg1
                       for index, block in enumerate(model.coarse_update_blocks)})
        layers['refine_pru'] = model.refine_update_block.motion_encoder.convg1
        for name, layer in layers.items():
            handle = layer.register_forward_hook(
                lambda _module, _inputs, output, name=name:
                observed_dtypes.setdefault(name, set()).add(output.dtype))
            self.addCleanup(handle.remove)
        left = torch.rand(1, 3, 64, 128, device='cuda') * 255
        right = torch.rand_like(left) * 255
        predictions, aux = model(left, right, iters=2, scale_iters=2, return_aux=True)
        self.assertEqual(len(predictions), 3)
        self.assertTrue(all(p.shape == (1, 1, 64, 128) for p in predictions))
        self.assertTrue(all(len(branch) == 3 for branch in aux['coarse_predictions']))
        for key in ('w_h', 'w_s'):
            torch.testing.assert_close(aux[key].sum(1), torch.ones(1, 16, 32, device='cuda'))
        loss = sum(p.square().mean() for p in predictions)
        loss += sum(p.square().mean() for branch in aux['coarse_predictions'] for p in branch)
        loss.backward()
        for name in layers:
            expected = torch.float32 if name == 'hypothesis_fusion' else torch.bfloat16
            self.assertEqual(observed_dtypes[name], {expected}, name)
        for label in ('coarse_match_encoder', 'refine_match_encoder', 'coarse_rope',
                      'refine_rope', 'coarse_update_blocks', 'hypothesis_weight_head',
                      'refine_state_projections', 'refine_update_block'):
            gradients = [p.grad for p in getattr(model, label).parameters() if p.requires_grad]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients), label)
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0, label)
        model.eval()
        with torch.no_grad():
            for iters, scale_iters in ((2, 2), (0, 0)):
                final = model(left, right, iters=iters, scale_iters=scale_iters, test_mode=True)
                self.assertEqual(final.shape, (1, 1, 64, 128))
                self.assertTrue(torch.isfinite(final).all())


if __name__ == '__main__':
    unittest.main()
