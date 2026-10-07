"""End-to-end CUDA smoke test with the real (random-weight) ViT-S encoder."""

import os
import copy
from types import SimpleNamespace
import unittest

import torch

from core.corga.defom_cor_ga_match import DEFOMStereo
from core.update_pru import MultiPromptUpdateBlock


selected_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
safe_gpu_selection = all(g.isdecimal() and int(g) != 1 for g in selected_gpus)


class RecurrentCheckpointTests(unittest.TestCase):
    def test_update_checkpoint_preserves_outputs_and_gradients(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        args = SimpleNamespace(n_gru_layers=3, corr_levels=2, corr_radius=4)
        reference = MultiPromptUpdateBlock(args, 32, 32, 1, 32)
        recomputed = copy.deepcopy(reference)
        tensors = [torch.randn(1, 32, 8 // (2 ** i), 16 // (2 ** i)) for i in range(3)]
        tensors += [torch.randn(1, 18, 8, 16), torch.rand(1, 1, 8, 16),
                    torch.randn(1, 32, 8, 16), torch.randn(1, 32, 8, 16),
                    torch.rand(1, 1, 8, 16)]
        results = []
        for block, enabled in ((reference, False), (recomputed, True)):
            inputs = [x.clone().requires_grad_() for x in tensors]
            owner = SimpleNamespace(checkpoint_updates=enabled, training=True)
            states, delta, mask = DEFOMStereo._run_update(
                owner, block, inputs[:3], *inputs[3:])
            outputs = states + [delta, mask]
            sum(x.square().mean() for x in outputs).backward()
            results.append(([x.detach() for x in outputs], [x.grad for x in inputs],
                            [p.grad for p in block.parameters()]))
        for expected, actual in zip(sum(results[0], []), sum(results[1], [])):
            if expected is None:
                self.assertIsNone(actual)
            else:
                torch.testing.assert_close(expected, actual)


@unittest.skipUnless(safe_gpu_selection and torch.cuda.is_available(),
                     'Explicitly select a CUDA GPU other than GPU 1 for the integration test')
class CorGAMatchTests(unittest.TestCase):
    def test_forward_backward_and_inference_contract(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        args = SimpleNamespace(
            dinov2_encoder='vits', hidden_dims=[32], context_norm='batch',
            n_gru_layers=3, n_downsample=2, corr_levels=2, corr_radius=4,
            hypothesis_scales=[4, 8, 16], mixed_precision=True, amp_dtype='bf16',
            max_delta_disp=8., debug_nan=True,
        )
        # The real encoder uses the repository's CUDA/xFormers execution path.
        with torch.cuda.device(0):
            model = DEFOMStereo(args).cuda()
            self.assertEqual(DEFOMStereo.__bases__, (torch.nn.Module,))
            model.train()
            model.freeze_bn()
            left = torch.rand(1, 3, 64, 128, device='cuda') * 255
            right = torch.rand_like(left) * 255
            predictions, aux = model(left, right, iters=2, scale_iters=1, return_aux=True)
            self.assertEqual(len(predictions), 3)
            self.assertTrue(all(tuple(p.shape) == (1, 1, 64, 128) for p in predictions))
            self.assertTrue(all(len(p) == 2 for p in aux['coarse_predictions']))
            for key in ('w_h', 'w_s'):
                torch.testing.assert_close(aux[key].sum(1), torch.ones(1, 16, 32, device='cuda'))
            loss = sum(p.abs().mean() for p in predictions)
            loss += sum(p.abs().mean() for branch in aux['coarse_predictions'] for p in branch)
            loss.backward()
            for label in ('coarse_match_encoder', 'refine_match_encoder',
                          'hypothesis_weight_head', 'refine_update_block'):
                gradients = [p.grad for p in getattr(model, label).parameters() if p.requires_grad]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients), label)
                self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0, label)
            model.eval()
            with torch.no_grad():
                final = model(left, right, iters=2, scale_iters=1, test_mode=True)
                initial = model(left, right, iters=0, scale_iters=0, test_mode=True)
            self.assertEqual(tuple(final.shape), (1, 1, 64, 128))
            self.assertEqual(final.shape, initial.shape)
            self.assertTrue(torch.isfinite(final).all() and torch.isfinite(initial).all())


if __name__ == '__main__':
    unittest.main()
