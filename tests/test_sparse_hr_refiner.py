import unittest

import torch
from core.corga.sparse_hr_diagnostics import optimal_candidate, optimal_gate

from core.corga.sparse_hr_refiner import (
    SparseHRRefiner, candidate_gate_loss, candidate_selection_loss, dilate, disparity_boundary, sample_map,
    sample_training_queries, scatter_queries, sparse_direction_loss,
    sparse_sequence_loss,
    update_quality_totals, summarize_update_quality,
)


class SparseHRRefinerTest(unittest.TestCase):
    def test_oracle_candidates_recover_subpixel_residuals_and_respect_radius(self):
        residual = torch.tensor([[[-12.], [-1.2], [-.2], [0.], [.2], [1.2], [12.]]])
        proposal, gate = optimal_candidate(residual, torch.arange(-4., 5.), 2.)
        self.assertTrue(torch.allclose(proposal * gate, residual.clamp(-2., 2.)))
        self.assertEqual(float(proposal[0, 3]), 0.)
        self.assertLessEqual(float(proposal.abs().max()), 2.)
        self.assertTrue(torch.isfinite(gate).all())

    def test_oracle_gate_rejects_wrong_direction_and_zero_proposal(self):
        residual = torch.tensor([[[.5], [-.5], [2.]]])
        proposal = torch.tensor([[[2.], [2.], [0.]]])
        gate = optimal_gate(residual, proposal)
        self.assertTrue(torch.allclose(gate, torch.tensor([[[.25], [0.], [0.]]])))

    def test_update_quality_separates_direction_from_error_improvement(self):
        base = torch.tensor([10., 10., 10., 10., 10.])
        gt = torch.tensor([10., 12., 8., 10., 12.])
        prediction = torch.tensor([10., 11., 9., 11., 15.])
        value = summarize_update_quality(update_quality_totals(
            base, prediction, gt, torch.ones_like(base, dtype=torch.bool)))
        self.assertEqual(value["update_coverage_percent"], 80.)
        self.assertEqual(value["update_improvement_percent"], 50.)
        self.assertEqual(value["direction_accuracy_updated_percent"], 75.)
        self.assertEqual(value["net_error_reduction_sum"], 0.)
        empty = summarize_update_quality(update_quality_totals(
            base, prediction, gt, torch.zeros_like(base, dtype=torch.bool)))
        self.assertIsNone(empty["update_improvement_percent"])

    def test_candidate_supervision_masks_unreachable_and_out_of_bounds(self):
        gt = torch.full((1, 1, 1, 24), 10.)
        gt[0, 0, 0, 20] = 10.5
        gt[0, 0, 0, 21] = 12.
        gt[0, 0, 0, 22] = 20.
        gt[0, 0, 0, 3] = 12.
        query = torch.tensor([[[20., 0.], [21., 0.], [22., 0.], [3., 0.]]])
        current = torch.full((1, 4, 1), 10.)
        scores = torch.zeros(1, 4, 9, requires_grad=True)
        info = [(current * 0, current * 0, scores)]
        loss = candidate_selection_loss(info, [current, current], gt,
                                        torch.ones_like(gt, dtype=torch.bool), query,
                                        torch.arange(-4., 5.))
        loss.backward()
        self.assertLess(float(scores.grad[0, 0, 4]), 0.)
        self.assertLess(float(scores.grad[0, 1, 6]), 0.)
        self.assertEqual(float(scores.grad[0, 2:].abs().sum()), 0.)

    def test_context_score_starts_at_zero_and_trains(self):
        model = SparseHRRefiner(coarse_hidden_channels=8, coarse_context_channels=8,
                                iterations=1, candidate_guided=True, context_scoring=True)
        left = torch.rand(1, 3, 32, 48) * 255
        full = torch.full((1, 1, 32, 48), 10.)
        query = torch.tensor([[[20., 10.], [25., 12.]]])
        args = (left, left, full, full[:, :, ::4, ::4],
                torch.randn(1, 8, 8, 12), torch.randn(1, 8, 8, 12), query)
        predictions, info = model(*args, return_candidate_info=True)
        model.context_scoring = False
        _, plain_info = model(*args, return_candidate_info=True)
        self.assertTrue(torch.allclose(info[0][2], plain_info[0][2]))
        gt = full + 2.
        loss = candidate_selection_loss(info, predictions, gt,
                                        torch.ones_like(gt, dtype=torch.bool), query,
                                        model.offsets)
        loss.backward()
        grad = model.context_candidate_scorer[-1].weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.)

    def test_expanded_hidden_and_context_receive_gradient(self):
        model = SparseHRRefiner(coarse_hidden_channels=8, coarse_context_channels=8,
                                hidden_channels=128, context_channels=128,
                                iterations=1, candidate_guided=True)
        left = torch.rand(1, 3, 32, 48) * 255
        full = torch.full((1, 1, 32, 48), 10.)
        low = torch.full((1, 1, 8, 12), 10.)
        query = torch.tensor([[[15., 10.], [20., 12.]]])
        with torch.no_grad():
            for parameter in model.candidate_scorer.parameters():
                parameter.zero_()
        predictions = model(left, left, full, low, torch.randn(1, 8, 8, 12),
                            torch.randn(1, 8, 8, 12), query)
        predictions[-1].sum().backward()
        for layer in (model.hidden_transfer[0], model.coarse_context_encoder[0],
                      model.fine_context_encoder[0], model.context_fusion[0]):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(float(layer.weight.grad.abs().sum()), 0.)

    def test_ratio_gate_targets_partial_update_and_masks_zero_proposal(self):
        gt = torch.tensor([[[[10.5, 9., 10.]]]])
        valid = torch.ones_like(gt, dtype=torch.bool)
        query = torch.tensor([[[0., 0.], [1., 0.], [2., 0.]]])
        current = torch.full((1, 3, 1), 10.)
        proposal = torch.tensor([[[2.], [2.], [0.]]])
        logits = torch.zeros_like(proposal, requires_grad=True)
        loss = candidate_gate_loss([(proposal, logits)], [current, current],
                                   gt, valid, query, target_mode="ratio")
        loss.backward()
        self.assertAlmostEqual(float(loss), .078125)
        self.assertTrue((logits.grad[0, :2] > 0).all())
        self.assertEqual(float(logits.grad[0, 2]), 0.)

    def test_sampling_uses_original_pixel_centers(self):
        horizontal = torch.arange(8.).view(1, 1, 1, 8).expand(1, 1, 6, 8)
        xy = torch.tensor([[[0., 0.], [3., 2.], [7., 5.]]])
        actual = sample_map(horizontal, xy, (6, 8))
        self.assertTrue(torch.allclose(actual[0, :, 0], xy[0, :, 0]))

    def test_queries_respect_valid_mask(self):
        gt = torch.ones(2, 1, 16, 16) * 20
        gt[:, :, :, 8:] = 40
        valid = torch.ones_like(gt, dtype=torch.bool)
        valid[:, :, :2] = False
        self.assertGreater(int(disparity_boundary(gt, valid).sum()), 0)
        query = sample_training_queries(gt, valid, n=128)
        self.assertEqual(tuple(query.shape), (2, 128, 2))
        for batch in range(2):
            self.assertTrue(valid[batch, 0, query[batch, :, 1].long(),
                                  query[batch, :, 0].long()].all())

    def test_boundary_only_queries_stay_in_requested_band(self):
        gt = torch.ones(1, 1, 16, 16) * 20
        gt[:, :, :, 8:] = 40
        valid = torch.ones_like(gt, dtype=torch.bool)
        band = dilate(disparity_boundary(gt, valid), 2)
        query = sample_training_queries(gt, valid, n=128, band_radius=2,
                                        query_mode="boundary_only")
        x, y = query[0, :, 0].long(), query[0, :, 1].long()
        self.assertTrue(band[0, 0, y, x].all())
        self.assertTrue(((x >= 5) & (x <= 10)).all())

    def test_arbitrary_query_count_and_chunk_equivalence(self):
        torch.manual_seed(4)
        model = SparseHRRefiner(coarse_hidden_channels=8,
                                coarse_context_channels=8, feature_channels=8,
                                hidden_channels=16, iterations=3)
        left = torch.rand(1, 3, 32, 48) * 255
        right = torch.rand(1, 3, 32, 48) * 255
        full = torch.ones(1, 1, 32, 48) * 10
        low = torch.ones(1, 1, 8, 12) * 10
        hidden = torch.randn(1, 8, 8, 12)
        context = torch.randn(1, 8, 8, 12)
        query = torch.tensor([[[5., 5.], [10., 6.], [15., 8.], [20., 11.],
                               [25., 14.], [30., 17.], [35., 20.]]])
        features = (model.feature(left), model.feature(right))
        args = (left, right, full, low, hidden, context)
        complete = model(*args, query, feature_left=features[0],
                         feature_right=features[1])
        chunks = [model(*args, query[:, start:start + 3],
                        feature_left=features[0], feature_right=features[1])
                  for start in range(0, query.shape[1], 3)]
        self.assertEqual(len(complete), 4)
        self.assertTrue(torch.allclose(complete[0], torch.ones_like(complete[0]) * 10))
        for step in range(4):
            self.assertTrue(torch.allclose(complete[step],
                                           torch.cat([chunk[step] for chunk in chunks], dim=1),
                                           atol=1e-5))
        complete[-1].sum().backward()
        self.assertIsNotNone(model.delta_head.weight.grad)
        self.assertTrue(torch.isfinite(model.delta_head.weight.grad).all())

    def test_scatter_preserves_unqueried_pixels(self):
        full = torch.ones(1, 1, 4, 5) * 7
        query = torch.tensor([[[1., 2.], [4., 0.]]])
        result = scatter_queries(full, query, torch.tensor([[[9.], [11.]]]))
        self.assertEqual(float(result[0, 0, 2, 1]), 9.)
        self.assertEqual(float(result[0, 0, 0, 4]), 11.)
        self.assertEqual(float(result[0, 0, 1, 1]), 7.)

    def test_hard_weight_targets_correctable_initial_errors(self):
        gt = torch.tensor([[[[10., 10.]]]])
        valid = torch.ones_like(gt, dtype=torch.bool)
        query = torch.tensor([[[0., 0.], [1., 0.]]])
        initial = torch.tensor([[[10.], [14.]]])
        refined = torch.tensor([[[10.], [13.]]])
        plain = sparse_sequence_loss([initial, refined], gt, valid, query)
        weighted = sparse_sequence_loss([initial, refined], gt, valid, query,
                                        hard_error_weight=4.)
        self.assertAlmostEqual(float(plain), 1.5)
        self.assertAlmostEqual(float(weighted), 2.5)

    def test_direction_head_controls_sign_and_receives_supervision(self):
        torch.manual_seed(7)
        model = SparseHRRefiner(coarse_hidden_channels=8,
                                coarse_context_channels=8, feature_channels=8,
                                hidden_channels=16, iterations=1,
                                direction_classes=True, gate_init_bias=4.)
        left = torch.rand(1, 3, 32, 48) * 255
        right = torch.rand(1, 3, 32, 48) * 255
        full = torch.ones(1, 1, 32, 48) * 10
        low = torch.ones(1, 1, 8, 12) * 10
        hidden = torch.randn(1, 8, 8, 12)
        context = torch.randn(1, 8, 8, 12)
        query = torch.tensor([[[10., 10.], [15., 12.]]])
        with torch.no_grad():
            model.delta_head.weight.zero_()
            model.delta_head.bias.fill_(1.)
            model.direction_head.weight.zero_()
            model.direction_head.bias.copy_(torch.tensor([-4., -4., 4.]))
        forward, logits = model(left, right, full, low, hidden, context, query,
                                return_direction_logits=True)
        self.assertEqual(tuple(logits[0].shape), (1, 2, 3))
        self.assertTrue((forward[-1] > forward[0]).all())
        gt = full.clone()
        gt[0, 0, 10, 10] = 12
        gt[0, 0, 12, 15] = 8
        loss = sparse_direction_loss(logits, forward, gt,
                                     torch.ones_like(gt, dtype=torch.bool), query)
        loss.backward()
        self.assertIsNotNone(model.direction_head.bias.grad)
        self.assertTrue(torch.isfinite(model.direction_head.bias.grad).all())
        with torch.no_grad():
            model.direction_head.bias.copy_(torch.tensor([4., -4., -4.]))
        backward = model(left, right, full, low, hidden, context, query)
        self.assertTrue((backward[-1] < backward[0]).all())

    def test_candidate_gate_relooks_up_and_receives_supervision(self):
        torch.manual_seed(11)
        model = SparseHRRefiner(coarse_hidden_channels=8,
                                coarse_context_channels=8, feature_channels=8,
                                hidden_channels=16, iterations=2,
                                candidate_guided=True)
        left = torch.rand(1, 3, 32, 48) * 255
        right = torch.rand(1, 3, 32, 48) * 255
        full = torch.ones(1, 1, 32, 48) * 10
        low = torch.ones(1, 1, 8, 12) * 10
        hidden = torch.randn(1, 8, 8, 12)
        context = torch.randn(1, 8, 8, 12)
        query = torch.tensor([[[10., 10.], [15., 12.]]])
        predictions, info = model(left, right, full, low, hidden, context,
                                  query, return_candidate_info=True)
        self.assertEqual(len(predictions), 3)
        self.assertEqual(len(info), 2)
        self.assertEqual(tuple(info[0][0].shape), (1, 2, 1))
        gt = full.clone()
        gt[0, 0, 10, 10] = 12
        gt[0, 0, 12, 15] = 8
        loss = candidate_gate_loss(info, predictions, gt,
                                   torch.ones_like(gt, dtype=torch.bool), query)
        loss.backward()
        self.assertTrue(torch.isfinite(model.candidate_gate[-1].weight.grad).all())


if __name__ == "__main__":
    unittest.main()
