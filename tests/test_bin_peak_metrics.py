"""CPU checks of candidate diagnostics, independent of the stereo model."""

import unittest

import torch

from scripts.bin.bin_peak_metrics import (
    candidate_errors,
    distinct_peak_indices,
    nearest_gt_bin_rank,
)


class BinPeakMetricsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_multimodal_distribution_can_have_a_bad_mean_and_good_peaks(self):
        bins = torch.arange(5, dtype=torch.float32)[None] * 20
        logits = torch.tensor([0., 10., 0., 10., 0.]).view(1, 5, 1, 1)
        mean = (logits.softmax(1) * bins[:, :, None, None]).sum(1)
        result = candidate_errors(logits, torch.ones_like(logits, dtype=torch.bool),
                                  bins, mean, mean[0], torch.tensor([[60.]]))
        self.assertAlmostEqual(result["errors"]["fused"].item(), 20., places=4)
        self.assertEqual(result["errors"]["top1_oracle"].item(), 40.)
        self.assertEqual(result["errors"]["top2_oracle"].item(), 0.)
        self.assertEqual(result["peak_indices"][0, :2, 0, 0].tolist(), [1, 3])
        self.assertEqual(result["peak_valid"].sum().item(), 2)

    def test_plateau_must_exceed_both_outer_neighbors(self):
        # The 2,2 plateau is rising toward 3 and must not become a peak.
        logits = torch.tensor([1., 2., 2., 3., 0., 4., 4., 1.]).view(1, 8, 1, 1)
        bins = torch.arange(8, dtype=torch.float32)[None]
        indices, valid = distinct_peak_indices(
            logits, torch.ones_like(logits, dtype=torch.bool), bins,
            max_peaks=4, min_separation_px=0.)
        self.assertEqual(indices[0, :, 0, 0].tolist(), [5, 3, 0, 0])
        self.assertEqual(valid[0, :, 0, 0].tolist(), [True, True, False, False])
        # A flat whole distribution is one plateau, not four independent peaks.
        _, flat_valid = distinct_peak_indices(torch.ones_like(logits),
                                              torch.ones_like(logits, dtype=torch.bool), bins,
                                              min_separation_px=0.)
        self.assertEqual(flat_valid.sum().item(), 1)

    def test_invalid_candidates_never_count_as_matches(self):
        logits = torch.tensor([5., 100., 1., float("nan"), 3.]).view(1, 5, 1, 1)
        logits = logits.expand(1, 5, 1, 2).clone()
        valid = torch.ones_like(logits, dtype=torch.bool)
        valid[:, 1] = False
        valid[..., 1] = False
        bins = torch.arange(5, dtype=torch.float32)[None] * 10
        result = candidate_errors(logits, valid, bins, torch.tensor([[[10., 10.]]]),
                                  torch.tensor([[10., 10.]]), torch.tensor([[10., 10.]]),
                                  min_separation_px=0.)
        self.assertEqual(result["peak_indices"][0, :2, 0, 0].tolist(), [0, 4])
        self.assertEqual(result["errors"]["all_bin_oracle"][0, 0].item(), 10.)
        for name, error in result["errors"].items():
            if name != "fused":
                self.assertTrue(torch.isinf(error[0, 1]))
            self.assertFalse(torch.isnan(error).any())
        self.assertFalse(result["peak_valid"][..., 1].any())
        self.assertTrue(torch.equal(result["peak_indices"][..., 1],
                                    torch.zeros_like(result["peak_indices"][..., 1])))
        rank = nearest_gt_bin_rank(logits, valid, bins, torch.tensor([[10., 10.]]))
        self.assertTrue(torch.isinf(rank[0, 0, 1]))

    def test_nms_uses_physical_distance_for_each_branch(self):
        logits = torch.tensor([9., 0., 8., 0., 7.]).view(1, 5, 1, 1).expand(2, -1, -1, -1)
        bins = torch.tensor([[0., 2., 4., 6., 8.], [0., 4., 8., 12., 16.]])
        indices, valid = distinct_peak_indices(logits, torch.ones_like(logits, dtype=torch.bool),
                                               bins, min_separation_px=8.)
        self.assertEqual(indices[0, :2, 0, 0].tolist(), [0, 4])
        self.assertEqual(indices[1, :3, 0, 0].tolist(), [0, 2, 4])
        self.assertEqual(valid.sum(dim=1).flatten().tolist(), [2, 3])

    def test_oracles_are_nested_and_nearest_bin_rank_handles_score_ties(self):
        logits = torch.tensor([[8., 1., 6., 0., 6., 1., 4.],
                               [1., 2., 3., 4., 5., 6., 7.]]).view(2, 7, 1, 1)
        bins = torch.tensor([[0., 10., 20., 30., 40., 50., 60.],
                             [0., 20., 40., 60., 80., 100., 120.]])
        valid = torch.ones_like(logits, dtype=torch.bool)
        gt = torch.tensor([[40.]])
        result = candidate_errors(logits, valid, bins, torch.tensor([[[30.]], [[50.]]]),
                                  torch.tensor([[40.]]), gt, min_separation_px=8.)
        e = result["errors"]
        self.assertLessEqual(e["all_bin_oracle"].item(), e["top4_oracle"].item())
        self.assertLessEqual(e["top4_oracle"].item(), e["top2_oracle"].item())
        self.assertLessEqual(e["top2_oracle"].item(), e["top1_oracle"].item())
        self.assertEqual(e["branch_oracle"].item(), 10.)
        rank = nearest_gt_bin_rank(logits, valid, bins, gt)
        self.assertEqual(rank.flatten().tolist(), [2., 5.])
        # At an equal physical distance, choose the lower-index GT bin.
        tie_rank = nearest_gt_bin_rank(logits[:1], valid[:1], bins[:1], torch.tensor([[5.]]))
        self.assertEqual(tie_rank.item(), 1.)

    def test_single_bin_and_nonfinite_ground_truth(self):
        logits = torch.zeros(1, 1, 1, 2)
        valid = torch.ones_like(logits, dtype=torch.bool)
        bins = torch.tensor([[12.]])
        gt = torch.tensor([[12., float("nan")]])
        result = candidate_errors(logits, valid, bins, torch.zeros(1, 1, 2),
                                  torch.zeros(1, 2), gt)
        self.assertEqual(result["peak_valid"].sum().item(), 2)
        self.assertEqual(result["errors"]["all_bin_oracle"][0, 0].item(), 0.)
        for error in result["errors"].values():
            self.assertTrue(torch.isinf(error[0, 1]))
        rank = nearest_gt_bin_rank(logits, valid, bins, gt)
        self.assertEqual(rank[0, 0, 0].item(), 1.)
        self.assertTrue(torch.isinf(rank[0, 0, 1]))


if __name__ == "__main__":
    unittest.main()
