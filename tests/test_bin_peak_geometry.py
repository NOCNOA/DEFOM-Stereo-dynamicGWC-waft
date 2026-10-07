"""Diagnostic coordinate and denominator checks; CPU only."""
import unittest

import torch

from scripts.bin.diagnose_bin_peaks import Metrics, native_geometry


class BinPeakGeometryTests(unittest.TestCase):
    def test_padding_does_not_create_gt_and_coordinates_stay_in_original_pixels(self):
        gt = torch.full((5, 7), 2.)
        gt[3, 4] = 5.
        gt[3, 0] = float('nan')
        valid = torch.ones_like(gt, dtype=torch.bool)
        low_gt, low_valid, in_right, _, x = native_geometry(
            gt, valid, [0, 1, 1, 2], (2, 2), factor=4)
        self.assertFalse(low_valid[0].any())
        self.assertFalse(low_valid[1, 0])
        self.assertEqual(low_gt[1, 1].item(), 5.)
        self.assertEqual(x[1, 1].item(), 4.)
        self.assertFalse(in_right[1, 1])

    def test_missing_candidates_count_as_recall_failures(self):
        stats = Metrics()
        stats.add(['example'], {
            'fused': torch.tensor([[0., 0.]]),
            'all_bin_oracle': torch.tensor([[2., float('inf')]]),
            'top4_oracle': torch.tensor([[10., float('inf')]]),
        }, {'valid_gt': torch.tensor([[True, True]])}, torch.tensor([[10., 20.]]))
        rows = {row['method']: row for row in stats.rows() if row['gt_band'] == 'all'}
        self.assertEqual(rows['top4_oracle']['epe'], 10.)
        self.assertEqual(rows['top4_oracle']['missing_candidate_pct'], 50.)
        self.assertEqual(rows['top4_oracle']['recall_16px_pct'], 50.)
        self.assertEqual(rows['top4_oracle']['reference_queries'], 1)
        self.assertEqual(rows['top4_oracle']['within_reference_plus_8px_pct'], 100.)
        self.assertEqual(rows['fused']['recall_2px_pct'], 100.)


if __name__ == '__main__':
    unittest.main()
