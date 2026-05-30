from __future__ import annotations

import unittest

import torch

from tower.train.vision_batch import (
    reconcile_vision_inputs,
    snap_grid_hw_to_vit_merge,
    vit_output_patch_total,
)


class VisionBatchTest(unittest.TestCase):
    def test_snap_odd_grid_to_even(self):
        feat = 768
        flat = torch.randn(89 * 91, feat)
        grid = torch.tensor([[89, 91]])
        out_flat, out_grid = snap_grid_hw_to_vit_merge(flat, grid, spatial_merge=2)
        self.assertEqual(out_grid.tolist(), [[88, 90]])
        self.assertEqual(out_flat.shape[0], 88 * 90)
        self.assertEqual(vit_output_patch_total(out_grid), out_flat.shape[0] // 4)

    def test_reconcile_partial_image_after_truncation(self):
        feat = 768
        flat = torch.randn(250, feat)
        grid = torch.tensor([[10, 10], [10, 10], [10, 10], [10, 10], [10, 10]])
        out_flat, out_grid = reconcile_vision_inputs(flat, grid, spatial_merge=2)
        self.assertEqual(out_flat.shape[0], int((out_grid[:, 0] * out_grid[:, 1]).sum()))
        for h, w in out_grid.tolist():
            self.assertEqual(h % 2, 0)
            self.assertEqual(w % 2, 0)
        total_in = out_flat.shape[0]
        total_out = sum((h // 2) * (w // 2) for h, w in out_grid.tolist())
        self.assertEqual(total_out, total_in // 4)


if __name__ == "__main__":
    unittest.main()
