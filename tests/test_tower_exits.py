from __future__ import annotations

import unittest

try:
    import torch
    import torch.nn as nn

    from tower.unify.tower_exits import CeTowerExit, ElfFlowTowerExit, JepaTowerExit

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class CeTowerExitTest(unittest.TestCase):
    def test_forward_returns_scalar_loss(self):
        hidden = torch.randn(8, 32)
        labels = torch.tensor([3, 7, 1, 0, -100, 5, 2, 9])
        exit_mod = CeTowerExit(hidden_size=32, vocab_size=16)
        loss = exit_mod(hidden, labels)
        self.assertEqual(loss.shape, ())
        self.assertGreater(loss.item(), 0.0)
        self.assertTrue(loss.requires_grad)

    def test_forward_all_ignore_labels_returns_zero(self):
        hidden = torch.randn(4, 32)
        labels = torch.tensor([-100, -100, -100, -100])
        exit_mod = CeTowerExit(hidden_size=32, vocab_size=16)
        loss = exit_mod(hidden, labels)
        self.assertEqual(loss.item(), 0.0)

    def test_forward_empty_hidden_returns_zero(self):
        hidden = torch.randn(0, 32)
        labels = torch.tensor([], dtype=torch.long)
        exit_mod = CeTowerExit(hidden_size=32, vocab_size=16)
        loss = exit_mod(hidden, labels)
        self.assertEqual(loss.item(), 0.0)

    def test_vocab_size_matches_head(self):
        exit_mod = CeTowerExit(hidden_size=64, vocab_size=1000)
        self.assertEqual(exit_mod.head.out_features, 1000)
        self.assertEqual(exit_mod.head.in_features, 64)

    def test_gradient_flows(self):
        hidden = torch.randn(4, 16, requires_grad=True)
        labels = torch.tensor([1, 2, 3, 4])
        exit_mod = CeTowerExit(hidden_size=16, vocab_size=10)
        loss = exit_mod(hidden, labels)
        loss.backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue((hidden.grad != 0).any())


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class ElfFlowTowerExitTest(unittest.TestCase):
    def test_forward_returns_scalar_loss(self):
        hidden_size = 32
        out_dim = 16
        n_tokens = 8
        exit_mod = ElfFlowTowerExit(hidden_size, out_dim, elf_depth=1)

        hidden = torch.randn(n_tokens, hidden_size)
        z = torch.randn(n_tokens, out_dim)
        t = torch.full((n_tokens,), 0.5)
        x_clean = torch.randn(n_tokens, out_dim)
        loss = exit_mod(hidden, z, t, x_clean, t_eps=0.05)
        self.assertEqual(loss.shape, ())
        self.assertGreater(loss.item(), 0.0)
        self.assertTrue(loss.requires_grad)

    def test_forward_empty_hidden_returns_zero(self):
        exit_mod = ElfFlowTowerExit(32, 16, elf_depth=1)
        hidden = torch.randn(0, 32)
        z = torch.randn(0, 16)
        t = torch.tensor([])
        x_clean = torch.randn(0, 16)
        loss = exit_mod(hidden, z, t, x_clean)
        self.assertEqual(loss.item(), 0.0)

    def test_predict_x_output_shape(self):
        hidden_size = 32
        out_dim = 48
        n_tokens = 5
        exit_mod = ElfFlowTowerExit(hidden_size, out_dim, elf_depth=2)
        hidden = torch.randn(n_tokens, hidden_size)
        t = torch.full((n_tokens,), 0.3)
        pred = exit_mod.predict_x(hidden, t)
        self.assertEqual(pred.shape, (n_tokens, out_dim))

    def test_predict_x_scalar_t_broadcasts(self):
        exit_mod = ElfFlowTowerExit(32, 16, elf_depth=1)
        hidden = torch.randn(6, 32)
        t = torch.tensor(0.5)
        pred = exit_mod.predict_x(hidden, t)
        self.assertEqual(pred.shape, (6, 16))

    def test_self_conditioning_changes_prediction(self):
        exit_mod = ElfFlowTowerExit(32, 16, elf_depth=1)
        hidden = torch.randn(4, 32)
        t = torch.full((4,), 0.5)
        pred_no_sc = exit_mod.predict_x(hidden, t)
        self_cond = torch.randn(4, 16)
        pred_with_sc = exit_mod.predict_x(hidden, t, self_cond=self_cond)
        self.assertFalse(torch.allclose(pred_no_sc, pred_with_sc))


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class JepaTowerExitTest(unittest.TestCase):
    def test_forward_returns_scalar_loss(self):
        hidden_size = 32
        n_tokens = 16
        exit_mod = JepaTowerExit(hidden_size)

        hidden = torch.randn(n_tokens, hidden_size)
        target_hidden = torch.randn(n_tokens, hidden_size)
        pred_mask = torch.zeros(n_tokens, dtype=torch.bool)
        pred_mask[:4] = True

        loss = exit_mod(hidden, target_hidden, pred_mask)
        self.assertEqual(loss.shape, ())
        self.assertGreater(loss.item(), 0.0)
        self.assertTrue(loss.requires_grad)

    def test_forward_empty_mask_returns_zero(self):
        exit_mod = JepaTowerExit(32)
        hidden = torch.randn(8, 32)
        target = torch.randn(8, 32)
        mask = torch.zeros(8, dtype=torch.bool)
        loss = exit_mod(hidden, target, mask)
        self.assertEqual(loss.item(), 0.0)

    def test_forward_empty_hidden_returns_zero(self):
        exit_mod = JepaTowerExit(32)
        hidden = torch.randn(0, 32)
        target = torch.randn(0, 32)
        mask = torch.zeros(0, dtype=torch.bool)
        loss = exit_mod(hidden, target, mask)
        self.assertEqual(loss.item(), 0.0)

    def test_ema_update_target(self):
        exit_mod = JepaTowerExit(32)
        before = {n: p.clone() for n, p in exit_mod.target_projector.named_parameters()}
        exit_mod.ema_update_target()
        for n, p in exit_mod.target_projector.named_parameters():
            self.assertFalse(torch.equal(before[n], p), f"EMA did not update {n}")

    def test_target_projector_no_grad(self):
        exit_mod = JepaTowerExit(32)
        for p in exit_mod.target_projector.parameters():
            self.assertFalse(p.requires_grad)


if __name__ == "__main__":
    unittest.main()
