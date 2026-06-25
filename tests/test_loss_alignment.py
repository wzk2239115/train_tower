from __future__ import annotations

import unittest

try:
    import torch

    from tower.unify.train_model import _weighted_ce

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class WeightedCeAlignmentTest(unittest.TestCase):
    """Regression guard for the next-token alignment fix (Phase 0.1).

    The NEO data processor pre-shifts labels so labels[p] is the target for
    position p (tower/neo/data/data_processor.py:468). ``_weighted_ce`` must
    therefore supervise logits[p] with labels[p] directly. An extra shift here
    would misalign supervision to T[p+2] and AR training would never converge.
    """

    def test_weighted_ce_learns_aligned_next_token(self):
        torch.manual_seed(0)
        hidden_size, vocab_size, seq_len = 16, 8, 6
        hidden = torch.randn(seq_len, hidden_size)
        head = torch.nn.Linear(hidden_size, vocab_size)
        opt = torch.optim.Adam(head.parameters(), lr=0.1)

        # Pre-shifted labels (as produced by the data processor):
        # position p must learn to emit labels[p].
        labels = torch.tensor([1, 2, 3, 4, 5, -100]).unsqueeze(0)

        first_loss = None
        for _ in range(200):
            logits = head(hidden).unsqueeze(0)  # [1, seq, vocab]
            loss = _weighted_ce(logits, labels, None, vocab_size)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if first_loss is None:
                first_loss = float(loss.item())

        with torch.no_grad():
            pred = head(hidden).argmax(-1)

        # Loss must drop substantially (learning happens).
        self.assertLess(float(loss.item()), first_loss * 0.2)
        # And predictions for supervised positions must match labels exactly —
        # this only holds when supervision is correctly aligned.
        self.assertEqual(pred[:5].tolist(), labels[0, :5].tolist())

    def test_weighted_ce_respects_loss_weight(self):
        torch.manual_seed(0)
        hidden_size, vocab_size = 8, 6
        hidden = torch.randn(4, hidden_size)
        head = torch.nn.Linear(hidden_size, vocab_size)
        labels = torch.tensor([[1, 2, 3, -100]])
        # Only positions 0-2 weighted; position 3 ignored via -100 in labels.
        loss_weight = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        logits = head(hidden).unsqueeze(0)
        loss = _weighted_ce(logits, labels, loss_weight, vocab_size)
        self.assertGreater(float(loss.item()), 0.0)
        self.assertTrue(loss.requires_grad)

    def test_weighted_ce_no_double_shift(self):
        """Explicit guard: a second shift would learn labels[p+1] instead.

        We construct a sequence where labels[p] != labels[p+1] for all p, so a
        correctly aligned head converges but a double-shifted one cannot.
        """
        torch.manual_seed(1)
        hidden_size, vocab_size, seq_len = 24, 10, 8
        hidden = torch.randn(seq_len, hidden_size)
        head = torch.nn.Linear(hidden_size, vocab_size)
        opt = torch.optim.Adam(head.parameters(), lr=0.1)
        # All-distinct targets so any off-by-one shift is detectable.
        labels = torch.tensor([0, 3, 1, 7, 2, 5, 9, -100]).unsqueeze(0)

        for _ in range(300):
            logits = head(hidden).unsqueeze(0)
            loss = _weighted_ce(logits, labels, None, vocab_size)
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            pred = head(hidden).argmax(-1)
        # Every supervised position must match — no off-by-one.
        self.assertEqual(pred[:7].tolist(), labels[0, :7].tolist())


if __name__ == "__main__":
    unittest.main()
