from __future__ import annotations

import unittest

try:
    import torch

    from tower.train.grad_norm import GradNormBalancer, GradNormWeights
    from tower.train.config import TrainConfig

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class TestGradNormWeights(unittest.TestCase):
    def test_default_weights_equal(self):
        gw = GradNormWeights(["ce_loss", "image_fm_loss"])
        w = gw.weights_dict()
        self.assertAlmostEqual(w["ce_loss"], 1.0, places=4)
        self.assertAlmostEqual(w["image_fm_loss"], 1.0, places=4)

    def test_custom_initial(self):
        gw = GradNormWeights(
            ["ce_loss", "image_fm_loss"],
            initial_weights={"ce_loss": 2.0, "image_fm_loss": 0.5},
        )
        w = gw.weights_dict()
        self.assertGreater(w["ce_loss"], w["image_fm_loss"])

    def test_update_from_grad_norms(self):
        gw = GradNormWeights(["ce_loss", "image_fm_loss"])
        grad_norms = {"ce_loss": 10.0, "image_fm_loss": 1.0}
        new_w = gw.update_from_grad_norms(grad_norms, target_norm=5.0, lr=0.1)
        self.assertIsInstance(new_w, dict)
        self.assertIn("ce_loss", new_w)
        self.assertIn("image_fm_loss", new_w)

    def test_update_reduces_high_grad_weight(self):
        gw = GradNormWeights(["ce_loss", "image_fm_loss"])
        grad_norms = {"ce_loss": 100.0, "image_fm_loss": 1.0}
        w_before = gw.get("ce_loss")
        gw.update_from_grad_norms(grad_norms, target_norm=50.0, lr=0.1)
        w_after = gw.get("ce_loss")
        self.assertLessEqual(w_after, w_before + 0.01)

    def test_empty_grad_norms_noop(self):
        gw = GradNormWeights(["ce_loss"])
        result = gw.update_from_grad_norms({}, 1.0)
        self.assertEqual(result, {})

    def test_sum_of_weights_near_n(self):
        gw = GradNormWeights(["ce_loss", "image_fm_loss", "audio_fm_loss"])
        w = gw.weights_dict()
        total = sum(w.values())
        self.assertAlmostEqual(total, 3.0, places=3)


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class TestGradNormBalancer(unittest.TestCase):
    def _make_cfg(self, *, enabled=False, target=1.0, interval=100):
        from unittest.mock import MagicMock
        cfg = TrainConfig()
        cfg.grad_norm_balance = enabled
        cfg.grad_norm_target = target
        cfg.grad_norm_update_interval = interval
        return cfg

    def _make_model(self):
        from unittest.mock import MagicMock
        model = MagicMock()
        model.tower_cfg = None
        model._current_stage = MagicMock(return_value="unified_mt")
        model._loss_breakdown_key = MagicMock(side_effect=lambda s: s.name)
        return model

    def test_disabled_returns_static_weights(self):
        cfg = self._make_cfg(enabled=False)
        model = self._make_model()
        balancer = GradNormBalancer(cfg, model)
        self.assertFalse(balancer.enabled)
        self.assertIsNone(balancer.weights_module)

    def test_enabled_creates_weights_module(self):
        cfg = self._make_cfg(enabled=True)
        model = self._make_model()
        balancer = GradNormBalancer(cfg, model)
        self.assertTrue(balancer.enabled)
        self.assertIsNotNone(balancer.weights_module)

    def test_weighted_total_disabled(self):
        cfg = self._make_cfg(enabled=False)
        model = self._make_model()
        balancer = GradNormBalancer(cfg, model)
        losses = {"ce_loss": torch.tensor(4.0), "image_fm_loss": torch.tensor(900.0)}
        total, weights = balancer.weighted_total(losses, fm_weight=0.005, ce_weight=1.0)
        self.assertAlmostEqual(float(total), 4.0 + 0.005 * 900.0, places=3)
        self.assertAlmostEqual(weights["ce_loss"], 1.0)
        self.assertAlmostEqual(weights["image_fm_loss"], 0.005)

    def test_weighted_total_enabled(self):
        cfg = self._make_cfg(enabled=True)
        model = self._make_model()
        balancer = GradNormBalancer(cfg, model)
        losses = {"ce_loss": torch.tensor(4.0), "image_fm_loss": torch.tensor(900.0)}
        total, weights = balancer.weighted_total(losses)
        self.assertGreater(float(total), 0)
        self.assertIn("ce_loss", weights)
        self.assertIn("image_fm_loss", weights)

    def test_record_grad_norms_skips_warmup(self):
        cfg = self._make_cfg(enabled=True, interval=1)
        model = self._make_model()
        balancer = GradNormBalancer(cfg, model)
        param = torch.nn.Parameter(torch.randn(10, 10))
        losses = {"ce_loss": torch.tensor(1.0, requires_grad=True)}
        result = balancer.record_grad_norms(losses, [param])
        self.assertEqual(result, {})

    def test_last_diagnostics(self):
        cfg = self._make_cfg(enabled=False)
        model = self._make_model()
        balancer = GradNormBalancer(cfg, model)
        diag = balancer.last_diagnostics()
        self.assertIn("grad_norm_step", diag)


if __name__ == "__main__":
    unittest.main()
