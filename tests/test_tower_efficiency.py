from __future__ import annotations

import unittest
from unittest import mock

import torch
import torch.nn as nn

from tower.train.freeze import (
    _llm_layer_index,
    _world_pt_shallow_trainable,
    apply_stage_freeze,
)
from tower.unify.build import _resolve_attn_implementation
from tower.unify.tower_config import load_tower_config


class TowerConfigEfficiencyTest(unittest.TestCase):
    def test_world_pt_max_hook_layer(self):
        cfg = load_tower_config()
        self.assertEqual(cfg.max_hook_layer("world_pt"), 15)
        self.assertEqual(cfg.max_hook_layer("understanding_warmup"), 21)
        self.assertEqual(cfg.max_hook_layer("generation_pt"), 25)

    def test_world_pt_shallow_train_layers(self):
        cfg = load_tower_config()
        self.assertEqual(cfg.shallow_train_layers("world_pt"), 8)
        self.assertIsNone(cfg.shallow_train_layers("understanding_warmup"))


class PartialForwardTest(unittest.TestCase):
    def test_run_backbone_layers_stops_early(self):
        from tower.unify.flow_tower import FlowJepaTowerTrainModel
        from tower.train.config import TrainConfig

        class _FakeLayer(nn.Module):
            def forward(self, hidden_states, **kwargs):
                return hidden_states + 1.0

        class _FakeLLM(nn.Module):
            def __init__(self, n: int):
                super().__init__()
                self.layers = nn.ModuleList([_FakeLayer() for _ in range(n)])

        class _FakeNeo(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = nn.Module()
                self.language_model.model = _FakeLLM(26)

        neo = _FakeNeo()
        tower = FlowJepaTowerTrainModel.__new__(FlowJepaTowerTrainModel)
        nn.Module.__init__(tower)
        tower.model = neo
        tower.tower_cfg = load_tower_config()
        tower.tower_exits = nn.ModuleDict()
        tower.cfg = TrainConfig()
        tower.audio_proj = nn.Identity()

        hidden = torch.zeros(1, 4, 1)
        indicators = torch.zeros(4, dtype=torch.bool)
        hooks = tower._run_backbone_layers(
            hidden,
            indexes=torch.zeros(3, 4, dtype=torch.long),
            attn=torch.ones(4, 4),
            indicators=indicators,
            stop_layer=15,
            hook_layers={7, 11, 15, 21},
        )
        self.assertEqual(set(hooks.keys()), {7, 11, 15})
        self.assertNotIn(21, hooks)
        self.assertEqual(hooks[15][0, 0, 0].item(), 16.0)


class WorldPtFreezeTest(unittest.TestCase):
    def test_llm_layer_index(self):
        name = "language_model.model.layers.7.self_attn.q_proj.weight"
        self.assertEqual(_llm_layer_index(name), 7)

    def test_shallow_trainable_rules(self):
        shallow = 8
        self.assertTrue(_world_pt_shallow_trainable("vision_model.blocks.0.weight", shallow))
        self.assertTrue(
            _world_pt_shallow_trainable(
                "language_model.model.layers.7.mlp.down_proj.weight", shallow
            )
        )
        self.assertFalse(
            _world_pt_shallow_trainable(
                "language_model.model.layers.8.mlp.down_proj.weight", shallow
            )
        )
        self.assertFalse(
            _world_pt_shallow_trainable(
                "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight", shallow
            )
        )
        self.assertFalse(_world_pt_shallow_trainable("fm_modules.timestep_embedder.mlp.0.weight", shallow))

    def test_apply_stage_freeze_world_pt(self):
        params = {
            "vision_model.proj.weight": nn.Parameter(torch.zeros(1)),
            "language_model.model.layers.0.weight": nn.Parameter(torch.zeros(1)),
            "language_model.model.layers.7.weight": nn.Parameter(torch.zeros(1)),
            "language_model.model.layers.8.weight": nn.Parameter(torch.zeros(1)),
            "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight": nn.Parameter(torch.zeros(1)),
            "fm_modules.head.weight": nn.Parameter(torch.zeros(1)),
            "language_model.model.embed_tokens.weight": nn.Parameter(torch.zeros(1)),
        }
        model = mock.Mock()
        model.named_parameters.return_value = params.items()
        model.parameters.return_value = params.values()
        apply_stage_freeze(model, "world_pt")
        self.assertTrue(params["vision_model.proj.weight"].requires_grad)
        self.assertTrue(params["language_model.model.layers.0.weight"].requires_grad)
        self.assertTrue(params["language_model.model.layers.7.weight"].requires_grad)
        self.assertFalse(params["language_model.model.layers.8.weight"].requires_grad)
        self.assertFalse(params["language_model.model.layers.0.self_attn.q_proj_mot_gen.weight"].requires_grad)
        self.assertFalse(params["fm_modules.head.weight"].requires_grad)


class AttnImplementationTest(unittest.TestCase):
    def test_default_is_sdpa(self):
        self.assertEqual(_resolve_attn_implementation("sdpa"), "sdpa")

    def test_flash_attn_fallback(self):
        import builtins

        real_import = builtins.__import__

        def _import(name, *args, **kwargs):
            if name == "flash_attn":
                raise ImportError("no flash_attn")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=_import):
            self.assertEqual(_resolve_attn_implementation("flash_attention_2"), "sdpa")


if __name__ == "__main__":
    unittest.main()
