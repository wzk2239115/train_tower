from __future__ import annotations

import unittest
from pathlib import Path

from tower.train.config import TrainConfig, load_train_config
from tower.train.size_preset import (
    apply_size_preset_to_train_config,
    assert_model_tower_layer_consistency,
    build_scaled_tower_raw,
    exit_after_layer,
    load_size_preset,
    merge_model_config_dict,
    resolve_model_config_dict,
)
from tower.unify.tower_config import clear_active_tower_overlay, load_tower_config


class SizePresetTest(unittest.TestCase):
    def tearDown(self) -> None:
        clear_active_tower_overlay()

    def test_500m_preset_matches_defaults(self) -> None:
        preset = load_size_preset("500m")
        tower_raw = build_scaled_tower_raw(preset)
        cfg = apply_size_preset_to_train_config(TrainConfig(size_preset="500m"))
        tower = load_tower_config()

        self.assertEqual(cfg.num_hidden_layers, 26)
        self.assertEqual(tower.num_hidden_layers, 26)
        self.assertEqual(tower.exit("world_elf").after_layer, 7)
        self.assertEqual(tower.exit("generative_elf").after_layer, 25)
        self.assertEqual(tower.shallow_train_layers("world_pt"), 8)
        assert_model_tower_layer_consistency(cfg)

        base = resolve_model_config_dict(TrainConfig(model_config_path=preset["model_config_path"]))
        merged = merge_model_config_dict(base, preset)
        self.assertEqual(merged["llm_config"]["num_hidden_layers"], 26)
        self.assertEqual(merged["llm_config"]["hidden_size"], 768)
        self.assertEqual(tower_raw["num_hidden_layers"], 26)

    def test_tiny_smoke_scales_exits(self) -> None:
        cfg = apply_size_preset_to_train_config(TrainConfig(size_preset="tiny_smoke"))
        tower = load_tower_config()
        llm_layers = 8

        self.assertEqual(cfg.num_hidden_layers, llm_layers)
        self.assertEqual(tower.num_hidden_layers, llm_layers)
        self.assertEqual(tower.exit("world_elf").after_layer, exit_after_layer(llm_layers, 0.28))
        self.assertEqual(tower.exit("generative_elf").after_layer, llm_layers - 1)
        for spec in tower.exits:
            self.assertLess(spec.after_layer, llm_layers)
        assert_model_tower_layer_consistency(cfg)

    def test_backward_compat_without_preset(self) -> None:
        cfg = load_train_config(config_path=Path("configs/train/understanding_warmup.yaml"))
        self.assertIsNone(cfg.size_preset)
        tower = load_tower_config()
        self.assertEqual(tower.num_hidden_layers, 26)
        self.assertEqual(tower.exit("understanding_elf").after_layer, 21)

    def test_yaml_model_size_alias(self) -> None:
        cfg = TrainConfig(
            model_config_path="configs/model/sensenova_500m_mot",
            model_size="tiny_smoke",
        )
        cfg = apply_size_preset_to_train_config(cfg)
        self.assertEqual(cfg.size_preset, "tiny_smoke")
        self.assertEqual(load_tower_config().num_hidden_layers, 8)

    def test_cli_size_overrides_yaml(self) -> None:
        cfg = TrainConfig(size_preset="1b")
        cfg = apply_size_preset_to_train_config(cfg, cli_size="tiny_smoke")
        self.assertEqual(cfg.size_preset, "tiny_smoke")
        self.assertEqual(load_tower_config().num_hidden_layers, 8)


if __name__ == "__main__":
    unittest.main()
