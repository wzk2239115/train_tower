from __future__ import annotations

import unittest

from tower.train.config import TrainConfig, load_train_config
from tower.train.experiment_profile import (
    load_experiment_profile,
    load_train_config_from_experiment,
    list_experiment_profiles,
    stages_dict_to_curriculum,
    summarize_profile,
)
from tower.train.size_preset import apply_size_preset_to_train_config
from tower.train.stage_boundaries import resolve_stage_boundaries
from tower.unify.tower_config import clear_active_tower_overlay


class ExperimentProfileTest(unittest.TestCase):
    def tearDown(self) -> None:
        clear_active_tower_overlay()

    def test_list_profiles_includes_builtins(self) -> None:
        names = list_experiment_profiles()
        self.assertIn("500m_continuous", names)
        self.assertIn("tiny_smoke", names)

    def test_load_500m_continuous_profile(self) -> None:
        profile = load_experiment_profile("500m_continuous")
        self.assertEqual(profile.size_preset, "500m")
        self.assertEqual(profile.train_config, "configs/train/continuous.yaml")

        cfg = load_train_config_from_experiment("500m_continuous")
        self.assertEqual(cfg.experiment_profile, "500m_continuous")
        self.assertEqual(cfg.size_preset, "500m")
        self.assertEqual(cfg.max_steps, 420_000)
        self.assertEqual(len(cfg.curriculum), 5)
        self.assertEqual(cfg.output_dir, "outputs/pretrain/500m_continuous")

    def test_tiny_smoke_profile_overrides(self) -> None:
        cfg = load_train_config_from_experiment("tiny_smoke")
        self.assertEqual(cfg.size_preset, "tiny_smoke")
        self.assertEqual(cfg.max_steps, 100)
        self.assertEqual(cfg.max_pixels, 262_144)

        world = cfg.curriculum_data_settings_for_step(0)
        self.assertEqual(world["datasets"], "blip3o_short_pt")
        self.assertEqual(world["max_pixels"], 262_144)

        sft = cfg.curriculum_data_settings_for_step(90)
        self.assertEqual(sft["stage"], "unified_sft")

    def test_stage_boundary_resolution(self) -> None:
        cfg = load_train_config_from_experiment("500m_continuous")
        boundaries = resolve_stage_boundaries(cfg)

        self.assertEqual(len(boundaries), 5)
        self.assertEqual(boundaries[0].stage, "world_pt")
        self.assertEqual(boundaries[0].start_step, 0)
        self.assertEqual(boundaries[0].end_step, 49_999)
        self.assertEqual(boundaries[0].step_count, 50_000)

        self.assertEqual(boundaries[1].stage, "understanding_warmup")
        self.assertEqual(boundaries[1].start_step, 50_000)
        self.assertEqual(boundaries[1].end_step, 249_999)

        self.assertEqual(boundaries[-1].stage, "unified_sft")
        self.assertEqual(boundaries[-1].end_step, 419_999)

    def test_tiny_smoke_stage_boundaries(self) -> None:
        cfg = load_train_config_from_experiment("tiny_smoke")
        boundaries = resolve_stage_boundaries(cfg)
        self.assertEqual([b.step_count for b in boundaries], [20, 20, 20, 20, 20])
        self.assertEqual(boundaries[-1].end_step, 99)

    def test_stages_dict_to_curriculum(self) -> None:
        curriculum = stages_dict_to_curriculum(
            {
                "world_pt": {"max_steps": 10, "datasets": ["blip3o_short_pt"]},
                "understanding_warmup": {"max_steps": 20, "datasets": "blip3o_short_pt"},
            }
        )
        self.assertEqual(len(curriculum), 2)
        self.assertEqual(curriculum[0]["until_step"], 9)
        self.assertEqual(curriculum[1]["until_step"], 29)
        self.assertEqual(curriculum[0]["datasets"], "blip3o_short_pt")

    def test_backward_compat_config_and_size(self) -> None:
        from pathlib import Path

        cfg = load_train_config(
            config_path=Path("configs/train/understanding_warmup.yaml"),
            size_preset="tiny_smoke",
        )
        self.assertIsNone(cfg.experiment_profile)
        self.assertEqual(cfg.size_preset, "tiny_smoke")

    def test_cli_size_overrides_profile_preset(self) -> None:
        cfg = load_train_config_from_experiment("500m_continuous", size_preset="tiny_smoke")
        self.assertEqual(cfg.size_preset, "tiny_smoke")

    def test_summarize_profile(self) -> None:
        summary = summarize_profile("tiny_smoke")
        self.assertEqual(summary["max_steps"], 100)
        self.assertEqual(len(summary["stages"]), 5)
        self.assertIn("blip3o_short_pt", summary["stages"][0]["datasets"])

    def test_single_stage_config_boundary(self) -> None:
        cfg = TrainConfig(stage="understanding_warmup", max_steps=1000, datasets="blip3o_short_pt")
        cfg = apply_size_preset_to_train_config(cfg)
        boundaries = resolve_stage_boundaries(cfg)
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(boundaries[0].start_step, 0)
        self.assertEqual(boundaries[0].end_step, 999)


if __name__ == "__main__":
    unittest.main()
