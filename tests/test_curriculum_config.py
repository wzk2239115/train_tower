from __future__ import annotations

import unittest

from tower.train.config import TrainConfig, load_train_config


class CurriculumConfigTest(unittest.TestCase):
    def test_phase_resolution_and_overrides(self):
        cfg = TrainConfig(
            stage="world_pt",
            max_seq_length=8192,
            max_pixels=8_388_608,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            curriculum=[
                {
                    "stage": "world_pt",
                    "until_step": 10,
                    "max_seq_length": 4096,
                    "max_pixels": 4_194_304,
                    "per_device_train_batch_size": 6,
                },
                {
                    "stage": "world_pt",
                    "until_step": 99,
                    "max_seq_length": 8192,
                    "max_pixels": 6_291_456,
                    "per_device_train_batch_size": 4,
                },
            ],
        )
        early = cfg.curriculum_data_settings_for_step(5)
        self.assertEqual(early["phase_index"], 0)
        self.assertEqual(early["max_seq_length"], 4096)
        self.assertEqual(early["max_pixels"], 4_194_304)
        self.assertEqual(early["per_device_train_batch_size"], 6)

        late = cfg.curriculum_data_settings_for_step(50)
        self.assertEqual(late["phase_index"], 1)
        self.assertEqual(late["max_seq_length"], 8192)
        self.assertEqual(late["max_pixels"], 6_291_456)
        self.assertEqual(late["per_device_train_batch_size"], 4)

    def test_curriculum_dataset_and_loss_overrides(self):
        cfg = TrainConfig(
            stage="world_pt",
            datasets="blip3o_short_pt",
            learning_rate=2e-4,
            loss_weights={"ce": 0.0, "fm": 0.0},
            curriculum=[
                {
                    "stage": "world_pt",
                    "until_step": 10,
                    "datasets": "blip3o_long_pt,blip3o_short_pt",
                },
                {
                    "stage": "understanding_warmup",
                    "until_step": 99,
                    "datasets": "blip3o_long_pt,blip3o_short_pt",
                    "loss_weights": {"ce": 1.0, "fm": 0.0},
                    "learning_rate": 1e-4,
                },
            ],
        )
        world = cfg.curriculum_data_settings_for_step(5)
        self.assertEqual(world["datasets"], "blip3o_long_pt,blip3o_short_pt")
        self.assertEqual(world["loss_weights"], {"ce": 0.0, "fm": 0.0})

        uw = cfg.curriculum_data_settings_for_step(50)
        self.assertEqual(uw["stage"], "understanding_warmup")
        self.assertEqual(uw["loss_weights"], {"ce": 1.0, "fm": 0.0})
        self.assertEqual(uw["learning_rate"], 1e-4)

    def test_yaml_curriculum_config_loads(self):
        cfg = load_train_config(
            config_path=__import__("pathlib").Path("configs/train/world_pt_h800_curriculum.yaml")
        )
        self.assertEqual(len(cfg.curriculum), 3)
        self.assertEqual(cfg.curriculum_data_settings_for_step(0)["max_seq_length"], 4096)
        self.assertEqual(cfg.curriculum_data_settings_for_step(20_000)["max_seq_length"], 8192)


    def test_continuous_config_completeness(self):
        from pathlib import Path

        cfg = load_train_config(config_path=Path("configs/train/continuous.yaml"))
        self.assertTrue(cfg.use_flow_tower)
        self.assertEqual(cfg.init_mode, "scratch")
        self.assertEqual(len(cfg.curriculum), 5)
        stages = [p["stage"] for p in cfg.curriculum]
        self.assertEqual(
            stages,
            [
                "world_pt",
                "understanding_warmup",
                "generation_pt",
                "unified_mt",
                "unified_sft",
            ],
        )

        expected_until = [49_999, 249_999, 349_999, 399_999, 419_999]
        for phase, until in zip(cfg.curriculum, expected_until):
            self.assertEqual(int(phase["until_step"]), until)
        self.assertEqual(cfg.max_steps, 420_000)

        world = cfg.curriculum_data_settings_for_step(0)
        self.assertEqual(world["stage"], "world_pt")
        self.assertEqual(world["loss_weights"], {"ce": 0.0, "fm": 0.0})
        self.assertEqual(world["tower_self_cond_prob"], 0.0)

        uw = cfg.curriculum_data_settings_for_step(50_000)
        self.assertEqual(uw["stage"], "understanding_warmup")
        self.assertEqual(uw["loss_weights"], {"ce": 1.0, "fm": 0.0})

        gen = cfg.curriculum_data_settings_for_step(250_000)
        self.assertEqual(gen["stage"], "generation_pt")
        self.assertEqual(gen["task_override"], "t2i")
        self.assertEqual(gen["tower_self_cond_prob"], 0.5)

        mt = cfg.curriculum_data_settings_for_step(350_000)
        self.assertEqual(mt["stage"], "unified_mt")
        self.assertEqual(mt["max_seq_length"], 16_384)
        self.assertEqual(mt["tower_decoder_prob"], 0.2)

        sft = cfg.curriculum_data_settings_for_step(410_000)
        self.assertEqual(sft["stage"], "unified_sft")
        self.assertEqual(sft["gradient_accumulation_steps"], 2)
        self.assertEqual(sft["loss_weights"], {"ce": 1.0, "fm": 0.1})
        self.assertEqual(sft["cfg_label_drop_prob"], 0.10)


if __name__ == "__main__":
    unittest.main()
