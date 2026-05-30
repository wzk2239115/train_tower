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

    def test_yaml_curriculum_config_loads(self):
        cfg = load_train_config(
            config_path=__import__("pathlib").Path("configs/train/world_pt_h800_curriculum.yaml")
        )
        self.assertEqual(len(cfg.curriculum), 3)
        self.assertEqual(cfg.curriculum_data_settings_for_step(0)["max_seq_length"], 4096)
        self.assertEqual(cfg.curriculum_data_settings_for_step(20_000)["max_seq_length"], 8192)


if __name__ == "__main__":
    unittest.main()
