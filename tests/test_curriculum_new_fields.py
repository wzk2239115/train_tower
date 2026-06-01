from __future__ import annotations

import unittest

from tower.train.config import TrainConfig, CURRICULUM_TRAIN_OVERRIDE_KEYS

try:
    from tower.train.curriculum import CurriculumRuntime

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class NewConfigFieldsTest(unittest.TestCase):
    def test_default_audio_cfg_drop_prob(self):
        cfg = TrainConfig()
        self.assertEqual(cfg.cfg_audio_drop_prob, 0.0)

    def test_default_video_cfg_drop_prob(self):
        cfg = TrainConfig()
        self.assertEqual(cfg.cfg_video_drop_prob, 0.0)

    def test_default_audio_cfg_scale(self):
        cfg = TrainConfig()
        self.assertEqual(cfg.audio_cfg_scale, 1.0)

    def test_default_video_cfg_scale(self):
        cfg = TrainConfig()
        self.assertEqual(cfg.video_cfg_scale, 1.0)

    def test_default_grad_norm_balance(self):
        cfg = TrainConfig()
        self.assertFalse(cfg.grad_norm_balance)

    def test_default_grad_norm_target(self):
        cfg = TrainConfig()
        self.assertEqual(cfg.grad_norm_target, 1.0)

    def test_default_grad_norm_update_interval(self):
        cfg = TrainConfig()
        self.assertEqual(cfg.grad_norm_update_interval, 100)

    def test_default_max_audio_duration_ms(self):
        cfg = TrainConfig()
        self.assertEqual(cfg.max_audio_duration_ms, 5000)

    def test_default_max_video_frames_gen(self):
        cfg = TrainConfig()
        self.assertEqual(cfg.max_video_frames_gen, 16)

    def test_custom_values(self):
        cfg = TrainConfig(
            cfg_audio_drop_prob=0.1,
            cfg_video_drop_prob=0.15,
            audio_cfg_scale=1.5,
            video_cfg_scale=2.0,
            grad_norm_balance=True,
            grad_norm_target=0.5,
            grad_norm_update_interval=50,
            max_audio_duration_ms=10000,
            max_video_frames_gen=32,
        )
        self.assertEqual(cfg.cfg_audio_drop_prob, 0.1)
        self.assertEqual(cfg.cfg_video_drop_prob, 0.15)
        self.assertEqual(cfg.audio_cfg_scale, 1.5)
        self.assertEqual(cfg.video_cfg_scale, 2.0)
        self.assertTrue(cfg.grad_norm_balance)
        self.assertEqual(cfg.grad_norm_target, 0.5)
        self.assertEqual(cfg.grad_norm_update_interval, 50)
        self.assertEqual(cfg.max_audio_duration_ms, 10000)
        self.assertEqual(cfg.max_video_frames_gen, 32)


class CurriculumOverrideKeysTest(unittest.TestCase):
    def test_all_new_fields_in_override_keys(self):
        new_fields = [
            "cfg_audio_drop_prob",
            "cfg_video_drop_prob",
            "audio_cfg_scale",
            "video_cfg_scale",
            "grad_norm_balance",
            "grad_norm_target",
            "grad_norm_update_interval",
            "max_audio_duration_ms",
            "max_video_frames_gen",
        ]
        for field in new_fields:
            self.assertIn(field, CURRICULUM_TRAIN_OVERRIDE_KEYS, f"{field} missing from override keys")

    def test_old_fields_still_present(self):
        old_fields = [
            "tower_decoder_prob",
            "cfg_label_drop_prob",
            "tower_self_cond_prob",
            "tower_self_cond_cfg_min",
            "tower_self_cond_cfg_max",
        ]
        for field in old_fields:
            self.assertIn(field, CURRICULUM_TRAIN_OVERRIDE_KEYS, f"{field} was removed")


class CurriculumPhaseOverrideTest(unittest.TestCase):
    def test_new_fields_resolved_from_curriculum(self):
        cfg = TrainConfig(
            stage="unified_mt",
            curriculum=[
                {
                    "stage": "mt_warmup",
                    "until_step": 100,
                    "max_audio_duration_ms": 3000,
                    "max_video_frames_gen": 8,
                    "audio_cfg_scale": 1.0,
                    "video_cfg_scale": 1.0,
                    "grad_norm_balance": False,
                },
                {
                    "stage": "mt_main",
                    "until_step": 999,
                    "max_audio_duration_ms": 5000,
                    "max_video_frames_gen": 16,
                    "audio_cfg_scale": 1.5,
                    "video_cfg_scale": 1.5,
                    "grad_norm_balance": True,
                    "grad_norm_update_interval": 100,
                },
            ],
        )
        early = cfg.curriculum_data_settings_for_step(50)
        self.assertEqual(early["max_audio_duration_ms"], 3000)
        self.assertEqual(early["max_video_frames_gen"], 8)
        self.assertEqual(early["audio_cfg_scale"], 1.0)
        self.assertEqual(early["grad_norm_balance"], False)

        late = cfg.curriculum_data_settings_for_step(500)
        self.assertEqual(late["max_audio_duration_ms"], 5000)
        self.assertEqual(late["max_video_frames_gen"], 16)
        self.assertEqual(late["audio_cfg_scale"], 1.5)
        self.assertEqual(late["grad_norm_balance"], True)

    def test_type_preservation_in_curriculum(self):
        cfg = TrainConfig(
            stage="test",
            curriculum=[
                {
                    "stage": "test",
                    "until_step": 99,
                    "grad_norm_balance": True,
                    "grad_norm_update_interval": 50,
                    "max_audio_duration_ms": 8000,
                    "audio_cfg_scale": 2.0,
                },
            ],
        )
        settings = cfg.curriculum_data_settings_for_step(0)
        self.assertIsInstance(settings["grad_norm_balance"], bool)
        self.assertIsInstance(settings["grad_norm_update_interval"], int)
        self.assertIsInstance(settings["max_audio_duration_ms"], int)
        self.assertIsInstance(settings["audio_cfg_scale"], float)


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class CurriculumSyncTypeAwareTest(unittest.TestCase):
    def test_sync_converts_types_correctly(self):
        cfg = TrainConfig(
            stage="test",
            curriculum=[
                {
                    "stage": "test",
                    "until_step": 99,
                    "grad_norm_balance": "true",
                    "grad_norm_update_interval": "25",
                    "max_audio_duration_ms": "6000",
                    "audio_cfg_scale": "1.5",
                },
            ],
        )

        runtime = CurriculumRuntime(cfg=cfg)

        class _FakeDataArgs:
            max_seq_length = 8192
            max_pixels = 262144
            min_pixels = 12544

        class _FakeTrainingArgs:
            per_device_train_batch_size = 1
            gradient_accumulation_steps = 1
            learning_rate = 1e-4

        data_args = _FakeDataArgs()
        training_args = _FakeTrainingArgs()

        runtime.sync(0, data_args=data_args, training_args=training_args, tokenizer=None)

        self.assertIsInstance(cfg.grad_norm_balance, bool)
        self.assertIsInstance(cfg.grad_norm_update_interval, int)
        self.assertIsInstance(cfg.max_audio_duration_ms, int)
        self.assertIsInstance(cfg.audio_cfg_scale, float)

    def test_sync_int_fields_remain_int(self):
        cfg = TrainConfig(
            stage="test",
            curriculum=[
                {
                    "stage": "test",
                    "until_step": 99,
                    "max_video_frames_gen": 24,
                    "grad_norm_update_interval": 50,
                },
            ],
        )

        runtime = CurriculumRuntime(cfg=cfg)

        class _FakeDataArgs:
            max_seq_length = 8192
            max_pixels = 262144
            min_pixels = 12544

        class _FakeTrainingArgs:
            per_device_train_batch_size = 1
            gradient_accumulation_steps = 1
            learning_rate = 1e-4

        runtime.sync(0, data_args=_FakeDataArgs(), training_args=_FakeTrainingArgs(), tokenizer=None)
        self.assertIsInstance(cfg.max_video_frames_gen, int)
        self.assertIsInstance(cfg.grad_norm_update_interval, int)
        self.assertEqual(cfg.max_video_frames_gen, 24)


if __name__ == "__main__":
    unittest.main()
