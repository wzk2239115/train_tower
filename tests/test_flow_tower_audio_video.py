from __future__ import annotations

import unittest
from unittest import mock

try:
    import torch
    import torch.nn as nn

    from tower.train.config import TrainConfig
    from tower.unify.flow_tower import FlowJepaTowerTrainModel
    from tower.unify.tower_config import TowerConfig, TowerExitSpec, load_tower_config
    from tower.unify.tower_exits import CeTowerExit, ElfFlowTowerExit

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _make_minimal_tower(**cfg_overrides):
    hidden = 32
    audio_dim = 16
    video_dim = 32

    class _FakeLLM(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.layers = nn.ModuleList([nn.Identity() for _ in range(n)])

    class _FakeConfig:
        class _LLM:
            hidden_size = hidden
            vocab_size = 100
            num_hidden_layers = 8
        llm_config = _LLM()
        patch_size = 4
        downsample_ratio = 0.5
        fm_head_layers = 2
        t_eps = 0.05
        P_mean = -0.8
        P_std = 0.8
        add_noise_scale_embedding = False
        self_cond_prob = 0.0

    class _FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _FakeConfig()
            self.language_model = nn.Module()
            self.language_model.model = _FakeLLM(8)
            self.language_model.get_input_embeddings = lambda: nn.Embedding(100, hidden)
            self.img_context_token_id = 99
            self.audio_context_token_id = 97
            self.video_context_token_id = 98
            self.fm_modules = nn.ModuleDict()

    model = _FakeModel()
    cfg = TrainConfig(
        audio_patch_dim=audio_dim,
        video_patch_dim=video_dim,
        audio_context_token_id=97,
        video_context_token_id=98,
        **cfg_overrides,
    )

    tower = FlowJepaTowerTrainModel.__new__(FlowJepaTowerTrainModel)
    nn.Module.__init__(tower)
    tower.model = model
    tower.cfg = cfg
    tower._tower_global_step = 0

    tower.tower_cfg = load_tower_config()

    tower.audio_und_proj = nn.Sequential(
        nn.Linear(audio_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
    )
    tower.audio_gen_proj = nn.Sequential(
        nn.Linear(audio_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
    )
    tower.video_und_proj = nn.Sequential(
        nn.Linear(video_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
    )
    tower.video_gen_proj = nn.Sequential(
        nn.Linear(video_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
    )
    tower.tower_exits = nn.ModuleDict()
    return tower


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class ProjectorInitTest(unittest.TestCase):
    def test_four_separate_projectors_created(self):
        tower = _make_minimal_tower()
        self.assertIsInstance(tower.audio_und_proj, nn.Sequential)
        self.assertIsInstance(tower.audio_gen_proj, nn.Sequential)
        self.assertIsInstance(tower.video_und_proj, nn.Sequential)
        self.assertIsInstance(tower.video_gen_proj, nn.Sequential)

    def test_audio_projector_input_dim_matches_config(self):
        tower = _make_minimal_tower(audio_patch_dim=16)
        self.assertEqual(tower.audio_und_proj[0].in_features, 16)
        self.assertEqual(tower.audio_gen_proj[0].in_features, 16)

    def test_video_projector_input_dim_matches_config(self):
        tower = _make_minimal_tower(video_patch_dim=32)
        self.assertEqual(tower.video_und_proj[0].in_features, 32)
        self.assertEqual(tower.video_gen_proj[0].in_features, 32)

    def test_und_and_gen_have_different_weights(self):
        tower = _make_minimal_tower()
        for p_und, p_gen in zip(
            tower.audio_und_proj.parameters(), tower.audio_gen_proj.parameters()
        ):
            self.assertFalse(torch.equal(p_und, p_gen))


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class ExtractAudioCleanGenModeTest(unittest.TestCase):
    def test_und_mode_uses_und_proj(self):
        tower = _make_minimal_tower(audio_patch_dim=16)
        audio = torch.randn(4, 16)
        batch = {"audio_values": [audio]}

        with mock.patch.object(tower, "device", torch.device("cpu")), \
             mock.patch.object(tower, "dtype", torch.float32):
            result = tower._extract_audio_clean(batch, gen_mode=False)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[-1], 32)

    def test_gen_mode_uses_gen_proj(self):
        tower = _make_minimal_tower(audio_patch_dim=16)
        audio = torch.randn(4, 16)
        batch = {"audio_values": [audio]}

        with mock.patch.object(tower, "device", torch.device("cpu")), \
             mock.patch.object(tower, "dtype", torch.float32):
            result_und = tower._extract_audio_clean(batch, gen_mode=False)
            result_gen = tower._extract_audio_clean(batch, gen_mode=True)
        self.assertFalse(torch.equal(result_und, result_gen))

    def test_returns_none_when_no_audio(self):
        tower = _make_minimal_tower()
        with mock.patch.object(tower, "device", torch.device("cpu")), \
             mock.patch.object(tower, "dtype", torch.float32):
            result = tower._extract_audio_clean({}, gen_mode=False)
        self.assertIsNone(result)


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class ExtractVideoCleanGenModeTest(unittest.TestCase):
    def test_und_mode_uses_und_proj(self):
        tower = _make_minimal_tower(video_patch_dim=32)
        video = torch.randn(8, 32)
        batch = {"video_values": [video]}

        with mock.patch.object(tower, "device", torch.device("cpu")), \
             mock.patch.object(tower, "dtype", torch.float32):
            result = tower._extract_video_clean(batch, gen_mode=False)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[-1], 32)

    def test_gen_mode_uses_gen_proj(self):
        tower = _make_minimal_tower(video_patch_dim=32)
        video = torch.randn(8, 32)
        batch = {"video_values": [video]}

        with mock.patch.object(tower, "device", torch.device("cpu")), \
             mock.patch.object(tower, "dtype", torch.float32):
            result_und = tower._extract_video_clean(batch, gen_mode=False)
            result_gen = tower._extract_video_clean(batch, gen_mode=True)
        self.assertFalse(torch.equal(result_und, result_gen))


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class ExtractRawAudioTest(unittest.TestCase):
    def test_returns_raw_patches_dim_80(self):
        tower = _make_minimal_tower(audio_patch_dim=80)
        audio = torch.randn(10, 80)
        batch = {"audio_values": [audio]}

        with mock.patch.object(tower, "device", torch.device("cpu")), \
             mock.patch.object(tower, "dtype", torch.float32):
            result = tower._extract_raw_audio(batch)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (10, 80))

    def test_pads_short_dim(self):
        tower = _make_minimal_tower(audio_patch_dim=80)
        audio = torch.randn(5, 40)
        batch = {"audio_values": [audio]}

        with mock.patch.object(tower, "device", torch.device("cpu")), \
             mock.patch.object(tower, "dtype", torch.float32):
            result = tower._extract_raw_audio(batch)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[-1], 80)

    def test_truncates_long_dim(self):
        tower = _make_minimal_tower(audio_patch_dim=80)
        audio = torch.randn(5, 120)
        batch = {"audio_values": [audio]}

        with mock.patch.object(tower, "device", torch.device("cpu")), \
             mock.patch.object(tower, "dtype", torch.float32):
            result = tower._extract_raw_audio(batch)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[-1], 80)

    def test_returns_none_for_missing(self):
        tower = _make_minimal_tower()
        with mock.patch.object(tower, "device", torch.device("cpu")), \
             mock.patch.object(tower, "dtype", torch.float32):
            result = tower._extract_raw_audio({})
        self.assertIsNone(result)


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class ExtractRawVideoTest(unittest.TestCase):
    def test_returns_raw_patches_dim_1024(self):
        tower = _make_minimal_tower(video_patch_dim=1024)
        video = torch.randn(16, 1024)
        batch = {"video_values": [video]}

        with mock.patch.object(tower, "device", torch.device("cpu")), \
             mock.patch.object(tower, "dtype", torch.float32):
            result = tower._extract_raw_video(batch)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (16, 1024))

    def test_pads_short_dim(self):
        tower = _make_minimal_tower(video_patch_dim=1024)
        video = torch.randn(8, 512)
        batch = {"video_values": [video]}

        with mock.patch.object(tower, "device", torch.device("cpu")), \
             mock.patch.object(tower, "dtype", torch.float32):
            result = tower._extract_raw_video(batch)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[-1], 1024)


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class LatentBundleTest(unittest.TestCase):
    def _make_ctx(self, **overrides):
        ctx = {
            "z_world": torch.randn(4, 32),
            "t_world": torch.rand(4),
            "clean_embed_und": torch.randn(4, 32),
            "z_embed": torch.randn(4, 32),
            "t_embed": torch.rand(4),
            "clean_embed_gen": torch.randn(4, 32),
            "z_pixel": torch.randn(4, 48),
            "t_pixel": torch.rand(4),
            "clean_pixel": torch.randn(4, 48),
            "z_audio": torch.randn(6, 32),
            "t_audio": torch.rand(6),
            "clean_audio": torch.randn(6, 32),
            "z_audio_raw": torch.randn(6, 16),
            "t_audio_raw": torch.rand(6),
            "clean_audio_raw": torch.randn(6, 16),
            "z_video": torch.randn(8, 32),
            "t_video": torch.rand(8),
            "clean_video": torch.randn(8, 32),
            "z_video_raw": torch.randn(8, 32),
            "t_video_raw": torch.rand(8),
            "clean_video_raw": torch.randn(8, 32),
            "z_text": torch.randn(5, 32),
            "t_text": torch.rand(5),
            "clean_text": torch.randn(5, 32),
        }
        ctx.update(overrides)
        return ctx

    def test_audio_patch_returns_raw_space(self):
        tower = _make_minimal_tower()
        ctx = self._make_ctx()
        spec = TowerExitSpec(
            name="audio_elf", after_layer=11, exit_type="elf_fm", latent="audio_patch",
        )
        bundle = tower._latent_bundle(spec, ctx)
        self.assertIsNotNone(bundle)
        z, t, clean = bundle
        self.assertEqual(z.shape[-1], 16)
        self.assertEqual(clean.shape[-1], 16)

    def test_video_patch_returns_raw_space(self):
        tower = _make_minimal_tower()
        ctx = self._make_ctx()
        spec = TowerExitSpec(
            name="video_elf", after_layer=13, exit_type="elf_fm", latent="video_patch",
        )
        bundle = tower._latent_bundle(spec, ctx)
        self.assertIsNotNone(bundle)
        z, t, clean = bundle
        self.assertEqual(z.shape[-1], 32)
        self.assertEqual(clean.shape[-1], 32)

    def test_vision_embed_und_bundle(self):
        tower = _make_minimal_tower()
        ctx = self._make_ctx()
        spec = TowerExitSpec(
            name="world_elf", after_layer=7, exit_type="jepa", latent="vision_embed_und",
        )
        bundle = tower._latent_bundle(spec, ctx)
        self.assertIsNotNone(bundle)
        z, t, clean = bundle
        self.assertEqual(z.shape[-1], 32)

    def test_pixel_patch_bundle(self):
        tower = _make_minimal_tower()
        ctx = self._make_ctx()
        spec = TowerExitSpec(
            name="generative_elf", after_layer=25, exit_type="elf_fm", latent="pixel_patch",
        )
        bundle = tower._latent_bundle(spec, ctx)
        self.assertIsNotNone(bundle)
        z, t, clean = bundle
        self.assertEqual(z.shape[-1], 48)

    def test_token_hidden_bundle(self):
        tower = _make_minimal_tower()
        ctx = self._make_ctx()
        spec = TowerExitSpec(
            name="understanding_elf", after_layer=21, exit_type="elf_fm", latent="token_hidden",
        )
        bundle = tower._latent_bundle(spec, ctx)
        self.assertIsNotNone(bundle)

    def test_returns_none_when_missing_data(self):
        tower = _make_minimal_tower()
        ctx = {}
        spec = TowerExitSpec(
            name="audio_elf", after_layer=11, exit_type="elf_fm", latent="audio_patch",
        )
        self.assertIsNone(tower._latent_bundle(spec, ctx))


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class SelectHiddenForExitTest(unittest.TestCase):
    def _make_ctx(self, seq_len=20):
        audio_mask = torch.zeros(seq_len, dtype=torch.bool)
        audio_mask[3:9] = True
        video_mask = torch.zeros(seq_len, dtype=torch.bool)
        video_mask[10:18] = True
        text_mask = torch.zeros(seq_len, dtype=torch.bool)
        text_mask[:3] = True
        text_mask[9:10] = True
        text_mask[18:] = True
        selected = torch.zeros(seq_len, dtype=torch.bool)
        selected[0:2] = True
        return {
            "audio_mask": audio_mask,
            "video_mask": video_mask,
            "text_mask": text_mask,
            "selected": selected,
        }

    def test_audio_patch_selects_audio_tokens(self):
        tower = _make_minimal_tower()
        hook = torch.randn(1, 20, 32)
        ctx = self._make_ctx()
        spec = TowerExitSpec(
            name="audio_elf", after_layer=11, exit_type="elf_fm", latent="audio_patch",
        )
        h = tower._select_hidden_for_exit(spec, hook, ctx)
        self.assertEqual(h.shape[0], 6)

    def test_video_patch_selects_video_tokens(self):
        tower = _make_minimal_tower()
        hook = torch.randn(1, 20, 32)
        ctx = self._make_ctx()
        spec = TowerExitSpec(
            name="video_elf", after_layer=13, exit_type="elf_fm", latent="video_patch",
        )
        h = tower._select_hidden_for_exit(spec, hook, ctx)
        self.assertEqual(h.shape[0], 8)

    def test_token_hidden_selects_text_tokens(self):
        tower = _make_minimal_tower()
        hook = torch.randn(1, 20, 32)
        ctx = self._make_ctx()
        spec = TowerExitSpec(
            name="understanding_elf", after_layer=21, exit_type="ce", latent="token_hidden",
        )
        h = tower._select_hidden_for_exit(spec, hook, ctx)
        self.assertEqual(h.shape[0], 5)


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class ComputeExitLossCETest(unittest.TestCase):
    def test_ce_exit_uses_labels_from_ctx(self):
        tower = _make_minimal_tower()
        hidden_size = 32
        vocab_size = 100
        tower.tower_exits["understanding_elf"] = CeTowerExit(hidden_size, vocab_size)

        seq_len = 10
        text_mask = torch.zeros(seq_len, dtype=torch.bool)
        text_mask[2:6] = True
        labels = torch.full((1, seq_len), -100, dtype=torch.long)
        labels[0, 2:6] = torch.tensor([5, 10, 3, 7])

        hook = torch.randn(1, seq_len, hidden_size)
        ctx = {
            "labels": labels,
            "text_mask": text_mask,
            "model_cfg": tower.model.config,
        }
        spec = TowerExitSpec(
            name="understanding_elf", after_layer=21, exit_type="ce", latent="token_hidden",
        )
        loss = tower._compute_exit_loss(spec, hook, ctx)
        self.assertEqual(loss.shape, ())
        self.assertGreater(loss.item(), 0.0)

    def test_ce_exit_no_labels_returns_zero(self):
        tower = _make_minimal_tower()
        tower.tower_exits["understanding_elf"] = CeTowerExit(32, 100)
        hook = torch.randn(1, 10, 32)
        ctx = {
            "labels": None,
            "text_mask": torch.ones(10, dtype=torch.bool),
            "model_cfg": tower.model.config,
        }
        spec = TowerExitSpec(
            name="understanding_elf", after_layer=21, exit_type="ce", latent="token_hidden",
        )
        loss = tower._compute_exit_loss(spec, hook, ctx)
        self.assertEqual(loss.item(), 0.0)


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class ComputeExitLossAudioPatchFMTest(unittest.TestCase):
    def test_audio_patch_fm_loss(self):
        tower = _make_minimal_tower(audio_patch_dim=16)
        hidden_size = 32
        audio_dim = 16

        tower.tower_exits["audio_elf"] = ElfFlowTowerExit(
            hidden_size, audio_dim, elf_depth=1,
        )

        n_audio = 6
        seq_len = 12
        audio_mask = torch.zeros(seq_len, dtype=torch.bool)
        audio_mask[3:3 + n_audio] = True

        hook = torch.randn(1, seq_len, hidden_size)
        ctx = {
            "z_audio_raw": torch.randn(n_audio, audio_dim),
            "t_audio_raw": torch.rand(n_audio),
            "clean_audio_raw": torch.randn(n_audio, audio_dim),
            "audio_mask": audio_mask,
            "model_cfg": tower.model.config,
            "noise_scale": 1.0,
        }
        spec = TowerExitSpec(
            name="audio_elf", after_layer=11, exit_type="elf_fm", latent="audio_patch",
        )
        loss = tower._compute_exit_loss(spec, hook, ctx)
        self.assertEqual(loss.shape, ())
        self.assertGreater(loss.item(), 0.0)
        self.assertTrue(loss.requires_grad)


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class BatchHasAudioVideoTest(unittest.TestCase):
    def test_batch_has_audio_true(self):
        tower = _make_minimal_tower()
        batch = {"audio_values": [torch.randn(4, 16)]}
        self.assertTrue(tower._batch_has_audio(batch))

    def test_batch_has_audio_false_empty(self):
        tower = _make_minimal_tower()
        self.assertFalse(tower._batch_has_audio({}))
        self.assertFalse(tower._batch_has_audio({"audio_values": []}))
        self.assertFalse(tower._batch_has_audio({"audio_values": [None]}))

    def test_batch_has_video_true(self):
        tower = _make_minimal_tower()
        batch = {"video_values": [torch.randn(8, 32)]}
        self.assertTrue(tower._batch_has_video(batch))

    def test_batch_has_video_false_empty(self):
        tower = _make_minimal_tower()
        self.assertFalse(tower._batch_has_video({}))
        self.assertFalse(tower._batch_has_video({"video_values": []}))
        self.assertFalse(tower._batch_has_video({"video_values": [None]}))


if __name__ == "__main__":
    unittest.main()
