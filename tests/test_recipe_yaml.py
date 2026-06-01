from __future__ import annotations

import unittest
from pathlib import Path

import yaml

RECIPE_ROOT = Path(__file__).resolve().parents[1] / "recipe"

REQUIRED_STAGE_FIELDS = {"stage", "description"}
REQUIRED_POOL_FIELDS = {"pool_id", "datasets"}
REQUIRED_SCALE_FIELDS = {"scale", "model_config_path"}
REQUIRED_CURRICULUM_ENTRY_FIELDS = {"stage", "until_step"}


class RecipeYamlSyntaxTest(unittest.TestCase):
    def test_all_pool_yamls_parse(self):
        pool_dir = RECIPE_ROOT / "pools"
        self.assertTrue(pool_dir.is_dir(), f"Missing pool dir: {pool_dir}")
        yamls = sorted(pool_dir.glob("*.yaml"))
        self.assertGreater(len(yamls), 0, "No pool YAML files found")
        for path in yamls:
            with self.subTest(pool=path.name):
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertIsInstance(data, dict)

    def test_all_stage_yamls_parse(self):
        stage_dir = RECIPE_ROOT / "stages"
        self.assertTrue(stage_dir.is_dir(), f"Missing stage dir: {stage_dir}")
        yamls = sorted(stage_dir.glob("*.yaml"))
        self.assertGreater(len(yamls), 0, "No stage YAML files found")
        for path in yamls:
            with self.subTest(stage=path.name):
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertIsInstance(data, dict)


class PoolYamlSchemaTest(unittest.TestCase):
    def _pools(self):
        pool_dir = RECIPE_ROOT / "pools"
        return sorted(pool_dir.glob("*.yaml"))

    def test_pools_have_required_fields(self):
        for path in self._pools():
            with self.subTest(pool=path.name):
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertIn("pool_id", data, f"{path.name} missing 'pool_id'")
                has_datasets = "datasets" in data
                has_sub_pools = "sub_pools" in data
                self.assertTrue(
                    has_datasets or has_sub_pools,
                    f"{path.name} must have 'datasets' or 'sub_pools'",
                )

    def test_datasets_are_list_of_dicts(self):
        for path in self._pools():
            with self.subTest(pool=path.name):
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                datasets = data.get("datasets", [])
                if not datasets and "sub_pools" in data:
                    for sp_name, sp in data["sub_pools"].items():
                        sp_datasets = sp.get("datasets", [])
                        self.assertIsInstance(sp_datasets, list)
                        for ds in sp_datasets:
                            self.assertIsInstance(ds, dict)
                            self.assertIn("name", ds)
                    continue
                self.assertIsInstance(datasets, list)
                for ds in datasets:
                    self.assertIsInstance(ds, dict)
                    self.assertIn("name", ds, f"Dataset in {path.name} missing 'name'")

    def test_pool_ids_match_filenames(self):
        for path in self._pools():
            with self.subTest(pool=path.name):
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                pool_id = data.get("pool_id", "")
                expected = path.stem.split("_", 1)[-1] if "_" in path.stem else path.stem
                self.assertEqual(pool_id, expected,
                                 f"pool_id '{pool_id}' doesn't match file stem '{expected}'")


class StageYamlSchemaTest(unittest.TestCase):
    def _stages(self):
        stage_dir = RECIPE_ROOT / "stages"
        return sorted(stage_dir.glob("*.yaml"))

    def test_stages_have_required_fields(self):
        for path in self._stages():
            with self.subTest(stage=path.name):
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertIn("stage", data, f"{path.name} missing 'stage'")
                self.assertIn("description", data, f"{path.name} missing 'description'")

    def test_hyperparams_section_exists(self):
        for path in self._stages():
            with self.subTest(stage=path.name):
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertIn("hyperparams", data, f"{path.name} missing 'hyperparams'")

    def test_hyperparams_have_required_keys(self):
        required = {"learning_rate", "max_steps", "max_seq_length", "per_device_train_batch_size"}
        for path in self._stages():
            with self.subTest(stage=path.name):
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                hp = data.get("hyperparams", {})
                for key in required:
                    self.assertIn(key, hp, f"{path.name} hyperparams missing '{key}'")


class StageCurriculumSchemaTest(unittest.TestCase):
    def test_unified_mt_curriculum_exists(self):
        path = RECIPE_ROOT / "stages" / "unified_mt.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        curriculum = data.get("curriculum")
        self.assertIsNotNone(curriculum, "unified_mt.yaml missing 'curriculum'")
        self.assertIsInstance(curriculum, list)
        self.assertGreaterEqual(len(curriculum), 2, "unified_mt curriculum needs >= 2 phases")

    def test_unified_sft_curriculum_exists(self):
        path = RECIPE_ROOT / "stages" / "unified_sft.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        curriculum = data.get("curriculum")
        self.assertIsNotNone(curriculum, "unified_sft.yaml missing 'curriculum'")
        self.assertIsInstance(curriculum, list)
        self.assertGreaterEqual(len(curriculum), 2)

    def test_curriculum_entries_have_required_fields(self):
        for stage_file in ["unified_mt.yaml", "unified_sft.yaml"]:
            path = RECIPE_ROOT / "stages" / stage_file
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            curriculum = data.get("curriculum", [])
            for i, entry in enumerate(curriculum):
                with self.subTest(stage=stage_file, phase=i):
                    for field in REQUIRED_CURRICULUM_ENTRY_FIELDS:
                        self.assertIn(field, entry, f"Phase {i} in {stage_file} missing '{field}'")

    def test_curriculum_until_steps_are_ascending(self):
        for stage_file in ["unified_mt.yaml", "unified_sft.yaml"]:
            path = RECIPE_ROOT / "stages" / stage_file
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            curriculum = data.get("curriculum", [])
            steps = [entry["until_step"] for entry in curriculum]
            for i in range(1, len(steps)):
                with self.subTest(stage=stage_file, phase=i):
                    self.assertGreater(steps[i], steps[i - 1],
                                       f"until_step not ascending in {stage_file}")

    def test_unified_mt_curriculum_has_new_fields(self):
        path = RECIPE_ROOT / "stages" / "unified_mt.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        curriculum = data.get("curriculum", [])
        new_fields = ["max_audio_duration_ms", "max_video_frames_gen", "audio_cfg_scale", "video_cfg_scale"]
        for i, entry in enumerate(curriculum):
            with self.subTest(phase=i):
                for field in new_fields:
                    self.assertIn(field, entry, f"Phase {i} missing '{field}'")

    def test_unified_sft_curriculum_has_new_fields(self):
        path = RECIPE_ROOT / "stages" / "unified_sft.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        curriculum = data.get("curriculum", [])
        new_fields = ["max_audio_duration_ms", "max_video_frames_gen", "audio_cfg_scale", "video_cfg_scale"]
        for i, entry in enumerate(curriculum):
            with self.subTest(phase=i):
                for field in new_fields:
                    self.assertIn(field, entry, f"Phase {i} missing '{field}'")

    def test_tower_section_has_new_fields(self):
        for stage_file in ["unified_mt.yaml", "unified_sft.yaml"]:
            path = RECIPE_ROOT / "stages" / stage_file
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            tower = data.get("tower", {})
            with self.subTest(stage=stage_file):
                self.assertIn("cfg_audio_drop_prob", tower, f"{stage_file} tower missing cfg_audio_drop_prob")
                self.assertIn("cfg_video_drop_prob", tower, f"{stage_file} tower missing cfg_video_drop_prob")
                self.assertIn("audio_cfg_scale", tower, f"{stage_file} tower missing audio_cfg_scale")
                self.assertIn("video_cfg_scale", tower, f"{stage_file} tower missing video_cfg_scale")
                self.assertIn("grad_norm_balance", tower, f"{stage_file} tower missing grad_norm_balance")
                self.assertIn("max_audio_duration_ms", tower, f"{stage_file} tower missing max_audio_duration_ms")
                self.assertIn("max_video_frames_gen", tower, f"{stage_file} tower missing max_video_frames_gen")


class TowerYamlTest(unittest.TestCase):
    def test_tower_yml_has_ce_exit(self):
        path = Path(__file__).resolve().parents[1] / "note" / "tower.yml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        exits = data.get("exits", {})
        understanding = exits.get("understanding_elf", {})
        self.assertEqual(understanding.get("type"), "ce",
                         "understanding_elf should have type=ce")

    def test_tower_yml_audio_patch_latent(self):
        path = Path(__file__).resolve().parents[1] / "note" / "tower.yml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        exits = data.get("exits", {})
        audio = exits.get("audio_elf", {})
        self.assertEqual(audio.get("latent"), "audio_patch",
                         "audio_elf should have latent=audio_patch")

    def test_tower_yml_video_patch_latent(self):
        path = Path(__file__).resolve().parents[1] / "note" / "tower.yml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        exits = data.get("exits", {})
        video = exits.get("video_elf", {})
        self.assertEqual(video.get("latent"), "video_patch",
                         "video_elf should have latent=video_patch")


class ScaleYamlTest(unittest.TestCase):
    def test_scale_yamls_parse(self):
        scale_dir = RECIPE_ROOT / "scales"
        if not scale_dir.is_dir():
            self.skipTest("No scales directory")
        yamls = sorted(scale_dir.glob("*.yaml"))
        schema = [y for y in yamls if y.name != "_schema.yaml"]
        self.assertGreater(len(schema), 0, "No scale YAML files found")
        for path in schema:
            with self.subTest(scale=path.name):
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertIn("scale", data, f"{path.name} missing 'scale'")


if __name__ == "__main__":
    unittest.main()
