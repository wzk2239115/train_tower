from __future__ import annotations

import tempfile
import unittest
from unittest import mock

try:
    import torch
    import torch.nn as nn

    from tower.unify.export import HEAD_EXPORT_MAP, export_multi_artifacts

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class ExportArtifactsTest(unittest.TestCase):
    def test_export_multi_artifacts_writes_all_heads(self):
        hidden = 8
        backbone = nn.Linear(hidden, hidden, bias=False)
        tower_exits = nn.ModuleDict(
            {name: nn.Linear(hidden, hidden, bias=False) for name in HEAD_EXPORT_MAP.values()}
        )

        model = mock.Mock()
        model.model = backbone
        model.tower_exits = tower_exits

        with tempfile.TemporaryDirectory() as tmp:
            export_dir = export_multi_artifacts(model, tmp)
            self.assertTrue((export_dir / "backbone.pt").is_file())
            for filename in HEAD_EXPORT_MAP:
                path = export_dir / filename
                self.assertTrue(path.is_file(), msg=filename)
                payload = torch.load(path, map_location="cpu", weights_only=False)
                self.assertIn("exit_name", payload)
                self.assertIn("state_dict", payload)


if __name__ == "__main__":
    unittest.main()
