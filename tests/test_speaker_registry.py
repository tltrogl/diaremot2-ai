import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _install_test_stubs() -> None:
    """Provide light-weight stubs for optional heavy dependencies."""
    if "librosa" not in sys.modules:
        librosa_stub = types.ModuleType("librosa")
        librosa_stub.util = types.SimpleNamespace(
            frame=lambda *args, **kwargs: np.zeros((0, 0), dtype=np.float32)
        )
        sys.modules["librosa"] = librosa_stub

    if "scipy" not in sys.modules:
        scipy_stub = types.ModuleType("scipy")
        sys.modules["scipy"] = scipy_stub
    else:
        scipy_stub = sys.modules["scipy"]

    signal_stub = types.ModuleType("scipy.signal")
    signal_stub.resample_poly = lambda audio, up, down: audio
    setattr(scipy_stub, "signal", signal_stub)
    sys.modules["scipy.signal"] = signal_stub


_install_test_stubs()

from diaremot.pipeline.speaker_diarization import SpeakerRegistry

import diaremot


class SpeakerRegistrySchemaTest(unittest.TestCase):
    def test_legacy_flat_schema_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            original = {
                "Alice": {
                    "centroid": [0.1, 0.0, 0.2],
                    "samples": 2,
                    "last_seen": "before",
                }
            }
            path.write_text(json.dumps(original), encoding="utf-8")

            registry = SpeakerRegistry(str(path))
            self.assertTrue(registry.has("Alice"))

            registry.update_centroid(
                "Alice", np.asarray([0.2, 0.1, 0.3], dtype=np.float32)
            )
            registry.enroll("Bob", np.asarray([0.3, 0.4, 0.5], dtype=np.float32))
            match_name, _ = registry.match(
                np.asarray([0.3, 0.4, 0.5], dtype=np.float32)
            )
            self.assertIn(match_name, {"Alice", "Bob"})

            registry.save()

            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("speakers", saved)
            self.assertIn("Alice", saved)
            self.assertIn("Bob", saved)
            self.assertIsInstance(saved["Bob"].get("last_seen"), str)

    def test_metadata_wrapped_schema_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            original = {
                "version": "1.0.0",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "config": {"max_speakers": 10},
                "metadata": {"total_speakers": 1, "total_sessions": 5},
                "speakers": {
                    "Alice": {
                        "centroid": [0.1, 0.0, 0.2],
                        "samples": 2,
                        "last_seen": "before",
                    }
                },
            }
            path.write_text(json.dumps(original), encoding="utf-8")

            registry = SpeakerRegistry(str(path))
            self.assertTrue(registry.has("Alice"))

            registry.update_centroid(
                "Alice", np.asarray([0.2, 0.1, 0.3], dtype=np.float32)
            )
            registry.enroll("Bob", np.asarray([0.3, 0.4, 0.5], dtype=np.float32))
            _match_name, score = registry.match(
                np.asarray([0.2, 0.1, 0.3], dtype=np.float32)
            )
            self.assertGreaterEqual(score, 0.0)

            registry.save()

            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("speakers", saved)
            self.assertIn("config", saved)
            self.assertEqual(set(saved["speakers"].keys()), {"Alice", "Bob"})
            self.assertIn("total_speakers", saved.get("metadata", {}))
            self.assertEqual(saved["metadata"]["total_speakers"], 2)
            self.assertNotEqual(saved["updated_at"], original["updated_at"])
            self.assertIsInstance(saved["speakers"]["Bob"].get("last_seen"), str)


class RegistryFactoryIntegrationTest(unittest.TestCase):
    def test_factory_creates_persisted_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "factory_registry.json"

            manager = diaremot.get_registry_manager(str(path))
            manager.enroll("Alice", np.asarray([0.3, 0.2, 0.1], dtype=np.float32))
            name, score = manager.match(np.asarray([0.3, 0.2, 0.1], dtype=np.float32))
            self.assertEqual(name, "Alice")
            self.assertGreater(score, 0.0)

            # Ensure persistence works by instantiating a fresh manager.
            reloaded = diaremot.get_registry_manager(str(path))
            match_name, reload_score = reloaded.match(
                np.asarray([0.3, 0.2, 0.1], dtype=np.float32)
            )
            self.assertEqual(match_name, "Alice")
            self.assertGreater(reload_score, 0.0)

            speakers = reloaded.get_speakers()
            self.assertIn("Alice", speakers)


if __name__ == "__main__":
    unittest.main()






