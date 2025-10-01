import builtins
from pathlib import Path

import pytest

from diaremot.pipeline import pipeline_checkpoint_system as pcs


@pytest.fixture()
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.wav"
    path.write_bytes(b"fake audio")
    return path


def _counting_open_factory(target: Path, original_open):
    def _wrapped(file, mode="r", *args, **kwargs):
        if Path(file) == target and "r" in mode and "b" in mode:
            _wrapped.calls += 1
        return original_open(file, mode, *args, **kwargs)

    _wrapped.calls = 0
    return _wrapped


def test_checkpoint_manager_reuses_cached_hash(monkeypatch, tmp_path, audio_file):
    manager = pcs.PipelineCheckpointManager(tmp_path / "checkpoints")
    counting_open = _counting_open_factory(audio_file, builtins.open)
    monkeypatch.setattr(pcs, "open", counting_open, raising=False)

    manager.create_checkpoint(
        str(audio_file),
        pcs.ProcessingStage.TRANSCRIPTION,
        {"payload": 1},
        progress=10.0,
    )
    manager.create_checkpoint(
        str(audio_file),
        pcs.ProcessingStage.DIARIZATION,
        {"payload": 2},
        progress=20.0,
    )

    assert counting_open.calls == 1


def test_seeded_hash_prevents_disk_reads(monkeypatch, tmp_path, audio_file):
    manager = pcs.PipelineCheckpointManager(tmp_path / "checkpoints")
    counting_open = _counting_open_factory(audio_file, builtins.open)
    monkeypatch.setattr(pcs, "open", counting_open, raising=False)

    seed = "0123456789abcdef0123456789abcdef"
    manager.seed_file_hash(audio_file, seed)
    manager.create_checkpoint(
        str(audio_file),
        pcs.ProcessingStage.TRANSCRIPTION,
        {"payload": 3},
        progress=5.0,
        file_hash=seed,
    )

    assert counting_open.calls == 0






