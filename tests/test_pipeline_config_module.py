from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import pytest

from diaremot.pipeline import config as config_mod
from diaremot.pipeline.config import PipelineConfig, build_pipeline_config


def test_build_pipeline_config_merges_and_validates() -> None:
    overrides = {"beam_size": 4, "speaker_limit": None}
    merged = build_pipeline_config(overrides)
    assert merged["beam_size"] == 4
    assert merged["speaker_limit"] is None

    with pytest.raises(ValueError):
        build_pipeline_config({"unknown": 1})

    cfg = PipelineConfig(beam_size=2)
    assert build_pipeline_config(cfg)["beam_size"] == 2


def test_dependency_summary_handles_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_iter() -> Iterator[
        Tuple[str, str, object | None, str | None, Exception | None, Exception | None]
    ]:
        yield ("missing", "1.0", None, None, ImportError("boom"), None)
        yield ("ok", "1.0", object(), "1.2", None, None)

    monkeypatch.setattr(config_mod, "_iter_dependency_status", fake_iter)

    summary = config_mod.dependency_health_summary()
    assert summary["missing"]["status"] == "error"
    assert summary["ok"]["status"] == "ok"


def test_pipeline_config_normalises_path_and_choice_fields(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        registry_path=tmp_path / "registry.json",
        cache_root=str(tmp_path / "cache"),
        log_dir=tmp_path / "logs",
        checkpoint_dir=tmp_path / "checkpoints",
        cache_roots=str(tmp_path / "alt_cache"),
        affect_text_model_dir=str(tmp_path / "text"),
        affect_intent_model_dir=tmp_path / "intent",
        affect_backend="AUTO",
        asr_backend="Faster",
        vad_backend="AUTO",
        loudness_mode="BROADCAST",
        language_mode="EN",
        intent_labels=("A", "B"),
    )

    assert isinstance(cfg.registry_path, Path)
    assert isinstance(cfg.cache_root, Path)
    assert all(isinstance(path, Path) for path in cfg.cache_roots)
    assert cfg.cache_roots == [tmp_path / "alt_cache"]
    assert cfg.affect_text_model_dir == tmp_path / "text"
    assert cfg.affect_intent_model_dir == tmp_path / "intent"
    assert cfg.affect_backend == "auto"
    assert cfg.asr_backend == "faster"
    assert cfg.vad_backend == "auto"
    assert cfg.loudness_mode == "broadcast"
    assert cfg.language_mode == "en"
    assert cfg.intent_labels == ["A", "B"]


def test_pipeline_config_rejects_invalid_formats() -> None:
    with pytest.raises(ValueError):
        PipelineConfig(intent_labels="bad")

    with pytest.raises(ValueError):
        PipelineConfig(vad_threshold=2.0)

    with pytest.raises(ValueError):
        PipelineConfig(cpu_threads=0)
