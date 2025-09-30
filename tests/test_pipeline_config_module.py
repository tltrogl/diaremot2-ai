from __future__ import annotations

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
