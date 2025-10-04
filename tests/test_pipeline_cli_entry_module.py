from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pytest

from diaremot.pipeline import cli_entry


def test_args_to_config_includes_flags() -> None:
    parser = cli_entry._build_arg_parser()
    args = parser.parse_args(
        [
            "--input",
            "sample.wav",
            "--outdir",
            "out",
            "--beam-size",
            "3",
            "--no-speech-threshold",
            "0.25",
        ]
    )
    config = cli_entry._args_to_config(args, ignore_tx_cache=True)
    assert config["beam_size"] == 3
    assert config["no_speech_threshold"] == 0.25
    assert config["ignore_tx_cache"] is True


def test_args_to_config_handles_chunk_toggle() -> None:
    parser = cli_entry._build_arg_parser()
    args = parser.parse_args(
        [
            "--input",
            "sample.wav",
            "--outdir",
            "out",
            "--no-chunk-enabled",
        ]
    )
    config = cli_entry._args_to_config(args, ignore_tx_cache=False)
    assert config["auto_chunk_enabled"] is False


def test_args_to_config_defaults_preserve_chunking() -> None:
    parser = cli_entry._build_arg_parser()
    args = parser.parse_args(
        [
            "--input",
            "sample.wav",
            "--outdir",
            "out",
        ]
    )
    config = cli_entry._args_to_config(args, ignore_tx_cache=False)
    assert config["auto_chunk_enabled"] is True


def test_args_to_config_accepts_affect_backend_auto() -> None:
    parser = cli_entry._build_arg_parser()
    args = parser.parse_args(
        [
            "--input",
            "sample.wav",
            "--outdir",
            "out",
            "--affect-backend",
            "auto",
        ]
    )
    config = cli_entry._args_to_config(args, ignore_tx_cache=False)
    assert config["affect_backend"] == "auto"


def test_main_verify_deps(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    called = {}

    def fake_verify(strict: bool) -> tuple[bool, list[str]]:
        called["strict"] = strict
        return True, []

    monkeypatch.setattr(cli_entry, "config_verify_dependencies", fake_verify)
    exit_code = cli_entry.main(["--verify_deps", "--strict_dependency_versions"])
    assert exit_code == 0
    assert called["strict"] is True
    captured = capsys.readouterr()
    assert "All core dependencies" in captured.out


def test_main_runs_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any) -> None:
    audio = tmp_path / "call.wav"
    audio.write_text("fake")
    outdir = tmp_path / "out"

    recorded = {}

    class DummyPipeline:
        def __init__(self, config: dict[str, Any]):
            recorded["config"] = config

        def process_audio_file(self, input_path: str, out_dir: str) -> dict[str, Any]:
            recorded["input"] = input_path
            recorded["out"] = out_dir
            return {"status": "ok"}

    monkeypatch.setattr(cli_entry, "AudioAnalysisPipelineV2", DummyPipeline)

    exit_code = cli_entry.main(
        [
            "--input",
            str(audio),
            "--outdir",
            str(outdir),
            "--clear-cache",
        ]
    )

    assert exit_code == 0
    raw_output = capsys.readouterr().out.splitlines()
    json_lines: list[str] = []
    started = False
    for line in raw_output:
        stripped = line.strip()
        if not started and stripped.startswith("{"):
            started = True
            json_lines.append(line)
            if stripped.endswith("}"):
                break
        elif started:
            json_lines.append(line)
            if stripped.endswith("}"):
                break
    if not json_lines:  # pragma: no cover - defensive fallback
        pytest.fail("No JSON payload emitted by CLI")
    payload = json.loads("\n".join(json_lines))
    assert payload["status"] == "ok"
    assert recorded["input"] == str(audio)
    assert recorded["out"] == str(outdir)
    assert recorded["config"]["ignore_tx_cache"] is True


@pytest.mark.parametrize(
    "module_name",
    [
        "diaremot.pipeline.audio_pipeline_core",
        "audio_pipeline_core",
    ],
)
def test_audio_pipeline_core_main_guard(monkeypatch: pytest.MonkeyPatch, module_name: str) -> None:
    import diaremot.pipeline.cli_entry as cli_entry_module

    recorded: dict[str, Any] = {}

    def fake_main(argv: list[str]) -> int:
        recorded["argv"] = list(argv)
        return 42

    monkeypatch.setattr(cli_entry_module, "main", fake_main)
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module(module_name, run_name="__main__")

    assert excinfo.value.code == 42
    assert recorded["argv"] == []
