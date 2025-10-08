import json
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

try:
    import typer
    from typer.testing import CliRunner
except ModuleNotFoundError:  # pragma: no cover - exercised only without typer installed
    pytest.skip("typer is required for CLI tests", allow_module_level=True)

# Ensure the src layout is importable when running tests directly from the repo root.
TESTS_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = TESTS_ROOT / "src"
for candidate in (SRC_DIR, TESTS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from diaremot.cli import (  # noqa: E402
    app,
    diagnostics,
)
from diaremot.pipeline.runtime_env import DEFAULT_WHISPER_MODEL  # noqa: E402


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_cli_requires_input_argument(runner: CliRunner) -> None:
    result = runner.invoke(app, ["run"])  # missing required options
    assert result.exit_code != 0
    assert "--input" in result.stdout or "--input" in result.stderr


def test_cli_run_invokes_pipeline(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    audio = tmp_path / "call.wav"
    audio.write_text("fake audio")
    outdir = tmp_path / "outputs"

    captured = {}

    def fake_run(input_path: str, outdir_path: str, *, config, clear_cache: bool):
        captured["input"] = input_path
        captured["outdir"] = outdir_path
        captured["config"] = config
        captured["clear_cache"] = clear_cache
        return {"status": "ok"}

    monkeypatch.setattr("diaremot.cli.core_run_pipeline", fake_run)

    result = runner.invoke(
        app,
        [
            "run",
            "--input",
            str(audio),
            "--outdir",
            str(outdir),
            "--profile",
            "fast",
            "--beam-size",
            "2",
            "--affect-backend",
            "onnx",
            "--clear-cache",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert captured["input"] == str(audio)
    assert captured["outdir"] == str(outdir)
    # fast profile should override whisper model, CLI override adjusts beam size
    assert captured["config"]["whisper_model"] == str(DEFAULT_WHISPER_MODEL)
    assert captured["config"]["beam_size"] == 2
    assert captured["clear_cache"] is True


def test_cli_validates_affect_backend_paths(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    audio = tmp_path / "call.wav"
    audio.write_text("fake audio")
    outdir = tmp_path / "outputs"

    def fail_run(*args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("pipeline should not run when validation fails")

    monkeypatch.setattr("diaremot.cli.core_run_pipeline", fail_run)

    missing = tmp_path / "missing"

    result = runner.invoke(
        app,
        [
            "run",
            "--input",
            str(audio),
            "--outdir",
            str(outdir),
            "--affect-backend",
            "onnx",
            "--affect-text-model-dir",
            str(missing),
        ],
    )

    assert result.exit_code != 0
    assert "affect_text_model_dir" in result.stdout or "affect_text_model_dir" in result.stderr


def test_cli_smoke_generates_audio_and_runs_pipeline(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    generated = {}

    def fake_generate(target: Path, duration: float, sample_rate: int, ffmpeg_bin: Optional[str]):
        generated["target"] = target
        target.write_bytes(b"WAV")
        return "python"

    captured_assemble = {}

    def fake_assemble(profile: Optional[str], overrides: dict[str, Any]):
        captured_assemble["profile"] = profile
        captured_assemble["overrides"] = overrides
        return {"config": "value"}

    captured_run = {}

    def fake_run(input_path: str, outdir_path: str, *, config, clear_cache: bool):
        captured_run["input"] = input_path
        captured_run["outdir"] = outdir_path
        captured_run["config"] = config
        captured_run["clear_cache"] = clear_cache
        return {"status": "ok"}

    monkeypatch.setattr("diaremot.cli._generate_sample_audio", fake_generate)
    monkeypatch.setattr("diaremot.cli._assemble_config", fake_assemble)
    monkeypatch.setattr("diaremot.cli._validate_assets", lambda *args, **kwargs: None)
    monkeypatch.setattr("diaremot.cli.core_run_pipeline", fake_run)

    result = runner.invoke(
        app,
        [
            "smoke",
            "--outdir",
            str(tmp_path),
            "--keep-audio",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload == {"status": "ok"}

    expected_sample = tmp_path / "diaremot_smoke_input.wav"
    assert generated["target"] == expected_sample
    assert captured_assemble["profile"] is None
    assert captured_assemble["overrides"]["disable_affect"] is True
    assert captured_run["input"] == str(expected_sample)
    assert captured_run["outdir"] == str(tmp_path)
    assert captured_run["config"] == {"config": "value"}
    assert captured_run["clear_cache"] is True
    assert expected_sample.exists()

def test_diagnostics_entrypoint_accepts_strict(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    captured = {}

    def fake_core_diagnostics(*, require_versions: bool):
        captured["strict"] = require_versions
        return {"status": "ok"}

    monkeypatch.setattr("diaremot.cli.core_diagnostics", fake_core_diagnostics)

    command = typer.Typer(add_completion=False)
    command.command()(diagnostics)
    result = runner.invoke(command, ["--strict"])

    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout) == {"status": "ok"}
    assert captured["strict"] is True
