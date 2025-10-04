import importlib

import pytest

MODULE_PATH = "diaremot.pipeline.run_pipeline"
CORE_PATH = "diaremot.pipeline.audio_pipeline_core"


@pytest.fixture()
def shim():
    import diaremot.pipeline.audio_pipeline_core as core
    import diaremot.pipeline.run_pipeline as module

    importlib.reload(module)
    return core, module


def test_shim_exports_core_types(shim):
    core, module = shim
    assert module.AudioAnalysisPipelineV2 is core.AudioAnalysisPipelineV2
    assert module.DEFAULT_PIPELINE_CONFIG is core.DEFAULT_PIPELINE_CONFIG


def test_shim_delegates_functions(monkeypatch, shim):
    core, module = shim
    calls = {}

    def fake_build(overrides=None):
        calls["build"] = overrides
        return {"config": overrides}

    def fake_run(input_path, outdir, *, config=None, clear_cache=False):
        calls["run"] = (input_path, outdir, config, clear_cache)
        return {"status": "ok"}

    def fake_resume(checkpoint_path, *, outdir=None, config=None):
        calls["resume"] = (checkpoint_path, outdir, config)
        return {"resume": True}

    def fake_diag(require_versions=False):
        calls["diagnostics"] = require_versions
        return {"diag": require_versions}

    def fake_verify(strict=False):
        calls["verify"] = strict
        return True, []

    def fake_clear(cache_root=None):
        calls["clear"] = cache_root
        return None

    monkeypatch.setattr(core, "build_pipeline_config", fake_build)
    monkeypatch.setattr(core, "run_pipeline", fake_run)
    monkeypatch.setattr(core, "resume", fake_resume)
    monkeypatch.setattr(core, "diagnostics", fake_diag)
    monkeypatch.setattr(core, "verify_dependencies", fake_verify)
    monkeypatch.setattr(core, "clear_pipeline_cache", fake_clear)

    importlib.reload(module)

    config = module.build_pipeline_config({"beam_size": 2})
    assert config == {"config": {"beam_size": 2}}
    assert calls["build"] == {"beam_size": 2}

    result = module.run_pipeline("input.wav", "out", config={"beam": 3}, clear_cache=True)
    assert result == {"status": "ok"}
    assert calls["run"] == ("input.wav", "out", {"beam": 3}, True)

    with pytest.warns(RuntimeWarning, match="allow_reprocess"):
        resume_payload = module.resume(
            "ckpt.json", outdir="out", config={"foo": 1}, allow_reprocess=True
        )
    assert resume_payload == {"resume": True}
    assert calls["resume"] == ("ckpt.json", "out", {"foo": 1})

    diag_payload = module.diagnostics(require_versions=True)
    assert diag_payload == {"diag": True}
    assert calls["diagnostics"] is True

    verify_payload = module.verify_dependencies(strict=True)
    assert verify_payload == (True, [])
    assert calls["verify"] is True

    module.clear_pipeline_cache(cache_root=".cache-test")
    assert calls["clear"] == ".cache-test"
