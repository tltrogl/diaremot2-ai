from pathlib import Path

from diaremot.pipeline import cache_env, runtime_env


def test_configure_local_cache_env_site_packages(monkeypatch, tmp_path):
    """Site-packages layout should fall back to a user cache when cwd is unwritable."""

    # Reset any environment that the module-level call might have set during import.
    for name in (
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
    ):
        monkeypatch.delenv(name, raising=False)

    fake_site_packages = (
        tmp_path
        / "venv"
        / "lib"
        / "python3.11"
        / "site-packages"
        / "diaremot"
        / "pipeline"
        / "runtime_env.py"
    )
    fake_site_packages.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(runtime_env, "__file__", str(fake_site_packages))

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home_dir)

    blocked_cache_root = (work_dir / ".cache").resolve()
    original_mkdir = Path.mkdir

    def guarded_mkdir(self, mode=0o777, parents=False, exist_ok=False):  # type: ignore[override]
        resolved = Path(self).resolve()
        if resolved == blocked_cache_root:
            raise PermissionError("cwd cache directory is not writable")
        return original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", guarded_mkdir)

    runtime_env.configure_local_cache_env()

    expected_root = (home_dir / ".cache" / "diaremot").resolve()
    assert expected_root.exists()

    for env_name, subdir in {
        "HF_HOME": "hf",
        "HUGGINGFACE_HUB_CACHE": "hf",
        "TRANSFORMERS_CACHE": "transformers",
        "TORCH_HOME": "torch",
        "XDG_CACHE_HOME": None,
    }.items():
        target_value = Path(runtime_env.os.environ[env_name]).resolve()
        expected_path = expected_root if subdir is None else (expected_root / subdir).resolve()
        assert target_value == expected_path


def test_cache_env_import_handles_readonly_prefix(monkeypatch, tmp_path):
    """Pipeline cache helper should fall back when interpreter prefix is read-only."""

    for name in (
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
    ):
        monkeypatch.delenv(name, raising=False)

    fake_prefix = tmp_path / "prefix" / "lib" / "python3.11"
    fake_site_packages = fake_prefix / "site-packages" / "diaremot" / "pipeline"
    fake_site_packages.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(runtime_env, "__file__", str(fake_site_packages / "runtime_env.py"))
    monkeypatch.setattr(cache_env, "__file__", str(fake_site_packages / "cache_env.py"))

    monkeypatch.chdir(fake_prefix)

    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home_dir)

    blocked_prefix = fake_prefix.resolve()
    original_mkdir = Path.mkdir

    def guarded_mkdir(self, mode=0o777, parents=False, exist_ok=False):  # type: ignore[override]
        resolved = Path(self).resolve()
        if str(resolved).startswith(str(blocked_prefix)):
            raise PermissionError("interpreter prefix is read-only")
        return original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", guarded_mkdir)

    cache_root = cache_env.configure_local_cache_env()

    expected_root = (home_dir / ".cache" / "diaremot").resolve()
    assert cache_root == expected_root
    assert expected_root.exists()

    for env_name, subdir in {
        "HF_HOME": "hf",
        "HUGGINGFACE_HUB_CACHE": "hf",
        "TRANSFORMERS_CACHE": "transformers",
        "TORCH_HOME": "torch",
        "XDG_CACHE_HOME": None,
    }.items():
        value = Path(runtime_env.os.environ[env_name]).resolve()
        expected_path = expected_root if subdir is None else (expected_root / subdir).resolve()
        assert value == expected_path
