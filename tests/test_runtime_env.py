from pathlib import Path

from pathlib import Path

from diaremot.pipeline import runtime_env


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

    expected_root = runtime_env.LINUX_CACHE_BASE.resolve()
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


def test_configure_local_cache_env_repo_cache_fallback(monkeypatch, tmp_path):
    """Project installs should fall back when the repo cache directory is unwritable."""

    for name in (
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
    ):
        monkeypatch.delenv(name, raising=False)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text("")

    script_path = repo_root / "src" / "diaremot" / "pipeline" / "runtime_env.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(runtime_env, "__file__", str(script_path))

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home_dir)

    canonical_root = runtime_env.LINUX_CACHE_BASE.resolve()
    blocked_repo_cache = (repo_root / ".cache").resolve()
    attempted: list[Path] = []
    original_ensure = runtime_env._ensure_writable_directory

    def tracking_ensure(path: Path) -> bool:
        attempted.append(path)
        if path in {canonical_root, blocked_repo_cache}:
            return False
        return original_ensure(path)

    monkeypatch.setattr(runtime_env, "_ensure_writable_directory", tracking_ensure)

    runtime_env.configure_local_cache_env()

    expected_root = (work_dir / ".cache").resolve()
    assert attempted[0] == canonical_root
    assert attempted[1] == blocked_repo_cache
    assert attempted[2] == expected_root

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
