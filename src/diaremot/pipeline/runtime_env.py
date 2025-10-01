"""Runtime environment helpers for pipeline execution."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Iterable

__all__ = [
    "WINDOWS_MODELS_ROOT",
    "DEFAULT_WHISPER_MODEL",
    "configure_local_cache_env",
    "resolve_default_whisper_model",
]


PROJECT_ROOT_MARKERS = ("pyproject.toml", ".git")


def _find_project_root(start: Path) -> Path | None:
    """Walk upward from ``start`` until a project marker is found."""

    current = start
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        for marker in PROJECT_ROOT_MARKERS:
            if (candidate / marker).exists():
                return candidate
    return None


def _candidate_cache_roots(script_path: Path) -> Iterable[Path]:
    project_root = _find_project_root(script_path)
    if project_root is not None:
        yield project_root / ".cache"
    else:
        yield Path.cwd() / ".cache"
        yield Path.home() / ".cache" / "diaremot"


def _ensure_writable_directory(path: Path) -> bool:
    """Return ``True`` when ``path`` can be created and written to."""

    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False

    probe = path / ".cache_write_test"
    try:
        probe.touch(exist_ok=True)
    except OSError:
        return False
    else:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            return False
    return os.access(path, os.W_OK | os.X_OK)


def configure_local_cache_env() -> None:
    """Ensure all cache directories resolve to a writable, local cache root."""

    cache_root = None
    for candidate in _candidate_cache_roots(Path(__file__).resolve()):
        resolved = candidate.resolve()
        if _ensure_writable_directory(resolved):
            cache_root = resolved
            break
    if cache_root is None:
        raise PermissionError("Unable to locate a writable cache directory for DiaRemot")

    targets = {
        "HF_HOME": cache_root / "hf",
        "HUGGINGFACE_HUB_CACHE": cache_root / "hf",
        "TRANSFORMERS_CACHE": cache_root / "transformers",
        "TORCH_HOME": cache_root / "torch",
        "XDG_CACHE_HOME": cache_root,
    }
    for env_name, target in targets.items():
        target_path = target.resolve()
        existing = os.environ.get(env_name)
        if existing:
            try:
                existing_path = Path(existing).resolve()
            except (OSError, RuntimeError, ValueError):
                existing_path = None
            if existing_path is not None:
                if existing_path == target_path:
                    continue
                try:
                    if existing_path.is_relative_to(cache_root):
                        continue
                except AttributeError:
                    # Python < 3.9 compatibility – fall back to manual check
                    try:
                        if str(existing_path).startswith(str(cache_root)):
                            continue
                    except Exception:
                        pass
        target_path.mkdir(parents=True, exist_ok=True)
        os.environ[env_name] = str(target_path)


configure_local_cache_env()

WINDOWS_MODELS_ROOT = Path("D:/models") if os.name == "nt" else None


def resolve_default_whisper_model() -> Path:
    env_override = os.environ.get("WHISPER_MODEL_PATH")
    if env_override:
        return Path(env_override)

    candidates = []
    if WINDOWS_MODELS_ROOT:
        candidates.append(WINDOWS_MODELS_ROOT / "faster-whisper-large-v3-turbo-ct2")
    candidates.append(Path.home() / "whisper_models" / "faster-whisper-large-v3-turbo-ct2")

    for candidate in candidates:
        if Path(candidate).exists():
            return Path(candidate)

    return Path(candidates[0])


DEFAULT_WHISPER_MODEL = resolve_default_whisper_model()
