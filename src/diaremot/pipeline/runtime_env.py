"""Runtime environment helpers for pipeline execution."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

__all__ = [
    "WINDOWS_MODELS_ROOT",
    "DEFAULT_MODELS_ROOT",
    "MODEL_ROOTS",
    "DEFAULT_WHISPER_MODEL",
    "configure_local_cache_env",
    "resolve_default_whisper_model",
    "iter_model_roots",
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
    candidates: list[Path] = []
    if project_root is not None:
        candidates.append(project_root / ".cache")

    candidates.extend(
        [
            Path.cwd() / ".cache",
            Path.home() / ".cache" / "diaremot",
        ]
    )

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        yield candidate


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
    script_path = Path(__file__).resolve()
    for candidate in _candidate_cache_roots(script_path):
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


def _discover_model_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("DIAREMOT_MODEL_DIR")
    if env_root:
        roots.append(Path(env_root).expanduser())

    if os.name == "nt":
        roots.append(Path("D:/models"))
    else:
        roots.append(Path("/models"))

    project_root = _find_project_root(Path(__file__).resolve())
    if project_root:
        roots.append(project_root / "models")

    roots.append(Path.cwd() / "models")
    roots.append(Path.home() / "models")

    seen: set[str] = set()
    deduped: list[Path] = []
    for candidate in roots:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


MODEL_ROOTS: tuple[Path, ...] = tuple(_discover_model_roots()) or (Path("/models"),)
DEFAULT_MODELS_ROOT: Path = MODEL_ROOTS[0]
WINDOWS_MODELS_ROOT = DEFAULT_MODELS_ROOT if os.name == "nt" else None


def iter_model_roots() -> tuple[Path, ...]:
    """Return an ordered tuple of candidate model roots."""

    return MODEL_ROOTS


def _iter_whisper_candidates() -> list[Path]:
    candidates: list[Path] = []
    for root in MODEL_ROOTS:
        candidates.extend(
            [
                root / "tiny.en",
                root / "faster-whisper" / "tiny.en",
                root / "faster-whisper-tiny.en",
                root / "ct2" / "tiny.en",
            ]
        )
    candidates.append(Path.home() / "whisper_models" / "tiny.en")
    return candidates


def resolve_default_whisper_model() -> Path:
    env_override = os.environ.get("WHISPER_MODEL_PATH")
    if env_override:
        return Path(env_override)

    for candidate in _iter_whisper_candidates():
        if candidate.exists():
            return candidate

    return _iter_whisper_candidates()[0]


DEFAULT_WHISPER_MODEL = resolve_default_whisper_model()
