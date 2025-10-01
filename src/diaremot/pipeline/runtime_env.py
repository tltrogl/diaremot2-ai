"""Runtime environment helpers for pipeline execution."""

from __future__ import annotations

from pathlib import Path
import os

__all__ = [
    "WINDOWS_MODELS_ROOT",
    "DEFAULT_WHISPER_MODEL",
    "configure_local_cache_env",
    "resolve_default_whisper_model",
]


def configure_local_cache_env() -> None:
    """Ensure all cache directories live under the repository's ``.cache`` folder."""

    cache_root = (Path(__file__).resolve().parents[3] / ".cache").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
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
