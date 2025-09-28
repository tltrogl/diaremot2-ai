"""Dependency alignment regression tests.

These checks keep the dependency manifests in sync with the runtime
imports exercised by the package. They intentionally avoid importing
heavy frameworks so they stay inexpensive in CI environments.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterable
import re

try:  # Python >=3.11
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    import tomli as tomllib  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

IMPORT_NAME_TO_DEP = {
    "av": "av",
    "faster_whisper": "faster-whisper",
    "huggingface_hub": "huggingface_hub",
    "librosa": "librosa",
    "numpy": "numpy",
    "onnxruntime": "onnxruntime",
    "packaging": "packaging",
    "panns_inference": "panns-inference",
    "parselmouth": "praat-parselmouth",
    "reportlab": "reportlab",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "soundfile": "soundfile",
    "torch": "torch",
    "transformers": "transformers",
    "typer": "typer",
    "whisper": "openai-whisper",
}

INTERNAL_IMPORTS = {
    "affect",
    "audio_pipeline_core",
    "audio_preprocessing",
    "cpu_optimized_diarizer",
    "diaremot",
    "intent_defaults",
    "paralinguistics",
    "pipeline",
    "pipeline_checkpoint_system",
    "speaker_diarization",
    "summaries",
    "transcription_module",
}

OPTIONAL_IMPORTS = {
    "suppress_warnings",  # optional environment helper
    "importlib_metadata",  # stdlib fallback only used on <3.8
}


_SPEC_SPLIT_RE = re.compile(r"[<>=!~]", re.ASCII)


def _normalize_requirement(spec: str) -> str:
    return _SPEC_SPLIT_RE.split(spec, 1)[0]


def _load_pyproject_dependencies() -> set[str]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    deps: Iterable[str] = data["project"]["dependencies"]
    return {_normalize_requirement(dep).split("[")[0] for dep in deps}


def _load_requirements() -> set[str]:
    reqs: set[str] = set()
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("--"):
            continue
        reqs.add(_normalize_requirement(stripped).split("[")[0])
    return reqs


def _iter_import_roots() -> set[str]:
    roots: set[str] = set()
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
    return roots


PYTHON_STDLIB = set(sys.stdlib_module_names)


def test_requirements_match_pyproject() -> None:
    deps = _load_pyproject_dependencies()
    reqs = _load_requirements()
    assert deps == reqs, (
        "pyproject.toml dependencies diverge from requirements.txt: "
        f"missing_in_requirements={sorted(deps - reqs)}, "
        f"missing_in_pyproject={sorted(reqs - deps)}"
    )


def test_third_party_imports_are_declared() -> None:
    deps = _load_pyproject_dependencies()
    missing: dict[str, str] = {}

    for name in _iter_import_roots():
        if name in PYTHON_STDLIB:
            continue
        if name in INTERNAL_IMPORTS or name.startswith("diaremot"):
            continue
        if name in OPTIONAL_IMPORTS:
            continue

        dep_name = IMPORT_NAME_TO_DEP.get(name)
        if dep_name is None:
            missing[name] = "no dependency mapping"
            continue
        if dep_name not in deps:
            missing[name] = f"expected dependency '{dep_name}' not listed"

    assert not missing, (
        "Detected imports without matching dependency pins: "
        + ", ".join(f"{mod} ({reason})" for mod, reason in sorted(missing.items()))
    )
