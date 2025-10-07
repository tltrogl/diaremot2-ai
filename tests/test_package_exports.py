"""Regression coverage for package-level __init__ exports."""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest

PACKAGE_EXPORTS = [
    "diaremot.affect",
    "diaremot.io",
    "diaremot.pipeline",
    "diaremot.summaries",
]


@pytest.mark.parametrize("package_name", PACKAGE_EXPORTS)
def test_star_import_populates_all(package_name: str) -> None:
    module = importlib.import_module(package_name)
    exported = getattr(module, "__all__", ())

    namespace: dict[str, object] = {}
    exec(f"from {package_name} import *", namespace)

    for name in exported:
        assert name in namespace, f"{package_name} missing {name} in star import"
        attr = getattr(module, name)
        if isinstance(attr, ModuleType):
            submodule = importlib.import_module(f"{package_name}.{name}")
            assert (
                attr is submodule
            ), f"{package_name}.{name} should resolve to its submodule"
