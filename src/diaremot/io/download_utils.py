"""Utility helpers for fetching remote assets with the standard library."""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.request import urlopen


def download_file(url: str, destination: Path) -> None:
    """Download a file from ``url`` and save it to ``destination``.

    Parameters
    ----------
    url:
        The URL pointing to the resource to download.
    destination:
        Path where the file will be stored. Parent directories are created
        automatically.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, destination.open("wb") as out_file:
        shutil.copyfileobj(response, out_file)
