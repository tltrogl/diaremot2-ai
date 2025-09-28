"""Hash utilities with a consistent BLAKE2s default."""

from __future__ import annotations

from hashlib import blake2s, new
from pathlib import Path
from typing import Iterable, Union


def _iter_chunks(handle, chunk_size: int) -> Iterable[bytes]:
    while True:
        chunk = handle.read(chunk_size)
        if not chunk:
            break
        yield chunk


def hash_file(
    path: Union[str, Path],
    *,
    algo: str = "blake2s",
    chunk_size: int = 8192,
    digest_size: int | None = None,
) -> str:
    """Return the hexadecimal digest for ``path`` using ``algo``.

    The default algorithm is **BLAKE2s** to provide modern security with
    excellent CPU performance. ``algo`` may be any algorithm accepted by
    :func:`hashlib.new`.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    if algo.lower() == "blake2s":
        hasher = blake2s(digest_size=digest_size or blake2s().digest_size)
    else:
        hasher = new(algo)

    with file_path.open("rb") as handle:
        for chunk in _iter_chunks(handle, chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


__all__ = ["hash_file"]

