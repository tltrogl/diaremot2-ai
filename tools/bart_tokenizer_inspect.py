"""Utility script to inspect a BART tokenizer checkpoint."""
<<<<<<< HEAD

=======
>>>>>>> 7b611bc33ae14a4cd702cb5f9355008663373325
from __future__ import annotations

import argparse
import sys
from pathlib import Path
<<<<<<< HEAD
=======
from typing import Optional
>>>>>>> 7b611bc33ae14a4cd702cb5f9355008663373325

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.bart_cli_utils import describe_bart_candidates, resolve_bart_dir


def inspect_tokenizer(model_dir: Path) -> None:
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "transformers is not installed. Please run 'pip install -r requirements.txt'"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    class_name = tokenizer.__class__.__name__
<<<<<<< HEAD
    vocab_size: int | None = getattr(tokenizer, "vocab_size", None)
=======
    vocab_size: Optional[int] = getattr(tokenizer, "vocab_size", None)
>>>>>>> 7b611bc33ae14a4cd702cb5f9355008663373325

    print(f"Loaded tokenizer from: {model_dir}")
    print(f"Tokenizer class: {class_name}")
    if vocab_size is not None:
        print(f"Vocabulary size: {vocab_size}")
    if hasattr(tokenizer, "get_vocab"):
        print(f"Unique entries in vocab: {len(tokenizer.get_vocab())}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a BART tokenizer checkpoint")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=(
            "Path to the BART model directory. If omitted, a set of sensible "
            "defaults will be tried."
        ),
    )
    return parser


<<<<<<< HEAD
def main(argv: list[str] | None = None) -> int:
=======
def main(argv: Optional[list[str]] = None) -> int:
>>>>>>> 7b611bc33ae14a4cd702cb5f9355008663373325
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        model_dir = resolve_bart_dir(args.model_dir, must_exist=True)
    except FileNotFoundError as exc:
        parser.error(
            "Could not locate the BART model directory. "
            "Tried the following candidates:\n" + describe_bart_candidates(args.model_dir)
        )
        raise SystemExit(2) from exc

    inspect_tokenizer(model_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
