from __future__ import annotations

import os
from pathlib import Path

from faster_whisper.utils import download_model
from faster_whisper import WhisperModel


def main() -> None:
    # Ensure online access for this one-time download
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

    dest_root = Path(r"D:\\models\\faster-whisper")
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / "tiny.en"

    # Download model into deterministic local path
    model_dir = download_model("tiny.en", cache_dir=str(dest), local_files_only=False)
    print(f"downloaded: {model_dir}")

    # Validate we can instantiate locally without network
    m = WhisperModel(str(model_dir), device="cpu", compute_type="int8", local_files_only=True)
    del m
    print("warmup: ok")


if __name__ == "__main__":
    main()

