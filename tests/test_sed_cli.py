import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("onnxruntime")

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "panns"


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="CNN14 model assets not available")
def test_cli_silence(tmp_path):
    wav_path = tmp_path / "silence.wav"
    sf.write(wav_path, np.zeros(16000 * 2, dtype=np.float32), 16000)

    output_csv = tmp_path / "events.csv"
    cmd = [
        sys.executable,
        "-m",
        "diaremot.sed.sed_panns_onnx",
        str(wav_path),
        str(output_csv),
        "--model",
        str(MODEL_DIR),
        "--frame",
        "1.0",
        "--hop",
        "0.5",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output_csv.exists()
    lines = output_csv.read_text().strip().splitlines()
    assert lines[0] == "file_id,start,end,label,score"
    assert len(lines) >= 1
