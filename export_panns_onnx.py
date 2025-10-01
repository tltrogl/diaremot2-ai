#!/usr/bin/env python
"""Export a PANNs checkpoint (CNN14 variants) to ONNX for offline SED inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

try:
    from panns_inference.models import (
        Cnn14,
        Cnn14_DecisionLevelAtt,
        Cnn14_DecisionLevelMax,
    )
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "panns_inference is required. Install it with 'pip install panns-inference'."
    ) from exc

MODEL_SPECS = {
    "cnn14": {
        "factory": Cnn14,
        "output_keys": ("clipwise_output", "embedding"),
        "description": "Baseline Cnn14 (clip probabilities + embeddings).",
    },
    "decision_att": {
        "factory": Cnn14_DecisionLevelAtt,
        "output_keys": ("clipwise_output", "segmentwise_output"),
        "description": "Cnn14 with attention pooling (clip + segment attention).",
    },
    "decision_max": {
        "factory": Cnn14_DecisionLevelMax,
        "output_keys": ("clipwise_output",),
        "description": "Cnn14 with max pooling (clip probabilities only).",
    },
}


class _PannsWrapper(torch.nn.Module):
    """Expose selected outputs from a PANNs model for ONNX export."""

    def __init__(self, model: torch.nn.Module, output_keys: Sequence[str]):
        super().__init__()
        self.model = model
        self.output_keys = tuple(output_keys)

    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        output = self.model(waveform)
        if isinstance(output, dict):
            tensors: List[torch.Tensor] = []
            for key in self.output_keys:
                tensor = output.get(key)
                if tensor is None:
                    raise RuntimeError(
                        f"Model output missing key '{key}'. Available: {list(output.keys())}"
                    )
                tensors.append(tensor)
            return tuple(tensors)
        # Some wrappers may return a tuple; align by position
        if len(self.output_keys) == 1:
            return (output[0],)
        if len(output) < len(self.output_keys):
            raise RuntimeError(
                f"Model returned {len(output)} tensors but {len(self.output_keys)} were requested"
            )
        return tuple(output[: len(self.output_keys)])


def _load_checkpoint(path: Path, model_type: str) -> torch.nn.Module:
    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    else:
        state_dict = ckpt

    spec = MODEL_SPECS[model_type]
    factory = spec["factory"]
    model = factory(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys during load: {missing}")
    if unexpected:
        print(f"[warn] Unexpected keys during load: {unexpected}")
    model.eval()
    return model


DYNAMIC_AXES = {
    "clipwise_output": {0: "batch", 1: "classes"},
    "embedding": {0: "batch", 1: "features"},
    "segmentwise_output": {0: "batch", 1: "frames", 2: "classes"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the PANNs checkpoint (.pth).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output ONNX file (e.g. models/panns/model.onnx).",
    )
    parser.add_argument(
        "--model-type",
        choices=sorted(MODEL_SPECS.keys()),
        default="cnn14",
        help="Which PANNs architecture to export.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=10.0,
        help="Dummy clip duration (seconds) used to trace the model.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Additionally write an INT8 quantized model alongside the fp32 export.",
    )
    return parser.parse_args()


def export_model(args: argparse.Namespace) -> None:
    ckpt_path: Path = args.checkpoint.expanduser().resolve()
    out_path: Path = args.out.expanduser().resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = _load_checkpoint(ckpt_path, args.model_type)
    output_keys = MODEL_SPECS[args.model_type]["output_keys"]
    wrapped = _PannsWrapper(model, output_keys)

    sample_rate = 32000
    num_samples = max(1, int(sample_rate * float(args.duration_sec)))
    dummy = torch.zeros(1, num_samples, dtype=torch.float32)

    dynamic_axes = {
        "waveform": {1: "samples"},
    }
    for key in output_keys:
        axis = DYNAMIC_AXES.get(key)
        if axis:
            dynamic_axes[key] = axis

    print(f"[info] Exporting ONNX -> {out_path}")
    torch.onnx.export(
        wrapped,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=int(args.opset),
        do_constant_folding=True,
        input_names=["waveform"],
        output_names=list(output_keys),
        dynamic_axes=dynamic_axes,
    )
    print("[info] Export complete")

    if args.quantize:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except Exception as exc:  # pragma: no cover - optional
            print(
                f"[warn] onnxruntime.quantization unavailable ({exc}); skipping INT8 export"
            )
            return

        int8_path = out_path.with_suffix(".int8.onnx")
        print(f"[info] Quantizing -> {int8_path}")
        quantize_dynamic(
            str(out_path),
            str(int8_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
        print("[info] INT8 model written")


def main() -> None:
    args = parse_args()
    try:
        export_model(args)
    except KeyboardInterrupt:  # pragma: no cover
        sys.exit(130)


if __name__ == "__main__":
    main()

