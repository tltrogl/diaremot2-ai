from __future__ import annotations

import math
import sys
import types


def pytest_configure() -> None:
    if "numpy" not in sys.modules:
        np = types.ModuleType("numpy")

        class _Array(list):  # type: ignore[type-arg]
            def __array__(self):  # pragma: no cover - compatibility hook
                return self

            def astype(self, dtype=None):
                if dtype is None:
                    converter = float
                elif callable(dtype):
                    converter = dtype
                else:
                    converter = float
                return _Array(converter(x) for x in self)

            @property
            def size(self) -> int:
                return len(self)

            def __getitem__(self, item):
                result = super().__getitem__(item)
                if isinstance(item, slice):
                    return _Array(result)
                return result

            def __mul__(self, other):
                if isinstance(other, (int, float)):
                    return _Array(float(x) * float(other) for x in self)
                return _Array(super().__mul__(other))

            __rmul__ = __mul__

            def __pow__(self, power):
                if isinstance(power, (int, float)):
                    return _Array(float(x) ** float(power) for x in self)
                raise TypeError("stub array only supports numeric powers")

        def _to_array(data, dtype=None):  # type: ignore[override]
            if isinstance(data, _Array):
                return data.astype(dtype)
            if isinstance(data, list):
                seq = _Array(data)
                return seq.astype(dtype) if dtype is not None else seq
            if isinstance(data, tuple):
                seq = _Array(data)
                return seq.astype(dtype) if dtype is not None else seq
            if hasattr(data, "__iter__") and not isinstance(data, (str, bytes)):
                seq = _Array(list(data))
                return seq.astype(dtype) if dtype is not None else seq
            seq = _Array([data])
            return seq.astype(dtype) if dtype is not None else seq

        def _ones(shape, dtype=None):  # type: ignore[attr-defined]
            total = shape if isinstance(shape, int) else int(math.prod(shape))
            return _to_array([1.0] * total, dtype)

        def _zeros(shape, dtype=None):  # type: ignore[attr-defined]
            total = shape if isinstance(shape, int) else int(math.prod(shape))
            return _to_array([0.0] * total, dtype)

        np.array = _to_array  # type: ignore[attr-defined]
        np.asarray = _to_array  # type: ignore[attr-defined]
        np.ascontiguousarray = lambda data: _to_array(data)  # type: ignore[attr-defined]
        np.zeros = _zeros  # type: ignore[attr-defined]
        np.ones = _ones  # type: ignore[attr-defined]
        np.float32 = float  # type: ignore[attr-defined]
        np.sqrt = lambda x: math.sqrt(x)  # type: ignore[attr-defined]
        np.mean = lambda arr: (sum(arr) / len(arr)) if arr else 0.0  # type: ignore[attr-defined]
        np.max = lambda arr: max(arr) if arr else 0.0  # type: ignore[attr-defined]
        np.min = lambda arr: min(arr) if arr else 0.0  # type: ignore[attr-defined]
        np.clip = lambda arr, a, b: _Array(max(a, min(b, x)) for x in arr)  # type: ignore[attr-defined]
        np.abs = lambda arr: _Array(abs(x) for x in arr)  # type: ignore[attr-defined]
        np.log = lambda arr: _Array(math.log(x) for x in arr)  # type: ignore[attr-defined]
        np.exp = lambda arr: _Array(math.exp(x) for x in arr)  # type: ignore[attr-defined]
        np.ndarray = _Array  # type: ignore[attr-defined]
        np.__stub__ = True  # type: ignore[attr-defined]
        sys.modules["numpy"] = np

    if "librosa" not in sys.modules:
        module = types.ModuleType("librosa")
        module.util = types.SimpleNamespace(frame=lambda *args, **kwargs: [])  # type: ignore[attr-defined]
        module.__stub__ = True  # type: ignore[attr-defined]
        sys.modules["librosa"] = module

    if "scipy" not in sys.modules:
        scipy = types.ModuleType("scipy")
        scipy.signal = types.SimpleNamespace(  # type: ignore[attr-defined]
            resample_poly=lambda audio, up, down: audio,
            butter=lambda order, cutoff, btype="highpass", fs=16000: ([1.0], [1.0]),
            filtfilt=lambda b, a, data: data,
        )
        scipy.ndimage = types.SimpleNamespace(median_filter=lambda data, size=1: data)  # type: ignore[attr-defined]
        sys.modules["scipy"] = scipy
        sys.modules["scipy.signal"] = scipy.signal
        sys.modules["scipy.ndimage"] = scipy.ndimage

    if "reportlab" not in sys.modules:
        reportlab = types.ModuleType("reportlab")
        pagesizes = types.ModuleType("reportlab.lib.pagesizes")
        pagesizes.letter = (612, 792)
        units = types.ModuleType("reportlab.lib.units")
        units.inch = 72
        styles = types.ModuleType("reportlab.lib.styles")

        class ParagraphStyle:  # type: ignore[too-few-public-methods]
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        styles.ParagraphStyle = ParagraphStyle  # type: ignore[attr-defined]
        colors = types.ModuleType("reportlab.lib.colors")
        colors.black = None
        colors.HexColor = lambda *args, **kwargs: None  # type: ignore[attr-defined]

        class _Dummy:  # type: ignore[too-few-public-methods]
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def __call__(self, *args, **kwargs):
                return None

        platypus = types.ModuleType("reportlab.platypus")
        platypus.Paragraph = _Dummy  # type: ignore[attr-defined]
        platypus.SimpleDocTemplate = _Dummy  # type: ignore[attr-defined]
        platypus.Spacer = _Dummy  # type: ignore[attr-defined]
        platypus.Table = _Dummy  # type: ignore[attr-defined]
        platypus.TableStyle = _Dummy  # type: ignore[attr-defined]

        reportlab.lib = types.SimpleNamespace(  # type: ignore[attr-defined]
            pagesizes=pagesizes,
            units=units,
            styles=styles,
            colors=colors,
        )
        reportlab.platypus = platypus  # type: ignore[attr-defined]

        sys.modules["reportlab"] = reportlab
        sys.modules["reportlab.lib"] = reportlab.lib
        sys.modules["reportlab.lib.pagesizes"] = pagesizes
        sys.modules["reportlab.lib.units"] = units
        sys.modules["reportlab.lib.styles"] = styles
        sys.modules["reportlab.lib.colors"] = colors
        sys.modules["reportlab.platypus"] = platypus

    if "soundfile" not in sys.modules:
        sf = types.ModuleType("soundfile")
        sf.SoundFile = object  # type: ignore[attr-defined]
        sf.read = lambda *args, **kwargs: ([0.0], 16000)  # type: ignore[attr-defined]
        sf.write = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        sys.modules["soundfile"] = sf
