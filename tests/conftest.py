from __future__ import annotations

import importlib
import importlib.util
import math
import random
import sys
import types


def pytest_configure() -> None:
    if "numpy" not in sys.modules:
        spec = importlib.util.find_spec("numpy")
        if spec is not None:
            importlib.import_module("numpy")
        else:
            np = types.ModuleType("numpy")

            class _StubArray(list):  # type: ignore[type-arg]
                def _convert(self, dtype):
                    if dtype is None:
                        return float
                    if callable(dtype):
                        return dtype
                    return lambda value: dtype(value)

                def astype(self, dtype=None, *_, **__):  # type: ignore[override]
                    converter = self._convert(dtype)
                    return _StubArray(converter(item) for item in self)

                def __mul__(self, other):  # type: ignore[override]
                    if isinstance(other, (int, float)):
                        return _StubArray(item * other for item in self)
                    return _StubArray(super().__mul__(other))

                __rmul__ = __mul__

                def __truediv__(self, other):  # type: ignore[override]
                    if isinstance(other, (int, float)) and other != 0:
                        return _StubArray(item / other for item in self)
                    return _StubArray(self)

                def __pow__(self, power):  # type: ignore[override]
                    if isinstance(power, (int, float)):
                        return _StubArray(item**power for item in self)
                    return _StubArray(self)

                def __array__(self):  # type: ignore[override]
                    return self

                @property
                def size(self) -> int:
                    return len(self)

            def _wrap(iterable):
                if isinstance(iterable, _StubArray):
                    return iterable
                return _StubArray(iterable)

            def _flatten_size(shape):
                if isinstance(shape, tuple):
                    total = 1
                    for dim in shape:
                        total *= int(dim)
                    return total
                return int(shape)

            def _to_array(data, dtype=None):  # type: ignore[override]
                if isinstance(data, (list, tuple, _StubArray)):
                    result = _wrap(list(data))
                elif hasattr(data, "__iter__") and not isinstance(data, (str, bytes)):
                    result = _wrap(list(data))
                else:
                    result = _wrap([data])
                return result.astype(dtype)

            np.array = _to_array  # type: ignore[attr-defined]
            np.asarray = _to_array  # type: ignore[attr-defined]
            np.ascontiguousarray = lambda data: _to_array(data)  # type: ignore[attr-defined]
            np.zeros = lambda shape, dtype=None: _to_array(
                [0.0] * _flatten_size(shape), dtype
            )  # type: ignore[attr-defined]
            np.ones = lambda shape, dtype=None: _to_array(
                [1.0] * _flatten_size(shape), dtype
            )  # type: ignore[attr-defined]
            np.float32 = float  # type: ignore[attr-defined]
            np.sqrt = lambda x: math.sqrt(x)  # type: ignore[attr-defined]
            np.mean = lambda arr: (sum(arr) / len(arr)) if arr else 0.0  # type: ignore[attr-defined]
            np.max = lambda arr: max(arr) if arr else 0.0  # type: ignore[attr-defined]
            np.min = lambda arr: min(arr) if arr else 0.0  # type: ignore[attr-defined]
            np.clip = lambda arr, a, b: _wrap(max(a, min(b, x)) for x in arr)  # type: ignore[attr-defined]
            np.abs = lambda arr: _wrap(abs(x) for x in arr)  # type: ignore[attr-defined]
            np.log = lambda arr: _wrap(math.log(x) for x in arr)  # type: ignore[attr-defined]
            np.exp = lambda arr: _wrap(math.exp(x) for x in arr)  # type: ignore[attr-defined]
            np.isscalar = lambda obj: isinstance(obj, (int, float))  # type: ignore[attr-defined]
            np.bool_ = bool  # type: ignore[attr-defined]
            np.ndarray = _StubArray  # type: ignore[attr-defined]

            class _RandomNamespace(types.SimpleNamespace):
                @staticmethod
                def _sample(generator, size, *args):
                    count = _flatten_size(size)
                    return _wrap(generator(*args) for _ in range(count))

                @staticmethod
                def randn(*size):  # type: ignore[attr-defined]
                    if not size:
                        size = (1,)
                    return _RandomNamespace._sample(random.gauss, size[0], 0.0, 1.0)

                @staticmethod
                def normal(loc=0.0, scale=1.0, size=1):  # type: ignore[attr-defined]
                    return _RandomNamespace._sample(random.gauss, size, loc, scale)

                @staticmethod
                def random(size=1):  # type: ignore[attr-defined]
                    return _RandomNamespace._sample(random.random, size)

            np.random = _RandomNamespace()  # type: ignore[attr-defined]
            np.__stub__ = True  # type: ignore[attr-defined]
            sys.modules["numpy"] = np

    if "librosa" not in sys.modules:
        try:
            importlib.import_module("librosa")
        except ModuleNotFoundError:
            module = types.ModuleType("librosa")
            module.util = types.SimpleNamespace(frame=lambda *args, **kwargs: [])  # type: ignore[attr-defined]
            module.__stub__ = True  # type: ignore[attr-defined]
            sys.modules["librosa"] = module

    if "scipy" not in sys.modules:
        try:
            importlib.import_module("scipy")
        except ModuleNotFoundError:
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
        try:
            importlib.import_module("reportlab")
        except ModuleNotFoundError:
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
        try:
            importlib.import_module("soundfile")
        except ModuleNotFoundError:
            sf = types.ModuleType("soundfile")
            sf.SoundFile = object  # type: ignore[attr-defined]
            sf.read = lambda *args, **kwargs: ([0.0], 16000)  # type: ignore[attr-defined]
            sf.write = lambda *args, **kwargs: None  # type: ignore[attr-defined]
            sys.modules["soundfile"] = sf
