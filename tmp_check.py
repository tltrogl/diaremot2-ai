from export_panns_onnx import _load_checkpoint, _Cnn14Wrapper
import torch

model = _load_checkpoint(__import__("pathlib").Path("models/panns/Cnn14_mAP=0.431.pth"))
wrapped = _Cnn14Wrapper(model)
wrapped.eval()
with torch.no_grad():
    out = wrapped(torch.zeros(1, 32000))
print(out)
