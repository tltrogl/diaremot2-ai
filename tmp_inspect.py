import torch
from panns_inference.models import Cnn14

model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
model.eval()
dummy = torch.zeros(1, 32000*10)
with torch.no_grad():
    out = model(dummy)
print(type(out))
print(out)
