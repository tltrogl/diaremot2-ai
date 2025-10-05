import os, json, numpy as np, soundfile as sf, librosa, torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

MODEL_DIR = r"D:\diaremot\diaremot2-1\models\dpngtm_ser"
AUDIO = r"D:\diaremot\diaremot2-ai\data\sample.wav"   # <-- change to a REAL file path
SR_TARGET = 16000

# Load locally only; no internet calls
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR, local_files_only=True)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True).eval()

# Read + conform audio
wav, sr = sf.read(AUDIO, dtype="float32", always_2d=False)
if sr != SR_TARGET:
    wav = librosa.resample(wav, orig_sr=sr, target_sr=SR_TARGET)
if getattr(wav, "ndim", 1) > 1:
    wav = wav.mean(axis=1)

# Inference
inputs = processor(wav, sampling_rate=SR_TARGET, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(**inputs).logits
probs = torch.softmax(logits, dim=-1)[0].tolist()

labels = ["angry","calm","disgust","fearful","happy","neutral","sad","surprised"]
top = int(np.argmax(probs))
print("TOP:", labels[top])
print("DISTR:", json.dumps({labels[i]: float(p) for i,p in enumerate(probs)}, indent=2))
