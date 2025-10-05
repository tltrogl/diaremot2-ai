import os, numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

ID2LABEL = ["angry","calm","disgust","fearful","happy","neutral","sad","surprised"]

class SERDpngtm:
    def __init__(self, model_dir=None):
        # Use local snapshot only (no internet)
        self.model_dir = model_dir or os.getenv("DIAREMOT_SER_MODEL_DIR", r"D:\diaremot\diaremot2-1\models\dpngtm_ser")
        self.proc  = Wav2Vec2Processor.from_pretrained(self.model_dir, local_files_only=True)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_dir, local_files_only=True).eval()

    def predict_16k_f32(self, wav_16k_f32: np.ndarray):
        # expects 16 kHz mono float32; average channels if needed
        if getattr(wav_16k_f32, "ndim", 1) > 1:
            wav_16k_f32 = wav_16k_f32.mean(axis=1)
        inputs = self.proc(wav_16k_f32, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        idx = int(np.argmax(probs))
        return ID2LABEL[idx], {ID2LABEL[i]: float(probs[i]) for i in range(len(ID2LABEL))}
