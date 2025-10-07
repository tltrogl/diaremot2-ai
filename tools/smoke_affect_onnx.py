import inspect
import os
import sys

sys.path.insert(0, os.path.abspath("src"))
print("Using PY code from:", os.path.abspath("src"))
import diaremot
from diaremot.affect import emotion_analyzer as ea

print("diaremot module file:", inspect.getfile(diaremot))
print("emotion_analyzer file:", inspect.getfile(ea))

intent_dir = r"D:\diaremot\diaremot2-1\models\bart"
text_dir = r"D:\diaremot\diaremot2-1\models\goemotions-onnx"

an = ea.EmotionIntentAnalyzer(
    affect_backend="onnx",
    affect_text_model_dir=text_dir,
    affect_intent_model_dir=intent_dir,
)

# initialize lazily
an._lazy_text()
an._lazy_intent()

print("TEXT onnx session:", an._text_session is not None)
print("TEXT tokenizer:", an._text_tokenizer is not None)
print("INTENT onnx session:", an._intent_session is not None)
print("INTENT tokenizer:", an._intent_tokenizer is not None)
print("INTENT entail/contra:", an._intent_entail_idx, an._intent_contra_idx)

# simple inferences
text = "I love this; it is wonderful!"
full, top5 = an._infer_text(text)
print("Text top5:", top5[:3])

intent_text = "book a meeting on Friday"
intent_top, intent_top3 = an._infer_intent(intent_text)
print("Intent top:", intent_top, "top3:", intent_top3)
