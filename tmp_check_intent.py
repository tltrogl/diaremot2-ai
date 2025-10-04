from diaremot.affect.emotion_analyzer import EmotionIntentAnalyzer
an = EmotionIntentAnalyzer(affect_backend="onnx", affect_intent_model_dir=r"D:\\diaremot\\diaremot2-1\\models\\bart")
an._lazy_intent()
print("session", bool(an._intent_session))
print("tokenizer", an._intent_tokenizer is not None)
print("entail", an._intent_entail_idx)
print("contra", an._intent_contra_idx)
