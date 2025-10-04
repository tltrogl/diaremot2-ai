from diaremot.affect.emotion_analyzer import EmotionIntentAnalyzer
analyzer = EmotionIntentAnalyzer(
    affect_backend='onnx',
    affect_intent_model_dir=r'D:\diaremot\diaremot2-1\models\bart'
)
analyzer._lazy_intent()
print('ONNX session:', analyzer._intent_session is not None)
print('Tokenizer:', analyzer._intent_tokenizer is not None)
print('Entail idx:', analyzer._intent_entail_idx)
print('Contra idx:', analyzer._intent_contra_idx)
print('Hypothesis template:', getattr(analyzer, '_intent_hypothesis_template', None))
try:
    top, top3 = analyzer._infer_intent('book a meeting on Friday')
    print('Infer OK:', top, top3)
except Exception as e:
    print('Infer error:', e)
