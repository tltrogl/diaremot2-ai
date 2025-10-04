from diaremot.affect.emotion_analyzer import EmotionIntentAnalyzer
an = EmotionIntentAnalyzer(affect_backend='onnx')
print('resolved text dir:', an.affect_text_model_dir)
an._lazy_text()
print('text session:', an._text_session is not None)
print('text tokenizer:', an._text_tokenizer is not None)
