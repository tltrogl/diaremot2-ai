import inspect
from diaremot.affect.emotion_analyzer import EmotionIntentAnalyzer
print('module file', inspect.getfile(EmotionIntentAnalyzer))
an=EmotionIntentAnalyzer(affect_backend='onnx')
an._lazy_intent()
print('session', bool(an._intent_session))
print('tokenizer', an._intent_tokenizer is not None)
print('entail', an._intent_entail_idx)

