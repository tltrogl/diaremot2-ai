import logging
from diaremot.affect.emotion_analyzer import EmotionIntentAnalyzer
logging.basicConfig(level=logging.INFO)
an = EmotionIntentAnalyzer(affect_backend='onnx')
print('model dir', an.affect_intent_model_dir)
an._lazy_intent()
print('session', an._intent_session)
print('tokenizer', type(an._intent_tokenizer) if an._intent_tokenizer else None)
print('entail', an._intent_entail_idx, 'contra', an._intent_contra_idx)

