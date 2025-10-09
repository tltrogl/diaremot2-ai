from faster_whisper import WhisperModel
m = WhisperModel('tiny.en', device='cpu', compute_type='int8')
print('model init ok')
