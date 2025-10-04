from transformers import AutoTokenizer
p=r'D:\diaremot\diaremot2-1\models\bart'
print('Loading tokenizer from', p)
tok=AutoTokenizer.from_pretrained(p)
print('ok:', tok.__class__.__name__)
print('vocab size:', tok.vocab_size if hasattr(tok,'vocab_size') else 'n/a')
