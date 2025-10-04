import json, os, sys
root = r"D:\\diaremot\\diaremot2-1\\models\\bart"
tok_json = os.path.join(root, 'tokenizer.json')
merges_txt = os.path.join(root, 'merges.txt')
vocab_json = os.path.join(root, 'vocab.json')

print('Reading', tok_json)
with open(tok_json, 'r', encoding='utf-8') as f:
    data = json.load(f)
model = data.get('model', {})
merges = model.get('merges', [])
vocab = model.get('vocab', {})

# Write merges if missing or suspiciously small
if merges and (not os.path.exists(merges_txt) or os.path.getsize(merges_txt) < 10):
    with open(merges_txt, 'w', encoding='utf-8') as f:
        for m in merges:
            line = ' '.join(m) if isinstance(m, (list, tuple)) else str(m)
            f.write(line + "\n")
    print('Wrote', merges_txt, len(merges), 'lines')
else:
    print('merges not written (exists?' , os.path.exists(merges_txt), ', size:', os.path.getsize(merges_txt) if os.path.exists(merges_txt) else 0, ', count:', len(merges), ')')

if vocab and not os.path.exists(vocab_json):
    # write as json mapping token->id
    inv = sorted(((tid, tok) for tok, tid in vocab.items()), key=lambda x: x[0])
    ordered = {tok: int(tid) for tid, tok in inv}
    with open(vocab_json, 'w', encoding='utf-8') as f:
        json.dump(ordered, f, ensure_ascii=False)
    print('Wrote', vocab_json, 'size', len(ordered))
else:
    print('vocab not written (exists?', os.path.exists(vocab_json), ', size:', len(vocab), ')')
