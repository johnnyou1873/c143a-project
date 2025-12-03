import json
from pathlib import Path

nb_path = Path('notebooks/phoneme_seq2seq.ipynb')
nb = json.loads(nb_path.read_text())

marker = 'librispeech_sentences.pkl'
if any(marker in ''.join(c.get('source','')) for c in nb['cells']):
    print('processing cell already present')
else:
    md = {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '# Rebuild LibriSpeech sentences\n',
            'Use LibriSpeech text in notebooks/data to regenerate data/librispeech_sentences.pkl.'
        ],
    }
    code = {
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': [
            'from pathlib import Path\n',
            'import re, pickle\n',
            "ROOT = Path.cwd()\n",
            "BOOKS_DIR = ROOT / 'notebooks' / 'data' / 'LibriSpeech' / 'books' / 'ascii'\n",
            "SENTENCE_PKL = ROOT / 'data' / 'librispeech_sentences.pkl'\n",
            '\n',
            "sentence_re = re.compile(r'[A-Za-z][^.!?]*[.!?]')\n",
            'sentences = []\n',
            "files = list(BOOKS_DIR.rglob('*.txt'))\n",
            "print(f'Found {len(files)} text files')\n",
            'for txt in files:\n',
            "    text = txt.read_text(encoding='utf-8', errors='ignore').replace('\\n', ' ')\n",
            '    for m in sentence_re.finditer(text):\n',
            '        s = m.group().strip()\n',
            '        wlen = len(s.split())\n',
            '        if 3 <= wlen <= 50:\n',
            '            sentences.append(s)\n',
            '\n',
            "print(f'Collected {len(sentences)} sentences')\n",
            "SENTENCE_PKL.parent.mkdir(parents=True, exist_ok=True)\n",
            "with SENTENCE_PKL.open('wb') as f:\n",
            "    pickle.dump(sentences, f)\n",
            "print('Saved to', SENTENCE_PKL)\n",
            'SENTENCE_PKL\n'
        ],
    }
    insert_at = 2 if len(nb['cells']) > 2 else len(nb['cells'])
    nb['cells'][insert_at:insert_at] = [md, code]
    nb_path.write_text(json.dumps(nb, indent=2))
    print('appended rebuild cells at', insert_at)
