import json
from pathlib import Path

nb_path = Path('notebooks/process_lexicon.ipynb')
nb = json.loads(nb_path.read_text())

if any('librispeech_sentences.pkl' in ''.join(c.get('source','')) for c in nb['cells']):
    print('processing cell already present')
else:
    nb['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '# Rebuild LibriSpeech sentences\n',
            'Use LibriSpeech text in notebooks/data to regenerate data/librispeech_sentences.pkl.'
        ],
    })
    nb['cells'].append({
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': [
            'from pathlib import Path\n',
            'import re, pickle\n',
            "ROOT = Path.cwd()\n",
            "BOOKS_DIR = ROOT / 'notebooks' / 'data' / 'LibriSpeech' / 'books' / 'ascii'\n",
            "SENTENCE_OUT = ROOT / 'data' / 'librispeech_sentences.pkl'\n",
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
            "SENTENCE_OUT.parent.mkdir(parents=True, exist_ok=True)\n",
            "with SENTENCE_OUT.open('wb') as f:\n",
            "    pickle.dump(sentences, f)\n",
            "print('Saved to', SENTENCE_OUT)\n",
            'SENTENCE_OUT\n',
        ],
    })
    nb_path.write_text(json.dumps(nb, indent=2))
    print('appended; cells now', len(nb['cells']))
