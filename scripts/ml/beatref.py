"""Beat reference labeller for the parity tools (test-only; nothing of it ships).

Runs Beat This! (CPJKU/beat_this, checkpoint final0, no DBN; offline, non-causal) over every
decoded PCM in .testdata/live/pcm/<key>.f32 (interleaved stereo float32, 44.1 kHz, written by
scripts/live/common.ts decode() and withBeatRef()) that has no .testdata/live/beatref/<key>.json yet; also keeps a shared copy in
~/personal/MusicVis-data/labels/beatthis/test/<key>.json for the other test tools.
Audio is read from disk, never played.

Setup once (Python 3.10/3.11; beat_this does not install on 3.14):
  uv venv --python 3.11 .testdata/ml/venv-beatthis
  uv pip install --python .testdata/ml/venv-beatthis torch torchaudio soundfile numpy \
      git+https://github.com/CPJKU/beat_this
Run:
  .testdata/ml/venv-beatthis/bin/python scripts/ml/beatref.py [key ...]
"""
import json, os, sys, time
import numpy as np
import torch
from beat_this.inference import Audio2Beats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../.testdata/live')
PCM = os.path.join(ROOT, 'pcm')
OUT = os.path.join(ROOT, 'beatref')
os.makedirs(OUT, exist_ok=True)
# Shared copy for the other test tools (AVQ): ~/personal/MusicVis-data/labels/beatthis/test/<key>.json
SHARED = os.path.expanduser('~/personal/MusicVis-data/labels/beatthis/test')
os.makedirs(SHARED, exist_ok=True)

keys = sys.argv[1:] or sorted(f[:-4] for f in os.listdir(PCM) if f.endswith('.f32'))
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
a2b = Audio2Beats(checkpoint_path='final0', device=dev, dbn=False)
for k in keys:
    out = os.path.join(OUT, k + '.json')
    shared = os.path.join(SHARED, k + '.json')
    if os.path.exists(out):
        if not os.path.exists(shared):
            with open(out) as fi, open(shared, 'w') as fo: fo.write(fi.read())
        continue
    t0 = time.time()
    x = np.fromfile(os.path.join(PCM, k + '.f32'), dtype='<f4')
    x = x[: len(x) // 2 * 2].reshape(-1, 2).mean(1)
    beats, downbeats = a2b(x, 44100)
    j = {'model': 'beat-this final0 (dbn=False)', 'key': k, 'beats': [round(float(b), 4) for b in beats],
         'downbeats': [round(float(d), 4) for d in downbeats]}
    for p in (out, shared):
        json.dump(j, open(p, 'w'))
    print(f'{k}: {len(beats)} beats, {len(downbeats)} downbeats, {time.time() - t0:.1f} s', flush=True)
