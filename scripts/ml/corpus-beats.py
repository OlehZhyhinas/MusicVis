"""Beat This! teacher labels (beats, downbeats) for a training corpus, same ids as corpus-feats.ts
(the path under the audio dir, '/' -> '__', no extension). Audio decoded with ffmpeg, never played.

  <beat_this venv>/bin/python scripts/ml/corpus-beats.py <audio dir> <out dir> [--shard i/N]

Writes <out dir>/<id>.json {beats, downbeats}; skips existing outputs.
"""
import json, os, subprocess, sys, time
import numpy as np
import torch
from beat_this.inference import Audio2Beats

args = [a for a in sys.argv[1:] if not a.startswith('--')]
src, out = args[0], args[1]
shard = (0, 1)
if '--shard' in sys.argv:
    i, n = sys.argv[sys.argv.index('--shard') + 1].split('/'); shard = (int(i), int(n))
os.makedirs(out, exist_ok=True)
files = []
for d, _, fs in os.walk(src):
    for f in fs:
        if f.lower().endswith(('.mp3', '.m4a', '.wav', '.flac', '.ogg', '.opus')):
            files.append(os.path.join(d, f))
files.sort()
files = files[shard[0]::shard[1]]
FF = '/opt/homebrew/bin/ffmpeg'
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
a2b = Audio2Beats(checkpoint_path='final0', device=dev, dbn=False)
t0 = time.time(); n = 0
for p in files:
    rid = os.path.splitext(os.path.relpath(p, src))[0].replace(os.sep, '__')
    o = os.path.join(out, rid + '.json')
    if os.path.exists(o):
        continue
    try:
        raw = subprocess.run([FF, '-v', 'error', '-i', p, '-f', 'f32le', '-ac', '1', '-ar', '44100', '-'], capture_output=True, check=True).stdout
        x = np.frombuffer(raw, dtype='<f4')
        if len(x) < 44100 * 5:
            continue
        beats, downbeats = a2b(x, 44100)
    except Exception as e:  # corrupt files exist in FMA
        print(rid, 'failed', e, flush=True); continue
    json.dump({'beats': [round(float(b), 4) for b in beats], 'downbeats': [round(float(d), 4) for d in downbeats]}, open(o, 'w'))
    n += 1
    if n % 200 == 0:
        print(f'{n} labelled, {(time.time() - t0) / n:.2f} s each', flush=True)
print('done', n, flush=True)
