"""Causal downbeat evidence from madmom's (bidirectional, offline) downbeat RNN run on a sliding
window of PAST audio: every H s the last W s are analysed, and the downbeat activations of the
frames D..D+H s before the window end are kept (they had D s of right context). So a frame's value
is known D + H s after it. Written at 100 fps to .testdata/ml/downbeat/<tag>/<slug>.f32 for
scripts/ml/eval.ts --downbeat <tag>.

  <madmom venv>/bin/python scripts/ml/downbeat-window.py [--W 6] [--H 1] [--D 1.5] [--nets 8] [slug ...]
"""
import argparse, os, time
import numpy as np
from madmom.audio.signal import Signal
from madmom.features.downbeats import RNNDownBeatProcessor

ap = argparse.ArgumentParser()
ap.add_argument('--W', type=float, default=6)
ap.add_argument('--H', type=float, default=1)
ap.add_argument('--D', type=float, default=1.5)
ap.add_argument('--nets', type=int, default=8)
ap.add_argument('slugs', nargs='*')
a = ap.parse_args()
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
PCM = os.path.join(ROOT, '.testdata/live/pcm')
tag = f'w{a.W:g}-h{a.H:g}-d{a.D:g}-n{a.nets}'
OUT = os.path.join(ROOT, '.testdata/ml/downbeat', tag)
os.makedirs(OUT, exist_ok=True)
proc = RNNDownBeatProcessor()
ens = proc.processors[1].processors[0]
ens.processors = ens.processors[:a.nets]  # fewer networks of the ensemble
slugs = a.slugs or sorted(f[:-4] for f in os.listdir(PCM) if f.endswith('.f32'))
FPS, SR = 100, 44100
for s in slugs:
    out = os.path.join(OUT, s + '.f32')
    if os.path.exists(out):
        continue
    t0 = time.time()
    x = np.fromfile(os.path.join(PCM, s + '.f32'), dtype='<f4')
    x = x[: len(x) // 2 * 2].reshape(-1, 2).mean(1)
    n = int(len(x) / SR * FPS)
    db = np.zeros(n, np.float32)
    bt = np.zeros(n, np.float32)
    end = a.W
    calls = 0
    while end <= len(x) / SR:
        seg = Signal(x[int((end - a.W) * SR): int(end * SR)], sample_rate=SR)
        act = proc(seg)  # (frames, 2): beat, downbeat
        calls += 1
        f0 = int(round((end - a.D - a.H) * FPS)); f1 = int(round((end - a.D) * FPS))
        base = int(round((end - a.W) * FPS))
        for f in range(max(f0, 0), min(f1, n)):
            i = f - base
            if 0 <= i < len(act):
                db[f] = act[i, 1]; bt[f] = act[i, 0]
        end += a.H
    db.astype('<f4').tofile(out)
    dt = time.time() - t0
    print(f'{s}: {calls} windows, {dt:.1f} s ({dt / calls * 1000:.0f} ms per window)', flush=True)
