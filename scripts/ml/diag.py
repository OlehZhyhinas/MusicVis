"""Frame-level diagnostic of the fold models on their held-out songs (and 2 training songs for the
over-fitting gap): beat / bar phase within 0.1 cycle of the offline labels, median phase error,
beat-activation correlation.

  .testdata/ml/venv/bin/python scripts/ml/diag.py gru128
"""
import json, os, sys
import numpy as np
import onnxruntime as ort

R = os.path.join(os.path.dirname(__file__), '../../.testdata/ml/')
idx = json.load(open(R + 'data/index.json')); LAB = idx['labels']; li = {n: i for i, n in enumerate(LAB)}
name = sys.argv[1]
folds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
rows = {'held': [], 'train': []}
for f in range(folds):
    m = json.load(open(R + f'models/{name}-f{f}of{folds}.json'))
    s = ort.InferenceSession(R + f'models/{name}-f{f}of{folds}.onnx')
    for split, songs in (('held', m['held']), ('train', [x for x in m['train'] if '@' not in x][:2])):
        for song in songs:
            x = np.fromfile(R + f'data/{song}.f32', dtype=np.float32).reshape(-1, 160)
            y = np.fromfile(R + f'data/{song}.lab.f32', dtype=np.float32).reshape(-1, len(LAB))
            h = np.zeros((m['layers'], 1, m['hidden']), np.float32); out = []
            for k in range(len(x)):
                o, h = s.run(None, {'x': x[k:k + 1][None], 'h': h}); out.append(o[0, 0])
            o = np.array(out); v = y[:, li['beatValid']] > 0
            e = np.arctan2(o[:, 3], o[:, 2]) - np.arctan2(y[:, li['beatSin']], y[:, li['beatCos']]); e = (e / (2 * np.pi) + 0.5) % 1 - 0.5
            be = np.arctan2(o[:, 5], o[:, 4]) - np.arctan2(y[:, li['barSin']], y[:, li['barCos']]); be = (be / (2 * np.pi) + 0.5) % 1 - 0.5
            c = np.corrcoef(o[:, 0], y[:, li['beat']])[0, 1]
            r = (np.mean(np.abs(e[v]) < 0.1), np.mean(np.abs(be[v]) < 0.1), c)
            rows[split].append(r)
            print(f'{split} {song[:24]:24s} beat in-phase {r[0]:.2f} bar {r[1]:.2f} median err {np.median(e[v]):+.2f} act corr {c:.2f} |phase vec| {np.mean(np.hypot(o[v, 2], o[v, 3])):.2f}', flush=True)
for k, v in rows.items():
    a = np.array(v); print(k, 'mean beat in-phase %.2f bar %.2f act corr %.2f' % tuple(a.mean(0)))
