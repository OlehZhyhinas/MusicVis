"""Dump peakpick.py onset frames (with the tuned per-class params from eval2-<tag>.json) for the
activation reference written by dump_torch_ref.py, for peak_parity.ts.
  <venv>/bin/python peak_parity_ref.py <tag> <corpus> <tid>
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import common
from model import CLASSES
from peakpick import peak_pick

W = os.path.join(common.WORK, 'drums')
tag, corpus, tid = sys.argv[1:4]
params = json.load(open(os.path.join(W, f'eval2-{tag}.json')))['summary']['peak_params']
act = np.fromfile(os.path.join(W, 'parity', f'{tag}-{corpus}-{tid}.ref.f32'), dtype='<f4').reshape(-1, len(CLASSES))
out = {'classes': CLASSES, 'params': {}, 'events': {}}
for ci, c in enumerate(CLASSES):
    p = {k: (float(v) if k in ('threshold', 'delta') else int(v)) for k, v in params[c].items() if k in ('threshold', 'refractory', 'lookahead', 'past', 'delta')}
    out['params'][c] = p
    out['events'][c] = peak_pick(act[:, ci], p['threshold'], p['refractory'], p['lookahead'], p['past'], p['delta'])
json.dump(out, open(os.path.join(W, 'parity', f'{tag}-{corpus}-{tid}.peaks.json'), 'w'))
print({c: len(v) for c, v in out['events'].items()})
