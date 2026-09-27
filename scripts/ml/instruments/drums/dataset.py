"""Cached features + on-the-fly frame targets for the drum transcription student.

Features: features_np.features_of (causal log-band frontend mirroring scripts/ml/features.ts,
80 log bands + flux, 2048 FFT / 512 hop / 44.1 kHz mono mid). Cached as float16 in
.testdata/instr/drums/cache/<corpus>/<id>.npz {feat} (older cache files also carry a 'lab'
array, which is ignored now: targets are rebuilt from the label JSON at load time).

Labels: ADTOF teacher onset times, ~/personal/MusicVis-data/labels/adtof/<corpus>/<id>.json
(kick/snare/hat, plus the tom/cymbal keys that add_tom_cymbal.py adds next to 'other').

TIME ALIGNMENT (the main bug of the first version). Feature frame k is computed from the 2048-sample
Hann window ending at sample (k+1)*512, i.e. frame k "is reported" at time (k+1)/fps. An onset at
time t first enters a window at frame k0 = floor(t*fps) (in its low-weight Hann tail) and its flux
peaks at k0+1 (onset near the window centre). Cross-correlating low/mid/high band flux with the
ADTOF onset trains on 80 tracks puts the peak at round(t*fps)+1 for snare and hat and +1..+3 for
kick (so ADTOF's times are fine and in seconds from the file start). The first version put the
target at round(t*fps) - 1, a frame whose window contains NO evidence of the onset yet, which a
causal model can only meet by guessing from rhythm: it learned broad, early, smeared activations.
Now the target centre is round(t*fps) + SHIFT (default 0: the first frame with evidence; the
prediction time is then (k+1)/fps ~ t + 6-17 ms).

  <venv>/bin/python dataset.py [own|test|fma] [--limit N] [--jobs 6]
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import common
from features_np import features_of, ML_FEAT, ML_HOP, ML_SR, ML_FPS

CLASSES = ['kick', 'snare', 'hat', 'tom', 'cymbal']
CACHE = os.path.join(common.WORK, 'drums', 'cache')
SHIFT = 0


def load_labels(corpus, tid):
    p = os.path.join(common.LABELS, 'adtof', corpus, f'{tid}.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def has_split(labels):
    return labels is not None and 'tom' in labels and 'cymbal' in labels


def build_targets(labels, T, shift=SHIFT, widen=0.5):
    """[T, 5] float32: 1.0 at frame round(t*fps)+shift, `widen` on the two neighbours."""
    lab = np.zeros((T, len(CLASSES)), dtype=np.float32)
    for ci, cls in enumerate(CLASSES):
        for t in labels.get(cls, []):
            k0 = int(round(t * ML_FPS)) + shift
            for d in (-1, 0, 1):
                k = k0 + d
                if 0 <= k < T:
                    v = 1.0 if d == 0 else widen
                    if v > lab[k, ci]:
                        lab[k, ci] = v
    return lab


def cache_path(corpus, tid):
    return os.path.join(CACHE, corpus, f'{tid}.npz')


def load_feat(corpus, tid):
    p = cache_path(corpus, tid)
    if not os.path.exists(p):
        return None
    return np.load(p)['feat']


def build_one(args):
    corpus, tid, path = args
    out = cache_path(corpus, tid)
    if os.path.exists(out):
        return 'cached'
    try:
        x = common.decode(path, ML_SR, 1)
        feat = features_of(x).astype(np.float16)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out[:-4] + '.tmp.npz'
        np.savez(tmp, feat=feat)
        os.replace(tmp, out)
        return 'built'
    except Exception as e:
        return f'FAILED {e!r}'[:200]


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else 'own'
    limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None
    jobs = int(sys.argv[sys.argv.index('--jobs') + 1]) if '--jobs' in sys.argv else 6
    todo = []
    for tid, path in common.tracks(corpus):
        if load_labels(corpus, tid) is None or os.path.exists(cache_path(corpus, tid)):
            continue
        todo.append((corpus, tid, path))
    if limit:
        todo = todo[:limit]
    t0 = time.time()
    from multiprocessing import Pool
    res = {}
    with Pool(jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(build_one, todo, chunksize=4)):
            k = r.split(' ')[0]
            res[k] = res.get(k, 0) + 1
            if r.startswith('FAILED'):
                print(r)
    print(f'{corpus}: {res} in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
