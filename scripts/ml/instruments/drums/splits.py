"""Train / validation / held-out split for the drum student.

held-out eval : common.is_eval() (13 test songs + every 5th owner track). Never trained or tuned on.
validation    : owner train tracks with NNN % 10 == 3 (~1 in 8 of the non-eval own tracks) plus FMA ids
                whose integer id % 20 == 7 (~5%). Used for early stopping and for tuning the peak
                picker (threshold / refractory), so tuning is not done on data the model has memorised.
train         : everything else that has an ADTOF label with the tom/cymbal split and a feature cache.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import common
from dataset import load_labels, has_split, cache_path


def split_of(corpus, tid, path):
    if common.is_eval(corpus, tid, path):
        return 'eval'
    if corpus == 'own':
        m = re.match(r'(\d+)', os.path.basename(path))
        return 'val' if m and int(m.group(1)) % 10 == 3 else 'train'
    if corpus == 'fma':
        return 'val' if int(tid) % 20 == 7 else 'train'
    return 'train'


def rows(corpora=('test', 'own', 'fma'), need_cache=True):
    out = []
    for corpus in corpora:
        for tid, path in common.tracks(corpus):
            lab = load_labels(corpus, tid)
            if not has_split(lab):
                continue
            if need_cache and not os.path.exists(cache_path(corpus, tid)):
                continue
            out.append({'corpus': corpus, 'tid': tid, 'path': path, 'split': split_of(corpus, tid, path)})
    return out
