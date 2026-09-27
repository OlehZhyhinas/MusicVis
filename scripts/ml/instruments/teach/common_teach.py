"""Shared helpers for the offline teacher-label runners (teach/*.py).

All runners follow the same processing order: 'test' songs first (all
teachers), then 'own' (eval tracks i.e. is_eval()==True first, then the
rest), then 'fma' (only once numbered folders exist under
~/personal/MusicVis-data/fma/fma_medium/, up to ~1500 clips spread across
folders). Every runner re-lists tracks on each pass (common.tracks() already
does this cheaply) and skips ids whose output file already exists, so they
are safe to stop/restart and pick up newly downloaded files.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import common

FMA_ROOT = os.path.join(common.DATA, 'fma', 'fma_medium')
FMA_CAP = 1500


def atomic_write_bytes(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_write_text(path, text):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, path)


def fma_ready():
    """True once at least one numbered subfolder with mp3s exists."""
    if not os.path.isdir(FMA_ROOT):
        return False
    for sub in os.listdir(FMA_ROOT):
        d = os.path.join(FMA_ROOT, sub)
        if os.path.isdir(d) and any(f.endswith('.mp3') for f in os.listdir(d)):
            return True
    return False


def ordered_plan():
    """Yield (corpus, tid, path) tuples in the required processing order:
    all test, then own-eval, then own-train, then fma (capped, spread across
    numbered folders). Re-lists each corpus fresh (called once per pass)."""
    for tid, path in common.tracks('test'):
        yield 'test', tid, path

    own = common.tracks('own')
    own_eval = [(t, p) for t, p in own if common.is_eval('own', t, p)]
    own_train = [(t, p) for t, p in own if not common.is_eval('own', t, p)]
    # Interleaved 1:2 (eval:train) so the note/instrument students get
    # training labels flowing early instead of waiting for all ~40 eval
    # tracks first; once eval is exhausted the rest of train follows.
    ei, ti = 0, 0
    while ei < len(own_eval) or ti < len(own_train):
        if ei < len(own_eval):
            yield 'own', own_eval[ei][0], own_eval[ei][1]
            ei += 1
        for _ in range(2):
            if ti < len(own_train):
                yield 'own', own_train[ti][0], own_train[ti][1]
                ti += 1

    if fma_ready():
        fma = common.tracks('fma')
        # spread across numbered folders: fma is already sorted by folder
        # then id since tracks() walks folders in sorted order; interleave
        # instead of taking a contiguous prefix so a cap still spans folders.
        by_folder = {}
        for tid, path in fma:
            folder = os.path.basename(os.path.dirname(path))
            by_folder.setdefault(folder, []).append((tid, path))
        folders = sorted(by_folder)
        i = 0
        count = 0
        cols = [by_folder[f] for f in folders]
        idx = [0] * len(cols)
        while count < FMA_CAP:
            progressed = False
            for ci in range(len(cols)):
                if idx[ci] < len(cols[ci]):
                    tid, path = cols[ci][idx[ci]]
                    idx[ci] += 1
                    yield 'fma', tid, path
                    count += 1
                    progressed = True
                    if count >= FMA_CAP:
                        break
            if not progressed:
                break


def log_line(log_path, msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(log_path, 'a') as f:
        f.write(line + '\n')
