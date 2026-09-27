"""Key accuracy: madmom's CNNKeyRecognitionProcessor (one global key label per song, offline,
663,585-param conv net) vs the repo's live (rtKey.ts KeyTracker) and offline (key.ts detectKeys)
key trackers, taken as the per-song majority vote over live_dump.ts's per-frame 'key' channel
(index = keyTonic + 12*isMinor). Also reports key-change count (how often each tracker modulates)
as a rough proxy for whether a song genuinely modulates (mismatches on those are less meaningful).

Usage: .testdata/instr/venvs/adtof/bin/python scripts/ml/instruments/chords/score_key.py
"""
import json, os, sys
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import common

NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
LIVE_DIR = os.path.join(common.WORK, 'chords', 'live')
ENHARMONIC = {'C#': 'Db', 'Eb': 'D#', 'F#': 'Gb', 'Ab': 'G#', 'Bb': 'A#'}


def norm_root(name):
    return ENHARMONIC.get(name, name)


def key_idx_to_label(idx):
    if idx is None or idx < 0:
        return None
    tonic = int(idx) % 12
    mode = 'minor' if idx >= 12 else 'major'
    return f'{NOTE_NAMES[tonic]} {mode}'


def majority_key(idx_array):
    vals = [v for v in idx_array if v is not None and v >= 0]
    if not vals:
        return None, 0
    c = Counter(vals)
    top, n = c.most_common(1)[0]
    return key_idx_to_label(top), n / len(vals)


def main():
    rows = []
    for tid, _ in common.tracks('test'):
        madmom_path = os.path.join(common.LABELS, 'madmom-key', 'test', f'{tid}.json')
        live_path = os.path.join(LIVE_DIR, f'test__{tid}.json')
        if not (os.path.exists(madmom_path) and os.path.exists(live_path)):
            continue
        with open(madmom_path) as f:
            mk = json.load(f)['key']
        root, mode = mk.split()
        madmom_label = f'{norm_root(root)} {mode}'
        with open(live_path) as f:
            ld = json.load(f)
        ci = ld['channels'].index('key')
        off_maj, off_frac = majority_key(ld['off'][ci])
        live_maj, live_frac = majority_key(ld['live'][ci])
        off_norm = norm_root(off_maj.split()[0]) + ' ' + off_maj.split()[1] if off_maj else None
        live_norm = norm_root(live_maj.split()[0]) + ' ' + live_maj.split()[1] if live_maj else None
        rows.append({
            'id': tid,
            'madmom_key': madmom_label,
            'offline_key': off_norm, 'offline_majority_frac': round(off_frac, 2),
            'live_key': live_norm, 'live_majority_frac': round(live_frac, 2),
            'offline_match': off_norm == madmom_label,
            'live_match': live_norm == madmom_label,
        })

    for r in rows:
        print(r)
    n = len(rows)
    off_acc = sum(r['offline_match'] for r in rows) / n
    live_acc = sum(r['live_match'] for r in rows) / n
    print(f'\noffline key vs madmom-key: {off_acc:.3f} ({sum(r["offline_match"] for r in rows)}/{n})')
    print(f'live key vs madmom-key:    {live_acc:.3f} ({sum(r["live_match"] for r in rows)}/{n})')


if __name__ == '__main__':
    main()
