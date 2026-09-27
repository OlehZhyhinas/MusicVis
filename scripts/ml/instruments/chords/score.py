"""Score chord/key candidates against madmom teacher labels with mir_eval.chord (root, majmin,
sevenths where supported), duration-weighted, on the held-out set (13 test songs + own eval
tracks). Also computes teacher-vs-teacher agreement and reproduces the repo's own exact-index
parity metric (scripts/live/parity.ts score(), cat branch) for the live tracker using the
per-frame dumps from live_dump.ts.

Usage:
  .testdata/instr/venvs/adtof/bin/python scripts/ml/instruments/chords/score.py
"""
import json, os, sys, glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import common
import mir_eval

NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']  # src/analysis/harmony.ts NAMES
LIVE_DIR = os.path.join(common.WORK, 'chords', 'live')


def chord_index_to_label(idx):
    """harmony.ts ChordIndex: -1 = N, 0-11 major on NAMES[idx], 12-23 minor on NAMES[idx-12]."""
    if idx is None or idx < 0:
        return 'N'
    if idx < 12:
        return f'{NOTE_NAMES[idx]}:maj'
    return f'{NOTE_NAMES[idx - 12]}:min'


def frames_to_intervals(times, idx_array, to_label):
    """Turn a per-frame index array (with a fixed frame period) into (intervals, labels) for mir_eval,
    collapsing consecutive equal labels. JSON-serialized NaN (no-chord/not-yet-set, see
    scripts/live/parity.ts stateChannels()'s 'chord'/'key' reads) comes back as null -> 'N'."""
    intervals, labels = [], []
    n = len(idx_array)
    if n == 0:
        return np.zeros((0, 2)), []
    period = times[1] - times[0] if n > 1 else 1.0
    start = times[0]
    prev_lab = to_label(idx_array[0])
    for i in range(1, n):
        lab = to_label(idx_array[i])
        if lab != prev_lab:
            intervals.append([start, times[i]])
            labels.append(prev_lab)
            start = times[i]
            prev_lab = lab
    intervals.append([start, times[-1] + period])
    labels.append(prev_lab)
    return np.array(intervals, dtype=float), labels


def load_segments_json(path):
    with open(path) as f:
        d = json.load(f)
    segs = d['segments']
    intervals = np.array([[s[0], s[1]] for s in segs], dtype=float)
    labels = [s[2] for s in segs]
    return intervals, labels


def mir_eval_compare(ref_int, ref_lab, est_int, est_lab):
    """root/majmin/sevenths/mirex, duration-weighted, using mir_eval.chord.evaluate machinery."""
    if len(ref_lab) == 0 or len(est_lab) == 0:
        return None
    est_int, est_lab = mir_eval.util.adjust_intervals(est_int, est_lab, ref_int.min(), ref_int.max(), mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD)
    comparisons = {}
    for name, fn in [('root', mir_eval.chord.root), ('majmin', mir_eval.chord.majmin), ('sevenths', mir_eval.chord.sevenths), ('mirex', mir_eval.chord.mirex)]:
        try:
            merged_int, r, e = mir_eval.util.merge_labeled_intervals(ref_int, ref_lab, est_int, est_lab)
            dur = mir_eval.util.intervals_to_durations(merged_int)
            comp = fn(r, e)
            valid = comp >= 0  # majmin/sevenths return -1 for out-of-vocab ref chords
            if valid.sum() == 0:
                comparisons[name] = None
                continue
            comparisons[name] = float(np.average(comp[valid], weights=dur[valid]))
        except Exception as ex:
            comparisons[name] = None
    return comparisons


def score_pair(name_a, ref_int, ref_lab, name_b, est_int, est_lab):
    r = mir_eval_compare(ref_int, ref_lab, est_int, est_lab)
    return r


def main():
    test_ids = [tid for tid, _ in common.tracks('test')]
    own_eval_ids = [tid for tid, p in common.tracks('own') if common.is_eval('own', tid, p)]

    rows = []
    for corpus, ids in [('test', test_ids), ('own', own_eval_ids)]:
        for tid in ids:
            dc_path = os.path.join(common.LABELS, 'madmom-dc-crf', corpus, f'{tid}.json')
            cnn_path = os.path.join(common.LABELS, 'madmom-cnn-crf', corpus, f'{tid}.json')
            causal_path = os.path.join(common.LABELS, 'causal-deepchroma', corpus, f'{tid}.json')
            key_path = os.path.join(common.LABELS, 'madmom-key', corpus, f'{tid}.json')
            live_path = os.path.join(LIVE_DIR, f'{corpus}__{tid}.json')

            have = {
                'dc': os.path.exists(dc_path),
                'cnn': os.path.exists(cnn_path),
                'causal': os.path.exists(causal_path),
                'key': os.path.exists(key_path),
                'live': os.path.exists(live_path),
            }
            if not (have['dc'] or have['cnn']):
                continue

            dc_int = dc_lab = cnn_int = cnn_lab = causal_int = causal_lab = None
            if have['dc']:
                dc_int, dc_lab = load_segments_json(dc_path)
            if have['cnn']:
                cnn_int, cnn_lab = load_segments_json(cnn_path)
            if have['causal']:
                causal_int, causal_lab = load_segments_json(causal_path)

            row = {'id': tid, 'corpus': corpus}

            # teacher vs teacher
            if have['dc'] and have['cnn']:
                row['dc_vs_cnn'] = score_pair('dc', dc_int, dc_lab, 'cnn', cnn_int, cnn_lab)

            # causal deep-chroma vs each teacher (dc is the closer comparison: same DNN, causal decoder vs offline CRF)
            if have['causal'] and have['dc']:
                row['causal_vs_dc'] = score_pair('causal', dc_int, dc_lab, 'dc', causal_int, causal_lab)
            if have['causal'] and have['cnn']:
                row['causal_vs_cnn'] = score_pair('causal', cnn_int, cnn_lab, 'dc', causal_int, causal_lab)

            # live tracker vs each teacher, and vs offline analysis (repo's own exact-index metric)
            if have['live']:
                with open(live_path) as f:
                    ld = json.load(f)
                chan = ld['channels']
                ci = chan.index('chord')
                off_idx = ld['off'][ci]
                live_idx = ld['live'][ci]
                fps = ld['fps']
                times = np.arange(len(off_idx)) / fps
                off_int, off_lab = frames_to_intervals(times, off_idx, chord_index_to_label)
                live_int, live_lab = frames_to_intervals(times, live_idx, chord_index_to_label)
                row['live_vs_offline_exactidx'] = next((s['quality'] for s in ld['parityScores'] if s['key'] == 'chord'), None)
                if have['dc']:
                    row['live_vs_dc'] = score_pair('live', dc_int, dc_lab, 'live', live_int, live_lab)
                if have['cnn']:
                    row['live_vs_cnn'] = score_pair('live', cnn_int, cnn_lab, 'live', live_int, live_lab)
                if off_lab:
                    row['offline_vs_dc'] = score_pair('offline', dc_int, dc_lab, 'off', off_int, off_lab) if have['dc'] else None
                    row['offline_vs_cnn'] = score_pair('offline', cnn_int, cnn_lab, 'off', off_int, off_lab) if have['cnn'] else None

            rows.append(row)

    out_path = os.path.join(common.WORK, 'chords', 'scores.json')
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'wrote {out_path} ({len(rows)} songs)', file=sys.stderr)

    def agg(key, metric):
        vals = [r[key][metric] for r in rows if r.get(key) and r[key].get(metric) is not None]
        return float(np.mean(vals)) if vals else None

    for key in ['dc_vs_cnn', 'causal_vs_dc', 'causal_vs_cnn', 'live_vs_dc', 'live_vs_cnn', 'offline_vs_dc', 'offline_vs_cnn']:
        for metric in ['root', 'majmin', 'sevenths', 'mirex']:
            v = agg(key, metric)
            if v is not None:
                print(f'{key:20s} {metric:10s} {v:.3f}  (n={sum(1 for r in rows if r.get(key) and r[key].get(metric) is not None)})')

    live_exact = [r['live_vs_offline_exactidx'] for r in rows if r.get('live_vs_offline_exactidx') is not None]
    if live_exact:
        print(f'live_vs_offline exact-index (repo parity metric): mean={np.mean(live_exact):.3f} n={len(live_exact)}')


if __name__ == '__main__':
    main()
