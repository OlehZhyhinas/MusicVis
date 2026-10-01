"""Score the teachers and the live/offline DSP path against the aligned ground truth.

References (labels/midi/<corpus>/<id>.json):
  MIDI   aligned notes / drums / beat grid / note-derived chords, only inside "regions"
  HT     Hooktheory human chords, melody and beats, inside each segment's audio span
Estimates:
  beatthis (beats, downbeats), madmom-cnn-crf / madmom-dc-crf / causal-deepchroma (chords),
  basicpitch on htdemucs_6s stems (notes per stem), adtof (kick/snare/hat),
  live + offline DSP (onBeat / onBar / chord from scripts/ml/groundtruth/live_dump.ts, when dumped).
Metrics: mir_eval beat F (70 ms), downbeat F (70 ms), chord majmin / root accuracy on a 10 Hz grid,
note onset F (50 ms, pitch within 50 cents; pitch class only for HT melody), drum onset F (50 ms).

Run: .testdata/gt/venv/bin/python scripts/ml/groundtruth/eval_teachers.py [--md]   (writes WORK/eval.json)
"""
import argparse, collections, glob, json, os, sys
import numpy as np
import mir_eval
sys.path.insert(0, os.path.dirname(__file__))
import gt_common as g

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
LIVE = os.path.join(REPO, '.testdata/gt/live')
IDX = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def in_regions(t, regs, pad=0.0):
    t = np.asarray(t, float)
    m = np.zeros(len(t), bool)
    for a, b in regs:
        m |= (t >= a - pad) & (t <= b + pad)
    return m


def beat_f(ref, est, regs):
    ref = np.asarray(ref, float); est = np.asarray(est, float)
    ref = ref[in_regions(ref, regs)]; est = est[in_regions(est, regs, 0.035)]
    if len(ref) < 4:
        return None
    return float(mir_eval.beat.f_measure(ref, est, 0.07)) if len(est) else 0.0


def beat_offset(ref, est, regs):
    """Median signed offset (ms, est - ref) of estimated beats matched within 70 ms."""
    ref = np.asarray(ref, float); est = np.sort(np.asarray(est, float))
    ref = ref[in_regions(ref, regs)]
    if len(ref) < 8 or len(est) < 2:
        return None
    k = np.clip(np.searchsorted(est, ref), 1, len(est) - 1)
    near = np.where(np.abs(est[k] - ref) < np.abs(est[k - 1] - ref), est[k], est[k - 1])
    dlt = near - ref
    dlt = dlt[np.abs(dlt) < 0.07]
    return float(np.median(dlt) * 1000) if len(dlt) >= 8 else None


def onset_f(ref, est, regs, win=0.05):
    ref = np.sort(np.asarray(ref, float)); est = np.sort(np.asarray(est, float))
    ref = ref[in_regions(ref, regs)]; est = est[in_regions(est, regs, win)]
    if len(ref) < 4:
        return None
    return float(mir_eval.onset.f_measure(ref, est, window=win)[0]) if len(est) else 0.0


def notes_f(ref, est, regs, pc_only=False):
    """ref/est: [[on, off, midi, ...]]; onset-only F with 50 ms onset tolerance."""
    ref = np.asarray(ref, float).reshape(-1, max(3, np.asarray(ref).shape[-1] if len(ref) else 3))
    est = np.asarray(est, float).reshape(-1, max(3, np.asarray(est).shape[-1] if len(est) else 3))
    if len(ref):
        ref = ref[in_regions(ref[:, 0], regs)]
    if len(est):
        est = est[in_regions(est[:, 0], regs, 0.05)]
    if len(ref) < 8:
        return None
    if not len(est):
        return 0.0
    rp, ep = ref[:, 2].copy(), est[:, 2].copy()
    if pc_only:
        rp = 60 + rp % 12; ep = 60 + ep % 12
    ri = np.stack([ref[:, 0], np.maximum(ref[:, 1], ref[:, 0] + 0.01)], 1)
    ei = np.stack([est[:, 0], np.maximum(est[:, 1], est[:, 0] + 0.01)], 1)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, mir_eval.util.midi_to_hz(rp), ei, mir_eval.util.midi_to_hz(ep), onset_tolerance=0.05, offset_ratio=None)
    return float(f)


def frames(segs, grid):
    """Label per grid time from [[t0, t1, label]] (else 'N')."""
    out = np.array(['N'] * len(grid), dtype=object)
    for a, b, l in segs:
        out[(grid >= a) & (grid < b)] = l
    return out


def chord_acc(ref_segs, est_segs, regs):
    if not ref_segs or not regs:
        return None
    grid = np.concatenate([np.arange(a, b, 0.1) for a, b in regs])
    if len(grid) < 50:
        return None
    r = frames(ref_segs, grid); e = frames(est_segs, grid)
    res = {}
    for name, fn in (('majmin', mir_eval.chord.majmin), ('root', mir_eval.chord.root)):
        s = np.asarray(fn(list(r), list(e)), float)
        ok = s >= 0
        res[name] = float(s[ok].mean()) if ok.sum() > 20 else None
    return res


def idx_to_label(i):
    i = int(i)
    if i < 0:
        return 'N'
    return f'{IDX[i]}:maj' if i < 12 else f'{IDX[i - 12]}:min'


def live_segments(d, which, key):
    ci = d['channels'].index(key)
    arr = np.asarray(d[which][ci]); fps = d['fps']
    segs, start = [], 0
    for i in range(1, len(arr) + 1):
        if i == len(arr) or arr[i] != arr[start]:
            segs.append([(start + 1) / fps, (i + 1) / fps, idx_to_label(arr[start])])
            start = i
    return segs


def live_events(d, which, key):
    ci = d['channels'].index(key)
    arr = np.asarray(d[which][ci])
    on = np.nonzero((arr[1:] > 0.5) & (arr[:-1] <= 0.5))[0] + 1
    return (on + 1) / d['fps']


def teacher(name, corpus, tid, slug, slug_count):
    if name == 'beatthis':
        p = os.path.join(g.LABELS, name, corpus, tid + '.json')
    else:
        if corpus == 'own' and slug_count[slug] > 1:
            return None                         # slug-keyed caches are ambiguous for colliding own titles
        p = os.path.join(g.LABELS, name, corpus, slug + '.json')
    return json.load(open(p)) if os.path.exists(p) else None


STEM_FAM = {'bass': ['Bass'], 'piano': ['Piano', 'Organ', 'Chromatic Percussion'], 'guitar': ['Guitar']}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--md', action='store_true'); a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(g.GT, '*', '*.json')))
    slug_count = collections.Counter(json.load(open(f))['slug'] for f in files)
    rows = []
    for f in files:
        d = json.load(open(f))
        corpus, tid, slug = d['corpus'], d['id'], d['slug']
        R = dict(id=tid, corpus=corpus)
        regs = d['regions']
        bt = teacher('beatthis', corpus, tid, slug, slug_count)
        bp = teacher('basicpitch', corpus, tid, slug, slug_count)
        ad = teacher('adtof', corpus, tid, slug, slug_count)
        chords_est = {n: teacher(n, corpus, tid, slug, slug_count) for n in ('madmom-cnn-crf', 'madmom-dc-crf', 'causal-deepchroma')}
        lv = os.path.join(LIVE, corpus, tid + '.json')
        lv = json.load(open(lv)) if os.path.exists(lv) else None
        # ---- MIDI reference
        if d.get('source') and regs:
            if d['beats_ok']:
                ref_b = [b[0] for b in d['beats']]
                ref_d = d['downbeats']
                if bt:
                    R['beat/beatthis'] = beat_f(ref_b, bt['beats'], regs)
                    R['beat_offset_ms/beatthis'] = beat_offset(ref_b, bt['beats'], regs)
                    R['downbeat/beatthis'] = beat_f(ref_d, bt.get('downbeats', []), regs)
                if lv:
                    for w in ('live', 'off'):
                        R[f'beat/{w}-dsp'] = beat_f(ref_b, live_events(lv, w, 'onBeat'), regs)
                        R[f'downbeat/{w}-dsp'] = beat_f(ref_d, live_events(lv, w, 'onBar'), regs)
            for n, e in chords_est.items():
                if e:
                    R[f'chord/{n}'] = chord_acc(d['chords'], e['segments'], regs)
            if lv:
                for w in ('live', 'off'):
                    R[f'chord/{w}-dsp'] = chord_acc(d['chords'], live_segments(lv, w, 'chord'), regs)
            pitched = {}
            for ins in d['instruments']:
                if not ins['is_drum']:
                    pitched.setdefault(ins['family'], []).extend(ins['notes'])
            allp = [n for v in pitched.values() for n in v]
            if bp:
                for stem, fams in STEM_FAM.items():
                    ref = [n for fm in fams for n in pitched.get(fm, [])]
                    if stem in bp and ref:
                        R[f'notes/basicpitch-{stem}'] = notes_f(ref, bp[stem], regs)
                est_all = [n for s in ('vocals', 'bass', 'guitar', 'piano', 'other') for n in bp.get(s, [])]
                R['notes/basicpitch-allstems'] = notes_f(allp, est_all, regs)
                R['notes/basicpitch-allstems-pc'] = notes_f(allp, est_all, regs, pc_only=True)
                bass_ref = [n for n in pitched.get('Bass', [])]
                if 'bass' in bp and bass_ref:
                    R['notes/basicpitch-bass-pc'] = notes_f(bass_ref, bp['bass'], regs, pc_only=True)
                if 'mix' in bp:
                    R['notes/basicpitch-mix'] = notes_f(allp, bp['mix'], regs)
            if ad and any(d['drums'].get(k) for k in ('kick', 'snare', 'hihat')):
                for k, ka in (('kick', 'kick'), ('snare', 'snare'), ('hihat', 'hat')):
                    if len(d['drums'].get(k, [])) >= 8:
                        R[f'drums/adtof-{k}'] = onset_f(d['drums'][k], ad.get(ka, []), regs)
        # ---- Hooktheory reference (human)
        for s in d.get('hooktheory', []):
            sreg = [s['audio']]
            if s['beats']:
                if d.get('beats_ok') and d['beats']:
                    R.setdefault('ht_beat/midi-grid', []).append(beat_f([b[0] for b in s['beats']], [b[0] for b in d['beats']], sreg))
                    R.setdefault('ht_downbeat/midi-grid', []).append(beat_f([b[0] for b in s['beats'] if b[1]], d['downbeats'], sreg))
                if bt:
                    R.setdefault('ht_beat/beatthis', []).append(beat_f([b[0] for b in s['beats']], bt['beats'], sreg))
                    R.setdefault('ht_downbeat/beatthis', []).append(beat_f([b[0] for b in s['beats'] if b[1]], bt.get('downbeats', []), sreg))
                if lv:
                    R.setdefault('ht_beat/live-dsp', []).append(beat_f([b[0] for b in s['beats']], live_events(lv, 'live', 'onBeat'), sreg))
            if s['chords']:
                for n, e in chords_est.items():
                    if e:
                        R.setdefault(f'ht_chord/{n}', []).append(chord_acc(s['chords'], e['segments'], sreg))
                if lv:
                    for w in ('live', 'off'):
                        R.setdefault(f'ht_chord/{w}-dsp', []).append(chord_acc(s['chords'], live_segments(lv, w, 'chord'), sreg))
                if d.get('source') and d['chords']:
                    ov = [[max(a0, s['audio'][0]), min(b0, s['audio'][1])] for a0, b0 in regs
                          if min(b0, s['audio'][1]) - max(a0, s['audio'][0]) > 4]
                    if ov:
                        R.setdefault('ht_chord/midi-notes2chords', []).append(chord_acc(s['chords'], d['chords'], ov))
            if s['melody'] and bp and 'vocals' in bp:
                R.setdefault('ht_melody/basicpitch-vocals-pc', []).append(
                    notes_f([[m[0], m[1], m[2]] for m in s['melody']], bp['vocals'], sreg, pc_only=True))
        rows.append(R)
    # aggregate (song-level means; chord dicts averaged per sub-metric)
    agg = collections.defaultdict(list)
    for R in rows:
        for k, v in R.items():
            if k in ('id', 'corpus'):
                continue
            vs = v if isinstance(v, list) else [v]
            for x in vs:
                if x is None:
                    continue
                if isinstance(x, dict):
                    for kk, vv in x.items():
                        if vv is not None:
                            agg[f'{k}:{kk}'].append(vv)
                else:
                    agg[k].append(x)
    table = {k: dict(mean=round(float(np.mean(v)), 3), median=round(float(np.median(v)), 3), n=len(v)) for k, v in sorted(agg.items())}
    g.save_json(dict(table=table, rows=rows), os.path.join(g.WORK, 'eval.json'), indent=1)
    w = max(len(k) for k in table) if table else 10
    for k, v in table.items():
        print(f'{k:{w}s}  mean {v["mean"]:.3f}  median {v["median"]:.3f}  n={v["n"]}')
    if a.md:
        print('\n| metric | mean | median | n |\n|---|---|---|---|')
        for k, v in table.items():
            print(f'| {k} | {v["mean"]:.3f} | {v["median"]:.3f} | {v["n"]} |')


if __name__ == '__main__':
    main()
