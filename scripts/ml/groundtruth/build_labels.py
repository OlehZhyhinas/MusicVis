"""Stage 2: full structure-aware alignment of the screened candidates and label export.

For every track whose best screened candidate passes the global gate (ratio < R_GATE, calibrated on the
null pairs in screen.json), the top candidates are aligned with align.align() (section chain), the best
one kept, and labels written to ~/personal/MusicVis-data/labels/midi/<corpus>/<id>.json (format in the
README next to them). Hooktheory clips (WORK/hooktheory.json) are merged into the same file.

Run (one process, under the machine-wide limiter):
  slot.sh env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib .testdata/gt/venv/bin/python scripts/ml/groundtruth/build_labels.py [--only substr] [--force]
"""
import argparse, json, os, re, sys, time
import numpy as np
import pretty_midi
sys.path.insert(0, os.path.dirname(__file__))
import gt_common as g
import align
import hooktheory as ht
from notes2chords import chords_from_notes

R_GATE = None          # set from nulls in calibrate()
P_GATE = 0.80          # per-placement path cost / tau; placements above it are dropped
H_GATE = None          # Hooktheory clip ratio gate, from its nulls
MIN_COVER = 0.35       # fraction of the recording that must be covered by confident placements
TOPC = 2

DRUMS = {'kick': {35, 36}, 'snare': {37, 38, 39, 40}, 'hihat': {42, 44, 46}, 'tom': {41, 43, 45, 47, 48, 50},
         'crash': {49, 52, 55, 57}, 'ride': {51, 53, 59}, 'perc': set(range(60, 82)) | {54, 56, 58}}


def calibrate(screen, htres):
    global R_GATE, H_GATE
    nul = np.array([r['ratio'] for v in screen.values() for r in v if r['kind'] == 'null' and 'ratio' in r])
    # strictest of: 0.5th percentile of nulls and a fixed ceiling
    R_GATE = float(min(np.percentile(nul, 0.5) - 0.02, 0.86)) if len(nul) > 20 else 0.8
    hn = np.array([r['ratio'] for v in htres.values() for r in v if r['null']])
    H_GATE = float(min(np.percentile(hn, 1) - 0.02, 0.8)) if len(hn) > 20 else 0.7
    return dict(R_GATE=R_GATE, null_n=int(len(nul)), null_min=float(nul.min()) if len(nul) else None,
                null_p1=float(np.percentile(nul, 1)) if len(nul) else None, H_GATE=H_GATE, ht_null_n=int(len(hn)))


def family(inst):
    if inst.is_drum:
        return 'drums'
    return pretty_midi.program_to_instrument_class(inst.program)


def mapped_times(t, maps):
    """For MIDI times t -> list of (index into t, audio time, placement idx) (repeats give several)."""
    t = np.asarray(t, float)
    res_i, res_a, res_p = [], [], []
    for pi_, (m0, m1, u, av, p) in enumerate(maps):
        sel = np.nonzero((t >= m0) & (t < m1))[0]
        if len(sel):
            res_i.append(sel); res_a.append(np.interp(t[sel], u, av)); res_p.append(np.full(len(sel), pi_))
    if not res_i:
        return np.zeros(0, int), np.zeros(0), np.zeros(0, int)
    return np.concatenate(res_i), np.concatenate(res_a), np.concatenate(res_p)


def lag_at(lagc, t):
    if lagc is None:
        return 0.0 * np.asarray(t)
    c, l = lagc
    return np.interp(t, c, l)


def grid_quality(pm):
    """Fraction of note onsets within 12% of a 16th-note step of the MIDI's own beat grid."""
    try:
        beats = pm.get_beats()
    except Exception:
        return 0.0
    if len(beats) < 8:
        return 0.0
    ons = np.array([n.start for i in pm.instruments for n in i.notes])
    ons = ons[(ons >= beats[0]) & (ons < beats[-1])]
    if len(ons) < 20:
        return 0.0
    k = np.clip(np.searchsorted(beats, ons) - 1, 0, len(beats) - 2)
    frac = (ons - beats[k]) / (beats[k + 1] - beats[k]) * 4
    return float(np.mean(np.abs(frac - np.round(frac)) < 0.12))


def export(corpus, tid, path, Ra, cand, pm, pl, st, lagc_src=None):
    shift = st['shift']
    keep = [p for p in pl if p['cost'] / st['tau'] < P_GATE]
    maps = align.time_mapper(keep)
    dur = len(Ra) / align.FPS
    # notes
    insts = []
    all_on = []
    for inst in pm.instruments:
        if not inst.notes:
            continue
        arr = np.array([[n.start, n.end, n.pitch, n.velocity] for n in inst.notes], float)
        i_on, a_on, p_idx = mapped_times(arr[:, 0], maps)
        if not len(i_on):
            continue
        # offset through the same placement's map (clamped to the placement's end)
        a_off = np.array([np.interp(min(arr[i, 1], maps[p][1]), maps[p][2], maps[p][3]) for i, p in zip(i_on, p_idx)])
        a_off = np.maximum(a_off, a_on + 0.02)
        pitch = arr[i_on, 2] + (0 if inst.is_drum else shift)
        insts.append((inst, np.stack([a_on, a_off, pitch, arr[i_on, 3]], 1)))
        all_on.append(a_on)
    all_on = np.concatenate(all_on) if all_on else np.zeros(0)
    y = g.decode(path, align.SR)
    lagc = align.onset_refine(all_on, y)
    del y
    out_insts = []
    for inst, a in insts:
        a[:, 0] += lag_at(lagc, a[:, 0]); a[:, 1] += lag_at(lagc, a[:, 1])
        a = a[np.argsort(a[:, 0])]
        out_insts.append(dict(name=inst.name.strip(), program=int(inst.program), is_drum=bool(inst.is_drum),
                              family=family(inst),
                              instrument=('Drums' if inst.is_drum else pretty_midi.program_to_instrument_name(inst.program)),
                              notes=np.round(a, 4).tolist()))
    # beats / downbeats
    gq = grid_quality(pm)
    beats_out, down_out = [], []
    try:
        mb = pm.get_beats(); md = pm.get_downbeats()
    except Exception:
        mb, md = np.zeros(0), np.zeros(0)
    if len(mb):
        pos = np.zeros(len(mb), int)
        di = np.searchsorted(md, mb, side='right') - 1
        for i in range(len(mb)):
            pos[i] = 0 if di[i] < 0 else int(np.sum((mb >= md[di[i]]) & (mb < mb[i])))
        i_b, a_b, _ = mapped_times(mb, maps)
        a_b = a_b + lag_at(lagc, a_b)
        o = np.argsort(a_b)
        beats_out = [[round(float(a_b[j]), 4), int(pos[i_b[j]])] for j in o]
        down_out = [b[0] for b in beats_out if b[1] == 0]
    # drums by class
    drums = {k: [] for k in DRUMS}
    for d in out_insts:
        if d['is_drum']:
            for on, off, p, v in d['notes']:
                for k, s in DRUMS.items():
                    if int(p) in s:
                        drums[k].append(round(on, 4))
    drums = {k: sorted(v) for k, v in drums.items()}
    # regions: merged audio spans of kept placements (after lag)
    regs = sorted([[p['audio'][0] / align.FPS, p['audio'][1] / align.FPS] for p in keep])
    merged = []
    for a, b in regs:
        a += float(lag_at(lagc, a)); b += float(lag_at(lagc, b))
        if merged and a - merged[-1][1] < 0.5:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    # chords from the pitched notes on the aligned beat grid, per region
    pitched = np.concatenate([np.array(d['notes']) for d in out_insts if not d['is_drum'] and d['notes']] or [np.zeros((0, 4))])
    chords = []
    bt = np.array([b[0] for b in beats_out]) if beats_out else np.zeros(0)
    for a, b in merged:
        bb = bt[(bt >= a - 1e-3) & (bt <= b + 1e-3)]
        chords += chords_from_notes(pitched, bb)
    cover = sum(b - a for a, b in merged) / dur
    conf = dict(global_ratio=cand['ratio'], transpose=shift,
                placements=len(keep), placements_dropped=len(pl) - len(keep),
                mean_cost_ratio=float(np.average([p['cost'] / st['tau'] for p in keep],
                                                 weights=[p['audio'][1] - p['audio'][0] + 1 for p in keep])) if keep else None,
                coverage=round(cover, 3), grid_quality=round(gq, 3),
                onset_lag_ms=None if lagc is None else [round(float(np.median(lagc[1])) * 1000, 1),
                                                         round(float(np.max(np.abs(lagc[1]))) * 1000, 1)])
    return dict(
        source=dict(type='freemidi' if cand['md5'].startswith('fm') else 'lakh', md5=cand['md5'], why=cand.get('why'),
                    names=cand.get('names')),
        confidence=conf,
        placements=[dict(section=p['k'], midi=[round(p['midi'][0] / align.FPS, 3), round(p['midi'][1] / align.FPS, 3)],
                         audio=[round(p['audio'][0] / align.FPS, 3), round(p['audio'][1] / align.FPS, 3)],
                         cost_ratio=round(p['cost'] / st['tau'], 4)) for p in keep],
        regions=[[round(a, 3), round(b, 3)] for a, b in merged],
        instruments=out_insts, drums=drums,
        beats=beats_out if gq >= 0.6 else [], downbeats=down_out if gq >= 0.6 else [],
        beats_ok=bool(gq >= 0.6 and len(beats_out) > 8),
        chords=chords), cover


def ht_segments(tid, htres, H):
    segs = []
    for r in htres.get(tid, []):
        if r['null']:
            continue
        direct = r['same_video'] and r['dtw_vs_identity'] < 0.3
        if not (direct or r['ratio'] < H_GATE):
            continue
        clip = H[r['clip']]
        mel, har, beats = ht.clip_events(clip)
        u, av = np.array(r['map_t']), np.array(r['map_a'])
        f = (lambda t: np.asarray(t, float)) if direct else (lambda t: np.interp(t, u, av))
        lo, hi = u[0], u[-1]
        ins = lambda t: direct or (lo - 0.05 <= t <= hi + 0.05)
        segs.append(dict(
            clip=r['clip'], song=r['song'], youtube=r['yt'], mode='direct' if direct else 'dtw', ratio=r['ratio'],
            audio=[round(float(f(lo)), 3), round(float(f(hi)), 3)],
            melody=[[round(float(f(s)), 4), round(float(f(e)), 4), int(p)] for s, e, p in mel if ins(s)],
            chords=[[round(float(f(s)), 4), round(float(f(e)), 4), lab] for s, e, lab, _ in har if ins(s)],
            beats=[[round(float(f(t)), 4), bool(d)] for t, d in beats if ins(t)]))
    # drop overlapping duplicates (two clips of the same section): keep the lower ratio
    segs.sort(key=lambda s: (s['mode'] != 'direct', s['ratio']))
    kept = []
    for s in segs:
        if all(s['audio'][1] <= k['audio'][0] or s['audio'][0] >= k['audio'][1] for k in kept):
            kept.append(s)
    return sorted(kept, key=lambda s: s['audio'][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    cands = json.load(open(os.path.join(g.WORK, 'candidates.json')))
    screen = json.load(open(os.path.join(g.WORK, 'screen.json')))
    htp = os.path.join(g.WORK, 'hooktheory.json')
    htres = json.load(open(htp)) if os.path.exists(htp) else {}
    H = ht.load() if htres else {}
    cal = calibrate(screen, htres)
    print('calibration', cal, flush=True)
    sump = os.path.join(g.WORK, 'summary.json')
    summary = json.load(open(sump)) if os.path.exists(sump) else {}
    summary['_calibration'] = cal
    ids = {tid: (c, p) for c, tid, p in g.all_tracks()}
    for tid, (corpus, path) in ids.items():
        if a.only and a.only not in tid:
            continue
        outp = os.path.join(g.GT, corpus, tid + '.json')
        if os.path.exists(outp) and not a.force:
            continue
        t0 = time.time()
        rs = sorted([r for r in screen.get(tid, []) if r['kind'] != 'null' and 'ratio' in r], key=lambda r: r['ratio'])
        rs = [r for r in rs if r['ratio'] < R_GATE][:TOPC]
        best = None
        info = []
        if rs:
            Ra = align.audio_raw(corpus, tid, path)
            for r in rs:
                c = next(c for c in cands[tid]['cands'] if c['md5'] == r['md5'])
                c = dict(c, ratio=r['ratio'])
                try:
                    pm, pl, st = align.align(Ra, os.path.join(g.WORK, 'midi', r['md5'] + '.mid'))
                    lab, cover = export(corpus, tid, path, Ra, c, pm, pl, st)
                except Exception as ex:
                    info.append(dict(md5=r['md5'], error=str(ex)[:100])); continue
                q = cover * (1.0 - (lab['confidence']['mean_cost_ratio'] or 1.0))
                info.append(dict(md5=r['md5'], ratio=r['ratio'], cover=round(cover, 3), q=round(q, 4)))
                if cover >= MIN_COVER and (best is None or q > best[0]):
                    best = (q, lab)
        segs = ht_segments(tid, htres, H) if htres else []
        summary[tid] = dict(corpus=corpus, midi=info, chosen=best[1]['source']['md5'] if best else None,
                            hooktheory=len(segs))
        if best or segs:
            lab = best[1] if best else dict(source=None, confidence=None, placements=[], regions=[], instruments=[],
                                             drums={}, beats=[], downbeats=[], beats_ok=False, chords=[])
            doc = dict(id=tid, slug=g.slug_of(re.sub(r'^\d+\s*-\s*', '', tid.split('__')[-1])) if corpus == 'own' else g.slug_of(tid),
                       corpus=corpus, audio=path, duration=round(len(align.audio_raw(corpus, tid, path)) / align.FPS, 3),
                       format=1, **lab, hooktheory=segs)
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            json.dump(doc, open(outp, 'w'), separators=(',', ':'))
        json.dump(summary, open(sump, 'w'), indent=1)
        print(f'{tid[:60]:60s} {time.time() - t0:5.1f}s midi={info} ht={len(segs)}', flush=True)


if __name__ == '__main__':
    main()
