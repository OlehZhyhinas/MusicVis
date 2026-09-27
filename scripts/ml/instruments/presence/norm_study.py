"""stems.* normalisation study on norm_dump.ts recordings (per analyzer frame: DbEnvelope input power p,
running top, current 0..1 output). Emulates candidate normalisers on the same inputs and reports, per
stem, the stems-eval r (normalised value EMA-smoothed vs the htdemucs dB reference) and the output
distribution (mean, p90, fraction of frames > 0.8) so preset reaction scales can be compared.
  N0  current DbEnvelope (40 ms release, peak memory -1.5 dB/s, range [max(top-60, hi-36), max(peak-2, top-30)])
  P   percentile range over a causal window: lo = p5, hi = p98 of the stem's smoothed dB over the last W s
      (history sampled every 8 frames), out = clamp01((db - lo) / (hi - lo)), hi - lo >= 12 dB
  .testdata/instr/venvs/torch/bin/python norm_study.py dsp|neural [W=20]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import stemscore as SS
import student as ST

FR = 44100 / 512
S4 = ['drums', 'bass', 'vocals', 'other']


def smooth_db(p):
    """DbEnvelope's AttackRelease(0, 0.04 s) on dB: instant rise, exponential fall."""
    d = 10 * np.log10(np.maximum(p, 1e-30))
    k = 1 - np.exp(-1 / (0.04 * FR))
    o = np.empty_like(d)
    a = -300.0
    for i, v in enumerate(d):
        a = v if v > a else a + (v - a) * k
        o[i] = a
    return o


def n0(db, top):
    out = np.zeros(len(db)); peak = -1e9; dec = 1.5 / FR
    for i in range(len(db)):
        peak = max(db[i], peak - dec)
        hi = max(peak - 2, top[i] - 30)
        lo = max(top[i] - 60, hi - 36)
        if hi < lo + 10:
            hi = lo + 10
        out[i] = min(1, max(0, (db[i] - lo) / (hi - lo)))
    return out


def ntop(db, top, lo_rel=-45, hi_rel=-5):
    """Fixed dB window against the mix's running top (linear in dB, no per-stem peak chasing)."""
    return np.clip((db - (top + lo_rel)) / (hi_rel - lo_rel), 0, 1)


def nslow(db, top, decay=0.2, span=30.0):
    """Per-stem peak memory falling `decay` dB/s (DbEnvelope: 1.5), range [peak - span, peak], floor top - 60."""
    out = np.zeros(len(db)); peak = -1e9; dec = decay / FR
    for i in range(len(db)):
        peak = max(db[i], peak - dec)
        hi = max(peak, top[i] - 30)
        lo = max(top[i] - 60, hi - span)
        out[i] = min(1, max(0, (db[i] - lo) / (hi - lo)))
    return out


def npct(db, top, W=20.0, plo=5, phi=98, every=8, minr=12.0):
    out = np.zeros(len(db))
    hist = []
    L = int(W * FR / every)
    lo = hi = None
    for i in range(len(db)):
        if i % every == 0 and top[i] > -149:
            hist.append(max(db[i], top[i] - 70))
            if len(hist) > L:
                hist.pop(0)
            h = np.asarray(hist)
            lo, hi = np.percentile(h, plo), np.percentile(h, phi)
            if hi - lo < minr:
                lo = hi - minr
        if lo is None:
            continue
        out[i] = min(1, max(0, (db[i] - lo) / (hi - lo)))
    return out


if __name__ == '__main__':
    mode = sys.argv[1]
    W = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    d = os.path.join(ST.PRES, 'norm', mode)
    res = {}
    for f in sorted(os.listdir(d)):
        slug = f[:-4]
        X = np.fromfile(os.path.join(d, f), '<f4').reshape(-1, 12)
        for i, s in enumerate(S4):
            rf = SS.beat_ref4(slug, s)
            n = min(len(rf), len(X))
            ref = SS.db_ref(rf[:n])
            p, top, cur = X[:n, i].astype(np.float64), X[:n, 4 + i].astype(np.float64), X[:n, 8 + i]
            db = smooth_db(p)
            live = top > -149
            outs = {'cur': cur, 'N0': np.where(live, n0(db, top), 0), f'P{W:g}': np.where(live, npct(db, top, W), 0),
                    'T45-5': np.where(live, ntop(db, top), 0), 'T40-10': np.where(live, ntop(db, top, -40, -10), 0),
                    'S.2/30': np.where(live, nslow(db, top), 0), 'S.5/36': np.where(live, nslow(db, top, 0.5, 36), 0),
                    'rawdB': np.where(live, db, np.nan)}
            for k, v in outs.items():
                if k == 'rawdB':
                    vv = np.nan_to_num(v, nan=np.nanmin(v))
                    r = SS.corr_lag(ref, SS.db_cand(vv))[0]
                    res.setdefault((s, k), []).append((r, np.nan, np.nan, np.nan))
                    continue
                r = SS.corr_lag(ref, SS.ema(v))[0]
                res.setdefault((s, k), []).append((r, v.mean(), np.percentile(v, 90), (v > 0.8).mean()))
    print(f'{mode}: per stem: r (normalised, stems-eval) | mean | p90 | frac>0.8')
    for (s, k), v in res.items():
        a = np.array(v, float)
        print(f'{s:7s} {k:6s} r={a[:, 0].mean():.3f}  mean={np.nanmean(a[:, 1]):.2f} p90={np.nanmean(a[:, 2]):.2f} >0.8={np.nanmean(a[:, 3]):.2f}')
