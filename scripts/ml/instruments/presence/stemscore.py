"""Python port of the beat agent's scripts/ml/stems-eval.ts metric (so numbers slot into its table):
reference RMS envelope -> dB, floored 60 dB under the song's 99th percentile, EMA-smoothed with
w = round(0.25 * fps) (a += (x - a) / w); candidate smoothed the same way; Pearson r at the best
lag in +-300 ms (lag > 0 = candidate late).
Reference envelopes (causal 2048-sample window ending at each 512-hop end, 44.1 kHz):
  4-stem htdemucs: the beat agent's .testdata/ml/stems/demucs/<slug>.<stem>.f32 (13 test songs)
  6-stem htdemucs_6s: computed here from .testdata/instr/stems6 (test) or the d6 / labels caches.
"""
import os
import numpy as np

FPS = 44100 / 512
W = round(0.25 * FPS)
L = round(0.3 * FPS)
BEAT_DEMUCS = '/Users/oleh/personal/MusicVis/.claude/worktrees/agent-adccb4c576b7c1097/.testdata/ml/stems/demucs'


def ema(x, w=W):
    o = np.empty(len(x), np.float64)
    a = float(x[0]) if len(x) else 0.0
    k = 1.0 / w
    for i, v in enumerate(x):
        a += (v - a) * k
        o[i] = a
    return o


def db_ref(rms):
    d = 20 * np.log10(np.asarray(rms, np.float64) + 1e-9)
    top = np.sort(d)[int(0.99 * (len(d) - 1))]
    return ema(np.maximum(d, top - 60))


def db_cand(dbv):
    """A candidate already in dB: same floor + smoothing as the reference."""
    d = np.asarray(dbv, np.float64)
    top = np.sort(d)[int(0.99 * (len(d) - 1))]
    return ema(np.maximum(d, top - 60))


def corr_lag(ref, y):
    n = min(len(ref), len(y))
    ref, y = ref[:n], y[:n]
    best, bl = -2.0, 0
    for lag in range(-L, L + 1):
        if lag >= 0:
            a, b = ref[:n - lag], y[lag:]
        else:
            a, b = ref[-lag:], y[:n + lag]
        sa, sb = a.std(), b.std()
        r = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)) if sa > 0 and sb > 0 else 0.0
        if r > best:
            best, bl = r, lag
    return best, bl / FPS


def causal_rms(x, hop, win):
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, np.float64) ** 2)])
    T = len(x) // hop
    ends = (np.arange(T) + 1) * hop
    return np.sqrt((c[ends] - c[np.maximum(0, ends - win)]) / win).astype(np.float32)


def beat_ref4(slug, stem):
    p = os.path.join(BEAT_DEMUCS, f'{slug}.{stem}.f32')
    return np.fromfile(p, '<f4') if os.path.exists(p) else None


def onset_latency(ref_rms, cand_db, rise_db=9.0):
    """Onset-to-response latency (s): at each reference onset (the causal ref dB rises >= rise_db within
    3 hops from a local floor), the time until the candidate covers half of its own rise (candidate max
    over the next 150 ms minus its value 2 hops before the onset). Returns (median latency, n onsets)."""
    r = 20 * np.log10(np.asarray(ref_rms, np.float64) + 1e-9)
    top = np.sort(r)[int(0.99 * (len(r) - 1))]
    r = np.maximum(r, top - 50)
    c = np.asarray(cand_db, np.float64)
    n = min(len(r), len(c))
    lat = []
    k = 3
    H = round(0.15 * FPS)
    while k < n - H:
        if r[k] - r[k - 3] >= rise_db and r[k] > top - 30:
            k0 = k - 2 + int(np.argmax(np.diff(r[k - 3:k + 1])))  # hop that lands the steepest rise
            base = c[max(0, k0 - 2)]
            seg = c[k0 - 2:k0 + H]
            peak = seg.max()
            if peak - base > 1.0:
                j = int(np.argmax(seg >= base + 0.5 * (peak - base)))
                lat.append((j - 2) / FPS)
            k += H
        else:
            k += 1
    return (float(np.median(lat)) if lat else float('nan')), len(lat)
