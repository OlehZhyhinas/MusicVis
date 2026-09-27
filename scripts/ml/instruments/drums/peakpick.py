"""Causal peak picking + onset scoring, shared by the student eval and the DSP baseline rescoring.

peak_pick() fires at frame k when
  act[k] >= threshold,
  act[k] >= mean(act[k-avg_win .. k-1]) + delta       (adaptive part; delta=0 disables),
  act[k] >  max(act[k-past .. k-1])                   (strict causal local max over the recent past),
  act[k] >= max(act[k+1 .. k+lookahead])              (<= 1-2 frames of look-ahead),
  k - last_onset > refractory.
The first version only had the look-ahead test, so on the decaying side of a wide peak every frame
above the threshold was a "local max over [k, k+1]" and fired again as soon as the refractory period
ran out: 2-3x as many onsets as references, and the tuner pushed the threshold to the top of its
grid to compensate.

A streaming implementation reports frame k's onset at frame k+lookahead (decision latency =
(k+1+lookahead)/fps - t_onset); the event's timestamp is (k+1)/fps.
"""
import bisect
import numpy as np


def peak_pick(act, threshold, refractory, lookahead=1, past=3, delta=0.0, avg_win=8):
    a = np.asarray(act, dtype=np.float64)
    T = len(a)
    if T == 0:
        return []
    cond = a >= threshold
    for j in range(1, past + 1):
        prev = np.concatenate([np.full(j, -np.inf), a[:-j]]) if j < T else np.full(T, -np.inf)
        cond &= a > prev
    for j in range(1, lookahead + 1):
        nxt = np.concatenate([a[j:], np.full(j, -np.inf)]) if j < T else np.full(T, -np.inf)
        cond &= a >= nxt
    if delta > 0:
        c = np.concatenate([[0.0], np.cumsum(a)])
        idx = np.arange(T)
        lo = np.maximum(0, idx - avg_win)
        n = np.maximum(1, idx - lo)
        mean = (c[idx] - c[lo]) / n
        cond &= a >= mean + delta
    events = []
    last = -10 ** 9
    for k in np.flatnonzero(cond):
        if k - last > refractory:
            events.append(int(k))
            last = k
    return events


def frames_to_times(frames, frame_rate):
    return [(k + 1) / frame_rate for k in frames]  # frame k's window ends at (k+1)*hop


def match(pred_times, ref_times, tol=(-0.010, 0.050)):
    """Greedy one-to-one matching: pred t hits ref r if r + tol[0] <= t <= r + tol[1].
    Returns (tp, delays) with delays = t - r of the matches."""
    ref = sorted(ref_times)
    used = [False] * len(ref)
    tp = 0
    delays = []
    for t in sorted(pred_times):
        i = bisect.bisect_left(ref, t - tol[1])
        best, best_d = -1, None
        while i < len(ref) and ref[i] <= t - tol[0]:
            if not used[i]:
                d = abs(t - ref[i])
                if best_d is None or d < best_d:
                    best, best_d = i, d
            i += 1
        if best >= 0:
            used[best] = True
            tp += 1
            delays.append(t - ref[best])
    return tp, delays


def prf(tp, n_pred, n_ref):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_ref if n_ref else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score_prf(pred_times, ref_times, tol=(-0.010, 0.050)):
    """Kept for compatibility: (precision, recall, f1, tp, delays). No 'perfect score' for an empty
    song; aggregate tp / n_pred / n_ref over songs (micro F1) instead of averaging per-song F1."""
    tp, delays = match(pred_times, ref_times, tol)
    p, r, f1 = prf(tp, len(pred_times), len(ref_times))
    return p, r, f1, tp, delays


def tune(acts, refs, frame_rate, thresholds, refractories, lookaheads=(1,), pasts=(3,), deltas=(0.0,), tol=(-0.010, 0.050)):
    """acts: list of 1-D activations, refs: list of reference time lists. Maximises MICRO F1 summed
    over all songs (songs without references still count their false positives)."""
    best = None
    n_ref = sum(len(r) for r in refs)
    for la in lookaheads:
        for pa in pasts:
            for de in deltas:
                for th in thresholds:
                    for rf in refractories:
                        tp = npred = 0
                        for a, r in zip(acts, refs):
                            pt = frames_to_times(peak_pick(a, th, rf, la, pa, de), frame_rate)
                            npred += len(pt)
                            tp += match(pt, r, tol)[0]
                        f1 = prf(tp, npred, n_ref)[2]
                        if best is None or f1 > best['train_f1']:
                            best = {'threshold': th, 'refractory': rf, 'lookahead': la, 'past': pa, 'delta': de, 'train_f1': f1, 'train_n_ref': n_ref}
    return best
