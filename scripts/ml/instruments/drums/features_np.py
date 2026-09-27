"""Numpy mirror of scripts/ml/features.ts (StreamFeatures): causal streaming features, one frame
per 512-sample hop at 44.1 kHz (86.13 fps). Frame = [80 log-spaced band log-magnitudes,
80 positive band differences (flux)] from a 2048-sample Hann window ending at the block's last
sample (mono mid = 0.5*(l+r)). Bit-for-bit filterbank match with the TS filterbank(); the FFT
power spectrum matches RealFFT.power() (a plain |rfft|^2, no extra normalisation).

If the browser captures at 48 kHz instead of 44.1 kHz: resample the incoming audio to 44.1 kHz
first (linear or polyphase resampler on the 512-sample blocks, buffered), so ML_SR/ML_HOP/ML_WIN
and this filterbank stay fixed; do NOT rescale the FFT size, since the filterbank's Hz edges (and
therefore the learned weights) are tied to a 44.1 kHz bin spacing.
"""
import numpy as np

ML_SR = 44100
ML_HOP = 512
ML_WIN = 2048
ML_BANDS = 80
ML_FEAT = 2 * ML_BANDS
ML_FPS = ML_SR / ML_HOP


def _filterbank(n=ML_WIN, sr=ML_SR, count=ML_BANDS, fmin=30.0, fmax=16000.0):
    nb = n // 2 + 1
    bin_hz = sr / n
    edges = [fmin * (fmax / fmin) ** (i / (count + 1)) for i in range(count + 2)]
    fb = []
    for b in range(count):
        f0, f1, f2 = edges[b], edges[b + 1], edges[b + 2]
        lo = int(np.floor(f0 / bin_hz))
        hi = int(np.ceil(f2 / bin_hz))
        c = f1 / bin_hz
        lo = max(0, min(lo, int(np.floor(c)) - 1))
        hi = min(nb - 1, max(hi, int(np.ceil(c)) + 1))
        w = np.zeros(hi - lo + 1, dtype=np.float64)
        for k in range(lo, hi + 1):
            f = k * bin_hz
            v = (f - f0) / (f1 - f0) if f < f1 else (f2 - f) / (f2 - f1)
            v = max(v, 0.0)
            if f2 - f0 < 2 * bin_hz:
                v = max(0.0, 1 - abs(k - c))
            w[k - lo] = v
        s = w.sum()
        if s > 0:
            w = w / s
        fb.append((lo, w))
    return fb


class StreamFeatures:
    """Causal streaming feature extractor matching TS StreamFeatures.push() frame-for-frame."""

    def __init__(self):
        self.fb = _filterbank()
        self.win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(ML_WIN) / ML_WIN)
        self.ring = np.zeros(ML_WIN, dtype=np.float64)
        self.prev = np.zeros(ML_BANDS, dtype=np.float64)
        self.frames = 0

    def push(self, l, r):
        """l, r: length-512 mono blocks (arrays). Returns a length-ML_FEAT frame."""
        mid = 0.5 * (np.asarray(l, dtype=np.float64) + np.asarray(r, dtype=np.float64))
        self.ring = np.concatenate([self.ring[ML_HOP:], mid])
        buf = self.ring * self.win
        spec = np.fft.rfft(buf, n=ML_WIN)
        pw = (spec.real ** 2 + spec.imag ** 2)
        out = np.zeros(ML_FEAT, dtype=np.float32)
        for b, (lo, w) in enumerate(self.fb):
            s = float(np.dot(w, pw[lo:lo + len(w)]))
            v = np.log1p(1000.0 * np.sqrt(max(s, 0.0)) / (ML_WIN / 4))
            out[b] = v
            out[ML_BANDS + b] = max(0.0, v - self.prev[b]) if self.frames > 0 else 0.0
            self.prev[b] = v
        self.frames += 1
        return out


def _dense_filterbank():
    fb = _filterbank()
    nb = ML_WIN // 2 + 1
    FB = np.zeros((ML_BANDS, nb), dtype=np.float64)
    for b, (lo, w) in enumerate(fb):
        FB[b, lo:lo + len(w)] = w
    return FB


_FB_DENSE = None


def features_of(mono, hop=ML_HOP):
    """mono: 1-D float array at 44.1 kHz. Returns [T, ML_FEAT] float32, T = len(mono)//hop.
    Vectorised equivalent of StreamFeatures.push() called frame by frame (bit-identical numbers,
    much faster: one batched STFT over all frames, then a single filterbank matmul)."""
    global _FB_DENSE
    if _FB_DENSE is None:
        _FB_DENSE = _dense_filterbank()
    x = np.asarray(mono, dtype=np.float64)
    T = len(x) // hop
    if T <= 0:
        return np.zeros((0, ML_FEAT), dtype=np.float32)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(ML_WIN) / ML_WIN)
    # frame k's window ends at sample (k+1)*hop, i.e. covers [(k+1)*hop - ML_WIN, (k+1)*hop)
    pad = np.zeros(ML_WIN - hop, dtype=np.float64)
    xp = np.concatenate([pad, x, np.zeros(max(0, (T * hop) - len(x)), dtype=np.float64)])
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(xp, ML_WIN)[::hop][:T]
    seg = windows * win
    spec = np.fft.rfft(seg, n=ML_WIN, axis=1)
    pw = spec.real ** 2 + spec.imag ** 2  # [T, nb]
    s = pw @ _FB_DENSE.T  # [T, ML_BANDS]
    logmag = np.log1p(1000.0 * np.sqrt(np.maximum(s, 0.0)) / (ML_WIN / 4))
    flux = np.zeros_like(logmag)
    flux[1:] = np.maximum(0.0, logmag[1:] - logmag[:-1])
    out = np.concatenate([logmag, flux], axis=1).astype(np.float32)
    return out


def frame_time(k, hop=ML_HOP, sr=ML_SR):
    return (k + 1) * hop / sr
