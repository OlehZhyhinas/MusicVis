"""Cheap causal log-mel front end for the student (plain-TS portable) and its feature cache.

Mono 22050 Hz (the browser decimates 44.1/48 kHz by 2 or resamples), hop 256 (= 512 at 44.1 kHz,
~86.1 fps, frame-aligned with the demucs6 envelopes), window N_FFT=1024 samples ENDING at the end
of the current hop (causal: frame k sees samples [(k+1)*256-1024, (k+1)*256)), periodic Hann, power
spectrum, N_MEL=64 triangular mel bands 30 Hz..11 kHz (HTK mel, area-normalised), log10(p + 1e-7).

Cache: .testdata/instr/presence/feat/<corpus>/<id>.npy (float16 [T, 64]).
  python features.py <corpus> <listfile>
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import common as C

SR, HOP, N_FFT, N_MEL, FMIN, FMAX = 22050, 256, 1024, 64, 30.0, 11000.0
FPS = SR / HOP
FEAT = os.path.join(C.WORK, 'presence', 'feat')


def mel_fb():
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(FMIN), hz2mel(FMAX), N_MEL + 2))
    freqs = np.arange(N_FFT // 2 + 1) * SR / N_FFT
    fb = np.zeros((N_MEL, len(freqs)), np.float32)
    for m in range(N_MEL):
        lo, c, hi = pts[m], pts[m + 1], pts[m + 2]
        up = (freqs - lo) / (c - lo)
        dn = (hi - freqs) / (hi - c)
        fb[m] = np.maximum(0, np.minimum(up, dn)) * (2 / (hi - lo))
    return fb


FB = mel_fb()
WIN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)


def logmel(x):
    """x: mono float32 at 22050 Hz -> [T, N_MEL] float32 with T = len(x) // HOP (causal frames)."""
    T = len(x) // HOP
    xp = np.concatenate([np.zeros(N_FFT - HOP, np.float32), x[:T * HOP].astype(np.float32)])
    out = np.empty((T, N_MEL), np.float32)
    for b in range(0, T, 4096):
        e = min(T, b + 4096)
        idx = (np.arange(b, e) * HOP)[:, None] + np.arange(N_FFT)[None, :]
        fr = xp[idx] * WIN
        p = np.abs(np.fft.rfft(fr, axis=1)) ** 2
        out[b:e] = np.log10(p @ FB.T + 1e-7)
    return out


def feat(corpus, tid, path):
    p = os.path.join(FEAT, corpus, tid + '.npy')
    if os.path.exists(p):
        return np.load(p).astype(np.float32)
    cached = os.path.exists(os.path.join(C.PCM22, corpus, tid + '.f32'))
    # decode to a pipe unless already cached: no new pcm22 files (disk)
    f = logmel(C.pcm22(corpus, tid, path) if cached else C.decode(path, SR, 1))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, f.astype(np.float16))
    return f


if __name__ == '__main__':
    corpus = sys.argv[1]
    for line in open(sys.argv[2]).read().split('\n'):
        if line:
            tid, path = line.split('=', 1)
            try:
                feat(corpus, tid, path)
            except Exception as e:
                print('fail', tid, e)
