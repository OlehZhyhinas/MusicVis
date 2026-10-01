"""Structure-aware audio-to-MIDI alignment (Raffel-style CQT DTW + JumpDTW-like section chaining).

Per (track, candidate MIDI):
  1. Synthesise the MIDI with fluidsynth (MuseScore_General GM soundfont) in memory, never to disk or speakers.
  2. Features for both: log-magnitude CQT, MIDI 24..95 (72 bins), hop 512 @ 22050 Hz (23 ms), audio tuning
     estimated and compensated, each frame L2-normalised; distance = cosine distance (Raffel & Ellis 2016).
  3. Transposition check: coarse global DTW for CQT shifts -3..+3 semitones (karaoke MIDIs are often in
     another key); the notes are output transposed to the recording's key.
  4. Split the MIDI into sections of SEG_BARS bars on its own downbeat grid. Each section is located in the
     audio with subsequence DTW (free start/end in the audio), giving for every audio end frame the section's
     mean path cost and start frame.
  5. Chain DP over audio time picks an ordered set of non-overlapping section placements maximising
     sum(len * (tau - cost)), with a bonus for playing section k right after k-1 and a penalty for any jump.
     That handles tempo drift (inside each DTW), omitted sections (skips), repeats (a section used twice),
     and intros/outros/skits (unlabelled gaps). tau is the median distance of random frame pairs.
  6. Each placement gets a full DTW path; MIDI time -> audio time is piecewise linear along it.
  7. Fine refinement: in 6 s windows the aligned note-onset train is cross-correlated with the audio onset
     envelope (hop 256) for lags within +-70 ms, lags are median-smoothed and applied.
Confidence: per placement, "contrast" = (tau - path cost) / std of random-pair distances; song level =
duration-weighted contrast over covered audio, plus coverage (fraction of audio inside placements). The
acceptance threshold is calibrated on null alignments (each track against unrelated MIDIs), see calibrate().

Library use: align(audio_y, midi_path) -> dict (see build_labels()).
"""
import os, sys, warnings
import numpy as np
import numba
import librosa
import pretty_midi
import scipy.ndimage as ndi
sys.path.insert(0, os.path.dirname(__file__))
import gt_common as g

warnings.filterwarnings('ignore')
SR = 22050
HOP = 512
FMIN_MIDI = 24
NBINS = 72
SEG_BARS = 4
FPS = SR / HOP
_SYNTH = None


def synth(pm):
    """In-memory fluidsynth render, mono float32 @ SR. No audio device is opened."""
    global _SYNTH
    import fluidsynth
    if _SYNTH is None:
        _SYNTH = fluidsynth.Synth(samplerate=SR)
        _SYNTH.sfid = _SYNTH.sfload(g.SOUNDFONT)
    y = pm.fluidsynth(fs=SR, synthesizer=_SYNTH, sfid=_SYNTH.sfid)
    return y.astype(np.float32)


def raw_cqt(y, tuning=None):
    """dB CQT relative to the track maximum, clipped to [-80, 0] and offset to [0, 80]; float16 [frames, 72]."""
    C = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.midi_to_hz(FMIN_MIDI), n_bins=NBINS,
                           bins_per_octave=12, tuning=tuning))
    C = librosa.amplitude_to_db(C, ref=np.max(C) + 1e-9, top_db=80.0) + 80.0
    return C.T.astype(np.float16)


FEAT_MODE = 'cqt_c'


def feat(raw, mode=None):
    """Frame features, L2-normalised, from raw_cqt output."""
    mode = mode or FEAT_MODE
    X = raw.astype(np.float32)
    if mode == 'cqt':                       # Raffel: log-magnitude CQT
        pass
    elif mode == 'cqt_c':                   # dB floor at -50, then per-frame mean removal (Pearson)
        X = np.clip(X - 30.0, 0, None)
        X = X - X.mean(axis=1, keepdims=True)
    elif mode == 'chroma':                  # octave-folded, floor at -50 dB
        X = np.clip(X - 30.0, 0, None)
        X = X.reshape(len(X), NBINS // 12, 12).sum(axis=1)
    elif mode == 'mix':                     # centred CQT + chroma, equal weight
        a = feat(raw, 'cqt_c'); b = feat(raw, 'chroma')
        return np.concatenate([a, b], axis=1) / np.sqrt(2)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-6
    return X


def cqt_feat(y, tuning=None):
    return feat(raw_cqt(y, tuning))


def shift_raw(R, k):
    """Transpose a raw CQT by k semitones (bins), zero fill."""
    if k == 0:
        return R
    out = np.zeros_like(R)
    if k > 0:
        out[:, k:] = R[:, :-k]
    else:
        out[:, :k] = R[:, -k:]
    return out


@numba.njit(cache=True)
def subseq_dtw(C, pen):
    """Subsequence DTW, MIDI rows free-start/free-end in audio columns.
    Returns per audio end frame: total cost, start column, path length."""
    m, n = C.shape
    D = np.empty(n); S = np.empty(n, np.int64)
    Dp = np.empty(n); Sp = np.empty(n, np.int64)
    for j in range(n):
        Dp[j] = C[0, j]; Sp[j] = j
    for i in range(1, m):
        D[0] = Dp[0] + C[i, 0] + pen; S[0] = Sp[0]
        for j in range(1, n):
            a = Dp[j - 1]; b = Dp[j] + pen; c = D[j - 1] + pen
            if a <= b and a <= c:
                D[j] = a + C[i, j]; S[j] = Sp[j - 1]
            elif b <= c:
                D[j] = b + C[i, j]; S[j] = Sp[j]
            else:
                D[j] = c + C[i, j]; S[j] = S[j - 1]
        for j in range(n):
            Dp[j] = D[j]; Sp[j] = S[j]
    return Dp, Sp


@numba.njit(cache=True)
def full_dtw(C, pen):
    """Standard DTW with fixed ends, additive penalty on non-diagonal steps; returns the path (i, j)."""
    m, n = C.shape
    D = np.full((m + 1, n + 1), np.inf); D[0, 0] = 0.0
    B = np.zeros((m + 1, n + 1), np.int8)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            a = D[i - 1, j - 1]; b = D[i - 1, j] + pen; c = D[i, j - 1] + pen
            if a <= b and a <= c:
                D[i, j] = a + C[i - 1, j - 1]; B[i, j] = 0
            elif b <= c:
                D[i, j] = b + C[i - 1, j - 1]; B[i, j] = 1
            else:
                D[i, j] = c + C[i - 1, j - 1]; B[i, j] = 2
    i, j = m, n
    pi = []; pj = []
    while i > 0 and j > 0:
        pi.append(i - 1); pj.append(j - 1)
        b = B[i, j]
        if b == 0:
            i -= 1; j -= 1
        elif b == 1:
            i -= 1
        else:
            j -= 1
    return np.array(pi[::-1]), np.array(pj[::-1])


def midi_sections(pm, n_frames_midi):
    """Section boundaries in MIDI frames, SEG_BARS bars each (falls back to 8 s blocks)."""
    try:
        db = pm.get_downbeats()
    except Exception:
        db = np.array([])
    end = n_frames_midi / FPS
    if len(db) >= 4:
        b = list(db[::SEG_BARS])
        if b[0] > 0.5:
            b = [0.0] + b
    else:
        b = list(np.arange(0, end, 8.0))
    b = [x for x in b if x < end - 1.0] + [end]
    fr = np.unique(np.round(np.array(b) * FPS).astype(int))
    return [(fr[i], fr[i + 1]) for i in range(len(fr) - 1) if fr[i + 1] - fr[i] >= 8]


def chain(ends, starts, lens, tau, n, cont_bonus, jump_pen, gap_pen):
    """ends[k][e] mean cost, starts[k][e] start col. DP over audio frames. Returns [(k, s, e)]."""
    K = len(ends)
    NEG = -1e18
    F = np.full((K, n), NEG)
    back = np.full((K, n, 2), -1, np.int64)          # (prev k, prev e) ; (-1,-1) = chain start
    Mk = np.full((K, n), NEG); Mk_arg = np.full((K, n), -1, np.int64)     # decaying max of F[k-1]
    Ma = np.full(n, NEG); Ma_arg = np.full((n, 2), -1, np.int64)          # decaying max of all F
    gain = [lens[k] * (tau - ends[k]) for k in range(K)]
    for e in range(n):
        for k in range(K):
            gk = gain[k][e]
            if gk <= 0:
                continue
            s = starts[k][e]
            best, arg = 0.0, (-1, -1)                   # start a chain here (anything before unlabelled)
            if s >= 1:
                if k >= 1 and Mk[k][s - 1] + cont_bonus > best:
                    best, arg = Mk[k][s - 1] + cont_bonus, (k - 1, Mk_arg[k][s - 1])
                if Ma[s - 1] - jump_pen > best:
                    best, arg = Ma[s - 1] - jump_pen, tuple(Ma_arg[s - 1])
            F[k, e] = gk + best
            back[k, e] = arg
        # update decaying maxima with frame e
        for k in range(1, K):
            prev = Mk[k][e - 1] - gap_pen if e else NEG
            if F[k - 1, e] > prev:
                Mk[k][e], Mk_arg[k][e] = F[k - 1, e], e
            else:
                Mk[k][e], Mk_arg[k][e] = prev, Mk_arg[k][e - 1]
        kb = int(np.argmax(F[:, e]))
        prev = Ma[e - 1] - gap_pen if e else NEG
        if F[kb, e] > prev:
            Ma[e], Ma_arg[e] = F[kb, e], (kb, e)
        else:
            Ma[e], Ma_arg[e] = prev, Ma_arg[e - 1]
    k, e = np.unravel_index(np.argmax(F), F.shape)
    if F[k, e] <= 0:
        return []
    out = []
    while k >= 0 and e >= 0:
        out.append((int(k), int(starts[k][e]), int(e)))
        k, e = back[k, e]
    return out[::-1]


def align_features(Fm, Fa, pm):
    """Returns placements [(k, (ms, me), path_midi_frames, path_audio_frames, cost)] and stats."""
    rng = np.random.default_rng(0)
    ia = rng.integers(0, len(Fa), 4000); im = rng.integers(0, len(Fm), 4000)
    rnd = 1.0 - np.sum(Fa[ia] * Fm[im], axis=1)
    tau, sd = float(np.median(rnd)), float(np.std(rnd) + 1e-6)
    pen = 0.5 * tau
    secs = midi_sections(pm, len(Fm))
    ends, starts, lens = [], [], []
    for ms, me in secs:
        C = 1.0 - Fm[ms:me] @ Fa.T
        D, S = subseq_dtw(C.astype(np.float64), pen)
        L = me - ms
        ends.append(D / L); starts.append(S); lens.append(L)
        # disallow placements whose audio span implies an absurd tempo ratio
        span = np.arange(len(Fa)) - S + 1
        bad = (span < 0.6 * L) | (span > 1.7 * L)
        ends[-1] = np.where(bad, np.inf, ends[-1])
    avgL = float(np.mean(lens)) if lens else 1.0
    pl = chain(ends, starts, lens, tau, len(Fa), cont_bonus=0.15 * avgL * tau,
               jump_pen=0.25 * avgL * tau, gap_pen=0.002 * tau)
    out = []
    for k, s, e in pl:
        ms, me = secs[k]
        C = 1.0 - Fm[ms:me] @ Fa[s:e + 1].T
        pi, pj = full_dtw(C.astype(np.float64), pen)
        cost = float(np.mean(C[pi, pj]))
        out.append(dict(k=k, midi=(int(ms), int(me)), audio=(int(s), int(e)), pi=pi + ms, pj=pj + s,
                        cost=cost, contrast=(tau - cost) / sd))
    return out, dict(tau=tau, sd=sd, n_sections=len(secs))


def time_mapper(placements):
    """List of (midi_t0, midi_t1, f(midi_t)->audio_t) per placement."""
    maps = []
    for p in placements:
        tm = p['pi'] / FPS; ta = p['pj'] / FPS
        # collapse to a monotone map: average audio time per MIDI frame
        u, inv = np.unique(tm, return_inverse=True)
        av = np.bincount(inv, weights=ta) / np.bincount(inv)
        av = np.maximum.accumulate(av)
        maps.append((p['midi'][0] / FPS, p['midi'][1] / FPS, u, av, p))
    return maps


def map_time(t, u, av):
    return np.interp(t, u, av, left=av[0] + (t - u[0]) if np.isscalar(t) else None)


# The onset-strength envelope peaks ~2 hops (23 ms) after the physical onset. Measured on the 12 songs whose
# Hooktheory beats are the human alignment of the very same video: aligned MIDI beats were +26 ms (median)
# late without this correction, while Beat This! was within +-3 ms of the humans.
ONSET_BIAS = 0.023


def onset_refine(notes_on, audio_y):
    """Estimate a smooth lag curve lag(t) (s) by cross-correlating aligned onsets with the audio onset envelope."""
    hop = 256
    env = librosa.onset.onset_strength(y=audio_y, sr=SR, hop_length=hop)
    env = (env - ndi.uniform_filter1d(env, 40)).clip(0)
    fps = SR / hop
    n = len(env)
    imp = np.zeros(n)
    idx = np.round(np.asarray(notes_on) * fps).astype(int)
    idx = idx[(idx >= 0) & (idx < n)]
    np.add.at(imp, idx, 1.0)
    imp = ndi.gaussian_filter1d(imp, 1.0)
    W, maxlag = int(6 * fps), int(0.07 * fps)
    centers, lags = [], []
    for c in range(W // 2, n - W // 2, W // 2):
        a, b = c - W // 2, c + W // 2
        if imp[a:b].sum() < 4:
            continue
        xs = [np.dot(imp[a:b], env[a + L:b + L]) if 0 <= a + L and b + L <= n else -1 for L in range(-maxlag, maxlag + 1)]
        xs = np.array(xs)
        if xs.max() <= 0:
            continue
        centers.append(c / fps); lags.append((np.argmax(xs) - maxlag) / fps)
    if len(lags) < 3:
        return None
    lags = ndi.median_filter(np.array(lags), size=5, mode='nearest') - ONSET_BIAS
    return np.array(centers), lags


def audio_raw(corpus, tid, path):
    """Cached raw CQT of the recording (tuning-compensated) under WORK/cqt/<corpus>/<id>.npy."""
    p = os.path.join(g.WORK, 'cqt', corpus, tid + '.npy')
    if os.path.exists(p):
        return np.load(p)
    y = g.decode(path, SR)
    R = raw_cqt(y, librosa.estimate_tuning(y=y, sr=SR))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, R)
    return R


def midi_raw(midi_path):
    """Cached raw CQT of the synthesised MIDI under WORK/cqt/midi/<md5>.npy (audio itself is not kept)."""
    p = os.path.join(g.WORK, 'cqt', 'midi', os.path.basename(midi_path)[:-4] + '.npy')
    if os.path.exists(p):
        return np.load(p)
    pm = pretty_midi.PrettyMIDI(midi_path)
    if pm.get_end_time() < 20 or pm.get_end_time() > 900:
        raise ValueError('midi length %.0f' % pm.get_end_time())
    R = raw_cqt(synth(pm), 0.0)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, R)
    return R


def best_shift(Rm, Ra, shifts=(-3, -2, -1, 0, 1, 2, 3)):
    fa4 = feat(Ra[::4])
    best = None
    for sh in shifts:
        fm4 = feat(shift_raw(Rm[::4], sh))
        C = 1.0 - fm4 @ fa4.T
        D, S = subseq_dtw(C.astype(np.float64), 0.5 * float(np.median(C)))
        sc = float(np.min(D) / len(fm4)) / float(np.median(C))
        if best is None or sc < best[0]:
            best = (sc, sh)
    return best


def align(Ra, midi_path):
    """Ra: raw CQT of the recording. Returns (pm, placements, stats)."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    Rm = midi_raw(midi_path)
    sc, shift = best_shift(Rm, Ra)
    placements, st = align_features(feat(shift_raw(Rm, shift)), feat(Ra), pm)
    st.update(shift=shift, coarse_ratio=sc)
    return pm, placements, st
