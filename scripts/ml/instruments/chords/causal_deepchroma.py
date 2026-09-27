"""Causal variant of madmom's deep-chroma chord recogniser.

madmom's DeepChromaProcessor (audio/chroma.py) is:
  1. spectrogram: FramedSignalProcessor(frame_size=8192, fps=10) -> STFT -> LogarithmicFilteredSpectrogram
     (24 bands, 65-2100 Hz, unique_filters) => ~105-bin log-spec frames every 100 ms. The base analysis
     window (8192 samples @ 44.1 kHz = 185.8 ms) is centred (madmom FramedSignalProcessor origin=0), so
     each frame already looks ~93 ms into the future.
  2. context stack: FramedSignalProcessor(frame_size=15, hop_size=1) over the spectrogram frames, i.e.
     +-7 frames (700 ms) of context CENTRED on the current frame -> flattened to 1575 (=105*15) -> fed to
     a small feedforward DNN (538,124 params: 1575->256->256->256->12, see param count check in this
     file's __main__) that outputs a 12-bin chroma vector.
  So total look-ahead of the stock pipeline is ~93 ms (base window) + 700 ms (context stack) = ~793 ms.
  That's the thing to fix for a live tracker: the context stack is the big offender.

This module reimplements only the context-stacking step CAUSALLY (context = the 15 MOST RECENT
spectrogram frames ending at the current one, zero-padded at the start of the song), keeping the same
base spectrogram window and the same pretrained DNN weights (nothing retrained). That leaves only the
~93 ms base-window look-ahead, comfortably under the "<=100ms" bar in the brief. Everything downstream
(root+quality decoding) is then a genuinely causal decoder: an online HMM forward-filter (not the
offline CRF/Viterbi shipped in madmom.features.chords), so the whole path can run in a live stream.

Usage (labelling / scoring, not for shipping - this is Python for evaluation only):
  .testdata/instr/venvs/adtof/bin/python scripts/ml/instruments/chords/causal_deepchroma.py --corpus test
"""
import argparse, json, os, sys, tempfile, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import common
import numpy as np

FPS = 10.0
CONTEXT = 15  # frames stacked (causal: all in the past, current frame last)
STAY_PROB = 0.93  # per 100ms-frame self-transition prob (~1.4s expected chord hold, tunable)
SHARPNESS = 10.0  # like harmony.ts HARMONY_PARAMS.sharp: exponent on template cosine score
NO_CHORD_FLOOR = 0.12  # chroma L2 energy below this -> favor N

# madmom's DeepChromaProcessor output is a plain pitch-class profile with bin 0 = C (standard chroma
# convention, e.g. librosa/PCP), NOT the chord-CLASS-id order used internally by majmin_targets_to_chord_labels
# (whose pred_to_cl starts at 'A' -- that's an index into the CRF's 25 learned classes, decoupled from
# the raw chroma bin layout). Verified empirically: for a segment madmom's own CRF calls 'C:maj', the
# dominant chroma bins are exactly {0, 4, 7} (root, major third, fifth of C major starting from bin 0).
PITCH_CLASS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def build_templates():
    """25 templates: 12 major, 12 minor (root+third+fifth, mean-centred like src/analysis/harmony.ts), + N."""
    labels = []
    T = np.zeros((25, 12), dtype=np.float64)
    for r in range(12):
        maj = np.zeros(12)
        maj[[r, (r + 4) % 12, (r + 7) % 12]] = [1.15, 1.0, 0.95]
        T[r] = maj - maj.mean()
        labels.append(f'{PITCH_CLASS[r]}:maj')
    for r in range(12):
        mnr = np.zeros(12)
        mnr[[r, (r + 3) % 12, (r + 7) % 12]] = [1.15, 1.0, 0.95]
        T[12 + r] = mnr - mnr.mean()
        labels.append(f'{PITCH_CLASS[r]}:min')
    T[24] = 0.0  # N: handled by the energy floor / uniform-low emission below
    labels.append('N')
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    T[24] = 0.0
    return T, labels


def deep_chroma_causal(wav_path):
    """Returns (times[T], chroma[T,12]) at 10 fps, using the pretrained madmom chroma DNN but a
    causal (all-past) context stack instead of the stock +-7-frame centred one."""
    from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
    from madmom.audio.stft import ShortTimeFourierTransformProcessor
    from madmom.audio.spectrogram import LogarithmicFilteredSpectrogramProcessor
    from madmom.ml.nn import NeuralNetworkEnsemble
    from madmom.models import CHROMA_DNN

    sig = SignalProcessor(num_channels=1, sample_rate=44100)
    frames = FramedSignalProcessor(frame_size=8192, fps=FPS)
    stft = ShortTimeFourierTransformProcessor()
    spec = LogarithmicFilteredSpectrogramProcessor(num_bands=24, fmin=65, fmax=2100, unique_filters=True)

    s = sig(wav_path)
    fr = frames(s)
    st = stft(fr)
    sp = np.asarray(spec(st))  # (T, nbins)
    T, nbins = sp.shape

    nn = NeuralNetworkEnsemble.load(CHROMA_DNN)

    stacked = np.zeros((T, CONTEXT * nbins), dtype=np.float32)
    padded = np.vstack([np.zeros((CONTEXT - 1, nbins), dtype=sp.dtype), sp])
    for i in range(T):
        window = padded[i : i + CONTEXT]  # causal: frames [i-14 .. i] (zero-padded at song start)
        stacked[i] = window.reshape(-1)

    chroma = nn.process(stacked)
    times = np.arange(T) / FPS
    return times, np.asarray(chroma)


def online_hmm_decode(chroma, templates, labels):
    """Causal HMM forward filter over the 25 chord classes (no future frames used)."""
    n_states = templates.shape[0]
    trans = np.full((n_states, n_states), (1 - STAY_PROB) / (n_states - 1))
    np.fill_diagonal(trans, STAY_PROB)
    alpha = np.full(n_states, 1.0 / n_states)
    out_idx = np.zeros(len(chroma), dtype=np.int64)
    out_conf = np.zeros(len(chroma), dtype=np.float64)
    for t in range(len(chroma)):
        c = chroma[t]
        energy = np.linalg.norm(c)
        c_n = c / (energy + 1e-9)
        sim = templates @ c_n  # cosine-ish score per state, N stays 0
        # N gets a competitive score only when energy is low (no strong harmonic content)
        sim[24] = NO_CHORD_FLOOR - energy if energy < NO_CHORD_FLOOR else -1.0
        emission = np.exp(SHARPNESS * sim)
        emission /= emission.sum()
        alpha = emission * (trans.T @ alpha)
        alpha /= alpha.sum() + 1e-12
        j = int(np.argmax(alpha))
        out_idx[t] = j
        out_conf[t] = alpha[j]
    return out_idx, out_conf, labels


def indices_to_segments(times, idx, labels, fps):
    spf = 1.0 / fps
    segs = []
    prev = None
    start = 0.0
    for i, j in enumerate(idx):
        lab = labels[j]
        if lab != prev:
            if prev is not None:
                segs.append([start, times[i], prev])
            start = times[i]
            prev = lab
    if prev is not None:
        segs.append([start, times[-1] + spf, prev])
    return segs


def atomic_write_json(path, obj):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix='.tmp-')
    with os.fdopen(fd, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='test', choices=['test', 'own'])
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--songs', default='')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    tracks = common.tracks(args.corpus)
    if args.songs:
        want = set(args.songs.split(','))
        tracks = [(t, p) for t, p in tracks if t in want]
    if args.eval_only:
        tracks = [(t, p) for t, p in tracks if common.is_eval(args.corpus, t, p)]
    if args.limit:
        tracks = tracks[: args.limit]

    templates, labels = build_templates()
    total_audio_s = 0.0
    total_cpu_s = 0.0

    for i, (tid, path) in enumerate(tracks):
        out_path = common.label_path('causal-deepchroma', args.corpus, tid, 'json')
        if os.path.exists(out_path):
            continue
        wav_path = os.path.join(common.TESTDATA, 'instr', 'wav', tid + '.wav')
        if not os.path.exists(wav_path):
            import soundfile as sf
            x = common.decode(path, sr=44100, channels=1)
            wav_path = os.path.join(common.WORK, 'wav', tid + '.wav')
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)
            sf.write(wav_path, x, 44100)

        t0 = time.time()
        times, chroma = deep_chroma_causal(wav_path)
        idx, conf, _ = online_hmm_decode(chroma, templates, labels)
        cpu_s = time.time() - t0
        audio_s = times[-1] + 1.0 / FPS if len(times) else 0.0
        total_audio_s += audio_s
        total_cpu_s += cpu_s

        segs = indices_to_segments(times, idx, labels, FPS)
        atomic_write_json(
            out_path,
            {
                'teacher': 'causal-deepchroma',
                'id': tid,
                'segments': segs,
                'fps': FPS,
                'lookahead_ms': 93,  # base spectrogram window only; context stack is causal
                'cpu_s': cpu_s,
                'audio_s': audio_s,
            },
        )
        print(f'[{i+1}/{len(tracks)}] {tid}: {audio_s:.1f}s audio in {cpu_s:.2f}s cpu ({cpu_s/max(audio_s,1e-6)*100:.2f}% of realtime)', file=sys.stderr)

    if total_audio_s > 0:
        print(f'TOTAL: {total_audio_s:.1f}s audio, {total_cpu_s:.2f}s cpu => {total_cpu_s/total_audio_s*100:.3f}% of one core per second of audio', file=sys.stderr)


if __name__ == '__main__':
    main()
