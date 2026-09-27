"""Export madmom's online beat RNN (BEATS_LSTM, 8 uni-directional 3x25 peephole LSTMs, Boeck et al.)
and its spectrogram frontend's filterbank to public/models/beat-lstm.bin for src/analysis/beatRnn.ts.

Also writes a verification reference (.testdata/ml/beatrnn-ref/<slug>.{feat,act}.f32): madmom's own
features and ensemble activation in its file mode (frames centred on i * 441), which beatRnn.ts
reproduces with centred framing (scripts/ml/beatrnn-check.ts).

  <venv with madmom>/bin/python scripts/ml/export-beat-lstm.py [slug]

File layout: uint32 LE header length, JSON header, zero padding to 4 bytes, then float32 LE data.
Header: {fps, frameSize, sampleRate, bands, fbStart[bands], fbLen[bands], models: [{layers: [...]}]}
with each array's offset (in floats) and length in the data block.
"""
import json, os, sys
import numpy as np
from madmom.models import BEATS_LSTM
from madmom.ml.nn import NeuralNetwork
from madmom.features.beats import RNNBeatProcessor

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
data = []
off = [0]

def put(a):
    a = np.asarray(a, dtype=np.float32).ravel()
    o = off[0]; data.append(a); off[0] += a.size
    return {'o': o, 'n': int(a.size)}

# Frontend: take the filterbank from madmom's own processor chain.
proc = RNNBeatProcessor(online=True)
chain = proc.processors[0].processors[1].processors[0].processors  # frames, stft, filt, spec, diff
frames_p, stft_p, filt_p, spec_p, diff_p = chain
from madmom.audio.signal import Signal
sig = Signal(np.random.RandomState(0).randn(44100).astype(np.float32) * 0.1, sample_rate=44100)
fs = frames_p(sig)
st = stft_p(fs)
fsp = filt_p(st)
fb = np.asarray(fsp.filterbank)  # (bins, bands)
bins, bands = fb.shape
starts, lens, w = [], [], []
for b in range(bands):
    nz = np.nonzero(fb[:, b])[0]
    s, e = int(nz[0]), int(nz[-1]) + 1
    starts.append(s); lens.append(e - s); w.append(fb[s:e, b])
fbw = put(np.concatenate(w))
hdr = {'fps': 100, 'frameSize': int(frames_p.frame_size), 'hop': 441, 'sampleRate': 44100, 'bins': int(bins), 'bands': int(bands),
       'fbStart': starts, 'fbLen': lens, 'fb': fbw, 'logMul': 1, 'logAdd': 1, 'diffFrames': 1, 'models': []}

for f in BEATS_LSTM:
    nn = NeuralNetwork.load(f)
    layers = []
    for l in nn.layers:
        if type(l).__name__ == 'LSTMLayer':
            g = {}
            for name in ('input_gate', 'forget_gate', 'cell', 'output_gate'):
                gate = getattr(l, name)
                g[name] = {'W': put(gate.weights), 'R': put(gate.recurrent_weights), 'b': put(gate.bias),
                           'p': put(gate.peephole_weights) if gate.peephole_weights is not None else None}
            layers.append({'type': 'lstm', 'in': int(l.cell.weights.shape[0]), 'units': int(l.cell.bias.size), **g})
        else:
            layers.append({'type': 'dense', 'in': int(l.weights.shape[0]), 'out': int(l.weights.shape[1]), 'W': put(l.weights), 'b': put(l.bias)})
    hdr['models'].append({'layers': layers})

blob = np.concatenate(data).astype('<f4').tobytes()
h = json.dumps(hdr, separators=(',', ':')).encode()
h += b' ' * ((4 - (len(h) + 4) % 4) % 4)
os.makedirs(os.path.join(ROOT, 'public/models'), exist_ok=True)
out = os.path.join(ROOT, 'public/models/beat-lstm.bin')
with open(out, 'wb') as fo:
    fo.write(np.uint32(len(h)).tobytes()); fo.write(h); fo.write(blob)
print('wrote', out, os.path.getsize(out), 'bytes;', bins, 'bins', bands, 'bands', len(BEATS_LSTM), 'models')

# Verification reference on one song (madmom file mode: centred frames).
slug = sys.argv[1] if len(sys.argv) > 1 else 'inna-morenito'
pcm = os.path.join(ROOT, '.testdata/live/pcm', slug + '.f32')
if os.path.exists(pcm):
    x = np.fromfile(pcm, dtype='<f4'); x = Signal(x[: len(x) // 2 * 2].reshape(-1, 2).mean(1)[: 44100 * 30], sample_rate=44100)
    feat = proc.processors[0](x)
    act = proc(x)
    d = os.path.join(ROOT, '.testdata/ml/beatrnn-ref'); os.makedirs(d, exist_ok=True)
    feat.astype('<f4').tofile(os.path.join(d, slug + '.feat.f32'))
    np.asarray(act, dtype='<f4').tofile(os.path.join(d, slug + '.act.f32'))
    print('reference', slug, feat.shape, act.shape)
