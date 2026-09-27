"""Export madmom's downbeat RNN (DOWNBEATS_BLSTM: 8 bidirectional 3x25 LSTM networks, Boeck, Krebs &
Widmer 2016) with its 3-resolution spectrogram frontend to public/models/downbeat-blstm.bin for
src/analysis/downbeatBlstm.ts, plus a verification reference (.testdata/ml/beatrnn-ref/<slug>.db.*).

  <venv with madmom>/bin/python scripts/ml/export-downbeat-blstm.py [slug]

Layout as beat-lstm.bin: uint32 LE header length, JSON header, padding, float32 LE data.
"""
import json, os, sys
import numpy as np
from madmom.audio.signal import Signal
from madmom.audio.spectrogram import _diff_frames
from madmom.features.downbeats import RNNDownBeatProcessor

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
data, off = [], [0]

def put(a):
    a = np.asarray(a, dtype=np.float32).ravel()
    o = off[0]; data.append(a); off[0] += a.size
    return {'o': o, 'n': int(a.size)}

proc = RNNDownBeatProcessor()
pre = proc.processors[0]
sig = Signal(np.random.RandomState(0).randn(44100).astype(np.float32) * 0.1, sample_rate=44100)
res = []
for seq in pre.processors[1].processors:
    fr_p, st_p, fi_p, sp_p, di_p = seq.processors
    fs = fr_p(sig); st = st_p(fs); fsp = fi_p(st)
    fb = np.asarray(fsp.filterbank)
    bins, bands = fb.shape
    starts, lens, w = [], [], []
    for b in range(bands):
        nz = np.nonzero(fb[:, b])[0]
        s, e = int(nz[0]), int(nz[-1]) + 1
        starts.append(s); lens.append(e - s); w.append(fb[s:e, b])
    df = _diff_frames(di_p.diff_ratio, hop_size=fs.hop_size, frame_size=fr_p.frame_size, window=np.hanning)
    res.append({'frameSize': int(fr_p.frame_size), 'bins': int(bins), 'bands': int(bands), 'fbStart': starts, 'fbLen': lens,
                'fb': put(np.concatenate(w)), 'diffFrames': int(df)})

def lstm(l):
    g = {}
    for name in ('input_gate', 'forget_gate', 'cell', 'output_gate'):
        gate = getattr(l, name)
        g[name] = {'W': put(gate.weights), 'R': put(gate.recurrent_weights), 'b': put(gate.bias),
                   'p': put(gate.peephole_weights) if gate.peephole_weights is not None else None}
    return {'in': int(l.cell.weights.shape[0]), 'units': int(l.cell.bias.size), **g}

models = []
for net in proc.processors[1].processors[0].processors:
    net = net.processors[0] if hasattr(net, 'processors') else net
    layers = []
    for l in net.layers:
        if type(l).__name__ == 'BidirectionalLayer':
            layers.append({'type': 'bilstm', 'fwd': lstm(l.fwd_layer), 'bwd': lstm(l.bwd_layer)})
        else:
            layers.append({'type': 'dense', 'in': int(l.weights.shape[0]), 'out': int(l.weights.shape[1]), 'W': put(l.weights), 'b': put(l.bias)})
    models.append({'layers': layers})

hdr = {'fps': 100, 'hop': 441, 'sampleRate': 44100, 'logMul': 1, 'logAdd': 1, 'resolutions': res, 'models': models}
blob = np.concatenate(data).astype('<f4').tobytes()
h = json.dumps(hdr, separators=(',', ':')).encode()
h += b' ' * ((4 - (len(h) + 4) % 4) % 4)
out = os.path.join(ROOT, 'public/models/downbeat-blstm.bin')
with open(out, 'wb') as fo:
    fo.write(np.uint32(len(h)).tobytes()); fo.write(h); fo.write(blob)
print('wrote', out, os.path.getsize(out), 'bytes;', [r['bands'] for r in res], 'bands', [r['diffFrames'] for r in res], 'diff frames', len(models), 'nets')

slug = sys.argv[1] if len(sys.argv) > 1 else 'inna-morenito'
pcm = os.path.join(ROOT, '.testdata/live/pcm', slug + '.f32')
if os.path.exists(pcm):
    x = np.fromfile(pcm, dtype='<f4'); x = x[: len(x) // 2 * 2].reshape(-1, 2).mean(1)
    seg = Signal(x[44100 * 20: 44100 * 25], sample_rate=44100)  # a 5 s window
    feat = pre(seg); act = proc(seg)
    d = os.path.join(ROOT, '.testdata/ml/beatrnn-ref'); os.makedirs(d, exist_ok=True)
    feat.astype('<f4').tofile(os.path.join(d, slug + '.db.feat.f32'))
    np.asarray(act, dtype='<f4').tofile(os.path.join(d, slug + '.db.act.f32'))
    print('reference', slug, feat.shape, np.asarray(act).shape)
