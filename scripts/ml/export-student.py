"""Export a trained beat / downbeat student (train-student.py .pt) for src/analysis/beatStudent.ts,
and (with --onnx) the ONNX streaming step for scripts/ml/downbeat-eval.ts.

  .testdata/ml/venv/bin/python scripts/ml/export-student.py <name> [--out public/models/beat-student.bin] [--onnx]

Layout as public/models/beat-lstm.bin: uint32 LE header length, JSON header, padding, float32 LE data.
Header: {inputs, hidden, layers, outputs, mu, sd, inp: {W, b}, lstm: [{Wih, Whh, bih, bhh}], out: {W, b}}
(torch layouts: Linear W [out, in]; LSTM gates stacked i, f, g, o).
"""
import json, os, sys
import numpy as np
import torch

name = sys.argv[1]
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
ML = os.path.join(ROOT, '.testdata/ml/models')
out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else os.path.join(ROOT, 'public/models/beat-student.bin')
sd = torch.load(os.path.join(ML, name + '.pt'), map_location='cpu')
meta = json.load(open(os.path.join(ML, name + '.json'))) if os.path.exists(os.path.join(ML, name + '.json')) else {}
H = sd['inp.0.weight'].shape[0]
L = len([k for k in sd if k.startswith('lstm.weight_ih_l')])
data, off = [], [0]

def put(t):
    a = t.detach().numpy().astype('<f4').ravel()
    o = off[0]; data.append(a); off[0] += a.size
    return {'o': o, 'n': int(a.size)}

hdr = {'dbLag': int(meta.get('dbLag', 0)), 'inputs': int(sd['inp.0.weight'].shape[1]), 'hidden': H, 'layers': L, 'outputs': int(sd['out.weight'].shape[0]),
       'mu': put(sd['mu']), 'sd': put(sd['sd']), 'inp': {'W': put(sd['inp.0.weight']), 'b': put(sd['inp.0.bias'])},
       'lstm': [{'Wih': put(sd[f'lstm.weight_ih_l{l}']), 'Whh': put(sd[f'lstm.weight_hh_l{l}']),
                 'bih': put(sd[f'lstm.bias_ih_l{l}']), 'bhh': put(sd[f'lstm.bias_hh_l{l}'])} for l in range(L)],
       'out': {'W': put(sd['out.weight']), 'b': put(sd['out.bias'])}}
blob = np.concatenate(data).tobytes()
h = json.dumps(hdr, separators=(',', ':')).encode()
h += b' ' * ((4 - (len(h) + 4) % 4) % 4)
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'wb') as f:
    f.write(np.uint32(len(h)).tobytes()); f.write(h); f.write(blob)
print('wrote', out, os.path.getsize(out), 'bytes')

if '--onnx' in sys.argv:
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.register_buffer('mu', sd['mu']); s.register_buffer('sd', sd['sd'])
            s.inp = nn.Sequential(nn.Linear(hdr['inputs'], H), nn.ReLU())
            s.lstm = nn.LSTM(H, H, L, batch_first=True)
            s.out = nn.Linear(H, hdr['outputs'])
        def forward(s, x, h, c):
            z, (h2, c2) = s.lstm(s.inp((x - s.mu) / s.sd), (h, c))
            return torch.sigmoid(s.out(z)), h2, c2
    net = Net(); net.load_state_dict(sd); net.eval()
    x1 = torch.zeros(1, 1, hdr['inputs']); h1 = torch.zeros(L, 1, H)
    p = os.path.join(ML, name + '.onnx')
    torch.onnx.export(net, (x1, h1, h1.clone()), p, input_names=['x', 'h', 'c'], output_names=['y', 'h_out', 'c_out'], opset_version=17, dynamo=False)
    meta.update({'hidden': H, 'layers': L})
    json.dump(meta, open(os.path.join(ML, name + '.json'), 'w'), indent=1)
    print('wrote', p)
