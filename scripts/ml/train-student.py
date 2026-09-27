"""Causal beat / downbeat student: an LSTM on the beat RNN's frontend features (corpus-feats.ts,
162 @ 100 fps) trained on Beat This! labels (corpus-beats.py) of a large corpus (FMA + own tracks).

  .testdata/ml/venv/bin/python scripts/ml/train-student.py --corpus fma,own [--hidden 64] [--steps 20000]

Data root ~/personal/MusicVis-data: ml/feats-<corpus>/<id>.f16 and labels/beatthis/<corpus>/<id>.json.
Held out: 3 % of each corpus by id hash (validation), every 3rd 'own' track, and the 'test' corpus
(the 13 test songs) entirely. Writes .testdata/ml/models/<name>.onnx (streaming step: x[1,1,162],
h[L,1,H], c[L,1,H] -> y[1,1,2] (beat, downbeat probabilities), h', c') and <name>.pt / <name>.json.
"""
import argparse, glob, hashlib, json, os, time
import numpy as np
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument('--corpus', default='fma')
ap.add_argument('--hidden', type=int, default=64)
ap.add_argument('--layers', type=int, default=2)
ap.add_argument('--steps', type=int, default=20000)
ap.add_argument('--seq', type=int, default=800)
ap.add_argument('--batch', type=int, default=48)
ap.add_argument('--lr', type=float, default=2e-3)
ap.add_argument('--name', default='')
ap.add_argument('--max-clips', type=int, default=0)
ap.add_argument('--smoke', action='store_true')
ap.add_argument('--db-lag', type=int, default=0)  # downbeat head predicts the frame this many frames back (right context)  # no held-out split (pipeline check only)
a = ap.parse_args()
DATA = os.path.expanduser('~/personal/MusicVis-data')
ML = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../.testdata/ml')
name = a.name or f'student-lstm{a.hidden}x{a.layers}'
torch.manual_seed(0); np.random.seed(0)

def held(corpus, rid):
    h = int(hashlib.md5(rid.encode()).hexdigest()[:8], 16)
    if corpus == 'test': return True
    if corpus == 'own': return h % 3 == 0
    return h % 100 < 3

def spikes(times, n):
    y = np.zeros(n, np.float32)
    for t in times:
        k = int(round(t * 100))
        for d, v in ((-1, 0.5), (0, 1.0), (1, 0.5)):
            if 0 <= k + d < n: y[k + d] = max(y[k + d], v)
    return y

def clips(corpus):
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, 'ml', f'feats-{corpus}', '*.f16'))):
        rid = os.path.basename(f)[:-4]
        lab = os.path.join(DATA, 'labels', 'beatthis', corpus, rid + '.json')
        if os.path.exists(lab): out.append((corpus, rid, f, lab))
    return out

train, val = [], []
for c in a.corpus.split(','):
    cl = clips(c)
    if a.max_clips: cl = cl[:a.max_clips]
    for x in cl: (val if held(c, x[1]) and not a.smoke else train).append(x)
print(f'{len(train)} train clips, {len(val)} held-out clips')

def load(item):
    _, _, f, lab = item
    x = np.fromfile(f, dtype='<f2').reshape(-1, 162)  # kept as float16 in RAM
    j = json.load(open(lab))
    d = spikes(j['downbeats'], len(x))
    if a.db_lag:
        d = np.concatenate([np.zeros(a.db_lag, np.float32), d[:-a.db_lag]])  # output k <- label k - lag
    y = np.stack([spikes(j['beats'], len(x)), d], 1)
    return x, y

t0 = time.time()
TR = [load(i) for i in train]
VA = [load(i) for i in val]
print(f'loaded in {time.time() - t0:.0f} s')
allx = np.concatenate([x for x, _ in TR[:2000]]).astype(np.float32)
mu, sd = allx.mean(0), allx.std(0) + 1e-3

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.register_buffer('mu', torch.tensor(mu)); s.register_buffer('sd', torch.tensor(sd))
        s.inp = nn.Sequential(nn.Linear(162, a.hidden), nn.ReLU())
        s.lstm = nn.LSTM(a.hidden, a.hidden, a.layers, batch_first=True)
        s.out = nn.Linear(a.hidden, 2)
    def forward(s, x, hc=None):
        z, hc = s.lstm(s.inp((x - s.mu) / s.sd), hc)
        return s.out(z), hc

class Step(nn.Module):
    def __init__(s, net): super().__init__(); s.net = net
    def forward(s, x, h, c):
        y, (h2, c2) = s.net(x, (h, c))
        return torch.sigmoid(y), h2, c2

dev = os.environ.get('DEV', 'mps' if torch.backends.mps.is_available() else 'cpu')
net = Net().to(dev)
print('params', sum(p.numel() for p in net.parameters()), 'device', dev)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
lens = np.array([len(x) for x, _ in TR], np.float64); P = lens / lens.sum()
bce = nn.BCEWithLogitsLoss(reduction='none')
W = torch.tensor([5.0, 12.0], device=dev)

def batch():
    xs, ys = [], []
    for i in np.random.choice(len(TR), a.batch, p=P):
        x, y = TR[i]
        s0 = np.random.randint(0, max(1, len(x) - a.seq))
        xb, yb = x[s0:s0 + a.seq].astype(np.float32), y[s0:s0 + a.seq]
        if len(xb) < a.seq:
            pad = a.seq - len(xb); xb = np.pad(xb, ((0, pad), (0, 0))); yb = np.pad(yb, ((0, pad), (0, 0)))
        # gain in the log domain: log10(1 + g*s) ~ shift; emulate +-10 dB on the band half
        g = 10 ** (np.random.uniform(-10, 10) / 20)
        lin = 10 ** xb[:, :81] - 1; xb[:, :81] = np.log10(1 + g * lin)
        xb[:, 81:] = np.maximum(0, np.diff(xb[:, :81], axis=0, prepend=xb[:1, :81]))
        xs.append(xb); ys.append(yb)
    return torch.tensor(np.stack(xs), device=dev), torch.tensor(np.stack(ys), device=dev)

def lossf(p, y):
    return (bce(p, y) * (1 + (W - 1) * y)).mean()

def validate():
    net.eval(); tot = 0; n = 0
    with torch.no_grad():
        for x, y in VA[:40]:
            x, y = x[:3000], y[:3000]  # 30 s of each held-out clip
            p, _ = net(torch.tensor(x[None].astype(np.float32), device=dev))
            tot += lossf(p, torch.tensor(y[None], device=dev)).item(); n += 1
    net.train(); return tot / max(1, n)

t0 = time.time()
for step in range(a.steps):
    x, y = batch()
    p, _ = net(x)
    loss = lossf(p, y)
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    if step % 1000 == 0 or step == a.steps - 1:
        print(f'{step} loss {loss.item():.4f} val {validate():.4f} {time.time() - t0:.0f}s', flush=True)
        torch.save(net.state_dict(), os.path.join(ML, 'models', name + '.pt'))

net = net.cpu().eval()
os.makedirs(os.path.join(ML, 'models'), exist_ok=True)
torch.save(net.state_dict(), os.path.join(ML, 'models', name + '.pt'))
x1 = torch.zeros(1, 1, 162); h1 = torch.zeros(a.layers, 1, a.hidden)
path = os.path.join(ML, 'models', name + '.onnx')
torch.onnx.export(Step(net), (x1, h1, h1.clone()), path, input_names=['x', 'h', 'c'], output_names=['y', 'h_out', 'c_out'], opset_version=17, dynamo=False)
json.dump({'hidden': a.hidden, 'layers': a.layers, 'dbLag': a.db_lag, 'corpus': a.corpus, 'train': len(train), 'val': len(val), 'bytes': os.path.getsize(path)},
          open(os.path.join(ML, 'models', name + '.json'), 'w'), indent=1)
print('wrote', path)
