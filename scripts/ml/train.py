"""Train the causal multi-task GRU prototype on the dumped features / offline labels.

  .testdata/ml/venv/bin/python scripts/ml/train.py --fold 0 --folds 3 [--hidden 128] [--steps 3000]
  .testdata/ml/venv/bin/python scripts/ml/train.py --fold -1          # all songs (for mixes / sizing)

Songs are split into folds by index (sorted slug order); fold k's songs are held out.
Writes .testdata/ml/models/<name>.onnx (streaming step: x[1,1,160], h[L,1,H] -> y[1,1,17], h')
and <name>.json (held-out songs, label order, output activations).
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.join(os.path.dirname(__file__), '../../.testdata/ml')
ap = argparse.ArgumentParser()
ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--folds', type=int, default=3)
ap.add_argument('--hidden', type=int, default=128)
ap.add_argument('--layers', type=int, default=2)
ap.add_argument('--steps', type=int, default=3000)
ap.add_argument('--seq', type=int, default=1720)  # 20 s
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--name', type=str, default='')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--aug', type=int, default=1)  # use the tempo/pitch-shifted copies
ap.add_argument('--eq', type=int, default=1)
ap.add_argument('--drop', type=float, default=0.0)  # dropout on the input layer and GRU output
ap.add_argument('--shift', type=int, default=0)  # random band roll +-N (pitch-shift-like)
ap.add_argument('--mask', type=int, default=0)  # SpecAugment: up to N masked bands, 2 masks
ap.add_argument('--inw', type=int, default=96)
ap.add_argument('--data', type=str, default='data')  # random spectral tilt / noise floor augmentation
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)

idx = json.load(open(os.path.join(ROOT, a.data, 'index.json')))
F = idx['feat']; LAB = idx['labels']; L = len(LAB)
# Folds by base song (augmented copies '<song>@<speed>' follow their song); held-out songs are scored unaugmented.
songs = list(dict.fromkeys(s.get('song', s['slug']) for s in idx['songs']))
held = [s for i, s in enumerate(songs) if a.fold >= 0 and i % a.folds == a.fold]
train = [s['slug'] for s in idx['songs'] if s.get('song', s['slug']) not in held and (a.aug or s.get('speed', 1) == 1)]
name = a.name or (f'gru{a.hidden}-f{a.fold}of{a.folds}' if a.fold >= 0 else f'gru{a.hidden}-all')
print('held out:', held)

def load(s):
    x = np.fromfile(os.path.join(ROOT, a.data, s + '.f32'), dtype=np.float32).reshape(-1, F)
    y = np.fromfile(os.path.join(ROOT, a.data, s + '.lab.f32'), dtype=np.float32).reshape(-1, L)
    return x, y

data = [load(s) for s in train]
allx = np.concatenate([d[0] for d in data])
mu = allx.mean(0); sd = allx.std(0) + 1e-3

# Output layout (17): beat, downbeat (logits), beatCos, beatSin, barCos, barSin (raw),
# stems x4, onsets x4, kick, snare, hat (logits -> sigmoid).
OUT = ['beat', 'downbeat', 'beatCos', 'beatSin', 'barCos', 'barSin',
       'stems.drums', 'stems.bass', 'stems.vocals', 'stems.other',
       'onset.drums', 'onset.bass', 'onset.vocals', 'onset.other', 'kick', 'snare', 'hat']
SIG = [0, 1] + list(range(6, 17))
li = {n: i for i, n in enumerate(LAB)}
tgt_cols = [li[n] for n in OUT]
valid_col = li['beatValid']

class Net(nn.Module):
    def __init__(s, H, NL):
        super().__init__()
        s.register_buffer('mu', torch.tensor(mu)); s.register_buffer('sd', torch.tensor(sd))
        W = a.inw
        s.inp = nn.Sequential(nn.Dropout(a.drop / 2), nn.Linear(F, W), nn.ReLU(), nn.Dropout(a.drop), nn.Linear(W, W), nn.ReLU())
        s.gru = nn.GRU(W, H, NL, batch_first=True, dropout=a.drop if NL > 1 else 0)
        s.dr = nn.Dropout(a.drop)
        s.out = nn.Linear(H, len(OUT))
    def forward(s, x, h=None):
        z = s.inp((x - s.mu) / s.sd)
        z, h = s.gru(z, h)
        return s.out(s.dr(z)), h

class Step(nn.Module):
    """Streaming step with the output activations applied (sigmoid on the event / energy heads)."""
    def __init__(s, net):
        super().__init__(); s.net = net
        m = torch.zeros(len(OUT)); m[SIG] = 1; s.register_buffer('m', m)
    def forward(s, x, h):
        y, h2 = s.net(x, h)
        return s.m * torch.sigmoid(y) + (1 - s.m) * y, h2

dev = os.environ.get('DEV', 'cpu')
net = Net(a.hidden, a.layers).to(dev)
nparams = sum(p.numel() for p in net.parameters())
print('params', nparams, 'device', dev)
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.steps, pct_start=0.1)
lens = np.array([len(d[0]) for d in data], dtype=np.float64)
pw = torch.tensor(4.0, device=dev)

def batch():
    xs, ys = [], []
    for _ in range(a.batch):
        i = np.random.choice(len(data), p=lens / lens.sum())
        x, y = data[i]
        s0 = np.random.randint(0, max(1, len(x) - a.seq))
        xb = x[s0:s0 + a.seq].copy(); yb = y[s0:s0 + a.seq]
        # Gain augmentation in feature space (capture level varies a lot live): +-12 dB.
        g = 10 ** (np.random.uniform(-12, 12) / 20)
        B = F // 2
        mag = np.expm1(xb[:, :B]) * g
        lm = np.log1p(mag)
        fl = np.maximum(0, np.diff(lm, axis=0, prepend=lm[:1]))
        if a.eq:
            # Random spectral tilt (+-6 dB across the bands) and a small noise floor (room tone / capture hiss).
            tilt = 10 ** (np.random.uniform(-6, 6) * np.linspace(-0.5, 0.5, B) / 20)
            noise = np.random.uniform(0, 0.02) * np.random.rand(1, B) * np.random.rand(len(mag), B)
            lm = np.log1p(mag * tilt + noise * mag.mean())
            fl = np.maximum(0, np.diff(lm, axis=0, prepend=lm[:1]))
        if a.shift:
            k = np.random.randint(-a.shift, a.shift + 1)
            if k:
                lm = np.roll(lm, k, 1); fl = np.roll(fl, k, 1)
                if k > 0: lm[:, :k] = lm[:, k:k + 1]; fl[:, :k] = 0
                else: lm[:, k:] = lm[:, k - 1:k]; fl[:, k:] = 0
        if a.mask:
            for _ in range(2):
                w = np.random.randint(0, a.mask + 1); b0 = np.random.randint(0, B - w + 1)
                lm[:, b0:b0 + w] = lm.mean(); fl[:, b0:b0 + w] = 0
        xb = np.concatenate([lm, fl], 1).astype(np.float32)
        xs.append(xb); ys.append(yb)
    return torch.tensor(np.stack(xs), device=dev), torch.tensor(np.stack(ys), device=dev)

bce = nn.BCEWithLogitsLoss(reduction='none')
vdata = [load(sg) for sg in held]

def losses(p, y):
    t = y[..., tgt_cols]; v = y[..., valid_col]
    lb = (bce(p[..., 0], t[..., 0]) * (1 + (pw - 1) * t[..., 0])).mean() + (bce(p[..., 1], t[..., 1]) * (1 + (pw * 2 - 1) * t[..., 1])).mean()
    lph = (((p[..., 2:6] - t[..., 2:6]) ** 2).mean(-1) * v).sum() / (v.sum() + 1)
    le = bce(p[..., 6:], t[..., 6:]).mean()
    return lb, lph, le

def val():
    if not vdata: return ''
    net.eval(); r = np.zeros(3)
    with torch.no_grad():
        for x, y in vdata:
            p, _ = net(torch.tensor(x[None], device=dev)); r += [q.item() for q in losses(p, torch.tensor(y[None], device=dev))]
    net.train(); r /= len(vdata)
    return f' | held-out beat {r[0]:.3f} phase {r[1]:.3f} energy {r[2]:.3f}'
t0 = time.time()
for step in range(a.steps):
    x, y = batch()
    p, _ = net(x)
    lb, lph, le = losses(p, y)
    loss = lb + 0.5 * lph + 2 * le
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    if step % 250 == 0 or step == a.steps - 1:
        print(f'{step} loss {loss.item():.4f} beat {lb.item():.3f} phase {lph.item():.3f} energy {le.item():.3f}{val() if step % 500 == 0 or step == a.steps - 1 else ""} {time.time() - t0:.0f}s', flush=True)

net = net.cpu().eval()
os.makedirs(os.path.join(ROOT, 'models'), exist_ok=True)
step_m = Step(net).eval()
x1 = torch.zeros(1, 1, F); h1 = torch.zeros(a.layers, 1, a.hidden)
path = os.path.join(ROOT, 'models', name + '.onnx')
torch.onnx.export(step_m, (x1, h1), path, input_names=['x', 'h'], output_names=['y', 'h_out'], opset_version=17, dynamo=False)
json.dump({'held': held, 'train': train, 'out': OUT, 'hidden': a.hidden, 'layers': a.layers, 'params': nparams,
           'bytes': os.path.getsize(path)}, open(os.path.join(ROOT, 'models', name + '.json'), 'w'), indent=1)
print('wrote', path, os.path.getsize(path), 'bytes')
