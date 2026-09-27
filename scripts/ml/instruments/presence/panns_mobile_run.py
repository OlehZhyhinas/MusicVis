"""Candidate (b): PANNs MobileNetV1 / V2 (AudioSet, 32 kHz) streamed causally: a window of WIN s ending
at t, every HOP s, stamped at t (window end). Clip-wise sigmoid outputs, 527 classes.
Run with the panns venv (torch + torchlibrosa):
  .testdata/instr/venvs/panns/bin/python scripts/ml/instruments/presence/panns_mobile_run.py V1|V2 <listfile> [--win 2] [--hop 0.5]
  ... --bench  prints ms per window (1 thread) and params
Writes .testdata/instr/presence/panns<V>_w<win>/<id>.npz {probs [N,527] f16, t [N], labels}
"""
import os, sys, time, csv
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import common as C
PRE = os.path.join(C.WORK, 'presence', 'pretrained')
sys.path.insert(0, PRE)
import panns_models as M

torch.set_num_threads(6)
V = sys.argv[1]
arg = lambda k, d: float(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
WIN, HOP = arg('--win', 2.0), arg('--hop', 0.5)
SR = 32000
cls = M.MobileNetV1 if V == 'V1' else M.MobileNetV2
model = cls(sample_rate=SR, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
ck = torch.load(os.path.join(PRE, f'MobileNet{V}.pth'), map_location='cpu', weights_only=False)
model.load_state_dict(ck['model'])
model.eval()
import panns_inference.config as PC
labels = PC.labels

if '--bench' in sys.argv:
    torch.set_num_threads(1)
    x = torch.randn(1, int(WIN * SR)) * 0.1
    with torch.no_grad():
        for _ in range(3):
            model(x)
        t0 = time.perf_counter(); n = 20
        for _ in range(n):
            model(x)
    ms = (time.perf_counter() - t0) / n * 1000
    npar = sum(p.numel() for p in model.parameters())
    print(f'PANNs MobileNet{V} win {WIN}s: {ms:.1f} ms per window (torch CPU, 1 thread); hop {HOP}s -> {ms / HOP:.1f} ms per s audio; params {npar / 1e6:.2f} M ({npar * 4 / 1e6:.1f} MB f32)')
    sys.exit()

OUT = os.path.join(C.WORK, 'presence', f'panns{V}_w{WIN:g}')
os.makedirs(OUT, exist_ok=True)
for line in open(sys.argv[2]).read().split('\n'):
    if not line:
        continue
    tid, path = line.split('=', 1)
    op = os.path.join(OUT, tid + '.npz')
    if os.path.exists(op):
        continue
    x = C.decode(path, SR, 1)
    w, h = int(WIN * SR), int(HOP * SR)
    ends = np.arange(w, len(x) + 1, h)
    t0 = time.perf_counter()
    P = []
    with torch.no_grad():
        for b in range(0, len(ends), 64):
            e = ends[b:b + 64]
            xb = torch.from_numpy(np.stack([x[k - w:k] for k in e]))
            P.append(model(xb)['clipwise_output'].numpy())
    P = np.concatenate(P)
    dt = time.perf_counter() - t0
    np.savez_compressed(op, probs=P.astype(np.float16), t=(ends / SR).astype(np.float32), labels=np.array(labels))
    print(tid, P.shape, f'{dt / (len(x) / SR) * 1000:.1f} ms/s', flush=True)
