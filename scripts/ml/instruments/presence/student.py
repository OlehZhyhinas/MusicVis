"""Candidate (d): distilled causal student on the cheap log-mel front end (features.py, 86 fps).

  in: log-mel [64] per 512-sample hop (44.1 kHz)  ->  Linear(64,C) + ReLU
      3 causal dilated Conv1d(C,C,k=3, d=1,2,4) with residual + ReLU  (receptive field 15 hops ~ 170 ms)
      GRU(C,H)
  heads: env  = Linear(C+H, 6)   per-stem level in dB RELATIVE TO THE MIX (drums bass vocals guitar piano other);
                                 absolute = rel + mix dB (the mix RMS of the same causal window, computed exactly)
         act  = Linear(H, G)     group activity logits (groups.py), trained on teacher.py targets where present
Training data: non-eval own tracks (+ FMA when present); teacher envelopes from our d6 cache (causal)
or the shared labels/demucs6 cache (centered window -> shifted by one hop).
  .testdata/instr/venvs/torch/bin/python student.py train --name s48 --C 48 --H 48 [--steps 4000]
  .testdata/instr/venvs/torch/bin/python student.py score --name s48
"""
import os, sys, json, time, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import common as C
import features as FE
import groups as GR
import stemscore as SS

torch.set_num_threads(6)
PRES = os.path.join(C.WORK, 'presence')
STEMS = GR.STEMS6
MODELS = os.path.join(PRES, 'models')
os.makedirs(MODELS, exist_ok=True)
TGT = os.path.join(PRES, 'targets')


def d6_env(corpus, tid):
    """[T, 7] causal RMS (6 stems + mix) at 86 fps, or None."""
    p = os.path.join(PRES, 'd6', corpus, tid + '.npz')
    shift = 0
    if not os.path.exists(p):
        p = os.path.join(C.LABELS, 'demucs6', corpus, tid + '.npz')
        shift = 1  # shared cache: centered window, frame k == our causal frame k+1
        if not os.path.exists(p):
            return None
    d = np.load(p)
    e = np.stack([d[s].astype(np.float32) for s in STEMS + ['mix']], 1)
    if shift:
        e = np.concatenate([e[:1], e[:-1]])
    return e


def mix_db_from_feat(f):
    """Mix level (dB) of the causal frame, from the mel power (a monotone proxy of RMS; exact in TS too)."""
    return 10 * np.log10((10 ** f.astype(np.float64)).sum(1) + 1e-10)


class Student(nn.Module):
    def __init__(s, Cc=48, H=48, G=GR.G, nst=6, nmel=FE.N_MEL):
        super().__init__()
        s.inp = nn.Linear(nmel, Cc)
        s.convs = nn.ModuleList([nn.Conv1d(Cc, Cc, 3, dilation=d) for d in (1, 2, 4)])
        s.dil = (1, 2, 4)
        s.gru = nn.GRU(Cc, H, batch_first=True)
        s.env = nn.Linear(Cc + H, nst)
        s.act = nn.Linear(H, G)
        s.register_buffer('mu', torch.zeros(nmel))
        s.register_buffer('sd', torch.ones(nmel))

    def forward(s, x, h=None):
        z = F.relu(s.inp((x - s.mu) / s.sd)).transpose(1, 2)  # [B,C,T]
        for c, d in zip(s.convs, s.dil):
            z = F.relu(z + c(F.pad(z, (2 * d, 0))))
        z = z.transpose(1, 2)
        g, h = s.gru(z, h)
        return s.env(torch.cat([z, g], -1)), s.act(g), h


FEATSET = os.environ.get('STUDENT_FEAT', 'mel')  # 'mel' (own 64-band log-mel, 86 fps) or 'beat' (beat RNN's 162-dim, 100 fps)
BEATFEAT = os.path.join(PRES, 'beatfeat')


def idx_86_to_100(T100):
    """For each 100 fps frame (ending at (j+1)*441), the newest 86 fps frame (ending at (k+1)*512) not after it."""
    return np.maximum(0, ((np.arange(T100) + 1) * 441) // 512 - 1)


def idx_100_to_86(T86):
    return np.maximum(0, ((np.arange(T86) + 1) * 512) // 441 - 1)


def load_beatfeat(corpus, tid):
    p = os.path.join(BEATFEAT, corpus, tid + '.f16')
    if not os.path.exists(p):
        return None
    return np.fromfile(p, '<f2').reshape(-1, 162).astype(np.float32)


def load_set(split, need_env=True, featset=None):
    featset = featset or FEATSET
    items = []
    lists = {'train': ['own-train'], 'fma': ['fma'], 'eval-own': ['own-eval'], 'test': ['test']}[split]
    for ln in lists:
        corpus = 'own' if ln.startswith('own') else ln
        lf = os.path.join(PRES, f'list-{ln}.txt')
        for line in open(lf).read().split('\n') if os.path.exists(lf) else []:
            if not line:
                continue
            tid, path = line.split('=', 1)
            fp = os.path.join(FE.FEAT, corpus, tid + '.npy')
            e = d6_env(corpus, tid)
            if not os.path.exists(fp) or (need_env and e is None):
                continue
            f = np.load(fp).astype(np.float32)
            tp = os.path.join(TGT, corpus, tid + '.npz')
            tz = np.load(tp) if os.path.exists(tp) else None
            a = tz['soft86'].astype(np.float32) if tz is not None else None
            have = tz['have'].astype(np.float32) if tz is not None else np.zeros(GR.G, np.float32)
            T = min(len(f), len(e) if e is not None else 10 ** 9, len(a) if a is not None else 10 ** 9)
            it = dict(corpus=corpus, tid=tid, f=f[:T], e=None if e is None else e[:T], a=None if a is None else a[:T], have=have)
            it['mixdb'] = mix_db_from_feat(it['f'])
            if featset == 'beat':
                fb = load_beatfeat(corpus, tid)
                if fb is None:
                    continue
                T100 = min(len(fb), int(T * 512 / 441))
                ix = idx_86_to_100(T100)
                it.update(f86len=T, f=fb[:T100], mixdb=it['mixdb'][ix], e=None if e is None else it['e'][ix], a=None if a is None else it['a'][ix])
            items.append(it)
    return items


def rel_target(e):
    db = 20 * np.log10(e + 1e-6)
    return np.clip(db[:, :6] - db[:, 6:7], -50, 6)


def train(name, Cc, H, steps, crop=512, bs=48, lr=2e-3, w_act=1.0):
    tr = load_set('train') + load_set('fma')
    print('train tracks', len(tr), 'with activity targets', sum(t['a'] is not None for t in tr), 'hours', round(sum(len(t['f']) for t in tr) / (100 if FEATSET == 'beat' else FE.FPS) / 3600, 2), flush=True)
    allf = np.concatenate([t['f'][::20] for t in tr])
    m = Student(Cc, H, nmel=tr[0]['f'].shape[1])
    m.mu.copy_(torch.tensor(allf.mean(0)))
    m.sd.copy_(torch.tensor(allf.std(0) + 1e-3))
    dev = os.environ.get('STUDENT_DEV', 'mps' if torch.backends.mps.is_available() else 'cpu')
    m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(0)
    lens = np.array([len(t['f']) for t in tr], float)
    pr = lens / lens.sum()
    t0 = time.time()
    for step in range(steps):
        X, Y, A, M = [], [], [], []
        for _ in range(bs):
            t = tr[rng.choice(len(tr), p=pr)]
            if len(t['f']) <= crop:
                continue
            k = rng.integers(0, len(t['f']) - crop)
            g = rng.uniform(-1.2, 1.2)  # +-12 dB input gain; targets are relative to the mix -> unchanged
            X.append(t['f'][k:k + crop] + g)
            Y.append(rel_target(t['e'][k:k + crop]))
            if t['a'] is not None:
                A.append(t['a'][k:k + crop]); M.append(t['have'])
            else:
                A.append(np.zeros((crop, GR.G), np.float32)); M.append(np.zeros(GR.G, np.float32))
        X = torch.tensor(np.stack(X), device=dev); Y = torch.tensor(np.stack(Y), device=dev)
        A = torch.tensor(np.stack(A), device=dev); M = torch.tensor(np.stack(M), device=dev)  # [B, G] per-group mask
        pe, pa, _ = m(X)
        warm = 32  # ignore the first hops of a crop (GRU state warm-up)
        le = F.smooth_l1_loss(pe[:, warm:].contiguous(), Y[:, warm:].contiguous(), beta=2.0)
        la = (F.binary_cross_entropy_with_logits(pa[:, warm:].contiguous(), A[:, warm:].contiguous(), reduction='none').mean(1) * M).sum() / (M.sum() + 1e-6)
        loss = le + w_act * la
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 250 == 0 or step == steps - 1:
            print(f'{step} env {le.item():.3f} act {la.item():.4f} {time.time() - t0:.0f}s', flush=True)
    m.cpu()
    torch.save({'state': m.state_dict(), 'C': Cc, 'H': H, 'ntrain': len(tr), 'nmel': tr[0]['f'].shape[1], 'feat': FEATSET}, os.path.join(MODELS, name + '.pt'))
    return m


def load_model(name):
    ck = torch.load(os.path.join(MODELS, name + '.pt'), weights_only=False)
    m = Student(ck['C'], ck['H'], nmel=ck.get('nmel', FE.N_MEL))
    m.load_state_dict(ck['state'])
    m.featset = ck.get('feat', 'mel')
    return m.eval()


def predict(m, f, mixdb=None):
    """-> (absolute stem dB, activity) at the model's frame rate."""
    with torch.no_grad():
        pe, pa, _ = m(torch.tensor(f[None]))
    rel = pe[0].numpy()
    absdb = rel + (mix_db_from_feat(f) if mixdb is None else mixdb)[:, None]
    return absdb, torch.sigmoid(pa[0]).numpy()


def predict86(m, it):
    """Predictions on the 86 fps grid for a load_set item (the beat-feature model runs at 100 fps)."""
    absdb, act = predict(m, it['f'], it['mixdb'])
    if getattr(m, 'featset', 'mel') == 'beat':
        ix = np.minimum(idx_100_to_86(it['f86len']), len(absdb) - 1)
        absdb, act = absdb[ix], act[ix]
    return absdb, act


def export(name):
    """Weights for the plain-TS forward pass: <name>.json {meta, tensors: {key: [shape, base64 f32 little-endian]}}."""
    import base64
    m = load_model(name)
    ten = {}
    for k, v in m.state_dict().items():
        a = v.detach().numpy().astype('<f4')
        ten[k] = [list(a.shape), base64.b64encode(a.tobytes()).decode()]
    meta = dict(C=m.inp.out_features, H=m.gru.hidden_size, dil=list(m.dil), groups=GR.NAMES, stems=STEMS,
                front=dict(sr=FE.SR, hop=FE.HOP, n_fft=FE.N_FFT, n_mel=FE.N_MEL, fmin=FE.FMIN, fmax=FE.FMAX, log_eps=1e-7),
                params=int(sum(v.numel() for v in m.parameters())))
    import json as _j
    op = os.path.join(MODELS, name + '.json')
    _j.dump(dict(meta=meta, tensors=ten), open(op, 'w'))
    # reference I/O for parity: features + outputs of one test song (first 30 s)
    f = np.load(os.path.join(FE.FEAT, 'test', 'owl-city-fireflies.npy')).astype(np.float32)[:2584]
    with torch.no_grad():
        pe, pa, _ = m(torch.tensor(f[None]))
    f.astype('<f4').tofile(os.path.join(MODELS, name + '.ref_in.f32'))
    pe[0].numpy().astype('<f4').tofile(os.path.join(MODELS, name + '.ref_env.f32'))
    torch.sigmoid(pa[0]).numpy().astype('<f4').tofile(os.path.join(MODELS, name + '.ref_act.f32'))
    print('exported', op, meta['params'], 'params', os.path.getsize(op) / 1e6, 'MB json')


def export_bin(name, out):
    """public/models/stems.bin for src/analysis/stemNet.ts: u32 header length, JSON header, float32 tensors."""
    import struct
    m = load_model(name)
    tens, parts, o = {}, [], 0
    for k, v in m.state_dict().items():
        a = v.detach().numpy().astype('<f4').ravel()
        tens[k] = dict(o=o, n=int(a.size))
        parts.append(a)
        o += a.size
    hdr = dict(version=1, sampleRate=44100, hop=512, frameSize=2048, nMel=FE.N_MEL, fmin=FE.FMIN, fmax=FE.FMAX,
               powScale=0.5, C=m.inp.out_features, H=m.gru.hidden_size, dil=list(m.dil), stems=STEMS, groups=GR.NAMES,
               tensors=tens, model=name)
    hb = json.dumps(hdr, separators=(',', ':')).encode()
    hb += b' ' * ((4 - (4 + len(hb)) % 4) % 4)  # keep the float data 4-byte aligned
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'wb') as f:
        f.write(struct.pack('<I', len(hb)))
        f.write(hb)
        f.write(np.concatenate(parts).tobytes())
    print('wrote', out, os.path.getsize(out), 'bytes')


def stems6_ref_test(slug):
    """6-stem causal reference for a test song from .testdata/instr/stems6 (mono 22050) -> 86 fps."""
    out = []
    for s in STEMS:
        p = os.path.join(C.WORK, 'stems6', f'{slug}.{s}.f32')
        if not os.path.exists(p):
            return None
        out.append(SS.causal_rms(np.fromfile(p, '<f4'), 256, 1024))
    T = min(map(len, out))
    return np.stack([o[:T] for o in out], 1)


def eval_env(name):
    m = load_model(name)
    res = {}
    rows = []
    for split in ['test', 'eval-own']:
        for t in load_set(split, need_env=False, featset=m.featset):
            absdb, act = predict86(m, t)
            T = len(absdb)
            r = {}
            if split == 'test':
                # 4-stem vs the beat agent's htdemucs reference (their table's metric)
                ref4 = {s: SS.beat_ref4(t['tid'], s) for s in ['drums', 'bass', 'vocals', 'other']}
                p4 = {s: absdb[:, STEMS.index(s)] for s in ['drums', 'bass', 'vocals']}
                p4['other'] = 10 * np.log10(sum(10 ** (absdb[:, STEMS.index(s)] / 10) for s in ['guitar', 'piano', 'other']))
                for s, rf in ref4.items():
                    if rf is not None:
                        n = min(T, len(rf))
                        r['4:' + s] = SS.corr_lag(SS.db_ref(rf[:n]), SS.db_cand(p4[s][:n]))
                        r['lat4:' + s] = SS.onset_latency(rf[:n], p4[s][:n])
                ref6 = stems6_ref_test(t['tid'])
            else:
                e = d6_env('own', t['tid'])
                ref6 = None if e is None else e[:, :6]
            if ref6 is not None:
                n = min(T, len(ref6))
                for i, s in enumerate(STEMS):
                    r['6:' + s] = SS.corr_lag(SS.db_ref(ref6[:n, i]), SS.db_cand(absdb[:n, i]))
            if r:
                res[f"{split}/{t['tid']}"] = r
                rows.append((split, t['tid'], r))
                print(split, t['tid'][:28].ljust(28), ' '.join(f'{k}={v[0]:.2f}' for k, v in r.items() if not k.startswith('lat')), flush=True)
    json.dump(res, open(os.path.join(MODELS, name + '.env.json'), 'w'))
    summarize(res, name)


def summarize(res, name):
    print(f'\n== {name}: mean r (median lag ms)')
    for split in ['test', 'eval-own']:
        keys = sorted({k for kk, r in res.items() if kk.startswith(split) for k in r})
        for k in keys:
            v = [r[k] for kk, r in res.items() if kk.startswith(split) and k in r]
            if k.startswith('lat'):
                print(f'{split:9s} {k:10s} n={len(v):3d} onset->half-response latency median {np.nanmedian([a for a, _ in v]) * 1000:.0f} ms ({sum(b for _, b in v)} onsets)')
            else:
                print(f'{split:9s} {k:10s} n={len(v):3d} r={np.mean([a for a, _ in v]):.3f} lag={np.median([b for _, b in v]) * 1000:.0f}ms')


if __name__ == '__main__':
    cmd = sys.argv[1]
    arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    name = arg('--name', 's48')
    if cmd == 'train':
        train(name, arg('--C', 48), arg('--H', 48), arg('--steps', 4000), w_act=arg('--wact', 1.0))
    elif cmd == 'score':
        eval_env(name)
    elif cmd == 'export':
        export(name)
    elif cmd == 'export-bin':
        export_bin(name, arg('--out', os.path.join(C.REPO, 'public', 'models', 'stems.bin')))
