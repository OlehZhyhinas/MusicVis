"""Train the causal drum-transcription student on own(train) + FMA(train) cached features with
ADTOF targets built at load time (dataset.build_targets: centre at round(t*fps)+shift, 0.5 on the
+-1 neighbours). Loss: BCE with a per-class pos_weight (sqrt of the neg/pos frame ratio, clipped),
chunks containing tom or cymbal onsets oversampled. Early stopping on validation loss.

Memory: with --pool N only N training tracks are held in RAM at a time (float16), and a quarter of
the pool is swapped for other random tracks every --swap steps (streams the corpus instead of
loading all ~2.9k tracks). Resumable: a checkpoint (model, optimiser, scheduler, rng, best) is
written to models/<tag>.ckpt every 500 steps; --resume continues from it.

Usage: <venv>/bin/python train.py --size small [--tag NAME] [--steps 6000] [--shift 0] [--widen 0.5]
         [--pw sqrt|none|full] [--oversample 3] [--no-fma] [--pool 400] [--swap 250] [--resume]
Writes .testdata/instr/drums/models/<tag>.pt/.json
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn
import common
from model import DrumStudent, SIZES, CLASSES, count_params
from dataset import load_labels, load_feat, build_targets
from splits import rows

torch.set_num_threads(6)
MODELS_DIR = os.path.join(common.WORK, 'drums', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


def load_set(rs, shift, widen):
    data = []
    for r in rs:
        feat = load_feat(r['corpus'], r['tid'])
        if feat is None or len(feat) < 64:
            continue
        lab = build_targets(load_labels(r['corpus'], r['tid']), len(feat), shift, widen)
        data.append((feat.astype(np.float16), lab.astype(np.float16)))
    return data


def make_chunks(data, chunk, oversample):
    idx, w = [], []
    for ti, (feat, lab) in enumerate(data):
        T = len(feat)
        starts = list(range(0, max(1, T - chunk + 1), chunk // 2))
        for s in starts:
            l = lab[s:s + chunk]
            rare = (l[:, 3:] >= 1).any()
            idx.append((ti, s))
            w.append(oversample if rare else 1.0)
    w = np.array(w, dtype=np.float64)
    return idx, w / w.sum()


def batch_of(data, idx, sel, chunk):
    F = np.zeros((len(sel), chunk, data[0][0].shape[1]), dtype=np.float32)
    L = np.zeros((len(sel), chunk, len(CLASSES)), dtype=np.float32)
    M = np.zeros((len(sel), chunk), dtype=np.float32)
    for b, i in enumerate(sel):
        ti, s = idx[i]
        f, l = data[ti][0][s:s + chunk], data[ti][1][s:s + chunk]
        F[b, :len(f)] = f
        L[b, :len(l)] = l
        M[b, :len(f)] = 1
    return torch.from_numpy(F), torch.from_numpy(L), torch.from_numpy(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', default='small')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--chunk', type=int, default=400)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--shift', type=int, default=0)
    ap.add_argument('--widen', type=float, default=0.5)
    ap.add_argument('--pw', default='sqrt')
    ap.add_argument('--oversample', type=float, default=3.0)
    ap.add_argument('--no-fma', action='store_true')
    ap.add_argument('--max-train', type=int, default=0)
    ap.add_argument('--pool', type=int, default=0, help='tracks held in RAM at once (0 = all)')
    ap.add_argument('--swap', type=int, default=250)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    tag = a.tag or a.size
    corpora = ('own',) if a.no_fma else ('own', 'fma')
    rs = rows(corpora)
    tr = [r for r in rs if r['split'] == 'train']
    va = [r for r in rs if r['split'] == 'val']
    if a.max_train:
        tr = tr[:a.max_train]
    t0 = time.time()
    rng = np.random.default_rng(0)
    n_own = sum(r['corpus'] == 'own' for r in tr)
    # class balance / pos_weight from the label files only (cheap, no features needed)
    tot = 0
    pos = np.zeros(len(CLASSES))
    for r in tr:
        lab = load_labels(r['corpus'], r['tid'])
        T = int(max([0.0] + [max(v) for k, v in lab.items() if k in CLASSES and v]) * 86.13) + 1
        T = max(T, 2583 if r['corpus'] == 'fma' else T)
        tot += T
        pos += [len(lab.get(c, [])) for c in CLASSES]
    pos = np.maximum(1, pos)
    dva = load_set(va, a.shift, a.widen)
    pool_n = a.pool if a.pool and a.pool < len(tr) else len(tr)
    pool_ids = list(rng.choice(len(tr), size=pool_n, replace=False))
    dtr = load_set([tr[i] for i in pool_ids], a.shift, a.widen)
    print(f'train {len(tr)} tracks (own {n_own}, fma {len(tr) - n_own}), pool {len(dtr)}, val {len(dva)}; loaded in {time.time()-t0:.0f}s', flush=True)

    ratio = (tot - pos) / pos
    pw = {'sqrt': np.clip(np.sqrt(ratio), 1, 30), 'full': np.clip(ratio, 1, 100), 'none': np.ones(len(CLASSES))}[a.pw]
    print('onsets/class', pos.astype(int).tolist(), 'pos_weight', np.round(pw, 1).tolist(), flush=True)
    pw_t = torch.tensor(pw, dtype=torch.float32, device=DEVICE)

    idx, prob = make_chunks(dtr, a.chunk, a.oversample)
    vidx, _ = make_chunks(dva, a.chunk, 1.0)
    model = DrumStudent(in_feat=160, **SIZES[a.size]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw_t, reduction='none')

    def val_loss():
        model.eval()
        tot_l = tot_m = 0.0
        with torch.no_grad():
            for i in range(0, len(vidx), 64):
                F, L, M = batch_of(dva, vidx, list(range(i, min(len(vidx), i + 64))), a.chunk)
                F, L, M = F.to(DEVICE), L.to(DEVICE), M.to(DEVICE)
                lo, _ = model(F)
                tot_l += (crit(lo, L).mean(-1) * M).sum().item()
                tot_m += M.sum().item()
        model.train()
        return tot_l / max(1, tot_m)

    print(f'[{tag}] params={count_params(model)} chunks={len(idx)} device={DEVICE}', flush=True)
    best = (1e9, None, 0)
    run = 0.0
    start = 1
    ckpt_p = os.path.join(MODELS_DIR, f'{tag}.ckpt')
    if a.resume and os.path.exists(ckpt_p):
        ck = torch.load(ckpt_p, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        best, start, run = (ck['best_val'], ck['best_state'], ck['best_step']), ck['step'] + 1, ck['run']
        rng = np.random.default_rng(ck['step'])
        print(f'resumed from step {ck["step"]} (best val {best[0]:.4f} @ {best[2]})', flush=True)
    model.train()
    for step in range(start, a.steps + 1):
        if pool_n < len(tr) and step % a.swap == 0:
            k = max(1, pool_n // 4)
            out_pos = rng.choice(len(pool_ids), size=k, replace=False)
            rest = np.setdiff1d(np.arange(len(tr)), pool_ids)
            new_ids = rng.choice(rest, size=k, replace=False)
            new_data = load_set([tr[i] for i in new_ids], a.shift, a.widen)
            for j, (pi, ni) in enumerate(zip(out_pos, new_ids)):
                if j < len(new_data):
                    pool_ids[pi] = ni
                    dtr[pi] = new_data[j]
            idx, prob = make_chunks(dtr, a.chunk, a.oversample)
        sel = rng.choice(len(idx), size=a.batch, p=prob)
        F, L, M = batch_of(dtr, idx, sel, a.chunk)
        F, L, M = F.to(DEVICE), L.to(DEVICE), M.to(DEVICE)
        lo, _ = model(F)
        loss = (crit(lo, L).mean(-1) * M).sum() / M.sum()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 500 == 0 or step == a.steps:
            vl = val_loss()
            if vl < best[0]:
                best = (vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, step)
            print(f'  step {step} train {run:.4f} val {vl:.4f} ({time.time()-t0:.0f}s)', flush=True)
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'step': step, 'run': run,
                        'best_val': best[0], 'best_state': best[1], 'best_step': best[2]}, ckpt_p + '.tmp')
            os.replace(ckpt_p + '.tmp', ckpt_p)
    torch.save(best[1], os.path.join(MODELS_DIR, f'{tag}.pt'))
    info = {'tag': tag, 'size': a.size, 'params': count_params(model), 'best_step': best[2], 'best_val_loss': best[0],
            'train_tracks': len(tr), 'pool': pool_n, 'train_own': n_own, 'val_tracks': len(dva), 'args': vars(a), 'pos_weight': pw.tolist(),
            'train_time_s': time.time() - t0}
    json.dump(info, open(os.path.join(MODELS_DIR, f'{tag}.json'), 'w'), indent=1)
    print(f'[{tag}] saved (best step {best[2]}, val {best[0]:.4f})', flush=True)


if __name__ == '__main__':
    main()
