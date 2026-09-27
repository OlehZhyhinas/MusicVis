"""Envelope baselines with the stems-eval metric (stemscore.py): the live DSP stems (dsp_dump.ts output,
the 0..1 env smoothed like stems-eval does, and 10*log10 of the private weighted power `rw`) and the
beat agent's HS-TasNet streaming envelopes, against the 4-stem htdemucs reference on the test songs,
and the DSP against the demucs6 teacher (4-stem sum) on the own eval tracks.
  .testdata/instr/venvs/torch/bin/python env_baselines.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import stemscore as SS
import student as ST

PRES = ST.PRES
S4 = ['drums', 'bass', 'vocals', 'other']
HS = '/Users/oleh/personal/MusicVis/.claude/worktrees/agent-adccb4c576b7c1097/.testdata/ml/stems/hstasnet'
acc = {}
for split, lf in [('test', 'list-test.txt'), ('eval-own', 'list-own-eval.txt')]:
    for line in open(os.path.join(PRES, lf)).read().split('\n'):
        if not line:
            continue
        tid = line.split('=')[0]
        dp = os.path.join(PRES, 'dsp', tid + '.f32')
        if not os.path.exists(dp):
            continue
        D = np.fromfile(dp, '<f4').reshape(-1, 12)
        if split == 'test':
            refs = {s: SS.beat_ref4(tid, s) for s in S4}
        else:
            e = ST.d6_env('own', tid)
            if e is None:
                continue
            refs = {'drums': e[:, 0], 'bass': e[:, 1], 'vocals': e[:, 2], 'other': np.sqrt(e[:, 3] ** 2 + e[:, 4] ** 2 + e[:, 5] ** 2)}
        for i, s in enumerate(S4):
            rf = refs[s]
            if rf is None:
                continue
            n = min(len(rf), len(D))
            ref = SS.db_ref(rf[:n])
            for name, y, raw in [('dsp.env', SS.ema(D[:n, i]), 30 * D[:n, i]), ('dsp.rwdb', SS.db_cand(10 * np.log10(D[:n, 8 + i] + 1e-12)), 10 * np.log10(D[:n, 8 + i] + 1e-12))]:
                acc.setdefault((split, name, s), []).append(SS.corr_lag(ref, y))
                if split == 'test':
                    acc.setdefault((split, 'lat.' + name, s), []).append(SS.onset_latency(rf[:n], raw))
            hp = os.path.join(HS, f'{tid}.{s}.f32')
            if split == 'test' and os.path.exists(hp):
                h = np.fromfile(hp, '<f4')
                m = min(n, len(h))
                acc.setdefault((split, 'hstasnet', s), []).append(SS.corr_lag(ref[:m], SS.db_ref(h[:m])))
                acc.setdefault((split, 'lat.hstasnet', s), []).append(SS.onset_latency(rf[:m], 20 * np.log10(h[:m] + 1e-9)))
out = {}
for (split, name, s), v in sorted(acc.items()):
    if name.startswith('lat.'):
        r, lag = np.nanmedian([a for a, _ in v]) * 1000, sum(b for _, b in v)
        print(f'{split:9s} {name:14s} {s:7s} onset->half-response latency median {r:.0f} ms ({lag} onsets)')
        out[f'{split}/{name}/{s}'] = (r, lag)
        continue
    r, lag = np.mean([a for a, _ in v]), np.median([b for _, b in v]) * 1000
    out[f'{split}/{name}/{s}'] = (r, lag, len(v))
    print(f'{split:9s} {name:10s} {s:7s} n={len(v):3d} r={r:.3f} lag={lag:.0f}ms')
json.dump(out, open(os.path.join(PRES, 'env_baselines.json'), 'w'), indent=1)
