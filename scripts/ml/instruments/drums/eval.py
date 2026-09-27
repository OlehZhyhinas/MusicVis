"""Evaluate the drum student (or the DSP baseline's single onset stream) on the held-out set
(13 test songs + every 5th owner track) against ADTOF onsets, causal tolerance [t-10 ms, t+50 ms].

Peak picking (peakpick.peak_pick: causal local max + look-ahead + refractory + optional adaptive
delta) is tuned PER CLASS on the VALIDATION split (splits.py; never trained on), maximising micro F1,
then frozen for held-out scoring.

  <venv>/bin/python eval.py <model-tag>        # student, .testdata/instr/drums/models/<tag>.pt
  <venv>/bin/python eval.py --dsp              # DSP stream dumped by dsp_baseline.ts (dsp_onsets/*.f32)

Delay columns: 'delay' = event timestamp (k+1)/fps minus reference time for matched onsets;
'latency' = delay + lookahead/fps, i.e. when a streaming picker can actually emit the event.
Writes .testdata/instr/drums/eval2-<tag>.json
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import common
from model import DrumStudent, SIZES, CLASSES
from peakpick import peak_pick, frames_to_times, match, prf, tune
from dataset import load_labels, load_feat
from features_np import ML_FPS
from splits import rows

torch.set_num_threads(6)
WORK = os.path.join(common.WORK, 'drums')
MODELS_DIR = os.path.join(WORK, 'models')
DSP_DIR = os.path.join(WORK, 'dsp_onsets')

THRESHOLDS = [round(x, 2) for x in np.arange(0.10, 0.96, 0.05)]
REFRACTORIES = [1, 2, 3, 5, 8]
LOOKAHEADS = (0, 1, 2)
DELTAS = (0.0, 0.1)
MAX_VAL = 150


def get_acts_fn(tag):
    if tag == 'dsp':
        def f(r):
            p = os.path.join(DSP_DIR, f"{r['corpus']}__{r['tid']}.f32")
            if not os.path.exists(p):
                return None
            x = np.fromfile(p, dtype='<f4')
            x = x / (np.percentile(x, 99.5) + 1e-9)  # per-song scale so one threshold grid fits all (uses the whole song: slightly optimistic for DSP)
            return np.repeat(np.clip(x, 0, 1.5)[:, None], len(CLASSES), axis=1)
        return f, {'params': 0, 'size': 'dsp'}
    info = json.load(open(os.path.join(MODELS_DIR, f'{tag}.json')))
    model = DrumStudent(in_feat=160, **SIZES[info['size']])
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f'{tag}.pt'), map_location='cpu'))
    model.eval()

    @torch.no_grad()
    def f(r):
        feat = load_feat(r['corpus'], r['tid'])
        if feat is None:
            return None
        lo, _ = model(torch.from_numpy(feat.astype(np.float32)).unsqueeze(0))
        return torch.sigmoid(lo)[0].numpy()
    return f, info


def main():
    tag = 'dsp' if '--dsp' in sys.argv else sys.argv[1]
    acts_of, info = get_acts_fn(tag)
    rs = rows(('test', 'own', 'fma'))
    val = [r for r in rs if r['split'] == 'val']
    # balance validation between own (full songs) and fma (30 s clips)
    vo = [r for r in val if r['corpus'] == 'own']
    vf = [r for r in val if r['corpus'] == 'fma']
    val = vo[:MAX_VAL // 2] + vf[:MAX_VAL - min(len(vo), MAX_VAL // 2)]
    ev = [r for r in rs if r['split'] == 'eval']

    vacts, vlabs = [], []
    for r in val:
        a = acts_of(r)
        if a is not None:
            vacts.append(a)
            vlabs.append(load_labels(r['corpus'], r['tid']))
    print(f'{tag}: tuning on {len(vacts)} val tracks, scoring {len(ev)} held-out', flush=True)
    params = {}
    for ci, cls in enumerate(CLASSES):
        params[cls] = tune([a[:, ci] for a in vacts], [l.get(cls, []) for l in vlabs], ML_FPS,
                           THRESHOLDS, REFRACTORIES, LOOKAHEADS, (3,), DELTAS)
        print(cls, params[cls], flush=True)

    groups = {'all': lambda r: True, 'test': lambda r: r['corpus'] == 'test', 'own': lambda r: r['corpus'] == 'own'}
    agg = {g: {c: {'tp': 0, 'np': 0, 'nr': 0, 'tp50': 0, 'delays': []} for c in CLASSES} for g in groups}
    per_song = {}
    n_scored = {g: 0 for g in groups}
    for r in ev:
        a = acts_of(r)
        if a is None:
            continue
        lab = load_labels(r['corpus'], r['tid'])
        song = {}
        for ci, cls in enumerate(CLASSES):
            p = params[cls]
            pt = frames_to_times(peak_pick(a[:, ci], p['threshold'], p['refractory'], p['lookahead'], p['past'], p['delta']), ML_FPS)
            ref = lab.get(cls, [])
            tp, delays = match(pt, ref)
            tp50, _ = match(pt, ref, (-0.05, 0.05))
            song[cls] = {'tp': tp, 'n_pred': len(pt), 'n_ref': len(ref)}
            for g, fn in groups.items():
                if fn(r):
                    s = agg[g][cls]
                    s['tp'] += tp; s['np'] += len(pt); s['nr'] += len(ref); s['tp50'] += tp50; s['delays'] += delays
        for g, fn in groups.items():
            n_scored[g] += fn(r)
        per_song[f"{r['corpus']}/{r['tid']}"] = song

    summary = {'tag': tag, 'params_count': info.get('params'), 'n_songs': n_scored, 'peak_params': params}
    for g in groups:
        summary[g] = {}
        for cls in CLASSES:
            s = agg[g][cls]
            p, rr, f1 = prf(s['tp'], s['np'], s['nr'])
            f50 = prf(s['tp50'], s['np'], s['nr'])[2]
            d = np.array(sorted(s['delays'])) * 1000
            la_ms = params[cls]['lookahead'] / ML_FPS * 1000
            summary[g][cls] = {'p': p, 'r': rr, 'f1': f1, 'f1_sym50': f50, 'n_ref': s['nr'], 'n_pred': s['np'],
                               'median_delay_ms': float(np.median(d)) if len(d) else None,
                               'p90_delay_ms': float(np.percentile(d, 90)) if len(d) else None,
                               'median_latency_ms': float(np.median(d) + la_ms) if len(d) else None,
                               'p90_latency_ms': float(np.percentile(d, 90) + la_ms) if len(d) else None}
    out = os.path.join(WORK, f'eval2-{tag}.json')
    json.dump({'summary': summary, 'per_song': per_song}, open(out, 'w'), indent=1)
    for g in groups:
        print(f'--- {g} ({n_scored[g]} songs)')
        for cls in CLASSES:
            s = summary[g][cls]
            print(f"  {cls:7s} P {s['p']:.3f} R {s['r']:.3f} F1 {s['f1']:.3f} (sym50 {s['f1_sym50']:.3f}) n_ref {s['n_ref']:6d} n_pred {s['n_pred']:6d} "
                  f"delay med {s['median_delay_ms'] or float('nan'):.1f} p90 {s['p90_delay_ms'] or float('nan'):.1f} latency med {s['median_latency_ms'] or float('nan'):.1f}")
    print('wrote', out)


if __name__ == '__main__':
    main()
