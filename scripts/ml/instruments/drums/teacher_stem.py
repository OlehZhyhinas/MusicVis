"""Teacher-reliability ceiling: run ADTOF (same Frame_RNN adtofAll fold0) on the demucs drums stem
of each test song (.testdata/instr/stems6/<slug>.drums.f32, mono 22050 Hz, upsampled to 44.1 kHz)
and score it against the ADTOF full-mix labels with the same causal tolerance used for the
student. Writes .testdata/instr/drums/teacher_stem/<slug>.json and teacher_agreement.json.

  .testdata/instr/venvs/adtof/bin/python scripts/ml/instruments/drums/teacher_stem.py
"""
import glob
import json
import os
import sys

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import common
from peakpick import score_prf

STEMS = os.path.join(common.WORK, 'stems6')
OUT = os.path.join(common.WORK, 'drums', 'teacher_stem')
PITCH = {'kick': 35, 'snare': 38, 'hat': 42, 'tom': 47, 'cymbal': 49}
CLASSES = list(PITCH)


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = os.path.join(OUT, '_raw')
    os.makedirs(raw, exist_ok=True)
    model = None
    agg = {c: {'tp': 0, 'np': 0, 'nr': 0} for c in CLASSES}
    for tid, _ in common.tracks('test'):
        stem = os.path.join(STEMS, f'{tid}.drums.f32')
        mix_p = common.label_path('adtof', 'test', tid, 'json')
        if not os.path.exists(stem) or not os.path.exists(mix_p):
            print('skip', tid)
            continue
        outp = os.path.join(OUT, f'{tid}.json')
        if not os.path.exists(outp):
            if model is None:
                from adtof.model.model import Model
                model, hp = Model.modelFactory(modelName='Frame_RNN', scenario='adtofAll', fold=0)
                assert model.weightLoadedFlag
            x = np.fromfile(stem, dtype='<f4')
            y = resample_poly(x, 2, 1).astype(np.float32)
            wav = os.path.join(raw, f'{tid}.wav')
            sf.write(wav, y, 44100, subtype='FLOAT')
            model.predictFolder(wav, raw, **{**hp, 'writeMidi': False})
            os.remove(wav)
            txt = sorted(glob.glob(os.path.join(raw, f'{tid}*.txt')), key=os.path.getmtime)[-1]
            ev = [(float(a), int(b)) for a, b in (l.strip().split('\t') for l in open(txt) if l.strip())]
            res = {c: sorted(t for t, q in ev if q == p) for c, p in PITCH.items()}
            json.dump(res, open(outp, 'w'))
        stem_lab = json.load(open(outp))
        mix_lab = json.load(open(mix_p))
        for c in CLASSES:
            # stem transcription treated as the "prediction", mix transcription as the reference, symmetric +-50ms
            _, _, _, tp, _ = score_prf(stem_lab[c], mix_lab.get(c, []), tol=(-0.05, 0.05))
            agg[c]['tp'] += tp
            agg[c]['np'] += len(stem_lab[c])
            agg[c]['nr'] += len(mix_lab.get(c, []))
    summ = {}
    for c, a in agg.items():
        p = a['tp'] / a['np'] if a['np'] else 0.0
        r = a['tp'] / a['nr'] if a['nr'] else 0.0
        summ[c] = {'p_stem_vs_mix': p, 'r': r, 'f1': 2 * p * r / (p + r) if p + r else 0.0, 'n_mix': a['nr'], 'n_stem': a['np']}
    json.dump(summ, open(os.path.join(common.WORK, 'drums', 'teacher_agreement.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
