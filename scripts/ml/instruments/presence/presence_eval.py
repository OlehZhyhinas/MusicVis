"""Frame-wise (10 fps) presence scoring of every candidate against the binarised teacher (teacher.py targets).
Candidates (all causal: the value at teacher frame k is the latest output stamped <= (k+1)/10 s):
  yamnet      YAMNet 0.96 s windows every 0.48 s (yamnet_run.py), group score = max class prob
  pannsV1/V2  PANNs MobileNet 2 s windows every 0.5 s (panns_mobile_run.py)
  dsp         RealtimeAnalyzer stemPresence (drums/bass/vocals; 'other' used for guitar/keys/synth*)
  <student>   student activity head (student.py)
Metrics per group: ROC-AUC, F1 at a threshold tuned on the training tracks (own-train tracks with targets and
candidate outputs), base rate. Pooled over the held-out tracks (13 test songs + own eval tracks).
  .testdata/instr/venvs/torch/bin/python presence_eval.py [--student NAME ...] [--split test|eval-own|all]
"""
import os, sys, json
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
sys.path.insert(0, os.path.dirname(__file__))
import groups as GR
import student as ST
import features as FE

PRES = ST.PRES
STEM_MAP_DSP = {'drums': 0, 'bass': 1, 'vocals': 2, 'guitar': 3, 'guitar_electric': 3, 'guitar_acoustic': 3, 'keys_piano': 3,
                'synth': 3, 'synth_lead': 3, 'synth_pad': 3, 'synth_pluck': 3}
TAGMAP = {'synth_lead': 'synth', 'synth_pad': 'synth', 'synth_pluck': 'synth'}


def ids(ln):
    lf = os.path.join(PRES, f'list-{ln}.txt')
    return [l.split('=')[0] for l in open(lf).read().split('\n') if l]


def sample_at(stamps, vals, T):
    """Causal hold: value for frame k = last row with stamp <= (k+1)/10, else 0."""
    k = np.searchsorted(stamps, (np.arange(T) + 1) / 10.0, side='right') - 1
    out = np.where(k[:, None] >= 0, vals[np.maximum(k, 0)], 0)
    return out


def cand_tag(dirname, corpus, tid, T):
    p = os.path.join(PRES, dirname, tid + '.npz')
    if not os.path.exists(p):
        return None
    d = np.load(p)
    P = d['probs'].astype(np.float32)
    idx = GR.class_index(list(d['labels']))
    S = np.zeros((len(P), GR.G), np.float32)
    for gi, g in enumerate(GR.NAMES):
        ii = idx[TAGMAP.get(g, g)]
        if g == 'vocals':
            ii = [i for i in ii if list(d['labels'])[i] != 'Speech'] or ii
        S[:, gi] = P[:, ii].max(1) if ii else 0
    return sample_at(d['t'], S, T)


def cand_dsp(corpus, tid, T):
    p = os.path.join(PRES, 'dsp', tid + '.f32')
    if not os.path.exists(p):
        return None
    D = np.fromfile(p, '<f4').reshape(-1, 12)
    st = (np.arange(len(D)) + 1) * 512 / 44100
    S = np.full((len(D), GR.G), np.nan, np.float32)
    for g, c in STEM_MAP_DSP.items():
        S[:, GR.NAMES.index(g)] = D[:, 4 + c]
    return sample_at(st, S, T)


_models = {}


ENV_GROUPS = {'vocals': 'vocals', 'drums': 'drums', 'bass': 'bass', 'guitar': 'guitar', 'guitar_electric': 'guitar',
              'guitar_acoustic': 'guitar', 'keys_piano': 'piano'}


def cand_student(name, corpus, tid, T):
    """<model> = the activity head; <model>+env = stem groups from the level head instead (causal 0.5 s
    power mean of the stem's level relative to the mix, in dB), the activity head for the rest."""
    use_env = name.endswith('+env')
    name = name[:-4] if use_env else name
    fp = os.path.join(FE.FEAT, corpus, tid + '.npy')
    if not os.path.exists(fp):
        return None
    if name not in _models:
        _models[name] = ST.load_model(name)
    m = _models[name]
    f = np.load(fp).astype(np.float32)
    it = dict(f=f, mixdb=ST.mix_db_from_feat(f))
    if m.featset == 'beat':
        fb = ST.load_beatfeat(corpus, tid)
        if fb is None:
            return None
        T100 = min(len(fb), int(len(f) * 512 / 441))
        it = dict(f=fb[:T100], mixdb=it['mixdb'][ST.idx_86_to_100(T100)], f86len=len(f))
    absdb, act = ST.predict86(m, it)
    if use_env:
        from scipy.ndimage import uniform_filter1d
        mixdb = ST.mix_db_from_feat(f)[:len(absdb)]
        rel = 10 ** ((absdb - mixdb[:, None]) / 10)
        w = int(0.5 * 86)
        relm = uniform_filter1d(rel, w, axis=0, origin=(w - 1) // 2)  # causal box: frames t-w+1..t
        act = act.copy()
        for g, stn in ENV_GROUPS.items():
            act[:, GR.NAMES.index(g)] = 10 * np.log10(relm[:, ST.STEMS.index(stn)] + 1e-9)
    st = (np.arange(len(act)) + 1) * 512 / 44100
    return sample_at(st, act, T)


def collect(cands, tracks):
    """-> {cand: (Y [N,G], S [N,G], H [N,G] have-mask)}"""
    out = {c: ([], [], []) for c in cands}
    for corpus, tid in tracks:
        tp = os.path.join(PRES, 'targets', corpus, tid + '.npz')
        if not os.path.exists(tp):
            continue
        t = np.load(tp)
        Y, have = t['bin10'], t['have'].astype(bool)
        T = len(Y)
        for c in cands:
            if c == 'yamnet':
                S = cand_tag('yamnet', corpus, tid, T)
            elif c.startswith('panns'):
                S = cand_tag(c, corpus, tid, T)
            elif c == 'dsp':
                S = cand_dsp(corpus, tid, T)
            else:
                S = cand_student(c, corpus, tid, T)
            if S is None:
                continue
            out[c][0].append(Y); out[c][1].append(S); out[c][2].append(np.repeat(have[None], T, 0))
    return {c: tuple(np.concatenate(x) for x in v) for c, v in out.items() if v[0]}


def best_thr(y, s):
    qs = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 49)))
    f = [f1_score(y, s >= q, zero_division=0) for q in qs]
    return qs[int(np.argmax(f))] if len(qs) else 0.5


if __name__ == '__main__':
    students = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == '--student']
    cands = ['yamnet', 'pannsV1_w2', 'pannsV2_w2', 'dsp'] + students
    held = [('test', t) for t in ids('test')] + [('own', t) for t in ids('own-eval')]
    train = [('own', t) for t in ids('own-train')]
    H = collect(cands, held)
    Tr = collect(cands, train)
    res = {}
    for c in cands:
        if c not in H:
            continue
        Y, S, M = H[c]
        for gi, g in enumerate(GR.NAMES):
            m = M[:, gi] & ~np.isnan(S[:, gi])
            if m.sum() < 100:
                continue
            y, s = Y[m, gi], S[m, gi]
            if y.min() == y.max():
                continue
            auc = roc_auc_score(y, s)
            thr = None
            if c in Tr:
                Yt, St, Mt = Tr[c]
                mt = Mt[:, gi] & ~np.isnan(St[:, gi])
                if mt.sum() > 100 and Yt[mt, gi].min() != Yt[mt, gi].max():
                    thr = best_thr(Yt[mt, gi], St[mt, gi])
            f1 = f1_score(y, s >= thr, zero_division=0) if thr is not None else float('nan')
            res[f'{c}/{g}'] = dict(auc=auc, f1=f1, base=float(y.mean()), n=int(m.sum()), thr=None if thr is None else float(thr))
    json.dump(res, open(os.path.join(PRES, 'presence_eval.json'), 'w'), indent=1)
    print('group'.ljust(16) + 'base  ' + ''.join(c[:12].rjust(14) for c in cands if c in H) + '   (AUC / F1)')
    for g in GR.NAMES:
        row = [res.get(f'{c}/{g}') for c in cands if c in H]
        base = next((r['base'] for r in row if r), None)
        if base is None:
            continue
        print(g.ljust(16) + f'{base:.2f}  ' + ''.join((f"{r['auc']:.2f}/{r['f1']:.2f}".rjust(14) if r else '-'.rjust(14)) for r in row))
