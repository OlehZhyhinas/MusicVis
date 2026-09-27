"""Per-group teacher targets (offline, non-causal allowed) at 10 fps, combining:
  - htdemucs_6s stem RMS relative to the mix (student.d6_env: our causal d6 cache or the shared labels/demucs6)
  - PANNs Cnn14_DecisionLevelMax framewise probabilities (10 fps) on the MIX (shared labels/panns or our
    panns_stem/<corpus>/<id>.mix.npz) and on the separated OTHER / VOCALS / GUITAR / PIANO stems
    (panns_stem/<corpus>/<id>.<stem>.npz), where present.
Rules (see groups.py for the class sets):
  vocals      vocals stem active, unless the vocals-stem tagger says it is a lead INSTRUMENT (sax, brass, flute,
              synth: demucs routes monophonic leads into 'vocals')
  drums, bass stem active
  guitar, keys_piano   stem active AND tag evidence (mix or that stem) within +-5 s (the 6s guitar/piano stems bleed)
  guitar_electric / _acoustic   guitar split by which tag family is stronger within +-5 s
  tag groups  max over sources of the smoothed group score > TAU[group]
  synth       'other' stem active AND (synth tag OR electronic-genre tag within +-5 s) AND no brass/woodwind/organ
  synth_pad / _pluck / _lead   synth split by the 'other' stem envelope: pad = sustained (low dB modulation, few
              onsets), pluck = many onsets with deep modulation, lead = the rest, or an instrumental lead in 'vocals'
Writes .testdata/instr/presence/targets/<corpus>/<id>.npz {bin10 [T,G] u8, soft86 [T86,G] f16, have [G] u8, names}
  .testdata/instr/venvs/torch/bin/python teacher.py <corpus> <listfile>   (or: teacher.py check  -> test-song sanity table)
"""
import os, sys
import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import common as C
import groups as GR
import student as ST

FPS86 = 44100 / 512
TAU = {'vocals_backing': 0.04, 'perc_hand': 0.05, 'organ': 0.04, 'brass': 0.04, 'strings': 0.06, 'woodwinds': 0.04,
       'mallets_bells': 0.04, 'fx': 0.04}
EV = 0.05          # guitar / piano tag evidence (window max)
STEM_THR = {'vocals': -15, 'drums': -15, 'bass': -15, 'guitar': -15, 'piano': -15, 'other': -15}
OUT = os.path.join(ST.PRES, 'targets')


def tags(corpus, tid):
    out = {}
    sp = os.path.join(C.LABELS, 'panns', corpus, tid + '.npz')
    mp = os.path.join(ST.PRES, 'panns_stem', corpus, tid + '.mix.npz')
    for p in [sp, mp]:
        if os.path.exists(p):
            d = np.load(p)
            out['mix'] = (d['probs'].astype(np.float32), list(d['labels']))
            break
    for st in ['other', 'vocals', 'guitar', 'piano']:
        p = os.path.join(ST.PRES, 'panns_stem', corpus, f'{tid}.{st}.npz')
        if os.path.exists(p):
            d = np.load(p)
            out[st] = (d['probs'].astype(np.float32), list(d['labels']))
    return out


def gscore(tg, src, group_names, T):
    if src not in tg:
        return None
    P, L = tg[src]
    idx = GR.class_index(L)
    ii = sorted({i for g in group_names for i in idx[g]})
    s = P[:, ii].max(1) if ii else np.zeros(len(P))
    s = s[:T] if len(s) >= T else np.pad(s, (0, T - len(s)))
    return uniform_filter1d(maximum_filter1d(s, 10), 10)  # 1 s max then 1 s mean (centered)


def genre_score(tg, T):
    if 'mix' not in tg:
        return None
    P, L = tg['mix']
    ii = [L.index(c) for c in GR.ELECTRONIC if c in L]
    s = P[:, ii].max(1)
    s = s[:T] if len(s) >= T else np.pad(s, (0, T - len(s)))
    return maximum_filter1d(uniform_filter1d(s, 20), 100)


def build(corpus, tid):
    e = ST.d6_env(corpus, tid)
    if e is None:
        return None
    tg = tags(corpus, tid)
    T86 = len(e)
    T = int(T86 / FPS86 * 10)
    fr = np.minimum((np.arange(T) / 10 * FPS86).astype(int), T86 - 1)
    db = 20 * np.log10(e + 1e-6)
    top = np.sort(db[:, 6])[int(0.99 * (T86 - 1))]
    names = GR.STEMS6
    act = {}
    rel86 = {}
    for i, s in enumerate(names):
        ps = uniform_filter1d(e[:, i].astype(np.float64) ** 2, int(0.5 * FPS86))  # 0.5 s power means
        pm = uniform_filter1d(e[:, 6].astype(np.float64) ** 2, int(0.5 * FPS86))
        a = 10 * np.log10(ps + 1e-12)
        r = a - 10 * np.log10(pm + 1e-12)
        rel86[s] = r
        act[s] = ((r > STEM_THR[s]) & (a > top - 40))[fr]
    W = lambda x: maximum_filter1d(x, 100)  # +-5 s window max
    B = {n: np.zeros(T, bool) for n in GR.NAMES}
    have = {n: False for n in GR.NAMES}
    mx = lambda *xs: (np.max(np.stack([x for x in xs if x is not None]), 0) if any(x is not None for x in xs) else None)
    # vocals vs instrumental lead in the vocals stem
    v_voice = gscore(tg, 'vocals', ['vocals'], T)
    v_inst = gscore(tg, 'vocals', ['brass', 'woodwinds', 'synth', 'strings'], T)
    inst_lead = (v_inst > 0.05) & (v_inst > 2 * v_voice) if v_voice is not None else np.zeros(T, bool)
    B['vocals'] = act['vocals'] & ~inst_lead; have['vocals'] = True
    B['drums'] = act['drums']; have['drums'] = True
    B['bass'] = act['bass']; have['bass'] = True
    for g, st in [('guitar', 'guitar'), ('keys_piano', 'piano')]:
        ev = mx(gscore(tg, 'mix', [g], T), gscore(tg, st, [g], T))
        if ev is not None:
            B[g] = act[st] & (W(ev) > EV); have[g] = True
    ge = mx(gscore(tg, 'mix', ['guitar_electric'], T), gscore(tg, 'guitar', ['guitar_electric'], T))
    ga = mx(gscore(tg, 'mix', ['guitar_acoustic'], T), gscore(tg, 'guitar', ['guitar_acoustic'], T))
    if ge is not None:
        we, wa = W(ge), W(ga)
        B['guitar_electric'] = B['guitar'] & (we >= wa) & (we > EV / 2)
        B['guitar_acoustic'] = B['guitar'] & (wa > we) & (wa > EV / 2)
        have['guitar_electric'] = have['guitar_acoustic'] = True
    for g, tau in TAU.items():
        srcs = ['mix', 'other'] + (['vocals'] if g in ('brass', 'woodwinds', 'vocals_backing') else [])
        s = mx(*[gscore(tg, src, [g], T) for src in srcs])
        if s is not None:
            B[g] = s > tau; have[g] = True
    sy = mx(gscore(tg, 'mix', ['synth'], T), gscore(tg, 'other', ['synth'], T))
    gen = genre_score(tg, T)
    if sy is not None and gen is not None:
        B['synth'] = act['other'] & ((W(sy) > 0.03) | (gen > 0.2)) & ~(B['brass'] | B['woodwinds'] | B['organ'])
        have['synth'] = True
        # envelope statistics of the 'other' stem over a centered 1 s window
        od = np.maximum(db[:, 5], top - 60)
        mod = np.sqrt(uniform_filter1d((od - uniform_filter1d(od, int(0.25 * FPS86))) ** 2, int(FPS86)))
        rise = np.diff(od, prepend=od[0])
        on = (rise > 1.5) & (np.roll(rise, 1) <= 1.5)
        rate = uniform_filter1d(on.astype(float), int(FPS86)) * FPS86
        mod, rate = mod[fr], rate[fr]
        pad = B['synth'] & (mod < 1.5) & (rate < 3)
        pluck = B['synth'] & (mod >= 2.5) & (rate >= 4)
        B['synth_pad'], B['synth_pluck'] = pad, pluck
        B['synth_lead'] = (B['synth'] & ~pad & ~pluck) | (inst_lead & act['vocals'] & (gen > 0.2))
        have['synth_pad'] = have['synth_pluck'] = have['synth_lead'] = True
    bin10 = np.stack([B[n] for n in GR.NAMES], 1).astype(np.uint8)
    t86 = np.minimum((np.arange(T86) / FPS86 * 10).astype(int), T - 1)
    soft86 = uniform_filter1d(bin10.astype(np.float32), 5, axis=0)[t86]
    return dict(bin10=bin10, soft86=soft86.astype(np.float16), have=np.array([have[n] for n in GR.NAMES], np.uint8),
                names=np.array(GR.NAMES), srcs=np.array(sorted(tg)))


def run(corpus, lst):
    n = 0
    for line in open(lst).read().split('\n'):
        if not line:
            continue
        tid = line.split('=')[0]
        r = build(corpus, tid)
        if r is None:
            continue
        os.makedirs(os.path.join(OUT, corpus), exist_ok=True)
        np.savez_compressed(os.path.join(OUT, corpus, tid + '.npz'), **r)
        n += 1
    print(corpus, n, 'targets')


def check():
    print('fraction of frames active per group (test songs)')
    print('song'.ljust(22) + ' '.join(n[:6].rjust(6) for n in GR.NAMES))
    for line in open(os.path.join(ST.PRES, 'list-test.txt')).read().split('\n'):
        if not line:
            continue
        tid = line.split('=')[0]
        r = build('test', tid)
        if r is None:
            continue
        fr = r['bin10'].mean(0)
        print(tid[:22].ljust(22) + ' '.join((f'{f:6.2f}' if h else '     -') for f, h in zip(fr, r['have'])), ' src:', ','.join(r['srcs']))


if __name__ == '__main__':
    if sys.argv[1] == 'check':
        check()
    else:
        run(sys.argv[1], sys.argv[2])
