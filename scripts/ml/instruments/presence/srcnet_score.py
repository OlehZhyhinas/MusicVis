"""stems-eval r (raw dB, 4-stem vs the beat agent's htdemucs reference) of the shipped TS network (srcnet_dump.ts)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import stemscore as SS
import student as ST

d = os.path.join(ST.PRES, 'srcnet')
acc = {}
for f in sorted(os.listdir(d)):
    slug = f[:-4]
    X = np.fromfile(os.path.join(d, f), '<f4').reshape(-1, 6)
    p4 = {s: X[:, ST.STEMS.index(s)] for s in ['drums', 'bass', 'vocals']}
    p4['other'] = 10 * np.log10(sum(10 ** (X[:, ST.STEMS.index(s)] / 10) for s in ['guitar', 'piano', 'other']))
    for s, y in p4.items():
        rf = SS.beat_ref4(slug, s)
        n = min(len(rf), len(y))
        acc.setdefault(s, []).append(SS.corr_lag(SS.db_ref(rf[:n]), SS.db_cand(y[:n])))
        acc.setdefault('lat.' + s, []).append(SS.onset_latency(rf[:n], y[:n]))
print(' | '.join(f"{s} r={np.mean([a for a, _ in acc[s]]):.3f} lag={np.median([b for _, b in acc[s]]) * 1000:.0f}ms resp={np.nanmedian([a for a, _ in acc['lat.' + s]]) * 1000:.0f}ms"
                 for s in ['drums', 'bass', 'vocals', 'other']))
