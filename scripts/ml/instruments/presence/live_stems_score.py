"""Score live_stems_dump.ts outputs (the app's stems.* 0..1 envelopes) with the stems-eval metric:
the envelope EMA-smoothed like stems-eval.ts does for the DSP, Pearson r at the best lag vs the
htdemucs 4-stem dB reference, plus the onset-to-half-response latency.
  .testdata/instr/venvs/torch/bin/python live_stems_score.py <tag> [<tag> ...]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import stemscore as SS
import student as ST

S4 = ['drums', 'bass', 'vocals', 'other']
for tag in sys.argv[1:]:
    d = os.path.join(ST.PRES, 'live', tag)
    acc = {}
    for f in sorted(os.listdir(d)):
        slug = f[:-4]
        X = np.fromfile(os.path.join(d, f), '<f4').reshape(-1, 4)
        for i, s in enumerate(S4):
            rf = SS.beat_ref4(slug, s)
            if rf is None:
                continue
            n = min(len(rf), len(X))
            acc.setdefault(s, []).append(SS.corr_lag(SS.db_ref(rf[:n]), SS.ema(X[:n, i])))
            acc.setdefault('lat.' + s, []).append(SS.onset_latency(rf[:n], 30 * X[:n, i]))
    print(tag + ': ' + ' | '.join(f"{s} r={np.mean([a for a, _ in acc[s]]):.3f} lag={np.median([b for _, b in acc[s]]) * 1000:.0f}ms "
                                 f"resp={np.nanmedian([a for a, _ in acc['lat.' + s]]) * 1000:.0f}ms" for s in S4 if s in acc))
