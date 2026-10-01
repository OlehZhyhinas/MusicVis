"""One-off: apply align.ONSET_BIAS to MIDI-derived times in label files written before the bias correction
existed (those without confidence.onset_bias_ms). Hooktheory parts are not touched.

Run: .testdata/gt/venv/bin/python scripts/ml/groundtruth/retime.py
"""
import glob, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import gt_common as g
from align import ONSET_BIAS as B


def main():
    n = 0
    for f in glob.glob(os.path.join(g.GT, '*', '*.json')):
        d = g.load_json(f)
        c = d.get('confidence')
        if not c or 'onset_bias_ms' in c:
            continue
        r = lambda t: round(t - B, 4)
        for ins in d['instruments']:
            ins['notes'] = [[r(a), r(b), p, v] for a, b, p, v in ins['notes']]
        d['drums'] = {k: [r(t) for t in v] for k, v in d['drums'].items()}
        d['beats'] = [[r(t), p] for t, p in d['beats']]
        d['downbeats'] = [r(t) for t in d['downbeats']]
        d['chords'] = [[r(a), r(b), l] for a, b, l in d['chords']]
        d['regions'] = [[r(a), r(b)] for a, b in d['regions']]
        if c.get('onset_lag_ms'):
            c['onset_lag_ms'] = [round(x - B * 1000, 1) for x in c['onset_lag_ms']]
        c['onset_bias_ms'] = round(B * 1000, 1)
        g.save_json(d, f, separators=(',', ':'))
        n += 1
    print('retimed', n)


if __name__ == '__main__':
    main()
