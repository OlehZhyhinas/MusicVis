"""Dump a JSON manifest of {corpus, tid, path, eval, split, label_path} for every track with an
ADTOF label (tom/cymbal split present), so the TS side (DSP baseline) doesn't reimplement
common.py's corpus listing and the train/val/eval split (splits.py).

Writes .testdata/instr/drums/manifest.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import common
from splits import rows

OUT = os.path.join(common.WORK, 'drums', 'manifest.json')


def main():
    rs = rows(('test', 'own', 'fma'), need_cache=False)
    for r in rs:
        r['eval'] = r['split'] == 'eval'
        r['label_path'] = os.path.join(common.LABELS, 'adtof', r['corpus'], f"{r['tid']}.json")
    with open(OUT, 'w') as f:
        json.dump(rs, f, indent=1)
    from collections import Counter
    print(len(rs), dict(Counter((r['corpus'], r['split']) for r in rs)), '->', OUT)


if __name__ == '__main__':
    main()
