"""Add separate 'tom' and 'cymbal' keys to the shared ADTOF label files
(~/personal/MusicVis-data/labels/adtof/<corpus>/<id>.json) ALONGSIDE the existing 'other'
(tom+cymbal merged) key, which is left untouched. The split comes from ADTOF's raw per-track
prediction files (time<TAB>midi pitch; 47 = tom, 49 = crash/ride) that the teacher runner keeps in
.testdata/instr/teach/adtof-raw/<id>.wav.txt (own/fma) or the drum agent's _raw dir.

The teacher runner (teach/run_adtof.py) writes only kick/snare/hat/other, so re-run this after it
has labelled more tracks. Safe to re-run: files that already have both keys are skipped, and a
file is only rewritten if its raw kick list matches the JSON's kick list (same inference run).
Atomic writes.

  <venv>/bin/python add_tom_cymbal.py [own,fma]
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import common

TOM, CYMBAL, KICK = 47, 49, 35
RAW_DIRS = [os.path.join(common.WORK, 'teach', 'adtof-raw')]


def raw_dirs(corpus):
    return RAW_DIRS + [os.path.join(common.LABELS, 'adtof', corpus, '_raw')]


def parse_txt(path):
    ev = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                t, p = line.split('\t')
                ev.append((float(t), int(p)))
    return ev


def main():
    corpora = (sys.argv[1] if len(sys.argv) > 1 else 'own,fma').split(',')
    for corpus in corpora:
        d = os.path.join(common.LABELS, 'adtof', corpus)
        if not os.path.isdir(d):
            continue
        n_add = n_have = n_noraw = n_mismatch = 0
        for f in sorted(os.listdir(d)):
            if not f.endswith('.json'):
                continue
            tid = f[:-5]
            p = os.path.join(d, f)
            try:
                lab = json.load(open(p))
            except Exception:
                continue  # being written
            if 'tom' in lab and 'cymbal' in lab:
                n_have += 1
                continue
            raw = None
            for rd in raw_dirs(corpus):
                for cand in (os.path.join(rd, tid + '.wav.txt'), os.path.join(rd, tid + '.txt')):
                    if os.path.exists(cand):
                        raw = cand
                        break
                if raw:
                    break
            if raw is None:
                n_noraw += 1
                continue
            ev = parse_txt(raw)
            kick = sorted(t for t, q in ev if q == KICK)
            if len(kick) != len(lab.get('kick', [])) or any(abs(a - b) > 1e-4 for a, b in zip(kick, lab['kick'])):
                n_mismatch += 1
                continue
            lab['tom'] = sorted(t for t, q in ev if q == TOM)
            lab['cymbal'] = sorted(t for t, q in ev if q == CYMBAL)
            lab['notes'] = lab.get('notes', '') + " 'tom' (47) and 'cymbal' (49) keys added from the raw ADTOF output; 'other' = tom + cymbal."
            tmp = p + '.tmp'
            with open(tmp, 'w') as fh:
                json.dump(lab, fh, indent=2)
            os.replace(tmp, p)
            n_add += 1
        print(f'{corpus}: added={n_add} already={n_have} no_raw={n_noraw} kick_mismatch={n_mismatch}')


if __name__ == '__main__':
    main()
