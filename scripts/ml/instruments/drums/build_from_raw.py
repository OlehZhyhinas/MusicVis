"""Rebuild labels/adtof/<corpus>/<id>.json (5-way kick/snare/hat/tom/cymbal split) from ADTOF's
raw per-track prediction .txt files (time\\tpitch), without re-running inference. Atomic writes.

  <venv>/bin/python build_from_raw.py <raw_dir> <out_dir> [--slugs a,b,c]
"""
import glob
import json
import os
import sys

KICK, SNARE, HAT, TOM, CYMBAL = 35, 38, 42, 47, 49


def parse_txt(path):
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t, pitch = line.split('\t')
            events.append((float(t), int(pitch)))
    return events


def atomic_write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    raw_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    txts = sorted(glob.glob(os.path.join(raw_dir, '*.txt')))
    # group by slug (strip trailing ".wav.txt" / ".txt")
    by_slug = {}
    for p in txts:
        base = os.path.basename(p)
        slug = base.replace('.wav.txt', '').replace('.txt', '')
        by_slug.setdefault(slug, []).append(p)

    done = 0
    for slug, paths in sorted(by_slug.items()):
        txt_path = sorted(paths, key=os.path.getmtime)[-1]
        events = parse_txt(txt_path)
        kick = sorted(t for t, p in events if p == KICK)
        snare = sorted(t for t, p in events if p == SNARE)
        hat = sorted(t for t, p in events if p == HAT)
        tom = sorted(t for t, p in events if p == TOM)
        cymbal = sorted(t for t, p in events if p == CYMBAL)
        other = sorted(t for t, p in events if p not in (KICK, SNARE, HAT, TOM, CYMBAL))
        result = {
            'model': 'adtof-Frame_RNN-adtofAll-fold0',
            'slug': slug,
            'beats': [],
            'downbeats': [],
            'kick': kick,
            'snare': snare,
            'hat': hat,
            'tom': tom,
            'cymbal': cymbal,
            'other': other,
            'notes': (
                "ADTOF drum transcription (bundled pretrained Frame_RNN CRNN model, "
                "scenario='adtofAll', fold=0). Classes [35,38,47,42,49]; "
                "35->kick, 38->snare, 42->hat, 47->tom, 49->cymbal(crash/ride)."
            ),
            'causal': False,
        }
        atomic_write_json(os.path.join(out_dir, f'{slug}.json'), result)
        done += 1
    print(f'wrote {done} label files to {out_dir}')


if __name__ == '__main__':
    main()
