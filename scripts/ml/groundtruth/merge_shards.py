"""Fold <name>.shard<i>.json partial results (GT_SHARD runs) back into the main work files and delete them.

Run: .testdata/gt/venv/bin/python scripts/ml/groundtruth/merge_shards.py
"""
import glob, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import gt_common as g


def main():
    for name in ('screen', 'screen_norm', 'hooktheory2', 'summary'):
        base = os.path.join(g.WORK, name + '.json')
        parts = sorted(glob.glob(os.path.join(g.WORK, name + '.shard*.json')))
        if not parts:
            continue
        out = g.load_json(base) if os.path.exists(base) else ({'midi': {}, 'audio': {}} if name == 'screen_norm' else {})
        for p in parts:
            d = g.load_json(p)
            if name == 'screen_norm':
                for k in ('midi', 'audio'):
                    out[k].update(d.get(k, {}))
            else:
                out.update(d)
        g.save_json(out, base, indent=0 if name == 'screen' else None)
        for p in parts:
            os.remove(p)
        print(name, 'merged', len(parts), 'shards')


if __name__ == '__main__':
    main()
