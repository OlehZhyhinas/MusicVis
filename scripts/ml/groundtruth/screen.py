"""Stage 1: global screening of every (track, candidate MIDI) pair plus null pairs (track vs unrelated MIDIs).

Score = coarse_ratio: min subsequence-DTW cost of the whole (transposition-corrected) MIDI against the
recording, per MIDI frame, divided by the median frame distance. ~0.9-1.0 for unrelated pairs.
Writes WORK/screen.json. Caches raw CQTs (WORK/cqt/...) that the full alignment reuses.

Run (one process, under the machine-wide limiter):
  slot.sh env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib .testdata/gt/venv/bin/python scripts/ml/groundtruth/screen.py
"""
import json, os, random, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import gt_common as g
import align

NULLS = 3


def main():
    d = json.load(open(os.path.join(g.WORK, 'candidates.json')))
    outp = os.path.join(g.WORK, 'screen.json')
    out = json.load(open(outp)) if os.path.exists(outp) else {}
    pool = sorted({c['md5'] for v in d.values() for c in v['cands'] if c['why'] == 'artist+title'})
    rnd = random.Random(0)
    t0 = time.time()
    for n, (tid, v) in enumerate(d.items()):
        mine = {c['md5'] for c in v['cands']}
        nulls = [m for m in rnd.sample(pool, min(len(pool), NULLS + 5)) if m not in mine][:NULLS]
        prev = out.get(tid, [])
        have = {r['md5'] for r in prev}
        if not v['cands'] or (tid in out and mine <= have):
            continue
        if tid in out:
            nulls = []                                  # re-screen only newly added candidates
        try:
            Ra = align.audio_raw(v['corpus'], tid, v['path'])
        except Exception as e:
            print('audio fail', tid, e); continue
        res = list(prev)
        for md5, kind in [(c['md5'], c['why']) for c in v['cands'] if c['md5'] not in have] + [(m, 'null') for m in nulls]:
            try:
                sc, sh = align.best_shift(align.midi_raw(os.path.join(g.WORK, 'midi', md5 + '.mid')), Ra)
                res.append(dict(md5=md5, kind=kind, ratio=round(sc, 4), shift=sh))
            except Exception as e:
                res.append(dict(md5=md5, kind=kind, error=str(e)[:80]))
        out[tid] = res
        json.dump(out, open(outp, 'w'), indent=0)
        best = min([r for r in res if r['kind'] != 'null' and 'ratio' in r], key=lambda r: r['ratio'], default=None)
        print(f'[{n}] {time.time()-t0:.0f}s {tid[:50]} best={best and best["ratio"]} '
              f'nulls={[r.get("ratio") for r in res if r["kind"] == "null"]}', flush=True)


if __name__ == '__main__':
    main()
