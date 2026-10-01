"""Stage 1b: two-way null normalisation of the screening ratios.

A raw screening ratio depends on the MIDI (sparse or short MIDIs fit anything) and on the recording
(repetitive EDM fits many MIDIs). So every candidate MIDI is also screened against K_A unrelated
recordings (-> mu_M, sd_M) and every recording against K_M unrelated MIDIs (-> mu_A, sd_A).
  z = (ratio - mu_M) / sd_M  and  zA = (ratio - mu_A) / sd_A;  score = max(z, zA)  (both must be low)
Writes WORK/screen_norm.json {"midi": {md5: [ratios]}, "audio": {id: [ratios]}}; build_labels reads it.

Run under the limiter (after screen.py): .testdata/gt/venv/bin/python scripts/ml/groundtruth/screen_norm.py
"""
import os, random, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import gt_common as g
import align

K_A = 8
K_M = 10


def main():
    cands = g.load_json(os.path.join(g.WORK, 'candidates.json'))
    screen = g.load_json(os.path.join(g.WORK, 'screen.json'))
    outp = os.path.join(g.WORK, 'screen_norm.json')
    out = g.load_sharded(outp, {'midi': {}, 'audio': {}})
    part = {'midi': {}, 'audio': {}}
    save = lambda: g.save_json(part if g.shard()[1] > 1 else out, g.shard_path(outp))
    rnd = random.Random(7)
    tids = [t for t in screen if screen[t]]
    title = {t: g.norm(cands[t]['parsed']['title']) for t in cands}
    midis = sorted({r['md5'] for v in screen.values() for r in v if r['kind'] != 'null' and 'ratio' in r})
    owner = {}
    for t, v in screen.items():
        for r in v:
            owner.setdefault(r['md5'], set()).add(title[t])
    # the null MIDI pool: well-formed candidate MIDIs of other songs
    t0 = time.time()
    raw_cache = {}

    def araw(t):
        if t not in raw_cache:
            if len(raw_cache) > 12:
                raw_cache.pop(next(iter(raw_cache)))
            raw_cache[t] = align.audio_raw(cands[t]['corpus'], t, cands[t]['path'])
        return raw_cache[t]

    for i, t in enumerate(tids):
        if t in out['audio'] or not g.mine(t):
            continue
        pool = [m for m in midis if title[t] not in owner.get(m, ())]
        res = []
        for m in rnd.sample(pool, min(K_M, len(pool))):
            try:
                res.append(align.best_shift(align.midi_raw(os.path.join(g.WORK, 'midi', m + '.mid')), araw(t))[0])
            except Exception:
                pass
        out['audio'][t] = part['audio'][t] = res
        save()
        print(f'audio {i}/{len(tids)} {time.time() - t0:.0f}s', flush=True)
    # MIDI side: reference recordings are random screened tracks of other titles
    for i, m in enumerate(midis):
        if m in out['midi'] or not g.mine(m):
            continue
        refs = [t for t in tids if title[t] not in owner.get(m, ())]
        res = []
        try:
            Rm = align.midi_raw(os.path.join(g.WORK, 'midi', m + '.mid'))
        except Exception:
            out['midi'][m] = part['midi'][m] = []; continue
        for t in rnd.sample(refs, min(K_A, len(refs))):
            res.append(align.best_shift(Rm, araw(t))[0])
        out['midi'][m] = part['midi'][m] = res
        if i % 10 == 0:
            save()
            print(f'midi {i}/{len(midis)} {time.time() - t0:.0f}s', flush=True)
    save()


if __name__ == '__main__':
    main()
