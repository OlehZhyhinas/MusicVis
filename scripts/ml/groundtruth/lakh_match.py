"""Match the owner's tracks and the test songs to Lakh MIDI files by normalised artist/title.

Name sources for every LMD-full md5:
  clean_midi.tar.gz      clean_midi/<Artist>/<Title>[.N].mid            (curated names)
  md5_to_paths.json      original scraped paths, e.g. 'Beatles/Help.mid', 'B/Billy_Joel_-_The_Stranger.mid'
  match_scores.json      LMD-matched: MSD track id -> {md5: score}, names from MSD unique_tracks.txt
Title-only file names (no artist) are matched on title alone; alignment later verifies them.

Only tracks not yet in WORK/candidates.json are matched; existing entries are kept.
Writes WORK/candidates.json: {id: {"corpus", "path", "parsed", "cands": [{"md5", "why", "names"}]}}
and extracts every candidate MIDI to WORK/midi/<md5>.mid (streamed out of lmd_full.tar.gz, no full unpack).

Run: .testdata/gt/venv/bin/python scripts/ml/groundtruth/lakh_match.py
"""
import collections, difflib, hashlib, json, os, re, sys, tarfile
sys.path.insert(0, os.path.dirname(__file__))
import gt_common as g

MAXC = 10


def load_names():
    names = collections.defaultdict(set)     # md5 -> {(artist_norm, title_norm)}; artist '' if unknown
    src = collections.defaultdict(set)
    cache = os.path.join(g.WORK, 'lakh_names.json')
    if os.path.exists(cache):
        d = json.load(open(cache))
        return ({k: set(map(tuple, v)) for k, v in d['names'].items()}, {k: set(v) for k, v in d['src'].items()})
    # clean_midi: hash the files so they join LMD-full md5s
    with tarfile.open(os.path.join(g.LAKH, 'clean_midi.tar.gz')) as tf:
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith('.mid'):
                continue
            parts = m.name.split('/')
            if len(parts) != 3:
                continue
            md5 = hashlib.md5(tf.extractfile(m).read()).hexdigest()
            title = re.sub(r'(\.\d+)?\.mid$', '', parts[2], flags=re.I)
            names[md5].add((g.norm(parts[1]), g.norm(title)))
            src[md5].add('clean')
    # LMD-matched via MSD metadata
    msd = {}
    with open(os.path.join(g.LAKH, 'unique_tracks.txt'), encoding='utf-8', errors='replace') as f:
        for line in f:
            p = line.rstrip('\n').split('<SEP>')
            if len(p) == 4:
                msd[p[0]] = (g.norm(p[2]), g.norm(p[3]))
    for tid, d in json.load(open(os.path.join(g.LAKH, 'match_scores.json'))).items():
        if tid in msd:
            for md5, sc in d.items():
                names[md5].add(msd[tid]); src[md5].add('matched')
    # scraped original paths
    for md5, paths in json.load(open(os.path.join(g.LAKH, 'md5_to_paths.json'))).items():
        for p in paths:
            parts = p.replace('\\', '/').split('/')
            stem = re.sub(r'\.(mid|midi|kar)$', '', parts[-1], flags=re.I).replace('_', ' ')
            stem = re.sub(r'\s+\d{1,2}$', '', stem)             # 'song2' variants
            d = parts[-2].replace('_', ' ') if len(parts) >= 2 else ''
            if re.search(r'\s-\s|--', stem):                    # 'Artist - Title'
                a, t = re.split(r'\s-\s|--', stem, maxsplit=1)
                names[md5].add((g.norm(a), g.norm(t))); names[md5].add((g.norm(t), g.norm(a)))
            if len(d) > 1 and not re.search(r'midi|unsorted|various|misc|^\w$|\d{4}', d, re.I):
                names[md5].add((g.norm(d), g.norm(stem)))
            names[md5].add(('', g.norm(stem)))
            src[md5].add('paths')
    os.makedirs(g.WORK, exist_ok=True)
    json.dump({'names': {k: sorted(v) for k, v in names.items()}, 'src': {k: sorted(v) for k, v in src.items()}},
              open(cache, 'w'))
    return names, src


def artist_hit(track_artists, a):
    if not a:
        return False
    toks = set(a.split())
    for ta in track_artists:
        n = g.norm(ta)
        if not n:
            continue
        if n == a or (len(n) > 3 and (n in a or a in n)) or difflib.SequenceMatcher(None, n, a).ratio() > 0.85:
            return True
        nt = set(n.split())
        if nt and len(nt & toks) >= max(1, len(nt) - 1) and len(nt) > 1:
            return True
    return False


def title_ok(tt, t):
    if not tt or not t:
        return False
    if tt == t:
        return True
    if len(tt) >= 6 and (t.startswith(tt + ' ') or t.endswith(' ' + tt)) and len(t) - len(tt) < 12:
        return True
    return len(tt) >= 6 and difflib.SequenceMatcher(None, tt, t).ratio() > 0.9


def main():
    names, src = load_names()
    by_title = collections.defaultdict(list)
    for md5, s in names.items():
        for a, t in s:
            by_title[t].append((md5, a))
    titles = list(by_title)
    cp = os.path.join(g.WORK, 'candidates.json')
    out = g.load_json(cp) if os.path.exists(cp) else {}       # keep earlier entries (incl. freemidi additions)
    for corpus, tid, path in g.all_tracks():
        if tid in out:
            continue
        p = g.parse_title(tid)
        tts = {g.norm(p['title'])}
        if p['alt_title']:
            tts.add(g.norm(p['alt_title']))
        tts.discard('')
        scored = {}
        for tt in tts:
            keys = [tt] + ([k for k in difflib.get_close_matches(tt, titles, n=30, cutoff=0.9)] if len(tt) >= 6 else [])
            keys += [k for k in titles if len(tt) >= 6 and k != tt and (k.startswith(tt + ' ') or k.endswith(' ' + tt))
                     and len(k) - len(tt) < 12] if len(tt) >= 8 else []
            for k in set(keys):
                if not title_ok(tt, k):
                    continue
                for md5, a in by_title[k]:
                    hit = artist_hit(p['artists'] + ([p['title']] if p['alt_title'] else []), a)
                    score = (2 if hit else 0) + (1 if k == tt else 0) + 0.5 * ('clean' in src.get(md5, ())) \
                        + 0.3 * ('matched' in src.get(md5, ())) + 0.01 * len(names[md5])
                    if not hit and p['artists'] and a:
                        continue                       # an artist is known on both sides and disagrees
                    if not hit and not p['artists'] is None and p['artists'] and len(tt) < 8:
                        continue                       # short title with a known artist: need the artist
                    if score > scored.get(md5, (-1,))[0]:
                        scored[md5] = (score, 'artist+title' if hit else 'title')
        c = sorted(scored.items(), key=lambda kv: -kv[1][0])
        if any(v[1] == 'artist+title' for _, v in c):
            c = [kv for kv in c if kv[1][1] == 'artist+title']
        c = c[:MAXC]
        out[tid] = dict(corpus=corpus, path=path, parsed=p,
                        cands=[dict(md5=m, why=v[1], score=round(v[0], 2), names=sorted(names[m])[:4]) for m, v in c])
    want = {c['md5'] for v in out.values() for c in v['cands']}
    mdir = os.path.join(g.WORK, 'midi'); os.makedirs(mdir, exist_ok=True)
    want -= {f[:-4] for f in os.listdir(mdir)}
    if want:
        with tarfile.open(os.path.join(g.LAKH, 'lmd_full.tar.gz')) as tf:
            for m in tf:
                md5 = os.path.basename(m.name)[:-4]
                if m.isfile() and md5 in want:
                    open(os.path.join(mdir, md5 + '.mid'), 'wb').write(tf.extractfile(m).read())
    g.save_json(out, cp, indent=1, ensure_ascii=False)
    n = sum(1 for v in out.values() if v['cands'])
    na = sum(1 for v in out.values() if any(c['why'] == 'artist+title' for c in v['cands']))
    print(f'{len(out)} tracks, {n} with Lakh candidates ({na} artist+title), {len(want)} MIDIs extracted')


if __name__ == '__main__':
    main()
