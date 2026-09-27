"""Second MIDI source: freemidi.org (public, no login), fetched politely (see GAP below).

For each track: search by title, keep song pages whose URL slug carries the title and (when known) an
artist, download up to MAXD files via the site's own download link (getter-<id>), store them as
WORK/midi/fm<id>.mid and append them to WORK/candidates.json with why "freemidi:artist+title" / "freemidi:title".
Polite: one request per GAP s. On a first HTTP 429 it waits (Retry-After or 5 min) and slows to 6 s per
request; a second 429, any 403 or a captcha page stops the run for good.

Run: .testdata/gt/venv/bin/python scripts/ml/groundtruth/freemidi_fetch.py
"""
import json, os, re, sys, time
import requests
sys.path.insert(0, os.path.dirname(__file__))
import gt_common as g
from lakh_match import artist_hit

BASE = 'https://freemidi.org/'
UA = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36'}
MAXD = 3
S = requests.Session(); S.headers.update(UA)
_last = [0.0]
GAP = [2.5]
N429 = [0]


def get(url, **kw):
    dt = time.time() - _last[0]
    if dt < GAP[0]:
        time.sleep(GAP[0] - dt)
    _last[0] = time.time()
    r = S.get(url, timeout=30, **kw)
    if r.status_code == 429 and N429[0] == 0:
        N429[0] += 1; GAP[0] = 6.0
        wait = int(r.headers.get('Retry-After', '300')) if r.headers.get('Retry-After', '').isdigit() else 300
        print(f'429 at {url}; waiting {wait}s and slowing down', flush=True)
        time.sleep(wait)
        return get(url, **kw)
    if r.status_code in (403, 429) or b'captcha' in r.content[:20000].lower():
        raise SystemExit(f'blocked at {url}: {r.status_code}')
    return r


def main():
    cp = os.path.join(g.WORK, 'candidates.json')
    d = json.load(open(cp))
    donep = os.path.join(g.WORK, 'freemidi_done.json')
    done = json.load(open(donep)) if os.path.exists(donep) else {}
    for tid, v in d.items():
        if tid in done:
            continue
        p = v['parsed']
        tt = g.norm(p['title'])
        if len(tt) < 2:
            done[tid] = []; continue
        hits = []
        for q in {p['title'].strip(), (p['alt_title'] or '').strip()} - {''}:
            r = get(BASE + 'search', params={'q': q})
            for m in re.finditer(r'href=(download3-(\d+)-([a-z0-9-]+))', r.text):
                url, sid, slug = m.groups()
                words = slug.replace('-', ' ')
                qn = g.norm(q)
                if not (words.startswith(qn + ' ') or words == qn):
                    continue
                rest = words[len(qn):].strip()          # artist part of the slug
                hit = artist_hit(p['artists'] + ([p['title']] if p['alt_title'] else []), rest)
                if p['artists'] and not hit:
                    continue
                hits.append((sid, url, 'freemidi:artist+title' if hit else 'freemidi:title', rest))
        seen, got = set(), []
        for sid, url, why, rest in hits:
            if sid in seen or len(got) >= MAXD:
                continue
            seen.add(sid)
            f = os.path.join(g.WORK, 'midi', f'fm{sid}.mid')
            if not os.path.exists(f):
                get(BASE + url)
                r = get(BASE + f'getter-{sid}', headers={'Referer': BASE + url})
                if not r.content.startswith(b'MThd'):
                    continue
                open(f, 'wb').write(r.content)
            got.append(dict(md5=f'fm{sid}', why=why, score=3 if 'artist' in why else 1, names=[[rest, tt]]))
        done[tid] = [x['md5'] for x in got]
        have = {c['md5'] for c in v['cands']}
        v['cands'] += [x for x in got if x['md5'] not in have]
        json.dump(d, open(cp, 'w'), indent=1, ensure_ascii=False)
        json.dump(done, open(donep, 'w'))
        if got:
            print(tid[:60], [x['md5'] + ' ' + x['why'] for x in got], flush=True)
    print('tracks with freemidi files:', sum(1 for x in done.values() if x))


if __name__ == '__main__':
    main()
