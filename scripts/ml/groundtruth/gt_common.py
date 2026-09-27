"""Shared paths, track listing and title parsing for the MIDI ground-truth pipeline.

Ids follow the path scheme: the audio path relative to its corpus dir, "/" replaced by "__", extension
dropped, e.g. own "pl-edm42__001 - Martin Garrix & Jex - Told You So (Official Video)", test
"Avicii - Waiting For Love". The instruments agent's slug id (scripts/ml/instruments/common.py slug_of)
is stored alongside in every output file as "slug".

Audio is decoded with ffmpeg to a pipe and never played.
"""
import os, re, subprocess, unicodedata
import numpy as np

HOME = os.path.expanduser('~')
DATA = os.path.join(HOME, 'personal/MusicVis-data')
LABELS = os.path.join(DATA, 'labels')
DATASETS = os.path.join(DATA, 'datasets')
LAKH = os.path.join(DATASETS, 'lakh')
GT = os.path.join(LABELS, 'midi')
WORK = os.path.join(DATA, 'gt-work')          # candidate MIDIs, caches, alignment debug
TEST_DIR = os.path.join(HOME, 'Downloads/YoutubeToMp3')
OWN_DIR = os.path.join(DATA, 'own')
SOUNDFONT = os.path.join(DATASETS, 'soundfonts/MuseScore_General.sf2')
AUDIO_RE = re.compile(r'\.(mp3|m4a)$', re.I)


def slug_of(path):
    base = re.sub(r'\.[^.]+$', '', os.path.basename(path))
    s = re.sub(r'\(.*?\)', '', base.lower())
    s = re.sub(r'[^a-z0-9]+', '-', s).strip('-')
    return s[:40] or 'song'


def tracks(corpus):
    """[(id, path)] with the path-scheme id. The test "(1)" duplicate is skipped."""
    out = []
    if corpus == 'test':
        for f in sorted(os.listdir(TEST_DIR)):
            if AUDIO_RE.search(f) and not re.search(r'\(\d+\)\.[a-z0-9]+$', f, re.I):
                out.append((AUDIO_RE.sub('', f), os.path.join(TEST_DIR, f)))
    elif corpus == 'own':
        for pl in sorted(os.listdir(OWN_DIR)):
            d = os.path.join(OWN_DIR, pl)
            if pl.startswith('pl-') and os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if AUDIO_RE.search(f):
                        out.append((pl + '__' + AUDIO_RE.sub('', f), os.path.join(d, f)))
    else:
        raise ValueError(corpus)
    return out


def all_tracks():
    return [('test', i, p) for i, p in tracks('test')] + [('own', i, p) for i, p in tracks('own')]


# ---------------------------------------------------------------- title parsing / normalisation
_JUNK = re.compile(r'''(official\s*(music\s*)?(video|audio|visuali[sz]er|lyric\s*video|single)|lyric(s)?(\s*video)?|
    visuali[sz]er|audio|video|hd|hq|4k|remaster(ed)?(\s*\d{4})?|explicit|clean|radio\s*edit|original\s*mix|
    extended(\s*(mix|version))?|club\s*mix|live(\s*performance.*)?|animated\s*video|car\s*music|sub\.?\s*espa\S+|
    alternative\s*music\s*video.*|\d{4})''', re.I | re.X)


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))


def norm(s):
    """Lowercase ascii alnum tokens joined by single spaces; '&'/'and'/'the' unified."""
    s = strip_accents(s).lower().replace('&', ' and ').replace('$', 's')
    s = re.sub(r"[’'`]", '', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    toks = [t for t in s.split() if t not in ('the', 'a')]
    return ' '.join(toks)


def parse_title(name):
    """YouTube-style file name (id tail) -> dict(artists=[...], title=str, raw=str, remix=bool).

    '001 - A, B & C - Song (feat. D) (Official Video)' -> artists [A,B,C,D], title 'Song'.
    Names without ' - ' give artists [] (title only; matched on title alone and verified by alignment).
    """
    raw = name.split('__')[-1]
    raw = re.sub(r'^\d{3}\s*-\s*', '', raw)
    raw = raw.replace('｜', '|').replace('–', '-').replace('—', '-')
    raw = raw.split('|')[0].strip()
    remix = bool(re.search(r'\b(remix|cover|edit|bootleg|vip|rework|flip)\b', raw, re.I)) and \
        not re.search(r'radio edit', raw, re.I)
    feats = []
    for m in re.finditer(r'[\(\[]\s*(?:feat\.?|ft\.?|featuring|with)\s+([^\)\]]+)[\)\]]', raw, re.I):
        feats += re.split(r'\s*(?:,|&| x | and )\s*', m.group(1))
    body = re.sub(r'[\(\[][^\)\]]*[\)\]]', ' ', raw)           # drop all bracketed parts
    parts = [p.strip() for p in re.split(r'\s+-\s+', body) if p.strip()]
    artists, title = [], body.strip()
    if len(parts) >= 2:
        a, title = parts[0], ' '.join(parts[1:])
        # handle "Title - Artist" orderings is ambiguous; keep both via 'alt'
        m = re.split(r'\s+(?:feat\.?|ft\.?|featuring)\s+', a, flags=re.I)
        a = m[0]; feats += m[1:]
        artists = re.split(r'\s*(?:,|&| x | X | vs\.? | and )\s*', a)
    m = re.split(r'\s+(?:feat\.?|ft\.?|featuring)\s+', title, flags=re.I)
    title = m[0]; feats += m[1:]
    title = _JUNK.sub(' ', title)
    artists = [x.strip() for x in artists + feats if x.strip()]
    alt = parts[0] if len(parts) >= 2 else None                # for "Title - Artist" names
    return dict(artists=artists, title=title.strip(), raw=raw, remix=remix, alt_title=alt)


def decode(path, sr=22050):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-f', 'f32le', '-ac', '1', '-ar', str(sr), '-'],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype='<f4').copy()


def pcm22(corpus, tid, path):
    """Mono 22050 Hz float32, decoded fresh (the shared slug-keyed pcm22 cache has slug collisions)."""
    return decode(path)


def load_json(path, tries=20):
    """json.load that tolerates a concurrent writer (retries on a half-written file)."""
    import json, time
    for i in range(tries):
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            if i == tries - 1:
                raise
            time.sleep(0.5)


def save_json(obj, path, **kw):
    """Atomic json write (tmp file + rename)."""
    import json
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, **kw)
    os.replace(tmp, path)
