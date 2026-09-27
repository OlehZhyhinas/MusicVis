"""Shared paths, corpora and audio loading for the instrument / drum / note prototypes (test-only).

Corpora (audio is decoded with ffmpeg and never played):
  test  the 13 test songs in ~/Downloads/YoutubeToMp3 (the "(1)" duplicate skipped). HELD OUT: never trained on.
  own   the owner's playlists, ~/personal/MusicVis-data/own/<playlist>/NNN - title.m4a (pl-pop100, pl-edm42,
        pl-pop2; still downloading). Every 5th track (NNN % 5 == 0) is HELD OUT for own-taste evaluation;
        the rest is training data. See is_eval().
  fma   FMA medium 30 s clips, ~/personal/MusicVis-data/fma/fma_medium/<ddd>/<id>.mp3 (training only)

Teacher label caches are shared with the other ML agents under
  ~/personal/MusicVis-data/labels/<teacher>/<corpus>/<id>.<ext>
Decoded audio is cached as mono 22050 Hz float32 under ~/personal/MusicVis-data/pcm22/<corpus>/<id>.f32
(44.1 kHz stereo is decoded on demand, not cached, except for the test songs which already live in
.testdata/live/pcm/<slug>.f32 as interleaved stereo 44.1 kHz).
"""
import os, re, subprocess
import numpy as np

HOME = os.path.expanduser('~')
DATA = os.path.join(HOME, 'personal/MusicVis-data')
LABELS = os.path.join(DATA, 'labels')
PCM22 = os.path.join(DATA, 'pcm22')
TEST_DIR = os.path.join(HOME, 'Downloads/YoutubeToMp3')
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
TESTDATA = os.path.join(REPO, '.testdata')
WORK = os.path.join(TESTDATA, 'instr')
AUDIO_EXT = re.compile(r'\.(mp3|m4a|wav|flac|ogg|webm|opus)$', re.I)


def slug_of(path):
    """Same as scripts/live/common.ts slugOf()."""
    base = re.sub(r'\.[^.]+$', '', os.path.basename(path))
    s = re.sub(r'\(.*?\)', '', base.lower())
    s = re.sub(r'[^a-z0-9]+', '-', s).strip('-')
    return s[:40] or 'song'


def tracks(corpus):
    """[(id, path)] for a corpus, sorted by id."""
    out = {}
    if corpus == 'test':
        for f in sorted(os.listdir(TEST_DIR)):
            if AUDIO_EXT.search(f) and not re.search(r'\(\d+\)\.[a-z0-9]+$', f, re.I):
                out[slug_of(f)] = os.path.join(TEST_DIR, f)
    elif corpus == 'own':
        root = os.path.join(DATA, 'own')
        for pl in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            d = os.path.join(root, pl)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                # only finished m4a/mp3 (skip .part/.webm/.temp files that yt-dlp is still writing)
                if re.search(r'\.(m4a|mp3)$', f, re.I):
                    k = slug_of(re.sub(r'^\d+\s*-\s*', '', f))
                    out[k] = os.path.join(d, f)
    elif corpus == 'fma':
        root = os.path.join(DATA, 'fma', 'fma_medium')
        for sub in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            d = os.path.join(root, sub)
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.endswith('.mp3'):
                        out[f[:-4]] = os.path.join(d, f)
    else:
        raise ValueError(corpus)
    return sorted(out.items())


def is_eval(corpus, tid, path):
    """Held-out split: every test song, and every 5th track of each owner playlist (by its NNN prefix)."""
    if corpus == 'test':
        return True
    if corpus == 'own':
        m = re.match(r'(\d+)', os.path.basename(path))
        return bool(m) and int(m.group(1)) % 5 == 0
    return False


def decode(path, sr=22050, channels=1):
    """float32 [n] (mono) or [n, 2] via ffmpeg to a pipe; nothing is played."""
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-f', 'f32le', '-ac', str(channels), '-ar', str(sr), '-'],
                         check=True, capture_output=True).stdout
    x = np.frombuffer(raw, dtype='<f4').copy()
    return x if channels == 1 else x.reshape(-1, channels)


def pcm22(corpus, tid, path):
    """Cached mono 22050 Hz float32."""
    p = os.path.join(PCM22, corpus, tid + '.f32')
    if os.path.exists(p):
        return np.fromfile(p, dtype='<f4')
    x = decode(path, 22050, 1)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    x.tofile(p)
    return x


def label_path(teacher, corpus, tid, ext):
    p = os.path.join(LABELS, teacher, corpus, f'{tid}.{ext}')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
