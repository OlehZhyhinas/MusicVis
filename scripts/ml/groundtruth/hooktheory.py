"""Human melody + chord annotations from the Hooktheory TheoryTab set (SheetSage release), aligned to our audio.

Hooktheory.json.gz (datasets/hooktheory) holds ~26k clips (usually a verse/chorus of 10-60 s) with melody
notes and chords in beats, plus a human-checked beat->time alignment to one YouTube video.

Per matched (track, clip):
  * The clip is rendered as MIDI (melody = lead, chords = sustained pad voicing, root in octave 3) on the
    original video's beat times, synthesised in memory and turned into chroma-folded CQT features.
  * Subsequence DTW (align.subseq_dtw) locates it in the recording; contrast vs. null clips of other songs
    gives the confidence.
  * If the clip's YouTube id is in the track's playlist archive and the durations agree within 0.5 s, the
    file is that very video: the human alignment is used as is when DTW agrees within 0.3 s ("direct").
    Otherwise times come from the DTW path ("dtw").
Output entries (list per track) go to WORK/hooktheory.json; build_labels.py merges them into the labels.
"""
import collections, glob, gzip, json, os, sys
import numpy as np
import pretty_midi
sys.path.insert(0, os.path.dirname(__file__))
import gt_common as g
import align
from lakh_match import artist_hit

HT_PATH = os.path.join(g.DATASETS, 'hooktheory', 'Hooktheory.json.gz')


def load():
    return json.load(gzip.open(HT_PATH))


def matches(H):
    byt = collections.defaultdict(list)
    for k, v in H.items():
        if 'HARMONY' in v['tags'] or 'MELODY' in v['tags']:
            byt[g.norm(v['hooktheory']['song'].replace('-', ' '))].append(k)
    out = {}
    for c, tid, p in g.all_tracks():
        pt = g.parse_title(tid)
        tt = g.norm(pt['title'])
        arts = pt['artists'] or []
        ks = [k for k in byt.get(tt, []) if arts and artist_hit(arts, g.norm(H[k]['hooktheory']['artist'].replace('-', ' ')))]
        if not ks and not arts:
            ks = byt.get(tt, [])[:6]                   # title-only file names: DTW decides
        if ks:
            out[tid] = (c, p, ks)
    return out


def beat_to_time(clip):
    al = clip['alignment'].get('refined') or clip['alignment']['user']
    b = np.array(al['beats'], float); t = np.array(al['times'], float)

    def f(x):
        x = np.asarray(x, float)
        y = np.interp(x, b, t)
        if len(b) >= 2:
            lo = (b[1] - b[0]); hi = (b[-1] - b[-2])
            y = np.where(x < b[0], t[0] + (x - b[0]) * (t[1] - t[0]) / lo, y)
            y = np.where(x > b[-1], t[-1] + (x - b[-1]) * (t[-1] - t[-2]) / hi, y)
        return y
    return f


def chord_pitches(h):
    root = 48 + h['root_pitch_class']
    ps = [root]
    for iv in h['root_position_intervals']:
        ps.append(ps[-1] + iv)
    for _ in range(h.get('inversion', 0)):
        ps = ps[1:] + [ps[0] + 12]
    return ps


QUAL = {(4, 3): 'maj', (3, 4): 'min', (3, 3): 'dim', (4, 4): 'aug', (4, 3, 3): '7', (4, 3, 4): 'maj7',
        (3, 4, 3): 'min7', (3, 3, 4): 'hdim7', (3, 3, 3): 'dim7', (2, 5): 'sus2', (5, 2): 'sus4',
        (5, 2, 3): '7(sus4)' , (4, 3, 7): 'maj(9)', (3, 4, 7): 'min(9)'}
PC = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def chord_label(h):
    q = QUAL.get(tuple(h['root_position_intervals']))
    lab = PC[h['root_pitch_class']] + ':' + (q or 'maj')
    if h.get('inversion'):
        ps = chord_pitches(h)
        bass_iv = (ps[0] - (48 + h['root_pitch_class'])) % 12
        lab += '/' + {3: 'b3', 4: '3', 7: '5', 6: 'b5', 8: '#5', 10: 'b7', 11: '7', 2: '2', 5: '4'}.get(bass_iv, '1')
    return lab


def clip_events(clip):
    """Notes/chords/beats of a clip in the ORIGINAL video's seconds."""
    a = clip['annotations']; f = beat_to_time(clip)
    mel = [(float(f(n['onset'])), float(f(n['offset'])), 60 + 12 * n['octave'] + n['pitch_class']) for n in a.get('melody') or []]
    har = [(float(f(h['onset'])), float(f(h['offset'])), chord_label(h), chord_pitches(h)) for h in a.get('harmony') or []]
    nb = a['num_beats']
    bpb = a['meters'][0]['beats_per_bar'] if a.get('meters') else 4
    beats = [(float(f(i)), (i % bpb) == 0) for i in range(nb + 1)]
    return mel, har, beats


def clip_midi(mel, har):
    pm = pretty_midi.PrettyMIDI()
    lead = pretty_midi.Instrument(program=80, name='melody')        # square lead: strong pitch
    pad = pretty_midi.Instrument(program=0, name='chords')
    for s, e, p in mel:
        lead.notes.append(pretty_midi.Note(100, int(p), s, max(e, s + 0.05)))
    for s, e, _, ps in har:
        for p in ps:
            pad.notes.append(pretty_midi.Note(80, int(p), s, max(e, s + 0.05)))
    pm.instruments += [lead, pad]
    return pm


def audio_id_durations():
    """{playlist: set(youtube ids)} from the yt-dlp archives."""
    out = {}
    for f in glob.glob(os.path.join(g.OWN_DIR, 'pl-*', 'archive.txt')):
        out[os.path.basename(os.path.dirname(f))] = {l.split()[1] for l in open(f) if len(l.split()) == 2}
    return out


def locate(Ra_chroma, pm_clip, t0):
    """Subsequence-DTW the synthesised clip (starting at original time t0) into the recording.
    Returns (cost ratio, path midi_t (orig secs), path audio_t)."""
    y = align.synth(pm_clip)
    i0 = int(t0 * align.SR)
    R = align.raw_cqt(y[i0:], 0.0)
    Fm = align.feat(R, 'chroma')
    C = 1.0 - Fm @ Ra_chroma.T
    med = float(np.median(C))
    D, S = align.subseq_dtw(C.astype(np.float64), med)
    e = int(np.argmin(D)); s = int(S[e])
    pi, pj = align.full_dtw(C[:, s:e + 1].astype(np.float64), med)
    ratio = float(np.mean(C[pi, pj + s])) / med
    return ratio, t0 + pi / align.FPS, (pj + s) / align.FPS


def main():
    H = load()
    M = matches(H)
    ids = audio_id_durations()
    outp = os.path.join(g.WORK, 'hooktheory.json')
    out = json.load(open(outp)) if os.path.exists(outp) else {}
    rng = np.random.default_rng(1)
    allk = [k for k, v in H.items() if 'HARMONY' in v['tags']]
    for tid, (corpus, path, ks) in M.items():
        if tid in out:
            continue
        Ra = align.audio_raw(corpus, tid, path)
        Fa = align.feat(Ra, 'chroma')
        dur = len(Ra) / align.FPS
        pl = tid.split('__')[0] if corpus == 'own' else None
        res = []
        nulls = [allk[i] for i in rng.integers(0, len(allk), 2)]
        for k in ks + nulls:
            clip = H[k]
            try:
                mel, har, beats = clip_events(clip)
                if not har and not mel:
                    continue
                t0 = max(0.0, min([x[0] for x in mel] + [x[0] for x in har]) - 0.5)
                ratio, pt, pa = locate(Fa, clip_midi(mel, har), t0)
            except Exception as ex:
                print('fail', tid[:40], k, ex); continue
            same_video = bool(pl and clip['youtube']['id'] in ids.get(pl, set())
                              and abs(clip['youtube']['duration'] - dur) < 0.5)
            u, inv = np.unique(pt, return_inverse=True)
            av = np.maximum.accumulate(np.bincount(inv, weights=pa) / np.bincount(inv))
            dev = float(np.median(np.abs(av - u)))                   # DTW vs identity (same video)
            res.append(dict(clip=k, null=k in nulls and k not in ks, ratio=round(ratio, 4), same_video=same_video,
                            dtw_vs_identity=round(dev, 3), map_t=np.round(u[::4], 3).tolist(),
                            map_a=np.round(av[::4], 3).tolist(), yt=clip['youtube']['id'],
                            song=clip['hooktheory']['artist'] + '/' + clip['hooktheory']['song']))
        out[tid] = res
        json.dump(out, open(outp, 'w'))
        print(tid[:50], [(r['ratio'], r['null'], r['same_video'], r['dtw_vs_identity']) for r in res], flush=True)


if __name__ == '__main__':
    main()
