"""Chord labels from symbolic notes (aligned MIDI), one decision per beat, Viterbi-smoothed.

Per beat span: pitch-class profile of all pitched (non-drum) notes weighted by overlap x velocity, the bass
(lowest sounding pitch class, duration weighted) counted extra. Scored against templates for
maj, min, 7, maj7, min7, dim, sus4 (root weight 1.5, other tones 1, bass-on-root bonus); "N" when the beat is
(near) silent. A Viterbi pass with a self-transition bonus removes one-beat flickers. Labels are in
mir_eval / Harte syntax ("C:maj", "A:min7", "N").
"""
import numpy as np

PC = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
QUALS = {'maj': (0, 4, 7), 'min': (0, 3, 7), '7': (0, 4, 7, 10), 'maj7': (0, 4, 7, 11), 'min7': (0, 3, 7, 10),
         'dim': (0, 3, 6), 'sus4': (0, 5, 7)}
PRIOR = {'maj': 0.0, 'min': 0.0, '7': -0.04, 'maj7': -0.05, 'min7': -0.04, 'dim': -0.08, 'sus4': -0.08}


def _templates():
    labs, T, roots = [], [], []
    for r in range(12):
        for q, ivs in QUALS.items():
            t = np.full(12, -0.6)
            for i, iv in enumerate(ivs):
                t[(r + iv) % 12] = 1.5 if i == 0 else 1.0
            labs.append(f'{PC[r]}:{q}'); T.append(t / np.linalg.norm(t)); roots.append(r)
    return labs, np.array(T), np.array(roots), np.array([PRIOR[l.split(':')[1]] for l in labs])


LABS, TEMPL, ROOTS, PRI = _templates()


def chords_from_notes(notes, beats, min_energy=0.05):
    """notes: array [n, 4] (on, off, pitch, vel) of pitched notes (audio time). beats: sorted times.
    Returns [[t0, t1, label], ...] covering beats[0]..beats[-1]."""
    beats = np.asarray(beats, float)
    if len(beats) < 2 or len(notes) == 0:
        return []
    notes = np.asarray(notes, float)
    nb = len(beats) - 1
    P = np.zeros((nb, 12)); B = np.zeros((nb, 12)); E = np.zeros(nb)
    order = np.argsort(notes[:, 0])
    notes = notes[order]
    for i in range(nb):
        a, b = beats[i], beats[i + 1]
        m = (notes[:, 0] < b) & (notes[:, 1] > a)
        if not m.any():
            continue
        sub = notes[m]
        ov = (np.minimum(sub[:, 1], b) - np.maximum(sub[:, 0], a)) / (b - a)
        w = ov * (sub[:, 3] / 127.0)
        pcs = (sub[:, 2].astype(int)) % 12
        np.add.at(P[i], pcs, w)
        lo = sub[:, 2].min()
        bm = sub[:, 2] <= lo + 2
        np.add.at(B[i], pcs[bm], w[bm])
        E[i] = w.sum()
    S = np.full((nb, len(LABS) + 1), -1.0)
    for i in range(nb):
        if E[i] < min_energy:
            S[i, -1] = 1.0
            continue
        v = P[i] / (np.linalg.norm(P[i]) + 1e-9)
        sc = TEMPL @ v + PRI
        if B[i].sum() > 0:
            bass = int(np.argmax(B[i]))
            sc = sc + 0.08 * (ROOTS == bass)
        S[i, :-1] = sc
        S[i, -1] = 0.2
    # Viterbi, self-transition bonus
    stay = 0.12
    K = S.shape[1]
    D = S[0].copy(); bp = np.zeros((nb, K), int)
    for i in range(1, nb):
        best = int(np.argmax(D))
        cand_stay = D + stay
        take_best = D[best]
        bp[i] = np.where(cand_stay >= take_best, np.arange(K), best)
        D = np.maximum(cand_stay, take_best) + S[i]
    path = np.zeros(nb, int); path[-1] = int(np.argmax(D))
    for i in range(nb - 1, 0, -1):
        path[i - 1] = bp[i, path[i]]
    labs = [LABS[k] if k < len(LABS) else 'N' for k in path]
    out = []
    for i, l in enumerate(labs):
        if out and out[-1][2] == l and abs(out[-1][1] - beats[i]) < 1e-6:
            out[-1][1] = float(beats[i + 1])
        else:
            out.append([float(beats[i]), float(beats[i + 1]), l])
    return out
