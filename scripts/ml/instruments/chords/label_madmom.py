"""Offline chord/key teacher labels via madmom, cached to the shared label store.

Two independent chord teachers (so we can measure teacher-vs-teacher agreement):
  madmom-dc-crf   DeepChromaProcessor (small feedforward DNN, ~538k params) -> CRF   (maj/min/N, 25 classes)
  madmom-cnn-crf  CNNChordFeatureProcessor (conv net)      -> CRF                    (maj/min/N, 25 classes)
And one key teacher:
  madmom-key      CNNKeyRecognitionProcessor                                        (24 major/minor keys)

All are the OFFLINE (non-causal) madmom pipelines: full-context feature extraction + Viterbi-style
CRF / global argmax decoding. These are ground truth for scoring the live tracker and any causal
candidate, not something we'd ship.

Usage:
  .testdata/instr/venvs/adtof/bin/python scripts/ml/instruments/chords/label_madmom.py --corpus test
  .testdata/instr/venvs/adtof/bin/python scripts/ml/instruments/chords/label_madmom.py --corpus own --eval-only
"""
import argparse, json, os, sys, tempfile, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import common

import numpy as np


def atomic_write_json(path, obj):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix='.tmp-')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def segments_to_json(segs):
    # madmom SEGMENT_DTYPE structured array: (start, end, label)
    return [[float(s['start']), float(s['end']), str(s['label'])] for s in segs]


def label_chords_deepchroma(wav_path):
    from madmom.audio.chroma import DeepChromaProcessor
    from madmom.features.chords import DeepChromaChordRecognitionProcessor
    dcp = DeepChromaProcessor()
    decode = DeepChromaChordRecognitionProcessor()
    chroma = dcp(wav_path)
    segs = decode(chroma)
    return segments_to_json(segs)


def label_chords_cnnfeat(wav_path):
    from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor
    featproc = CNNChordFeatureProcessor()
    decode = CRFChordRecognitionProcessor()
    feats = featproc(wav_path)
    segs = decode(feats)
    return segments_to_json(segs)


def label_key(wav_path):
    from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label
    proc = CNNKeyRecognitionProcessor()
    pred = proc(wav_path)
    label = key_prediction_to_label(pred)
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='test', choices=['test', 'own', 'fma'])
    ap.add_argument('--eval-only', action='store_true', help='own corpus: only every-5th (held-out) track')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--songs', default='')
    args = ap.parse_args()

    tracks = common.tracks(args.corpus)
    if args.songs:
        want = set(args.songs.split(','))
        tracks = [(tid, p) for tid, p in tracks if tid in want]
    if args.eval_only:
        tracks = [(tid, p) for tid, p in tracks if common.is_eval(args.corpus, tid, p)]
    if args.limit:
        tracks = tracks[: args.limit]

    print(f'{len(tracks)} tracks in corpus={args.corpus} eval_only={args.eval_only}', file=sys.stderr)

    wav_dir = os.path.join(common.WORK, 'wav')
    os.makedirs(wav_dir, exist_ok=True)

    for i, (tid, path) in enumerate(tracks):
        t0 = time.time()
        wav_path = os.path.join(common.TESTDATA, 'instr', 'wav', tid + '.wav') if args.corpus == 'test' else os.path.join(wav_dir, tid + '.wav')
        if not os.path.exists(wav_path):
            # decode to a temp mono 44.1k wav madmom can read directly
            import soundfile as sf
            x = common.decode(path, sr=44100, channels=1)
            sf.write(wav_path, x, 44100)

        out_dc = common.label_path('madmom-dc-crf', args.corpus, tid, 'json')
        out_cnn = common.label_path('madmom-cnn-crf', args.corpus, tid, 'json')
        out_key = common.label_path('madmom-key', args.corpus, tid, 'json')

        did = []
        if not os.path.exists(out_dc):
            segs = label_chords_deepchroma(wav_path)
            atomic_write_json(out_dc, {'teacher': 'madmom-dc-crf', 'id': tid, 'segments': segs})
            did.append('dc-crf')
        if not os.path.exists(out_cnn):
            segs = label_chords_cnnfeat(wav_path)
            atomic_write_json(out_cnn, {'teacher': 'madmom-cnn-crf', 'id': tid, 'segments': segs})
            did.append('cnn-crf')
        if not os.path.exists(out_key):
            label = label_key(wav_path)
            atomic_write_json(out_key, {'teacher': 'madmom-key', 'id': tid, 'key': label})
            did.append('key')

        print(f'[{i+1}/{len(tracks)}] {tid}: {did or "cached"} ({time.time()-t0:.1f}s)', file=sys.stderr)


if __name__ == '__main__':
    main()
