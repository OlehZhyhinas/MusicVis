#!/usr/bin/env .testdata/instr/venvs/bpitch/bin/python
"""Teacher 2: Spotify basic-pitch (ONNX) on separated stems + full mix.

TEST corpus: reads stems from .testdata/instr/stems6/<slug>.<stem>.f32
(mono 22050 Hz float32, written by run_demucs6.py) for vocals/bass/guitar/
piano/other, plus the full mix via common.pcm22(). Also saves frame-level
note/onset posteriors (float16) to .testdata/instr/teach/bp-frames/<slug>.<stem>.npz.

own/fma: reads stems from the shared ~/personal/MusicVis-data/stems6/<corpus>/
<id>.<stem>.f16 (written by run_demucs6.py; skipped until that file exists so
demucs never runs twice). 'mix' is NOT run for own/fma (bottleneck; test only).
No frame posteriors saved for own/fma.

Output: ~/personal/MusicVis-data/labels/basicpitch/<corpus>/<id>.json
  {"vocals": [[onset_s, offset_s, midi, amplitude], ...], "bass": [...],
   "guitar": [...], "piano": [...], "other": [...], "mix": [...] (test only)}

Basic-pitch is the throughput bottleneck (~1.8-2x realtime single-process),
so run several shards in parallel, e.g.:
  for i in 0 1 2 3; do OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
    .testdata/instr/venvs/bpitch/bin/python scripts/ml/instruments/teach/run_basicpitch.py \\
    --corpus own --shard $i/4 & done

Run: .testdata/instr/venvs/bpitch/bin/python scripts/ml/instruments/teach/run_basicpitch.py [--limit N] [--corpus test] [--shard i/N]
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time


def stable_hash(s):
    """Deterministic across processes/runs (unlike built-in hash(), which is
    salted per-process), so sharding gives full, non-overlapping coverage."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16)

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import common
import common_teach as ct
import bp_core

STEMS = ['vocals', 'bass', 'guitar', 'piano', 'other']
STEMS6_DIR = os.path.join(common.WORK, 'stems6')  # test, float32
SHARED_STEMS6_DIR = os.path.join(common.DATA, 'stems6')  # own/fma, float16
FRAMES_DIR = os.path.join(common.WORK, 'teach', 'bp-frames')
SCRATCH = os.path.join(common.WORK, 'teach', 'tmp')
LOG_DIR = os.path.join(common.WORK, 'teach', 'logs')
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(SCRATCH, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def savez_atomic(path, **arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    ct.atomic_write_bytes(path, buf.getvalue())


def infer_one(mono22, model_path):
    tmp_wav = os.path.join(SCRATCH, f'bp_{os.getpid()}_{time.time_ns()}.wav')
    sf.write(tmp_wav, mono22, 22050, subtype='FLOAT')
    try:
        notes, frames = bp_core.predict_wav(tmp_wav, model_path)
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
    return notes, frames


def load_stem_test(tid, stem):
    if stem == 'mix':
        return None  # handled by caller via common.pcm22
    p = os.path.join(STEMS6_DIR, f'{tid}.{stem}.f32')
    if not os.path.exists(p):
        return None
    return np.fromfile(p, dtype='<f4')


def load_stem_shared(corpus, tid, stem):
    p = os.path.join(SHARED_STEMS6_DIR, corpus, f'{tid}.{stem}.f16')
    if not os.path.exists(p):
        return None
    return np.fromfile(p, dtype='<f2').astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--corpus', default=None)
    ap.add_argument('--ids', default=None)
    ap.add_argument('--shard', default=None, help='i/N: process only tracks where hash(id) % N == i')
    args = ap.parse_args()

    shard_i = shard_n = None
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split('/'))

    model_path = bp_core.get_model_path()
    tag = f'shard{shard_i}' if args.shard else 'main'
    log_path = os.path.join(LOG_DIR, f'basicpitch-{tag}.log')
    ct.log_line(log_path, f'model={model_path} corpus={args.corpus} shard={args.shard}')

    done = 0
    total_audio_s = 0.0
    total_wall_s = 0.0

    for corpus, tid, path in ct.ordered_plan():
        if args.corpus and corpus != args.corpus:
            continue
        if args.ids and tid not in args.ids.split(','):
            continue
        if shard_n and (stable_hash(tid) % shard_n) != shard_i:
            continue

        out_path = common.label_path('basicpitch', corpus, tid, 'json')
        is_test = corpus == 'test'
        stems_needed = STEMS + (['mix'] if is_test else [])
        frame_paths = {s: os.path.join(FRAMES_DIR, f'{tid}.{s}.npz') for s in stems_needed} if is_test else {}

        if is_test:
            stem_audio = {}
            for s in STEMS:
                a = load_stem_test(tid, s)
                if a is None:
                    stem_audio = None
                    break
                stem_audio[s] = a
            if stem_audio is None:
                continue  # stems not ready yet
            stem_audio['mix'] = common.pcm22('test', tid, path)
        else:
            stem_audio = {}
            ready = True
            for s in STEMS:
                a = load_stem_shared(corpus, tid, s)
                if a is None:
                    ready = False
                    break
                stem_audio[s] = a
            if not ready:
                continue  # demucs hasn't separated this track yet

        if os.path.exists(out_path) and (not is_test or all(os.path.exists(p) for p in frame_paths.values())):
            continue

        t0 = time.time()
        audio_s = stem_audio['mix' if is_test else 'vocals'].shape[0] / 22050.0
        result = {}
        for stem, audio in stem_audio.items():
            try:
                notes, frames = infer_one(audio, model_path)
            except Exception as e:
                ct.log_line(log_path, f'FAIL {corpus}/{tid}/{stem}: {e}')
                continue
            result[stem] = notes
            if is_test:
                fp = frame_paths[stem]
                if not os.path.exists(fp):
                    savez_atomic(fp, onset=frames['onset'], note=frames['note'])

        ct.atomic_write_text(out_path, json.dumps(result))

        dt = time.time() - t0
        total_audio_s += audio_s
        total_wall_s += dt
        done += 1
        rate = (total_audio_s / 60.0) / (total_wall_s / 60.0) if total_wall_s > 0 else 0
        n_notes = {k: len(v) for k, v in result.items()}
        ct.log_line(log_path, f'{corpus}/{tid} audio={audio_s:.1f}s wall={dt:.1f}s notes={n_notes} cum_ratio={rate:.2f}')

        if args.limit and done >= args.limit:
            break

    ct.log_line(log_path, f'PASS DONE processed={done} total_audio_min={total_audio_s/60:.2f} total_wall_min={total_wall_s/60:.2f}')


if __name__ == '__main__':
    main()
