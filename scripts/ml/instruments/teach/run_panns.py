#!/usr/bin/env .testdata/instr/venvs/panns/bin/python
"""Teacher 3: PANNs Cnn14_DecisionLevelMax sound event detection on the full mix.

32 kHz input, framewise output at 100 fps (hop=320 @ 32kHz), max-pooled down
to 10 frames/second. All 527 AudioSet classes are kept (broad instrument
taxonomy covers vocals, drums/percussion, bass, guitars, keys/synths, brass,
strings, woodwinds, plucked/mallets/bells, FX -- do not drop any classes).

Output: ~/personal/MusicVis-data/labels/panns/<corpus>/<id>.npz
  probs: float16 [T, 527] (max-pooled to 10 fps)
  labels: [527] class names
  fps: 10

Run: .testdata/instr/venvs/panns/bin/python scripts/ml/instruments/teach/run_panns.py [--limit N] [--corpus test]
"""
import argparse
import io
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import common
import common_teach as ct

SR = 32000
NATIVE_FPS = 100  # hop_size=320 @ 32kHz
POOL = 10  # -> 10 fps
LOG_PATH = os.path.join(common.WORK, 'teach', 'logs', 'panns.log')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

torch.set_num_threads(6)


def pool_max(framewise, factor):
    """framewise: [T,C] at NATIVE_FPS -> max-pooled [ceil(T/factor), C]."""
    t = framewise.shape[0]
    n_out = (t + factor - 1) // factor
    pad = n_out * factor - t
    if pad:
        framewise = np.concatenate([framewise, np.full((pad, framewise.shape[1]), -np.inf, dtype=framewise.dtype)], axis=0)
    return framewise.reshape(n_out, factor, framewise.shape[1]).max(axis=1)


def savez_atomic(path, **arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    ct.atomic_write_bytes(path, buf.getvalue())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--corpus', default=None)
    ap.add_argument('--ids', default=None)
    args = ap.parse_args()

    from panns_inference import SoundEventDetection, labels as panns_labels

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    sed = SoundEventDetection(checkpoint_path=os.path.expanduser('~/panns_data/Cnn14_DecisionLevelMax.pth'), device=device)
    ct.log_line(LOG_PATH, f'device={device} classes={len(panns_labels)}')
    labels_arr = np.array(panns_labels)

    done = 0
    total_audio_s = 0.0
    total_wall_s = 0.0

    for corpus, tid, path in ct.ordered_plan():
        if args.corpus and corpus != args.corpus:
            continue
        if args.ids and tid not in args.ids.split(','):
            continue
        out_path = common.label_path('panns', corpus, tid, 'npz')
        if os.path.exists(out_path):
            continue

        t0 = time.time()
        try:
            mono = common.decode(path, SR, 1)
        except Exception as e:
            ct.log_line(LOG_PATH, f'FAIL decode {corpus}/{tid}: {e}')
            continue
        audio_s = mono.shape[0] / SR

        try:
            audio = mono[None, :].astype(np.float32)
            framewise = sed.inference(audio)[0]  # [T, 527]
        except Exception as e:
            ct.log_line(LOG_PATH, f'FAIL infer {corpus}/{tid}: {e}')
            continue

        pooled = pool_max(framewise, POOL).astype(np.float16)
        savez_atomic(out_path, probs=pooled, labels=labels_arr, fps=np.int32(10))

        dt = time.time() - t0
        total_audio_s += audio_s
        total_wall_s += dt
        done += 1
        rate = (total_audio_s / 60.0) / (total_wall_s / 60.0) if total_wall_s > 0 else 0
        ct.log_line(LOG_PATH, f'{corpus}/{tid} audio={audio_s:.1f}s wall={dt:.1f}s T={pooled.shape[0]} cum_ratio={rate:.2f}')

        if args.limit and done >= args.limit:
            break

    ct.log_line(LOG_PATH, f'PASS DONE processed={done} total_audio_min={total_audio_s/60:.2f} total_wall_min={total_wall_s/60:.2f}')


if __name__ == '__main__':
    main()
