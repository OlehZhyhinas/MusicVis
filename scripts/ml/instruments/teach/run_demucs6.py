#!/usr/bin/env .testdata/instr/venvs/demucs/bin/python
"""Teacher 1: htdemucs_6s stem separation -> per-stem causal RMS envelopes.

Envelope definition (same as .testdata/ml/stems/run_demucs.py, vectorised):
  value k = RMS of the stem's mono mix over samples [(k+1)*512-2048, (k+1)*512)
  at 44.1 kHz, zero-padded before the start. hop=512, win=2048.

Output: ~/personal/MusicVis-data/labels/demucs6/<corpus>/<id>.npz
  keys: drums, bass, other, vocals, guitar, piano, mix (float16 [n_hops]),
        sr=44100, hop=512

For the TEST corpus, also writes separated stems as mono 22050 Hz float32 to
.testdata/instr/stems6/<slug>.<stem>.f32 (for the note/instrument agents).
For own/fma, writes stems as mono 22050 Hz float16 to the shared
~/personal/MusicVis-data/stems6/<corpus>/<id>.<stem>.f16 (used by the
basic-pitch teacher and the presence/note agents, so demucs never runs
twice for the same track).

Run: .testdata/instr/venvs/demucs/bin/python scripts/ml/instruments/teach/run_demucs6.py [--limit N] [--corpus test]
"""
import argparse
import io
import os
import shutil
import sys
import time

import shutil

import numpy as np
import torch

MIN_FREE_GB = 30  # pause writing (large) shared stem caches below this; envelopes/labels always continue

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import common
import common_teach as ct

SR = 44100
HOP = 512
WIN = 2048
STEMS = ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano']

MIN_FREE = 30 << 30  # stop caching stem audio below 30 GB free (envelopes still written)
STEMS6_DIR = os.path.join(common.WORK, 'stems6')  # test only, float32 (note/instrument agents)
SHARED_STEMS6_DIR = os.path.join(common.DATA, 'stems6')  # own/fma, float16 (shared with other agents + basicpitch)
LOG_PATH = os.path.join(common.WORK, 'teach', 'logs', 'demucs6.log')
os.makedirs(STEMS6_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

torch.set_num_threads(6)


def causal_env(mono):
    """Vectorised causal RMS envelope, hop=512 win=2048, zero-padded front."""
    n = mono.shape[0]
    n_hops = (n + HOP - 1) // HOP
    padded_len = n_hops * HOP
    x = np.zeros(padded_len, dtype=np.float64)
    x[:n] = mono
    block_ss = (x.reshape(n_hops, HOP) ** 2).sum(axis=1)  # sum of squares per 512-block
    bp = np.concatenate([np.zeros(3, dtype=np.float64), block_ss])
    windows = np.lib.stride_tricks.sliding_window_view(bp, 4)  # [n_hops, 4]
    window_sum = windows.sum(axis=1)
    env = np.sqrt(window_sum / WIN)
    return env.astype(np.float32)


def stereo_to_mono(stereo):
    return stereo.mean(axis=1)


def load_stereo44(corpus, tid, path):
    if corpus == 'test':
        cached = os.path.join(common.TESTDATA, 'live', 'pcm', f'{tid}.f32')
        if os.path.exists(cached):
            raw = np.fromfile(cached, dtype='<f4')
            if raw.size % 2:
                raw = raw[:-1]
            return raw.reshape(-1, 2)
    return common.decode(path, SR, 2)


def resample_2x_down(mono_f32):
    """44100 -> 22050 via polyphase resampling (exact 2:1 ratio)."""
    from scipy.signal import resample_poly
    return resample_poly(mono_f32, up=1, down=2).astype(np.float32)


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def savez_atomic(path, **arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    ct.atomic_write_bytes(path, buf.getvalue())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--corpus', default=None, help='restrict to one corpus (for verification runs)')
    ap.add_argument('--ids', default=None, help='comma-separated ids to force (verification)')
    args = ap.parse_args()

    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    ct.log_line(LOG_PATH, f'device={device}')

    model = get_model('htdemucs_6s')
    model.eval()
    model.to(device)
    assert model.sources == STEMS, model.sources
    n_params = sum(p.numel() for p in model.parameters())
    ct.log_line(LOG_PATH, f'model=htdemucs_6s params={n_params/1e6:.2f}M sources={model.sources} device={device}')

    done = 0
    total_audio_s = 0.0
    total_wall_s = 0.0

    plan = ct.ordered_plan()
    for corpus, tid, path in plan:
        if args.corpus and corpus != args.corpus:
            continue
        if args.ids and tid not in args.ids.split(','):
            continue
        out_path = common.label_path('demucs6', corpus, tid, 'npz')
        if corpus == 'test':
            stems_paths = [os.path.join(STEMS6_DIR, f'{tid}.{s}.f32') for s in STEMS]
        else:
            shared_dir = os.path.join(SHARED_STEMS6_DIR, corpus)
            stems_paths = [os.path.join(shared_dir, f'{tid}.{s}.f16') for s in STEMS]
        stems_done = all(os.path.exists(p) for p in stems_paths)
        low_disk = shutil.disk_usage(common.DATA).free < MIN_FREE
        if os.path.exists(out_path) and (stems_done or low_disk):
            continue

        t0 = time.time()
        try:
            stereo = load_stereo44(corpus, tid, path)
        except Exception as e:
            ct.log_line(LOG_PATH, f'FAIL decode {corpus}/{tid}: {e}')
            continue
        audio_s = stereo.shape[0] / SR
        mix_mono = stereo_to_mono(stereo)

        wav = torch.from_numpy(np.ascontiguousarray(stereo.T)).float().unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                out = apply_model(model, wav, device=device, progress=False, num_workers=0)
        except Exception as e:
            ct.log_line(LOG_PATH, f'FAIL separate {corpus}/{tid}: {e}')
            continue
        out = out[0].cpu().numpy()  # (n_sources, 2, n)

        envs = {}
        stem_monos = {}
        for si, stem in enumerate(model.sources):
            mono = out[si].mean(axis=0)
            stem_monos[stem] = mono
            envs[stem] = causal_env(mono).astype(np.float16)
        envs['mix'] = causal_env(mix_mono).astype(np.float16)

        savez_atomic(out_path, sr=np.int32(SR), hop=np.int32(HOP), **envs)

        if corpus == 'test':
            for stem, mono in stem_monos.items():
                p = os.path.join(STEMS6_DIR, f'{tid}.{stem}.f32')
                if not os.path.exists(p):
                    m22 = resample_2x_down(mono)
                    tmp = p + '.tmp'
                    m22.tofile(tmp)
                    os.replace(tmp, p)
        elif low_disk:
            ct.log_line(LOG_PATH, f'LOW DISK (<{MIN_FREE >> 30} GB free): envelopes only, no stems for {corpus}/{tid}')
        else:
            fgb = free_gb(common.DATA)
            if fgb < MIN_FREE_GB:
                ct.log_line(LOG_PATH, f'LOW DISK {fgb:.1f}GB free < {MIN_FREE_GB}GB: skipping stems6 write for {corpus}/{tid} (envelope still saved)')
            else:
                shared_dir = os.path.join(SHARED_STEMS6_DIR, corpus)
                os.makedirs(shared_dir, exist_ok=True)
                for stem, mono in stem_monos.items():
                    p = os.path.join(shared_dir, f'{tid}.{stem}.f16')
                    if not os.path.exists(p):
                        m22 = resample_2x_down(mono).astype(np.float16)
                        tmp = p + '.tmp'
                        m22.tofile(tmp)
                        os.replace(tmp, p)

        dt = time.time() - t0
        total_audio_s += audio_s
        total_wall_s += dt
        done += 1
        rate = (total_audio_s / 60.0) / (total_wall_s / 60.0) if total_wall_s > 0 else 0
        ct.log_line(LOG_PATH, f'{corpus}/{tid} audio={audio_s:.1f}s wall={dt:.1f}s cum_ratio_audiomin_per_wallmin={rate:.2f}')

        if args.limit and done >= args.limit:
            break

    ct.log_line(LOG_PATH, f'PASS DONE processed={done} total_audio_min={total_audio_s/60:.2f} total_wall_min={total_wall_s/60:.2f}')


if __name__ == '__main__':
    main()
