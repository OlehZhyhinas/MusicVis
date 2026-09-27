#!/usr/bin/env .testdata/instr/venvs/adtof/bin/python
"""Teacher 4: ADTOF drum onsets on the full mix.

For the TEST corpus, just copies the 13 pre-computed files from
.testdata/instr/teach/adtof-mix/<slug>.json (never recomputed).

For own/fma, runs ADTOF's bundled Frame_RNN (scenario='adtofAll', fold=0)
model on the full mix. predictFolder wants a WAV file path, so a temp mono
44.1 kHz WAV is written to .testdata/instr/teach/tmp/ and deleted afterwards.
Output format matches the existing adtof-mix files exactly: keys model,
slug, beats, downbeats, kick, snare, hat, other, notes, causal,
cpu_time_total_s.

Runs as its own background process, in parallel with demucs6. Do own/fma
before demucs6 reaches them (drum agent needs this first).

Run: .testdata/instr/venvs/adtof/bin/python scripts/ml/instruments/teach/run_adtof.py [--limit N] [--corpus own]
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import common
import common_teach as ct

KICK = 35
SNARE = 38
HAT = 42

TEST_PRECOMPUTED = os.path.join(common.WORK, 'teach', 'adtof-mix')
TMP_DIR = os.path.join(common.WORK, 'teach', 'tmp')
RAW_DIR = os.path.join(common.WORK, 'teach', 'adtof-raw')
LOG_PATH = os.path.join(common.WORK, 'teach', 'logs', 'adtof.log')
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def parse_txt(path):
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t, pitch = line.split('\t')
            events.append((float(t), int(pitch)))
    return events


def copy_test_precomputed():
    n = 0
    for f in sorted(os.listdir(TEST_PRECOMPUTED)):
        if not f.endswith('.json') or f == '_summary.json':
            continue
        slug = f[:-5]
        out_path = common.label_path('adtof', 'test', slug, 'json')
        if os.path.exists(out_path):
            continue
        with open(os.path.join(TEST_PRECOMPUTED, f)) as fh:
            data = fh.read()
        ct.atomic_write_text(out_path, data)
        n += 1
    ct.log_line(LOG_PATH, f'copied {n} precomputed test files')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--corpus', default=None)
    ap.add_argument('--ids', default=None)
    ap.add_argument('--skip-test-copy', action='store_true')
    args = ap.parse_args()

    if not args.skip_test_copy and (args.corpus is None or args.corpus == 'test'):
        copy_test_precomputed()
        if args.corpus == 'test':
            return

    from adtof.model.model import Model
    model, hparams = Model.modelFactory(modelName='Frame_RNN', scenario='adtofAll', fold=0)
    assert model.weightLoadedFlag
    ct.log_line(LOG_PATH, 'model loaded: Frame_RNN adtofAll fold0')

    done = 0
    total_audio_s = 0.0
    total_wall_s = 0.0

    for corpus, tid, path in ct.ordered_plan():
        if corpus == 'test':
            continue  # handled by copy above
        if args.corpus and corpus != args.corpus:
            continue
        if args.ids and tid not in args.ids.split(','):
            continue
        out_path = common.label_path('adtof', corpus, tid, 'json')
        if os.path.exists(out_path):
            continue

        t0 = time.time()
        try:
            mono = common.pcm22(corpus, tid, path) if False else common.decode(path, 44100, 1)
        except Exception as e:
            ct.log_line(LOG_PATH, f'FAIL decode {corpus}/{tid}: {e}')
            continue
        audio_s = mono.shape[0] / 44100.0

        wav_path = os.path.join(TMP_DIR, f'{tid}.wav')
        try:
            sf.write(wav_path, mono, 44100, subtype='FLOAT')
            model.predictFolder(wav_path, RAW_DIR, **hparams)
            candidates = glob.glob(os.path.join(RAW_DIR, f'{tid}*.txt'))
            if not candidates:
                ct.log_line(LOG_PATH, f'FAIL {corpus}/{tid}: no output txt produced')
                continue
            txt_path = sorted(candidates, key=os.path.getmtime)[-1]
            events = parse_txt(txt_path)
        except Exception as e:
            ct.log_line(LOG_PATH, f'FAIL predict {corpus}/{tid}: {e}')
            continue
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

        kick = sorted(t for t, p in events if p == KICK)
        snare = sorted(t for t, p in events if p == SNARE)
        hat = sorted(t for t, p in events if p == HAT)
        other = sorted(t for t, p in events if p not in (KICK, SNARE, HAT))

        result = {
            'model': 'adtof-Frame_RNN-adtofAll-fold0',
            'slug': tid,
            'beats': [],
            'downbeats': [],
            'kick': kick,
            'snare': snare,
            'hat': hat,
            'other': other,
            'notes': (
                "ADTOF drum transcription (bundled pretrained Frame_RNN CRNN "
                "model, scenario='adtofAll', fold=0) via "
                "Model.predictFolder/peak-picking. Not a beat tracker: beats/"
                "downbeats left empty. Output classes (MIDI pitches) "
                "[35,38,47,42,49]; 35->kick, 38->snare, 42->hat, 47 (tom) and "
                "49 (crash/ride) grouped under 'other'."
            ),
            'causal': False,
            'cpu_time_total_s': time.time() - t0,
        }
        ct.atomic_write_text(out_path, json.dumps(result, indent=2))

        dt = time.time() - t0
        total_audio_s += audio_s
        total_wall_s += dt
        done += 1
        rate = (total_audio_s / 60.0) / (total_wall_s / 60.0) if total_wall_s > 0 else 0
        ct.log_line(LOG_PATH, f'{corpus}/{tid} audio={audio_s:.1f}s wall={dt:.1f}s kick={len(kick)} snare={len(snare)} hat={len(hat)} cum_ratio={rate:.2f}')

        if args.limit and done >= args.limit:
            break

    ct.log_line(LOG_PATH, f'PASS DONE processed={done} total_audio_min={total_audio_s/60:.2f} total_wall_min={total_wall_s/60:.2f}')


if __name__ == '__main__':
    main()
