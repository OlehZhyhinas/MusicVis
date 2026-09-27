"""Run ADTOF (Frame_RNN, adtofAll, fold0) on the 'own' corpus (and optionally 'fma'),
writing ~/personal/MusicVis-data/labels/adtof/<corpus>/<id>.json in the same format as
.testdata/instr/teach/adtof-mix/<slug>.json (kick/snare/hat/other onset times in seconds,
beats/downbeats left empty since ADTOF is not a beat tracker). Atomic writes (tmp + os.replace).

Run with the adtof venv:
  .testdata/instr/venvs/adtof/bin/python scripts/ml/instruments/drums/teach_adtof_own.py [own|fma] [--limit N]
"""
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

import common

from adtof.model.model import Model

KICK, SNARE, HAT, TOM, CYMBAL = 35, 38, 42, 47, 49


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


def atomic_write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else 'own'
    limit = None
    if '--limit' in sys.argv:
        limit = int(sys.argv[sys.argv.index('--limit') + 1])

    wav_dir = os.path.join(common.WORK, f'wav-{corpus}')
    os.makedirs(wav_dir, exist_ok=True)
    out_dir = os.path.join(common.LABELS, 'adtof', corpus)
    raw_dir = os.path.join(out_dir, '_raw')
    os.makedirs(raw_dir, exist_ok=True)

    tracks = common.tracks(corpus)
    if limit:
        tracks = tracks[:limit]
    tracks = [(tid, path) for tid, path in tracks if not os.path.exists(os.path.join(out_dir, f'{tid}.json'))]
    print(f'{corpus}: {len(tracks)} tracks left to label')

    # decode whatever is missing
    for tid, path in tracks:
        wav_path = os.path.join(wav_dir, tid + '.wav')
        if not os.path.exists(wav_path):
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', path, '-ac', '1', '-ar', '44100', wav_path], check=True)
    print('decoded.')

    model, hparams = Model.modelFactory(modelName='Frame_RNN', scenario='adtofAll', fold=0)
    assert model.weightLoadedFlag

    done = failed = 0
    total_t = 0.0
    for tid, path in tracks:
        wav_path = os.path.join(wav_dir, tid + '.wav')
        t0 = time.time()
        try:
            model.predictFolder(wav_path, raw_dir, **{**hparams, 'writeMidi': False})
        except Exception as e:
            print(tid, 'FAILED', repr(e))
            failed += 1
            continue
        t1 = time.time()
        total_t += t1 - t0
        candidates = glob.glob(os.path.join(raw_dir, f'{tid}*.txt'))
        if not candidates:
            print(tid, 'FAILED: no output txt')
            failed += 1
            continue
        txt_path = sorted(candidates, key=os.path.getmtime)[-1]
        events = parse_txt(txt_path)
        kick = sorted(t for t, p in events if p == KICK)
        snare = sorted(t for t, p in events if p == SNARE)
        hat = sorted(t for t, p in events if p == HAT)
        tom = sorted(t for t, p in events if p == TOM)
        cymbal = sorted(t for t, p in events if p == CYMBAL)
        other = sorted(t for t, p in events if p not in (KICK, SNARE, HAT, TOM, CYMBAL))

        result = {
            'model': 'adtof-Frame_RNN-adtofAll-fold0',
            'slug': tid,
            'beats': [],
            'downbeats': [],
            'kick': kick,
            'snare': snare,
            'hat': hat,
            'tom': tom,
            'cymbal': cymbal,
            'other': other,
            'notes': (
                "ADTOF drum transcription (bundled pretrained Frame_RNN CRNN model, "
                "scenario='adtofAll', fold=0). Classes [35,38,47,42,49]; "
                "35->kick, 38->snare, 42->hat, 47->tom, 49->cymbal(crash/ride). Unlike the shared "
                "adtof-mix teacher (which groups 47+49 under 'other'), this drum-transcription "
                "teacher keeps tom and cymbal separate."
            ),
            'causal': False,
            'cpu_time_total_s': t1 - t0,
        }
        atomic_write_json(os.path.join(out_dir, f'{tid}.json'), result)
        done += 1
        print(tid, f'{t1 - t0:.1f}s', 'kick', len(kick), 'snare', len(snare), 'hat', len(hat), 'tom', len(tom), 'cymbal', len(cymbal))

    print(f'{corpus}: done={done} failed={failed} total_predict_s={total_t:.1f}')


if __name__ == '__main__':
    main()
