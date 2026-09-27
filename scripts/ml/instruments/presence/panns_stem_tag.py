"""Teacher helper: PANNs Cnn14_DecisionLevelMax framewise tagging of SEPARATED stems (mix tagging is
dominated by genre classes and misses instruments under vocals/drums).
Input: mono 22050 Hz float32 stems <dir>/<id>.<stem>.f32 (e.g. .testdata/instr/stems6 or our own
demucs output), resampled to 32 kHz. Output: .testdata/instr/presence/panns_stem/<corpus>/<id>.<stem>.npz
{probs [T,527] f16 at 10 fps (100 frames/s max-pooled by 10), labels}
  .testdata/instr/venvs/panns/bin/python panns_stem_tag.py <corpus> <stemdir> <id> [<id> ...] [--stems other,guitar,piano,vocals]
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import common as C
from panns_inference import SoundEventDetection
import panns_inference.config as PC
from scipy.signal import resample_poly

torch.set_num_threads(6)
args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith('--') and sys.argv[i - 1] != '--stems']
corpus, stemdir, ids = args[0], args[1], args[2:]
stems = sys.argv[sys.argv.index('--stems') + 1].split(',') if '--stems' in sys.argv else ['other', 'guitar', 'piano', 'vocals']
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
sed = SoundEventDetection(checkpoint_path=os.path.expanduser('~/panns_data/Cnn14_DecisionLevelMax.pth'), device=dev)
if dev == 'mps':  # the wrapper only knows cuda/cpu: move it to MPS by hand
    sed.model.to('mps'); sed.device = 'mps'
OUT = os.path.join(C.WORK, 'presence', 'panns_stem', corpus)
os.makedirs(OUT, exist_ok=True)
if len(ids) == 1 and ids[0].startswith('@'):
    ids = [l.split('=')[0] for l in open(ids[0][1:]).read().split('\n') if l]
for tid in ids:
    for st in stems:
        op = os.path.join(OUT, f'{tid}.{st}.npz')
        if os.path.exists(op) or (st == 'mix' and os.path.exists(os.path.join(C.LABELS, 'panns', corpus, tid + '.npz'))):
            continue
        cands = [(os.path.join(stemdir, f'{tid}.{st}.f32'), '<f4'), (os.path.join(stemdir, f'{tid}.{st}.f16'), '<f2'),
                 (os.path.join(C.WORK, 'presence', 'd6', corpus, f'{tid}.{st}.f16'), '<f2'),
                 (os.path.join(C.DATA, 'stems6', corpus, f'{tid}.{st}.f16'), '<f2'),
                 (os.path.join(C.WORK, 'stems6', f'{tid}.{st}.f32'), '<f4'), (os.path.join(C.PCM22, corpus, f'{tid}.f32') if st == 'mix' else '/nonexistent', '<f4')]
        src = next(((q, dt) for q, dt in cands if os.path.exists(q)), None)
        if src is None:
            continue
        x = np.fromfile(src[0], dtype=src[1]).astype(np.float32)
        x = resample_poly(x, 640, 441).astype(np.float32)
        outs = []
        seg = 32000 * 60
        for b in range(0, len(x), seg):
            xb = x[b:b + seg]
            if len(xb) < 32000:
                xb = np.pad(xb, (0, 32000 - len(xb)))
            fw = sed.inference(xb[None, :])[0]  # [frames at 100 fps, 527]
            outs.append(fw[:int(np.ceil(len(x[b:b + seg]) / 320))])
        fw = np.concatenate(outs)
        T = len(fw) // 10
        p10 = fw[:T * 10].reshape(T, 10, -1).max(1)
        np.savez_compressed(op, probs=p10.astype(np.float16), labels=np.array(PC.labels), fps=10)
        print(tid, st, p10.shape, flush=True)
