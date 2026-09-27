"""Own teacher run (used where the shared labels/demucs6 cache lacks a track): htdemucs_6s on MPS.
Writes .testdata/instr/presence/d6/<corpus>/<id>.npz with float16 RMS envelopes per stem + mix,
CAUSAL 2048-sample window ending at the end of each 512-sample hop at 44.1 kHz (the same
definition as the beat agent's stems-eval reference), and mono 22050 Hz float16 stems for
tagging (other, vocals, guitar, piano) at .testdata/instr/presence/d6/<corpus>/<id>.<stem>.f16.
  .testdata/instr/venvs/demucs/bin/python run_demucs6.py <corpus> <listfile> [--max N] [--seconds S]
FMA clips are 30 s; own tracks may be cropped to --seconds (default whole track).
"""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import common as C
from demucs.pretrained import get_model
from demucs.apply import apply_model
from scipy.signal import resample_poly

torch.set_num_threads(6)
SR, HOP, WIN = 44100, 512, 2048
corpus, lst = sys.argv[1], sys.argv[2]
MAXN = int(sys.argv[sys.argv.index('--max') + 1]) if '--max' in sys.argv else 10 ** 9
SECS = float(sys.argv[sys.argv.index('--seconds') + 1]) if '--seconds' in sys.argv else None
OUT = os.path.join(C.WORK, 'presence', 'd6', corpus)
os.makedirs(OUT, exist_ok=True)
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = get_model('htdemucs_6s').to(dev).eval()
names = model.sources  # drums bass other vocals guitar piano


def env(m):
    n = len(m)
    T = n // HOP
    c = np.concatenate([[0.0], np.cumsum(m.astype(np.float64) ** 2)])
    ends = (np.arange(T) + 1) * HOP
    st = np.maximum(0, ends - WIN)
    return np.sqrt((c[ends] - c[st]) / WIN).astype(np.float32)


done = 0
for line in open(lst).read().split('\n'):
    if not line or done >= MAXN:
        continue
    tid, path = line.split('=', 1)
    op = os.path.join(OUT, tid + '.npz')
    if os.path.exists(op):
        done += 1
        continue
    if os.path.exists(os.path.join(C.LABELS, 'demucs6', corpus, tid + '.npz')):
        continue  # the shared teacher cache has it: never separate twice
    t0 = time.perf_counter()
    try:
        x = C.decode(path, SR, 2)
    except Exception as e:
        print('decode fail', tid, e); continue
    if SECS:
        x = x[:int(SECS * SR)]
    wav = torch.from_numpy(x.T.copy()).float()
    ref = wav.mean(0)
    mu, sd = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        s = apply_model(model, ((wav - mu) / sd)[None].to(dev), shifts=0, split=True, overlap=0.25, progress=False, device=dev)[0]
    s = (s * sd + mu).cpu().numpy()  # [6, 2, n]
    mono = s.mean(1)
    out = {'sr': SR, 'hop': HOP, 'mix': env(x.mean(1)).astype(np.float16)}
    for i, nm in enumerate(names):
        out[nm] = env(mono[i]).astype(np.float16)
    np.savez_compressed(op, **out)
    for nm in ['other', 'vocals', 'guitar', 'piano']:
        y = resample_poly(mono[names.index(nm)], 1, 2).astype(np.float16)
        y.tofile(os.path.join(OUT, f'{tid}.{nm}.f16'))
    done += 1
    print(tid, f'{len(x) / SR:.0f}s audio in {time.perf_counter() - t0:.1f}s', flush=True)
