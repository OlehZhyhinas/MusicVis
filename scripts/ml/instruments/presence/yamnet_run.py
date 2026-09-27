"""Candidate (a): YAMNet streamed causally. YAMNet frames 0.96 s patches every 0.48 s; score i covers
[i*0.48, i*0.48+0.96] and is stamped at its window END (i*0.48+0.96), i.e. it only uses past audio.
Also times the 4 MB TFLite classifier per 0.975 s window (single thread) as a CPU proxy.
Run with the adtof venv (TF 2.21):
  TF_USE_LEGACY_KERAS=1 .testdata/instr/venvs/adtof/bin/python scripts/ml/instruments/presence/yamnet_run.py <listfile>
Writes .testdata/instr/presence/yamnet/<id>.npz {probs [N,521] f16, t [N] (window-end s), labels}
"""
import os, sys, time, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import common as C
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(4)
PRE = os.path.join(C.WORK, 'presence', 'pretrained')
OUT = os.path.join(C.WORK, 'presence', 'yamnet')
os.makedirs(OUT, exist_ok=True)
labels = [r[2] for r in list(csv.reader(open(os.path.join(PRE, 'yamnet_class_map.csv'))))[1:]]

if sys.argv[1] == '--bench':
    it = tf.lite.Interpreter(os.path.join(PRE, 'yamnet.tflite'), num_threads=1)
    inp = it.get_input_details()[0]
    it.resize_tensor_input(inp['index'], [15600]); it.allocate_tensors()
    x = (np.random.randn(15600) * 0.1).astype(np.float32)
    for _ in range(5):
        it.set_tensor(inp['index'], x); it.invoke()
    t0 = time.perf_counter(); n = 50
    for _ in range(n):
        it.set_tensor(inp['index'], x); it.invoke()
    ms = (time.perf_counter() - t0) / n * 1000
    print(f'tflite yamnet: {ms:.2f} ms per 0.975 s window (1 thread); at hop 0.48 s = {ms / 0.48:.1f} ms per s audio;',
          'input', inp['dtype'], 'size MB', os.path.getsize(os.path.join(PRE, 'yamnet.tflite')) / 1e6)
    sys.exit()

model = tf.saved_model.load(os.path.join(PRE, 'yamnet_sm'))
for line in open(sys.argv[1]).read().split('\n'):
    if not line:
        continue
    tid, path = line.split('=', 1)
    op = os.path.join(OUT, tid + '.npz')
    if os.path.exists(op):
        continue
    x = C.decode(path, 16000, 1)
    t0 = time.perf_counter()
    scores, emb, spec = model(x)
    scores = scores.numpy()
    dt = time.perf_counter() - t0
    t = np.arange(len(scores)) * 0.48 + 0.96
    np.savez_compressed(op, probs=scores.astype(np.float16), t=t.astype(np.float32), labels=np.array(labels))
    print(tid, scores.shape, f'{dt / (len(x) / 16000) * 1000:.1f} ms/s', flush=True)
