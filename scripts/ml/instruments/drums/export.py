"""Export a trained DrumStudent (2 causal conv layers + 1 GRU layer + linear head) to a compact
binary for the plain-TS forward pass: BatchNorm folded into the preceding conv (inference-only),
weights as float32 (a float16 cast is applied on the TS side's .bin if smaller size is required;
this repo's weights are already small enough to ship as float32 raw, see the size report).

Writes .testdata/instr/drums/models/<size>.bin (raw float32, concatenated) and
.testdata/instr/drums/models/<size>.header.json (shapes + offsets, so the TS loader knows how to
slice the buffer).

Usage: <venv>/bin/python export.py <model-tag>   (reads models/<tag>.pt and <tag>.json for the size)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch
from model import DrumStudent, SIZES, CLASSES

WORK = os.path.join(os.path.dirname(__file__), '../../../../.testdata/instr/drums')
MODELS_DIR = os.path.join(WORK, 'models')


def fold_bn(conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
    scale = bn_w / np.sqrt(bn_var + eps)
    w2 = conv_w * scale[:, None, None]
    b2 = (conv_b - bn_mean) * scale + bn_b
    return w2.astype(np.float32), b2.astype(np.float32)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'small'
    info_p = os.path.join(MODELS_DIR, f'{tag}.json')
    size = json.load(open(info_p)).get('size', tag) if os.path.exists(info_p) else tag
    model = DrumStudent(in_feat=160, **SIZES[size])
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f'{tag}.pt'), map_location='cpu'))
    model.eval()
    sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}

    # conv.0 = CausalConv1d.conv (layer0), conv.1 = BatchNorm1d, conv.3 = CausalConv1d.conv (layer1), conv.4 = BatchNorm1d
    w0, b0 = fold_bn(sd['conv.0.conv.weight'], sd['conv.0.conv.bias'], sd['conv.1.weight'], sd['conv.1.bias'], sd['conv.1.running_mean'], sd['conv.1.running_var'])
    w1, b1 = fold_bn(sd['conv.3.conv.weight'], sd['conv.3.conv.bias'], sd['conv.4.weight'], sd['conv.4.bias'], sd['conv.4.running_mean'], sd['conv.4.running_var'])

    gru_wih = sd['gru.weight_ih_l0']  # [3H, C1]
    gru_whh = sd['gru.weight_hh_l0']  # [3H, H]
    gru_bih = sd['gru.bias_ih_l0']
    gru_bhh = sd['gru.bias_hh_l0']
    lin_w = sd['out.weight']  # [5, H]
    lin_b = sd['out.bias']

    conv_ch = SIZES[size]['conv_ch']
    gru_hidden = SIZES[size]['gru_hidden']
    kernel = 5
    dilations = [2 ** i for i in range(len(conv_ch))]

    arrays = [w0, b0, w1, b1, gru_wih, gru_whh, gru_bih, gru_bhh, lin_w, lin_b]
    names = ['conv0_w', 'conv0_b', 'conv1_w', 'conv1_b', 'gru_wih', 'gru_whh', 'gru_bih', 'gru_bhh', 'lin_w', 'lin_b']
    offsets = []
    off = 0
    flat = []
    for name, a in zip(names, arrays):
        a = np.ascontiguousarray(a.astype(np.float32))
        flat.append(a.tobytes())
        offsets.append({'name': name, 'shape': list(a.shape), 'offset': off, 'count': int(a.size)})
        off += a.size * 4

    header = {
        'size': size,
        'tag': tag,
        'in_feat': 160,
        'conv_ch': list(conv_ch),
        'kernel': kernel,
        'dilations': dilations,
        'gru_hidden': gru_hidden,
        'n_classes': len(CLASSES),
        'classes': CLASSES,
        'tensors': offsets,
        'total_bytes': off,
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, f'{tag}.bin'), 'wb') as f:
        f.write(b''.join(flat))
    with open(os.path.join(MODELS_DIR, f'{tag}.header.json'), 'w') as f:
        json.dump(header, f, indent=1)
    print(f'{tag} ({size}): {off} bytes ({off / 1024:.1f} KB) -> {MODELS_DIR}/{tag}.bin')


if __name__ == '__main__':
    main()
