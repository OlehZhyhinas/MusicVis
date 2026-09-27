"""Dump a torch forward pass reference (activations) for one cached track, for TS/torch parity
checking: <venv>/bin/python dump_torch_ref.py <model-tag> <corpus> <tid>
Writes .testdata/instr/drums/parity/<size>-<corpus>-<tid>.{feat,ref}.f32
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import common
from model import DrumStudent, SIZES

CACHE = os.path.join(common.WORK, 'drums', 'cache')
MODELS_DIR = os.path.join(common.WORK, 'drums', 'models')
OUT = os.path.join(common.WORK, 'drums', 'parity')


def main():
    size, corpus, tid = sys.argv[1], sys.argv[2], sys.argv[3]
    info_p = os.path.join(MODELS_DIR, f'{size}.json')
    arch = json.load(open(info_p)).get('size', size) if os.path.exists(info_p) else size
    model = DrumStudent(in_feat=160, **SIZES[arch])
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f'{size}.pt'), map_location='cpu'))
    model.eval()
    d = np.load(os.path.join(CACHE, corpus, f'{tid}.npz'))
    feat = d['feat'].astype(np.float32)
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(feat).unsqueeze(0))
        act = torch.sigmoid(logits)[0].numpy().astype(np.float32)
    os.makedirs(OUT, exist_ok=True)
    feat.astype(np.float32).tofile(os.path.join(OUT, f'{size}-{corpus}-{tid}.feat.f32'))
    act.tofile(os.path.join(OUT, f'{size}-{corpus}-{tid}.ref.f32'))
    print('T=', feat.shape[0], '-> ', OUT)


if __name__ == '__main__':
    main()
