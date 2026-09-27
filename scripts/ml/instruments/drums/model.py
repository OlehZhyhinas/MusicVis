"""Small causal student for drum transcription: causal dilated Conv1D stack (left-padded, no
future leakage) feeding a single-layer GRU, multi-label frame output (kick/snare/hat/tom/cymbal).

Causal: every conv is left-padded only (padding=(k-1)*dilation on the left, none on the right),
and the GRU processes frames in order with no bidirectional pass, so output frame k depends only
on input frames <= k (peak picking may add up to 2 frames of look-ahead on top of this).
"""
import torch
import torch.nn as nn

CLASSES = ['kick', 'snare', 'hat', 'tom', 'cymbal']
N_CLASSES = len(CLASSES)


class CausalConv1d(nn.Module):
    def __init__(self, cin, cout, k, dilation=1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation)

    def forward(self, x):  # x: [B, C, T]
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class DrumStudent(nn.Module):
    def __init__(self, in_feat=160, conv_ch=(32, 32), gru_hidden=64, kernel=5):
        super().__init__()
        layers = []
        cin = in_feat
        for i, cout in enumerate(conv_ch):
            layers += [CausalConv1d(cin, cout, kernel, dilation=2 ** i), nn.BatchNorm1d(cout), nn.ReLU()]
            cin = cout
        self.conv = nn.Sequential(*layers)
        self.gru = nn.GRU(cin, gru_hidden, num_layers=1, batch_first=True)
        self.out = nn.Linear(gru_hidden, N_CLASSES)
        self.gru_hidden = gru_hidden

    def forward(self, x, h0=None):  # x: [B, T, in_feat]
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)  # [B, T, C]
        y, h = self.gru(z, h0)
        return self.out(y), h  # logits [B, T, 5]


SIZES = {
    'tiny': dict(conv_ch=(24, 24), gru_hidden=32),
    'small': dict(conv_ch=(32, 32), gru_hidden=64),
    'medium': dict(conv_ch=(48, 64), gru_hidden=96),
    'large': dict(conv_ch=(96, 96), gru_hidden=160),
    'xl': dict(conv_ch=(128, 128), gru_hidden=256),
}


def count_params(m):
    return sum(p.numel() for p in m.parameters())
