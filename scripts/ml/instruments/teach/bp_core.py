"""Core basic-pitch inference helpers (bpitch venv only).

predict_wav(wav_path, model) -> (notes, frames) where
  notes: [[onset_s, offset_s, midi, amplitude], ...]
  frames: {'onset': float32 [T,88], 'note': float32 [T,88]} model output posteriors
          (frame-level, ANNOTATIONS_FPS ~ 43.07 Hz / hop 256 @ 22050 Hz)
"""
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

_ONNX_MODEL_PATH = os.path.join(os.path.dirname(ICASSP_2022_MODEL_PATH), 'nmp.onnx')


def get_model_path():
    if os.path.exists(_ONNX_MODEL_PATH):
        return _ONNX_MODEL_PATH
    return ICASSP_2022_MODEL_PATH


def predict_wav(wav_path, model_path=None):
    model_path = model_path or get_model_path()
    model_output, midi_data, note_events = predict(wav_path, model_or_model_path=model_path)
    notes = [
        [float(start), float(end), int(pitch), float(amp)]
        for start, end, pitch, amp, _bends in note_events
    ]
    frames = {
        'onset': model_output['onset'].astype(np.float16),
        'note': model_output['note'].astype(np.float16),
    }
    return notes, frames
