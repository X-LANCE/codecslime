"""Small audio I/O helpers."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


def load_audio(path: str | Path, sample_rate: int = 16000) -> torch.Tensor:
    audio, source_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = torch.from_numpy(audio.mean(axis=1))
    if source_rate != sample_rate:
        mono = AF.resample(mono, source_rate, sample_rate)
    return mono


def save_audio(path: str | Path, waveform: torch.Tensor, sample_rate: int = 16000):
    sf.write(path, waveform.detach().float().cpu().numpy(), sample_rate)


def save_tokens(path, codes, durations, *, variant, frame_rate, length):
    np.savez_compressed(
        path,
        codes=codes.detach().cpu().numpy().astype(np.int32),
        durations=durations.detach().cpu().numpy().astype(np.int16),
        variant=np.array(variant),
        frame_rate=np.float32(frame_rate),
        sample_rate=np.int32(16000),
        length=np.int64(length),
    )


def load_tokens(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
