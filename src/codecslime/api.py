"""Public Python API for CodecSlime."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .nn.models import CodecSlimeNet
from .segmentation import average_segments, segment


RELEASE_BASE = "https://github.com/X-LANCE/codecslime/releases/download/v0.1.0"
VARIANTS = {
    "vq8k": "codecslime-vq8k.safetensors",
    "fsq18k": "codecslime-fsq18k.safetensors",
}


class CodecSlime:
    sample_rate = 16000
    encoder_frame_rate = 80

    def __init__(self, variant: str, checkpoint: str | Path, device: str | torch.device = "cpu"):
        self.variant = variant.lower()
        self.device = torch.device(device)
        self.model = CodecSlimeNet(self.variant)
        self.model.load_state_dict(load_file(str(checkpoint), device=str(self.device)))
        self.model.to(self.device).eval()

    @classmethod
    def from_pretrained(cls, variant="vq8k", device="cpu", cache_dir=None):
        variant = variant.lower()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
        cache = Path(cache_dir or Path.home() / ".cache" / "codecslime")
        cache.mkdir(parents=True, exist_ok=True)
        destination = cache / VARIANTS[variant]
        if not destination.exists():
            with urlopen(f"{RELEASE_BASE}/{destination.name}") as response, destination.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        return cls(variant, destination, device=device)

    @torch.inference_mode()
    def encode(self, waveform: torch.Tensor, frame_rate: float = 40.0):
        waveform = waveform.to(self.device).float().flatten()
        original_length = waveform.numel()
        padded_length = ((original_length + 199) // 200) * 200
        waveform = F.pad(waveform, (0, padded_length - original_length))[None, None]
        features = self.model.encoder(waveform)
        durations = segment(features, frame_rate)
        compact = average_segments(features, durations)
        _, codes = self.model.decoder.quantize(compact)
        if codes.shape[0] == 1:
            codes = codes.squeeze(0)
        return codes.long(), durations, original_length

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor, durations: torch.Tensor, length: int | None = None):
        codes = torch.as_tensor(codes, device=self.device, dtype=torch.long)
        durations = torch.as_tensor(durations, device=self.device, dtype=torch.long)
        if codes.ndim == 1:
            codes = codes[:, None]
        embeddings = self.model.decoder.codes_to_embeddings(codes[None])
        expanded = embeddings.repeat_interleave(durations, dim=-1)
        waveform = self.model.decoder.decode_embeddings(expanded).flatten()
        return waveform[:length] if length is not None else waveform

    @torch.inference_mode()
    def reconstruct(self, waveform: torch.Tensor, frame_rate: float = 40.0):
        codes, durations, length = self.encode(waveform, frame_rate)
        return self.decode(codes, durations, length)
