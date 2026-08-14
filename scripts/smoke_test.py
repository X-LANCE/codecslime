"""Load a release checkpoint and exercise representative frame rates."""

import sys
import torch

from codecslime import CodecSlime


variant, checkpoint = sys.argv[1:3]
samples = int(sys.argv[3]) if len(sys.argv) > 3 else 16000
device = "cuda" if torch.cuda.is_available() else "cpu"
codec = CodecSlime(variant, checkpoint, device=device)
waveform = torch.randn(samples, device=device) * 0.01
frame_rates = (36, 40, 67, 80) if samples >= 16000 else (40,)
for frame_rate in frame_rates:
    codes, durations, length = codec.encode(waveform, frame_rate)
    reconstruction = codec.decode(codes, durations, length)
    assert reconstruction.shape == waveform.shape
    assert durations.sum().item() == (samples + 199) // 200
    assert torch.isfinite(reconstruction).all()
    print(variant, frame_rate, tuple(codes.shape), tuple(reconstruction.shape))
