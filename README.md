# CodecSlime

[![Paper](https://img.shields.io/badge/arXiv-2506.21074-b31b1b.svg)](https://arxiv.org/abs/2506.21074)
[![Checkpoints](https://img.shields.io/badge/Checkpoints-Download-2ea44f.svg)](https://github.com/X-LANCE/codecslime/releases/tag/v0.1.0)
[![Demo Page](https://img.shields.io/badge/Demo-Audio_Samples-6f42c1.svg)](https://x-lance.github.io/codecslime/)
[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE)
[![Model License: CC BY 4.0](https://img.shields.io/badge/Weights-CC_BY_4.0-lightgrey.svg)](MODEL_LICENSE)

Official inference implementation and model checkpoints for **CodecSlime: Temporal Redundancy Compression of Neural Speech Codec via Dynamic Frame Rate**, accepted at **ICASSP 2026**.

CodecSlime removes temporal redundancy from neural speech codec representations. Its offline ScheDFR algorithm uses dynamic programming to merge locally redundant encoder frames, allowing one checkpoint to operate at any target frame rate from 36 to 80 Hz. This repository contains waveform encoding, token decoding, reconstruction, and the two paper model variants. Training, evaluation pipelines, and streaming code are not included.

## Models

| Model | Quantizer | Vocabulary | Default rate | Approx. bitrate | WER | STOI | PESQ | SECS | ViSQOL | UTMOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CodecSlime-VQ8k | VQ | 8,192 | 40 Hz | 520 bps | 4.25 | .903 | 2.02 | .935 | 3.96 | 4.02 |
| CodecSlime-FSQ18k | FSQ | 18,225 | 40 Hz | 600 bps | 3.80 | .904 | 2.00 | .918 | 3.91 | 4.05 |

The table reports the 40 Hz LibriSpeech test-clean results from the paper. Both checkpoints are available in the [v0.1.0 release](https://github.com/X-LANCE/codecslime/releases/tag/v0.1.0).

## Installation

Python 3.10+ and PyTorch 2.1+ are supported.

```bash
git clone https://github.com/X-LANCE/codecslime.git
cd CodecSlime
pip install -e .
```

The first `from_pretrained` call downloads the selected checkpoint from the `v0.1.0` release into `~/.cache/codecslime`. You can also download the assets manually and pass `--checkpoint`.

## Quick start

Reconstruct a 16 kHz waveform (other input sample rates are resampled automatically):

```bash
codecslime reconstruct input.wav output.wav --variant vq8k --frame-rate 40
```

Store compact discrete tokens, then decode them later:

```bash
codecslime encode input.wav tokens.npz --variant fsq18k --frame-rate 40
codecslime decode tokens.npz output.wav --variant fsq18k
```

Python API:

```python
from codecslime import CodecSlime
from codecslime.audio import load_audio, save_audio

codec = CodecSlime.from_pretrained("vq8k", device="cuda")
waveform = load_audio("input.wav")
codes, durations, length = codec.encode(waveform, frame_rate=40)
reconstruction = codec.decode(codes, durations, length)
save_audio("output.wav", reconstruction)
```

`codes` has shape `(num_segments, 1)`. `durations` records how many 80 Hz encoder frames each token represents; its sum is the full latent length. The `.npz` format uses numeric arrays only and is loaded with `allow_pickle=False`.

## Checkpoints

| Release asset | Quantizer | Contents |
|---|---|---|
| [`codecslime-vq8k.safetensors`](https://github.com/X-LANCE/codecslime/releases/download/v0.1.0/codecslime-vq8k.safetensors) | 8,192-entry VQ | Encoder, quantizer, and waveform decoder parameters |
| [`codecslime-fsq18k.safetensors`](https://github.com/X-LANCE/codecslime/releases/download/v0.1.0/codecslime-fsq18k.safetensors) | 18,225-entry FSQ | Encoder, quantizer, and waveform decoder parameters |

The release also contains `checksums.txt`. Checkpoints contain only inference parameters—no optimizer, scheduler, global step, trainer state, discriminator, or loss-module state.

## Scope and limitations

- This release supports mono speech inference at 16 kHz and processes one utterance at a time.
- ScheDFR is offline: segmentation uses the full utterance's encoder features.
- The implementation is intended for speech reconstruction research; robustness on music, environmental audio, clipped recordings, or out-of-domain sample rates has not been established.
- Generated/reconstructed audio can preserve or alter speaker traits. Follow applicable consent, privacy, and disclosure requirements.

## Citation

```bibtex
@inproceedings{wang2026codecslime,
  title={CodecSlime: Temporal Redundancy Compression of Neural Speech Codec via Dynamic Frame Rate},
  author={Wang, Hankun and Guo, Yiwei and Shao, Chongtian and Li, Bohan and Yu, Kai},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={17017--17021},
  year={2026},
  publisher={IEEE},
  doi={10.1109/ICASSP55912.2026.11463766}
}
```

## License and acknowledgements

The inference code is released under the [MIT License](LICENSE). Checkpoint weights are released under [CC BY 4.0](MODEL_LICENSE). See [NOTICE](NOTICE) for upstream code attribution. Contributions are welcome through GitHub issues and pull requests.
