"""CodecSlime inference-only model definition."""

from __future__ import annotations

import torch
from torch import nn

from . import activations
from .alias_free import Activation1d
from .layers import DecoderBlock, EncoderBlock, ResLSTM, WNConv1d
from .vq.factorized_vector_quantize import FactorizedVectorQuantize
from .vq.residual_fsq import ResidualFSQ


class ResidualVQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            FactorizedVectorQuantize(
                dim=1024,
                codebook_size=8192,
                codebook_dim=8,
                commitment=0.25,
            )
        ])

    def forward(self, x):
        quantized, indices, loss = self.layers[0](x)
        return quantized, indices.unsqueeze(0), loss.unsqueeze(0)

    def idx2emb(self, indices):
        return self.layers[0].idx2emb(indices[:, :, 0])


class CodecEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = 48
        blocks: list[nn.Module] = [WNConv1d(1, channels, kernel_size=7, padding=3)]
        for stride in (2, 2, 2, 5, 5):
            channels *= 2
            blocks.append(EncoderBlock(channels, stride=stride, dilations=(1, 3, 9)))
        blocks.extend([
            ResLSTM(channels, num_layers=2),
            Activation1d(activations.SnakeBeta(channels, alpha_logscale=True)),
            WNConv1d(channels, 1024, kernel_size=3, padding=1),
        ])
        self.block = nn.Sequential(*blocks)

    def forward(self, waveform):
        return self.block(waveform)


class CodecDecoder(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        if variant == "vq8k":
            self.quantizer = ResidualVQ()
        elif variant == "fsq18k":
            self.quantizer = ResidualFSQ(
                levels=[5, 5, 3, 3, 3, 3, 3, 3],
                num_quantizers=1,
                dim=1024,
                is_channel_first=True,
                quantize_dropout=False,
            )
        else:
            raise ValueError(f"Unknown CodecSlime variant: {variant}")

        channels = 1536
        layers: list[nn.Module] = [
            WNConv1d(1024, channels, kernel_size=7, padding=3),
            ResLSTM(channels, num_layers=2),
        ]
        for index, stride in enumerate((5, 5, 2, 2, 2)):
            input_dim = channels // (2**index)
            output_dim = channels // (2 ** (index + 1))
            layers.append(DecoderBlock(input_dim, output_dim, stride, (1, 3, 9)))
        layers.extend([
            Activation1d(activations.SnakeBeta(output_dim, alpha_logscale=True)),
            WNConv1d(output_dim, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        ])
        self.model = nn.Sequential(*layers)

    def quantize(self, features):
        quantized, codes, _ = self.quantizer(features)
        if codes.ndim == 3 and codes.shape[0] == 1 and features.shape[0] == 1:
            codes = codes.permute(1, 2, 0)
        return quantized, codes

    def codes_to_embeddings(self, codes):
        embeddings = self.quantizer.idx2emb(codes)
        return embeddings.transpose(1, 2)

    def decode_embeddings(self, embeddings):
        return self.model(embeddings)


class CodecSlimeNet(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        self.encoder = CodecEncoder()
        self.decoder = CodecDecoder(variant)
