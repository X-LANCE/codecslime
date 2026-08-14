"""Neural network layers used by CodecSlime's encoder and decoder."""

import torch
from einops import rearrange
from torch import nn
from torch.nn.utils import weight_norm

from . import activations
from .alias_free import Activation1d


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        padding = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Activation1d(activations.SnakeBeta(dim, alpha_logscale=True)),
            WNConv1d(dim, dim, 7, dilation=dilation, padding=padding),
            Activation1d(activations.SnakeBeta(dim, alpha_logscale=True)),
            WNConv1d(dim, dim, 1),
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int, dilations=(1, 3, 9)):
        super().__init__()
        self.block = nn.Sequential(
            *(ResidualUnit(dim // 2, dilation) for dilation in dilations),
            Activation1d(activations.SnakeBeta(dim // 2, alpha_logscale=True)),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2 + stride % 2,
            ),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int, dilations=(1, 3, 9)):
        super().__init__()
        self.block = nn.Sequential(
            Activation1d(activations.SnakeBeta(input_dim, alpha_logscale=True)),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2 + stride % 2,
                output_padding=stride % 2,
            ),
            *(ResidualUnit(output_dim, dilation) for dilation in dilations),
        )

    def forward(self, x):
        return self.block(x)


class ResLSTM(nn.Module):
    def __init__(self, dimension: int, num_layers: int = 2):
        super().__init__()
        self.skip = True
        self.lstm = nn.LSTM(dimension, dimension, num_layers, batch_first=True)

    def forward(self, x):
        sequence = rearrange(x, "b f t -> b t f")
        output, _ = self.lstm(sequence)
        return rearrange(output + sequence, "b t f -> b f t")
