"""Offline dynamic-programming segmentation from the CodecSlime paper."""

from itertools import combinations
import math
from numbers import Real

import numpy as np
import torch


def _segment_costs(features: np.ndarray, length: int, cost_type: str = "std_1.0"):
    costs = [0.0] * (len(features) - length + 1)
    for start in range(len(costs)):
        segment = features[start : start + length]
        if cost_type.startswith("std"):
            scale = float(cost_type.split("_")[1])
            mean = segment.mean(axis=0)
            costs[start] = np.linalg.norm(segment - mean, axis=1).mean() * ((length - 1) ** scale)
        elif cost_type.startswith("pairwise"):
            exponent = 1.0 if cost_type == "pairwise" else float(cost_type.split("_")[1])
            if length > 1:
                costs[start] = sum(
                    np.linalg.norm(segment[p] - segment[q]) ** exponent
                    for p, q in combinations(range(length), 2)
                ) / length
        else:
            raise ValueError(f"Unsupported cost type: {cost_type}")
    return costs


def segment(features: torch.Tensor, frame_rate: float) -> torch.Tensor:
    """Return segment durations for an 80 Hz encoder feature sequence."""
    if isinstance(frame_rate, bool) or not isinstance(frame_rate, Real):
        raise TypeError("frame_rate must be a real number")
    frame_rate = float(frame_rate)
    if not math.isfinite(frame_rate) or not 36 <= frame_rate <= 80:
        raise ValueError("frame_rate must be between 36 and 80 Hz")
    total = features.shape[-1]
    if total == 0:
        return torch.empty(0, dtype=torch.long, device=features.device)
    if frame_rate == 80:
        return torch.ones(total, dtype=torch.long, device=features.device)
    allowed = (1, 2, 3, 4)
    target_segments = min(total, max(math.ceil(total / allowed[-1]), round(total * frame_rate / 80)))
    array = features.detach().float().cpu().squeeze(0).transpose(0, 1).numpy()
    costs = {length: _segment_costs(array, length) for length in allowed}
    table = [[float("inf")] * (target_segments + 1) for _ in range(total + 1)]
    previous = [[None] * (target_segments + 1) for _ in range(total + 1)]
    table[0][0] = 0.0
    for count in range(1, target_segments + 1):
        for length in allowed:
            for used in range(count, min(total, count * allowed[-1]) + 1):
                if used < length:
                    continue
                prior = table[used - length][count - 1]
                if prior != float("inf"):
                    candidate = prior + costs[length][used - length]
                    if candidate < table[used][count]:
                        table[used][count] = candidate
                        previous[used][count] = length
    if table[total][target_segments] == float("inf"):
        raise ValueError("No valid segmentation found")
    durations = []
    used, count = total, target_segments
    while count:
        length = previous[used][count]
        durations.append(length)
        used -= length
        count -= 1
    durations.reverse()
    return torch.tensor(durations, dtype=torch.long, device=features.device)


def average_segments(features: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
    pieces = torch.split(features, durations.tolist(), dim=-1)
    return torch.cat([piece.mean(dim=-1, keepdim=True) for piece in pieces], dim=-1)
