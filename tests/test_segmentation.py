import math

import pytest
import torch

from codecslime.segmentation import average_segments, segment


@pytest.mark.parametrize("frame_rate", (36, 37.5, 40, 67, 80))
def test_supported_frame_rates_cover_input(frame_rate):
    features = torch.randn(1, 8, 80)
    durations = segment(features, frame_rate)
    assert len(durations) == round(80 * frame_rate / 80)
    assert durations.sum().item() == 80
    assert durations.min().item() >= 1
    assert durations.max().item() <= 4
    assert average_segments(features, durations).shape[-1] == len(durations)


@pytest.mark.parametrize("frame_rate", (35.9, 80.1, math.nan))
def test_invalid_frame_rates_are_rejected(frame_rate):
    with pytest.raises(ValueError):
        segment(torch.randn(1, 8, 80), frame_rate)
