"""Tests for correspondence trace-channel validation."""

import pytest

from astroglial_morphology.correspondence import _normalize_trace_channels


def test_trace_channel_count_above_two_is_rejected():
    with pytest.raises(ValueError, match="only one or two channels"):
        _normalize_trace_channels([0, 1], nchannels=3)


def test_trace_channel_index_above_one_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        _normalize_trace_channels([2], nchannels=2)
