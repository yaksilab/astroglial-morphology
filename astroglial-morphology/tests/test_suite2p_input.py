"""Tests for direct Suite2p plane0 input handling."""

from __future__ import annotations

import json

import numpy as np
import pytest

from astroglial_morphology.pipeline import Pipeline
from astroglial_morphology.io.file_detection import InputFormat


def _make_plane(tmp_path, *, nchannels=1):
    plane = tmp_path / "suite2p" / "plane0"
    plane.mkdir(parents=True)
    (plane / "data.bin").write_bytes(b"\0" * 64)
    if nchannels == 2:
        (plane / "data_chan2.bin").write_bytes(b"\0" * 64)
    np.save(
        plane / "ops.npy",
        {"Ly": 4, "Lx": 4, "nframes": 2, "nchannels": nchannels, "nplanes": 1, "fs": 2.0},
        allow_pickle=True,
    )
    return plane


def test_direct_suite2p_input_uses_nearest_metadata_calibration(tmp_path):
    plane = _make_plane(tmp_path)
    (plane / "pipeline_metadata.json").write_text(
        json.dumps({"pixels_per_micron": 3.168}), encoding="utf-8"
    )
    pipeline = Pipeline(str(plane), segmentation_mode="ensemble")

    pipeline.detect_input()
    pipeline.load_metadata()
    pipeline.prepare_data()

    assert pipeline.file_info.format == InputFormat.SUITE2P
    assert pipeline.input_mode == "suite2p"
    assert pipeline.pixels_per_micron == pytest.approx(3.168)
    assert pipeline.metadata.frames_per_channel_per_plane == 2
    assert pipeline.suite2p_options["input_format"] == "suite2p"
    assert pipeline.run_registration() is False


def test_cli_calibration_overrides_direct_suite2p_metadata(tmp_path):
    plane = _make_plane(tmp_path)
    (plane / "pipeline_metadata.json").write_text(
        json.dumps({"pixels_per_micron": 3.168}), encoding="utf-8"
    )
    pipeline = Pipeline(
        str(plane), segmentation_mode="ensemble", pixels_per_micron=2.0
    )
    pipeline.detect_input()
    pipeline.load_metadata()

    assert pipeline.pixels_per_micron == pytest.approx(2.0)
    assert pipeline.calibration_source == "cli"


def test_direct_suite2p_rejects_missing_second_channel_binary(tmp_path):
    plane = _make_plane(tmp_path, nchannels=2)
    (plane / "data_chan2.bin").unlink()
    pipeline = Pipeline(str(plane), segmentation_mode="ensemble")
    pipeline.detect_input()

    with pytest.raises(FileNotFoundError, match="data_chan2.bin"):
        pipeline.load_metadata()


def test_direct_suite2p_rejects_force_registration(tmp_path):
    plane = _make_plane(tmp_path)
    pipeline = Pipeline(str(plane), segmentation_mode="ensemble")

    with pytest.raises(ValueError, match="force-registration"):
        pipeline.run(force_registration=True)


def test_direct_suite2p_ensemble_requires_calibration_before_model_download(tmp_path):
    plane = _make_plane(tmp_path)
    pipeline = Pipeline(str(plane), segmentation_mode="ensemble")
    pipeline.detect_input()
    pipeline.load_metadata()
    pipeline.projections = {"mean": np.zeros((4, 4), dtype=np.float32)}

    with pytest.raises(ValueError, match="requires pixels-per-micron calibration"):
        pipeline.segment_cells()
