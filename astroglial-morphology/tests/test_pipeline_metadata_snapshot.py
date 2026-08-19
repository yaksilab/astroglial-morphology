"""Verify the structured parameter snapshot recorded in pipeline_metadata.json.

These tests focus on the GUI-facing additions: the expanded registration
compatibility keys and the parameter/defaults/overrides fields.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np

from astroglial_morphology.io.file_detection import InputFileInfo, InputFormat
from astroglial_morphology.pipeline import Pipeline
from astroglial_morphology.utils.tiff_utils import Metadata


def _build_pipeline(temp_dir):
    lif_file = temp_dir / "test.lif"
    lif_file.write_bytes(b"fake-lif")

    suite2p_dir = temp_dir / "suite2p"
    plane_path = suite2p_dir / "plane0"
    plane_path.mkdir(parents=True)
    complete_flag = suite2p_dir / ".registration_complete"
    complete_flag.touch()

    data_bin = plane_path / "data.bin"
    data_bin.touch()

    np.save(
        plane_path / "ops.npy",
        {
            "Ly": 12,
            "Lx": 34,
            "nframes": 5,
            "nchannels": 1,
            "channel_indices": [0],
            "reg_file": str(data_bin),
            "meanImg": np.zeros((12, 34), dtype=np.float32),
            "refImg": np.zeros((12, 34), dtype=np.float32),
            "badframes": np.array([False, False, False, False, False]),
            "xoff": np.array([0.0, 0.0, 0.0]),
            "yoff": np.array([0.0, 0.0, 0.0]),
            "corrXY": np.array([0.5, 0.5, 0.5]),
            "timing": {"registration": np.float32(0.5)},
        },
        allow_pickle=True,
    )

    with patch("astroglial_morphology.pipeline.Segmentation"):
        pipeline = Pipeline(data_path=str(temp_dir))
    pipeline.file_info = InputFileInfo(path=lif_file, format=InputFormat.LIF)
    pipeline.metadata = Metadata(
        nframes=5,
        nchannels=1,
        nplanes=1,
        finterval=2.0,
        pix_resolution=1.76,
    )
    return pipeline, plane_path


class TestParameterSnapshot:
    def test_defaults_run_records_no_overrides(self, temp_dir):
        pipeline, plane_path = _build_pipeline(temp_dir)
        pipeline.suite2p_options = {
            "nplanes": 1,
            "nchannels": 1,
            "fs": 0.5,
            "do_registration": True,
            "two_step_registration": False,
            "nonrigid": False,
            "maxregshift": 0.11,
            "subpixel": 10,
            "smooth_sigma_time": 1,
            "tau": 3,
            "align_by_chan": 1,
            "functional_chan": 1,
            "batch_size": 100,
            "nimg_init": 30,
            "do_regmetrics": False,
            "reg_tif": False,
            "reg_tif_chan2": False,
            "roidetect": False,
            "spikedetect": False,
        }
        pipeline.write_pipeline_metadata()

        payload = json.loads((plane_path / "pipeline_metadata.json").read_text())
        params = payload["parameters"]
        assert "registration" in params
        assert params["registration"]["maxregshift"] == 0.11
        assert params["registration"]["nonrigid"] is False
        assert payload["parameter_overrides"] == {} or "registration" not in payload["parameter_overrides"]

    def test_overrides_are_reported(self, temp_dir):
        pipeline, plane_path = _build_pipeline(temp_dir)
        pipeline.suite2p_options = {
            "nplanes": 1,
            "nchannels": 1,
            "fs": 0.5,
            "do_registration": True,
            "two_step_registration": True,
            "nonrigid": True,
            "maxregshift": 0.2,
            "subpixel": 20,
            "smooth_sigma_time": 2,
            "tau": 5,
            "align_by_chan": 2,
            "functional_chan": 1,
            "batch_size": 100,
            "nimg_init": 30,
            "do_regmetrics": True,
            "reg_tif": True,
            "reg_tif_chan2": False,
            "roidetect": False,
            "spikedetect": False,
        }
        pipeline.segmentation_eval_params = {
            "flow_threshold": 0.6,
            "cellprob_threshold": 0.0,
            "diameter": None,
            "augment": True,
            "resample": True,
            "min_size": 80,
        }
        pipeline.write_pipeline_metadata()

        payload = json.loads((plane_path / "pipeline_metadata.json").read_text())
        overrides = payload["parameter_overrides"]
        assert overrides["registration"]["maxregshift"]["used"] == 0.2
        assert overrides["registration"]["nonrigid"]["used"] is True
        assert overrides["segmentation"]["flow_threshold"]["used"] == 0.6

    def test_expanded_registration_compat_keys(self, temp_dir):
        pipeline, _ = _build_pipeline(temp_dir)
        for key in (
            "maxregshift",
            "nonrigid",
            "two_step_registration",
            "subpixel",
            "smooth_sigma",
            "smooth_sigma_time",
            "tau",
            "nimg_init",
            "batch_size",
            "1Preg",
        ):
            assert key in pipeline._REGISTRATION_COMPATIBILITY_KEYS
