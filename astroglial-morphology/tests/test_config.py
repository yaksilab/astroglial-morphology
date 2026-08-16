"""Tests for the instance-backed pipeline runtime configuration."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from astroglial_morphology.config import PipelineConfig
from astroglial_morphology.utils.tiff_utils import Metadata


def test_defaults_match_the_legacy_pipeline() -> None:
    config = PipelineConfig()

    assert config.ASTROCYTE_DIAMETER_MICRONS == 31.35
    assert config.DIAMETER_BUFFER_MICRONS == 10.0
    assert config.NECK_DISTANCE_RATIO == 0.47
    assert config.SUITE2P_DEFAULTS["maxregshift"] == 0.11
    assert config.SUITE2P_DEFAULTS["subpixel"] == 10
    assert config.SEGMENTATION_DEFAULTS["min_size"] == 80
    assert config.FILE_FORMAT_PRIORITY == [".lif", ".tif"]


def test_config_instances_do_not_share_mutable_defaults() -> None:
    first = PipelineConfig()
    second = PipelineConfig()

    first.SUITE2P_DEFAULTS["maxregshift"] = 0.2
    first.SEGMENTATION_DEFAULTS["normalize"]["invert"] = True

    assert second.SUITE2P_DEFAULTS["maxregshift"] == 0.11
    assert second.SEGMENTATION_DEFAULTS["normalize"]["invert"] is False


def test_get_model_path_uses_environment_then_packaged_default() -> None:
    with patch.dict(os.environ, {}, clear=True):
        default_path = PipelineConfig.get_model_path()
        assert Path(default_path).is_absolute()
        assert default_path.endswith(PipelineConfig.DEFAULT_MODEL_NAME)

    with patch.dict(os.environ, {"ASTROGLIAL_MODEL_PATH": "/custom/model"}):
        assert PipelineConfig.get_model_path() == "/custom/model"


def test_calculations_honor_instance_overrides() -> None:
    config = PipelineConfig(
        NIMG_INIT_RATIO=0.2,
        NIMG_INIT_MAX=150,
        BATCH_SIZE_RATIO=0.5,
        BATCH_SIZE_MAX=400,
        ASTROCYTE_DIAMETER_MICRONS=35.0,
        DIAMETER_BUFFER_MICRONS=5.0,
        NECK_DISTANCE_RATIO=0.5,
    )

    assert config.calculate_batch_params(1_000) == {
        "nimg_init": 150,
        "batch_size": 400,
    }
    assert config.calculate_diameter(2.0) == 75.0
    assert config.calculate_neck_distance(75.9) == 37


def test_build_suite2p_options_merges_metadata_and_overrides() -> None:
    metadata = Metadata(
        nframes=2_000,
        nchannels=2,
        nplanes=1,
        finterval=1.0,
        pix_resolution=8.36,
    )
    config = PipelineConfig()

    options = config.build_suite2p_options(
        metadata, reg_tif=True, maxregshift=0.2
    )

    assert options["nchannels"] == 2
    assert options["nimg_init"] == 150
    assert options["batch_size"] == 500
    assert options["reg_tif"] is True
    assert options["maxregshift"] == pytest.approx(0.2)
