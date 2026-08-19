"""Tests for the Hydra application dispatcher."""

from unittest.mock import Mock, patch
from pathlib import Path

import pytest

from astroglial_morphology.app_config import AppConfig
from astroglial_morphology.__main__ import run_application


@patch("astroglial_morphology.__main__.get_logger")
@patch("astroglial_morphology.__main__.Pipeline")
def test_run_application_maps_typed_config_to_pipeline(
    mock_pipeline_class, mock_get_logger
) -> None:
    config = AppConfig()
    data_path = Path(__file__).parents[1] / "data" / "seed"
    config.input.data_path = str(data_path)
    config.input.pixels_per_micron = 3.168
    config.runtime.use_gpu = True
    config.runtime.manual_correction = True
    config.registration.reg_tif = True
    config.registration.force = True
    config.registration.channel = 1
    config.registration.regmetrics = True
    config.segmentation.projection = "max_projection"
    config.segmentation.channel = "both"
    config.segmentation.skip = True
    config.correspondence.enabled = False
    config.correspondence.trace_channels = [0, 1]
    mock_pipeline_class.return_value.run.return_value = {
        "classification": "ok",
        "correspondence": None,
    }

    result = run_application(config)

    assert result == {"classification": "ok", "correspondence": None}
    constructor_kwargs = mock_pipeline_class.call_args.kwargs
    assert constructor_kwargs["data_path"] == str(data_path)
    assert constructor_kwargs["use_gpu"] is True
    assert constructor_kwargs["reg_tif"] is True
    assert constructor_kwargs["pixels_per_micron"] == pytest.approx(3.168)
    assert constructor_kwargs["config"].SUITE2P_DEFAULTS["maxregshift"] == 0.11

    run_kwargs = mock_pipeline_class.return_value.run.call_args.kwargs
    assert run_kwargs["force_registration"] is True
    assert run_kwargs["manual_correction"] is True
    assert run_kwargs["registration_channel"] == 1
    assert run_kwargs["segmentation_projection"] == "max_projection"
    assert run_kwargs["segmentation_channel"] == "both"
    assert run_kwargs["skip_segmentation"] is True
    assert run_kwargs["trace_channels"] == [0, 1]
    assert run_kwargs["do_regmetrics"] is True


@patch("astroglial_morphology.__main__.get_logger")
@patch("astroglial_morphology.__main__.ModelAssetResolver")
@patch("astroglial_morphology.__main__.load_ensemble_profile")
def test_prefetch_does_not_construct_a_pipeline(
    mock_profile, mock_resolver, mock_get_logger
) -> None:
    config = AppConfig(action="prefetch_models")
    profile = Mock()
    assets = {"complete_cell": Mock()}
    mock_profile.return_value = (profile, assets)
    mock_resolver.return_value.prefetch.return_value = {"complete_cell": "/model"}

    assert run_application(config) is None

    mock_profile.assert_called_once_with(
        profile_name=config.segmentation.ensemble_profile,
        config_path=None,
    )
    mock_resolver.return_value.prefetch.assert_called_once_with(profile)


def test_run_requires_input_path() -> None:
    with pytest.raises(ValueError, match="input.data_path is required"):
        run_application(AppConfig())
