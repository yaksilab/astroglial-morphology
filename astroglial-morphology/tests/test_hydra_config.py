"""Tests for packaged Hydra configuration composition."""

import importlib.resources
import os
import subprocess
import sys

import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import ConfigCompositionException
from omegaconf.errors import ValidationError

from astroglial_morphology.app_config import compose_app_config


@pytest.fixture(autouse=True)
def clear_hydra_state():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def _compose(*overrides: str):
    with initialize_config_module(
        config_module="astroglial_morphology.conf", version_base="1.3"
    ):
        return compose(
            config_name="config",
            overrides=list(overrides),
            return_hydra_config=True,
        )


def test_default_config_matches_existing_runtime_defaults() -> None:
    config = compose_app_config(_compose())

    assert config.action == "run"
    assert config.input.data_path is None
    assert config.segmentation.mode == "single"
    assert config.pipeline.suite2p_defaults["maxregshift"] == pytest.approx(0.11)
    assert config.pipeline.segmentation_defaults["min_size"] == 80
    assert config.correspondence.enabled is True
    assert config.segmentation.skip is False
    assert config.pipeline.suite2p_defaults["smooth_sigma"] == pytest.approx(1.15)
    assert config.pipeline.suite2p_defaults["one_photon_reg"] is False


def test_ensemble_group_and_native_overrides_are_composed() -> None:
    config = compose_app_config(
        _compose(
            "segmentation=ensemble",
            "input.data_path=/data/example",
            "registration.force=true",
            "correspondence.trace_channels=[0,1]",
        )
    )

    assert config.segmentation.mode == "ensemble"
    assert config.input.data_path == "/data/example"
    assert config.registration.force is True
    assert config.correspondence.trace_channels == [0, 1]


def test_one_photon_suite2p_overrides_are_composed() -> None:
    config = compose_app_config(
        _compose(
            "pipeline.suite2p_defaults.one_photon_reg=true",
            "pipeline.suite2p_defaults.smooth_sigma=3",
        )
    )

    assert config.pipeline.suite2p_defaults["one_photon_reg"] is True
    assert config.pipeline.suite2p_defaults["smooth_sigma"] == pytest.approx(3.0)


def test_segmentation_skip_override_is_composed() -> None:
    config = compose_app_config(_compose("segmentation.skip=true"))

    assert config.segmentation.skip is True


def test_environment_defaults_are_overridden_by_explicit_hydra_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASTROGLIAL_MODEL_PATH", "/environment/model")
    monkeypatch.setenv("ASTROGLIAL_LOGLEVEL", "DEBUG")

    from_environment = compose_app_config(_compose())
    explicit = compose_app_config(
        _compose("segmentation.model_path=/command-line/model", "logging.level=WARNING")
    )

    assert from_environment.segmentation.model_path == "/environment/model"
    assert from_environment.logging.level == "DEBUG"
    assert explicit.segmentation.model_path == "/command-line/model"
    assert explicit.logging.level == "WARNING"


def test_schema_rejects_unknown_keys_and_invalid_types() -> None:
    with pytest.raises(ConfigCompositionException):
        _compose("pipeline.unknown_setting=true")

    with pytest.raises(ValidationError):
        compose_app_config(_compose("input.pixels_per_micron=not-a-number"))


def test_hydra_settings_preserve_working_directory_and_record_runs() -> None:
    config = _compose()

    assert config.hydra.job.chdir is False
    assert str(config.hydra.run.dir).startswith("outputs/")
    assert str(config.hydra.sweep.dir).startswith("multirun/")


def test_config_files_are_packaged_resources() -> None:
    config_root = importlib.resources.files("astroglial_morphology.conf")

    assert config_root.joinpath("config.yaml").is_file()
    assert config_root.joinpath("segmentation", "ensemble.yaml").is_file()


def test_module_cli_prints_resolved_config_without_running_pipeline() -> None:
    project_root = os.path.dirname(os.path.dirname(__file__))
    result = subprocess.run(
        [sys.executable, "-m", "astroglial_morphology", "--cfg", "job"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )

    assert result.returncode == 0, result.stderr
    assert "input:" in result.stdout
    assert "segmentation:" in result.stdout
