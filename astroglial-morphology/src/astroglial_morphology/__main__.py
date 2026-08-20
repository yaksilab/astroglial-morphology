"""Hydra entry point for the astroglial morphology pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig

from .app_config import AppConfig, compose_app_config, make_pipeline_config
from .correspondence import VALID_SUBSEGMENTATION_MODES
from .ensemble import ModelAssetResolver, load_ensemble_profile
from .logging_config import adopt_existing_logging, add_log_file_handler, get_logger
from .pipeline import Pipeline


_VALID_ACTIONS = {"run", "prefetch_models"}
_VALID_SEGMENTATION_MODES = {"single", "ensemble"}
_VALID_PROJECTIONS = {"mean", "max_projection"}
_VALID_SEGMENTATION_CHANNELS = {"auto", "both", "0", "1"}


def _validate_config(config: AppConfig) -> None:
    """Reject invalid cross-field combinations with actionable messages."""

    if config.action not in _VALID_ACTIONS:
        raise ValueError(
            f"action must be one of {sorted(_VALID_ACTIONS)}, got {config.action!r}"
        )
    if config.segmentation.mode not in _VALID_SEGMENTATION_MODES:
        raise ValueError(
            "segmentation.mode must be 'single' or 'ensemble', "
            f"got {config.segmentation.mode!r}"
        )
    if (
        config.segmentation.mode == "ensemble"
        and config.segmentation.model_path is not None
    ):
        raise ValueError(
            "segmentation.model_path applies to segmentation=single; "
            "use segmentation.ensemble_config for an ensemble profile"
        )
    if config.segmentation.projection not in _VALID_PROJECTIONS:
        raise ValueError(
            "segmentation.projection must be 'mean' or 'max_projection'"
        )
    if config.segmentation.channel not in _VALID_SEGMENTATION_CHANNELS:
        raise ValueError("segmentation.channel must be auto, both, 0, or 1")
    if config.registration.channel not in {0, 1}:
        raise ValueError("registration.channel must be 0 or 1")
    if config.correspondence.subsegmentation_mode not in VALID_SUBSEGMENTATION_MODES:
        raise ValueError(
            "correspondence.subsegmentation_mode must be one of "
            f"{sorted(VALID_SUBSEGMENTATION_MODES)}"
        )
    if config.action == "run" and not config.input.data_path:
        raise ValueError("input.data_path is required when action=run")


def run_application(config: AppConfig) -> Optional[dict[str, object]]:
    """Execute a validated Hydra application configuration.

    This thin dispatcher is intentionally separate from Hydra's decorated entry
    point so library users and unit tests can exercise the mapping without
    creating a Hydra global runtime.
    """

    _validate_config(config)
    logger = get_logger(__name__)

    if config.action == "prefetch_models":
        profile, assets = load_ensemble_profile(
            profile_name=config.segmentation.ensemble_profile,
            config_path=config.segmentation.ensemble_config,
        )
        downloaded = ModelAssetResolver(
            assets, config.segmentation.model_cache_dir
        ).prefetch(profile)
        for role, path in downloaded.items():
            logger.info("Verified %s model: %s", role, path)
        return None

    data_path = Path(config.input.data_path)
    if not data_path.exists():
        raise ValueError(f"Input data path does not exist: {data_path}")
    if not data_path.is_dir():
        raise ValueError(f"Input data path is not a directory: {data_path}")

    pipeline = Pipeline(
        data_path=str(data_path),
        model_path=config.segmentation.model_path,
        use_gpu=config.runtime.use_gpu,
        reg_tif=config.registration.reg_tif,
        config=make_pipeline_config(config.pipeline),
        segmentation_mode=config.segmentation.mode,
        ensemble_profile=config.segmentation.ensemble_profile,
        ensemble_config=config.segmentation.ensemble_config,
        pixels_per_micron=config.input.pixels_per_micron,
        model_cache_dir=config.segmentation.model_cache_dir,
    )
    results = pipeline.run(
        skip_registration=config.registration.skip,
        force_registration=config.registration.force,
        manual_correction=config.runtime.manual_correction,
        export_correspondence=config.correspondence.enabled,
        correspondence_segment_length=config.correspondence.segment_length,
        correspondence_delta_x=config.correspondence.delta_x,
        correspondence_subsegmentation_mode=config.correspondence.subsegmentation_mode,
        segmentation_projection=config.segmentation.projection,
        segmentation_channel=config.segmentation.channel,
        registration_channel=config.registration.channel,
        trace_channels=config.correspondence.trace_channels,
        do_regmetrics=config.registration.regmetrics,
        alignment_only=config.runtime.alignment_only,
        skip_segmentation=config.segmentation.skip,
        existing_seg_path=config.segmentation.existing_seg_path,
    )

    logger.info("Pipeline completed successfully")
    logger.info("Results: %s", results.get("classification"))
    correspondence = results.get("correspondence")
    if config.correspondence.enabled and correspondence:
        logger.info(
            "Correspondence matrix: %s (npy), %s (mat)",
            correspondence["correspondence_matrix_path"],
            correspondence["correspondence_matrix_mat_path"],
        )
    return results


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def _hydra_main(cfg: DictConfig) -> None:
    """Compose, validate, and dispatch the packaged Hydra configuration."""

    config = compose_app_config(cfg)
    adopt_existing_logging()
    add_log_file_handler(
        config.logging.log_file,
        log_dir=config.logging.log_dir,
        use_environment=False,
    )
    run_application(config)


def main() -> None:
    """Run the package through Hydra's command-line interface."""

    _hydra_main()


if __name__ == "__main__":
    main()
