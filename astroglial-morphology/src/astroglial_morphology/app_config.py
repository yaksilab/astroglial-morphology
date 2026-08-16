"""Typed application configuration and Hydra-to-pipeline conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from .config import PipelineConfig
from .ensemble import DEFAULT_PROFILE_NAME


def _default_suite2p_defaults() -> dict[str, Any]:
    return PipelineConfig().SUITE2P_DEFAULTS


def _default_segmentation_defaults() -> dict[str, Any]:
    return PipelineConfig().SEGMENTATION_DEFAULTS


@dataclass
class MorphologySettings:
    astrocyte_diameter_microns: float = 31.35
    diameter_buffer_microns: float = 10.0
    neck_distance_ratio: float = 0.47


@dataclass
class BatchSettings:
    nimg_init_ratio: float = 0.15
    nimg_init_max: int = 300
    batch_size_ratio: float = 1.0
    batch_size_max: int = 500


@dataclass
class LifSettings:
    series_index: int = 0
    channel_index: int = 0
    plane_index: int = 0


@dataclass
class PipelineSettings:
    morphology: MorphologySettings = field(default_factory=MorphologySettings)
    suite2p_defaults: dict[str, Any] = field(default_factory=_default_suite2p_defaults)
    batch: BatchSettings = field(default_factory=BatchSettings)
    segmentation_defaults: dict[str, Any] = field(
        default_factory=_default_segmentation_defaults
    )
    file_format_priority: list[str] = field(default_factory=lambda: [".lif", ".tif"])
    lif: LifSettings = field(default_factory=LifSettings)


@dataclass
class InputSettings:
    data_path: Optional[str] = None
    pixels_per_micron: Optional[float] = None


@dataclass
class RegistrationSettings:
    reg_tif: bool = False
    skip: bool = False
    force: bool = False
    channel: int = 0
    regmetrics: bool = False


@dataclass
class SegmentationSettings:
    mode: str = "single"
    model_path: Optional[str] = None
    ensemble_profile: str = DEFAULT_PROFILE_NAME
    ensemble_config: Optional[str] = None
    model_cache_dir: Optional[str] = None
    projection: str = "mean"
    channel: str = "auto"


@dataclass
class CorrespondenceSettings:
    enabled: bool = True
    segment_length: int = 5
    delta_x: float = 20.0
    subsegmentation_mode: str = "equal_length"
    trace_channels: Optional[list[int]] = None


@dataclass
class RuntimeSettings:
    use_gpu: bool = False
    manual_correction: bool = False
    alignment_only: bool = False


@dataclass
class LoggingSettings:
    level: str = "INFO"
    log_file: Optional[str] = None
    log_dir: Optional[str] = None


@dataclass
class AppConfig:
    action: str = "run"
    input: InputSettings = field(default_factory=InputSettings)
    registration: RegistrationSettings = field(default_factory=RegistrationSettings)
    segmentation: SegmentationSettings = field(default_factory=SegmentationSettings)
    correspondence: CorrespondenceSettings = field(default_factory=CorrespondenceSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)


def compose_app_config(cfg: DictConfig) -> AppConfig:
    """Validate a composed Hydra config and convert it into typed dataclasses."""

    schema = OmegaConf.structured(AppConfig)
    app_keys = [item.name for item in fields(AppConfig)]
    merged = OmegaConf.merge(schema, OmegaConf.masked_copy(cfg, app_keys))
    OmegaConf.resolve(merged)
    return OmegaConf.to_object(merged)


def make_pipeline_config(settings: PipelineSettings) -> PipelineConfig:
    """Translate nested YAML settings into the pipeline's runtime config object."""

    return PipelineConfig(
        ASTROCYTE_DIAMETER_MICRONS=settings.morphology.astrocyte_diameter_microns,
        DIAMETER_BUFFER_MICRONS=settings.morphology.diameter_buffer_microns,
        NECK_DISTANCE_RATIO=settings.morphology.neck_distance_ratio,
        SUITE2P_DEFAULTS=dict(settings.suite2p_defaults),
        NIMG_INIT_RATIO=settings.batch.nimg_init_ratio,
        NIMG_INIT_MAX=settings.batch.nimg_init_max,
        BATCH_SIZE_RATIO=settings.batch.batch_size_ratio,
        BATCH_SIZE_MAX=settings.batch.batch_size_max,
        SEGMENTATION_DEFAULTS=dict(settings.segmentation_defaults),
        FILE_FORMAT_PRIORITY=list(settings.file_format_priority),
        LIF_SERIES_INDEX=settings.lif.series_index,
        LIF_CHANNEL_INDEX=settings.lif.channel_index,
        LIF_PLANE_INDEX=settings.lif.plane_index,
    )
