"""Runtime configuration used by the astroglial morphology pipeline."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def _suite2p_defaults() -> Dict[str, Any]:
    """Return a fresh Suite2p default mapping for every config instance."""

    return {
        "save_path0": "",
        "save_folder": [],
        "functional_chan": 1,
        "tau": 3,
        "multiplane_parallel": False,
        "combined": True,
        "do_registration": True,
        "two_step_registration": False,
        "keep_movie_raw": False,
        "maxregshift": 0.11,
        "align_by_chan": 1,
        "smooth_sigma_time": 1,
        "do_regmetrics": False,
        "subpixel": 10,
        "nonrigid": False,
        "roidetect": False,
        "spikedetect": False,
    }


def _segmentation_defaults() -> Dict[str, Any]:
    """Return the legacy Cellpose defaults without sharing nested mappings."""

    return {
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "diameter": None,
        "augment": True,
        "resample": True,
        "min_size": 80,
        "normalize": {
            "lowhigh": None,
            "percentile": [1.0, 99.0],
            "normalize": True,
            "norm3D": True,
            "sharpen_radius": 0,
            "smooth_radius": 0,
            "tile_norm_blocksize": 0,
            "tile_norm_smooth3D": 1,
            "invert": False,
        },
    }


@dataclass
class PipelineConfig:
    """Typed, instance-backed settings for the analysis pipeline.

    Hydra composes these values from YAML for command-line runs. The class is
    also deliberately usable by programmatic callers that inject a custom
    configuration into :class:`~astroglial_morphology.pipeline.Pipeline`.
    """

    DEFAULT_MODEL_DIR = Path(__file__).parent / "models"
    DEFAULT_MODEL_NAME = "CP3_S4_1_0001_3000"

    ASTROCYTE_DIAMETER_MICRONS: float = 31.35
    DIAMETER_BUFFER_MICRONS: float = 10.0
    NECK_DISTANCE_RATIO: float = 0.47

    SUITE2P_DEFAULTS: Dict[str, Any] = field(default_factory=_suite2p_defaults)

    NIMG_INIT_RATIO: float = 0.15
    NIMG_INIT_MAX: int = 300
    BATCH_SIZE_RATIO: float = 1.0
    BATCH_SIZE_MAX: int = 500

    SEGMENTATION_DEFAULTS: Dict[str, Any] = field(
        default_factory=_segmentation_defaults
    )
    FILE_FORMAT_PRIORITY: list[str] = field(default_factory=lambda: [".lif", ".tif"])
    LIF_SERIES_INDEX: int = 0
    LIF_CHANNEL_INDEX: int = 0
    LIF_PLANE_INDEX: int = 0

    @classmethod
    def get_model_path(cls, model_name: Optional[str] = None) -> str:
        """Resolve a model path, retaining the legacy environment fallback."""

        env_path = os.environ.get("ASTROGLIAL_MODEL_PATH")
        if env_path:
            return env_path
        return str(cls.DEFAULT_MODEL_DIR / (model_name or cls.DEFAULT_MODEL_NAME))

    def calculate_batch_params(
        self, frames_per_channel_per_plane: int
    ) -> Dict[str, int]:
        """Calculate Suite2p initialization and batch sizes for a frame count."""

        nimg_init = min(
            int(frames_per_channel_per_plane * self.NIMG_INIT_RATIO),
            self.NIMG_INIT_MAX,
        )
        batch_size = min(
            int(frames_per_channel_per_plane * self.BATCH_SIZE_RATIO),
            self.BATCH_SIZE_MAX,
        )
        return {"nimg_init": nimg_init, "batch_size": batch_size}

    def build_suite2p_options(
        self, metadata: Any, reg_tif: bool = False, **overrides: Any
    ) -> Dict[str, Any]:
        """Build Suite2p options from defaults, metadata, and explicit overrides."""

        options = deepcopy(self.SUITE2P_DEFAULTS)
        options.update(
            {
                "nplanes": metadata.nplanes,
                "nchannels": metadata.nchannels,
                "fs": metadata.fs,
                "reg_tif": reg_tif,
            }
        )
        options.update(
            self.calculate_batch_params(metadata.frames_per_channel_per_plane)
        )
        options.update(overrides)
        return options

    def calculate_diameter(self, pix_resolution: float) -> float:
        """Calculate segmentation diameter in pixels from physical calibration."""

        return (
            pix_resolution * self.ASTROCYTE_DIAMETER_MICRONS
            + self.DIAMETER_BUFFER_MICRONS
        )

    def calculate_neck_distance(self, diameter: float) -> int:
        """Calculate the morphology-classification neck-distance threshold."""

        return int(diameter * self.NECK_DISTANCE_RATIO)
