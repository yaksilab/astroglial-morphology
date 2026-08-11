"""Configuration module for astroglial morphology pipeline."""

import os
from pathlib import Path
from typing import Dict, Any, Optional


class PipelineConfig:
    """Configuration for the astroglial morphology analysis pipeline."""

    # Model paths
    DEFAULT_MODEL_DIR = Path(__file__).parent / "models"
    DEFAULT_MODEL_NAME = "CP3_S4_1_0001_3000"

    @classmethod
    def get_model_path(cls, model_name: Optional[str] = None) -> str:
        """
        Get the path to the Cellpose model.

        Args:
            model_name: Name of the model directory. If None, uses default.

        Returns:
            Absolute path to the model directory
        """
        if model_name is None:
            model_name = cls.DEFAULT_MODEL_NAME

        # Check environment variable first
        env_path = os.environ.get("ASTROGLIAL_MODEL_PATH")
        if env_path:
            return env_path

        # Use default path
        return str(cls.DEFAULT_MODEL_DIR / model_name)

    # Morphology parameters
    ASTROCYTE_DIAMETER_MICRONS = 31.35  # Average astrocyte diameter in microns
    DIAMETER_BUFFER_MICRONS = 10.0  # Additional buffer for segmentation
    NECK_DISTANCE_RATIO = 0.47  # Ratio of diameter for neck distance calculation

    # Suite2p registration parameters
    SUITE2P_DEFAULTS = {
        "save_path0": "",
        "save_folder": [],
        "functional_chan": 1,
        "tau": 3,
        "multiplane_parallel": False,
        "combined": True,
        "do_registration": True,
        "two_step_registration": False,
        "keep_movie_raw": False,
        "maxregshift": 0.11,  # 11% of frame dimension
        "align_by_chan": 1,
        # Stabilize the registration reference for low-SNR recordings without
        # adding the stronger temporal smoothing used in experimental trials.
        "smooth_sigma_time": 1,
        "do_regmetrics": False,
        "subpixel": 10,  # High-precision registration
        "nonrigid": False,  # Rigid registration only
        "roidetect": False,  # We don't need ROI detection
        "spikedetect": False,  # We don't need spike detection
    }

    # Batch size calculations
    NIMG_INIT_RATIO = 0.15  # 15% of frames for initialization
    NIMG_INIT_MAX = 300
    BATCH_SIZE_RATIO = 1.0  # 100% of frames per batch
    BATCH_SIZE_MAX = 500

    @classmethod
    def calculate_batch_params(
        cls, frames_per_channel_per_plane: int
    ) -> Dict[str, int]:
        """
        Calculate nimg_init and batch_size based on frame count.

        Args:
            frames_per_channel_per_plane: Number of frames per channel per plane

        Returns:
            Dictionary with 'nimg_init' and 'batch_size' keys
        """
        nimg_init = min(
            int(frames_per_channel_per_plane * cls.NIMG_INIT_RATIO), cls.NIMG_INIT_MAX
        )
        batch_size = min(
            int(frames_per_channel_per_plane * cls.BATCH_SIZE_RATIO), cls.BATCH_SIZE_MAX
        )

        return {
            "nimg_init": nimg_init,
            "batch_size": batch_size,
        }

    @classmethod
    def build_suite2p_options(
        cls, metadata: Any, reg_tif: bool = False, **overrides
    ) -> Dict[str, Any]:
        """
        Build Suite2p options dictionary.

        Args:
            metadata: Metadata object with nframes, nchannels, nplanes, fs
            reg_tif: Whether to save registered TIFF
            **overrides: Additional options to override defaults

        Returns:
            Dictionary of Suite2p options
        """
        # Start with defaults
        options = cls.SUITE2P_DEFAULTS.copy()

        # Add metadata-derived parameters
        options.update(
            {
                "nplanes": metadata.nplanes,
                "nchannels": metadata.nchannels,
                "fs": metadata.fs,
                "reg_tif": reg_tif,
            }
        )

        # Add batch parameters
        batch_params = cls.calculate_batch_params(metadata.frames_per_channel_per_plane)
        options.update(batch_params)

        # Apply any overrides
        options.update(overrides)

        return options

    @classmethod
    def calculate_diameter(cls, pix_resolution: float) -> float:
        """
        Calculate segmentation diameter based on pixel resolution.

        Args:
            pix_resolution: Pixel resolution in microns/pixel

        Returns:
            Diameter in pixels
        """
        return (
            pix_resolution * cls.ASTROCYTE_DIAMETER_MICRONS
            + cls.DIAMETER_BUFFER_MICRONS
        )

    @classmethod
    def calculate_neck_distance(cls, diameter: float) -> int:
        """
        Calculate neck distance for cell classification.

        Args:
            diameter: Diameter in pixels

        Returns:
            Neck distance in pixels
        """
        return int(diameter * cls.NECK_DISTANCE_RATIO)

    # Segmentation parameters
    SEGMENTATION_DEFAULTS = {
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "augment": True,
        "resample": True,
        "min_size": 50,
    }

    # File format priorities
    FILE_FORMAT_PRIORITY = [".lif", ".tif"]  # Check LIF first, then TIFF

    # LIF conversion parameters
    LIF_SERIES_INDEX = 0  # Process first series only
    LIF_CHANNEL_INDEX = 0  # Process first channel only
    LIF_PLANE_INDEX = 0  # Process first plane only
