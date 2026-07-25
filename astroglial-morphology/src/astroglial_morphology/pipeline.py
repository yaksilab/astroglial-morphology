"""Main pipeline orchestration for astroglial morphology analysis."""

import logging
import json
import os
import platform
import socket
import sys
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Optional, Dict, Any, Sequence, Union
import numpy as np

from .config import PipelineConfig
from .io import detect_input_file, load_metadata, InputFormat
from .registration import (
    do_registration,
    check_registration_complete,
    get_suite2p_output_dir,
)
from .binary_utils import create_projections
from .segmentation import Segmentation
from .classifier import classify_cells
from .correspondence import (
    export_correspondence_products,
    SUBSEGMENTATION_MODE_EQUAL_LENGTH,
)
from .utils.lif_utils import lif_to_suite2p_binary, Metadata

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main pipeline for astroglial morphology analysis.

    Orchestrates the complete workflow from raw microscopy data to
    classified cell morphologies.
    """

    _REGISTRATION_COMPATIBILITY_KEYS = (
        "align_by_chan",
        "functional_chan",
        "nchannels",
    )
    _REGISTRATION_INPUT_FILENAMES = (
        "ops.npy",
        "data.bin",
        "data_chan2.bin",
        "data_raw.bin",
        "data_chan2_raw.bin",
    )

    def __init__(
        self,
        data_path: str,
        model_path: Optional[str] = None,
        use_gpu: bool = False,
        reg_tif: bool = False,
        config: Optional[PipelineConfig] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            data_path: Path to directory containing input files
            model_path: Path to Cellpose model. If None, uses default.
            use_gpu: Whether to use GPU for segmentation
            reg_tif: Whether to save registered TIFF files
            config: Custom configuration object. If None, uses defaults.
        """
        self.data_path = data_path
        self.use_gpu = use_gpu
        self.reg_tif = reg_tif
        self.config = config or PipelineConfig()

        if model_path is None:
            model_path = self.config.get_model_path()
        self.model_path = model_path

        self.segmenter = Segmentation(
            model_path=self.model_path,
            gpu=self.use_gpu,
        )

        self.file_info = None
        self.metadata: Optional[Metadata] = None
        self.suite2p_options = None
        self.projections = None
        self.masks = None
        self.classification = None
        self.neck_distance: Optional[int] = None
        self.segmentation_base_path: Optional[str] = None
        self.registration_channel = 0
        self.segmentation_channel = "both"
        self.trace_channels: Optional[list[int]] = None
        self.segmentation_projection = "mean"
        self.do_regmetrics = False
        self.manual_correction = False
        self.export_correspondence = False
        self.alignment_only = False

    @staticmethod
    def _normalize_trace_channels(
        trace_channels: Optional[Union[str, Sequence[int]]],
    ) -> Optional[list[int]]:
        if trace_channels is None:
            return None
        if isinstance(trace_channels, str):
            if not trace_channels.strip():
                return None
            channels = [int(part.strip()) for part in trace_channels.split(",")]
        else:
            channels = [int(channel) for channel in trace_channels]

        if not channels:
            return None
        if len(set(channels)) != len(channels):
            raise ValueError("Duplicate trace channels are not allowed")
        return channels

    @staticmethod
    def _normalize_image(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32, copy=False)
        low, high = np.percentile(image, [1.0, 99.0])
        if high <= low:
            return np.zeros_like(image, dtype=np.float32)
        return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)

    def _available_channel_count(self) -> int:
        if self.suite2p_options is not None:
            return int(self.suite2p_options.get("nchannels", 1))
        if self.metadata is not None:
            return int(self.metadata.nchannels)
        return 1

    def _validate_channel_index(self, channel: int, label: str) -> None:
        if channel < 0 or channel >= self._available_channel_count():
            raise ValueError(
                f"{label} channel {channel} is out of range for "
                f"{self._available_channel_count()} channel(s)"
            )

    def _validate_supported_channel_count(self) -> None:
        if (
            self.file_info is not None
            and self.file_info.format == InputFormat.TIFF
            and self._available_channel_count() > 2
        ):
            raise ValueError(
                "TIFF processing supports at most two channels; "
                f"found {self._available_channel_count()}"
            )

    def _validate_run_channels(self, export_correspondence: bool) -> None:
        self._validate_supported_channel_count()
        self._validate_channel_index(self.registration_channel, "Registration")

        if self.segmentation_channel not in {"both", "0", "1"}:
            raise ValueError(
                "segmentation_channel must be one of: both, 0, 1"
            )
        if self.segmentation_channel in {"0", "1"}:
            self._validate_channel_index(int(self.segmentation_channel), "Segmentation")

        if export_correspondence:
            if self._available_channel_count() > 1 and self.trace_channels is None:
                raise ValueError(
                    "trace_channels must be specified when exporting traces "
                    "from multi-channel data"
                )
            if self.trace_channels is None:
                self.trace_channels = [0]
            for channel in self.trace_channels:
                self._validate_channel_index(channel, "Trace")

    def _suite2p_output_dir(self) -> Path:
        return get_suite2p_output_dir(self.data_path, self.suite2p_options)

    def _invalidate_registration_completion(self, reason: str) -> None:
        flag_path = self._suite2p_output_dir() / ".registration_complete"
        if flag_path.exists():
            flag_path.unlink()
            logger.info(
                "Invalidated completed registration because %s: %s",
                reason,
                flag_path,
            )

    def _convert_lif_to_binary(self) -> Dict[str, Any]:
        if self.file_info is None or self.file_info.format != InputFormat.LIF:
            raise RuntimeError("LIF input is required for binary conversion")

        self._invalidate_registration_completion("the LIF binaries are being rebuilt")
        return lif_to_suite2p_binary(
            lif_path=self.file_info.path_str,
            output_dir=self.data_path,
            series_index=self.config.LIF_SERIES_INDEX,
            channel_index=None,
            plane_index=self.config.LIF_PLANE_INDEX,
        )

    def _registration_configuration_matches(self) -> bool:
        if self.suite2p_options is None:
            return False

        ops_path = self._suite2p_output_dir() / "plane0" / "ops.npy"
        existing_ops = self._load_suite2p_ops(ops_path)
        if not existing_ops:
            logger.info(
                "Completed registration has no readable ops metadata; rebuilding inputs"
            )
            return False

        mismatches = {
            key: {
                "completed": self._json_safe(existing_ops.get(key)),
                "requested": self._json_safe(self.suite2p_options.get(key)),
            }
            for key in self._REGISTRATION_COMPATIBILITY_KEYS
            if existing_ops.get(key) != self.suite2p_options.get(key)
        }
        if mismatches:
            logger.info(
                "Registration configuration changed; rebuilding inputs: %s",
                mismatches,
            )
            return False
        return True

    def _remove_registration_inputs(self) -> None:
        suite2p_dir = self._suite2p_output_dir()
        self._invalidate_registration_completion("registration is being rebuilt")

        for plane_path in suite2p_dir.glob("plane*"):
            if not plane_path.is_dir():
                continue
            for filename in self._REGISTRATION_INPUT_FILENAMES:
                input_path = plane_path / filename
                if input_path.exists():
                    input_path.unlink()
                    logger.debug("Removed stale registration input: %s", input_path)

    def _rebuild_registration_inputs(self) -> None:
        if self.file_info is None:
            raise RuntimeError("Input must be detected before rebuilding registration")

        self._remove_registration_inputs()
        if self.file_info.format == InputFormat.LIF:
            self._convert_lif_to_binary()
        else:
            logger.info(
                "Removed Suite2p binaries so they will be recreated from the source TIFF"
            )

    def detect_input(self) -> None:
        """Detect input file in the data directory."""
        logger.info("Detecting input file...")
        self.file_info = detect_input_file(
            self.data_path, format_priority=self.config.FILE_FORMAT_PRIORITY
        )

    def load_metadata(self) -> None:
        """Load metadata from the detected input file."""
        if self.file_info is None:
            raise RuntimeError("Must call detect_input() before load_metadata()")

        logger.info("Loading metadata...")
        self.metadata = load_metadata(self.file_info)

    def prepare_data(self) -> None:
        """
        Prepare data for Suite2p processing.

        For LIF files, this converts to binary format if not already converted.
        For TIFF files, no preparation is needed.
        """
        if self.file_info is None or self.metadata is None:
            raise RuntimeError("Must call detect_input() and load_metadata() first")

        if self.file_info.format == InputFormat.LIF:
            # Check if LIF has already been converted to binary
            suite2p_path = Path(self.data_path) / "suite2p" / "plane0"
            data_bin_path = suite2p_path / "data.bin"
            data_chan2_bin_path = suite2p_path / "data_chan2.bin"
            ops_npy_path = suite2p_path / "ops.npy"

            if data_bin_path.exists() and ops_npy_path.exists():
                try:
                    existing_ops = np.load(ops_npy_path, allow_pickle=True).item()
                    expected_channels = min(int(self.metadata.nchannels), 2)
                    existing_channels = int(existing_ops.get("nchannels", 1))
                    needs_chan2 = expected_channels > 1
                    if needs_chan2 and (
                        existing_channels < 2 or not data_chan2_bin_path.exists()
                    ):
                        logger.info(
                            "Existing LIF binary is single-channel; re-converting "
                            "source LIF with two-channel support"
                        )
                        ops_from_lif = self._convert_lif_to_binary()
                    else:
                        logger.info("LIF file already converted to Suite2p binary format")
                        ops_from_lif = existing_ops
                except Exception as e:
                    logger.warning(f"Failed to load existing ops.npy: {e}")
                    logger.info("Re-converting LIF file...")
                    ops_from_lif = self._convert_lif_to_binary()
            else:
                logger.info("Converting LIF to Suite2p binary format...")
                ops_from_lif = self._convert_lif_to_binary()
                logger.info("LIF conversion completed")

            converted_channels = int(ops_from_lif.get("nchannels", 1))
            if self.registration_channel >= converted_channels:
                raise ValueError(
                    f"Registration channel {self.registration_channel} is out of "
                    f"range for converted LIF data with {converted_channels} channel(s)"
                )
            batch_params = self.config.calculate_batch_params(
                int(ops_from_lif["nframes"])
            )
            suite2p_overrides = {
                "input_format": "binary",
                "look_one_level_down": None,
                "Lys": ops_from_lif.get("Lys", [ops_from_lif["Ly"]]),
                "Lxs": ops_from_lif.get("Lxs", [ops_from_lif["Lx"]]),
                "nchannels": converted_channels,
                "nframes": int(ops_from_lif["nframes"]),
                "reg_file": ops_from_lif.get("reg_file", str(data_bin_path)),
                "functional_chan": 1,
                "align_by_chan": self.registration_channel + 1,
                "do_regmetrics": self.do_regmetrics,
                "reg_tif_chan2": self.reg_tif and converted_channels > 1,
                **batch_params,
            }
            if converted_channels > 1:
                reg_file_chan2 = ops_from_lif.get(
                    "reg_file_chan2", str(data_chan2_bin_path)
                )
                if not Path(reg_file_chan2).exists():
                    raise FileNotFoundError(
                        "Two-channel LIF conversion is missing reg_file_chan2"
                    )
                suite2p_overrides["reg_file_chan2"] = reg_file_chan2

            # Build Suite2p options for binary input
            self.suite2p_options = self.config.build_suite2p_options(
                metadata=self.metadata,
                reg_tif=self.reg_tif,
                **suite2p_overrides,
            )
            logger.info("Using binary input format for LIF-converted data")
        else:
            self._validate_supported_channel_count()
            self._validate_channel_index(self.registration_channel, "Registration")
            # TIFF files - no conversion needed
            self.suite2p_options = self.config.build_suite2p_options(
                metadata=self.metadata,
                reg_tif=self.reg_tif,
                align_by_chan=self.registration_channel + 1,
                do_regmetrics=self.do_regmetrics,
                reg_tif_chan2=self.reg_tif and self.metadata.nchannels > 1,
            )

    def run_registration(self, force: bool = False) -> bool:
        """Run Suite2p motion correction.

        Args:
            force: If True, run registration even if already complete

        Returns:
            True if registration was performed, False if skipped
        """
        if self.suite2p_options is None:
            raise RuntimeError("Must call prepare_data() before run_registration()")

        registration_complete = check_registration_complete(
            self.data_path, self.suite2p_options
        )
        if registration_complete and not force:
            if self._registration_configuration_matches():
                logger.info("Registration already complete - skipping")
                return False
            self._rebuild_registration_inputs()
        elif force:
            logger.info("Force registration requested; rebuilding inputs from source")
            self._rebuild_registration_inputs()

        logger.info("Starting motion correction...")
        logger.debug(f"Suite2p options: {self.suite2p_options}")

        do_registration(self.data_path, self.suite2p_options)
        logger.info("Registration completed")
        return True

    def create_projections(self) -> Dict[str, np.ndarray]:
        """Create projection images from registered data."""
        if self.suite2p_options is None:
            raise RuntimeError(
                "Must complete prepare_data() before creating projections"
            )

        logger.info("Creating projections...")
        batch_size = self.suite2p_options.get("batch_size", 500)

        suite2p_path = os.path.join(self.data_path, "suite2p")
        self.projections = create_projections(suite2p_path, batch_size=batch_size)

        logger.info("Projections created")
        return self.projections

    def _projection_key(self, projection_type: str, channel: int) -> str:
        channel_key = f"{projection_type}_ch{channel}"
        if self.projections is not None and channel_key in self.projections:
            return channel_key
        if channel == 0 and self.projections is not None and projection_type in self.projections:
            return projection_type
        raise ValueError(
            f"Projection '{projection_type}' for channel {channel} is not available. "
            f"Available projections: {', '.join(sorted(self.projections or {}))}"
        )

    def _build_segmentation_image(
        self, projection_type: str, segmentation_channel: str
    ) -> tuple[np.ndarray, str, dict[str, Any]]:
        if self.projections is None:
            raise RuntimeError("Must create projections before segmentation")

        available_channels = self._available_channel_count()
        if segmentation_channel == "both" and available_channels > 1:
            images = [
                self._normalize_image(
                    self.projections[self._projection_key(projection_type, channel)]
                )
                for channel in range(min(available_channels, 2))
            ]
            return np.stack(images, axis=-1), "both", {"channel_axis": -1}

        channel = 0 if segmentation_channel == "both" else int(segmentation_channel)
        self._validate_channel_index(channel, "Segmentation")
        image = self.projections[self._projection_key(projection_type, channel)]
        return image, f"ch{channel}", {}

    def segment_cells(
        self,
        interactive_correction: bool = False,
        projection_type: str = "mean",
        segmentation_channel: str = "both",
    ) -> np.ndarray:
        """Segment cells using Cellpose.

        Args:
            interactive_correction: If True, prompts user to manually correct masks
                                   in Cellpose before continuing
            projection_type: Projection image to segment on ("mean" or "max_projection")
            segmentation_channel: "both", "0", or "1"
        """
        if self.projections is None:
            raise RuntimeError("Must create projections before segmentation")

        if self.metadata is None:
            raise RuntimeError("Metadata is required for segmentation")

        segmentation_image, channel_suffix, segmentation_kwargs = (
            self._build_segmentation_image(projection_type, segmentation_channel)
        )

        logger.info(
            "Segmenting cells on %s projection using channel mode %s...",
            projection_type,
            segmentation_channel,
        )

        save_name = f"{projection_type}_{channel_suffix}_image"
        save_path = os.path.join(self.data_path, "suite2p", "plane0", save_name)
        self.segmentation_base_path = save_path
        diameter = self.config.calculate_diameter(self.metadata.pix_resolution)

        self.masks = self.segmenter.segment_img(
            segmentation_image,
            save_path,
            diameter=diameter,
            **segmentation_kwargs,
        )

        labels = np.unique(self.masks)
        labels = labels[labels != 0]
        logger.info(f"Found {len(labels)} cell masks")

        if interactive_correction:
            self._prompt_for_manual_correction(save_path)

        return self.masks

    def _prompt_for_manual_correction(self, seg_path: str) -> None:
        """
        Prompt user to manually correct masks in Cellpose.

        Args:
            seg_path: Path to the segmentation file (without _seg.npy extension)
        """
        seg_file = f"{seg_path}_seg.npy"

        print("\n" + "=" * 70)
        print("MANUAL CORRECTION REQUIRED")
        print("=" * 70)
        print(f"\nAutomatic segmentation completed.")
        print(f"Segmentation file saved at:\n  {seg_file}\n")
        print("Please follow these steps:")
        print("  1. Open the Cellpose GUI")
        print("  2. Load the segmentation file")
        print("  3. Inspect and correct the masks as needed")
        print("  4. Save the corrected masks (overwrite the same file)")
        print("  5. Return here and confirm\n")
        print("=" * 70)

        while True:
            response = (
                input("\nHave you completed manual correction? (yes/no): ")
                .strip()
                .lower()
            )
            if response in ["yes", "y"]:
                logger.info("User confirmed manual correction completed")
                self.load_manual_corrections()
                break
            elif response in ["no", "n"]:
                logger.info(
                    "User skipped manual correction, using automatic segmentation"
                )
                break
            else:
                print("Please answer 'yes' or 'no'")

    def load_manual_corrections(self) -> np.ndarray:
        """
        Load manually corrected masks.

        Returns:
            Corrected masks array
        """
        logger.info("Loading manually corrected masks...")

        if self.segmentation_base_path:
            mask_path = f"{self.segmentation_base_path}_seg.npy"
        else:
            mask_path = os.path.join(
                self.data_path, "suite2p", "plane0", "mean_image_seg.npy"
            )

        self.masks = np.load(mask_path, allow_pickle=True).item()["masks"]
        logger.info(f"Loaded corrected masks with {len(np.unique(self.masks))} labels")

        return self.masks

    def classify_cells(self) -> Any:
        """Classify cell morphology."""
        if self.masks is None:
            raise RuntimeError("Must segment cells before classification")

        if self.metadata is None:
            raise RuntimeError("Metadata is required for classification")

        logger.info("Classifying astrocyte morphology...")

        diameter = self.config.calculate_diameter(self.metadata.pix_resolution)
        neck_distance = self.config.calculate_neck_distance(diameter)
        self.neck_distance = neck_distance

        self.classification = classify_cells(
            masks=self.masks, neck_distance=neck_distance
        )

        logger.info(f"Classification completed: {self.classification}")
        return self.classification

    def export_correspondence_data(
        self,
        segment_length: int = 10,
        delta_x: float = 20.0,
        subsegmentation_mode: str = SUBSEGMENTATION_MODE_EQUAL_LENGTH,
        mask_filename: str = "subsegmented_masks_seg.npy",
        trace_channels: Optional[Sequence[int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create correspondence matrix, subsegment masks, and extract traces."""

        if self.masks is None or self.classification is None:
            raise RuntimeError(
                "Must complete segmentation and classification before exporting correspondence"
            )

        classification_rows = (
            self.classification[0]
            if isinstance(self.classification, tuple)
            else self.classification
        )
        if not classification_rows:
            logger.warning("No cells were segmented; skipping correspondence export")
            return None

        if self.segmentation_base_path:
            template_seg_path = Path(f"{self.segmentation_base_path}_seg.npy")
        else:
            template_seg_path = (
                Path(self.data_path) / "suite2p" / "plane0" / "mean_image_seg.npy"
            )
        if not template_seg_path.exists():
            raise FileNotFoundError(
                f"Cannot locate segmentation file for template metadata: {template_seg_path}"
            )

        logger.info(
            "Exporting correspondence data (segment_length=%d, delta_x=%.2f, mode=%s)",
            segment_length,
            delta_x,
            subsegmentation_mode,
        )

        neck_distance = self.neck_distance
        if neck_distance is None and self.metadata is not None:
            diameter = self.config.calculate_diameter(self.metadata.pix_resolution)
            neck_distance = self.config.calculate_neck_distance(diameter)
        subsegment_pixel_length = round(segment_length * self.metadata.pix_resolution)

        outputs = export_correspondence_products(
            data_path=Path(self.data_path) / "suite2p" / "plane0",
            template_seg_path=template_seg_path,
            masks=self.masks,
            classifications=classification_rows,
            segment_length=subsegment_pixel_length,
            delta_x=delta_x,
            subsegmentation_mode=subsegmentation_mode,
            neck_distance=neck_distance,
            mask_filename=mask_filename,
            trace_channels=trace_channels,
        )
        if outputs is None:
            logger.info("Correspondence export skipped")
            return None

        logger.info("Correspondence export completed")
        return outputs

    @staticmethod
    def _timestamp_from_path(path: Path) -> Optional[str]:
        if not path.exists():
            return None
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")

    @staticmethod
    def _package_version(package_name: str) -> Optional[str]:
        try:
            return importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            return None

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): Pipeline._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [Pipeline._json_safe(item) for item in value]
        return value

    @staticmethod
    def _array_shape(value: Any) -> Optional[list[int]]:
        if value is None:
            return None
        return [int(size) for size in np.asarray(value).shape]

    @staticmethod
    def _array_stats(values: Any) -> Dict[str, Optional[float]]:
        if values is None:
            return {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
            }

        array = np.asarray(values, dtype=float).ravel()
        array = array[np.isfinite(array)]
        if array.size == 0:
            return {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
            }

        return {
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
        }

    def _load_suite2p_ops(self, ops_path: Path) -> Dict[str, Any]:
        if not ops_path.exists():
            return {}
        try:
            return np.load(ops_path, allow_pickle=True).item()
        except Exception as exc:
            logger.warning(
                "Failed to load Suite2p ops metadata from %s: %s", ops_path, exc
            )
            return {}

    def _lif_series_name(self, series_index: int) -> Optional[str]:
        if self.file_info is None or self.file_info.format != InputFormat.LIF:
            return None

        try:
            from readlif.reader import LifFile

            lif = LifFile(self.file_info.path_str)
            if series_index >= len(lif.image_list):
                return None
            series_name = lif.image_list[series_index].get("name")
            return str(series_name) if series_name is not None else None
        except Exception as exc:
            logger.debug(
                "Could not read LIF series name from %s: %s",
                self.file_info.path_str,
                exc,
            )
            return None

    def _input_file_metadata_payload(self) -> Dict[str, Any]:
        if self.file_info is None:
            return {
                "input_file": None,
                "input_filename": None,
                "input_file_extension": None,
                "input_file_type": None,
                "input_file_size_bytes": None,
                "input_file_modified_time": None,
                "series_index": None,
                "series_name": None,
                "plane_index": None,
            }

        input_path = self.file_info.path
        series_index = (
            self.config.LIF_SERIES_INDEX
            if self.file_info.format == InputFormat.LIF
            else None
        )
        plane_index = (
            self.config.LIF_PLANE_INDEX
            if self.file_info.format == InputFormat.LIF
            else None
        )
        series_name = getattr(self.metadata, "series_name", None)

        return {
            "input_file": str(input_path),
            "input_filename": input_path.name,
            "input_file_extension": input_path.suffix,
            "input_file_type": self.file_info.format.value,
            "input_file_size_bytes": (
                input_path.stat().st_size if input_path.exists() else None
            ),
            "input_file_modified_time": self._timestamp_from_path(input_path),
            "series_index": series_index,
            "series_name": series_name,
            "plane_index": plane_index,
        }

    def _registration_qc_payload(
        self,
        ops: Dict[str, Any],
        ops_path: Path,
        suite2p_dir: Path,
    ) -> Dict[str, Any]:
        nframes_registered = ops.get(
            "nframes",
            self.suite2p_options.get("nframes") if self.suite2p_options else None,
        )
        badframes = ops.get("badframes")
        if badframes is None:
            num_badframes = None
            badframes_fraction = None
        else:
            badframes_array = np.asarray(badframes, dtype=bool)
            num_badframes = int(np.sum(badframes_array))
            denominator = int(badframes_array.size or nframes_registered or 0)
            badframes_fraction = (
                float(num_badframes / denominator) if denominator > 0 else None
            )

        xoff_stats = self._array_stats(ops.get("xoff"))
        yoff_stats = self._array_stats(ops.get("yoff"))
        corrxy_stats = self._array_stats(ops.get("corrXY"))
        flag_path = suite2p_dir / ".registration_complete"

        return {
            "registration_complete": flag_path.exists(),
            "registration_completed_at": self._timestamp_from_path(flag_path),
            "suite2p_output_dir": str(suite2p_dir),
            "ops_path": str(ops_path),
            "meanImg_shape": self._array_shape(ops.get("meanImg")),
            "meanImg_chan2_shape": self._array_shape(ops.get("meanImg_chan2")),
            "refImg_shape": self._array_shape(ops.get("refImg")),
            "num_badframes": num_badframes,
            "badframes_fraction": badframes_fraction,
            "xoff_min": xoff_stats["min"],
            "xoff_max": xoff_stats["max"],
            "xoff_mean": xoff_stats["mean"],
            "xoff_std": xoff_stats["std"],
            "yoff_min": yoff_stats["min"],
            "yoff_max": yoff_stats["max"],
            "yoff_mean": yoff_stats["mean"],
            "yoff_std": yoff_stats["std"],
            "corrXY_mean": corrxy_stats["mean"],
            "corrXY_min": corrxy_stats["min"],
            "corrXY_max": corrxy_stats["max"],
            "suite2p_timing": self._json_safe(ops.get("timing")),
        }

    def write_pipeline_metadata(self) -> None:
        if self.suite2p_options is None:
            return

        suite2p_dir = get_suite2p_output_dir(self.data_path, self.suite2p_options)
        plane_path = suite2p_dir / "plane0"
        metadata_path = plane_path / "pipeline_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        ops_path = plane_path / "ops.npy"
        ops = self._load_suite2p_ops(ops_path)

        metadata = self.metadata
        nframes_registered = ops.get("nframes", self.suite2p_options.get("nframes"))
        channel_indices = ops.get(
            "channel_indices",
            self.suite2p_options.get("channel_indices"),
        )

        payload = {
            **self._input_file_metadata_payload(),
            "source_nframes": metadata.nframes if metadata else None,
            "nframes_registered": (
                int(nframes_registered) if nframes_registered is not None else None
            ),
            "source_nchannels": metadata.nchannels if metadata else None,
            "converted_nchannels": self._available_channel_count(),
            "nchannels": self._available_channel_count(),
            "channel_indices": self._json_safe(channel_indices),
            "nplanes": (
                metadata.nplanes if metadata else self.suite2p_options.get("nplanes")
            ),
            "Ly": self._json_safe(ops.get("Ly", self.suite2p_options.get("Ly"))),
            "Lx": self._json_safe(ops.get("Lx", self.suite2p_options.get("Lx"))),
            "pixel_resolution": metadata.pix_resolution if metadata else None,
            "frame_interval_seconds": metadata.finterval if metadata else None,
            "fs": metadata.fs if metadata else self.suite2p_options.get("fs"),
            "frames_per_channel_per_plane": (
                metadata.frames_per_channel_per_plane if metadata else None
            ),
            "do_registration": self.suite2p_options.get("do_registration"),
            "two_step_registration": self.suite2p_options.get("two_step_registration"),
            "nonrigid": self.suite2p_options.get("nonrigid"),
            "maxregshift": self.suite2p_options.get("maxregshift"),
            "subpixel": self.suite2p_options.get("subpixel"),
            "align_by_chan": self.suite2p_options.get("align_by_chan"),
            "functional_chan": self.suite2p_options.get("functional_chan"),
            "batch_size": self.suite2p_options.get("batch_size"),
            "nimg_init": self.suite2p_options.get("nimg_init"),
            "do_regmetrics": self.suite2p_options.get(
                "do_regmetrics", self.do_regmetrics
            ),
            "reg_tif": self.suite2p_options.get("reg_tif", self.reg_tif),
            "reg_tif_chan2": self.suite2p_options.get("reg_tif_chan2"),
            "roidetect": self.suite2p_options.get("roidetect"),
            "spikedetect": self.suite2p_options.get("spikedetect"),
            "input_format": self.suite2p_options.get("input_format"),
            **self._registration_qc_payload(ops, ops_path, suite2p_dir),
            "registration_channel": self.registration_channel,
            "suite2p_align_by_chan": self.suite2p_options.get("align_by_chan"),
            "segmentation_channel": self.segmentation_channel,
            "segmentation_projection": self.segmentation_projection,
            "trace_channels": self.trace_channels,
            "alignment_only": self.alignment_only,
            "export_correspondence": self.export_correspondence,
            "manual_correction": self.manual_correction,
            "model_path": self.model_path,
            "use_gpu": self.use_gpu,
            "reg_file": self._json_safe(
                ops.get("reg_file", self.suite2p_options.get("reg_file"))
            ),
            "reg_file_chan2": self._json_safe(
                ops.get("reg_file_chan2", self.suite2p_options.get("reg_file_chan2"))
            ),
            "pipeline_version": self._package_version("astroglial-morphology"),
            "python_version": sys.version.split()[0],
            "suite2p_version": self._json_safe(
                ops.get("suite2p_version") or self._package_version("suite2p")
            ),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "hostname": socket.gethostname(),
        }
        payload = self._json_safe(payload)
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved pipeline metadata to %s", metadata_path)

    def run(
        self,
        skip_registration: bool = False,
        force_registration: bool = False,
        manual_correction: bool = False,
        export_correspondence: bool = False,
        correspondence_segment_length: int = 100,
        correspondence_delta_x: float = 20.0,
        correspondence_subsegmentation_mode: str = SUBSEGMENTATION_MODE_EQUAL_LENGTH,
        segmentation_projection: str = "mean",
        segmentation_channel: str = "both",
        registration_channel: int = 0,
        trace_channels: Optional[Union[str, Sequence[int]]] = None,
        do_regmetrics: bool = False,
        alignment_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline.

        Args:
            skip_registration: If True, unconditionally skip registration (deprecated, auto-detected now)
            force_registration: If True, run registration even if already complete
            manual_correction: If True, load manually corrected masks
            export_correspondence: Build correspondence/trace outputs when True
            correspondence_segment_length: Segment length in pixels for subsegmentation
            correspondence_delta_x: X-axis grouping distance for correspondence alignment
            correspondence_subsegmentation_mode: Strategy for subsegmenting cells
            segmentation_channel: Channel mode for Cellpose input ("both", "0", or "1")
            registration_channel: Zero-based channel used to calculate registration shifts
            trace_channels: Zero-based channels to export traces for
            do_regmetrics: Whether Suite2p should compute optional registration metrics
            alignment_only: Stop after registration and projection creation

        Returns:
            Dictionary with pipeline results
        """
        logger.info("Starting astroglial morphology pipeline")
        self.registration_channel = int(registration_channel)
        self.segmentation_channel = str(segmentation_channel)
        self.trace_channels = self._normalize_trace_channels(trace_channels)
        self.segmentation_projection = segmentation_projection
        self.do_regmetrics = do_regmetrics
        self.manual_correction = manual_correction
        self.export_correspondence = export_correspondence
        self.alignment_only = alignment_only
        should_export_correspondence = export_correspondence and not alignment_only

        # Step 1: Detect input
        self.detect_input()

        # Step 2: Load metadata
        self.load_metadata()
        self._validate_run_channels(should_export_correspondence)

        # Step 3: Prepare data (always needed for suite2p_options)
        self.prepare_data()
        self._validate_run_channels(should_export_correspondence)

        # Step 4: Run registration (auto-detected or forced)
        if skip_registration:
            logger.info("Registration explicitly skipped by user")
        else:
            self.run_registration(force=force_registration)

        # Step 5: Create projections
        self.create_projections()
        self.write_pipeline_metadata()

        if alignment_only:
            logger.info("Alignment-only mode requested; stopping before segmentation")
            return {
                "file_info": self.file_info,
                "metadata": self.metadata,
                "projections": self.projections,
                "masks": None,
                "classification": None,
                "correspondence": None,
            }

        # Step 6: Segment cells (with optional interactive correction)
        self.segment_cells(
            interactive_correction=manual_correction,
            projection_type=segmentation_projection,
            segmentation_channel=self.segmentation_channel,
        )

        # Step 7: Classify cells
        self.classify_cells()

        correspondence_outputs = None
        if should_export_correspondence:
            correspondence_outputs = self.export_correspondence_data(
                segment_length=correspondence_segment_length,
                delta_x=correspondence_delta_x,
                subsegmentation_mode=correspondence_subsegmentation_mode,
                trace_channels=self.trace_channels,
            )

        logger.info("Pipeline completed successfully")

        return {
            "file_info": self.file_info,
            "metadata": self.metadata,
            "projections": self.projections,
            "masks": self.masks,
            "classification": self.classification,
            "correspondence": correspondence_outputs,
        }
