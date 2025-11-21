"""Main pipeline orchestration for astroglial morphology analysis."""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from .config import PipelineConfig
from .io import detect_input_file, load_metadata, InputFormat
from .registration import do_registration
from .binary_utils import create_projections
from .segmentation import Segmentation
from .classifier import classify_cells
from .correspondence import (
    export_correspondence_products,
    SUBSEGMENTATION_MODE_EQUAL_LENGTH,
)
from .utils.lif_utils import lif_to_suite2p_binary

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main pipeline for astroglial morphology analysis.

    Orchestrates the complete workflow from raw microscopy data to
    classified cell morphologies.
    """

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
        self.metadata = None
        self.suite2p_options = None
        self.projections = None
        self.masks = None
        self.classification = None
        self.neck_distance: Optional[int] = None

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

        For LIF files, this converts to binary format.
        For TIFF files, no preparation is needed.
        """
        if self.file_info is None or self.metadata is None:
            raise RuntimeError("Must call detect_input() and load_metadata() first")

        if self.file_info.format == InputFormat.LIF:
            logger.info("Converting LIF to Suite2p binary format...")
            ops_from_lif = lif_to_suite2p_binary(
                lif_path=self.file_info.path_str,
                output_dir=self.data_path,
                series_index=self.config.LIF_SERIES_INDEX,
                channel_index=self.config.LIF_CHANNEL_INDEX,
                plane_index=self.config.LIF_PLANE_INDEX,
            )
            logger.info("LIF conversion completed")

            # Build Suite2p options for binary input
            self.suite2p_options = self.config.build_suite2p_options(
                metadata=self.metadata,
                reg_tif=self.reg_tif,
                input_format="binary",
                look_one_level_down=None,
                Lys=ops_from_lif["Lys"],
                Lxs=ops_from_lif["Lxs"],
            )
            logger.info("Using binary input format for LIF-converted data")
        else:
            # TIFF files - no conversion needed
            self.suite2p_options = self.config.build_suite2p_options(
                metadata=self.metadata,
                reg_tif=self.reg_tif,
            )

    def run_registration(self) -> None:
        """Run Suite2p motion correction."""
        if self.suite2p_options is None:
            raise RuntimeError("Must call prepare_data() before run_registration()")

        logger.info("Starting motion correction...")
        logger.debug(f"Suite2p options: {self.suite2p_options}")

        do_registration(self.data_path, self.suite2p_options)
        logger.info("Registration completed")

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

    def segment_cells(self, interactive_correction: bool = False) -> np.ndarray:
        """Segment cells using Cellpose.

        Args:
            interactive_correction: If True, prompts user to manually correct masks
                                   in Cellpose before continuing
        """
        if self.projections is None:
            raise RuntimeError("Must create projections before segmentation")

        if self.metadata is None:
            raise RuntimeError("Metadata is required for segmentation")

        logger.info("Segmenting cells on mean projection...")

        save_path = os.path.join(self.data_path, "suite2p", "plane0", "mean_image")
        diameter = self.config.calculate_diameter(self.metadata.pix_resolution)

        self.masks = self.segmenter.segment_img(
            self.projections["mean"], save_path, diameter=diameter
        )

        labels = np.unique(self.masks)
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

        outputs = export_correspondence_products(
            data_path=Path(self.data_path) / "suite2p" / "plane0",
            template_seg_path=template_seg_path,
            masks=self.masks,
            classifications=classification_rows,
            segment_length=segment_length,
            delta_x=delta_x,
            subsegmentation_mode=subsegmentation_mode,
            neck_distance=neck_distance,
            mask_filename=mask_filename,
        )
        if outputs is None:
            logger.info("Correspondence export skipped")
            return None

        logger.info("Correspondence export completed")
        return outputs

    def run(
        self,
        skip_registration: bool = False,
        manual_correction: bool = False,
        export_correspondence: bool = False,
        correspondence_segment_length: int = 100,
        correspondence_delta_x: float = 20.0,
        correspondence_subsegmentation_mode: str = SUBSEGMENTATION_MODE_EQUAL_LENGTH,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline.

        Args:
            skip_registration: If True, skip registration step (assumes already done)
            manual_correction: If True, load manually corrected masks
            export_correspondence: Build correspondence/trace outputs when True
            correspondence_segment_length: Segment length in pixels for subsegmentation
            correspondence_delta_x: X-axis grouping distance for correspondence alignment
            correspondence_subsegmentation_mode: Strategy for subsegmenting cells

        Returns:
            Dictionary with pipeline results
        """
        logger.info("Starting astroglial morphology pipeline")

        # Step 1: Detect input
        self.detect_input()

        # Step 2: Load metadata
        self.load_metadata()

        # Step 3: Prepare data (always needed for suite2p_options)
        self.prepare_data()

        # Step 4: Run registration (unless skipped)
        if not skip_registration:
            self.run_registration()

        # Step 5: Create projections
        self.create_projections()

        # Step 6: Segment cells (with optional interactive correction)
        self.segment_cells(interactive_correction=manual_correction)

        # Step 7: Classify cells
        self.classify_cells()

        correspondence_outputs = None
        if export_correspondence:
            correspondence_outputs = self.export_correspondence_data(
                segment_length=correspondence_segment_length,
                delta_x=correspondence_delta_x,
                subsegmentation_mode=correspondence_subsegmentation_mode,
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
