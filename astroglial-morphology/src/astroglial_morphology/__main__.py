"""
Main entry point for the astroglial morphology pipeline.

Usage:
    python -m astroglial_morphology <data_path> [--reg-tif] [--model PATH] [--gpu]
"""

import argparse
import sys
from pathlib import Path

from astroglial_morphology import setup_logging, get_logger
from astroglial_morphology.correspondence import (
    VALID_SUBSEGMENTATION_MODES,
    SUBSEGMENTATION_MODE_EQUAL_LENGTH,
)
from astroglial_morphology.pipeline import Pipeline


def main() -> None:
    """Run the astroglial morphology analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Astroglial morphology analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process TIFF or LIF file (auto-detects if registration already done)
  python -m astroglial_morphology /path/to/data
  
  # Force re-registration even if already complete
  python -m astroglial_morphology /path/to/data --force-registration
  
  # Save registered TIFF (large files!)
  python -m astroglial_morphology /path/to/data --reg-tif
  
  # Use custom model
  python -m astroglial_morphology /path/to/data --model /path/to/model
  
  # Use GPU for segmentation
  python -m astroglial_morphology /path/to/data --gpu
        """,
    )
    parser.add_argument(
        "data_path",
        help="Path to directory containing input files (.tif or .lif)",
    )
    parser.add_argument(
        "--reg-tif",
        action="store_true",
        help="Save registered TIFF files (warning: creates large files)",
        default=False,
    )
    parser.add_argument(
        "--model",
        help="Path to custom Cellpose model (optional)",
        default=None,
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for segmentation",
        default=False,
    )
    parser.add_argument(
        "--skip-registration",
        action="store_true",
        help="Unconditionally skip registration step (deprecated - auto-detected now)",
        default=False,
    )
    parser.add_argument(
        "--force-registration",
        action="store_true",
        help="Force registration even if already complete",
        default=False,
    )
    parser.add_argument(
        "--alignment-only",
        action="store_true",
        help="Stop after registration and projection creation",
        default=False,
    )
    parser.add_argument(
        "--registration-channel",
        type=int,
        choices=[0, 1],
        default=0,
        help="Zero-based channel used to calculate registration shifts",
    )
    parser.add_argument(
        "--regmetrics",
        action="store_true",
        help="Enable Suite2p registration metrics (can require substantial memory)",
        default=False,
    )
    parser.add_argument(
        "--manual-correction",
        action="store_true",
        help="Load manually corrected masks instead of running segmentation",
        default=False,
    )
    parser.add_argument(
        "--export-correspondence",
        action="store_true",
        help="Generate correspondence matrix, subsegmented masks, and trace exports",
        default=True,
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=5,
        help="Segment length of subsegment (in microns) used when exporting correspondence data",
    )
    parser.add_argument(
        "--correspondence-delta-x",
        type=float,
        default=20.0,
        help="Maximum x-distance for grouping cells while aligning correspondence",
    )
    parser.add_argument(
        "--subsegmentation-mode",
        choices=sorted(VALID_SUBSEGMENTATION_MODES),
        default=SUBSEGMENTATION_MODE_EQUAL_LENGTH,
        help="How to subsegment each cell: equal_length (fixed pixels) or compartments (soma/middle/distal)",
    )
    parser.add_argument(
        "--segmentation-image",
        choices=["mean", "max_projection"],
        default="mean",
        help="Projection image to segment on: mean or max_projection",
    )
    parser.add_argument(
        "--segmentation-channel",
        choices=["both", "0", "1"],
        default="both",
        help="Channel mode to segment: both, 0, or 1",
    )
    parser.add_argument(
        "--trace-channels",
        default=None,
        help="Comma-separated zero-based channels to export traces for, e.g. 0 or 0,1",
    )

    args = parser.parse_args()

    setup_logging()
    logger = get_logger(__name__)

    # Validate data path
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data path does not exist: {data_path}")
        sys.exit(1)

    if not data_path.is_dir():
        logger.error(f"Data path is not a directory: {data_path}")
        sys.exit(1)

    try:
        pipeline = Pipeline(
            data_path=str(data_path),
            model_path=args.model,
            use_gpu=args.gpu,
            reg_tif=args.reg_tif,
        )

        results = pipeline.run(
            skip_registration=args.skip_registration,
            force_registration=args.force_registration,
            manual_correction=args.manual_correction,
            export_correspondence=args.export_correspondence,
            correspondence_segment_length=args.segment_length,
            correspondence_delta_x=args.correspondence_delta_x,
            correspondence_subsegmentation_mode=args.subsegmentation_mode,
            segmentation_projection=args.segmentation_image,
            segmentation_channel=args.segmentation_channel,
            registration_channel=args.registration_channel,
            trace_channels=args.trace_channels,
            do_regmetrics=args.regmetrics,
            alignment_only=args.alignment_only,
        )

        logger.info("Pipeline completed successfully")
        logger.info(f"Results: {results['classification']}")
        if args.export_correspondence and results.get("correspondence"):
            corr_outputs = results["correspondence"]
            logger.info(
                "Correspondence matrix: %s (npy), %s (mat)",
                corr_outputs["correspondence_matrix_path"],
                corr_outputs["correspondence_matrix_mat_path"],
            )
            trace_paths = corr_outputs.get("trace_matrix_paths")
            trace_mat_paths = corr_outputs.get("trace_matrix_mat_paths")
            if trace_paths and trace_mat_paths:
                logger.info(
                    "Trace matrices: %s (npy), %s (mat)",
                    trace_paths,
                    trace_mat_paths,
                )
            else:
                logger.info(
                    "Trace matrix: %s (npy), %s (mat)",
                    corr_outputs["trace_matrix_path"],
                    corr_outputs["trace_matrix_mat_path"],
                )

    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
