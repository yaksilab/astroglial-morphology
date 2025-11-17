"""Utilities for building correspondence matrices and exporting trace data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Tuple, Dict, Any, Optional

import numpy as np
from scipy.io import savemat

from astroglial_analysis.determine_line import get_line, uniform_align_comp_cell
from astroglial_analysis.sub_segmentation import sub_segment
from astroglial_analysis.create_masks import create_cp_mask
from astroglial_analysis.map_cellpose_suite2p import (
    create_suite2p_cellpose_roi_mapping,
    map_trace,
)

from .logging_config import get_logger

logger = get_logger(__name__)


Classifications = Sequence[Tuple[int, int]]


def _partition_labels(classifications: Classifications) -> tuple[list[int], list[int]]:
    upper = [label for cls, label in classifications if cls == 1]
    lower = [label for cls, label in classifications if cls == 2]
    return upper, lower


def _align_group(
    masks: np.ndarray,
    labels: Sequence[int],
    is_upper: bool,
    delta_x: float,
) -> np.ndarray:
    if not labels:
        return np.empty((0, 5), dtype=np.int32)

    line, _ = get_line(labels, masks, is_upper, delta_x=delta_x)
    if not line:
        logger.warning(
            "No valid regions found for %s group", "upper" if is_upper else "lower"
        )
        return np.empty((0, 5), dtype=np.int32)

    _, correspondence = uniform_align_comp_cell(line, masks, is_upper)
    return correspondence.astype(np.int32)


def build_correspondence_matrix(
    masks: np.ndarray,
    classifications: Classifications,
    delta_x: float = 20.0,
) -> np.ndarray:
    """Construct correspondence matrix for upper/lower complete cells only."""
    upper_labels, lower_labels = _partition_labels(classifications)

    matrices = []
    upper_count = 0
    lower_count = 0

    if upper_labels:
        upper_matrix = _align_group(masks, upper_labels, True, delta_x)
        if upper_matrix.size:
            class_upper = np.full((upper_matrix.shape[0], 1), 1, dtype=np.int16)
            matrices.append(np.hstack((upper_matrix, class_upper)))
            upper_count = upper_matrix.shape[0]
        else:
            logger.warning("Upper group alignment produced no correspondence rows")

    if lower_labels:
        lower_matrix = _align_group(masks, lower_labels, False, delta_x)
        if lower_matrix.size:
            class_lower = np.full((lower_matrix.shape[0], 1), 2, dtype=np.int16)
            matrices.append(np.hstack((lower_matrix, class_lower)))
            lower_count = lower_matrix.shape[0]
        else:
            logger.warning("Lower group alignment produced no correspondence rows")

    if not matrices:
        raise ValueError(
            "No classified cells available to build correspondence matrix."
        )

    total = matrices[0] if len(matrices) == 1 else np.vstack(matrices)
    total = total.astype(np.int32)

    logger.info(
        "Built correspondence matrix with %d points (upper=%d, lower=%d)",
        total.shape[0],
        upper_count,
        lower_count,
    )
    return total


def _save_as_mat(matrix: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(path), {"matrix": matrix})
    return path


def _save_subsegmented_mask(
    template_seg_path: Path,
    destination: Path,
    subsegmented_mask: np.ndarray,
) -> None:
    template = np.load(template_seg_path, allow_pickle=True)
    metadata = template.item().copy()
    metadata["masks"] = subsegmented_mask
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, metadata)


def export_correspondence_products(
    *,
    data_path: Path,
    template_seg_path: Path,
    masks: np.ndarray,
    classifications: Classifications,
    segment_length: int = 10,
    delta_x: float = 20.0,
    mask_filename: str = "subsegmented_masks_seg.npy",
) -> Optional[Dict[str, Any]]:
    """Run the full correspondence workflow and write artifacts to disk."""

    data_path = Path(data_path)
    template_seg_path = Path(template_seg_path)

    try:
        correspondence_matrix = build_correspondence_matrix(
            masks=masks, classifications=classifications, delta_x=delta_x
        )
    except ValueError as exc:
        logger.warning("Skipping correspondence export: %s", exc)
        return None
    sub_segmented_data = sub_segment(correspondence_matrix, segment_length)
    sub_segmented_data = sub_segmented_data.astype(np.int32)

    subsegmented_mask = create_cp_mask(sub_segmented_data, masks)
    subseg_path = data_path / mask_filename

    _save_subsegmented_mask(template_seg_path, subseg_path, subsegmented_mask)
    logger.info("Saved subsegmented masks to %s", subseg_path)

    try:
        from astroglial_segmentation import create_suite2p_masks_extract_traces
    except ImportError as exc:  # pragma: no cover - optional dependency runtime check
        raise ImportError(
            "The astroglial_segmentation package is required to extract traces. "
            "Install it or update your environment."
        ) from exc

    create_suite2p_masks_extract_traces(str(data_path), mask_filename)
    logger.info("Created Suite2p traces for subsegmented masks")

    suite2p_folder = data_path / "cellpose_suite2p_output"
    mapping = create_suite2p_cellpose_roi_mapping(
        subsegmented_mask, str(suite2p_folder)
    )
    reversed_mapping = {
        cp_label: s2p_idx
        for s2p_idx, cp_label in mapping.items()
        if cp_label is not None
    }

    traces_path = suite2p_folder / "F.npy"
    if not traces_path.exists():
        raise FileNotFoundError("Suite2p trace file F.npy not found after extraction.")

    traces = np.load(traces_path, allow_pickle=True)
    mapped_traces_matrix = map_trace(traces, mapping)

    suite2p_column = np.array(
        [reversed_mapping.get(label, -1) for label in sub_segmented_data[:, 1]],
        dtype=np.int32,
    )
    mapped_subsegmented_data = np.column_stack(
        (suite2p_column, sub_segmented_data)
    ).astype(np.int32)

    outputs = {
        "subsegmented_mask_path": subseg_path,
        "correspondence_matrix": correspondence_matrix,
        "sub_segmented_data": sub_segmented_data,
        "mapped_subsegmented_data": mapped_subsegmented_data,
        "mapped_traces_matrix": mapped_traces_matrix,
    }

    trace_npy = data_path / "trace_matrix.npy"
    corr_npy = data_path / "correspondence_matrix.npy"
    np.save(trace_npy, mapped_traces_matrix)
    np.save(corr_npy, mapped_subsegmented_data)
    logger.info(
        "Saved trace matrix to %s and correspondence matrix to %s (NumPy format)",
        trace_npy,
        corr_npy,
    )

    trace_mat = data_path / "trace_matrix.mat"
    corr_mat = data_path / "correspondence_matrix.mat"
    _save_as_mat(mapped_traces_matrix, trace_mat)
    _save_as_mat(mapped_subsegmented_data, corr_mat)
    logger.info(
        "Saved trace matrix to %s and correspondence matrix to %s (MAT format)",
        trace_mat,
        corr_mat,
    )

    outputs.update(
        {
            "trace_matrix_path": trace_npy,
            "trace_matrix_mat_path": trace_mat,
            "correspondence_matrix_path": corr_npy,
            "correspondence_matrix_mat_path": corr_mat,
        }
    )

    return outputs
