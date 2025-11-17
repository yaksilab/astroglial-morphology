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
SUBSEGMENTATION_MODE_EQUAL_LENGTH = "equal_length"
SUBSEGMENTATION_MODE_COMPARTMENTS = "compartments"
VALID_SUBSEGMENTATION_MODES = {
    SUBSEGMENTATION_MODE_EQUAL_LENGTH,
    SUBSEGMENTATION_MODE_COMPARTMENTS,
}


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


def _build_subsegment_matrix(
    data_matrix: np.ndarray, subsegment_number: np.ndarray
) -> np.ndarray:
    if data_matrix.size == 0:
        return np.empty((0, data_matrix.shape[1] + 2), dtype=np.int32)

    cell_labels = data_matrix[:, 0].astype(np.int32)
    unique_pairs = np.unique(
        np.column_stack((cell_labels, subsegment_number)), axis=0
    )

    max_cell_label = int(cell_labels.max()) if cell_labels.size else 0
    new_labels_start = max_cell_label + 1
    mapping = {
        (pair[0], pair[1]): new_labels_start + idx
        for idx, pair in enumerate(unique_pairs)
    }

    sub_segment_label = np.array(
        [mapping[(cl, sn)] for cl, sn in zip(cell_labels, subsegment_number)],
        dtype=np.int32,
    )

    new_data = np.column_stack(
        (cell_labels, sub_segment_label, subsegment_number, data_matrix[:, 1:])
    )
    return new_data.astype(np.int32)


def _subsegment_with_compartments(
    data_matrix: np.ndarray, neck_distance: int
) -> np.ndarray:
    if neck_distance is None or neck_distance <= 0:
        raise ValueError("neck_distance must be positive for compartment mode")

    if data_matrix.size == 0:
        return np.empty((0, data_matrix.shape[1] + 2), dtype=np.int32)

    cell_labels = data_matrix[:, 0].astype(np.int32)
    y_rotated = data_matrix[:, 4].astype(np.float32)
    subsegment_number = np.zeros_like(cell_labels, dtype=np.int32)

    for label in np.unique(cell_labels):
        indices = np.where(cell_labels == label)[0]
        cell_y = y_rotated[indices]
        if cell_y.size == 0:
            continue

        max_length = float(cell_y.max())
        soma_limit = min(float(neck_distance), max_length)
        distal_start = max(max_length - float(neck_distance), soma_limit)
        middle_span = max(distal_start - soma_limit, 0.0)
        segments = np.full(cell_y.shape, 4, dtype=np.int32)

        if max_length <= 0:
            segments[:] = 1
        else:
            segments[cell_y <= soma_limit] = 1
            if middle_span > 0:
                mid_split = soma_limit + middle_span / 2.0
                near_soma = (cell_y > soma_limit) & (cell_y <= mid_split)
                near_distal = (cell_y > mid_split) & (cell_y < distal_start)
                segments[near_soma] = 2
                segments[near_distal] = 3
            segments[cell_y >= distal_start] = 4

        subsegment_number[indices] = segments

    return _build_subsegment_matrix(data_matrix, subsegment_number)


def _subsegment_correspondence(
    correspondence_matrix: np.ndarray,
    mode: str,
    segment_length: int,
    neck_distance: Optional[int],
) -> np.ndarray:
    normalized_mode = mode.lower()
    if normalized_mode not in VALID_SUBSEGMENTATION_MODES:
        raise ValueError(
            f"Unsupported subsegmentation mode '{mode}'. "
            f"Valid options are: {', '.join(sorted(VALID_SUBSEGMENTATION_MODES))}"
        )

    if normalized_mode == SUBSEGMENTATION_MODE_EQUAL_LENGTH:
        if segment_length <= 0:
            raise ValueError("segment_length must be positive for equal_length mode")
        return sub_segment(correspondence_matrix, segment_length)

    if neck_distance is None:
        raise ValueError(
            "neck_distance must be provided when using compartment subsegmentation"
        )
    return _subsegment_with_compartments(correspondence_matrix, int(neck_distance))


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
    subsegmentation_mode: str = SUBSEGMENTATION_MODE_EQUAL_LENGTH,
    neck_distance: Optional[int] = None,
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
    sub_segmented_data = _subsegment_correspondence(
        correspondence_matrix,
        mode=subsegmentation_mode,
        segment_length=segment_length,
        neck_distance=neck_distance,
    )
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
