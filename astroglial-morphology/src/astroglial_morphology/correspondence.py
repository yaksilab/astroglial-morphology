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


def _normalize_trace_channels(
    trace_channels: Optional[Sequence[int]],
    nchannels: int,
) -> list[int]:
    if trace_channels is None:
        if nchannels > 1:
            raise ValueError(
                "trace_channels must be specified when exporting traces "
                "from multi-channel data"
            )
        return [0]

    channels = [int(channel) for channel in trace_channels]
    if not channels:
        raise ValueError("At least one trace channel must be selected")
    if len(set(channels)) != len(channels):
        raise ValueError("Duplicate trace channels are not allowed")
    for channel in channels:
        if channel < 0 or channel >= nchannels:
            raise ValueError(
                f"Trace channel {channel} is out of range for {nchannels} channel(s)"
            )
    return channels


def _build_suite2p_stat(masks: np.ndarray, ly: int, lx: int, ops: dict) -> np.ndarray:
    from suite2p.detection import roi_stats
    from suite2p.extraction.masks import create_masks

    stat = []
    for label in np.unique(masks)[1:]:
        ypix, xpix = np.nonzero(masks == label)
        npix = len(ypix)
        stat.append(
            {
                "ypix": ypix,
                "xpix": xpix,
                "npix": npix,
                "lam": np.ones(npix, np.float32),
                "med": [np.mean(ypix), np.mean(xpix)],
            }
        )

    stat = roi_stats(np.array(stat), ly, lx)
    create_masks(stat, ly, lx, ops)
    return stat


def _extract_suite2p_traces_for_channels(
    *,
    data_path: Path,
    mask_filename: str,
    trace_channels: Optional[Sequence[int]],
) -> tuple[Path, dict[int, np.ndarray]]:
    import contextlib
    import os

    from suite2p import extraction_wrapper, classify
    from suite2p.classification import builtin_classfile
    from suite2p.extraction import preprocess, oasis
    from suite2p.io import BinaryFile

    ops_file = data_path / "ops.npy"
    if not ops_file.exists():
        raise FileNotFoundError(f"Ops file not found: {ops_file}")

    ops = np.load(ops_file, allow_pickle=True).item()
    nchannels = int(ops.get("nchannels", 1))
    channels = _normalize_trace_channels(trace_channels, nchannels)
    ly = int(ops["Ly"])
    lx = int(ops["Lx"])
    nframes = int(ops["nframes"])

    mask_path = data_path / mask_filename
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    masks = np.load(mask_path, allow_pickle=True).item()["masks"]
    stat = _build_suite2p_stat(masks, ly, lx, ops)

    data_bin = data_path / "data.bin"
    data_chan2_bin = data_path / "data_chan2.bin"
    if nchannels > 1 and not data_chan2_bin.exists():
        raise FileNotFoundError(
            f"ops.npy declares {nchannels} channels but {data_chan2_bin} is missing"
        )

    output_dir = data_path / "cellpose_suite2p_output"
    os.makedirs(output_dir, exist_ok=True)

    chan2_context = (
        BinaryFile(ly, lx, str(data_chan2_bin), n_frames=nframes)
        if data_chan2_bin.exists()
        else contextlib.nullcontext(None)
    )
    with BinaryFile(ly, lx, str(data_bin), n_frames=nframes) as f_reg, chan2_context as f_reg_chan2:
        stat_after_extraction, f_ch0, fneu_ch0, f_ch1, fneu_ch1 = extraction_wrapper(
            stat, f_reg, f_reg_chan2=f_reg_chan2, ops=ops
        )

    iscell = classify(stat=stat_after_extraction, classfile=builtin_classfile)
    np.save(output_dir / "stat.npy", stat_after_extraction)
    np.save(output_dir / "iscell.npy", iscell)
    np.save(output_dir / "ops.npy", ops)

    extracted: dict[int, np.ndarray] = {}
    for channel in channels:
        if channel == 0:
            traces = f_ch0
            neuropil = fneu_ch0
            legacy_prefix = ""
        else:
            if f_ch1 is None or fneu_ch1 is None:
                raise RuntimeError("Suite2p did not return channel 1 traces")
            traces = f_ch1
            neuropil = fneu_ch1
            legacy_prefix = "_chan2"

        dff = traces.copy() - ops["neucoeff"] * neuropil
        dff = preprocess(
            F=dff,
            baseline=ops["baseline"],
            win_baseline=ops["win_baseline"],
            sig_baseline=ops["sig_baseline"],
            fs=ops["fs"],
            prctile_baseline=ops["prctile_baseline"],
        )
        spks = oasis(F=dff, batch_size=ops["batch_size"], tau=ops["tau"], fs=ops["fs"])

        np.save(output_dir / f"F_ch{channel}.npy", traces)
        np.save(output_dir / f"Fneu_ch{channel}.npy", neuropil)
        np.save(output_dir / f"spks_ch{channel}.npy", spks)
        if legacy_prefix == "":
            np.save(output_dir / "F.npy", traces)
            np.save(output_dir / "Fneu.npy", neuropil)
            np.save(output_dir / "spks.npy", spks)
        else:
            np.save(output_dir / f"F{legacy_prefix}.npy", traces)
            np.save(output_dir / f"Fneu{legacy_prefix}.npy", neuropil)

        extracted[channel] = traces

    return output_dir, extracted


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
    trace_channels: Optional[Sequence[int]] = None,
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

    suite2p_folder, extracted_traces = _extract_suite2p_traces_for_channels(
        data_path=data_path,
        mask_filename=mask_filename,
        trace_channels=trace_channels,
    )
    logger.info(
        "Created Suite2p traces for channels: %s",
        ", ".join(str(channel) for channel in sorted(extracted_traces)),
    )

    mapping = create_suite2p_cellpose_roi_mapping(
        subsegmented_mask, str(suite2p_folder)
    )
    reversed_mapping = {
        cp_label: s2p_idx
        for s2p_idx, cp_label in mapping.items()
        if cp_label is not None
    }

    mapped_trace_matrices: dict[int, np.ndarray] = {}
    for channel, traces in extracted_traces.items():
        mapped_trace_matrices[channel] = map_trace(traces, mapping)

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
        "mapped_traces_matrices": mapped_trace_matrices,
    }

    corr_npy = data_path / "correspondence_matrix.npy"
    np.save(corr_npy, mapped_subsegmented_data)
    logger.info("Saved correspondence matrix to %s (NumPy format)", corr_npy)

    corr_mat = data_path / "correspondence_matrix.mat"
    _save_as_mat(mapped_subsegmented_data, corr_mat)
    logger.info("Saved correspondence matrix to %s (MAT format)", corr_mat)

    trace_paths: dict[int, Path] = {}
    trace_mat_paths: dict[int, Path] = {}
    for channel, mapped_traces_matrix in mapped_trace_matrices.items():
        trace_npy = data_path / f"trace_matrix_ch{channel}.npy"
        trace_mat = data_path / f"trace_matrix_ch{channel}.mat"
        np.save(trace_npy, mapped_traces_matrix)
        _save_as_mat(mapped_traces_matrix, trace_mat)
        trace_paths[channel] = trace_npy
        trace_mat_paths[channel] = trace_mat
        logger.info(
            "Saved channel %d trace matrix to %s and %s",
            channel,
            trace_npy,
            trace_mat,
        )

    first_channel = sorted(mapped_trace_matrices)[0]
    outputs["mapped_traces_matrix"] = mapped_trace_matrices[first_channel]
    if list(sorted(mapped_trace_matrices)) == [0]:
        legacy_trace_npy = data_path / "trace_matrix.npy"
        legacy_trace_mat = data_path / "trace_matrix.mat"
        np.save(legacy_trace_npy, mapped_trace_matrices[0])
        _save_as_mat(mapped_trace_matrices[0], legacy_trace_mat)

    outputs.update(
        {
            "trace_matrix_paths": trace_paths,
            "trace_matrix_mat_paths": trace_mat_paths,
            "trace_matrix_path": trace_paths[first_channel],
            "trace_matrix_mat_path": trace_mat_paths[first_channel],
            "correspondence_matrix_path": corr_npy,
            "correspondence_matrix_mat_path": corr_mat,
        }
    )

    return outputs
