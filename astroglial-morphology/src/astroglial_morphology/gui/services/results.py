"""Helpers for loading pipeline artifacts into the Streamlit GUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


_PROJECTION_PREFIXES = ("mean_", "max_projection_")
_QC_SCALAR_KEYS = ("nframes", "Ly", "Lx")
_QC_ARRAY_KEYS = ("xoff", "yoff", "corrXY", "badframes")
_QC_IMAGE_KEYS = ("meanImg", "meanImg_chan2", "refImg")


def file_mtime(path: Path) -> float:
    """Return ``st_mtime`` or ``0`` when the path cannot be stat'd."""

    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def downsample_for_display(image: np.ndarray, max_side: int = 768) -> np.ndarray:
    """Stride-downsample an image so Streamlit PNG encoding stays cheap."""

    array = np.asarray(image)
    if array.ndim < 2 or max_side <= 0:
        return array
    height, width = array.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return array
    step = int(np.ceil(longest / max_side))
    return array[::step, ::step]


def minmax_downsample(
    series: Dict[str, np.ndarray], max_points: int = 1500
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Downsample 1-D traces while keeping per-window peaks and troughs.

    Vega/Altair zoom is client-side, but it still has to rasterize every
    vertex.  Suite2p recordings are often tens of thousands of frames, which
    makes the default Streamlit line chart lag during pan/zoom.
    """

    arrays = {
        key: np.asarray(values, dtype=float).ravel() for key, values in series.items()
    }
    arrays = {key: values for key, values in arrays.items() if values.size}
    if not arrays:
        return np.array([], dtype=float), {}

    length = min(values.size for values in arrays.values())
    arrays = {key: values[:length] for key, values in arrays.items()}
    if length <= max_points or max_points < 4:
        return np.arange(length, dtype=float), arrays

    bins = max(2, max_points // 2)
    edges = np.linspace(0, length, bins + 1).astype(int)
    indices: list[int] = []
    for start, end in zip(edges[:-1], edges[1:]):
        if start >= end:
            continue
        chosen = {start, end - 1}
        for values in arrays.values():
            window = values[start:end]
            chosen.add(start + int(np.argmin(window)))
            chosen.add(start + int(np.argmax(window)))
        indices.extend(sorted(chosen))
    idx = np.asarray(indices, dtype=int)
    return idx.astype(float), {key: values[idx] for key, values in arrays.items()}


def load_ops(plane_dir: Path) -> Dict[str, Any]:
    """Load ``ops.npy`` for a plane directory, returning an empty dict on failure."""

    ops_path = Path(plane_dir) / "ops.npy"
    if not ops_path.is_file():
        return {}
    try:
        return np.load(ops_path, allow_pickle=True).item()
    except Exception:
        return {}


def load_registration_qc(plane_dir: Path, max_side: int = 768) -> Dict[str, Any]:
    """Return a small QC payload from ``ops.npy``.

    Suite2p's ops dict often contains extra images and metric cubes.  Keeping
    those around makes every Streamlit rerun expensive, so this copies only
    the scalars, 1-D traces, and downsampled preview images the GUI shows.
    """

    ops = load_ops(plane_dir)
    if not ops:
        return {}

    qc: Dict[str, Any] = {}
    for key in _QC_SCALAR_KEYS:
        qc[key] = ops.get(key)
    for key in _QC_ARRAY_KEYS:
        value = ops.get(key)
        qc[key] = np.asarray(value).ravel() if value is not None else np.array([])
    timing = ops.get("timing") or {}
    if isinstance(timing, dict):
        qc["timing"] = {
            str(key): float(value)
            for key, value in timing.items()
            if np.isscalar(value)
        }
    else:
        qc["timing"] = {}
    for key in _QC_IMAGE_KEYS:
        image = ops.get(key)
        qc[key] = (
            None
            if image is None
            else downsample_for_display(np.asarray(image, dtype=np.float32), max_side)
        )
    return qc


def load_projections(plane_dir: Path) -> Dict[str, np.ndarray]:
    """Load projection PNGs saved by the pipeline into arrays."""

    plane = Path(plane_dir)
    if not plane.is_dir():
        return {}
    import matplotlib.image as mpimg

    result: Dict[str, np.ndarray] = {}
    for png in sorted(plane.glob("*.png")):
        if not png.stem.startswith(_PROJECTION_PREFIXES):
            continue
        try:
            result[png.stem] = mpimg.imread(str(png))
        except Exception:
            continue
    return result


def load_seg_file(seg_path: Path) -> Dict[str, Any]:
    """Load a Cellpose ``*_seg.npy`` payload."""

    path = Path(seg_path)
    payload = np.load(path, allow_pickle=True).item()
    if "masks" not in payload:
        raise ValueError(f"Segmentation file missing 'masks': {path}")
    return payload


def load_seg_masks(seg_path: Path) -> np.ndarray:
    """Load only the label image from a Cellpose seg file."""

    return np.asarray(load_seg_file(seg_path)["masks"])


def save_seg_masks(
    seg_path: Path,
    masks: np.ndarray,
    *,
    manual_correction: bool = True,
    backup: bool = True,
) -> Path:
    """Persist edited *masks* back into a Cellpose seg file.

    A one-time ``.orig`` backup is written the first time we edit a file so
    users can revert manual corrections.
    """

    path = Path(seg_path)
    payload: Dict[str, Any]
    if path.is_file():
        payload = np.load(path, allow_pickle=True).item()
    else:
        payload = {}

    if backup and path.is_file():
        backup_path = path.with_suffix(path.suffix + ".orig")
        if not backup_path.exists():
            backup_path.write_bytes(path.read_bytes())

    payload["masks"] = np.asarray(masks, dtype=np.int32)
    payload["manual"] = np.asarray(masks, dtype=bool)
    payload["manual_edited"] = bool(manual_correction)
    np.save(path, payload, allow_pickle=True)
    return path


def load_metadata_payload(metadata_path: Path) -> Optional[Dict[str, Any]]:
    """Read a ``pipeline_metadata.json`` file."""

    path = Path(metadata_path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def normalize_image(image: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Percentile-normalize an image for display in the GUI."""

    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    if array.ndim == 3 and array.shape[-1] in (2, 3):
        # Normalize each channel independently and average for display.
        channels = [normalize_image(array[..., i], low, high) for i in range(array.shape[-1])]
        return np.stack(channels, axis=-1)
    lo, hi = np.percentile(array[np.isfinite(array)], [low, high]) if array.size else (0.0, 1.0)
    if hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    scaled = (array - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def mask_overlay(image: np.ndarray, masks: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Return a colored overlay of *masks* on top of *image*.

    Uses a fixed hash-based palette so ROI ids keep their colour across reruns.
    """

    base = normalize_image(image)
    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)
    elif base.ndim == 3 and base.shape[-1] == 2:
        padded = np.zeros((*base.shape[:2], 3), dtype=base.dtype)
        padded[..., 0] = base[..., 0]
        padded[..., 1] = base[..., 1]
        base = padded

    masks_arr = np.asarray(masks, dtype=np.int64)
    labels = np.unique(masks_arr)
    labels = labels[labels != 0]
    if labels.size == 0:
        return base

    palette = np.zeros((int(labels.max()) + 1, 3), dtype=np.float32)
    rng = np.random.default_rng(seed=42)
    palette[1:] = rng.random((int(labels.max()), 3)).astype(np.float32)

    overlay = base.copy()
    mask_pixels = masks_arr > 0
    if mask_pixels.any():
        colors = palette[masks_arr[mask_pixels]]
        overlay[mask_pixels] = (1.0 - alpha) * overlay[mask_pixels] + alpha * colors
    return np.clip(overlay, 0.0, 1.0)
