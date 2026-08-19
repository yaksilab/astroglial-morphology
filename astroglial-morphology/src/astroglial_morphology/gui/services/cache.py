"""Streamlit cache wrappers around expensive GUI loaders.

The underlying loaders in :mod:`results` stay Streamlit-free so unit tests can
import them.  Pages should call these helpers so ``ops.npy`` and seg files are
not unpickled on every widget rerun.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st

from .results import (
    downsample_for_display,
    file_mtime,
    load_metadata_payload,
    load_projections,
    load_registration_qc,
    load_seg_masks,
    mask_overlay,
    normalize_image,
)


def _plane_signature(plane_dir: Path) -> Tuple[str, float]:
    plane = Path(plane_dir)
    ops_mtime = file_mtime(plane / "ops.npy")
    png_mtime = max((file_mtime(path) for path in plane.glob("*.png")), default=0.0)
    return str(plane), max(ops_mtime, png_mtime)


@st.cache_data(show_spinner="Loading registration QC…", max_entries=8)
def cached_registration_qc(plane_dir: str, mtime: float, max_side: int = 768) -> Dict[str, Any]:
    return load_registration_qc(Path(plane_dir), max_side=max_side)


@st.cache_data(show_spinner="Loading projections…", max_entries=8)
def cached_display_projections(plane_dir: str, mtime: float, max_side: int = 768) -> Dict[str, np.ndarray]:
    loaded = load_projections(Path(plane_dir))
    return {
        name: downsample_for_display(normalize_image(image), max_side)
        for name, image in loaded.items()
    }


@st.cache_data(show_spinner="Loading masks…", max_entries=16)
def cached_seg_masks(seg_path: str, mtime: float) -> np.ndarray:
    return load_seg_masks(Path(seg_path))


@st.cache_data(show_spinner="Building overlay…", max_entries=16)
def cached_mask_overlay(
    seg_path: str,
    image_path: str,
    seg_mtime: float,
    image_mtime: float,
    max_side: int = 768,
) -> np.ndarray:
    import matplotlib.image as mpimg

    masks = load_seg_masks(Path(seg_path))
    image = mpimg.imread(image_path)
    return downsample_for_display(mask_overlay(image, masks), max_side)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_metadata(metadata_path: str, mtime: float) -> Optional[Dict[str, Any]]:
    return load_metadata_payload(Path(metadata_path))


def registration_qc_for(plane_dir: Path, max_side: int = 768) -> Dict[str, Any]:
    plane = Path(plane_dir)
    return cached_registration_qc(str(plane), file_mtime(plane / "ops.npy"), max_side)


def display_projections_for(plane_dir: Path, max_side: int = 768) -> Dict[str, np.ndarray]:
    plane, mtime = _plane_signature(plane_dir)
    return cached_display_projections(plane, mtime, max_side)


def mask_overlay_for(seg_path: Path, image_path: Path, max_side: int = 768) -> np.ndarray:
    return cached_mask_overlay(
        str(seg_path),
        str(image_path),
        file_mtime(seg_path),
        file_mtime(image_path),
        max_side,
    )


def metadata_for(metadata_path: Path) -> Optional[Dict[str, Any]]:
    path = Path(metadata_path)
    return cached_metadata(str(path), file_mtime(path))
