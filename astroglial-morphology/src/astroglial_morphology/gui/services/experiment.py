"""Discover experiment folders and the artifacts each contains.

The GUI accepts three kinds of paths:

* A directory holding a raw ``.lif`` or ``.tif`` acquisition.
* A directory that already contains a ``suite2p`` sub-folder from a prior run.
* A direct Suite2p ``plane0`` folder.

`describe_experiment` normalises all three into a single status object so the
pages can enable/disable steps based on which artifacts already exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from ...io.file_detection import (
    InputFileInfo,
    InputFormat,
    detect_input_file,
    is_suite2p_plane,
)


@dataclass
class SegmentationFile:
    """A Cellpose ``*_seg.npy`` file with its paired projection image."""

    seg_path: Path
    image_path: Optional[Path]
    projection: Optional[str]
    channel: Optional[str]

    @property
    def label(self) -> str:
        return self.seg_path.stem


@dataclass
class ExperimentStatus:
    """Snapshot of what has already been produced for a data folder."""

    data_path: Path
    is_directory: bool = False
    input_mode: Optional[str] = None
    input_file: Optional[InputFileInfo] = None
    suite2p_dir: Optional[Path] = None
    plane_dir: Optional[Path] = None
    has_ops: bool = False
    has_data_bin: bool = False
    has_registration_flag: bool = False
    projections: List[Path] = field(default_factory=list)
    segmentation_files: List[SegmentationFile] = field(default_factory=list)
    metadata_path: Optional[Path] = None
    correspondence_files: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def has_registration(self) -> bool:
        return self.has_ops and self.has_data_bin and self.has_registration_flag

    @property
    def has_projections(self) -> bool:
        return bool(self.projections)

    @property
    def has_segmentation(self) -> bool:
        return bool(self.segmentation_files)

    @property
    def has_metadata(self) -> bool:
        return self.metadata_path is not None and self.metadata_path.is_file()

    @property
    def is_valid_input(self) -> bool:
        return self.input_mode is not None

    @property
    def pipeline_data_path(self) -> Path:
        """Return the path the command-line pipeline should consume."""

        if self.input_mode == "suite2p" and self.plane_dir is not None:
            return self.plane_dir
        return self.data_path


def is_experiment_folder(path: str | Path) -> bool:
    """Return whether *path* is a directory the pipeline could consume."""

    p = Path(path)
    if not p.is_dir():
        return False
    if is_suite2p_plane(p):
        return True
    candidate_plane = p / "suite2p" / "plane0"
    if is_suite2p_plane(candidate_plane):
        return True
    try:
        detect_input_file(str(p))
        return True
    except FileNotFoundError:
        return False
    except Exception:
        return False


def plane0_path(data_path: str | Path) -> Optional[Path]:
    """Return the ``suite2p/plane0`` folder for *data_path* if it exists."""

    p = Path(data_path)
    if is_suite2p_plane(p):
        return p
    candidate = p / "suite2p" / "plane0"
    return candidate if candidate.is_dir() else None


_PROJECTION_KEYS = ("mean", "max_projection", "std", "sum")
_CHANNEL_SUFFIXES = ("ch0", "ch1", "both")


def _parse_projection_from_name(stem: str) -> tuple[Optional[str], Optional[str]]:
    """Extract projection type and channel key from an image file stem."""

    for projection in _PROJECTION_KEYS:
        for channel in _CHANNEL_SUFFIXES:
            expected_prefix = f"{projection}_{channel}"
            if stem.startswith(expected_prefix):
                return projection, channel
    for projection in _PROJECTION_KEYS:
        if stem.startswith(projection):
            return projection, None
    return None, None


def _seg_stem(seg_path: Path) -> str:
    stem = seg_path.stem
    return stem[: -len("_seg")] if stem.endswith("_seg") else stem


def _filename_recorded_in_seg(seg_path: Path) -> Optional[Path]:
    """Return the Cellpose ``filename`` stored in *seg_path*, if readable."""

    try:
        payload = np.load(seg_path, allow_pickle=True)
        item = payload.item() if getattr(payload, "shape", ()) == () else payload
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(item, dict):
        return None
    recorded = item.get("filename")
    if not recorded:
        return None
    return Path(str(recorded))


def _fallback_projection_png(plane_dir: Path) -> Optional[Path]:
    """Pick a Cellpose source PNG, preferring mean over max, in *plane_dir*."""

    pngs = sorted(path for path in plane_dir.glob("*.png") if path.is_file())
    ranked: List[tuple[int, Path]] = []
    for png in pngs:
        stem = png.stem
        if stem.endswith("_image") and stem.startswith("mean_"):
            ranked.append((0, png))
        elif stem.endswith("_image") and stem.startswith("max_projection_"):
            ranked.append((1, png))
        elif stem.startswith("mean"):
            ranked.append((2, png))
    if ranked:
        ranked.sort(key=lambda item: (item[0], item[1].name))
        return ranked[0][1]
    return pngs[0] if pngs else None


def resolve_seg_image_path(seg_path: Path, plane_dir: Path) -> Optional[Path]:
    """Pair a ``*_seg.npy`` file with the projection it should overlay.

    Cellpose writes ``{stem}.png`` next to ``{stem}_seg.npy``. Derived masks
    such as ``subsegmented_masks_seg.npy`` do not get their own PNG; they
    inherit the template image via the payload ``filename`` or fall back to
    the plane's Cellpose source projection.
    """

    stem = _seg_stem(seg_path)
    same_prefix = plane_dir / f"{stem}.png"
    if same_prefix.is_file():
        return same_prefix

    recorded = _filename_recorded_in_seg(seg_path)
    if recorded is not None:
        if recorded.is_file():
            return recorded
        sibling = plane_dir / recorded.name
        if sibling.is_file():
            return sibling

    return _fallback_projection_png(plane_dir)


def find_segmentation_files(plane_dir: Path) -> List[SegmentationFile]:
    """Enumerate ``*_seg.npy`` files in *plane_dir* with paired images."""

    if not plane_dir.is_dir():
        return []
    results: List[SegmentationFile] = []
    for seg_path in sorted(plane_dir.glob("*_seg.npy")):
        image_path = resolve_seg_image_path(seg_path, plane_dir)
        image_stem = image_path.stem if image_path is not None else _seg_stem(seg_path)
        projection, channel = _parse_projection_from_name(image_stem)
        results.append(
            SegmentationFile(
                seg_path=seg_path,
                image_path=image_path,
                projection=projection,
                channel=channel,
            )
        )
    return results


_CORRESPONDENCE_FILES: Iterable[str] = (
    "subsegmented_masks_seg.npy",
    "correspondence_matrix.npy",
    "correspondence_matrix.mat",
    "trace_matrix.npy",
    "trace_matrix.mat",
    "trace_matrix_ch0.npy",
    "trace_matrix_ch1.npy",
)


def describe_experiment(data_path: str | Path) -> ExperimentStatus:
    """Build an :class:`ExperimentStatus` for *data_path*."""

    p = Path(data_path)
    status = ExperimentStatus(data_path=p)
    if not p.exists():
        status.errors.append(f"Path does not exist: {p}")
        return status
    if not p.is_dir():
        status.errors.append(f"Path is not a directory: {p}")
        return status
    status.is_directory = True

    if is_suite2p_plane(p):
        status.input_mode = "suite2p"
        status.plane_dir = p
        status.suite2p_dir = p.parent if p.parent.name == "suite2p" else None
    else:
        detection_error: Optional[str] = None
        try:
            file_info = detect_input_file(str(p))
            status.input_file = file_info
            status.input_mode = (
                "lif" if file_info.format == InputFormat.LIF else "tif"
            )
        except FileNotFoundError as exc:
            detection_error = str(exc)
        except Exception as exc:  # pragma: no cover - defensive
            detection_error = f"Failed to detect input file: {exc}"
        candidate_suite2p = p / "suite2p"
        if candidate_suite2p.is_dir():
            status.suite2p_dir = candidate_suite2p
            candidate_plane = candidate_suite2p / "plane0"
            if candidate_plane.is_dir():
                status.plane_dir = candidate_plane
                if status.input_mode is None and is_suite2p_plane(candidate_plane):
                    status.input_mode = "suite2p"
        if status.input_mode is None and detection_error is not None:
            status.errors.append(detection_error)

    plane_dir = status.plane_dir
    if plane_dir is not None:
        status.has_ops = (plane_dir / "ops.npy").is_file()
        status.has_data_bin = (plane_dir / "data.bin").is_file()
        parent = plane_dir.parent
        status.has_registration_flag = (
            parent.is_dir() and (parent / ".registration_complete").is_file()
        )
        status.projections = sorted(plane_dir.glob("*.png"))
        status.segmentation_files = find_segmentation_files(plane_dir)
        metadata_path = plane_dir / "pipeline_metadata.json"
        if metadata_path.is_file():
            status.metadata_path = metadata_path
        status.correspondence_files = [
            plane_dir / name
            for name in _CORRESPONDENCE_FILES
            if (plane_dir / name).is_file()
        ]

    return status


def load_metadata_json(status: ExperimentStatus) -> Optional[dict]:
    """Convenience helper for pages that only need the parsed JSON."""

    if not status.has_metadata or status.metadata_path is None:
        return None
    try:
        return json.loads(status.metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
