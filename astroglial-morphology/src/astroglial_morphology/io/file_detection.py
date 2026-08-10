"""File detection and validation for input data."""

import logging
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class InputFormat(Enum):
    """Supported input file formats."""

    LIF = "lif"
    TIFF = "tif"
    SUITE2P = "suite2p"


@dataclass
class InputFileInfo:
    """Information about detected input file."""

    path: Path
    format: InputFormat

    @property
    def path_str(self) -> str:
        """Get path as string."""
        return str(self.path)


def is_suite2p_plane(data_path: str | Path) -> bool:
    """Return whether *data_path* is a direct Suite2p plane directory.

    The cheap file check intentionally happens before raw-file discovery.  Full
    ``ops.npy`` validation is performed by the pipeline when the input is used.
    """

    path = Path(data_path)
    return path.is_dir() and (path / "ops.npy").is_file() and (path / "data.bin").is_file()


def detect_input_file(
    data_path: str, format_priority: Optional[List[str]] = None
) -> InputFileInfo:
    """
    Detect input file in the data directory.

    Searches for supported file formats in priority order and returns
    information about the first file found.

    Args:
        data_path: Path to directory containing input files
        format_priority: List of file extensions in priority order.
                        Defaults to [".lif", ".tif"]

    Returns:
        InputFileInfo object with path and format

    Raises:
        FileNotFoundError: If no supported files are found
        ValueError: If multiple files of the same type are found (warning only)
    """
    if format_priority is None:
        format_priority = [".lif", ".tif"]

    data_path_obj = Path(data_path)

    if not data_path_obj.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    if not data_path_obj.is_dir():
        raise ValueError(f"Data path is not a directory: {data_path}")

    # Search for files in priority order
    for ext in format_priority:
        pattern = f"*{ext}"
        files = list(data_path_obj.glob(pattern))

        if files:
            if len(files) > 1:
                logger.warning(
                    f"Multiple {ext} files found in directory, using first one: {files[0].name}"
                )

            format_map = {
                ".lif": InputFormat.LIF,
                ".tif": InputFormat.TIFF,
            }

            file_info = InputFileInfo(
                path=files[0], format=format_map.get(ext, InputFormat.TIFF)
            )

            logger.info(f"Found {ext.upper()} file: {file_info.path_str}")
            return file_info

    supported_formats = ", ".join(format_priority)
    raise FileNotFoundError(
        f"No supported files ({supported_formats}) found in directory: {data_path}"
    )
