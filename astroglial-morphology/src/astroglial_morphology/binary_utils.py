"""
Binary data utility module for loading and processing Suite2p binary files.

This module provides functions to load binary data files and create various
projections and statistical images from the data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from suite2p.io import BinaryFile

from .logging_config import get_logger

logger = get_logger(__name__)


class BinaryDataProcessor:
    """
    A class to handle loading and processing of Suite2p binary data files.
    Currently supports single plane, single channel data only.
    """

    def __init__(self, suite2p_folder_path: str, plane_idx: int = 0):
        """
        Initialize the binary data processor.

        Args:
            suite2p_folder_path: Path to the suite2p folder containing plane directories
            plane_idx: Index of the plane to load (default: 0)
        """
        self.suite2p_folder_path = Path(suite2p_folder_path)
        self.plane_idx = plane_idx

        # Construct paths to plane-specific files
        self.plane_path = self.suite2p_folder_path / f"plane{plane_idx}"
        self.ops_path = self.plane_path / "ops.npy"
        self.bin_file_path = self.plane_path / "data.bin"

        self.ops: Optional[Dict[str, Any]] = None
        self.data: Optional[BinaryFile] = None
        self._load_data()

    def _load_data(self) -> None:
        """Load the ops metadata and binary data."""
        try:
            if not self.plane_path.exists():
                raise FileNotFoundError(f"Plane directory not found: {self.plane_path}")

            if not self.ops_path.exists():
                raise FileNotFoundError(f"ops.npy not found: {self.ops_path}")
            if not self.bin_file_path.exists():
                raise FileNotFoundError(f"data.bin not found: {self.bin_file_path}")

            # Load ops metadata
            self.ops = np.load(self.ops_path, allow_pickle=True).item()
            logger.info(f"Loaded ops from {self.ops_path}")

            # Extract dimensions
            if self.ops is None:
                raise RuntimeError("Failed to load ops data")

            Lx = self.ops["Lx"]
            Ly = self.ops["Ly"]
            nframes = self.ops["nframes"]

            # Check for multiple channels and warn if found
            nchannels = self.ops.get("nchannels", 1)
            if nchannels > 1:
                logger.warning(
                    f"Multiple channels detected (nchannels={nchannels}). "
                    f"Only single channel data is currently supported. "
                    f"Channel 0 will be used, other channels will be ignored."
                )

            # Check for multiple planes and warn if found
            nplanes = self.ops.get("nplanes", 1)
            if nplanes > 1:
                logger.info(
                    f"Multiple planes detected (nplanes={nplanes}). "
                    f"Currently loading plane {self.plane_idx} only."
                )
            # Load binary data
            self.data = BinaryFile(
                Ly=Ly, Lx=Lx, filename=str(self.bin_file_path), n_frames=nframes
            )
            logger.info(f"Loaded binary data from {self.bin_file_path}")

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def get_mean_image(self) -> np.ndarray:
        """
        Get the mean image from ops metadata.

        Returns:
            Mean image as numpy array
        """
        if self.ops is None or "meanImg" not in self.ops:
            logger.warning("No meanImg found in ops, calculating from data")
            if self.data is None:
                raise RuntimeError("No data loaded")
            return np.mean(self.data.data, axis=0)
        return self.ops["meanImg"]

    def get_max_projection(self) -> np.ndarray:
        """
        Calculate maximum projection across all frames.

        Returns:
            Maximum projection image as numpy array
        """
        if self.data is None:
            raise RuntimeError("No data loaded")
        logger.info("Calculating max projection")
        return np.max(self.data.data, axis=0)

    def get_std_image(self) -> np.ndarray:
        """
        Calculate standard deviation image across all frames.

        Returns:
            Standard deviation image as numpy array
        """
        if self.data is None:
            raise RuntimeError("No data loaded")
        logger.info("Calculating standard deviation image")
        return np.std(self.data.data, axis=0)

    def get_sum_image(self) -> np.ndarray:
        """
        Calculate sum image across all frames.

        Returns:
            Sum image as numpy array
        """
        if self.data is None:
            raise RuntimeError("No data loaded")
        logger.info("Calculating sum image")
        return np.sum(self.data.data, axis=0)

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """
        Get a specific frame from the data.

        Args:
            frame_idx: Index of the frame to retrieve

        Returns:
            Frame data as numpy array
        """
        if self.ops is None or self.data is None:
            raise RuntimeError("No data loaded")

        if frame_idx >= self.ops["nframes"]:
            raise IndexError(
                f"Frame index {frame_idx} out of range (max: {self.ops['nframes']-1})"
            )

        return self.data.data[frame_idx]

    def save_image(
        self, image: np.ndarray, output_path: str, cmap: str = "gray"
    ) -> None:
        """
        Save an image to file.

        Args:
            image: Image data as numpy array
            output_path: Path where to save the image (relative to plane directory if not absolute)
            cmap: Colormap to use for saving
        """
        output_path_obj = Path(output_path)

        if not output_path_obj.is_absolute():
            output_path_obj = self.plane_path / output_path_obj

        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        plt.imsave(str(output_path_obj), image, cmap=cmap)
        logger.info(f"Saved image to {output_path_obj}")

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded data.

        Returns:
            Dictionary containing data information
        """
        if self.ops is None:
            raise RuntimeError("No data loaded")

        return {
            "Lx": self.ops["Lx"],
            "Ly": self.ops["Ly"],
            "nframes": self.ops["nframes"],
            "nchannels": self.ops.get("nchannels", 1),
            "nplanes": self.ops.get("nplanes", 1),
            "plane_idx": self.plane_idx,
            "ops_path": str(self.ops_path),
            "bin_file_path": str(self.bin_file_path),
            "data_shape": self.data.data.shape if self.data else None,
        }


def load_binary_data(
    suite2p_folder_path: str, plane_idx: int = 0
) -> BinaryDataProcessor:
    """
    Convenience function to load binary data.

    Args:
        suite2p_folder_path: Path to the suite2p folder containing plane directories
        plane_idx: Index of the plane to load (default: 0)

    Returns:
        BinaryDataProcessor instance
    """
    return BinaryDataProcessor(suite2p_folder_path, plane_idx)


def create_projections(
    suite2p_folder_path: str,
    plane_idx: int = 0,
    output_dir: Optional[str] = None,
    save_images: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Create all standard projections and save them as images.

    Args:
        suite2p_folder_path: Path to the suite2p folder containing plane directories
        plane_idx: Index of the plane to load (default: 0)
        output_dir: Directory to save images (default: plane directory)
        save_images: Whether to save images to disk

    Returns:
        Dictionary containing all projection images
    """
    processor = load_binary_data(suite2p_folder_path, plane_idx)

    projections = {
        "mean": processor.get_mean_image(),
        "max_projection": processor.get_max_projection(),
        "std": processor.get_std_image(),
        "sum": processor.get_sum_image(),
    }

    if save_images:
        if output_dir is None:
            output_dir_obj = processor.plane_path
        else:
            output_dir_obj = Path(output_dir)

        output_dir_obj.mkdir(parents=True, exist_ok=True)

        for name, image in projections.items():
            output_path = output_dir_obj / f"{name}_image.png"
            processor.save_image(image, str(output_path))

    return projections
