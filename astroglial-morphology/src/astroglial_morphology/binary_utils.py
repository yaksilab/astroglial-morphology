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

    def get_mean_image(self, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Get the mean image from ops metadata or calculate it from data.

        Args:
            batch_size: Number of frames to process at a time. If None, processes all frames at once.
                       Use smaller batch sizes to avoid memory issues with large datasets.
                       Only used if meanImg is not available in ops.

        Returns:
            Mean image as numpy array
        """
        if self.ops is None or "meanImg" not in self.ops:
            logger.warning("No meanImg found in ops, calculating from data")
            if self.data is None or self.ops is None:
                raise RuntimeError("No data loaded")

            if batch_size is None or batch_size >= self.ops["nframes"]:
                return np.mean(self.data.data, axis=0)

            # Batch processing for mean
            logger.info(f"Calculating mean image (batch size: {batch_size})")
            nframes = self.ops["nframes"]
            mean_image = None

            for start_idx in range(0, nframes, batch_size):
                end_idx = min(start_idx + batch_size, nframes)
                batch = self.data.data[start_idx:end_idx]
                batch_sum = np.sum(batch, axis=0)

                if mean_image is None:
                    mean_image = batch_sum.astype(np.float64)
                else:
                    mean_image += batch_sum

                logger.debug(f"Processed frames {start_idx}-{end_idx}/{nframes}")

            if mean_image is None:
                raise RuntimeError("Failed to compute mean image")

            return mean_image / nframes

        return self.ops["meanImg"]

    def get_max_projection(self, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Calculate maximum projection across all frames.

        Args:
            batch_size: Number of frames to process at a time. If None, processes all frames at once.
                       Use smaller batch sizes to avoid memory issues with large datasets.

        Returns:
            Maximum projection image as numpy array
        """
        if self.data is None or self.ops is None:
            raise RuntimeError("No data loaded")

        if batch_size is None or batch_size >= self.ops["nframes"]:
            logger.info("Calculating max projection (single batch)")
            return np.max(self.data.data, axis=0)

        logger.info(f"Calculating max projection (batch size: {batch_size})")
        nframes = self.ops["nframes"]
        max_image = None

        for start_idx in range(0, nframes, batch_size):
            end_idx = min(start_idx + batch_size, nframes)
            batch = self.data.data[start_idx:end_idx]
            batch_max = np.max(batch, axis=0)

            if max_image is None:
                max_image = batch_max
            else:
                max_image = np.maximum(max_image, batch_max)

            logger.debug(f"Processed frames {start_idx}-{end_idx}/{nframes}")

        if max_image is None:
            raise RuntimeError("Failed to compute max projection")

        return max_image

    def get_std_image(self, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Calculate standard deviation image across all frames.

        Args:
            batch_size: Number of frames to process at a time. If None, processes all frames at once.
                       Use smaller batch sizes to avoid memory issues with large datasets.

        Returns:
            Standard deviation image as numpy array
        """
        if self.data is None or self.ops is None:
            raise RuntimeError("No data loaded")

        if batch_size is None or batch_size >= self.ops["nframes"]:
            logger.info("Calculating standard deviation image (single batch)")
            return np.std(self.data.data, axis=0)

        # For batch processing std, we use Welford's online algorithm
        logger.info(f"Calculating standard deviation image (batch size: {batch_size})")
        nframes = self.ops["nframes"]

        # Initialize accumulators
        mean = None
        M2 = None
        n = 0

        for start_idx in range(0, nframes, batch_size):
            end_idx = min(start_idx + batch_size, nframes)
            batch = self.data.data[start_idx:end_idx]

            for frame in batch:
                n += 1
                if mean is None:
                    mean = np.zeros_like(frame, dtype=np.float64)
                    M2 = np.zeros_like(frame, dtype=np.float64)

                delta = frame - mean
                mean += delta / n
                delta2 = frame - mean
                M2 += delta * delta2

            logger.debug(f"Processed frames {start_idx}-{end_idx}/{nframes}")

        if M2 is None or n == 0:
            raise RuntimeError("Failed to compute standard deviation")

        std = np.sqrt(M2 / n)
        return std

    def get_sum_image(self, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Calculate sum image across all frames.

        Args:
            batch_size: Number of frames to process at a time. If None, processes all frames at once.
                       Use smaller batch sizes to avoid memory issues with large datasets.

        Returns:
            Sum image as numpy array
        """
        if self.data is None or self.ops is None:
            raise RuntimeError("No data loaded")

        if batch_size is None or batch_size >= self.ops["nframes"]:
            logger.info("Calculating sum image (single batch)")
            return np.sum(self.data.data, axis=0)

        logger.info(f"Calculating sum image (batch size: {batch_size})")
        nframes = self.ops["nframes"]
        sum_image = None

        for start_idx in range(0, nframes, batch_size):
            end_idx = min(start_idx + batch_size, nframes)
            batch = self.data.data[start_idx:end_idx]
            batch_sum = np.sum(batch, axis=0)

            if sum_image is None:
                sum_image = batch_sum.astype(np.float64)
            else:
                sum_image += batch_sum

            logger.debug(f"Processed frames {start_idx}-{end_idx}/{nframes}")

        if sum_image is None:
            raise RuntimeError("Failed to compute sum image")

        return sum_image

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
    batch_size: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Create all standard projections and save them as images.

    Args:
        suite2p_folder_path: Path to the suite2p folder containing plane directories
        plane_idx: Index of the plane to load (default: 0)
        output_dir: Directory to save images (default: plane directory)
        save_images: Whether to save images to disk
        batch_size: Number of frames to process at a time. If None, processes all frames at once.
                   Use smaller batch sizes (e.g., 1000-5000) to avoid memory issues with large datasets.

    Returns:
        Dictionary containing all projection images
        - mean : Mean projection image
        - max_projection : Max projection image
        - std : Standard deviation projection image
        - sum : Sum projection image

    """
    processor = load_binary_data(suite2p_folder_path, plane_idx)

    projections = {
        "mean": processor.get_mean_image(batch_size=batch_size),
        "max_projection": processor.get_max_projection(batch_size=batch_size),
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
