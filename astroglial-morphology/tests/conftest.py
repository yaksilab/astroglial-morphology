"""Shared pytest fixtures for astroglial_morphology tests."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any
from unittest.mock import Mock, MagicMock

from astroglial_morphology.utils.tiff_utils import Metadata
from astroglial_morphology.io.file_detection import InputFileInfo, InputFormat
from astroglial_morphology.config import PipelineConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return Metadata(
        nframes=1000,
        nchannels=1,
        nplanes=1,
        finterval=6.8181818181818175,
        pix_resolution=8.360028765690377,
    )


@pytest.fixture
def pipeline_config():
    """Create a default pipeline configuration."""
    return PipelineConfig()


@pytest.fixture
def mock_lif_file(temp_dir):
    """Create a mock LIF file structure."""
    lif_path = temp_dir / "test.lif"
    lif_path.touch()
    return lif_path


@pytest.fixture
def mock_tiff_file(temp_dir):
    """Create a mock TIFF file structure."""
    tiff_path = temp_dir / "test.tif"
    tiff_path.touch()
    return tiff_path


@pytest.fixture
def sample_binary_data(temp_dir):
    """Create sample Suite2p binary data for testing."""
    suite2p_root = temp_dir / "suite2p"
    plane_path = suite2p_root / "plane0"
    plane_path.mkdir(parents=True, exist_ok=True)

    # Create binary file with sample data
    Ly, Lx = 512, 1024
    nframes = 100
    data = np.random.randint(-1000, 1000, size=(nframes, Ly, Lx), dtype=np.int16)

    bin_path = plane_path / "data.bin"
    with open(bin_path, "wb") as f:
        f.write(bytearray(data))

    # Create ops file
    ops = {
        "Ly": Ly,
        "Lx": Lx,
        "nframes": nframes,
        "meanImg": np.random.rand(Ly, Lx).astype(np.float32),
        "fs": 0.1467,
    }
    np.save(plane_path / "ops.npy", ops, allow_pickle=True)

    return suite2p_root, ops


@pytest.fixture
def sample_mean_image():
    """Create a sample mean image for segmentation testing."""
    # Create a simple synthetic image with 2 bright regions (cells)
    Ly, Lx = 512, 1024
    img = np.zeros((Ly, Lx), dtype=np.float32)
    
    # Add two "cells" (bright circular regions)
    y1, x1 = 200, 300
    y2, x2 = 300, 700
    radius = 50
    
    for y in range(Ly):
        for x in range(Lx):
            dist1 = np.sqrt((y - y1)**2 + (x - x1)**2)
            dist2 = np.sqrt((y - y2)**2 + (x - x2)**2)
            if dist1 < radius:
                img[y, x] = 255 * (1 - dist1/radius)
            if dist2 < radius:
                img[y, x] = 255 * (1 - dist2/radius)
    
    return img


@pytest.fixture
def sample_masks():
    """Create sample segmentation masks for testing."""
    Ly, Lx = 512, 1024
    masks = np.zeros((Ly, Lx), dtype=np.uint16)
    
    # Create two circular masks
    y1, x1 = 200, 300
    y2, x2 = 300, 700
    radius = 50
    
    for y in range(Ly):
        for x in range(Lx):
            dist1 = np.sqrt((y - y1)**2 + (x - x1)**2)
            dist2 = np.sqrt((y - y2)**2 + (x - x2)**2)
            if dist1 < radius:
                masks[y, x] = 1
            elif dist2 < radius:
                masks[y, x] = 2
    
    return masks


@pytest.fixture
def mock_cellpose_model():
    """Create a mock Cellpose model."""
    model = Mock()
    model.eval = Mock(return_value=(None, None, None))
    model.diam_labels = 200.0
    return model


@pytest.fixture
def mock_suite2p_ops():
    """Create mock Suite2p ops for testing."""
    return {
        "do_registration": True,
        "two_step_registration": False,
        "keep_movie_raw": False,
        "smooth_sigma": 1.15,
        "maxregshift": 0.11,
        "align_by_chan": 1,
        "subpixel": 10,
        "nonrigid": False,
        "Ly": 512,
        "Lx": 1024,
        "Lys": [512],
        "Lxs": [1024],
        "nframes": 1000,
        "fs": 0.1467,
    }


@pytest.fixture
def real_lif_data_path():
    """
    Path to real LIF test data (if available).
    Returns None if the test data doesn't exist.
    """
    test_data_path = Path(r"C:\Users\javid.rezai\YaksiLab\duygu\data\Lif_data")
    if test_data_path.exists():
        return test_data_path
    return None


@pytest.fixture
def input_file_info_lif(mock_lif_file):
    """Create InputFileInfo for LIF file."""
    return InputFileInfo(
        path=mock_lif_file,
        format=InputFormat.LIF
    )


@pytest.fixture
def input_file_info_tiff(mock_tiff_file):
    """Create InputFileInfo for TIFF file."""
    return InputFileInfo(
        path=mock_tiff_file,
        format=InputFormat.TIFF
    )
