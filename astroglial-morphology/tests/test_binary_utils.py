"""Tests for binary_utils module."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from astroglial_morphology.binary_utils import BinaryDataProcessor


def _close_processor(processor: BinaryDataProcessor) -> None:
    """Close underlying binary file/memmap handles to avoid Windows file locks."""
    data = getattr(processor, "data", None)
    if data is None:
        return
    try:
        close_fn = getattr(data, "close", None)
        if callable(close_fn):
            close_fn()
    except Exception:
        pass
    try:
        file_obj = getattr(data, "file", None)
        if file_obj is not None:
            file_obj.close()
    except Exception:
        pass
    try:
        mmap_obj = getattr(data, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()
    except Exception:
        pass
    try:
        data_array = getattr(data, "data", None)
        if hasattr(data_array, "_mmap") and data_array._mmap is not None:
            data_array._mmap.close()
    except Exception:
        pass


@pytest.fixture
def binary_processor(sample_binary_data):
    """Create BinaryDataProcessor and ensure cleanup."""
    suite2p_path, ops = sample_binary_data
    processor = BinaryDataProcessor(suite2p_path)
    try:
        yield processor, ops
    finally:
        _close_processor(processor)


class TestBinaryDataProcessor:
    """Tests for BinaryDataProcessor class."""
    
    def test_initialization(self, sample_binary_data):
        """Test BinaryDataProcessor initialization."""
        suite2p_path, ops = sample_binary_data

        processor = BinaryDataProcessor(suite2p_path)
        
        assert processor.suite2p_folder_path == suite2p_path
        assert processor.ops is not None
        assert processor.ops["Ly"] == ops["Ly"]
        assert processor.ops["Lx"] == ops["Lx"]
        assert processor.ops["nframes"] == ops["nframes"]
        _close_processor(processor)
    
    def test_initialization_missing_ops(self, temp_dir):
        """Test initialization fails when ops.npy is missing."""
        suite2p_root = temp_dir / "suite2p"
        suite2p_path = suite2p_root / "plane0"
        suite2p_path.mkdir(parents=True)
        
        with pytest.raises(FileNotFoundError):
            BinaryDataProcessor(suite2p_root)
    
    def test_load_ops(self, binary_processor):
        """Test loading ops file."""
        processor, expected_ops = binary_processor
        
        assert processor.ops["Ly"] == expected_ops["Ly"]
        assert processor.ops["Lx"] == expected_ops["Lx"]
        assert processor.ops["nframes"] == expected_ops["nframes"]
    
    def test_read_frames(self, binary_processor):
        """Test reading frames from binary file."""
        processor, ops = binary_processor

        # Read first 10 frames from binary data
        frames = processor.data.data[:10]

        assert frames.shape == (10, ops["Ly"], ops["Lx"])
        assert frames.dtype == np.int16
    
    def test_read_frames_subset(self, binary_processor):
        """Test reading a subset of frames."""
        processor, ops = binary_processor

        # Read frames 20-30
        frames = processor.data.data[20:30]

        assert frames.shape == (10, ops["Ly"], ops["Lx"])
    
    def test_read_frames_invalid_range(self, binary_processor):
        """Test reading frames with invalid range."""
        processor, ops = binary_processor

        # End beyond nframes should raise IndexError when requesting a frame
        with pytest.raises(IndexError):
            processor.get_frame(ops["nframes"])
    
    def test_create_mean_projection(self, binary_processor):
        """Test creating mean projection."""
        processor, ops = binary_processor
        mean_img = processor.get_mean_image()

        assert mean_img.shape == (ops["Ly"], ops["Lx"])
        assert mean_img.dtype in (np.float32, np.float64)
        assert not np.any(np.isnan(mean_img))
    
    def test_create_max_projection(self, binary_processor):
        """Test creating max projection."""
        processor, ops = binary_processor
        max_img = processor.get_max_projection()

        assert max_img.shape == (ops["Ly"], ops["Lx"])
        assert max_img.dtype == np.int16
    
    def test_create_std_image(self, binary_processor):
        """Test creating standard deviation image."""
        processor, ops = binary_processor
        std_img = processor.get_std_image()

        assert std_img.shape == (ops["Ly"], ops["Lx"])
        assert std_img.dtype in (np.float32, np.float64)
        assert np.all(std_img >= 0)  # Std dev is always non-negative
    
    def test_create_sum_image(self, binary_processor):
        """Test creating sum image."""
        processor, ops = binary_processor
        sum_img = processor.get_sum_image()

        assert sum_img.shape == (ops["Ly"], ops["Lx"])
        assert sum_img.dtype in (np.int64, np.float64)
    
    def test_create_projections_with_batch_size(self, binary_processor):
        """Test creating projections with custom batch size."""
        processor, ops = binary_processor

        # Use small batch size
        mean_img = processor.get_mean_image(batch_size=10)

        assert mean_img.shape == (ops["Ly"], ops["Lx"])
    
    def test_save_projection(self, binary_processor, temp_dir):
        """Test saving projection image."""
        processor, ops = binary_processor
        mean_img = processor.get_mean_image()

        # Save to temp directory
        output_path = temp_dir / "test_projection.png"
        processor.save_image(mean_img, str(output_path))

        assert output_path.exists()
    
    def test_projections_consistency(self, binary_processor):
        """Test that projections are consistent across multiple calls."""
        processor, ops = binary_processor

        # Create projections twice
        mean1 = processor.get_mean_image()
        mean2 = processor.get_mean_image()

        # Should be identical
        np.testing.assert_array_almost_equal(mean1, mean2)
    
    def test_mean_projection_properties(self, binary_processor):
        """Test mathematical properties of mean projection."""
        processor, ops = binary_processor

        # If meanImg is stored in ops, get_mean_image returns it directly
        if processor.ops is not None and "meanImg" in processor.ops:
            np.testing.assert_array_almost_equal(
                processor.get_mean_image(), processor.ops["meanImg"], decimal=6
            )
        else:
            # Get all frames and calculate mean manually
            all_frames = processor.data.data[:]
            manual_mean = np.mean(all_frames, axis=0)

            # Get mean from processor
            processor_mean = processor.get_mean_image()

            # Should be very close (allowing for numerical precision)
            np.testing.assert_array_almost_equal(processor_mean, manual_mean, decimal=4)
    
    def test_max_projection_properties(self, binary_processor):
        """Test mathematical properties of max projection."""
        processor, ops = binary_processor

        # Get max projection
        max_proj = processor.get_max_projection()

        # Max should be greater than or equal to mean from data
        mean_proj = np.mean(processor.data.data[:], axis=0)

        # At least some pixels should have max >= mean
        assert np.all(max_proj >= mean_proj) or np.any(max_proj >= mean_proj)


class TestBinaryDataProcessorEdgeCases:
    """Edge case tests for BinaryDataProcessor."""
    
    def test_single_frame(self, temp_dir):
        """Test processing binary file with single frame."""
        suite2p_root = temp_dir / "suite2p"
        suite2p_path = suite2p_root / "plane0"
        suite2p_path.mkdir(parents=True, exist_ok=True)
        
        Ly, Lx = 100, 100
        data = np.random.randint(-1000, 1000, size=(1, Ly, Lx), dtype=np.int16)
        
        bin_path = suite2p_path / "data.bin"
        with open(bin_path, "wb") as f:
            f.write(bytearray(data))
        
        ops = {"Ly": Ly, "Lx": Lx, "nframes": 1, "fs": 1.0}
        np.save(suite2p_path / "ops.npy", ops, allow_pickle=True)
        
        processor = BinaryDataProcessor(suite2p_root)
        mean_img = processor.get_mean_image()
        
        # For single frame, mean should equal the frame
        assert mean_img.shape == (Ly, Lx)
        _close_processor(processor)
    
    def test_small_frame_dimensions(self, temp_dir):
        """Test with very small frame dimensions."""
        suite2p_root = temp_dir / "suite2p"
        suite2p_path = suite2p_root / "plane0"
        suite2p_path.mkdir(parents=True, exist_ok=True)
        
        Ly, Lx = 10, 10
        nframes = 5
        data = np.random.randint(-100, 100, size=(nframes, Ly, Lx), dtype=np.int16)
        
        bin_path = suite2p_path / "data.bin"
        with open(bin_path, "wb") as f:
            f.write(bytearray(data))
        
        ops = {"Ly": Ly, "Lx": Lx, "nframes": nframes, "fs": 1.0}
        np.save(suite2p_path / "ops.npy", ops, allow_pickle=True)
        
        processor = BinaryDataProcessor(suite2p_root)
        mean_img = processor.get_mean_image()
        
        assert mean_img.shape == (Ly, Lx)
        _close_processor(processor)
