"""Tests for TIFF utilities module."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from astroglial_morphology.utils.tiff_utils import Metadata, extract_tiff_metadata


class TestMetadata:
    """Tests for Metadata dataclass."""
    
    def test_metadata_creation(self):
        """Test creating a Metadata object."""
        metadata = Metadata(
            nframes=1000,
            nchannels=1,
            nplanes=1,
            finterval=6.8181818181818175,
            pix_resolution=8.360028765690377
        )
        
        assert metadata.nframes == 1000
        assert metadata.nchannels == 1
        assert metadata.nplanes == 1
        assert metadata.finterval == 6.8181818181818175
        assert metadata.pix_resolution == 8.360028765690377
    
    def test_fs_property(self):
        """Test fs (sampling frequency) property calculation."""
        metadata = Metadata(
            nframes=1000,
            nchannels=1,
            nplanes=1,
            finterval=6.8181818181818175,
            pix_resolution=8.360028765690377
        )
        
        expected_fs = 1.0 / 6.8181818181818175
        assert metadata.fs == pytest.approx(expected_fs, rel=1e-10)
    
    def test_frames_per_channel_per_plane_property(self):
        """Test frames_per_channel_per_plane property calculation."""
        # Single channel, single plane
        metadata1 = Metadata(
            nframes=1000,
            nchannels=1,
            nplanes=1,
            finterval=1.0,
            pix_resolution=1.0
        )
        assert metadata1.frames_per_channel_per_plane == 1000
        
        # Multiple channels
        metadata2 = Metadata(
            nframes=1000,
            nchannels=2,
            nplanes=1,
            finterval=1.0,
            pix_resolution=1.0
        )
        assert metadata2.frames_per_channel_per_plane == 500
        
        # Multiple planes
        metadata3 = Metadata(
            nframes=1000,
            nchannels=1,
            nplanes=2,
            finterval=1.0,
            pix_resolution=1.0
        )
        assert metadata3.frames_per_channel_per_plane == 500
        
        # Multiple channels and planes
        metadata4 = Metadata(
            nframes=1000,
            nchannels=2,
            nplanes=2,
            finterval=1.0,
            pix_resolution=1.0
        )
        assert metadata4.frames_per_channel_per_plane == 250


class TestExtractTiffMetadata:
    """Tests for extract_tiff_metadata function."""
    
    def test_extract_metadata_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            extract_tiff_metadata("nonexistent.tif")
    
    @patch('astroglial_morphology.utils.tiff_utils.tifffile.TiffFile')
    def test_extract_metadata_success(self, mock_tiff_class, tmp_path):
        """Test successful metadata extraction from TIFF file."""
        # Create a temporary TIFF file
        tiff_path = tmp_path / "test.tif"
        tiff_path.touch()
        
        # Mock TiffFile structure
        mock_page = Mock()
        mock_page.shape = (512, 1024)  # (height, width)
        
        mock_series = Mock()
        mock_series.shape = (1000, 512, 1024)  # (frames, height, width)
        
        mock_tiff = Mock()
        mock_tiff.pages = [mock_page] * 1000
        mock_tiff.series = [mock_series]
        
        # Mock ImageJ metadata with Info field
        info_str = (
            "Series 1 Name = TestSeries\n"
            "TestSeries SizeC = 1\n"
            "TestSeries SizeT = 1000\n"
            "TestSeries SizeZ = 1\n"
            "CycleTime = 6.818\n"
            "DimensionDescription #1| DimID = 1 | Length = 0.00836 | NumberOfElements = 1024 | Unit = mm\n"
        )
        mock_tiff.imagej_metadata = {
            "Info": info_str,
        }
        
        mock_tiff_class.return_value.__enter__ = Mock(return_value=mock_tiff)
        mock_tiff_class.return_value.__exit__ = Mock(return_value=False)
        
        # Extract metadata
        metadata = extract_tiff_metadata(str(tiff_path))
        
        # Assertions
        assert isinstance(metadata, Metadata)
        assert metadata.nframes == 1000
        assert metadata.nchannels == 1
        assert metadata.nplanes == 1
        assert metadata.finterval == pytest.approx(6.818, rel=1e-6)
        assert metadata.pix_resolution > 0
    
    @patch('astroglial_morphology.utils.tiff_utils.tifffile.TiffFile')
    def test_extract_metadata_no_imagej_metadata(self, mock_tiff_class, tmp_path):
        """Test metadata extraction with missing ImageJ metadata."""
        tiff_path = tmp_path / "test.tif"
        tiff_path.touch()
        
        mock_page = Mock()
        mock_page.shape = (512, 1024)
        
        mock_series = Mock()
        mock_series.shape = (100, 512, 1024)
        
        mock_tiff = Mock()
        mock_tiff.pages = [mock_page] * 100
        mock_tiff.series = [mock_series]
        mock_tiff.imagej_metadata = None  # No ImageJ metadata
        
        mock_tiff_class.return_value.__enter__ = Mock(return_value=mock_tiff)
        mock_tiff_class.return_value.__exit__ = Mock(return_value=False)
        
        # Extract metadata should fail when ImageJ metadata missing
        with pytest.raises(ValueError, match="No ImageJ metadata found"):
            extract_tiff_metadata(str(tiff_path))
    
    @patch('astroglial_morphology.utils.tiff_utils.tifffile.TiffFile')
    def test_extract_metadata_multi_channel(self, mock_tiff_class, tmp_path):
        """Test metadata extraction for multi-channel TIFF."""
        tiff_path = tmp_path / "test.tif"
        tiff_path.touch()
        
        mock_page = Mock()
        mock_page.shape = (512, 1024)
        
        mock_series = Mock()
        mock_series.shape = (2000, 512, 1024)  # 1000 frames × 2 channels
        
        mock_tiff = Mock()
        mock_tiff.pages = [mock_page] * 2000
        mock_tiff.series = [mock_series]
        info_str = (
            "Series 1 Name = TestSeries\n"
            "TestSeries SizeC = 2\n"
            "TestSeries SizeT = 1000\n"
            "TestSeries SizeZ = 1\n"
            "CycleTime = 1.0\n"
            "DimensionDescription #1| DimID = 1 | Length = 0.01024 | NumberOfElements = 1024 | Unit = mm\n"
        )
        mock_tiff.imagej_metadata = {
            "Info": info_str,
        }
        
        mock_tiff_class.return_value.__enter__ = Mock(return_value=mock_tiff)
        mock_tiff_class.return_value.__exit__ = Mock(return_value=False)
        
        metadata = extract_tiff_metadata(str(tiff_path))
        
        assert metadata.nchannels in (1, 2)
        assert metadata.nframes == 2000
        assert metadata.frames_per_channel_per_plane == 1000
