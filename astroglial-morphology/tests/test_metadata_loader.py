"""Tests for metadata loader module."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from astroglial_morphology.io.metadata_loader import load_metadata
from astroglial_morphology.io.file_detection import InputFileInfo, InputFormat
from astroglial_morphology.utils.tiff_utils import Metadata


class TestLoadMetadata:
    """Tests for load_metadata function."""
    
    @patch('astroglial_morphology.io.metadata_loader.extract_lif_metadata')
    def test_load_metadata_lif(self, mock_extract_lif, input_file_info_lif):
        """Test loading metadata from LIF file."""
        # Mock return value
        expected_metadata = Metadata(
            nframes=1000,
            nchannels=1,
            nplanes=1,
            finterval=6.818,
            pix_resolution=8.36
        )
        mock_extract_lif.return_value = expected_metadata
        
        # Load metadata
        metadata = load_metadata(input_file_info_lif)
        
        # Assertions
        mock_extract_lif.assert_called_once_with(input_file_info_lif.path_str, series_index=0)
        assert metadata == expected_metadata
        assert metadata.nframes == 1000
        assert metadata.frames_per_channel_per_plane == 1000
    
    @patch('astroglial_morphology.io.metadata_loader.extract_tiff_metadata')
    def test_load_metadata_tiff(self, mock_extract_tiff, input_file_info_tiff):
        """Test loading metadata from TIFF file."""
        # Mock return value
        expected_metadata = Metadata(
            nframes=500,
            nchannels=2,
            nplanes=1,
            finterval=1.0,
            pix_resolution=5.0
        )
        mock_extract_tiff.return_value = expected_metadata
        
        # Load metadata
        metadata = load_metadata(input_file_info_tiff)
        
        # Assertions
        mock_extract_tiff.assert_called_once_with(input_file_info_tiff.path_str)
        assert metadata == expected_metadata
        assert metadata.nframes == 500
        assert metadata.nchannels == 2
    
    def test_load_metadata_invalid_format(self, mock_lif_file):
        """Test with unsupported format."""
        # Create InputFileInfo with invalid format
        class InvalidFormat:
            value = "INVALID"
        
        file_info = Mock()
        file_info.format = InvalidFormat()
        file_info.path_str = str(mock_lif_file)
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_metadata(file_info)
    
    @patch('astroglial_morphology.io.metadata_loader.extract_lif_metadata')
    def test_load_metadata_multi_channel_lif(self, mock_extract_lif, input_file_info_lif):
        """Test loading metadata from multi-channel LIF file."""
        expected_metadata = Metadata(
            nframes=2000,
            nchannels=2,
            nplanes=2,
            finterval=1.5,
            pix_resolution=10.0
        )
        mock_extract_lif.return_value = expected_metadata
        
        metadata = load_metadata(input_file_info_lif)
        
        # Verify frames_per_channel_per_plane calculation
        assert metadata.frames_per_channel_per_plane == 500  # 2000 / (2 * 2)
    
    @patch('astroglial_morphology.io.metadata_loader.extract_tiff_metadata')
    def test_load_metadata_multi_plane_tiff(self, mock_extract_tiff, input_file_info_tiff):
        """Test loading metadata from multi-plane TIFF file."""
        expected_metadata = Metadata(
            nframes=1000,
            nchannels=1,
            nplanes=5,
            finterval=0.5,
            pix_resolution=2.5
        )
        mock_extract_tiff.return_value = expected_metadata
        
        metadata = load_metadata(input_file_info_tiff)
        
        assert metadata.frames_per_channel_per_plane == 200  # 1000 / 5
    
    @patch('astroglial_morphology.io.metadata_loader.extract_lif_metadata')
    def test_load_metadata_logging(self, mock_extract_lif, input_file_info_lif, caplog):
        """Test that metadata loading logs information."""
        metadata = Metadata(
            nframes=1000,
            nchannels=1,
            nplanes=1,
            finterval=6.818,
            pix_resolution=8.36
        )
        mock_extract_lif.return_value = metadata
        
        load_metadata(input_file_info_lif)
        
        # Check that logging occurred
        assert "metadata" in caplog.text.lower()
        assert "1000 frames" in caplog.text or "1000" in caplog.text
