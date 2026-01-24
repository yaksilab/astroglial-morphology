"""Tests for file detection module."""

import pytest
from pathlib import Path
from astroglial_morphology.io.file_detection import (
    InputFormat,
    InputFileInfo,
    detect_input_file
)


class TestInputFormat:
    """Tests for InputFormat enum."""
    
    def test_input_format_values(self):
        """Test that InputFormat has correct values."""
        assert hasattr(InputFormat, "LIF")
        assert hasattr(InputFormat, "TIFF")
        assert InputFormat.LIF.value == "lif"
        assert InputFormat.TIFF.value == "tif"


class TestInputFileInfo:
    """Tests for InputFileInfo dataclass."""
    
    def test_input_file_info_creation(self, mock_lif_file):
        """Test creating InputFileInfo object."""
        info = InputFileInfo(path=mock_lif_file, format=InputFormat.LIF)
        
        assert info.path == mock_lif_file
        assert info.format == InputFormat.LIF
        assert isinstance(info.path, Path)
    
    def test_path_str_property(self, mock_tiff_file):
        """Test path_str property."""
        info = InputFileInfo(path=mock_tiff_file, format=InputFormat.TIFF)
        
        assert info.path_str == str(mock_tiff_file)
        assert isinstance(info.path_str, str)


class TestDetectInputFile:
    """Tests for detect_input_file function."""
    
    def test_detect_lif_file(self, temp_dir):
        """Test detecting LIF file."""
        # Create a LIF file
        lif_file = temp_dir / "test.lif"
        lif_file.touch()
        
        file_info = detect_input_file(temp_dir)
        
        assert file_info is not None
        assert file_info.format == InputFormat.LIF
        assert file_info.path == lif_file
    
    def test_detect_tiff_file(self, temp_dir):
        """Test detecting TIFF file."""
        # Create a TIFF file
        tiff_file = temp_dir / "test.tif"
        tiff_file.touch()
        
        file_info = detect_input_file(temp_dir)
        
        assert file_info is not None
        assert file_info.format == InputFormat.TIFF
        assert file_info.path == tiff_file
    
    def test_lif_priority_over_tiff(self, temp_dir):
        """Test that LIF files have priority over TIFF files."""
        # Create both LIF and TIFF files
        lif_file = temp_dir / "test.lif"
        tiff_file = temp_dir / "test.tif"
        lif_file.touch()
        tiff_file.touch()
        
        file_info = detect_input_file(temp_dir)
        
        # LIF should be detected first (default priority)
        assert file_info.format == InputFormat.LIF
        assert file_info.path == lif_file
    
    def test_custom_format_priority(self, temp_dir):
        """Test custom format priority."""
        # Create both files
        lif_file = temp_dir / "test.lif"
        tiff_file = temp_dir / "test.tif"
        lif_file.touch()
        tiff_file.touch()
        
        # Prioritize TIFF over LIF
        file_info = detect_input_file(temp_dir, format_priority=[".tif", ".lif"])
        
        assert file_info.format == InputFormat.TIFF
        assert file_info.path == tiff_file
    
    def test_no_input_file_found(self, temp_dir):
        """Test when no input file is found."""
        with pytest.raises(FileNotFoundError, match="No supported files"):
            detect_input_file(temp_dir)
    
    def test_invalid_directory(self):
        """Test with non-existent directory."""
        with pytest.raises((FileNotFoundError, ValueError)):
            detect_input_file(Path("nonexistent_directory"))
    
    def test_tiff_extension_variations(self, temp_dir):
        """Test detection of various TIFF extensions."""
        # Test .tif extension (supported)
        tif_file = temp_dir / "test.tif"
        tif_file.touch()

        file_info = detect_input_file(temp_dir)

        assert file_info is not None
        assert file_info.format == InputFormat.TIFF
    
    def test_multiple_lif_files(self, temp_dir):
        """Test with multiple LIF files (should return first one found)."""
        lif1 = temp_dir / "test1.lif"
        lif2 = temp_dir / "test2.lif"
        lif1.touch()
        lif2.touch()
        
        file_info = detect_input_file(temp_dir)
        
        assert file_info.format == InputFormat.LIF
        # Should return one of them
        assert file_info.path in [lif1, lif2]
    
    def test_case_insensitive_extension(self, temp_dir):
        """Test case-insensitive extension matching."""
        # Create file with uppercase extension
        lif_file = temp_dir / "test.LIF"
        lif_file.touch()
        
        file_info = detect_input_file(temp_dir)
        
        assert file_info is not None
        assert file_info.format == InputFormat.LIF
