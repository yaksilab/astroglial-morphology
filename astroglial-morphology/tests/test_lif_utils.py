"""Tests for LIF utilities module."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from astroglial_morphology.utils.lif_utils import extract_lif_metadata, lif_to_suite2p_binary
from astroglial_morphology.utils.tiff_utils import Metadata


class TestExtractLifMetadata:
    """Tests for extract_lif_metadata function."""
    
    def test_extract_metadata_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            extract_lif_metadata("nonexistent.lif")
    
    @patch('astroglial_morphology.utils.lif_utils.LifFile')
    def test_extract_metadata_no_images(self, mock_lif_class, tmp_path):
        """Test that ValueError is raised when LIF file has no images."""
        mock_lif = Mock()
        mock_lif.image_list = []
        mock_lif_class.return_value = mock_lif
        
        lif_path = tmp_path / "test.lif"
        lif_path.touch()

        with pytest.raises(ValueError, match="No images found"):
            extract_lif_metadata(str(lif_path))
    
    @patch('astroglial_morphology.utils.lif_utils.LifFile')
    def test_extract_metadata_invalid_series_index(self, mock_lif_class, tmp_path):
        """Test that ValueError is raised for invalid series index."""
        mock_lif = Mock()
        mock_lif.image_list = [{"name": "series1"}]
        mock_lif_class.return_value = mock_lif
        
        lif_path = tmp_path / "test.lif"
        lif_path.touch()

        with pytest.raises(ValueError, match="Series index 5 out of range"):
            extract_lif_metadata(str(lif_path), series_index=5)
    
    @patch('astroglial_morphology.utils.lif_utils.LifFile')
    def test_extract_metadata_success(self, mock_lif_class, tmp_path):
        """Test successful metadata extraction."""
        # Create a temporary LIF file
        lif_path = tmp_path / "test.lif"
        lif_path.touch()
        
        # Mock LIF file structure
        mock_dims = Mock()
        mock_dims.x = 1024
        mock_dims.y = 512
        mock_dims.z = 1
        mock_dims.t = 1000
        mock_dims.m = 1
        
        mock_img = Mock()
        mock_img.name = "test-series"
        mock_img.dims = mock_dims
        mock_img.channels = 1
        mock_img.scale = (8.36, 8.36, 1.0, 6.818)  # x, y, z, t scales
        
        mock_lif = Mock()
        mock_lif.image_list = [{"name": "test-series"}]
        mock_lif.get_image = Mock(return_value=mock_img)
        mock_lif_class.return_value = mock_lif
        
        # Extract metadata
        metadata = extract_lif_metadata(str(lif_path))
        
        # Assertions
        assert isinstance(metadata, Metadata)
        assert metadata.nframes == 1000
        assert metadata.nchannels == 1
        assert metadata.nplanes == 1
        assert metadata.pix_resolution == 8.36
        assert metadata.finterval == 6.818
        assert metadata.fs == pytest.approx(1.0 / 6.818, rel=1e-5)
    
    @patch('astroglial_morphology.utils.lif_utils.LifFile')
    def test_extract_metadata_multiple_series_warning(self, mock_lif_class, tmp_path, caplog):
        """Test that warning is logged for multiple series."""
        lif_path = tmp_path / "test.lif"
        lif_path.touch()
        
        mock_dims = Mock()
        mock_dims.x = 1024
        mock_dims.y = 512
        mock_dims.z = 1
        mock_dims.t = 1000
        mock_dims.m = 1
        
        mock_img = Mock()
        mock_img.name = "series1"
        mock_img.dims = mock_dims
        mock_img.channels = 1
        mock_img.scale = (8.36, 8.36, 1.0, 6.818)
        
        mock_lif = Mock()
        mock_lif.image_list = [{"name": "series1"}, {"name": "series2"}]
        mock_lif.get_image = Mock(return_value=mock_img)
        mock_lif_class.return_value = mock_lif
        
        metadata = extract_lif_metadata(str(lif_path))
        
        # Check warning was logged
        assert "Multiple series detected" in caplog.text
    
    @patch('astroglial_morphology.utils.lif_utils.LifFile')
    def test_extract_metadata_multiple_channels_warning(self, mock_lif_class, tmp_path, caplog):
        """Test that warning is logged for multiple channels."""
        lif_path = tmp_path / "test.lif"
        lif_path.touch()
        
        mock_dims = Mock()
        mock_dims.x = 1024
        mock_dims.y = 512
        mock_dims.z = 1
        mock_dims.t = 1000
        mock_dims.m = 1
        
        mock_img = Mock()
        mock_img.name = "series1"
        mock_img.dims = mock_dims
        mock_img.channels = 3  # Multiple channels
        mock_img.scale = (8.36, 8.36, 1.0, 6.818)
        
        mock_lif = Mock()
        mock_lif.image_list = [{"name": "series1"}]
        mock_lif.get_image = Mock(return_value=mock_img)
        mock_lif_class.return_value = mock_lif
        
        metadata = extract_lif_metadata(str(lif_path))
        
        assert "Multiple channels detected" in caplog.text


class TestLifToSuite2pBinary:
    """Tests for lif_to_suite2p_binary function."""
    
    @patch('astroglial_morphology.utils.lif_utils.extract_lif_metadata')
    @patch('astroglial_morphology.utils.lif_utils.LifFile')
    def test_conversion_creates_directory_structure(self, mock_lif_class, mock_extract_metadata, tmp_path):
        """Test that conversion creates correct directory structure."""
        # Mock metadata
        mock_extract_metadata.return_value = Metadata(
            nframes=10,
            nchannels=1,
            nplanes=1,
            finterval=1.0,
            pix_resolution=8.36
        )
        
        # Mock LIF file
        mock_dims = Mock()
        mock_dims.x = 100
        mock_dims.y = 50
        mock_dims.t = 10
        
        mock_img = Mock()
        mock_img.dims = mock_dims
        mock_img.get_frame = Mock(return_value=Mock())
        
        # Mock PIL Image conversion
        mock_frame = Mock()
        mock_img.get_frame.return_value = mock_frame
        
        with patch('numpy.array', return_value=np.zeros((50, 100), dtype=np.uint8)):
            mock_lif = Mock()
            mock_lif.get_image = Mock(return_value=mock_img)
            mock_lif_class.return_value = mock_lif
            
            # Run conversion
            ops = lif_to_suite2p_binary(str(tmp_path / "test.lif"), str(tmp_path))
            
            # Check directory structure
            assert (tmp_path / "suite2p" / "plane0").exists()
            assert (tmp_path / "suite2p" / "plane0" / "data.bin").exists()
            assert (tmp_path / "suite2p" / "plane0" / "ops.npy").exists()
    
    @patch('astroglial_morphology.utils.lif_utils.extract_lif_metadata')
    def test_conversion_invalid_channel_index(self, mock_extract_metadata, tmp_path):
        """Test that ValueError is raised for invalid channel index."""
        mock_extract_metadata.return_value = Metadata(
            nframes=10,
            nchannels=1,
            nplanes=1,
            finterval=1.0,
            pix_resolution=8.36
        )
        
        with pytest.raises(ValueError, match="Channel index 5 out of range"):
            lif_to_suite2p_binary(str(tmp_path / "test.lif"), str(tmp_path), channel_index=5)
    
    @patch('astroglial_morphology.utils.lif_utils.extract_lif_metadata')
    def test_conversion_invalid_plane_index(self, mock_extract_metadata, tmp_path):
        """Test that ValueError is raised for invalid plane index."""
        mock_extract_metadata.return_value = Metadata(
            nframes=10,
            nchannels=1,
            nplanes=1,
            finterval=1.0,
            pix_resolution=8.36
        )
        
        with pytest.raises(ValueError, match="Plane index 5 out of range"):
            lif_to_suite2p_binary(str(tmp_path / "test.lif"), str(tmp_path), plane_index=5)
    
    @patch('astroglial_morphology.utils.lif_utils.extract_lif_metadata')
    @patch('astroglial_morphology.utils.lif_utils.LifFile')
    def test_conversion_uint8_to_int16_scaling(self, mock_lif_class, mock_extract_metadata, tmp_path):
        """Test uint8 to int16 conversion with proper scaling."""
        mock_extract_metadata.return_value = Metadata(
            nframes=5,
            nchannels=1,
            nplanes=1,
            finterval=1.0,
            pix_resolution=8.36
        )
        
        mock_dims = Mock()
        mock_dims.x = 10
        mock_dims.y = 10
        mock_dims.t = 5
        
        mock_img = Mock()
        mock_img.dims = mock_dims
        
        # Create uint8 test data
        test_data = np.full((10, 10), 255, dtype=np.uint8)
        
        with patch('numpy.array', return_value=test_data):
            mock_img.get_frame = Mock(return_value=Mock())
            
            mock_lif = Mock()
            mock_lif.get_image = Mock(return_value=mock_img)
            mock_lif_class.return_value = mock_lif
            
            ops = lif_to_suite2p_binary(str(tmp_path / "test.lif"), str(tmp_path))
            
            # Check ops contains correct metadata
            assert ops["Ly"] == 10
            assert ops["Lx"] == 10
            assert ops["nframes"] == 5
            assert "Lys" in ops
            assert "Lxs" in ops
    
    @patch('astroglial_morphology.utils.lif_utils.extract_lif_metadata')
    @patch('astroglial_morphology.utils.lif_utils.LifFile')
    def test_conversion_ops_structure(self, mock_lif_class, mock_extract_metadata, tmp_path):
        """Test that ops dictionary has correct structure."""
        mock_extract_metadata.return_value = Metadata(
            nframes=10,
            nchannels=1,
            nplanes=1,
            finterval=1.0,
            pix_resolution=8.36
        )
        
        mock_dims = Mock()
        mock_dims.x = 100
        mock_dims.y = 50
        mock_dims.t = 10
        
        mock_img = Mock()
        mock_img.dims = mock_dims
        
        with patch('numpy.array', return_value=np.zeros((50, 100), dtype=np.uint8)):
            mock_img.get_frame = Mock(return_value=Mock())
            
            mock_lif = Mock()
            mock_lif.get_image = Mock(return_value=mock_img)
            mock_lif_class.return_value = mock_lif
            
            ops = lif_to_suite2p_binary(str(tmp_path / "test.lif"), str(tmp_path))
            
            # Check required ops keys
            required_keys = ["Ly", "Lx", "Lys", "Lxs", "nframes", "nchannels", 
                           "nplanes", "fs", "do_registration", "maxregshift", "subpixel"]
            for key in required_keys:
                assert key in ops
            
            # Check array types
            assert isinstance(ops["Lys"], list)
            assert isinstance(ops["Lxs"], list)
            assert len(ops["Lys"]) == 1
            assert len(ops["Lxs"]) == 1


@pytest.mark.skipif(
    not Path(r"C:\Users\javid.rezai\YaksiLab\duygu\data\Lif_data").exists(),
    reason="Real LIF test data not available"
)
class TestLifUtilsWithRealData:
    """Integration tests with real LIF data (if available)."""
    
    def test_extract_real_lif_metadata(self, real_lif_data_path):
        """Test metadata extraction with real LIF file."""
        if real_lif_data_path is None:
            pytest.skip("Real LIF data not available")
        
        lif_files = list(real_lif_data_path.glob("*.lif"))
        if not lif_files:
            pytest.skip("No LIF files found in test data directory")
        
        lif_file = lif_files[0]
        metadata = extract_lif_metadata(str(lif_file))
        
        # Check that metadata is valid
        assert metadata.nframes > 0
        assert metadata.nchannels > 0
        assert metadata.nplanes > 0
        assert metadata.pix_resolution > 0
        assert metadata.finterval > 0
