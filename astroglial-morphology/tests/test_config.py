"""Tests for config module."""

import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch

from astroglial_morphology.config import PipelineConfig
from astroglial_morphology.utils.tiff_utils import Metadata


class TestPipelineConfig:
    """Tests for PipelineConfig class."""
    
    def test_default_constants(self):
        """Test default configuration constants."""
        assert PipelineConfig.ASTROCYTE_DIAMETER_MICRONS == 31.35
        assert PipelineConfig.DIAMETER_BUFFER_MICRONS == 10.0
        assert PipelineConfig.NECK_DISTANCE_RATIO == 0.47
        assert PipelineConfig.NIMG_INIT_RATIO == 0.15
        assert PipelineConfig.NIMG_INIT_MAX == 300
        assert PipelineConfig.BATCH_SIZE_RATIO == 1.0
        assert PipelineConfig.BATCH_SIZE_MAX == 500
    
    def test_default_model_name(self):
        """Test default model name."""
        assert PipelineConfig.DEFAULT_MODEL_NAME == "CP3_S4_1_0001_3000"
    
    def test_suite2p_defaults(self):
        """Test Suite2p default parameters."""
        defaults = PipelineConfig.SUITE2P_DEFAULTS
        
        assert defaults["maxregshift"] == 0.11
        assert defaults["subpixel"] == 10
        assert defaults["smooth_sigma_time"] == 1
        assert defaults["nonrigid"] is False
        assert defaults["do_registration"] is True
        assert defaults["roidetect"] is False
        assert defaults["spikedetect"] is False


class TestGetModelPath:
    """Tests for get_model_path method."""
    
    def test_get_default_model_path(self):
        """Test getting default model path."""
        # Clear environment variable if set
        with patch.dict(os.environ, {}, clear=True):
            model_path = PipelineConfig.get_model_path()
            
            assert "CP3_S4_1_0001_3000" in model_path
            assert Path(model_path).is_absolute()
    
    def test_get_custom_model_path(self):
        """Test getting custom model path."""
        with patch.dict(os.environ, {}, clear=True):
            model_path = PipelineConfig.get_model_path("custom_model")
            
            assert "custom_model" in model_path
            assert Path(model_path).is_absolute()
    
    def test_get_model_path_from_env(self):
        """Test getting model path from environment variable."""
        custom_path = "/custom/path/to/model"
        with patch.dict(os.environ, {"ASTROGLIAL_MODEL_PATH": custom_path}):
            model_path = PipelineConfig.get_model_path()
            
            assert model_path == custom_path
    
    def test_env_variable_overrides_default(self):
        """Test that environment variable overrides default."""
        custom_path = "/env/model/path"
        with patch.dict(os.environ, {"ASTROGLIAL_MODEL_PATH": custom_path}):
            # Even with custom model name, env variable takes precedence
            model_path = PipelineConfig.get_model_path("ignored_name")
            
            assert model_path == custom_path


class TestCalculateBatchParams:
    """Tests for calculate_batch_params method."""
    
    def test_batch_params_small_dataset(self):
        """Test batch parameters for small dataset."""
        frames = 100
        params = PipelineConfig.calculate_batch_params(frames)
        
        # nimg_init = min(100 * 0.15, 300) = 15
        # batch_size = min(100 * 1.0, 500) = 100
        assert params["nimg_init"] == 15
        assert params["batch_size"] == 100
    
    def test_batch_params_medium_dataset(self):
        """Test batch parameters for medium dataset."""
        frames = 1000
        params = PipelineConfig.calculate_batch_params(frames)
        
        # nimg_init = min(1000 * 0.15, 300) = 150
        # batch_size = min(1000 * 1.0, 500) = 500
        assert params["nimg_init"] == 150
        assert params["batch_size"] == 500
    
    def test_batch_params_large_dataset(self):
        """Test batch parameters for large dataset."""
        frames = 10000
        params = PipelineConfig.calculate_batch_params(frames)
        
        # nimg_init = min(10000 * 0.15, 300) = 300 (capped)
        # batch_size = min(10000 * 1.0, 500) = 500 (capped)
        assert params["nimg_init"] == 300
        assert params["batch_size"] == 500
    
    def test_batch_params_very_small_dataset(self):
        """Test batch parameters for very small dataset."""
        frames = 10
        params = PipelineConfig.calculate_batch_params(frames)
        
        # nimg_init = min(10 * 0.15, 300) = 1
        # batch_size = min(10 * 1.0, 500) = 10
        assert params["nimg_init"] == 1
        assert params["batch_size"] == 10


class TestBuildSuite2pOptions:
    """Tests for build_suite2p_options method."""
    
    def test_build_basic_options(self, sample_metadata):
        """Test building basic Suite2p options."""
        options = PipelineConfig.build_suite2p_options(sample_metadata)
        
        # Check metadata-derived parameters
        assert options["nplanes"] == sample_metadata.nplanes
        assert options["nchannels"] == sample_metadata.nchannels
        assert options["fs"] == sample_metadata.fs
        assert options["reg_tif"] is False  # Default
        
        # Check defaults are included
        assert options["maxregshift"] == 0.11
        assert options["subpixel"] == 10
        assert options["smooth_sigma_time"] == 1
        assert options["nonrigid"] is False
        
        # Check batch parameters
        assert "nimg_init" in options
        assert "batch_size" in options
    
    def test_build_options_with_reg_tif(self, sample_metadata):
        """Test building options with registered TIFF saving enabled."""
        options = PipelineConfig.build_suite2p_options(sample_metadata, reg_tif=True)
        
        assert options["reg_tif"] is True
    
    def test_build_options_with_overrides(self, sample_metadata):
        """Test building options with custom overrides."""
        overrides = {
            "maxregshift": 0.2,
            "subpixel": 5,
            "custom_param": "test_value"
        }
        
        options = PipelineConfig.build_suite2p_options(sample_metadata, **overrides)
        
        # Check overrides are applied
        assert options["maxregshift"] == 0.2
        assert options["subpixel"] == 5
        assert options["custom_param"] == "test_value"
        
        # Check other defaults still present
        assert options["nonrigid"] is False
    
    def test_build_options_multi_channel(self):
        """Test building options for multi-channel data."""
        metadata = Metadata(
            nframes=2000,
            nchannels=2,
            nplanes=1,
            finterval=1.0,
            pix_resolution=8.36
        )
        
        options = PipelineConfig.build_suite2p_options(metadata)
        
        assert options["nchannels"] == 2
        # Batch params should be based on frames_per_channel_per_plane = 1000
        assert options["nimg_init"] == 150  # min(1000 * 0.15, 300)
        assert options["batch_size"] == 500  # min(1000 * 1.0, 500)


class TestCalculateDiameter:
    """Tests for calculate_diameter method."""
    
    def test_calculate_diameter_standard_resolution(self):
        """Test diameter calculation with standard resolution."""
        pix_resolution = 8.360028765690377
        diameter = PipelineConfig.calculate_diameter(pix_resolution)
        
        # Expected: 8.36 * 31.35 + 10.0 ≈ 272.09
        expected = pix_resolution * 31.35 + 10.0
        assert diameter == pytest.approx(expected, rel=1e-5)
    
    def test_calculate_diameter_high_resolution(self):
        """Test diameter calculation with high resolution (small pixels)."""
        pix_resolution = 1.0  # 1 micron per pixel
        diameter = PipelineConfig.calculate_diameter(pix_resolution)
        
        expected = 1.0 * 31.35 + 10.0  # = 41.35 pixels
        assert diameter == pytest.approx(expected, rel=1e-5)
    
    def test_calculate_diameter_low_resolution(self):
        """Test diameter calculation with low resolution (large pixels)."""
        pix_resolution = 20.0  # 20 microns per pixel
        diameter = PipelineConfig.calculate_diameter(pix_resolution)
        
        expected = 20.0 * 31.35 + 10.0  # = 637.0 pixels
        assert diameter == pytest.approx(expected, rel=1e-5)
    
    def test_diameter_always_positive(self):
        """Test that diameter is always positive."""
        for resolution in [0.1, 1.0, 5.0, 10.0, 20.0]:
            diameter = PipelineConfig.calculate_diameter(resolution)
            assert diameter > 0


class TestCalculateNeckDistance:
    """Tests for calculate_neck_distance method."""
    
    def test_calculate_neck_distance_standard(self):
        """Test neck distance calculation with standard diameter."""
        diameter = 272.08690180439334  # From the pipeline run
        neck_distance = PipelineConfig.calculate_neck_distance(diameter)
        
        # Expected: int(272.09 * 0.47) = 127
        expected = int(diameter * 0.47)
        assert neck_distance == expected
        assert isinstance(neck_distance, int)
    
    def test_calculate_neck_distance_small_diameter(self):
        """Test neck distance with small diameter."""
        diameter = 50.0
        neck_distance = PipelineConfig.calculate_neck_distance(diameter)
        
        expected = int(50.0 * 0.47)  # = 23
        assert neck_distance == expected
    
    def test_calculate_neck_distance_large_diameter(self):
        """Test neck distance with large diameter."""
        diameter = 500.0
        neck_distance = PipelineConfig.calculate_neck_distance(diameter)
        
        expected = int(500.0 * 0.47)  # = 235
        assert neck_distance == expected
    
    def test_neck_distance_rounds_down(self):
        """Test that neck distance rounds down (int conversion)."""
        diameter = 100.9
        neck_distance = PipelineConfig.calculate_neck_distance(diameter)
        
        # 100.9 * 0.47 = 47.423, int() = 47
        assert neck_distance == 47


class TestPipelineConfigIntegration:
    """Integration tests for PipelineConfig."""
    
    def test_full_config_workflow(self):
        """Test complete configuration workflow."""
        # Create metadata
        metadata = Metadata(
            nframes=1000,
            nchannels=1,
            nplanes=1,
            finterval=6.818,
            pix_resolution=8.36
        )
        
        # Calculate diameter
        diameter = PipelineConfig.calculate_diameter(metadata.pix_resolution)
        assert diameter > 0
        
        # Calculate neck distance
        neck_distance = PipelineConfig.calculate_neck_distance(diameter)
        assert neck_distance > 0
        assert isinstance(neck_distance, int)
        
        # Build Suite2p options
        options = PipelineConfig.build_suite2p_options(metadata)
        assert "nimg_init" in options
        assert "batch_size" in options
        assert options["fs"] == metadata.fs
        
        # Get model path
        model_path = PipelineConfig.get_model_path()
        assert len(model_path) > 0
