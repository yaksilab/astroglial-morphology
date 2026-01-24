"""Integration tests for the complete pipeline."""

import pytest
import numpy as np
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from astroglial_morphology.pipeline import Pipeline
from astroglial_morphology.config import PipelineConfig
from astroglial_morphology.io.file_detection import InputFileInfo, InputFormat
from astroglial_morphology.utils.tiff_utils import Metadata


class TestPipelineInitialization:
    """Tests for Pipeline initialization."""
    
    def test_pipeline_creation_with_path(self, temp_dir):
        """Test creating pipeline with data path."""
        with patch('astroglial_morphology.pipeline.Segmentation'):
            pipeline = Pipeline(data_path=str(temp_dir))

            assert pipeline.data_path == str(temp_dir)
            assert isinstance(pipeline.config, PipelineConfig)
    
    def test_pipeline_creation_with_custom_config(self, temp_dir):
        """Test creating pipeline with custom config."""
        config = PipelineConfig()
        config.ASTROCYTE_DIAMETER_MICRONS = 35.0
        with patch('astroglial_morphology.pipeline.Segmentation'):
            pipeline = Pipeline(data_path=str(temp_dir), config=config)

            assert pipeline.config.ASTROCYTE_DIAMETER_MICRONS == 35.0
    
    def test_pipeline_with_gpu_flag(self, temp_dir):
        """Test creating pipeline with GPU enabled."""
        with patch('astroglial_morphology.pipeline.Segmentation'):
            pipeline = Pipeline(data_path=str(temp_dir), use_gpu=True)

            assert pipeline.use_gpu is True
    
    def test_pipeline_with_custom_model(self, temp_dir):
        """Test creating pipeline with custom model path."""
        model_path = "/custom/model/path"
        with patch('astroglial_morphology.pipeline.Segmentation'):
            pipeline = Pipeline(data_path=str(temp_dir), model_path=model_path)

            assert pipeline.model_path == model_path


class TestPipelineDetectInput:
    """Tests for Pipeline.detect_input method."""
    
    @patch('astroglial_morphology.pipeline.detect_input_file')
    def test_detect_lif_file(self, mock_detect, temp_dir):
        """Test detecting LIF input file."""
        lif_file = temp_dir / "test.lif"
        mock_detect.return_value = InputFileInfo(
            path=lif_file,
            format=InputFormat.LIF
        )
        with patch('astroglial_morphology.pipeline.Segmentation'):
            pipeline = Pipeline(data_path=str(temp_dir))
            pipeline.detect_input()

            assert pipeline.file_info.format == InputFormat.LIF
            mock_detect.assert_called_once_with(str(temp_dir), format_priority=pipeline.config.FILE_FORMAT_PRIORITY)
    
    @patch('astroglial_morphology.pipeline.detect_input_file')
    def test_detect_tiff_file(self, mock_detect, temp_dir):
        """Test detecting TIFF input file."""
        tiff_file = temp_dir / "test.tif"
        mock_detect.return_value = InputFileInfo(
            path=tiff_file,
            format=InputFormat.TIFF
        )
        with patch('astroglial_morphology.pipeline.Segmentation'):
            pipeline = Pipeline(data_path=str(temp_dir))
            pipeline.detect_input()

            assert pipeline.file_info.format == InputFormat.TIFF


class TestPipelineLoadMetadata:
    """Tests for Pipeline.load_metadata method."""
    
    @patch('astroglial_morphology.pipeline.load_metadata')
    def test_load_metadata_success(self, mock_load, temp_dir, input_file_info_lif):
        """Test loading metadata successfully."""
        expected_metadata = Metadata(
            nframes=1000,
            nchannels=1,
            nplanes=1,
            finterval=6.818,
            pix_resolution=8.36
        )
        mock_load.return_value = expected_metadata
        
        with patch('astroglial_morphology.pipeline.Segmentation'):
            pipeline = Pipeline(data_path=str(temp_dir))
            pipeline.file_info = input_file_info_lif

            pipeline.load_metadata()

            assert pipeline.metadata == expected_metadata
            mock_load.assert_called_once_with(input_file_info_lif)


class TestPipelineCreateProjections:
    """Tests for Pipeline.create_projections method."""
    
    @patch('astroglial_morphology.pipeline.create_projections')
    def test_create_projections(self, mock_create_projections, temp_dir, sample_metadata):
        """Test creating projections."""
        mock_mean = np.random.rand(512, 1024).astype(np.float32)
        mock_max = np.random.randint(-1000, 1000, (512, 1024), dtype=np.int16)
        mock_std = np.random.rand(512, 1024).astype(np.float32)
        mock_sum = np.random.rand(512, 1024).astype(np.float64)
        mock_create_projections.return_value = {
            "mean": mock_mean,
            "max_projection": mock_max,
            "std": mock_std,
            "sum": mock_sum,
        }
        
        # Setup pipeline
        with patch('astroglial_morphology.pipeline.Segmentation'):
            pipeline = Pipeline(data_path=str(temp_dir))
            pipeline.metadata = sample_metadata
            pipeline.suite2p_options = {"batch_size": 500}
        
            # Create projections
            projections = pipeline.create_projections()
        
            # Verify projections were created
            assert "mean" in projections
            assert "max_projection" in projections
            assert "std" in projections
            assert "sum" in projections


class TestPipelineSegmentCells:
    """Tests for Pipeline.segment_cells method."""
    
    @patch('astroglial_morphology.pipeline.Segmentation')
    def test_segment_cells(self, mock_seg_class, temp_dir, sample_metadata):
        """Test cell segmentation."""
        # Mock Segmentation
        mock_seg = Mock()
        mock_masks = np.zeros((512, 1024), dtype=np.uint16)
        mock_masks[100:200, 100:200] = 1
        mock_masks[300:400, 700:800] = 2
        
        mock_seg.segment_img = Mock(return_value=mock_masks)
        mock_seg_class.return_value = mock_seg
        
        # Setup pipeline
        pipeline = Pipeline(data_path=str(temp_dir))
        pipeline.metadata = sample_metadata
        pipeline.projections = {
            "mean": np.random.rand(512, 1024).astype(np.float32)
        }

        # Segment cells
        masks = pipeline.segment_cells()

        assert masks.shape == (512, 1024)
        assert np.max(masks) == 2  # Two cells
        mock_seg.segment_img.assert_called_once()


class TestPipelineClassifyCells:
    """Tests for Pipeline.classify_cells method."""
    
    @patch('astroglial_morphology.pipeline.classify_cells')
    def test_classify_cells(self, mock_classify, temp_dir, sample_metadata):
        """Test cell classification."""
        # Mock classification results
        mock_pairs = [(2, 1), (1, 2)]
        mock_dict = {
            1: {"type": "lower", "area": 100.0},
            2: {"type": "upper", "area": 50000.0}
        }
        mock_classify.return_value = (mock_pairs, mock_dict)
        
        # Setup pipeline
        with patch('astroglial_morphology.pipeline.Segmentation'):
            pipeline = Pipeline(data_path=str(temp_dir))
            pipeline.metadata = sample_metadata
            pipeline.masks = np.zeros((512, 1024), dtype=np.uint16)
            pipeline.masks[100:200, 100:200] = 1
            pipeline.masks[300:400, 700:800] = 2

            # Classify cells
            pairs, cell_dict = pipeline.classify_cells()

            assert len(pairs) == 2
            assert len(cell_dict) == 2
            mock_classify.assert_called_once()


class TestPipelineRun:
    """Integration tests for full pipeline execution."""
    
    @patch('astroglial_morphology.pipeline.detect_input_file')
    @patch('astroglial_morphology.pipeline.load_metadata')
    @patch('astroglial_morphology.pipeline.lif_to_suite2p_binary')
    @patch('astroglial_morphology.pipeline.do_registration')
    @patch('astroglial_morphology.pipeline.check_registration_complete')
    @patch('astroglial_morphology.pipeline.create_projections')
    @patch('astroglial_morphology.pipeline.Segmentation')
    @patch('astroglial_morphology.pipeline.classify_cells')
    def test_full_pipeline_lif(
        self,
        mock_classify,
        mock_seg_class,
        mock_create_projections,
        mock_check_registration,
        mock_do_registration,
        mock_lif_convert,
        mock_load_metadata,
        mock_detect_file,
        temp_dir
    ):
        """Test full pipeline execution with LIF file."""
        # Setup mocks
        lif_file = temp_dir / "test.lif"
        lif_file.touch()
        
        mock_detect_file.return_value = InputFileInfo(
            path=lif_file,
            format=InputFormat.LIF
        )
        
        mock_load_metadata.return_value = Metadata(
            nframes=100,
            nchannels=1,
            nplanes=1,
            finterval=1.0,
            pix_resolution=8.36
        )
        
        mock_lif_convert.return_value = {
            "Ly": 512,
            "Lx": 1024,
            "Lys": [512],
            "Lxs": [1024]
        }
        
        mock_check_registration.return_value = False

        # Mock projections
        mock_create_projections.return_value = {
            "mean": np.random.rand(512, 1024).astype(np.float32),
            "max_projection": np.random.randint(-1000, 1000, (512, 1024), dtype=np.int16),
            "std": np.random.rand(512, 1024).astype(np.float32),
            "sum": np.random.rand(512, 1024).astype(np.float64),
        }
        
        # Mock segmentation
        mock_seg = Mock()
        mock_masks = np.zeros((512, 1024), dtype=np.uint16)
        mock_masks[100:200, 100:200] = 1
        mock_seg.segment_img = Mock(return_value=mock_masks)
        mock_seg_class.return_value = mock_seg
        
        # Mock classification
        mock_classify.return_value = (
            [(1, 1)],
            {1: {"type": "upper", "area": 10000.0}}
        )
        
        # Run pipeline
        pipeline = Pipeline(data_path=str(temp_dir))
        results = pipeline.run()
        
        # Verify all steps were called
        mock_detect_file.assert_called_once()
        mock_load_metadata.assert_called_once()
        mock_lif_convert.assert_called_once()
        mock_do_registration.assert_called_once()
        mock_seg.segment_img.assert_called_once()
        mock_classify.assert_called_once()
        
        # Verify results structure
        assert isinstance(results, dict)
        assert "metadata" in results
        assert "masks" in results


@pytest.mark.skipif(
    not Path(r"C:\Users\javid.rezai\YaksiLab\duygu\data\Lif_data").exists(),
    reason="Real LIF test data not available"
)
class TestPipelineWithRealData:
    """Integration tests with real LIF data (if available)."""
    
    def test_full_pipeline_with_real_lif(self, real_lif_data_path):
        """Test complete pipeline with real LIF file."""
        if real_lif_data_path is None:
            pytest.skip("Real LIF data not available")
        
        # This test would run the actual pipeline on real data
        # It's slow and requires actual data, so it's skipped by default
        with patch('astroglial_morphology.pipeline.Segmentation'):
            pipeline = Pipeline(data_path=str(real_lif_data_path))
        
            # Test just the detection and metadata loading
            pipeline.detect_input()
            assert pipeline.file_info is not None

            pipeline.load_metadata()
            assert pipeline.metadata.nframes > 0
            assert pipeline.metadata.pix_resolution > 0

    def test_real_data_metadata_and_prepare(self, real_lif_data_path, caplog):
        """Test real data metadata and preparation align with expected log output."""
        if real_lif_data_path is None:
            pytest.skip("Real LIF data not available")

        caplog.set_level(logging.INFO)

        with patch('astroglial_morphology.pipeline.Segmentation'):
            pipeline = Pipeline(data_path=str(real_lif_data_path))

            # Detect input and verify
            pipeline.detect_input()
            assert pipeline.file_info is not None
            assert pipeline.file_info.format == InputFormat.LIF

            # Load metadata and verify values match log output
            pipeline.load_metadata()
            assert pipeline.metadata is not None
            assert pipeline.metadata.nframes == 1000
            assert pipeline.metadata.nplanes == 1
            assert pipeline.metadata.nchannels == 1
            assert pipeline.metadata.pix_resolution == pytest.approx(8.3600, rel=1e-4)
            assert pipeline.metadata.finterval == pytest.approx(6.8182, rel=1e-4)

            # Prepare data (LIF conversion or reuse)
            pipeline.prepare_data()

            # Suite2p options should reflect binary input for LIF
            assert pipeline.suite2p_options is not None
            assert pipeline.suite2p_options.get("input_format") == "binary"
            assert pipeline.suite2p_options.get("Lys") == [512]
            assert pipeline.suite2p_options.get("Lxs") == [1024]
            assert pipeline.suite2p_options.get("nimg_init") == 150
            assert pipeline.suite2p_options.get("batch_size") == 500

            # Confirm expected log output
            assert "Using binary input format for LIF-converted data" in caplog.text

            # Verify conversion outputs exist
            suite2p_plane0 = Path(real_lif_data_path) / "suite2p" / "plane0"
            data_bin = suite2p_plane0 / "data.bin"
            ops_npy = suite2p_plane0 / "ops.npy"
            assert data_bin.exists()
            assert ops_npy.exists()

            # Load ops and verify dimensions
            ops = np.load(ops_npy, allow_pickle=True).item()
            assert ops.get("Ly") == 512
            assert ops.get("Lx") == 1024

    @pytest.mark.slow
    def test_real_data_full_pipeline_logs(self, real_lif_data_path, caplog):
        """Run full pipeline on real data and assert key log messages."""
        if real_lif_data_path is None:
            pytest.skip("Real LIF data not available")

        caplog.set_level(logging.INFO)

        pipeline = Pipeline(data_path=str(real_lif_data_path))
        results = pipeline.run()

        # Required log messages for successful completion
        expected_messages = [
            "Starting astroglial morphology pipeline",
            "Detecting input file...",
            "Loading metadata...",
            "Using binary input format for LIF-converted data",
            "Creating projections...",
            "Segmenting cells on mean projection...",
            "Classifying astrocyte morphology...",
            "Pipeline completed successfully",
        ]

        for msg in expected_messages:
            assert msg in caplog.text

        # Registration can be either performed or skipped depending on existing outputs
        registration_messages = [
            "Starting motion correction...",
            "Registration completed",
            "Registration already complete - skipping",
        ]
        assert any(msg in caplog.text for msg in registration_messages)

        # Validate results structure
        assert isinstance(results, dict)
        assert results.get("metadata") is not None
        assert results.get("projections") is not None
        assert results.get("masks") is not None
        assert results.get("classification") is not None


class TestPipelineSkipRegistration:
    """Tests for pipeline with skip_registration flag."""
    
    @patch('astroglial_morphology.pipeline.detect_input_file')
    @patch('astroglial_morphology.pipeline.load_metadata')
    @patch('astroglial_morphology.pipeline.create_projections')
    @patch('astroglial_morphology.pipeline.Segmentation')
    @patch('astroglial_morphology.pipeline.classify_cells')
    @patch('astroglial_morphology.pipeline.do_registration')
    def test_skip_registration(
        self,
        mock_do_registration,
        mock_classify,
        mock_seg_class,
        mock_create_projections,
        mock_load_metadata,
        mock_detect_file,
        temp_dir
    ):
        """Test pipeline with skip_registration=True."""
        # Setup mocks
        tiff_file = temp_dir / "test.tif"
        tiff_file.touch()
        
        mock_detect_file.return_value = InputFileInfo(
            path=tiff_file,
            format=InputFormat.TIFF
        )
        
        mock_load_metadata.return_value = Metadata(
            nframes=100,
            nchannels=1,
            nplanes=1,
            finterval=1.0,
            pix_resolution=8.36
        )
        
        # Mock projections
        mock_create_projections.return_value = {
            "mean": np.random.rand(512, 1024).astype(np.float32),
            "max_projection": np.random.randint(-1000, 1000, (512, 1024), dtype=np.int16),
            "std": np.random.rand(512, 1024).astype(np.float32),
            "sum": np.random.rand(512, 1024).astype(np.float64),
        }
        
        # Mock segmentation
        mock_seg = Mock()
        mock_masks = np.zeros((512, 1024), dtype=np.uint16)
        mock_seg.segment_img = Mock(return_value=mock_masks)
        mock_seg_class.return_value = mock_seg
        
        # Mock classification
        mock_classify.return_value = ([], {})
        
        # Run pipeline with skip_registration
        pipeline = Pipeline(data_path=str(temp_dir))

        results = pipeline.run(skip_registration=True)

        # Registration should NOT be called
        mock_do_registration.assert_not_called()
