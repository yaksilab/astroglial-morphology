"""Integration tests for the complete pipeline."""

import json
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

    @patch('astroglial_morphology.pipeline.Segmentation')
    def test_segment_cells_with_both_channels(self, mock_seg_class, temp_dir):
        """Test two-channel segmentation passes HxWx2 data to Cellpose."""
        mock_seg = Mock()
        mock_masks = np.zeros((512, 1024), dtype=np.uint16)
        mock_seg.segment_img = Mock(return_value=mock_masks)
        mock_seg_class.return_value = mock_seg

        pipeline = Pipeline(data_path=str(temp_dir))
        pipeline.metadata = Metadata(
            nframes=1000,
            nchannels=2,
            nplanes=1,
            finterval=1.0,
            pix_resolution=8.36
        )
        pipeline.suite2p_options = {"nchannels": 2}
        pipeline.projections = {
            "mean_ch0": np.random.rand(512, 1024).astype(np.float32),
            "mean_ch1": np.random.rand(512, 1024).astype(np.float32),
        }

        pipeline.segment_cells(
            projection_type="mean",
            segmentation_channel="both",
        )

        img_arg = mock_seg.segment_img.call_args.args[0]
        save_path = mock_seg.segment_img.call_args.args[1]
        kwargs = mock_seg.segment_img.call_args.kwargs
        assert img_arg.shape == (512, 1024, 2)
        assert kwargs["channel_axis"] == -1
        assert save_path.endswith("mean_both_image")

    @patch('astroglial_morphology.pipeline.Segmentation')
    def test_segment_cells_with_selected_channel(self, mock_seg_class, temp_dir):
        """Test selected-channel segmentation uses the requested channel."""
        mock_seg = Mock()
        mock_masks = np.zeros((512, 1024), dtype=np.uint16)
        mock_seg.segment_img = Mock(return_value=mock_masks)
        mock_seg_class.return_value = mock_seg

        channel0 = np.zeros((512, 1024), dtype=np.float32)
        channel1 = np.ones((512, 1024), dtype=np.float32)
        pipeline = Pipeline(data_path=str(temp_dir))
        pipeline.metadata = Metadata(
            nframes=1000,
            nchannels=2,
            nplanes=1,
            finterval=1.0,
            pix_resolution=8.36
        )
        pipeline.suite2p_options = {"nchannels": 2}
        pipeline.projections = {"mean_ch0": channel0, "mean_ch1": channel1}

        pipeline.segment_cells(projection_type="mean", segmentation_channel="1")

        img_arg = mock_seg.segment_img.call_args.args[0]
        save_path = mock_seg.segment_img.call_args.args[1]
        np.testing.assert_array_equal(img_arg, channel1)
        assert "channel_axis" not in mock_seg.segment_img.call_args.kwargs
        assert save_path.endswith("mean_ch1_image")


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


class TestPipelineCorrespondenceExport:
    """Tests for correspondence and trace export wiring."""

    @patch('astroglial_morphology.pipeline.export_correspondence_products')
    @patch('astroglial_morphology.pipeline.Segmentation')
    def test_export_correspondence_passes_trace_channels(
        self, mock_seg_class, mock_export, temp_dir
    ):
        """Test selected trace channels are passed to the export layer."""
        seg_path = temp_dir / "suite2p" / "plane0" / "mean_both_image_seg.npy"
        seg_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(seg_path, {"masks": np.ones((10, 10), dtype=np.uint16)})
        mock_export.return_value = {"trace_matrix_paths": {1: temp_dir / "trace.npy"}}

        pipeline = Pipeline(data_path=str(temp_dir))
        pipeline.metadata = Metadata(
            nframes=100,
            nchannels=2,
            nplanes=1,
            finterval=1.0,
            pix_resolution=8.36
        )
        pipeline.masks = np.ones((10, 10), dtype=np.uint16)
        pipeline.classification = [(1, 1)]
        pipeline.segmentation_base_path = str(seg_path).removesuffix("_seg.npy")

        pipeline.export_correspondence_data(trace_channels=[1])

        assert mock_export.call_args.kwargs["trace_channels"] == [1]


class TestPipelineMetadata:
    """Tests for pipeline metadata export."""

    @patch('astroglial_morphology.pipeline.Segmentation')
    def test_write_pipeline_metadata_includes_acquisition_settings_and_qc(
        self, mock_seg_class, temp_dir
    ):
        lif_file = temp_dir / "test.lif"
        lif_file.write_bytes(b"fake-lif")
        suite2p_dir = temp_dir / "suite2p"
        plane_path = suite2p_dir / "plane0"
        plane_path.mkdir(parents=True)
        complete_flag = suite2p_dir / ".registration_complete"
        complete_flag.touch()

        data_bin = plane_path / "data.bin"
        data_chan2_bin = plane_path / "data_chan2.bin"
        data_bin.touch()
        data_chan2_bin.touch()
        np.save(
            plane_path / "ops.npy",
            {
                "Ly": 12,
                "Lx": 34,
                "nframes": 5,
                "nchannels": 2,
                "channel_indices": [0, 1],
                "reg_file": str(data_bin),
                "reg_file_chan2": str(data_chan2_bin),
                "meanImg": np.zeros((12, 34), dtype=np.float32),
                "meanImg_chan2": np.zeros((12, 34), dtype=np.float32),
                "refImg": np.zeros((12, 34), dtype=np.float32),
                "badframes": np.array([False, True, False, True, False]),
                "xoff": np.array([-1.0, 0.0, 2.0]),
                "yoff": np.array([1.0, 3.0, 5.0]),
                "corrXY": np.array([0.2, 0.5, 0.8]),
                "timing": {"registration": np.float32(1.25)},
                "suite2p_version": "0.14.5",
            },
            allow_pickle=True,
        )

        pipeline = Pipeline(data_path=str(temp_dir), use_gpu=True, reg_tif=True)
        pipeline.file_info = InputFileInfo(path=lif_file, format=InputFormat.LIF)
        pipeline.metadata = Metadata(
            nframes=10,
            nchannels=2,
            nplanes=1,
            finterval=2.0,
            pix_resolution=1.76,
            series_name="Series001",
            series_index=0,
        )
        pipeline.suite2p_options = {
            "nplanes": 1,
            "nchannels": 2,
            "fs": 0.5,
            "do_registration": True,
            "two_step_registration": False,
            "nonrigid": False,
            "maxregshift": 0.11,
            "subpixel": 10,
            "align_by_chan": 2,
            "functional_chan": 1,
            "batch_size": 500,
            "nimg_init": 300,
            "do_regmetrics": True,
            "reg_tif": True,
            "reg_tif_chan2": True,
            "roidetect": False,
            "spikedetect": False,
            "reg_file": str(data_bin),
            "reg_file_chan2": str(data_chan2_bin),
            "input_format": "binary",
        }
        pipeline.registration_channel = 1
        pipeline.segmentation_channel = "both"
        pipeline.segmentation_projection = "max_projection"
        pipeline.trace_channels = [0, 1]
        pipeline.do_regmetrics = True
        pipeline.manual_correction = True
        pipeline.export_correspondence = True
        pipeline.alignment_only = True

        pipeline.write_pipeline_metadata()

        payload = json.loads((plane_path / "pipeline_metadata.json").read_text())
        assert payload["input_file"] == str(lif_file)
        assert payload["input_filename"] == "test.lif"
        assert payload["input_file_extension"] == ".lif"
        assert payload["input_file_type"] == "lif"
        assert payload["input_file_size_bytes"] == 8
        assert payload["series_index"] == 0
        assert payload["series_name"] == "Series001"
        assert payload["plane_index"] == 0
        assert payload["source_nframes"] == 10
        assert payload["nframes_registered"] == 5
        assert payload["source_nchannels"] == 2
        assert payload["converted_nchannels"] == 2
        assert payload["channel_indices"] == [0, 1]
        assert payload["nplanes"] == 1
        assert payload["Ly"] == 12
        assert payload["Lx"] == 34
        assert payload["pixel_resolution"] == 1.76
        assert payload["frame_interval_seconds"] == 2.0
        assert payload["fs"] == 0.5
        assert payload["frames_per_channel_per_plane"] == 5
        assert payload["registration_complete"] is True
        assert payload["registration_completed_at"] is not None
        assert payload["suite2p_output_dir"] == str(suite2p_dir)
        assert payload["ops_path"] == str(plane_path / "ops.npy")
        assert payload["meanImg_shape"] == [12, 34]
        assert payload["meanImg_chan2_shape"] == [12, 34]
        assert payload["refImg_shape"] == [12, 34]
        assert payload["num_badframes"] == 2
        assert payload["badframes_fraction"] == 0.4
        assert payload["xoff_min"] == -1.0
        assert payload["xoff_max"] == 2.0
        assert payload["xoff_mean"] == pytest.approx(1.0 / 3.0)
        assert payload["yoff_std"] == pytest.approx(np.std([1.0, 3.0, 5.0]))
        assert payload["corrXY_mean"] == pytest.approx(0.5)
        assert payload["suite2p_timing"]["registration"] == pytest.approx(1.25)
        assert payload["registration_channel"] == 1
        assert payload["suite2p_align_by_chan"] == 2
        assert payload["do_registration"] is True
        assert payload["do_regmetrics"] is True
        assert payload["reg_tif"] is True
        assert payload["reg_tif_chan2"] is True
        assert payload["input_format"] == "binary"
        assert payload["alignment_only"] is True
        assert payload["export_correspondence"] is True
        assert payload["manual_correction"] is True
        assert payload["model_path"] == pipeline.model_path
        assert payload["use_gpu"] is True
        assert payload["reg_file"] == str(data_bin)
        assert payload["reg_file_chan2"] == str(data_chan2_bin)
        assert payload["python_version"]
        assert payload["suite2p_version"] == "0.14.5"
        assert payload["numpy_version"] == np.__version__
        assert payload["platform"]
        assert payload["created_at"]
        assert payload["hostname"]


class TestPipelineRegistrationReuse:
    """Tests for safe reuse and rebuilding of Suite2p registration inputs."""

    @patch('astroglial_morphology.pipeline.lif_to_suite2p_binary')
    @patch('astroglial_morphology.pipeline.Segmentation')
    def test_lif_reconversion_invalidates_registration_completion(
        self, mock_seg_class, mock_lif_convert, temp_dir
    ):
        lif_file = temp_dir / "test.lif"
        lif_file.touch()
        suite2p_dir = temp_dir / "suite2p"
        plane_path = suite2p_dir / "plane0"
        plane_path.mkdir(parents=True)
        data_bin = plane_path / "data.bin"
        data_chan2_bin = plane_path / "data_chan2.bin"
        data_bin.write_bytes(b"old-registered-data")
        np.save(
            plane_path / "ops.npy",
            {"Ly": 6, "Lx": 8, "nframes": 4, "nchannels": 1},
            allow_pickle=True,
        )
        complete_flag = suite2p_dir / ".registration_complete"
        complete_flag.touch()

        def convert_two_channels(**kwargs):
            data_bin.write_bytes(b"raw-channel-0")
            data_chan2_bin.write_bytes(b"raw-channel-1")
            return {
                "Ly": 6,
                "Lx": 8,
                "Lys": [6],
                "Lxs": [8],
                "nframes": 4,
                "nchannels": 2,
                "reg_file": str(data_bin),
                "reg_file_chan2": str(data_chan2_bin),
            }

        mock_lif_convert.side_effect = convert_two_channels
        pipeline = Pipeline(data_path=str(temp_dir))
        pipeline.file_info = InputFileInfo(path=lif_file, format=InputFormat.LIF)
        pipeline.metadata = Metadata(
            nframes=8,
            nchannels=2,
            nplanes=1,
            finterval=1.0,
            pix_resolution=8.36,
        )

        pipeline.prepare_data()

        mock_lif_convert.assert_called_once()
        assert not complete_flag.exists()

    @patch('astroglial_morphology.pipeline.do_registration')
    @patch('astroglial_morphology.pipeline.Segmentation')
    def test_changed_registration_channel_rebuilds_tiff_inputs(
        self, mock_seg_class, mock_do_registration, temp_dir
    ):
        tiff_file = temp_dir / "test.tif"
        tiff_file.touch()
        suite2p_dir = temp_dir / "suite2p"
        plane_path = suite2p_dir / "plane0"
        plane_path.mkdir(parents=True)
        data_bin = plane_path / "data.bin"
        data_chan2_bin = plane_path / "data_chan2.bin"
        data_bin.write_bytes(b"registered-channel-0")
        data_chan2_bin.write_bytes(b"registered-channel-1")
        np.save(
            plane_path / "ops.npy",
            {
                "align_by_chan": 1,
                "functional_chan": 1,
                "nchannels": 2,
            },
            allow_pickle=True,
        )
        complete_flag = suite2p_dir / ".registration_complete"
        complete_flag.touch()

        pipeline = Pipeline(data_path=str(temp_dir))
        pipeline.file_info = InputFileInfo(path=tiff_file, format=InputFormat.TIFF)
        pipeline.metadata = Metadata(
            nframes=200,
            nchannels=2,
            nplanes=1,
            finterval=1.0,
            pix_resolution=8.36,
        )
        pipeline.registration_channel = 1
        pipeline.prepare_data()

        performed = pipeline.run_registration()

        assert performed is True
        mock_do_registration.assert_called_once()
        assert mock_do_registration.call_args.args[1]["align_by_chan"] == 2
        assert not complete_flag.exists()
        assert not data_bin.exists()
        assert not data_chan2_bin.exists()
        assert not (plane_path / "ops.npy").exists()

    @patch('astroglial_morphology.pipeline.do_registration')
    @patch('astroglial_morphology.pipeline.Segmentation')
    def test_matching_registration_channel_reuses_completed_inputs(
        self, mock_seg_class, mock_do_registration, temp_dir
    ):
        tiff_file = temp_dir / "test.tif"
        tiff_file.touch()
        suite2p_dir = temp_dir / "suite2p"
        plane_path = suite2p_dir / "plane0"
        plane_path.mkdir(parents=True)
        data_bin = plane_path / "data.bin"
        data_bin.write_bytes(b"registered-channel-0")
        np.save(
            plane_path / "ops.npy",
            {
                "align_by_chan": 1,
                "functional_chan": 1,
                "nchannels": 1,
            },
            allow_pickle=True,
        )
        (suite2p_dir / ".registration_complete").touch()

        pipeline = Pipeline(data_path=str(temp_dir))
        pipeline.file_info = InputFileInfo(path=tiff_file, format=InputFormat.TIFF)
        pipeline.metadata = Metadata(
            nframes=200,
            nchannels=1,
            nplanes=1,
            finterval=1.0,
            pix_resolution=8.36,
        )
        pipeline.prepare_data()

        performed = pipeline.run_registration()

        assert performed is False
        mock_do_registration.assert_not_called()
        assert data_bin.exists()


class TestPipelineChannelValidation:
    """Tests for the one/two-channel processing limit."""

    @patch('astroglial_morphology.pipeline.Segmentation')
    def test_three_channel_tiff_is_rejected(self, mock_seg_class, temp_dir):
        tiff_file = temp_dir / "test.tif"
        tiff_file.touch()
        pipeline = Pipeline(data_path=str(temp_dir))
        pipeline.file_info = InputFileInfo(path=tiff_file, format=InputFormat.TIFF)
        pipeline.metadata = Metadata(
            nframes=300,
            nchannels=3,
            nplanes=1,
            finterval=1.0,
            pix_resolution=8.36,
        )

        with pytest.raises(ValueError, match="supports at most two channels"):
            pipeline.prepare_data()


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
            "Lxs": [1024],
            "nchannels": 1,
            "nframes": 100,
            "reg_file": str(temp_dir / "suite2p" / "plane0" / "data.bin"),
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

    @patch('astroglial_morphology.pipeline.detect_input_file')
    @patch('astroglial_morphology.pipeline.load_metadata')
    @patch('astroglial_morphology.pipeline.do_registration')
    @patch('astroglial_morphology.pipeline.check_registration_complete')
    @patch('astroglial_morphology.pipeline.create_projections')
    @patch('astroglial_morphology.pipeline.Segmentation')
    @patch('astroglial_morphology.pipeline.classify_cells')
    def test_registration_channel_maps_to_suite2p_align_by_chan(
        self,
        mock_classify,
        mock_seg_class,
        mock_create_projections,
        mock_check_registration,
        mock_do_registration,
        mock_load_metadata,
        mock_detect_file,
        temp_dir,
    ):
        """Test zero-based registration channel maps to Suite2p's one-based option."""
        tiff_file = temp_dir / "test.tif"
        tiff_file.touch()
        mock_detect_file.return_value = InputFileInfo(
            path=tiff_file,
            format=InputFormat.TIFF
        )
        mock_load_metadata.return_value = Metadata(
            nframes=200,
            nchannels=2,
            nplanes=1,
            finterval=1.0,
            pix_resolution=8.36
        )
        mock_check_registration.return_value = False
        mock_create_projections.return_value = {
            "mean_ch0": np.random.rand(512, 1024).astype(np.float32),
            "mean_ch1": np.random.rand(512, 1024).astype(np.float32),
            "mean": np.random.rand(512, 1024).astype(np.float32),
            "max_projection": np.random.rand(512, 1024).astype(np.float32),
        }
        mock_seg = Mock()
        mock_seg.segment_img = Mock(return_value=np.zeros((512, 1024), dtype=np.uint16))
        mock_seg_class.return_value = mock_seg
        mock_classify.return_value = ([], {})

        pipeline = Pipeline(data_path=str(temp_dir))
        pipeline.run(registration_channel=1, export_correspondence=False)

        options = mock_do_registration.call_args.args[1]
        assert options["align_by_chan"] == 2

    @patch('astroglial_morphology.pipeline.detect_input_file')
    @patch('astroglial_morphology.pipeline.load_metadata')
    @patch('astroglial_morphology.pipeline.Segmentation')
    def test_trace_channels_required_for_two_channel_export(
        self,
        mock_seg_class,
        mock_load_metadata,
        mock_detect_file,
        temp_dir,
    ):
        """Test multi-channel correspondence export requires explicit trace channels."""
        tiff_file = temp_dir / "test.tif"
        tiff_file.touch()
        mock_detect_file.return_value = InputFileInfo(
            path=tiff_file,
            format=InputFormat.TIFF
        )
        mock_load_metadata.return_value = Metadata(
            nframes=200,
            nchannels=2,
            nplanes=1,
            finterval=1.0,
            pix_resolution=8.36
        )

        pipeline = Pipeline(data_path=str(temp_dir))
        with pytest.raises(ValueError, match="trace_channels must be specified"):
            pipeline.run(export_correspondence=True)

    @patch('astroglial_morphology.pipeline.detect_input_file')
    @patch('astroglial_morphology.pipeline.load_metadata')
    @patch('astroglial_morphology.pipeline.do_registration')
    @patch('astroglial_morphology.pipeline.check_registration_complete')
    @patch('astroglial_morphology.pipeline.create_projections')
    @patch('astroglial_morphology.pipeline.Segmentation')
    def test_alignment_only_stops_before_segmentation(
        self,
        mock_seg_class,
        mock_create_projections,
        mock_check_registration,
        mock_do_registration,
        mock_load_metadata,
        mock_detect_file,
        temp_dir,
    ):
        """Test alignment-only mode returns before segmentation."""
        tiff_file = temp_dir / "test.tif"
        tiff_file.touch()
        mock_detect_file.return_value = InputFileInfo(
            path=tiff_file,
            format=InputFormat.TIFF
        )
        mock_load_metadata.return_value = Metadata(
            nframes=200,
            nchannels=2,
            nplanes=1,
            finterval=1.0,
            pix_resolution=8.36
        )
        mock_check_registration.return_value = False
        mock_create_projections.return_value = {
            "mean_ch0": np.random.rand(512, 1024).astype(np.float32),
            "mean_ch1": np.random.rand(512, 1024).astype(np.float32),
        }

        pipeline = Pipeline(data_path=str(temp_dir))
        results = pipeline.run(alignment_only=True)

        mock_do_registration.assert_called_once()
        mock_seg_class.return_value.segment_img.assert_not_called()
        assert results["masks"] is None


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
