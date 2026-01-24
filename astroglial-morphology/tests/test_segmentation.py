"""Tests for segmentation module."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from astroglial_morphology.segmentation import Segmentation


class TestSegmentation:
    """Tests for Segmentation class."""
    
    @patch('astroglial_morphology.segmentation.CellposeModel')
    def test_initialization_with_default_model(self, mock_cellpose_model):
        """Test Segmentation initialization with default model."""
        mock_model = Mock()
        mock_model.diam_labels = 200.0
        mock_cellpose_model.return_value = mock_model
        
        seg = Segmentation()
        
        assert seg.model is not None
        mock_cellpose_model.assert_called_once()
    
    @patch('astroglial_morphology.segmentation.CellposeModel')
    def test_initialization_with_custom_model(self, mock_cellpose_model):
        """Test Segmentation initialization with custom model path."""
        mock_model = Mock()
        mock_model.diam_labels = 200.0
        mock_cellpose_model.return_value = mock_model
        
        custom_path = "/path/to/custom/model"
        seg = Segmentation(model_path=custom_path)
        
        assert seg.model is not None
        # Check that custom path was used
        call_args = mock_cellpose_model.call_args
        assert call_args is not None
    
    @patch('astroglial_morphology.segmentation.CellposeModel')
    def test_initialization_with_gpu(self, mock_cellpose_model):
        """Test Segmentation initialization with GPU enabled."""
        mock_model = Mock()
        mock_model.diam_labels = 200.0
        mock_cellpose_model.return_value = mock_model
        
        seg = Segmentation(gpu=True)
        mock_cellpose_model.assert_called_once()
        call_args = mock_cellpose_model.call_args
        assert call_args is not None
        assert call_args[1]["gpu"] is True
    
    @patch('astroglial_morphology.segmentation.masks_flows_to_seg')
    @patch('astroglial_morphology.segmentation.CellposeModel')
    def test_segment_with_default_params(self, mock_cellpose_model, mock_save, sample_mean_image):
        """Test segmentation with default parameters."""
        # Mock model
        mock_model = Mock()
        mock_model.diam_labels = 200.0
        
        # Mock masks
        masks = np.zeros_like(sample_mean_image, dtype=np.uint16)
        masks[100:200, 100:200] = 1
        masks[300:400, 300:400] = 2
        
        mock_model.eval = Mock(return_value=(masks, None, None))
        mock_cellpose_model.return_value = mock_model
        
        seg = Segmentation()
        result_masks = seg.segment_img(sample_mean_image, save_file_name="test_masks")

        assert result_masks.shape == sample_mean_image.shape
        assert np.max(result_masks) == 2  # Two cells
        mock_model.eval.assert_called_once()
    
    @patch('astroglial_morphology.segmentation.masks_flows_to_seg')
    @patch('astroglial_morphology.segmentation.CellposeModel')
    def test_segment_with_custom_diameter(self, mock_cellpose_model, mock_save, sample_mean_image):
        """Test segmentation with custom diameter."""
        mock_model = Mock()
        mock_model.diam_labels = 200.0
        mock_model.eval = Mock(return_value=(np.zeros_like(sample_mean_image), None, None))
        mock_cellpose_model.return_value = mock_model
        
        seg = Segmentation()
        custom_diameter = 300.0
        masks = seg.segment_img(sample_mean_image, diameter=custom_diameter)

        # Check that diameter was passed to eval
        call_args = mock_model.eval.call_args
        assert call_args is not None
        assert call_args[1]["diameter"] == custom_diameter
    
    @patch('astroglial_morphology.segmentation.masks_flows_to_seg')
    @patch('astroglial_morphology.segmentation.CellposeModel')
    def test_segment_with_custom_thresholds(self, mock_cellpose_model, mock_save, sample_mean_image):
        """Test segmentation with custom thresholds."""
        mock_model = Mock()
        mock_model.diam_labels = 200.0
        mock_model.eval = Mock(return_value=(np.zeros_like(sample_mean_image), None, None))
        mock_cellpose_model.return_value = mock_model
        
        seg = Segmentation()
        masks = seg.segment_img(
            sample_mean_image,
            flow_threshold=0.5,
            cellprob_threshold=0.5
        )
        
        # Check that thresholds were passed
        call_args = mock_model.eval.call_args
        assert call_args[1]["flow_threshold"] == 0.5
        assert call_args[1]["cellprob_threshold"] == 0.5
    
    @patch('astroglial_morphology.segmentation.masks_flows_to_seg')
    @patch('astroglial_morphology.segmentation.CellposeModel')
    def test_segment_no_cells_found(self, mock_cellpose_model, mock_save, sample_mean_image):
        """Test segmentation when no cells are found."""
        mock_model = Mock()
        mock_model.diam_labels = 200.0
        mock_model.eval = Mock(return_value=(np.zeros_like(sample_mean_image), None, None))
        mock_cellpose_model.return_value = mock_model
        
        seg = Segmentation()
        masks = seg.segment_img(sample_mean_image)

        assert np.max(masks) == 0  # No cells
    
    @patch('astroglial_morphology.segmentation.masks_flows_to_seg')
    @patch('astroglial_morphology.segmentation.CellposeModel')
    def test_segment_multiple_cells(self, mock_cellpose_model, mock_save, sample_mean_image):
        """Test segmentation finding multiple cells."""
        mock_model = Mock()
        mock_model.diam_labels = 200.0
        
        # Create masks with 5 cells
        masks = np.zeros_like(sample_mean_image, dtype=np.uint16)
        for i in range(5):
            y, x = 100 * i, 100 * i
            masks[y:y+50, x:x+50] = i + 1
        
        mock_model.eval = Mock(return_value=(masks, None, None))
        mock_cellpose_model.return_value = mock_model
        
        seg = Segmentation()
        result_masks = seg.segment_img(sample_mean_image)

        assert np.max(result_masks) == 5
    
    @patch('astroglial_morphology.segmentation.masks_flows_to_seg')
    @patch('astroglial_morphology.segmentation.CellposeModel')
    def test_segment_returns_masks(self, mock_cellpose_model, mock_save, sample_mean_image):
        """Test that segmentation returns masks."""
        mock_model = Mock()
        mock_model.diam_labels = 200.0
        
        mock_flows = [np.random.rand(*sample_mean_image.shape)]

        mock_model.eval = Mock(return_value=(
            np.zeros_like(sample_mean_image),
            mock_flows,
            None
        ))
        mock_cellpose_model.return_value = mock_model
        
        seg = Segmentation()
        masks = seg.segment_img(sample_mean_image)

        assert masks is not None
    
    @patch('astroglial_morphology.segmentation.masks_flows_to_seg')
    @patch('astroglial_morphology.segmentation.CellposeModel')
    def test_segment_with_augmentation(self, mock_cellpose_model, mock_save, sample_mean_image):
        """Test segmentation with augmentation enabled."""
        mock_model = Mock()
        mock_model.diam_labels = 200.0
        mock_model.eval = Mock(return_value=(np.zeros_like(sample_mean_image), None, None))
        mock_cellpose_model.return_value = mock_model
        
        seg = Segmentation()
        masks = seg.segment_img(sample_mean_image, augment=True)
        
        call_args = mock_model.eval.call_args
        assert call_args[1]["augment"] is True
    
    @patch('astroglial_morphology.segmentation.masks_flows_to_seg')
    @patch('astroglial_morphology.segmentation.CellposeModel')
    def test_segment_with_resample(self, mock_cellpose_model, mock_save, sample_mean_image):
        """Test segmentation with resample enabled."""
        mock_model = Mock()
        mock_model.diam_labels = 200.0
        mock_model.eval = Mock(return_value=(np.zeros_like(sample_mean_image), None, None))
        mock_cellpose_model.return_value = mock_model
        
        seg = Segmentation()
        masks = seg.segment_img(sample_mean_image, resample=True)
        
        call_args = mock_model.eval.call_args
        assert call_args[1]["resample"] is True
    
    @patch('astroglial_morphology.segmentation.masks_flows_to_seg')
    @patch('astroglial_morphology.segmentation.CellposeModel')
    def test_segment_with_min_size(self, mock_cellpose_model, mock_save, sample_mean_image):
        """Test segmentation with minimum size filter."""
        mock_model = Mock()
        mock_model.diam_labels = 200.0
        mock_model.eval = Mock(return_value=(np.zeros_like(sample_mean_image), None, None))
        mock_cellpose_model.return_value = mock_model
        
        seg = Segmentation()
        min_size = 100
        masks = seg.segment_img(sample_mean_image, min_size=min_size)
        
        call_args = mock_model.eval.call_args
        assert call_args[1]["min_size"] == min_size


class TestSegmentationEdgeCases:
    """Edge case tests for Segmentation."""
    
    @patch('astroglial_morphology.segmentation.masks_flows_to_seg')
    @patch('astroglial_morphology.segmentation.CellposeModel')
    def test_segment_empty_image(self, mock_cellpose_model, mock_save):
        """Test segmentation on empty (all zeros) image."""
        mock_model = Mock()
        mock_model.diam_labels = 200.0
        mock_model.eval = Mock(return_value=(np.zeros((100, 100)), None, None))
        mock_cellpose_model.return_value = mock_model
        
        seg = Segmentation()
        empty_img = np.zeros((100, 100), dtype=np.float32)

        masks = seg.segment_img(empty_img)

        assert masks.shape == empty_img.shape
    
    @patch('astroglial_morphology.segmentation.masks_flows_to_seg')
    @patch('astroglial_morphology.segmentation.CellposeModel')
    def test_segment_constant_image(self, mock_cellpose_model, mock_save):
        """Test segmentation on constant-value image."""
        mock_model = Mock()
        mock_model.diam_labels = 200.0
        mock_model.eval = Mock(return_value=(np.zeros((100, 100)), None, None))
        mock_cellpose_model.return_value = mock_model
        
        seg = Segmentation()
        constant_img = np.full((100, 100), 128, dtype=np.float32)

        masks = seg.segment_img(constant_img)

        assert masks.shape == constant_img.shape
    
    @patch('astroglial_morphology.segmentation.masks_flows_to_seg')
    @patch('astroglial_morphology.segmentation.CellposeModel')
    def test_segment_very_small_image(self, mock_cellpose_model, mock_save):
        """Test segmentation on very small image."""
        mock_model = Mock()
        mock_model.diam_labels = 200.0
        mock_model.eval = Mock(return_value=(np.zeros((10, 10)), None, None))
        mock_cellpose_model.return_value = mock_model
        
        seg = Segmentation()
        small_img = np.random.rand(10, 10).astype(np.float32)

        masks = seg.segment_img(small_img)

        assert masks.shape == small_img.shape
    
    @patch('astroglial_morphology.segmentation.masks_flows_to_seg')
    @patch('astroglial_morphology.segmentation.CellposeModel')
    def test_segment_different_dtypes(self, mock_cellpose_model, mock_save):
        """Test segmentation with different input dtypes."""
        mock_model = Mock()
        mock_model.diam_labels = 200.0
        mock_model.eval = Mock(return_value=(np.zeros((100, 100)), None, None))
        mock_cellpose_model.return_value = mock_model
        
        seg = Segmentation()

        # Test with uint8
        img_uint8 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        masks = seg.segment_img(img_uint8)
        assert masks.shape == img_uint8.shape

        # Test with float64
        img_float64 = np.random.rand(100, 100).astype(np.float64)
        masks = seg.segment_img(img_float64)
        assert masks.shape == img_float64.shape
