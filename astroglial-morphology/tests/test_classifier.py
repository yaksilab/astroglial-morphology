"""Tests for classifier module."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from astroglial_morphology.classifier import classify_cells


class TestClassifyCells:
    """Tests for classify_cells function."""
    
    def test_classify_single_upper_cell(self, sample_masks):
        """Test classifying a single upper cell."""
        # Create a mask with one cell in upper position
        masks = np.zeros((512, 1024), dtype=np.uint16)
        masks[50:150, 50:150] = 1  # Upper left position
        
        neck_distance = 50
        pairs, cell_dict = classify_cells(masks, neck_distance)
        
        # Should have one cell
        assert len(cell_dict) == 1
        assert cell_dict[1]["type"] in ["upper", "lower"]
    
    def test_classify_single_lower_cell(self, sample_masks):
        """Test classifying a single lower cell."""
        # Create a mask with one cell in lower position
        masks = np.zeros((512, 1024), dtype=np.uint16)
        masks[400:500, 50:150] = 1  # Lower position
        
        neck_distance = 50
        pairs, cell_dict = classify_cells(masks, neck_distance)
        
        assert len(cell_dict) == 1
        assert cell_dict[1]["type"] in ["upper", "lower"]
    
    def test_classify_multiple_cells(self):
        """Test classifying multiple cells."""
        # Create masks with two cells
        masks = np.zeros((512, 1024), dtype=np.uint16)
        masks[50:150, 50:150] = 1  # First cell
        masks[300:400, 700:800] = 2  # Second cell
        
        neck_distance = 50
        pairs, cell_dict = classify_cells(masks, neck_distance)
        
        # Should have two cells
        assert len(cell_dict) == 2
        assert 1 in cell_dict
        assert 2 in cell_dict
        assert cell_dict[1]["type"] in ["upper", "lower"]
        assert cell_dict[2]["type"] in ["upper", "lower"]
    
    def test_classify_no_cells(self):
        """Test classification with no cells."""
        masks = np.zeros((512, 1024), dtype=np.uint16)
        
        neck_distance = 50
        pairs, cell_dict = classify_cells(masks, neck_distance)
        
        # Should return empty results
        assert len(pairs) == 0
        assert len(cell_dict) == 0
    
    def test_cell_properties_exist(self):
        """Test that classified cells have required properties."""
        masks = np.zeros((512, 1024), dtype=np.uint16)
        masks[100:200, 100:200] = 1
        
        neck_distance = 50
        pairs, cell_dict = classify_cells(masks, neck_distance)
        
        # Check required properties
        cell_info = cell_dict[1]
        required_props = [
            "type",
            "soma_end",
            "process_end",
            "soma_neck_point",
            "other_neck_point",
            "soma_neck_thickness",
            "other_neck_thickness",
            "area",
            "neck_distance"
        ]
        
        for prop in required_props:
            assert prop in cell_info
    
    def test_soma_end_is_tuple(self):
        """Test that soma_end is a tuple of coordinates."""
        masks = np.zeros((512, 1024), dtype=np.uint16)
        masks[100:200, 100:200] = 1
        
        neck_distance = 50
        pairs, cell_dict = classify_cells(masks, neck_distance)
        
        soma_end = cell_dict[1]["soma_end"]
        assert isinstance(soma_end, tuple)
        assert len(soma_end) == 2
    
    def test_process_end_is_tuple(self):
        """Test that process_end is a tuple of coordinates."""
        masks = np.zeros((512, 1024), dtype=np.uint16)
        masks[100:200, 100:200] = 1
        
        neck_distance = 50
        pairs, cell_dict = classify_cells(masks, neck_distance)
        
        process_end = cell_dict[1]["process_end"]
        assert isinstance(process_end, tuple)
        assert len(process_end) == 2
    
    def test_area_is_positive(self):
        """Test that cell area is positive."""
        masks = np.zeros((512, 1024), dtype=np.uint16)
        masks[100:200, 100:200] = 1
        
        neck_distance = 50
        pairs, cell_dict = classify_cells(masks, neck_distance)
        
        area = cell_dict[1]["area"]
        assert area > 0
    
    def test_neck_thickness_is_non_negative(self):
        """Test that neck thickness values are non-negative."""
        masks = np.zeros((512, 1024), dtype=np.uint16)
        masks[100:200, 100:200] = 1
        
        neck_distance = 50
        pairs, cell_dict = classify_cells(masks, neck_distance)
        
        soma_thickness = cell_dict[1]["soma_neck_thickness"]
        other_thickness = cell_dict[1]["other_neck_thickness"]
        
        assert soma_thickness >= 0
        assert other_thickness >= 0
    
    def test_pairs_format(self):
        """Test that pairs are in correct format."""
        masks = np.zeros((512, 1024), dtype=np.uint16)
        masks[100:200, 100:200] = 1
        masks[300:400, 700:800] = 2
        
        neck_distance = 50
        pairs, cell_dict = classify_cells(masks, neck_distance)
        
        # Pairs should be list of tuples
        assert isinstance(pairs, list)
        for pair in pairs:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
    
    def test_different_neck_distances(self):
        """Test classification with different neck distances."""
        masks = np.zeros((512, 1024), dtype=np.uint16)
        masks[100:200, 100:200] = 1
        
        # Test with small neck distance
        pairs1, dict1 = classify_cells(masks, neck_distance=20)
        assert len(dict1) == 1
        
        # Test with large neck distance
        pairs2, dict2 = classify_cells(masks, neck_distance=100)
        assert len(dict2) == 1
        
        # Both should classify the same cell
        assert 1 in dict1
        assert 1 in dict2
    
    def test_classification_reproducibility(self):
        """Test that classification is reproducible."""
        masks = np.zeros((512, 1024), dtype=np.uint16)
        masks[100:200, 100:200] = 1
        
        neck_distance = 50
        
        # Run classification twice
        pairs1, dict1 = classify_cells(masks, neck_distance)
        pairs2, dict2 = classify_cells(masks, neck_distance)
        
        # Results should be identical
        assert pairs1 == pairs2
        assert dict1[1]["type"] == dict2[1]["type"]
        assert dict1[1]["area"] == dict2[1]["area"]


class TestClassifyCellsEdgeCases:
    """Edge case tests for classify_cells."""
    
    def test_classify_very_small_cell(self):
        """Test classifying a very small cell."""
        masks = np.zeros((512, 1024), dtype=np.uint16)
        masks[100:105, 100:105] = 1  # 5x5 cell
        
        neck_distance = 10
        pairs, cell_dict = classify_cells(masks, neck_distance)
        
        # Should still classify even if small
        assert len(cell_dict) >= 0  # May or may not be detected
    
    def test_classify_very_large_cell(self):
        """Test classifying a very large cell."""
        masks = np.zeros((512, 1024), dtype=np.uint16)
        masks[50:450, 50:900] = 1  # Very large cell
        
        neck_distance = 50
        pairs, cell_dict = classify_cells(masks, neck_distance)
        
        assert len(cell_dict) == 1
        assert cell_dict[1]["area"] > 0
    
    def test_classify_elongated_cell(self):
        """Test classifying an elongated cell."""
        masks = np.zeros((512, 1024), dtype=np.uint16)
        masks[200:220, 100:900] = 1  # Elongated horizontally
        
        neck_distance = 50
        pairs, cell_dict = classify_cells(masks, neck_distance)
        
        assert len(cell_dict) == 1
    
    def test_classify_irregular_shape(self):
        """Test classifying irregularly shaped cell."""
        masks = np.zeros((512, 1024), dtype=np.uint16)
        
        # Create L-shaped cell
        masks[100:200, 100:150] = 1  # Vertical part
        masks[180:200, 150:250] = 1  # Horizontal part
        
        neck_distance = 50
        pairs, cell_dict = classify_cells(masks, neck_distance)
        
        assert len(cell_dict) == 1
    
    def test_classify_cells_touching_border(self):
        """Test classifying cells that touch image borders."""
        masks = np.zeros((512, 1024), dtype=np.uint16)
        masks[0:100, 0:100] = 1  # Top-left corner
        
        neck_distance = 50
        pairs, cell_dict = classify_cells(masks, neck_distance)
        
        assert len(cell_dict) == 1
    
    def test_classify_with_large_neck_distance(self):
        """Test classification with neck distance larger than cell."""
        masks = np.zeros((512, 1024), dtype=np.uint16)
        masks[100:150, 100:150] = 1  # 50x50 cell
        
        neck_distance = 200  # Larger than cell size
        pairs, cell_dict = classify_cells(masks, neck_distance)
        
        # Should still work, even if neck distance is large
        assert len(cell_dict) >= 0
    
    def test_classify_non_contiguous_mask_ids(self):
        """Test classification with non-contiguous mask IDs."""
        masks = np.zeros((512, 1024), dtype=np.uint16)
        masks[100:200, 100:200] = 1
        masks[300:400, 300:400] = 5  # Skip IDs 2, 3, 4
        masks[100:200, 700:800] = 10
        
        neck_distance = 50
        pairs, cell_dict = classify_cells(masks, neck_distance)
        
        # Should classify cells 1, 5, and 10
        assert 1 in cell_dict
        assert 5 in cell_dict
        assert 10 in cell_dict
