"""Tests for registration module."""

import pytest
from pathlib import Path
from astroglial_morphology.registration import get_suite2p_output_dir


class TestGetSuite2pOutputDir:
    """Tests for get_suite2p_output_dir function."""

    def test_default_no_options(self):
        """Test default behavior with no options."""
        data_path = "/data/experiment"
        result = get_suite2p_output_dir(data_path)
        assert result == Path("/data/experiment/suite2p")

    def test_empty_save_path0(self):
        """Test with empty save_path0 (should use data_path)."""
        data_path = "/data/experiment"
        options = {"save_path0": "", "save_folder": []}
        result = get_suite2p_output_dir(data_path, options)
        assert result == Path("/data/experiment/suite2p")

    def test_custom_save_path0(self):
        """Test with custom save_path0."""
        data_path = "/data/experiment"
        options = {"save_path0": "/custom/path", "save_folder": []}
        result = get_suite2p_output_dir(data_path, options)
        assert result == Path("/custom/path/suite2p")

    def test_save_folder_with_default_path(self):
        """Test with save_folder and default path."""
        data_path = "/data/experiment"
        options = {"save_path0": "", "save_folder": ["run1"]}
        result = get_suite2p_output_dir(data_path, options)
        assert result == Path("/data/experiment/run1/suite2p")

    def test_serialized_suite2p_save_folder_string(self):
        """Suite2p saves its resolved output directory as a scalar string."""
        data_path = "/data/experiment"
        options = {"save_path0": data_path, "save_folder": "suite2p"}
        result = get_suite2p_output_dir(data_path, options)
        assert result == Path("/data/experiment/suite2p")

    def test_save_folder_with_custom_path(self):
        """Test with save_folder and custom save_path0."""
        data_path = "/data/experiment"
        options = {"save_path0": "/custom/path", "save_folder": ["run1"]}
        result = get_suite2p_output_dir(data_path, options)
        assert result == Path("/custom/path/run1/suite2p")

    def test_empty_save_folder_list(self):
        """Test with empty save_folder list."""
        data_path = "/data/experiment"
        options = {"save_path0": "/custom/path", "save_folder": []}
        result = get_suite2p_output_dir(data_path, options)
        assert result == Path("/custom/path/suite2p")

    def test_multiple_save_folders_uses_first(self):
        """Test that only the first save_folder is used."""
        data_path = "/data/experiment"
        options = {"save_path0": "", "save_folder": ["run1", "run2"]}
        result = get_suite2p_output_dir(data_path, options)
        assert result == Path("/data/experiment/run1/suite2p")

    def test_none_options(self):
        """Test with None options (should behave like empty dict)."""
        data_path = "/data/experiment"
        result = get_suite2p_output_dir(data_path, None)
        assert result == Path("/data/experiment/suite2p")

    def test_options_without_save_keys(self):
        """Test with options that don't have save_path0 or save_folder."""
        data_path = "/data/experiment"
        options = {"nplanes": 1, "nchannels": 1}
        result = get_suite2p_output_dir(data_path, options)
        assert result == Path("/data/experiment/suite2p")

    def test_relative_path(self):
        """Test with relative path."""
        data_path = "data/experiment"
        result = get_suite2p_output_dir(data_path)
        assert result == Path("data/experiment/suite2p")

    def test_path_with_trailing_slash(self):
        """Test with path that has trailing slash."""
        data_path = "/data/experiment/"
        result = get_suite2p_output_dir(data_path)
        # Path normalization should handle this
        assert result == Path("/data/experiment/suite2p")
