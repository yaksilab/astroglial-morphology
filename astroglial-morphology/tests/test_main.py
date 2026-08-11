"""Tests for command-line defaults."""

import sys
from unittest.mock import patch

from astroglial_morphology.__main__ import main


@patch("astroglial_morphology.__main__.get_logger")
@patch("astroglial_morphology.__main__.setup_logging")
@patch("astroglial_morphology.__main__.Pipeline")
def test_correspondence_export_is_opt_in(
    mock_pipeline_class,
    mock_setup_logging,
    mock_get_logger,
    tmp_path,
    monkeypatch,
):
    mock_pipeline_class.return_value.run.return_value = {
        "classification": None,
        "correspondence": None,
    }
    monkeypatch.setattr(sys, "argv", ["astroglial-morphology", str(tmp_path)])

    main()

    run_kwargs = mock_pipeline_class.return_value.run.call_args.kwargs
    assert run_kwargs["export_correspondence"] is False
    assert run_kwargs["trace_channels"] is None
