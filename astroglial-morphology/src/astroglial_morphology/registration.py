from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping, MutableMapping, Optional

from suite2p.default_ops import default_ops
from suite2p.run_s2p import run_s2p

from .logging_config import get_logger
import logging

logger = get_logger(__name__)
suite2p_logger = get_logger("Suite2p")


def get_suite2p_output_dir(
    data_path: str, options: Optional[Mapping[str, object]] = None
) -> Path:
    """
    Determine the Suite2p output directory based on data_path and options.

    Suite2p's output location is determined by:
    - save_path0: Root save directory (defaults to data_path if empty/not set)
    - save_folder: List of subdirectories under save_path0 (optional)
    - Final path: save_path0/save_folder[0]/suite2p (or save_path0/suite2p if no save_folder)

    Args:
        data_path: Path to the data directory
        options: Suite2p options dictionary that may contain save_path0/save_folder

    Returns:
        Path object representing the suite2p output directory
    """
    if options is None:
        options = {}

    # Get save_path0, defaulting to data_path if empty or not set
    save_path0 = options.get("save_path0", "")
    if not save_path0 or save_path0 == "":
        base_path = Path(data_path)
    else:
        base_path = Path(save_path0)

    # Add save_folder subdirectory if specified
    save_folder = options.get("save_folder", [])
    if save_folder and len(save_folder) > 0:
        base_path = base_path / save_folder[0]

    # Suite2p always creates a 'suite2p' subdirectory
    return base_path / "suite2p"


class StreamToLogger:
    """Redirect stdout/stderr to a logger."""

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            if line.strip():  # Only log non-empty lines
                self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


@contextmanager
def capture_suite2p_output():
    """Context manager to capture Suite2p's stdout/stderr output."""
    import logging

    capture_logger = get_logger("s2p")

    # Store original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Redirect stdout and stderr to loggers
    sys.stdout = StreamToLogger(capture_logger, logging.INFO)
    sys.stderr = StreamToLogger(capture_logger, logging.ERROR)

    try:
        yield
    finally:
        # Restore original streams
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def do_registration(
    data_path: str,
    user_options: Optional[Mapping[str, object]] = None,
) -> None:
    """Run Suite2p registration for the supplied dataset."""

    logger.info("Starting registration for %s", data_path)

    options: MutableMapping[str, object] = default_ops()

    if user_options is not None:
        logger.debug("Applying user overrides: %s", user_options)
        options.update(user_options)

    db = {
        "data_path": [data_path],
        "subfolders": [],
        "fast_disk": [],
    }

    with capture_suite2p_output():
        run_s2p(options, db)
    logger.info("Registration completed successfully")

    # Create registration completion flag in the actual Suite2p output directory
    suite2p_dir = get_suite2p_output_dir(data_path, options)
    flag_path = suite2p_dir / ".registration_complete"
    flag_path.touch()
    logger.debug(f"Created registration flag: {flag_path}")


def check_registration_complete(
    data_path: str, user_options: Optional[Mapping[str, object]] = None
) -> bool:
    """
    Check if registration has been completed for the given data path.

    Args:
        data_path: Path to the data directory
        user_options: Suite2p options that may affect output location

    Returns:
        True if registration is complete, False otherwise
    """
    suite2p_dir = get_suite2p_output_dir(data_path, user_options)
    flag_path = suite2p_dir / ".registration_complete"
    return flag_path.exists()
