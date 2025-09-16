from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Mapping, MutableMapping, Optional

from suite2p.default_ops import default_ops
from suite2p.run_s2p import run_s2p

from .logging_config import get_logger
import logging

logger = get_logger(__name__)
suite2p_logger = get_logger("Suite2p")


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
