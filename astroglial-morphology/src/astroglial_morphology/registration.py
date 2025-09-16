from __future__ import annotations

from typing import Mapping, MutableMapping, Optional

from suite2p.default_ops import default_ops
from suite2p.run_s2p import run_s2p

from .logging_config import get_logger

logger = get_logger(__name__)


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

    logger.debug("Database configuration: %s", db)
    run_s2p(options, db)
    logger.info("Registration completed successfully")
