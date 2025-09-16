"""Centralized logging configuration for the astroglial_morphology package."""
from __future__ import annotations

import logging
import os
from logging.config import dictConfig
from pathlib import Path
from typing import Dict, Optional, Union

_LOGGING_INITIALIZED = False

DEFAULT_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL = "INFO"

LevelType = Union[int, str]


def _coerce_level(level: LevelType) -> int:
    """Translate a logging level expressed as an int or name into the numeric value."""
    if isinstance(level, int):
        return level

    normalized = level.strip().upper()
    if normalized.isdigit():
        return int(normalized)

    if normalized in logging._nameToLevel:  # type: ignore[attr-defined]
        return logging._nameToLevel[normalized]  # type: ignore[attr-defined]

    raise ValueError(f"Unsupported log level: {level!r}")


def _resolve_log_path(log_file: Optional[str]) -> Optional[Path]:
    env_log_file = os.getenv("ASTROGLIAL_LOGFILE")
    env_log_dir = os.getenv("ASTROGLIAL_LOGDIR")

    target = log_file or env_log_file
    if not target:
        return None

    path = Path(target)
    if not path.is_absolute() and env_log_dir:
        path = Path(env_log_dir) / path

    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _build_handlers(log_path: Optional[Path], include_console: bool) -> Dict[str, Dict[str, object]]:
    handlers: Dict[str, Dict[str, object]] = {}

    if include_console:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stderr",
        }

    if log_path is not None:
        handlers["file"] = {
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": str(log_path),
            "encoding": "utf-8",
        }

    return handlers


def setup_logging(
    *,
    level: Optional[LevelType] = None,
    log_file: Optional[str] = None,
    console: bool = True,
    force: bool = False,
) -> None:
    """Configure the root logger for the package.

    Parameters
    ----------
    level:
        Optional log level to use. Accepts numeric values or case-insensitive names.
        Defaults to the ``ASTROGLIAL_LOGLEVEL`` environment variable, falling back to INFO.
    log_file:
        Optional file path for log output. Relative paths can be combined with the
        ``ASTROGLIAL_LOGDIR`` environment variable. The directory is created when needed.
    console:
        Controls whether a console stream handler is added.
    force:
        When ``True`` existing logging configuration is replaced.
    """

    global _LOGGING_INITIALIZED

    if _LOGGING_INITIALIZED and not force:
        return

    env_level = os.getenv("ASTROGLIAL_LOGLEVEL", DEFAULT_LEVEL)
    resolved_level = _coerce_level(level or env_level)

    log_path = _resolve_log_path(log_file)
    handlers = _build_handlers(log_path, console)

    if not handlers:
        raise ValueError("At least one logging handler must be enabled")

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": DEFAULT_FORMAT,
                    "datefmt": DEFAULT_DATEFMT,
                }
            },
            "handlers": handlers,
            "root": {
                "handlers": list(handlers.keys()),
                "level": resolved_level,
            },
        }
    )

    _LOGGING_INITIALIZED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger configured for the package, initialising defaults on first use."""
    if not _LOGGING_INITIALIZED:
        setup_logging()

    return logging.getLogger(name)
