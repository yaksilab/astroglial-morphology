"""Centralized logging configuration for the astroglial_morphology package."""
from __future__ import annotations

import logging
import os
from logging.config import dictConfig
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

_LOGGING_INITIALIZED = False
_CELLPOSE_LOGGER_PATCHED = False
_CELLPOSE_ORIGINAL_LOGGER_SETUP: Optional[
    Callable[[str, str, Optional[str]], Tuple[logging.Logger, Path]]
] = None

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


def _ensure_file_handler(path: Path) -> None:
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            filename = getattr(handler, "baseFilename", None)
            if filename and Path(filename) == path:
                return

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATEFMT))
    root_logger.addHandler(handler)


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


def configure_cellpose_logging(
    *,
    cp_path: str = ".cellpose",
    logfile_name: str = "run.log",
    stdout_file_replacement: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Ensure Cellpose uses the shared logging configuration.

    Returns the resolved path to the primary Cellpose log file.
    """

    try:
        from cellpose import io as cellpose_io
        from cellpose import version_str as cellpose_version
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Cellpose is not installed; cannot configure logging") from exc

    global _CELLPOSE_LOGGER_PATCHED, _CELLPOSE_ORIGINAL_LOGGER_SETUP

    if _CELLPOSE_LOGGER_PATCHED and not force:
        _, log_path = cellpose_io.logger_setup(
            cp_path=cp_path,
            logfile_name=logfile_name,
            stdout_file_replacement=stdout_file_replacement,
        )
        return log_path

    _CELLPOSE_ORIGINAL_LOGGER_SETUP = cellpose_io.logger_setup

    def _patched_logger_setup(
        cp_path: str = ".cellpose",
        logfile_name: str = "run.log",
        stdout_file_replacement: Optional[str] = None,
    ) -> Tuple[logging.Logger, Path]:
        log_dir = Path.home() / cp_path
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / logfile_name

        try:
            log_file.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logging.getLogger(__name__).warning(
                "Unable to reset existing Cellpose log file %s", log_file, exc_info=True
            )

        if not _LOGGING_INITIALIZED:
            setup_logging(log_file=str(log_file))
        else:
            _ensure_file_handler(log_file)

        if stdout_file_replacement is not None:
            replacement_path = Path(stdout_file_replacement)
            replacement_path.parent.mkdir(parents=True, exist_ok=True)
            _ensure_file_handler(replacement_path)

        logger = get_logger("cellpose")
        logger.info("WRITING LOG OUTPUT TO %s", log_file)
        logger.info(cellpose_version)
        return logger, log_file

    cellpose_io.logger_setup = _patched_logger_setup
    _CELLPOSE_LOGGER_PATCHED = True

    _, log_path = _patched_logger_setup(
        cp_path=cp_path,
        logfile_name=logfile_name,
        stdout_file_replacement=stdout_file_replacement,
    )
    return log_path


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger configured for the package, initialising defaults on first use."""
    if not _LOGGING_INITIALIZED:
        setup_logging()

    return logging.getLogger(name)
