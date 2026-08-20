"""Run the pipeline in a child process so Streamlit stays responsive.

Suite2p and Cellpose can each take minutes and print heavily to stdout.
Jobs are spawned with ``python -m astroglial_morphology``; a reader thread
copies the child's stdout/stderr into a log file and an in-memory ring buffer
that the GUI tails without waiting for the process to exit.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Deque, Iterable, List, Optional, TextIO


class JobStatus(str, Enum):
    """Lifecycle states for a spawned pipeline job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


def _append_line(fp: TextIO, buffer: Deque[str], lock: threading.Lock, line: str) -> None:
    if not line.endswith("\n"):
        line = f"{line}\n"
    with lock:
        buffer.append(line)
        fp.write(line)
        fp.flush()


def _pump_output(
    stream: TextIO,
    log_fp: TextIO,
    buffer: Deque[str],
    lock: threading.Lock,
) -> None:
    """Copy child output into the log file, splitting carriage-return updates."""

    leftover = ""
    try:
        while True:
            chunk = stream.read(256)
            if not chunk:
                break
            leftover += chunk.replace("\r\n", "\n").replace("\r", "\n")
            while "\n" in leftover:
                line, leftover = leftover.split("\n", 1)
                _append_line(log_fp, buffer, lock, line)
        if leftover.strip():
            _append_line(log_fp, buffer, lock, leftover)
    finally:
        try:
            stream.close()
        except OSError:
            pass
        with lock:
            try:
                log_fp.flush()
                log_fp.close()
            except (OSError, ValueError):
                pass


@dataclass
class JobHandle:
    """Handle to a running pipeline subprocess."""

    job_id: str
    command: List[str]
    log_path: Path
    started_at: datetime
    process: Optional[subprocess.Popen] = None
    status: JobStatus = JobStatus.PENDING
    return_code: Optional[int] = None
    stopped_at: Optional[datetime] = None
    label: str = ""
    results_consumed: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _lines: Deque[str] = field(default_factory=lambda: deque(maxlen=4000), repr=False)
    _pump: Optional[threading.Thread] = None
    _log_fp: Optional[TextIO] = None

    def is_running(self) -> bool:
        return self.status in {JobStatus.PENDING, JobStatus.RUNNING}

    def elapsed_label(self) -> str:
        end = self.stopped_at or datetime.now()
        seconds = max(0, int((end - self.started_at).total_seconds()))
        minutes, secs = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        return f"{minutes}m {secs:02d}s"

    def poll(self) -> JobStatus:
        """Update :attr:`status` based on the child process state."""

        process = self.process
        if process is None:
            return self.status
        code = process.poll()
        if code is None:
            with self._lock:
                if self.status == JobStatus.PENDING:
                    self.status = JobStatus.RUNNING
                return self.status
        if self._pump is not None:
            self._pump.join(timeout=0.2)
        with self._lock:
            self.return_code = code
            if self.status == JobStatus.CANCELLED:
                self.stopped_at = self.stopped_at or datetime.now()
                return self.status
            self.status = JobStatus.SUCCESS if code == 0 else JobStatus.FAILED
            self.stopped_at = datetime.now()
            return self.status

    def cancel(self, timeout: float = 5.0) -> None:
        """Terminate the child process if it is still running."""

        with self._lock:
            if self.process is None or self.process.poll() is not None:
                return
            self.status = JobStatus.CANCELLED
        try:
            self.process.terminate()
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
        finally:
            self.stopped_at = datetime.now()

    def latest_line(self) -> str:
        """Return the last non-empty log line, stripped."""

        text = self.tail_log(max_lines=40)
        for line in reversed(text.splitlines()):
            stripped = line.strip()
            if stripped:
                return stripped
        return "Waiting for output…"

    def tail_log(self, max_lines: int = 200) -> str:
        """Return the last *max_lines* lines from memory, falling back to disk."""

        with self._lock:
            if self._lines:
                lines = list(self._lines)
                return "".join(lines[-max_lines:])
        if not self.log_path.is_file():
            return ""
        try:
            with self.log_path.open("r", encoding="utf-8", errors="replace") as fp:
                lines = fp.readlines()
        except OSError:
            return ""
        return "".join(lines[-max_lines:])


def _resolve_log_dir(data_path: Path) -> Path:
    """Return a directory for job logs, inside the experiment when possible."""

    if data_path.is_dir():
        log_dir = data_path / "gui_logs"
    else:
        log_dir = Path.cwd() / "gui_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _build_command(executable: str, overrides: Iterable[str]) -> List[str]:
    """Build a pipeline command without assigning the capture log to the child."""

    return [executable, "-m", "astroglial_morphology", *list(overrides)]


def run_pipeline_subprocess(
    overrides: Iterable[str],
    *,
    data_path: str | Path,
    label: str = "pipeline",
    python_executable: Optional[str] = None,
) -> JobHandle:
    """Spawn ``python -m astroglial_morphology`` with the given Hydra *overrides*."""

    data = Path(data_path)
    log_dir = _resolve_log_dir(data)
    job_id = uuid.uuid4().hex[:12]
    log_path = log_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}-{job_id}.log"
    executable = python_executable or sys.executable

    command = _build_command(executable, overrides)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    log_fp = log_path.open("w", encoding="utf-8", buffering=1)
    handle = JobHandle(
        job_id=job_id,
        command=command,
        log_path=log_path,
        started_at=datetime.now(),
        status=JobStatus.RUNNING,
        label=label,
        _log_fp=log_fp,
    )
    _append_line(log_fp, handle._lines, handle._lock, f"$ {' '.join(command)}")
    _append_line(
        log_fp,
        handle._lines,
        handle._lock,
        f"Started {label} at {handle.started_at.isoformat(timespec='seconds')}",
    )

    popen_kwargs: dict = {
        "args": command,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "env": env,
        "cwd": str(data if data.is_dir() else Path.cwd()),
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    process = subprocess.Popen(**popen_kwargs)
    handle.process = process
    assert process.stdout is not None
    pump = threading.Thread(
        target=_pump_output,
        args=(process.stdout, log_fp, handle._lines, handle._lock),
        name=f"job-log-{job_id}",
        daemon=True,
    )
    pump.start()
    handle._pump = pump
    return handle
