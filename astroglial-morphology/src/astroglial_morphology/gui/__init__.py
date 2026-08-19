"""Streamlit GUI for the astroglial morphology pipeline."""

from __future__ import annotations

__all__ = ["main"]


def main() -> None:
    """Launch the Streamlit app via ``streamlit run``."""

    import subprocess
    import sys
    from pathlib import Path

    entry = Path(__file__).with_name("streamlit_app.py")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(entry)],
        check=False,
    )
