"""Streamlit entry point for the astroglial morphology GUI."""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit executes this file as a script, not as a package module.
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import streamlit as st

from astroglial_morphology.gui.job_panel import render_sidebar_job_chip
from astroglial_morphology.gui.services.experiment import describe_experiment


def _init_session_state() -> None:
    st.session_state.setdefault("data_path", "")
    st.session_state.setdefault("status", None)
    st.session_state.setdefault("active_job", None)
    st.session_state.setdefault("active_seg_file", None)
    st.session_state.setdefault("mask_editor_dirty", False)


def _sync_status() -> None:
    """Refresh the experiment status when the data path changes."""

    path = (st.session_state.get("data_path") or "").strip()
    current = st.session_state.get("status")
    if not path:
        st.session_state.status = None
        return
    if current is None or str(current.data_path) != path:
        st.session_state.status = describe_experiment(path)


def _sidebar() -> None:
    st.sidebar.header("Experiment")

    # A form keeps keystrokes from rerunning the whole app (and reloading ops.npy).
    with st.sidebar.form("load-experiment"):
        st.text_input(
            "Data folder",
            value=st.session_state.get("data_path", ""),
            key="data_path_input",
            placeholder=r"C:\path\to\experiment",
            help=(
                "Directory containing a LIF/TIFF acquisition, a folder with a "
                "'suite2p' sub-directory, or a Suite2p 'plane0' folder."
            ),
        )
        loaded = st.form_submit_button("Load folder", type="primary")
    if loaded:
        st.session_state.data_path = (st.session_state.data_path_input or "").strip()
        st.session_state.status = None
    if st.sidebar.button("Reload", width="stretch"):
        st.session_state.status = None
    _sync_status()

    status = st.session_state.status
    if status is None:
        st.sidebar.info("Enter a folder to get started.")
        return

    if status.errors:
        for err in status.errors:
            st.sidebar.error(err)

    st.sidebar.markdown("### Status")
    st.sidebar.write(
        {
            "Input mode": status.input_mode or "unknown",
            "Registration": "yes" if status.has_registration else "no",
            "Projections": len(status.projections),
            "Segmentation files": len(status.segmentation_files),
            "Metadata": "yes" if status.has_metadata else "no",
        }
    )
    if status.plane_dir is not None:
        st.sidebar.caption(f"plane0: {status.plane_dir}")
    render_sidebar_job_chip()


def _build_navigation() -> None:
    pages_dir = Path(__file__).parent / "app_pages"

    home = st.Page(str(pages_dir / "home.py"), title="Home", icon=":material/home:")
    pipeline = st.Page(
        str(pages_dir / "pipeline.py"),
        title="Full pipeline",
        icon=":material/account_tree:",
    )
    registration = st.Page(
        str(pages_dir / "registration.py"),
        title="Registration",
        icon=":material/rotate_right:",
    )
    segmentation = st.Page(
        str(pages_dir / "segmentation.py"),
        title="Segmentation",
        icon=":material/blur_on:",
    )
    correction = st.Page(
        str(pages_dir / "mask_correction.py"),
        title="Mask correction",
        icon=":material/brush:",
    )
    inspect = st.Page(
        str(pages_dir / "inspect.py"),
        title="Inspect",
        icon=":material/visibility:",
    )
    metadata = st.Page(
        str(pages_dir / "metadata.py"),
        title="Metadata",
        icon=":material/description:",
    )

    nav = st.navigation(
        {
            "Overview": [home, inspect, metadata],
            "Processing": [pipeline, registration, segmentation, correction],
        }
    )
    nav.run()


def main() -> None:
    st.set_page_config(
        page_title="Astroglial morphology",
        page_icon=":material/biotech:",
        layout="wide",
    )
    _init_session_state()
    _sidebar()
    _build_navigation()


main()
