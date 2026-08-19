"""Home page: describe the loaded experiment and its artifacts."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from astroglial_morphology.gui.services.experiment import describe_experiment

st.title("Astroglial morphology")
st.caption(
    "Run the full pipeline with a pause for mask correction, or use the individual processing pages."
)

status = st.session_state.get("status")
data_path = (st.session_state.get("data_path") or "").strip()

if not data_path:
    st.info("Enter a data folder in the sidebar to begin.")
    st.stop()

if st.button("Reload folder", icon=":material/refresh:"):
    st.session_state.status = describe_experiment(data_path)
    status = st.session_state.status

if status is None:
    st.warning("Could not read the folder. Check the path in the sidebar.")
    st.stop()

if status.errors:
    for err in status.errors:
        st.error(err)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Input")
    st.write({"path": str(status.data_path), "mode": status.input_mode or "unknown"})
    if status.input_file is not None:
        st.write(
            {
                "file": str(status.input_file.path),
                "format": status.input_file.format.value,
            }
        )
    if status.plane_dir is not None:
        st.write({"plane0": str(status.plane_dir)})

with col_right:
    st.subheader("Artifacts")
    st.write(
        {
            "Registration complete": bool(status.has_registration),
            "Projections": len(status.projections),
            "Segmentation files": len(status.segmentation_files),
            "Pipeline metadata": bool(status.has_metadata),
            "Correspondence outputs": len(status.correspondence_files),
        }
    )

st.divider()
st.subheader("Next steps")
if status.input_mode is None:
    st.warning(
        "No supported input detected. Provide a folder containing a LIF/TIFF or a "
        "Suite2p `plane0` directory."
    )
else:
    if not status.has_registration and status.input_mode != "suite2p":
        st.info("Start with the full pipeline to align the acquisition and segment cells.")
    elif not status.has_segmentation:
        st.info("Start with the full pipeline to run Cellpose, then correct masks.")
    elif not status.correspondence_files:
        st.info(
            "Correct masks if needed, then confirm on the full pipeline page to "
            "export correspondence."
        )
    else:
        st.info("Correspondence outputs are present. Inspect them or re-run a step.")
    st.page_link(
        str(Path(__file__).with_name("pipeline.py")),
        label="Open full pipeline",
        icon=":material/account_tree:",
    )
    if status.has_metadata:
        st.info("Head to **Metadata** to compare the run against pipeline defaults.")
