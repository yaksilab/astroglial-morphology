"""Inspect page: browse projections, masks, and QC without running anything."""

from __future__ import annotations

import numpy as np
import streamlit as st

from astroglial_morphology.gui.services.cache import (
    display_projections_for,
    mask_overlay_for,
    metadata_for,
)
from astroglial_morphology.gui.services.qc_charts import show_registration_qc
from astroglial_morphology.gui.services.results import load_seg_masks

st.title("Inspect")
st.caption("Browse an existing run without recomputing anything.")

status = st.session_state.get("status")
if status is None or status.plane_dir is None:
    st.info("Load a folder containing a Suite2p `plane0` directory.")
    st.stop()

plane_dir = status.plane_dir
view = st.segmented_control(
    "View",
    options=["Projections", "Segmentation", "Registration QC", "Files"],
    default="Projections",
    key="inspect-view",
    label_visibility="collapsed",
)
if view is None:
    view = "Projections"

if view == "Projections":
    projections = display_projections_for(plane_dir)
    if not projections:
        st.info("No projection PNGs found in this plane0.")
    else:
        cols = st.columns(min(3, len(projections)))
        for idx, (name, image) in enumerate(projections.items()):
            cols[idx % len(cols)].image(image, caption=name, clamp=True, width="stretch")

elif view == "Segmentation":
    if not status.segmentation_files:
        st.info("No segmentation files found.")
    else:
        seg_options = {sf.seg_path.name: sf for sf in status.segmentation_files}
        seg_name = st.selectbox(
            "Segmentation file", options=list(seg_options.keys()), key="inspect-seg"
        )
        seg_file = seg_options[seg_name]
        masks = load_seg_masks(seg_file.seg_path)
        labels = np.unique(masks)
        labels = labels[labels != 0]
        st.metric("ROIs", int(labels.size))
        if seg_file.image_path is not None and seg_file.image_path.is_file():
            st.image(
                mask_overlay_for(seg_file.seg_path, seg_file.image_path),
                caption=f"{seg_file.seg_path.stem} overlay",
                clamp=True,
                width="stretch",
            )
        else:
            st.warning(
                f"No projection image is paired with {seg_file.seg_path.name}."
            )

elif view == "Registration QC":
    show_registration_qc(plane_dir)

else:
    st.markdown("**plane0 contents**")
    files = sorted(plane_dir.iterdir())
    st.dataframe(
        [
            {
                "name": p.name,
                "size": p.stat().st_size if p.is_file() else None,
                "kind": "file" if p.is_file() else "dir",
            }
            for p in files
        ],
        width="stretch",
    )
    if status.correspondence_files:
        st.markdown("**Correspondence outputs**")
        st.write([str(p) for p in status.correspondence_files])
    metadata = metadata_for(plane_dir / "pipeline_metadata.json")
    if metadata is not None:
        st.markdown("**pipeline_metadata.json (raw)**")
        st.json(metadata, expanded=False)
