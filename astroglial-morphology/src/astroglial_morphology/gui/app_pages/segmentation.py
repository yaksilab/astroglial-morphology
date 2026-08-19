"""Segmentation page: run Cellpose against existing projections."""

from __future__ import annotations

import numpy as np
import streamlit as st

from astroglial_morphology.gui.form_widgets import render_specs
from astroglial_morphology.gui.job_panel import render_job_panel
from astroglial_morphology.gui.services.jobs import run_pipeline_subprocess
from astroglial_morphology.gui.services.parameters import (
    PARAMETER_CATALOG,
    build_hydra_overrides,
    default_registration_params,
    default_segmentation_params,
    diff_against_defaults,
)
from astroglial_morphology.gui.services.cache import (
    display_projections_for,
    mask_overlay_for,
)
from astroglial_morphology.gui.services.results import load_seg_masks

st.title("Segmentation")
st.caption(
    "Run Cellpose on the mean or max projection with parameter overrides. "
    "For the pause-and-correct workflow, use Full pipeline instead."
)

render_job_panel()

status = st.session_state.get("status")
if status is None or not status.is_valid_input:
    st.info("Select a valid data folder in the sidebar.")
    st.stop()

if not (status.has_registration or status.input_mode == "suite2p"):
    st.warning(
        "Segmentation requires a registered plane0 folder. Run registration first."
    )

specs = PARAMETER_CATALOG["segmentation"]

with st.form("segmentation-form"):
    defaults = default_segmentation_params()
    values = render_specs(st, specs, defaults, "seg::")
    submitted = st.form_submit_button(
        "Run segmentation", type="primary", icon=":material/play_arrow:"
    )

overrides_preview = build_hydra_overrides(
    registration_values={
        **default_registration_params(),
    },
    segmentation_values=values,
    data_path=str(status.data_path),
    alignment_only=False,
    skip_registration=True,
    correspondence_enabled=False,
)

diffs = diff_against_defaults("segmentation", values)
if diffs:
    st.markdown("**Non-default parameters:**")
    st.write(
        {
            key: f"default={info['default']} → used={info['used']}"
            for key, info in diffs.items()
        }
    )
else:
    st.caption("All parameters are at their package defaults.")

with st.expander("Command that will be run"):
    st.code("python -m astroglial_morphology " + " ".join(overrides_preview))

active_job = st.session_state.get("active_job")

if submitted:
    if active_job is not None and active_job.is_running():
        st.warning("A job is already running. Wait for it to finish or cancel it.")
    else:
        job = run_pipeline_subprocess(
            overrides_preview,
            data_path=status.data_path,
            label="segmentation",
        )
        st.session_state.active_job = job
        st.rerun()

st.divider()
st.subheader("Segmentation preview")

if not status.has_segmentation:
    st.info("No segmentation files yet. Run segmentation to see the overlay.")
else:
    seg_options = {sf.seg_path.name: sf for sf in status.segmentation_files}
    seg_name = st.selectbox(
        "Segmentation file", options=list(seg_options.keys()), index=0
    )
    seg_file = seg_options[seg_name]
    masks = load_seg_masks(seg_file.seg_path)
    labels = np.unique(masks)
    labels = labels[labels != 0]
    st.metric("ROIs", int(labels.size))

    if seg_file.image_path is not None and seg_file.image_path.is_file():
        cols = st.columns(2)
        projections = display_projections_for(status.plane_dir) if status.plane_dir else {}
        projection = projections.get(seg_file.image_path.stem)
        if projection is not None:
            cols[0].image(projection, caption="Projection", clamp=True, width="stretch")
        cols[1].image(
            mask_overlay_for(seg_file.seg_path, seg_file.image_path),
            caption="Overlay",
            clamp=True,
            width="stretch",
        )
    else:
        st.warning(f"Paired projection image not found next to {seg_file.seg_path.name}.")
