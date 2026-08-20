"""Registration page: configure and launch Suite2p, then review QC."""

from __future__ import annotations

import streamlit as st

from astroglial_morphology.gui.form_widgets import render_specs
from astroglial_morphology.gui.job_panel import render_job_panel
from astroglial_morphology.gui.services.jobs import run_pipeline_subprocess
from astroglial_morphology.gui.services.parameters import (
    PARAMETER_CATALOG,
    build_hydra_overrides,
    default_registration_params,
    diff_against_defaults,
)
from astroglial_morphology.gui.services.qc_charts import show_registration_qc

st.title("Registration")
st.caption(
    "Suite2p motion correction with parameter overrides and QC preview. "
    "For the pause-and-correct workflow, use Full pipeline instead."
)

render_job_panel()

status = st.session_state.get("status")
if status is None or not status.is_valid_input:
    st.info("Select a valid data folder in the sidebar.")
    st.stop()

if status.input_mode == "suite2p":
    st.warning(
        "This folder is a direct Suite2p input. Registration cannot be re-run "
        "without the raw LIF/TIFF files."
    )

st.subheader("Suite2p parameters")

specs = PARAMETER_CATALOG["registration"]
common_specs = [spec for spec in specs if not spec.advanced]
advanced_specs = [spec for spec in specs if spec.advanced]

with st.form("registration-form", clear_on_submit=False):
    defaults = default_registration_params()
    values = render_specs(st, common_specs, defaults, "reg::")

    with st.expander("Advanced options"):
        values.update(render_specs(st, advanced_specs, defaults, "reg::"))

    submitted = st.form_submit_button(
        "Run registration", type="primary", icon=":material/play_arrow:"
    )

overrides_preview = build_hydra_overrides(
    registration_values=values,
    segmentation_values={},
    data_path=str(status.pipeline_data_path),
    alignment_only=True,
    skip_registration=False,
    correspondence_enabled=False,
)

diffs = diff_against_defaults("registration", values)
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
    elif status.input_mode == "suite2p":
        st.error("Cannot run registration on direct Suite2p input.")
    else:
        job = run_pipeline_subprocess(
            overrides_preview,
            data_path=status.data_path,
            label="registration",
        )
        st.session_state.active_job = job
        st.rerun()

st.divider()
st.subheader("Registration QC")
show_registration_qc(status.plane_dir)
