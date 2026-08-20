"""Full pipeline page: register, segment, correct, then finish correspondence."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from astroglial_morphology.gui.form_widgets import render_specs, values_from_session
from astroglial_morphology.gui.job_panel import render_job_panel
from astroglial_morphology.gui.services.jobs import run_pipeline_subprocess
from astroglial_morphology.gui.services.parameters import (
    PARAMETER_CATALOG,
    build_hydra_overrides,
    default_correspondence_params,
    default_registration_params,
    default_segmentation_params,
)

_MASK_CORRECTION_PAGE = str(Path(__file__).with_name("mask_correction.py"))

st.title("Full pipeline")
st.caption(
    "Run registration and automatic segmentation, correct masks in the browser, "
    "then continue to classification, subsegmentation, traces, and correspondence."
)

render_job_panel()

status = st.session_state.get("status")
if status is None or not status.is_valid_input:
    st.info("Select a valid data folder in the sidebar.")
    st.stop()

has_registration = bool(status.has_registration or status.input_mode == "suite2p")
has_segmentation = bool(status.has_segmentation)
has_correspondence = bool(status.correspondence_files)

st.markdown(
    f"1. Registration: **{'done' if has_registration else 'needed'}**  \n"
    f"2. Automatic segmentation: **{'done' if has_segmentation else 'needed'}**  \n"
    f"3. Mask correction: **{'ready' if has_segmentation else 'waiting'}**  \n"
    f"4. Correspondence export: **{'done' if has_correspondence else 'waiting'}**"
)

reg_defaults = default_registration_params()
seg_defaults = default_segmentation_params()
corr_defaults = default_correspondence_params()
reg_specs = PARAMETER_CATALOG["registration"]
seg_specs = PARAMETER_CATALOG["segmentation"]
corr_specs = PARAMETER_CATALOG["correspondence"]
reg_common = [spec for spec in reg_specs if not spec.advanced]
reg_advanced = [spec for spec in reg_specs if spec.advanced]

active_job = st.session_state.get("active_job")
busy = active_job is not None and active_job.is_running()


def _skip_registration(force: bool) -> bool:
    if status.input_mode == "suite2p":
        return True
    return bool(has_registration and not force)


st.subheader("1. Run through segmentation")
st.write(
    "Starts Suite2p (unless this folder is already registered), runs Cellpose, "
    "and stops before correspondence so you can correct masks."
)

with st.form("pipeline-to-segmentation"):
    reg_col, seg_col = st.columns(2, gap="large")
    with reg_col:
        with st.container(border=True):
            st.markdown("**Registration**")
            if status.input_mode == "suite2p":
                st.caption("Direct Suite2p input: registration will be skipped.")
            reg_values = render_specs(st, reg_common, reg_defaults, "reg::")
            with st.expander("Advanced registration"):
                reg_values.update(render_specs(st, reg_advanced, reg_defaults, "reg::"))
    with seg_col:
        with st.container(border=True):
            st.markdown("**Segmentation**")
            seg_values = render_specs(st, seg_specs, seg_defaults, "seg::")
    start_submitted = st.form_submit_button(
        "Run registration and segmentation",
        type="primary",
        icon=":material/play_arrow:",
        disabled=busy,
    )

start_reg = values_from_session(reg_specs, reg_defaults, "reg::")
start_overrides = build_hydra_overrides(
    start_reg,
    values_from_session(seg_specs, seg_defaults, "seg::"),
    data_path=str(status.pipeline_data_path),
    alignment_only=False,
    skip_registration=_skip_registration(bool(start_reg.get("force"))),
    correspondence_enabled=False,
)

with st.expander("Command for step 1"):
    st.code("python -m astroglial_morphology " + " ".join(start_overrides))

if start_submitted:
    if busy:
        st.warning("A job is already running. Wait for it to finish or cancel it.")
    else:
        overrides = build_hydra_overrides(
            reg_values,
            seg_values,
            data_path=str(status.pipeline_data_path),
            alignment_only=False,
            skip_registration=_skip_registration(bool(reg_values.get("force"))),
            correspondence_enabled=False,
        )
        st.session_state.active_job = run_pipeline_subprocess(
            overrides,
            data_path=status.data_path,
            label="pipeline_segmentation",
        )
        st.rerun()

st.divider()
st.subheader("2. Correct masks")
if not has_segmentation:
    st.info("Segmentation must finish before you can edit masks.")
else:
    st.write(
        "Edit ROIs in the mask editor and save them into the seg file. "
        "Then continue here, or use the same confirm button on that page."
    )
    st.page_link(
        _MASK_CORRECTION_PAGE,
        label="Open mask correction",
        icon=":material/brush:",
    )

st.divider()
st.subheader("3. Continue after correction")
st.write(
    "Loads the saved masks (no Cellpose rerun) and runs classification, "
    "subsegmentation, trace extraction, and correspondence export."
)

segmentation_options = {
    seg_file.seg_path.name: seg_file for seg_file in status.segmentation_files
}

with st.form("pipeline-after-correction"):
    if segmentation_options:
        selected_seg_name = st.selectbox(
            "Segmentation file",
            options=list(segmentation_options),
            help="The exact corrected mask file that the pipeline will resume from.",
        )
    else:
        selected_seg_name = st.selectbox(
            "Segmentation file",
            options=["No segmentation available"],
            disabled=True,
        )
    with st.container(border=True):
        st.markdown("**Correspondence**")
        corr_values = render_specs(st, corr_specs, corr_defaults, "corr::")
    continue_submitted = st.form_submit_button(
        "Masks are corrected — continue pipeline",
        type="primary",
        icon=":material/check:",
        disabled=busy or not has_segmentation,
    )

selected_seg_path = (
    segmentation_options[selected_seg_name].seg_path
    if selected_seg_name in segmentation_options
    else None
)

continue_overrides = build_hydra_overrides(
    values_from_session(reg_specs, reg_defaults, "reg::"),
    values_from_session(seg_specs, seg_defaults, "seg::"),
    data_path=str(status.pipeline_data_path),
    alignment_only=False,
    skip_registration=True,
    correspondence_enabled=True,
    skip_segmentation=True,
    existing_seg_path=str(selected_seg_path) if selected_seg_path is not None else None,
    correspondence_values=values_from_session(corr_specs, corr_defaults, "corr::"),
)

with st.expander("Command for step 3"):
    st.code("python -m astroglial_morphology " + " ".join(continue_overrides))

if continue_submitted:
    if not has_segmentation:
        st.error("Save a segmentation first.")
    elif busy:
        st.warning("A job is already running. Wait for it to finish or cancel it.")
    else:
        overrides = build_hydra_overrides(
            values_from_session(reg_specs, reg_defaults, "reg::"),
            values_from_session(seg_specs, seg_defaults, "seg::"),
            data_path=str(status.pipeline_data_path),
            alignment_only=False,
            skip_registration=True,
            correspondence_enabled=True,
            skip_segmentation=True,
            existing_seg_path=(
                str(selected_seg_path) if selected_seg_path is not None else None
            ),
            correspondence_values=corr_values,
        )
        st.session_state.active_job = run_pipeline_subprocess(
            overrides,
            data_path=status.data_path,
            label="pipeline_correspondence",
        )
        st.rerun()
