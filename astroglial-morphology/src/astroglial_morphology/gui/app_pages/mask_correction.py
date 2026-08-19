"""Mask correction page: edit ROIs in-browser with the CCv2 canvas."""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import streamlit as st

from astroglial_morphology.gui.components.mask_editor import (
    decode_rle_to_masks,
    mask_editor,
)
from astroglial_morphology.gui.form_widgets import values_from_session
from astroglial_morphology.gui.job_panel import render_job_panel
from astroglial_morphology.gui.services.experiment import describe_experiment
from astroglial_morphology.gui.services.jobs import run_pipeline_subprocess
from astroglial_morphology.gui.services.parameters import (
    PARAMETER_CATALOG,
    build_hydra_overrides,
    default_correspondence_params,
    default_registration_params,
    default_segmentation_params,
)
from astroglial_morphology.gui.services.results import load_seg_file, save_seg_masks

_PIPELINE_PAGE = str(Path(__file__).with_name("pipeline.py"))

st.title("Mask correction")
st.caption("Paint, split, merge, and delete ROIs, then save back into the seg file.")

render_job_panel()

status = st.session_state.get("status")
if status is None or not status.has_segmentation:
    st.info("Run segmentation first, then return here to edit masks.")
    st.stop()

seg_options = {sf.seg_path.name: sf for sf in status.segmentation_files}
seg_name = st.selectbox(
    "Segmentation file", options=list(seg_options.keys()), key="editor-seg"
)
seg_file = seg_options[seg_name]

if seg_file.image_path is None or not seg_file.image_path.is_file():
    st.error(
        f"Cannot find the projection image paired with {seg_file.seg_path.name}. "
        "Ensure the PNG saved during segmentation is still present."
    )
    st.stop()

payload = load_seg_file(seg_file.seg_path)
masks = np.asarray(payload["masks"], dtype=np.int32)
image = mpimg.imread(str(seg_file.image_path))

st.write(
    {
        "seg file": str(seg_file.seg_path),
        "projection image": str(seg_file.image_path),
        "shape": list(masks.shape),
        "ROIs": int(np.unique(masks[masks != 0]).size),
    }
)

editor_key = f"mask-editor::{seg_file.seg_path}"


def _on_save() -> None:
    component_state = st.session_state.get(editor_key)
    payload = getattr(component_state, "save", None)
    if payload is None and isinstance(component_state, dict):
        payload = component_state.get("save")
    if not isinstance(payload, dict) or not payload.get("masks_b64"):
        st.session_state["mask_editor_notice"] = (
            "error",
            "Save did not include a mask payload.",
        )
        return
    save_token = payload.get("timestamp")
    if save_token is not None and st.session_state.get("mask_editor_saved_token") == save_token:
        return
    try:
        edited = decode_rle_to_masks(payload)
        path = save_seg_masks(seg_file.seg_path, edited)
    except (ValueError, OSError) as exc:
        st.session_state["mask_editor_notice"] = ("error", f"Failed to save masks: {exc}")
        return
    st.session_state["mask_editor_notice"] = (
        "success",
        f"Saved {path.name}. Original backed up to {path.name}.orig.",
    )
    if save_token is not None:
        st.session_state["mask_editor_saved_token"] = save_token
    st.session_state.status = describe_experiment(status.data_path)
    st.rerun()


mask_editor(
    image=image,
    masks=masks,
    key=editor_key,
    on_save=_on_save,
)

st.caption(
    "Tools: Select (S), Brush (B) traces an outline and fills on release, "
    "Erase (E), Split (X), Overlay (O), Pan (space). "
    "Shift+click to add to selection. Wheel to zoom. Ctrl+Z / Ctrl+Y for undo/redo."
)

notice = st.session_state.pop("mask_editor_notice", None)
if notice:
    kind, text = notice
    if kind == "error":
        st.error(text)
    else:
        st.success(text)

if not status.correspondence_files:
    st.divider()
    st.subheader("Continue pipeline")
    st.write(
        "When the saved masks look right, continue classification, "
        "subsegmentation, traces, and correspondence export. This does not "
        "re-run Cellpose."
    )
    active_job = st.session_state.get("active_job")
    busy = active_job is not None and active_job.is_running()
    if st.button(
        "Masks are corrected — continue pipeline",
        type="primary",
        icon=":material/check:",
        disabled=busy,
        key="mask-continue-pipeline",
    ):
        overrides = build_hydra_overrides(
            values_from_session(
                PARAMETER_CATALOG["registration"],
                default_registration_params(),
                "reg::",
            ),
            values_from_session(
                PARAMETER_CATALOG["segmentation"],
                default_segmentation_params(),
                "seg::",
            ),
            data_path=str(status.data_path),
            alignment_only=False,
            skip_registration=True,
            correspondence_enabled=True,
            skip_segmentation=True,
            correspondence_values=values_from_session(
                PARAMETER_CATALOG["correspondence"],
                default_correspondence_params(),
                "corr::",
            ),
        )
        st.session_state.active_job = run_pipeline_subprocess(
            overrides,
            data_path=status.data_path,
            label="pipeline_correspondence",
        )
        st.rerun()
    st.page_link(
        _PIPELINE_PAGE,
        label="Tune correspondence parameters",
        icon=":material/account_tree:",
    )
