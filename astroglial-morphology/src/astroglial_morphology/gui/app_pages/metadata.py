"""Metadata page: highlight default vs non-default pipeline parameters."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from astroglial_morphology.gui.services.cache import metadata_for
from astroglial_morphology.gui.services.parameters import PARAMETER_CATALOG

st.title("Pipeline metadata")
st.caption("Compare the last saved run against the package defaults.")

status = st.session_state.get("status")
if status is None or status.plane_dir is None:
    st.info("Load a folder containing a Suite2p `plane0` directory.")
    st.stop()

metadata_path = status.plane_dir / "pipeline_metadata.json"
payload = metadata_for(metadata_path)
if payload is None:
    st.warning(f"No pipeline metadata found at {metadata_path}.")
    st.stop()

st.write(
    {
        "created_at": payload.get("created_at"),
        "pipeline_version": payload.get("pipeline_version"),
        "python_version": payload.get("python_version"),
        "suite2p_version": payload.get("suite2p_version"),
        "hostname": payload.get("hostname"),
        "input_mode": payload.get("input_mode"),
    }
)


def _rows_from_snapshot(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    parameters = payload.get("parameters") or {}
    defaults = payload.get("parameter_defaults") or {}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for group in ("registration", "segmentation", "runtime"):
        used = parameters.get(group) or {}
        default_map = defaults.get(group) or {}
        rows: List[Dict[str, Any]] = []
        for key in sorted(set(used) | set(default_map)):
            u_val = used.get(key)
            d_val = default_map.get(key)
            state = "Default" if u_val == d_val else "Overridden"
            rows.append(
                {
                    "Parameter": key,
                    "Used": u_val,
                    "Default": d_val,
                    "State": state,
                }
            )
        if rows:
            grouped[group] = rows
    return grouped


def _rows_from_catalog(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Fallback when a run predates the structured snapshot."""

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for group, specs in PARAMETER_CATALOG.items():
        rows: List[Dict[str, Any]] = []
        for spec in specs:
            used = payload.get(spec.key)
            state = "Default" if used == spec.default or used is None else "Overridden"
            rows.append(
                {
                    "Parameter": spec.key,
                    "Used": used,
                    "Default": spec.default,
                    "State": state,
                }
            )
        grouped[group] = rows
    return grouped


snapshot = payload.get("parameters")
if isinstance(snapshot, dict):
    grouped = _rows_from_snapshot(payload)
else:
    st.info(
        "This run does not contain a structured parameter snapshot yet. "
        "Comparing against catalog defaults instead."
    )
    grouped = _rows_from_catalog(payload)

for group, rows in grouped.items():
    st.subheader(group.title())
    st.dataframe(rows, width="stretch")

overrides = payload.get("parameter_overrides") or {}
if overrides:
    st.subheader("Overrides only")
    st.json(overrides, expanded=True)
