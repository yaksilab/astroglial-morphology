"""Shared registration QC rendering for the inspect and registration pages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .cache import registration_qc_for
from .results import minmax_downsample


def _trace_chart(
    frame: np.ndarray, series: Mapping[str, np.ndarray], y_title: str
) -> alt.Chart:
    frame = np.asarray(frame, dtype=float)
    payload = {"frame": frame}
    for name, values in series.items():
        payload[name] = np.asarray(values, dtype=float)
    table = pd.DataFrame(payload).melt("frame", var_name="series", value_name="value")
    return (
        alt.Chart(table)
        .mark_line(clip=True)
        .encode(
            x=alt.X("frame:Q", title="Frame"),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color("series:N", legend=alt.Legend(title="")),
            tooltip=["frame", "series", "value"],
        )
        .properties(height=240)
        .interactive()
    )


def _show_chart(chart: alt.Chart) -> None:
    # Zoom/pan must stay in the browser.  A selection rerun would reload ops
    # previews and make the chart feel frozen.
    st.altair_chart(chart, width="stretch", on_select="ignore")


@st.fragment
def show_registration_qc(plane_dir: Optional[Path], *, show_images: bool = True) -> None:
    """Render metrics and interactive, downsampled shift traces.

    This runs as a fragment so chart zoom does not rebuild the rest of the page.
    Mean/reference images stay hidden until requested because encoding them is
    what made inspect feel stuck while interacting with the plots.
    """

    if plane_dir is None or not (Path(plane_dir) / "ops.npy").is_file():
        st.info("No registration output yet. Run registration to see QC metrics.")
        return

    qc = registration_qc_for(Path(plane_dir), max_side=512)
    if not qc:
        st.warning("Could not load ops.npy for QC.")
        return

    badframes = np.asarray(qc.get("badframes", []), dtype=bool)
    num_badframes = int(np.sum(badframes)) if badframes.size else 0
    frac_bad = float(num_badframes / badframes.size) if badframes.size else 0.0
    corr = np.asarray(qc.get("corrXY", []), dtype=float)
    cols = st.columns(4)
    cols[0].metric("Frames", int(qc.get("nframes") or 0))
    cols[1].metric("Bad frames", num_badframes, f"{frac_bad * 100:.1f}%")
    cols[2].metric("corrXY mean", f"{float(np.mean(corr)):.3f}" if corr.size else "n/a")
    timing = qc.get("timing") or {}
    registration_s = timing.get("registration") if isinstance(timing, dict) else None
    cols[3].metric(
        "Registration time (s)",
        f"{float(registration_s):.1f}" if registration_s is not None else "n/a",
    )

    xoff = np.asarray(qc.get("xoff", []), dtype=float)
    yoff = np.asarray(qc.get("yoff", []), dtype=float)
    if xoff.size:
        frame, traces = minmax_downsample(
            {"xoff": xoff, "yoff": yoff[: xoff.size]}, max_points=1500
        )
        st.markdown("**Rigid shifts per frame**")
        st.caption("Scroll or drag to zoom. Traces are peak-preserving downsampled for speed.")
        _show_chart(_trace_chart(frame, traces, "Shift (pixels)"))
    if corr.size:
        frame, traces = minmax_downsample({"corrXY": corr}, max_points=1500)
        st.markdown("**Frame-to-mean correlation (corrXY)**")
        _show_chart(_trace_chart(frame, traces, "corrXY"))

    if not show_images:
        return

    if st.toggle("Show mean and reference images", value=False, key="qc-show-images"):
        image_cols = st.columns(3)
        for col, key, caption in zip(
            image_cols,
            ("meanImg", "meanImg_chan2", "refImg"),
            ("Mean image (ch0)", "Mean image (ch1)", "Reference frame"),
        ):
            image = qc.get(key)
            if image is not None:
                col.image(
                    np.asarray(image), caption=caption, width="stretch", clamp=True
                )
