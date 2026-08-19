"""Live job status and log viewer used by the processing pages."""

from __future__ import annotations

import streamlit as st

from astroglial_morphology.gui.services.experiment import describe_experiment
from astroglial_morphology.gui.services.jobs import JobHandle, JobStatus


_STATUS_LABELS = {
    JobStatus.PENDING: (":material/hourglass_empty:", "Queued", "blue"),
    JobStatus.RUNNING: (":material/play_circle:", "Running", "blue"),
    JobStatus.SUCCESS: (":material/check_circle:", "Succeeded", "green"),
    JobStatus.FAILED: (":material/error:", "Failed", "red"),
    JobStatus.CANCELLED: (":material/cancel:", "Cancelled", "orange"),
}


def _refresh_experiment_if_needed(job: JobHandle) -> None:
    if job.results_consumed:
        return
    if job.status != JobStatus.SUCCESS:
        if job.status in {JobStatus.FAILED, JobStatus.CANCELLED}:
            job.results_consumed = True
        return
    job.results_consumed = True
    data_path = (st.session_state.get("data_path") or "").strip()
    if data_path:
        st.session_state.status = describe_experiment(data_path)


def _render_job_body() -> None:
    job = st.session_state.get("active_job")
    if not isinstance(job, JobHandle):
        return

    was_running = job.is_running()
    job.poll()
    _refresh_experiment_if_needed(job)
    if was_running and not job.is_running():
        st.rerun()

    icon, label, color = _STATUS_LABELS[job.status]
    title = job.label.replace("_", " ") or "pipeline"
    log_text = job.tail_log(max_lines=250)

    with st.container(border=True):
        top = st.columns([3, 1])
        top[0].markdown(f":{color}[{icon} **{label}**] · {title}")
        top[1].caption(f"Elapsed {job.elapsed_label()}")
        st.caption(job.latest_line())
        st.code(
            log_text or "Waiting for the worker to print its first line…",
            language="text",
            height=360,
            wrap_lines=True,
            width="stretch",
        )

        if job.is_running():
            if st.button("Cancel", icon=":material/stop:", key=f"job-cancel-{job.job_id}"):
                job.cancel()
                st.rerun()
        else:
            cols = st.columns(2)
            with cols[0]:
                if st.button(
                    "Dismiss", icon=":material/close:", key=f"job-dismiss-{job.job_id}"
                ):
                    st.session_state.active_job = None
                    st.rerun()
            with cols[1]:
                if job.log_path.is_file():
                    st.download_button(
                        "Download log",
                        data=job.log_path.read_bytes(),
                        file_name=job.log_path.name,
                        mime="text/plain",
                        icon=":material/download:",
                        key=f"job-download-{job.job_id}",
                    )
        st.caption(f"Log file · `{job.log_path}`")
        if job.status == JobStatus.FAILED:
            st.error(
                f"The {title} job exited with code {job.return_code}. "
                "The log above is the captured worker output."
            )
        elif job.status == JobStatus.SUCCESS:
            st.success(f"The {title} job finished successfully.")


@st.fragment(run_every=1.0)
def _live_job_panel() -> None:
    _render_job_body()


def render_job_panel() -> None:
    """Show the active job, auto-refreshing while it is running."""

    job = st.session_state.get("active_job")
    if not isinstance(job, JobHandle):
        return
    st.subheader("Run log")
    job.poll()
    if job.is_running():
        _live_job_panel()
        return
    _render_job_body()


def render_sidebar_job_chip() -> None:
    """Compact running-job indicator for the shared sidebar."""

    job = st.session_state.get("active_job")
    if not isinstance(job, JobHandle):
        return
    job.poll()
    icon, label, color = _STATUS_LABELS[job.status]
    st.sidebar.markdown("### Current run")
    st.sidebar.markdown(
        f":{color}[{icon} **{label}**] {job.label.replace('_', ' ')} · {job.elapsed_label()}"
    )
    st.sidebar.caption(job.latest_line())
