"""Catalog of GUI-exposed pipeline parameters and their defaults.

The catalog is the source of truth for widget labels, ranges, and how a
parameter maps back onto Hydra CLI overrides. It lets pages render forms and
lets the metadata page decide which values are still at their defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from ...config import PipelineConfig
from ...ensemble import DEFAULT_PROFILE_NAME


@dataclass
class ParameterSpec:
    """Metadata about a single parameter exposed by the GUI."""

    key: str
    label: str
    group: str
    default: Any
    hydra_path: Optional[str] = None
    kind: str = "text"
    choices: Optional[Sequence[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    help: Optional[str] = None
    advanced: bool = False
    section: Optional[str] = None
    parser: Optional[Callable[[Any], Any]] = None

    def format_default(self) -> str:
        if self.default is None:
            return "auto"
        if isinstance(self.default, bool):
            return "on" if self.default else "off"
        if isinstance(self.default, (list, tuple)):
            return ",".join(str(item) for item in self.default)
        return str(self.default)


def _config_defaults() -> tuple[Dict[str, Any], Dict[str, Any]]:
    config = PipelineConfig()
    return dict(config.SUITE2P_DEFAULTS), dict(config.SEGMENTATION_DEFAULTS)


_SUITE2P_DEFAULTS, _SEGMENTATION_DEFAULTS = _config_defaults()


_REGISTRATION_SPECS: List[ParameterSpec] = [
    ParameterSpec(
        key="channel",
        label="Alignment channel",
        group="registration",
        default=0,
        hydra_path="registration.channel",
        kind="choice",
        choices=[0, 1],
        help="Zero-based channel used to compute registration shifts.",
    ),
    ParameterSpec(
        key="maxregshift",
        label="Max registration shift",
        group="registration",
        default=_SUITE2P_DEFAULTS.get("maxregshift", 0.11),
        hydra_path="pipeline.suite2p_defaults.maxregshift",
        kind="number",
        min_value=0.0,
        max_value=0.5,
        step=0.01,
        help="Fraction of frame width that the alignment is allowed to move.",
    ),
    ParameterSpec(
        key="smooth_sigma",
        label="Spatial smooth sigma",
        group="registration",
        default=_SUITE2P_DEFAULTS.get("smooth_sigma", 1.15),
        hydra_path="pipeline.suite2p_defaults.smooth_sigma",
        kind="number",
        min_value=0.0,
        max_value=20.0,
        step=0.05,
        help="Gaussian smoothing in pixels before phase correlation. Raise this for 1-photon data.",
    ),
    ParameterSpec(
        key="smooth_sigma_time",
        label="Temporal smooth sigma",
        group="registration",
        default=_SUITE2P_DEFAULTS.get("smooth_sigma_time", 1),
        hydra_path="pipeline.suite2p_defaults.smooth_sigma_time",
        kind="number",
        min_value=0.0,
        max_value=20.0,
        step=0.1,
        help="Gaussian smoothing sigma along the time axis before alignment.",
    ),
    ParameterSpec(
        key="th_badframes",
        label="Bad-frame threshold",
        group="registration",
        default=_SUITE2P_DEFAULTS.get("th_badframes", 1.0),
        hydra_path="pipeline.suite2p_defaults.th_badframes",
        kind="number",
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        help="Frames with peak correlation below this times the median are marked bad.",
    ),
    ParameterSpec(
        key="nimg_init",
        label="Reference frames (nimg_init)",
        group="registration",
        default=None,
        hydra_path="pipeline.suite2p_defaults.nimg_init",
        kind="integer",
        min_value=1,
        max_value=2000,
        step=1,
        help="Frames used to build the reference image. Leave blank to size from the recording.",
    ),
    ParameterSpec(
        key="batch_size",
        label="Batch size",
        group="registration",
        default=None,
        hydra_path="pipeline.suite2p_defaults.batch_size",
        kind="integer",
        min_value=1,
        max_value=2000,
        step=1,
        help="Frames registered per batch. Leave blank to size from the recording.",
    ),
    ParameterSpec(
        key="two_step_registration",
        label="Two-step registration",
        group="registration",
        default=_SUITE2P_DEFAULTS.get("two_step_registration", False),
        hydra_path="pipeline.suite2p_defaults.two_step_registration",
        kind="bool",
        help="Run an initial rigid pass before the main registration.",
    ),
    ParameterSpec(
        key="keep_movie_raw",
        label="Keep raw movie",
        group="registration",
        default=_SUITE2P_DEFAULTS.get("keep_movie_raw", False),
        hydra_path="pipeline.suite2p_defaults.keep_movie_raw",
        kind="bool",
        help="Keep unregistered binaries so registration can be rebuilt without reconverting.",
    ),
    ParameterSpec(
        key="nonrigid",
        label="Non-rigid registration",
        group="registration",
        default=_SUITE2P_DEFAULTS.get("nonrigid", False),
        hydra_path="pipeline.suite2p_defaults.nonrigid",
        kind="bool",
        section="Nonrigid",
        help="Enable Suite2p's block-wise non-rigid alignment.",
    ),
    ParameterSpec(
        key="block_size",
        label="Block size",
        group="registration",
        default=_SUITE2P_DEFAULTS.get("block_size", [128, 128]),
        hydra_path="pipeline.suite2p_defaults.block_size",
        kind="int_list",
        section="Nonrigid",
        help="Non-rigid block size as y,x pixels, e.g. 128,128.",
    ),
    ParameterSpec(
        key="snr_thresh",
        label="SNR threshold",
        group="registration",
        default=_SUITE2P_DEFAULTS.get("snr_thresh", 1.2),
        hydra_path="pipeline.suite2p_defaults.snr_thresh",
        kind="number",
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        section="Nonrigid",
        help="Minimum SNR for a non-rigid block to contribute to the offset.",
    ),
    ParameterSpec(
        key="maxregshiftNR",
        label="Max non-rigid shift",
        group="registration",
        default=_SUITE2P_DEFAULTS.get("maxregshiftNR", 5),
        hydra_path="pipeline.suite2p_defaults.maxregshiftNR",
        kind="integer",
        min_value=0,
        max_value=50,
        step=1,
        section="Nonrigid",
        help="Maximum extra shift in pixels allowed for each non-rigid block.",
    ),
    ParameterSpec(
        key="one_photon_reg",
        label="1-photon registration",
        group="registration",
        default=_SUITE2P_DEFAULTS.get("one_photon_reg", False),
        hydra_path="pipeline.suite2p_defaults.one_photon_reg",
        kind="bool",
        section="1P",
        help="Enable Suite2p 1-photon registration (high-pass spatial filter). Use this for 1-photon data.",
    ),
    ParameterSpec(
        key="spatial_hp_reg",
        label="Spatial high-pass",
        group="registration",
        default=_SUITE2P_DEFAULTS.get("spatial_hp_reg", 42),
        hydra_path="pipeline.suite2p_defaults.spatial_hp_reg",
        kind="integer",
        min_value=0,
        max_value=200,
        step=1,
        section="1P",
        help="Spatial high-pass window in pixels used when 1-photon registration is on.",
    ),
    ParameterSpec(
        key="pre_smooth",
        label="Pre-smooth",
        group="registration",
        default=_SUITE2P_DEFAULTS.get("pre_smooth", 0),
        hydra_path="pipeline.suite2p_defaults.pre_smooth",
        kind="integer",
        min_value=0,
        max_value=20,
        step=1,
        section="1P",
        help="Gaussian smoothing applied before the 1-photon high-pass filter. 0 disables it.",
    ),
    ParameterSpec(
        key="spatial_taper",
        label="Spatial taper",
        group="registration",
        default=_SUITE2P_DEFAULTS.get("spatial_taper", 40),
        hydra_path="pipeline.suite2p_defaults.spatial_taper",
        kind="integer",
        min_value=0,
        max_value=200,
        step=1,
        section="1P",
        help="Pixels tapered at the edges during 1-photon registration.",
    ),
    ParameterSpec(
        key="regmetrics",
        label="Compute registration metrics",
        group="registration",
        default=False,
        hydra_path="registration.regmetrics",
        kind="bool",
        section="Run",
        help="Ask Suite2p to save its optional registration-quality metrics.",
    ),
    ParameterSpec(
        key="reg_tif",
        label="Save registered TIFFs",
        group="registration",
        default=False,
        hydra_path="registration.reg_tif",
        kind="bool",
        section="Run",
        help="Write registered frames as TIFF files (large output).",
    ),
    ParameterSpec(
        key="force",
        label="Force re-registration",
        group="registration",
        default=False,
        hydra_path="registration.force",
        kind="bool",
        section="Run",
        help="Ignore cached registration and rebuild inputs from source.",
    ),
    ParameterSpec(
        key="subpixel",
        label="Subpixel factor",
        group="registration",
        default=_SUITE2P_DEFAULTS.get("subpixel", 10),
        hydra_path="pipeline.suite2p_defaults.subpixel",
        kind="integer",
        min_value=1,
        max_value=50,
        step=1,
        advanced=True,
        help="Suite2p subpixel interpolation factor.",
    ),
    ParameterSpec(
        key="tau",
        label="Indicator tau (s)",
        group="registration",
        default=_SUITE2P_DEFAULTS.get("tau", 3),
        hydra_path="pipeline.suite2p_defaults.tau",
        kind="number",
        min_value=0.1,
        max_value=30.0,
        step=0.1,
        advanced=True,
        help="Indicator decay constant used by Suite2p defaults.",
    ),
]


_SEGMENTATION_SPECS: List[ParameterSpec] = [
    ParameterSpec(
        key="mode",
        label="Segmentation mode",
        group="segmentation",
        default="single",
        hydra_path="segmentation",
        kind="choice",
        choices=["single", "ensemble"],
        help="Single-model Cellpose or the three-model ensemble.",
    ),
    ParameterSpec(
        key="projection",
        label="Projection",
        group="segmentation",
        default="mean",
        hydra_path="segmentation.projection",
        kind="choice",
        choices=["mean", "max_projection"],
        help="Projection image used as Cellpose input.",
    ),
    ParameterSpec(
        key="channel",
        label="Channel",
        group="segmentation",
        default="auto",
        hydra_path="segmentation.channel",
        kind="choice",
        choices=["auto", "both", "0", "1"],
        help="Channel selection for segmentation (auto uses ch1 if two channels).",
    ),
    ParameterSpec(
        key="model_path",
        label="Cellpose model path",
        group="segmentation",
        default=None,
        hydra_path="segmentation.model_path",
        kind="text",
        help="Optional path to a custom Cellpose 3 model.",
    ),
    ParameterSpec(
        key="ensemble_profile",
        label="Ensemble profile",
        group="segmentation",
        default=DEFAULT_PROFILE_NAME,
        hydra_path="segmentation.ensemble_profile",
        kind="text",
        help="Named ensemble profile used when mode is 'ensemble'.",
    ),
    ParameterSpec(
        key="pixels_per_micron",
        label="Pixels per micron",
        group="segmentation",
        default=None,
        hydra_path="input.pixels_per_micron",
        kind="number",
        min_value=0.0,
        max_value=50.0,
        step=0.01,
        help="Required for ensemble mode; overrides metadata otherwise.",
    ),
    ParameterSpec(
        key="flow_threshold",
        label="Flow threshold",
        group="segmentation",
        default=_SEGMENTATION_DEFAULTS.get("flow_threshold", 0.4),
        hydra_path="pipeline.segmentation_defaults.flow_threshold",
        kind="number",
        min_value=0.0,
        max_value=3.0,
        step=0.05,
        help="Cellpose flow error threshold; lower is stricter.",
    ),
    ParameterSpec(
        key="cellprob_threshold",
        label="Cell probability threshold",
        group="segmentation",
        default=_SEGMENTATION_DEFAULTS.get("cellprob_threshold", 0.0),
        hydra_path="pipeline.segmentation_defaults.cellprob_threshold",
        kind="number",
        min_value=-6.0,
        max_value=6.0,
        step=0.1,
        help="Cellpose cellprob threshold; lower keeps more masks.",
    ),
    ParameterSpec(
        key="diameter",
        label="Diameter override (px)",
        group="segmentation",
        default=None,
        hydra_path="pipeline.segmentation_defaults.diameter",
        kind="number",
        min_value=0.0,
        max_value=500.0,
        step=1.0,
        help="Optional override; otherwise derived from calibration.",
    ),
    ParameterSpec(
        key="min_size",
        label="Minimum mask size (px)",
        group="segmentation",
        default=_SEGMENTATION_DEFAULTS.get("min_size", 80),
        hydra_path="pipeline.segmentation_defaults.min_size",
        kind="integer",
        min_value=0,
        max_value=100000,
        step=1,
        help="Masks below this pixel count are discarded.",
    ),
    ParameterSpec(
        key="use_gpu",
        label="Use GPU",
        group="segmentation",
        default=False,
        hydra_path="runtime.use_gpu",
        kind="bool",
        help="Enable GPU inference (requires a compatible CUDA install).",
    ),
]


_CORRESPONDENCE_SPECS: List[ParameterSpec] = [
    ParameterSpec(
        key="segment_length",
        label="Subsegment length (px)",
        group="correspondence",
        default=5,
        hydra_path="correspondence.segment_length",
        kind="integer",
        min_value=1,
        max_value=200,
        step=1,
        help="Length of each subsegment along the cell axis, in pixels.",
    ),
    ParameterSpec(
        key="delta_x",
        label="Correspondence delta x",
        group="correspondence",
        default=20.0,
        hydra_path="correspondence.delta_x",
        kind="number",
        min_value=0.0,
        max_value=200.0,
        step=1.0,
        help="X-axis grouping distance used when aligning correspondence.",
    ),
    ParameterSpec(
        key="subsegmentation_mode",
        label="Subsegmentation mode",
        group="correspondence",
        default="equal_length",
        hydra_path="correspondence.subsegmentation_mode",
        kind="choice",
        choices=["equal_length", "compartments"],
        help="equal_length splits along the axis; compartments uses neck distance.",
    ),
    ParameterSpec(
        key="trace_channels",
        label="Trace channels",
        group="correspondence",
        default=None,
        hydra_path="correspondence.trace_channels",
        kind="text",
        help="Comma-separated zero-based channels, e.g. 0 or 0,1. Leave blank for auto.",
    ),
]


PARAMETER_CATALOG: Dict[str, List[ParameterSpec]] = {
    "registration": _REGISTRATION_SPECS,
    "segmentation": _SEGMENTATION_SPECS,
    "correspondence": _CORRESPONDENCE_SPECS,
}


def default_registration_params() -> Dict[str, Any]:
    """Return the registration-form defaults keyed by :attr:`ParameterSpec.key`."""

    return {spec.key: spec.default for spec in _REGISTRATION_SPECS}


def default_segmentation_params() -> Dict[str, Any]:
    """Return the segmentation-form defaults keyed by :attr:`ParameterSpec.key`."""

    return {spec.key: spec.default for spec in _SEGMENTATION_SPECS}


def default_correspondence_params() -> Dict[str, Any]:
    """Return the correspondence-form defaults keyed by :attr:`ParameterSpec.key`."""

    return {spec.key: spec.default for spec in _CORRESPONDENCE_SPECS}


def diff_against_defaults(
    group: str, values: Mapping[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Compare *values* to the catalog defaults for *group*.

    Returns a mapping of parameter key to ``{"default": ..., "used": ...}`` for
    every value that differs from the catalog default.
    """

    specs = PARAMETER_CATALOG.get(group, [])
    diffs: Dict[str, Dict[str, Any]] = {}
    for spec in specs:
        if spec.key not in values:
            continue
        used = values[spec.key]
        if used != spec.default:
            diffs[spec.key] = {"default": spec.default, "used": used}
    return diffs


def iter_specs(group: Optional[str] = None) -> Iterable[ParameterSpec]:
    """Iterate over specs, optionally filtered by *group*."""

    if group is None:
        for specs in PARAMETER_CATALOG.values():
            yield from specs
        return
    yield from PARAMETER_CATALOG.get(group, [])


def build_hydra_overrides(
    registration_values: Mapping[str, Any],
    segmentation_values: Mapping[str, Any],
    *,
    data_path: str,
    alignment_only: bool,
    skip_registration: bool,
    correspondence_enabled: bool,
    skip_segmentation: bool = False,
    correspondence_values: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Translate GUI form values into Hydra ``key=value`` overrides.

    Only values that differ from the catalog defaults are emitted so the
    resulting command mirrors what the user actually changed.
    """

    overrides: List[str] = [f"input.data_path={data_path!s}"]

    if alignment_only:
        overrides.append("runtime.alignment_only=true")
    if skip_registration:
        overrides.append("registration.skip=true")
    if skip_segmentation:
        overrides.append("segmentation.skip=true")
    overrides.append(
        f"correspondence.enabled={'true' if correspondence_enabled else 'false'}"
    )

    def _emit(spec: ParameterSpec, value: Any) -> None:
        if spec.hydra_path is None:
            return
        if value == spec.default:
            return
        if spec.key == "mode":
            overrides.append(f"segmentation={value}")
            return
        if spec.key == "trace_channels":
            parsed = _parse_trace_channels(value)
            if parsed is None:
                return
            joined = ",".join(str(item) for item in parsed)
            overrides.append(f"{spec.hydra_path}=[{joined}]")
            return
        if spec.kind == "int_list":
            parsed = _parse_int_list(value)
            if parsed is None:
                return
            if parsed == list(spec.default or []):
                return
            joined = ",".join(str(item) for item in parsed)
            overrides.append(f"{spec.hydra_path}=[{joined}]")
            return
        if value is None:
            overrides.append(f"{spec.hydra_path}=null")
        elif isinstance(value, bool):
            overrides.append(f"{spec.hydra_path}={'true' if value else 'false'}")
        else:
            overrides.append(f"{spec.hydra_path}={value}")

    for spec in _REGISTRATION_SPECS:
        if spec.key in registration_values:
            _emit(spec, registration_values[spec.key])
    for spec in _SEGMENTATION_SPECS:
        if spec.key in segmentation_values:
            _emit(spec, segmentation_values[spec.key])
    if correspondence_values:
        for spec in _CORRESPONDENCE_SPECS:
            if spec.key in correspondence_values:
                _emit(spec, correspondence_values[spec.key])
    return overrides


def _parse_trace_channels(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_int_list(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]
