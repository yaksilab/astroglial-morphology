"""Shared Streamlit widgets for pipeline parameter catalogs."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import streamlit as st

from astroglial_morphology.gui.services.parameters import ParameterSpec

_GRID_COLUMNS = 2


def render_specs(
    container,
    specs: Sequence[ParameterSpec],
    defaults: Mapping[str, Any],
    key_prefix: str,
) -> Dict[str, Any]:
    """Render catalog *specs* into *container* and return the widget values."""

    values: Dict[str, Any] = {}
    if not specs:
        return values

    for title, group in _grouped_specs(specs):
        if title:
            container.caption(title)
        values.update(_render_group(container, group, defaults, key_prefix))
    return values


def values_from_session(
    specs: Sequence[ParameterSpec],
    defaults: Mapping[str, Any],
    key_prefix: str,
) -> Dict[str, Any]:
    """Read previously rendered widget values from session state."""

    values: Dict[str, Any] = {}
    for spec in specs:
        widget_key = f"{key_prefix}{spec.key}"
        if widget_key in st.session_state:
            values[spec.key] = _coerce_session_value(
                spec, st.session_state[widget_key], defaults[spec.key]
            )
        else:
            values[spec.key] = defaults[spec.key]
    return values


def _grouped_specs(
    specs: Sequence[ParameterSpec],
) -> List[Tuple[str, List[ParameterSpec]]]:
    groups: List[Tuple[str, List[ParameterSpec]]] = []
    for spec in specs:
        title = spec.section or ""
        if not groups or groups[-1][0] != title:
            groups.append((title, [spec]))
        else:
            groups[-1][1].append(spec)
    return groups


def _render_group(
    container,
    specs: Sequence[ParameterSpec],
    defaults: Mapping[str, Any],
    key_prefix: str,
) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    field_specs = [spec for spec in specs if spec.kind != "bool"]
    bool_specs = [spec for spec in specs if spec.kind == "bool"]

    if field_specs:
        ncols = min(_GRID_COLUMNS, len(field_specs))
        columns = container.columns(ncols, vertical_alignment="top")
        for index, spec in enumerate(field_specs):
            values[spec.key] = _render_spec(
                columns[index % ncols], spec, defaults, key_prefix
            )

    if bool_specs:
        with container.container(horizontal=True, gap="medium"):
            for spec in bool_specs:
                values[spec.key] = _render_spec(st, spec, defaults, key_prefix)
    return values


def _render_spec(
    container,
    spec: ParameterSpec,
    defaults: Mapping[str, Any],
    key_prefix: str,
) -> Any:
    default = defaults[spec.key]
    widget_key = f"{key_prefix}{spec.key}"
    label = spec.label
    if spec.kind == "bool":
        return container.checkbox(
            label, value=bool(default), key=widget_key, help=spec.help
        )
    if spec.kind == "choice" and spec.choices:
        options = list(spec.choices)
        index = options.index(default) if default in options else 0
        return container.selectbox(
            label, options=options, index=index, key=widget_key, help=spec.help
        )
    if spec.kind == "integer":
        if spec.default is None:
            return _optional_number_input(container, spec, widget_key, integer=True)
        return int(
            container.number_input(
                label,
                value=int(default),
                min_value=int(spec.min_value) if spec.min_value is not None else None,
                max_value=int(spec.max_value) if spec.max_value is not None else None,
                step=int(spec.step or 1),
                key=widget_key,
                help=spec.help,
            )
        )
    if spec.kind == "int_list":
        display = ",".join(str(item) for item in (default or []))
        raw = container.text_input(
            label,
            value=display,
            key=widget_key,
            help=spec.help,
        )
        parsed = _parse_int_list(raw)
        if parsed is None:
            container.warning(f"{label}: expected comma-separated integers.")
            return list(default or [])
        return parsed
    if spec.kind == "number":
        if spec.default is None:
            return _optional_number_input(container, spec, widget_key, integer=False)
        return float(
            container.number_input(
                label,
                value=float(default),
                min_value=(
                    float(spec.min_value) if spec.min_value is not None else None
                ),
                max_value=(
                    float(spec.max_value) if spec.max_value is not None else None
                ),
                step=float(spec.step or 0.01),
                key=widget_key,
                help=spec.help,
            )
        )
    raw = container.text_input(
        label,
        value="" if default is None else str(default),
        key=widget_key,
        help=spec.help,
    )
    return None if raw.strip() == "" else raw.strip()


def _optional_number_input(container, spec: ParameterSpec, widget_key: str, *, integer: bool):
    help_text = spec.help or ""
    if "blank" not in help_text.lower():
        help_text = f"{help_text} Leave blank for auto.".strip()
    raw = container.text_input(
        spec.label,
        value="",
        key=widget_key,
        help=help_text or None,
    )
    raw = raw.strip()
    if raw == "":
        return None
    try:
        return int(raw) if integer else float(raw)
    except ValueError:
        container.warning(
            f"{spec.label}: expected an integer." if integer else f"{spec.label}: expected a number."
        )
        return spec.default


def _coerce_session_value(spec: ParameterSpec, value: Any, default: Any) -> Any:
    if spec.kind == "int_list":
        parsed = _parse_int_list(value)
        return parsed if parsed is not None else list(default or [])
    if spec.kind == "integer" and spec.default is None:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    if spec.kind == "number" and spec.default is None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return value


def _parse_int_list(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        try:
            return [int(item) for item in value]
        except (TypeError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError:
        return None
