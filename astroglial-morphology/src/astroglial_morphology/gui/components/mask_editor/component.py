"""Python wrapper for the CCv2 mask editor component.

The frontend lives in adjacent ``editor.html``, ``editor.css``, and
``editor.js`` files. We register the component inline via
``st.components.v2.component`` so no build step is required.

Asset mtime is part of the registration cache key so HTML/JS/CSS edits
reload on the next Streamlit rerun without restarting the process.
"""

from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import numpy as np

_ASSET_DIR = Path(__file__).parent


@lru_cache(maxsize=4)
def _register_component(mtime: int):
    import streamlit as st  # Local import so the codec functions remain usable

    assets = {
        "html": (_ASSET_DIR / "editor.html").read_text(encoding="utf-8"),
        "css": (_ASSET_DIR / "editor.css").read_text(encoding="utf-8"),
        "js": (_ASSET_DIR / "editor.js").read_text(encoding="utf-8"),
    }
    return st.components.v2.component(
        name="astroglial_mask_editor",
        html=assets["html"],
        css=assets["css"],
        js=assets["js"],
    )


def _asset_mtime() -> int:
    return max(
        (_ASSET_DIR / name).stat().st_mtime_ns
        for name in ("editor.html", "editor.css", "editor.js")
    )


def encode_image_to_data_url(image: np.ndarray) -> str:
    """Encode a 2D/3D image array into a base64 PNG data URL for the canvas."""

    from PIL import Image

    array = np.asarray(image)
    if array.dtype != np.uint8:
        finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.floating) else array
        if finite.size == 0:
            scaled = np.zeros_like(array, dtype=np.uint8)
        else:
            lo, hi = np.percentile(finite, [1.0, 99.0])
            if hi <= lo:
                scaled = np.zeros_like(array, dtype=np.uint8)
            else:
                scaled = np.clip((array - lo) / (hi - lo), 0.0, 1.0) * 255.0
                scaled = scaled.astype(np.uint8)
    else:
        scaled = array

    if scaled.ndim == 3 and scaled.shape[-1] == 2:
        padded = np.zeros((*scaled.shape[:2], 3), dtype=np.uint8)
        padded[..., 0] = scaled[..., 0]
        padded[..., 1] = scaled[..., 1]
        scaled = padded

    if scaled.ndim == 2:
        pil = Image.fromarray(scaled, mode="L").convert("RGB")
    else:
        pil = Image.fromarray(scaled[..., :3])

    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_masks_to_rle(masks: np.ndarray) -> Dict[str, Any]:
    """Encode masks as a base64 Int32 buffer.

    The name preserves the plan's terminology but the payload is a raw
    row-major buffer; that keeps the JS side simple and is small enough for
    typical projections (< 2 MB even at 2048x2048).
    """

    array = np.ascontiguousarray(np.asarray(masks, dtype=np.int32))
    return {
        "masks_b64": base64.b64encode(array.tobytes()).decode("ascii"),
        "height": int(array.shape[0]),
        "width": int(array.shape[1]),
        "max_label": int(array.max()) if array.size else 0,
    }


def decode_rle_to_masks(payload: Dict[str, Any]) -> np.ndarray:
    """Inverse of :func:`encode_masks_to_rle`."""

    if not payload:
        raise ValueError("Empty payload from mask editor")
    data = base64.b64decode(payload["masks_b64"])
    array = np.frombuffer(data, dtype=np.int32).copy()
    height = int(payload["height"])
    width = int(payload["width"])
    if array.size != height * width:
        raise ValueError(
            f"Mask payload size {array.size} does not match {height}x{width}"
        )
    return array.reshape((height, width))


def mask_editor(
    *,
    image: np.ndarray,
    masks: np.ndarray,
    key: str,
    on_save=None,
) -> Any:
    """Render the mask editor.

    Returns the CCv2 result object. When the user clicks *Save*, the
    ``save`` trigger fires and ``result.save`` contains ``masks_b64``,
    ``width``, ``height``, and ``max_label``.

    ``on_save_change`` is always registered so Streamlit exposes the trigger
    attribute. Do not pass ``default=`` unless every key has a matching
    ``on_<key>_change`` callback.
    """

    component = _register_component(_asset_mtime())
    encoded = encode_masks_to_rle(masks)
    data = {
        "image": encode_image_to_data_url(image),
        **encoded,
    }
    return component(
        key=key,
        data=data,
        on_save_change=on_save or (lambda: None),
    )
