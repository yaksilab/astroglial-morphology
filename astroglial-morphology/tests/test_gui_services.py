"""Tests for the Streamlit GUI service layer.

These cover experiment discovery, parameter diffs, and mask serialization —
i.e. the pure-Python helpers that back the pages, so they can run without a
real Streamlit runtime.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from astroglial_morphology.gui.services.experiment import (
    describe_experiment,
    is_experiment_folder,
    plane0_path,
)
from astroglial_morphology.gui.services.parameters import (
    PARAMETER_CATALOG,
    build_hydra_overrides,
    default_correspondence_params,
    default_registration_params,
    default_segmentation_params,
    diff_against_defaults,
)
from astroglial_morphology.gui.services.results import (
    downsample_for_display,
    load_metadata_payload,
    load_ops,
    load_projections,
    load_registration_qc,
    load_seg_file,
    mask_overlay,
    minmax_downsample,
    save_seg_masks,
)


def _make_plane(temp_dir: Path, *, with_metadata: bool = True) -> Path:
    plane = temp_dir / "suite2p" / "plane0"
    plane.mkdir(parents=True, exist_ok=True)
    ops = {
        "Ly": 8,
        "Lx": 8,
        "nframes": 4,
        "xoff": np.array([-1.0, 0.0, 1.0, 2.0]),
        "yoff": np.array([0.0, 0.5, 1.0, 1.5]),
        "corrXY": np.array([0.9, 0.8, 0.7, 0.6]),
    }
    np.save(plane / "ops.npy", ops, allow_pickle=True)
    (plane / "data.bin").write_bytes(b"\0" * 16)
    (plane.parent / ".registration_complete").touch()
    (plane / "mean_ch0_image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    seg_payload: Dict[str, Any] = {
        "masks": np.zeros((8, 8), dtype=np.int32),
        "filename": "mean_ch0_image.png",
    }
    np.save(plane / "mean_ch0_image_seg.npy", seg_payload, allow_pickle=True)
    if with_metadata:
        (plane / "pipeline_metadata.json").write_text(json.dumps({"created_at": "now"}))
    return plane


class TestExperimentInventory:
    def test_describe_direct_plane(self, temp_dir):
        plane = _make_plane(temp_dir)
        status = describe_experiment(plane)
        assert status.input_mode == "suite2p"
        assert status.plane_dir == plane
        assert status.has_registration is True
        assert status.has_segmentation is True
        assert status.has_metadata is True

    def test_describe_raw_folder(self, temp_dir):
        plane = _make_plane(temp_dir)
        (temp_dir / "test.tif").touch()
        status = describe_experiment(temp_dir)
        assert status.input_mode == "tif"
        assert status.plane_dir == plane
        assert status.has_registration is True

    def test_describe_missing_input(self, temp_dir):
        status = describe_experiment(temp_dir)
        assert not status.is_valid_input
        assert status.errors

    def test_is_experiment_folder(self, temp_dir):
        assert is_experiment_folder(temp_dir) is False
        plane = _make_plane(temp_dir)
        # Raw input alone should qualify.
        (temp_dir / "test.tif").touch()
        assert is_experiment_folder(temp_dir) is True
        # Direct plane0 folder should also qualify.
        assert is_experiment_folder(plane) is True

    def test_plane0_path_resolution(self, temp_dir):
        plane = _make_plane(temp_dir)
        assert plane0_path(temp_dir) == plane
        assert plane0_path(plane) == plane

    def test_subsegmented_masks_pair_with_parent_projection(self, temp_dir):
        plane = _make_plane(temp_dir)
        np.save(
            plane / "subsegmented_masks_seg.npy",
            {
                "masks": np.ones((8, 8), dtype=np.int32),
                "filename": "mean_ch0_image.png",
            },
            allow_pickle=True,
        )
        status = describe_experiment(plane)
        by_name = {sf.seg_path.name: sf for sf in status.segmentation_files}
        subseg = by_name["subsegmented_masks_seg.npy"]
        assert subseg.image_path == plane / "mean_ch0_image.png"
        assert subseg.projection == "mean"
        assert subseg.channel == "ch0"

    def test_subsegmented_masks_fall_back_to_cellpose_png(self, temp_dir):
        plane = _make_plane(temp_dir)
        np.save(
            plane / "subsegmented_masks_seg.npy",
            {"masks": np.ones((8, 8), dtype=np.int32)},
            allow_pickle=True,
        )
        status = describe_experiment(plane)
        by_name = {sf.seg_path.name: sf for sf in status.segmentation_files}
        subseg = by_name["subsegmented_masks_seg.npy"]
        assert subseg.image_path == plane / "mean_ch0_image.png"


class TestParameterCatalog:
    def test_registration_defaults_match_catalog(self):
        defaults = default_registration_params()
        assert defaults["channel"] == 0
        for spec in PARAMETER_CATALOG["registration"]:
            assert defaults[spec.key] == spec.default

    def test_segmentation_defaults_match_catalog(self):
        defaults = default_segmentation_params()
        assert defaults["mode"] == "single"
        for spec in PARAMETER_CATALOG["segmentation"]:
            assert defaults[spec.key] == spec.default

    def test_diff_detects_changes(self):
        defaults = default_registration_params()
        modified = {**defaults, "channel": 1, "nonrigid": True}
        diffs = diff_against_defaults("registration", modified)
        assert set(diffs.keys()) == {"channel", "nonrigid"}
        assert diffs["channel"]["used"] == 1
        assert diffs["nonrigid"]["used"] is True

    def test_hydra_overrides_emit_only_deltas(self, tmp_path):
        reg = default_registration_params()
        seg = default_segmentation_params()
        reg["channel"] = 1
        seg["mode"] = "ensemble"
        seg["pixels_per_micron"] = 3.14
        overrides = build_hydra_overrides(
            reg,
            seg,
            data_path=str(tmp_path),
            alignment_only=True,
            skip_registration=False,
            correspondence_enabled=False,
        )
        assert f"input.data_path={tmp_path}" in overrides
        assert "runtime.alignment_only=true" in overrides
        assert "correspondence.enabled=false" in overrides
        assert "registration.channel=1" in overrides
        assert "segmentation=ensemble" in overrides
        assert "input.pixels_per_micron=3.14" in overrides
        # Untouched values should not appear.
        assert not any(o.startswith("pipeline.suite2p_defaults.maxregshift") for o in overrides)

    def test_hydra_overrides_emit_one_photon_registration(self, tmp_path):
        reg = default_registration_params()
        reg["one_photon_reg"] = True
        reg["smooth_sigma"] = 3.0
        reg["block_size"] = [64, 64]
        overrides = build_hydra_overrides(
            reg,
            default_segmentation_params(),
            data_path=str(tmp_path),
            alignment_only=True,
            skip_registration=False,
            correspondence_enabled=False,
        )
        assert "pipeline.suite2p_defaults.one_photon_reg=true" in overrides
        assert "pipeline.suite2p_defaults.smooth_sigma=3.0" in overrides
        assert "pipeline.suite2p_defaults.block_size=[64,64]" in overrides

    def test_hydra_overrides_resume_after_correction(self, tmp_path):
        corr = default_correspondence_params()
        corr["segment_length"] = 12
        corr["trace_channels"] = "0,1"
        overrides = build_hydra_overrides(
            default_registration_params(),
            default_segmentation_params(),
            data_path=str(tmp_path),
            alignment_only=False,
            skip_registration=True,
            correspondence_enabled=True,
            skip_segmentation=True,
            correspondence_values=corr,
        )
        assert "registration.skip=true" in overrides
        assert "segmentation.skip=true" in overrides
        assert "correspondence.enabled=true" in overrides
        assert "correspondence.segment_length=12" in overrides
        assert "correspondence.trace_channels=[0,1]" in overrides
        assert "runtime.alignment_only=true" not in overrides

    def test_correspondence_defaults_match_catalog(self):
        defaults = default_correspondence_params()
        for spec in PARAMETER_CATALOG["correspondence"]:
            assert defaults[spec.key] == spec.default


class TestResultsLoaders:
    def test_load_ops(self, temp_dir):
        plane = _make_plane(temp_dir)
        ops = load_ops(plane)
        assert ops["Ly"] == 8
        assert ops["nframes"] == 4

    def test_load_seg_file(self, temp_dir):
        plane = _make_plane(temp_dir)
        payload = load_seg_file(plane / "mean_ch0_image_seg.npy")
        assert payload["masks"].shape == (8, 8)

    def test_save_seg_masks_writes_backup(self, temp_dir):
        plane = _make_plane(temp_dir)
        seg_path = plane / "mean_ch0_image_seg.npy"
        edited = np.zeros((8, 8), dtype=np.int32)
        edited[2:5, 2:5] = 3
        result_path = save_seg_masks(seg_path, edited)
        assert result_path == seg_path
        assert (plane / "mean_ch0_image_seg.npy.orig").is_file()
        payload = np.load(seg_path, allow_pickle=True).item()
        assert (payload["masks"] == edited).all()
        assert payload["manual_edited"] is True

    def test_save_seg_masks_backup_only_once(self, temp_dir):
        plane = _make_plane(temp_dir)
        seg_path = plane / "mean_ch0_image_seg.npy"
        original_bytes = seg_path.read_bytes()
        save_seg_masks(seg_path, np.zeros((8, 8), dtype=np.int32))
        backup_bytes = (plane / "mean_ch0_image_seg.npy.orig").read_bytes()
        assert backup_bytes == original_bytes
        # Second save should not overwrite the original backup.
        save_seg_masks(seg_path, np.ones((8, 8), dtype=np.int32))
        assert (plane / "mean_ch0_image_seg.npy.orig").read_bytes() == original_bytes

    def test_load_metadata_payload(self, temp_dir):
        plane = _make_plane(temp_dir)
        payload = load_metadata_payload(plane / "pipeline_metadata.json")
        assert payload["created_at"] == "now"
        assert load_metadata_payload(plane / "missing.json") is None

    def test_mask_overlay_preserves_shape(self):
        image = np.random.rand(10, 12).astype(np.float32)
        masks = np.zeros((10, 12), dtype=np.int32)
        masks[3:6, 3:6] = 1
        overlay = mask_overlay(image, masks)
        assert overlay.shape == (10, 12, 3)
        assert overlay.min() >= 0.0
        assert overlay.max() <= 1.0

    def test_downsample_for_display_strides_large_images(self):
        image = np.zeros((2000, 1000), dtype=np.float32)
        small = downsample_for_display(image, max_side=500)
        assert small.shape[0] <= 500
        assert small.shape[1] <= 500

    def test_load_registration_qc_drops_extra_ops_arrays(self, temp_dir):
        plane = _make_plane(temp_dir)
        ops = {
            "Ly": 64,
            "Lx": 64,
            "nframes": 3,
            "xoff": np.array([0.0, 1.0, 2.0]),
            "yoff": np.array([0.0, 0.0, 1.0]),
            "corrXY": np.array([0.9, 0.8, 0.7]),
            "meanImg": np.ones((64, 64), dtype=np.float32),
            "regPC": np.zeros((20, 64, 64), dtype=np.float32),
        }
        np.save(plane / "ops.npy", ops, allow_pickle=True)
        qc = load_registration_qc(plane, max_side=32)
        assert "regPC" not in qc
        assert qc["nframes"] == 3
        assert qc["meanImg"].shape[0] <= 32
        assert qc["xoff"].tolist() == [0.0, 1.0, 2.0]

    def test_load_projections_skips_non_projection_pngs(self, temp_dir):
        plane = _make_plane(temp_dir)
        (plane / "notes.png").write_bytes((plane / "mean_ch0_image.png").read_bytes())
        loaded = load_projections(plane)
        assert "notes" not in loaded

    def test_minmax_downsample_keeps_peaks(self):
        y = np.zeros(10000, dtype=float)
        y[1234] = 10.0
        y[5678] = -7.0
        frame, traces = minmax_downsample({"xoff": y}, max_points=400)
        assert traces["xoff"].size <= 800
        assert traces["xoff"].max() == 10.0
        assert traces["xoff"].min() == -7.0
        assert frame.size == traces["xoff"].size



class TestMaskEditorCodec:
    def test_roundtrip(self):
        from astroglial_morphology.gui.components.mask_editor.component import (
            decode_rle_to_masks,
            encode_masks_to_rle,
        )

        masks = np.zeros((8, 12), dtype=np.int32)
        masks[1:4, 1:5] = 5
        masks[5:7, 6:11] = 7
        encoded = encode_masks_to_rle(masks)
        assert encoded["width"] == 12
        assert encoded["height"] == 8
        assert encoded["max_label"] == 7
        # payload must round-trip losslessly.
        decoded = decode_rle_to_masks(encoded)
        assert decoded.dtype == np.int32
        assert (decoded == masks).all()

    def test_decode_rejects_wrong_size(self):
        from astroglial_morphology.gui.components.mask_editor.component import (
            decode_rle_to_masks,
        )

        buffer = np.zeros(4, dtype=np.int32).tobytes()
        with pytest.raises(ValueError):
            decode_rle_to_masks(
                {
                    "masks_b64": base64.b64encode(buffer).decode("ascii"),
                    "width": 8,
                    "height": 8,
                }
            )


class TestJobLogging:
    def test_hydra_log_override_uses_posix_path(self, tmp_path):
        from astroglial_morphology.gui.services.jobs import _hydra_log_override

        log_path = tmp_path / "run.log"
        override = _hydra_log_override(log_path)
        assert override.startswith("logging.log_file=")
        assert "\\" not in override.split("=", 1)[1]

    def test_pump_splits_carriage_returns(self, tmp_path):
        import io
        import threading
        from collections import deque

        from astroglial_morphology.gui.services.jobs import _pump_output

        stream = io.StringIO("frame 1\rframe 2\nRegistration completed\n")
        log_path = tmp_path / "job.log"
        buffer = deque()
        lock = threading.Lock()
        with log_path.open("w", encoding="utf-8") as fp:
            _pump_output(stream, fp, buffer, lock)
        text = log_path.read_text(encoding="utf-8")
        assert "frame 1" in text
        assert "frame 2" in text
        assert "Registration completed" in text
        assert "\r" not in text

    def test_tail_log_prefers_memory_buffer(self, tmp_path):
        from datetime import datetime

        from astroglial_morphology.gui.services.jobs import JobHandle

        handle = JobHandle(
            job_id="abc",
            command=["python"],
            log_path=tmp_path / "missing.log",
            started_at=datetime.now(),
        )
        handle._lines.extend(["one\n", "two\n", "three\n"])
        assert handle.tail_log(max_lines=2) == "two\nthree\n"
        assert handle.latest_line() == "three"

