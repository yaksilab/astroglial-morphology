"""Focused tests for physical scaling and in-memory three-model merging."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from astroglial_morphology.ensemble import (
    EnsembleProfile,
    ModelAsset,
    ModelAssetResolver,
    RoleSpec,
    ThreeModelEnsembleSegmenter,
    calculate_diameter_pixels,
    extend_and_merge_masks,
    load_ensemble_profile,
    merge_ensemble_masks,
)


def test_physical_diameter_conversion_supports_diameter_and_area():
    assert calculate_diameter_pixels(
        pixels_per_micron=3.168, effective_diameter_um=6.0
    ) == pytest.approx(19.008)
    assert calculate_diameter_pixels(
        pixels_per_micron=3.168, effective_diameter_um=7.89
    ) == pytest.approx(24.99552)
    assert calculate_diameter_pixels(
        pixels_per_micron=2.0, projected_area_um2=np.pi * 9.0
    ) == pytest.approx(12.0)


def test_physical_diameter_rejects_missing_or_duplicate_scale():
    with pytest.raises(ValueError, match="exactly one"):
        calculate_diameter_pixels(pixels_per_micron=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        calculate_diameter_pixels(
            pixels_per_micron=1.0,
            effective_diameter_um=2.0,
            projected_area_um2=3.0,
        )


def test_merge_ensemble_masks_is_in_memory_and_relabels():
    complete = np.zeros((8, 8), dtype=np.int32)
    complete[1:3, 1:3] = 5
    processes = np.zeros_like(complete)
    processes[1:3, 1:3] = 7
    processes[5:7, 5:7] = 9
    body = np.zeros_like(complete)
    body[1:3, 1:3] = 12

    merged = merge_ensemble_masks(complete, processes, body)

    np.testing.assert_array_equal(complete[1:3, 1:3], np.full((2, 2), 5))
    assert set(np.unique(merged)) == {0, 1, 2}
    assert merged[1, 1] != 0
    assert merged[5, 5] != 0
    assert merged[1, 1] != merged[5, 5]


def test_merge_handles_one_to_many_and_empty_roles():
    base = np.zeros((8, 8), dtype=np.int32)
    base[2:6, 2:6] = 1
    additional = np.zeros_like(base)
    additional[2:4, 2:4] = 1
    additional[4:6, 4:6] = 2

    merged = extend_and_merge_masks(base, additional, overlap_threshold=0.15)
    assert set(np.unique(merged)) == {0, 2}
    empty = np.zeros_like(base)
    assert np.array_equal(
        merge_ensemble_masks(base, empty, empty),
        np.where(base != 0, 1, 0),
    )


def _role(role: str, model_path: Path, diameter: float) -> RoleSpec:
    return RoleSpec(
        role=role,
        asset=None,
        model_path=str(model_path),
        effective_diameter_um=diameter,
        projected_area_um2=None,
        cellpose={"flow_threshold": 0.5, "cellprob_threshold": 0.4, "min_size": 15},
    )


def test_ensemble_uses_role_parameters_and_saves_components_and_combined(tmp_path):
    model_file = tmp_path / "model"
    model_file.write_bytes(b"model")
    profile = EnsembleProfile(
        name="test",
        version="1",
        roles={
            "complete_cell": _role("complete_cell", model_file, 7.89),
            "processes": _role("processes", model_file, 7.89),
            "cell_body": _role("cell_body", model_file, 6.0),
        },
        processes_overlap_threshold=0.15,
        cell_body_overlap_threshold=0.35,
    )
    component_masks = []
    complete = np.zeros((10, 10), dtype=np.int32)
    complete[1:3, 1:3] = 1
    processes = np.zeros_like(complete)
    processes[5:7, 5:7] = 1
    body = np.zeros_like(complete)
    body[1:3, 1:3] = 1
    component_masks.extend([complete, processes, body])

    models = []
    def model_factory(**kwargs):
        model = Mock()
        model.eval.return_value = (component_masks[len(models)], ["flow"], None)
        models.append(model)
        return model

    saved = []
    def save_function(image, masks, flows, filename, **kwargs):
        saved.append((masks.copy(), flows, filename, kwargs))

    segmenter = ThreeModelEnsembleSegmenter(
        profile=profile,
        assets={},
        pixels_per_micron=3.168,
        model_factory=model_factory,
        save_function=save_function,
    )
    result = segmenter.segment_img(np.ones((10, 10)), str(tmp_path / "mean_ch0_image"))

    assert [call[2] for call in saved] == [
        str(tmp_path / "mean_ch0_image_complete_cell"),
        str(tmp_path / "mean_ch0_image_processes"),
        str(tmp_path / "mean_ch0_image_cell_body"),
        str(tmp_path / "mean_ch0_image"),
    ]
    assert models[0].eval.call_args.kwargs["diameter"] == pytest.approx(24.99552)
    assert models[2].eval.call_args.kwargs["diameter"] == pytest.approx(19.008)
    assert saved[0][3]["diams"] == pytest.approx(24.99552)
    assert result.combined_path.endswith("mean_ch0_image_seg.npy")
    assert result.role_statistics["combined"]["count"] == 2


def test_model_asset_resolver_uses_verified_cache_without_network(tmp_path, monkeypatch):
    contents = b"verified model"
    asset = ModelAsset(
        name="test",
        version="v1",
        filename="model",
        url="https://example.invalid/model",
        sha256=hashlib.sha256(contents).hexdigest(),
    )
    resolver = ModelAssetResolver({"test": asset}, cache_dir=str(tmp_path))
    monkeypatch.setattr(
        "astroglial_morphology.ensemble.urlopen", lambda url: io.BytesIO(contents)
    )
    path = resolver.resolve("test")
    assert path.read_bytes() == contents
    monkeypatch.setattr(
        "astroglial_morphology.ensemble.urlopen",
        lambda url: pytest.fail("cache hit should not fetch"),
    )
    assert resolver.resolve("test") == path


def test_packaged_profile_has_expected_role_diameters():
    profile, _ = load_ensemble_profile()
    assert profile.roles["complete_cell"].diameter_pixels(3.168) == pytest.approx(24.99552)
    assert profile.roles["processes"].diameter_pixels(3.168) == pytest.approx(24.99552)
    assert profile.roles["cell_body"].diameter_pixels(3.168) == pytest.approx(19.008)
