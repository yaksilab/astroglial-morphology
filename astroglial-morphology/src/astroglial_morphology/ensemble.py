"""Three-model Cellpose ensemble support.

The ensemble intentionally keeps model prediction, mask merging, and model
asset resolution separate.  This makes the physical calibration and merge
behaviour testable without requiring Cellpose or a GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.request import urlopen

import numpy as np
from cellpose.io import masks_flows_to_seg
from cellpose.models import CellposeModel

from .logging_config import get_logger

logger = get_logger(__name__)

ROLE_ORDER = ("complete_cell", "processes", "cell_body")
DEFAULT_PROFILE_NAME = "cp3-three-part"


def _read_package_json(filename: str) -> dict[str, Any]:
    resource = resources.files("astroglial_morphology").joinpath("profiles", filename)
    return json.loads(resource.read_text(encoding="utf-8"))


def _positive_float(value: Any, label: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite positive number") from exc
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return converted


def calculate_diameter_pixels(
    *,
    pixels_per_micron: float,
    effective_diameter_um: Optional[float] = None,
    projected_area_um2: Optional[float] = None,
) -> float:
    """Convert one physical scale representation into a Cellpose diameter."""

    pixels_per_micron = _positive_float(pixels_per_micron, "pixels_per_micron")
    supplied = sum(value is not None for value in (effective_diameter_um, projected_area_um2))
    if supplied != 1:
        raise ValueError(
            "Specify exactly one of effective_diameter_um or projected_area_um2"
        )
    if effective_diameter_um is not None:
        return _positive_float(effective_diameter_um, "effective_diameter_um") * pixels_per_micron

    area = _positive_float(projected_area_um2, "projected_area_um2")
    return 2.0 * math.sqrt(area / math.pi) * pixels_per_micron


def _mask_statistics(masks: np.ndarray, pixels_per_micron: float) -> dict[str, Any]:
    labels, areas = np.unique(masks, return_counts=True)
    areas = areas[labels != 0].astype(float)
    if areas.size == 0:
        return {
            "count": 0,
            "pixel_area": {"min": None, "max": None, "median": None},
            "equivalent_diameter_px": {"min": None, "max": None, "median": None},
            "physical_area_um2": {"min": None, "max": None, "median": None},
            "equivalent_diameter_um": {"min": None, "max": None, "median": None},
        }

    equivalent_px = 2.0 * np.sqrt(areas / math.pi)
    physical_area = areas / (pixels_per_micron**2)
    equivalent_um = equivalent_px / pixels_per_micron

    def summary(values: np.ndarray) -> dict[str, float]:
        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }

    return {
        "count": int(areas.size),
        "pixel_area": summary(areas),
        "equivalent_diameter_px": summary(equivalent_px),
        "physical_area_um2": summary(physical_area),
        "equivalent_diameter_um": summary(equivalent_um),
    }


def reassign_consecutive_labels(masks: np.ndarray) -> np.ndarray:
    """Return a copy of *masks* with labels numbered from 1 without gaps."""

    result = np.zeros_like(masks)
    labels = np.unique(masks)
    labels = labels[labels != 0]
    for new_label, old_label in enumerate(labels, start=1):
        result[masks == old_label] = new_label
    return result


def extend_and_merge_masks(
    base_masks: np.ndarray,
    additional_masks: np.ndarray,
    overlap_threshold: float,
) -> np.ndarray:
    """Merge masks using the established complete/process/body overlap rule.

    A pair is related when overlap exceeds the threshold relative to either
    component.  One-to-many and many-to-one related components become one
    label.  Inputs are never modified.
    """

    if base_masks.shape != additional_masks.shape:
        raise ValueError("Masks must have the same shape to be merged")
    if not 0 <= overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be between 0 and 1")

    base_labels = np.unique(base_masks)
    base_labels = base_labels[base_labels != 0]
    additional_labels = np.unique(additional_masks)
    additional_labels = additional_labels[additional_labels != 0]
    result = base_masks.copy()
    new_label = int(base_labels.max()) + 1 if base_labels.size else 1

    related: dict[int, list[int]] = {}
    for additional_label in additional_labels:
        additional_region = additional_masks == additional_label
        additional_area = int(np.sum(additional_region))
        related_base: list[int] = []
        for base_label in base_labels:
            base_region = base_masks == base_label
            base_area = int(np.sum(base_region))
            overlap = int(np.sum(additional_region & base_region))
            if overlap / additional_area > overlap_threshold or overlap / base_area > overlap_threshold:
                related_base.append(int(base_label))
        if related_base:
            related[int(additional_label)] = related_base
        else:
            result[additional_region] = new_label
            new_label += 1

    one_to_many: dict[int, list[int]] = {}
    many_to_one: dict[int, list[int]] = {}
    for additional_label, related_base in related.items():
        if len(related_base) > 1:
            many_to_one[additional_label] = related_base
        else:
            one_to_many.setdefault(related_base[0], []).append(additional_label)

    for base_label, additional_for_base in one_to_many.items():
        result[base_masks == base_label] = new_label
        for additional_label in additional_for_base:
            result[additional_masks == additional_label] = new_label
        new_label += 1

    for additional_label, bases_for_additional in many_to_one.items():
        result[additional_masks == additional_label] = new_label
        for base_label in bases_for_additional:
            result[base_masks == base_label] = new_label
        new_label += 1

    return result


def merge_ensemble_masks(
    complete_cell: np.ndarray,
    processes: np.ndarray,
    cell_body: np.ndarray,
    *,
    processes_overlap_threshold: float = 0.15,
    cell_body_overlap_threshold: float = 0.35,
) -> np.ndarray:
    """Merge all ensemble roles in the prescribed, deterministic order."""

    merged = extend_and_merge_masks(complete_cell, processes, processes_overlap_threshold)
    merged = extend_and_merge_masks(merged, cell_body, cell_body_overlap_threshold)
    return reassign_consecutive_labels(merged)


@dataclass(frozen=True)
class ModelAsset:
    name: str
    version: str
    url: str
    sha256: str
    filename: str


@dataclass(frozen=True)
class RoleSpec:
    role: str
    asset: Optional[str]
    model_path: Optional[str]
    effective_diameter_um: Optional[float]
    projected_area_um2: Optional[float]
    cellpose: dict[str, Any]

    def diameter_pixels(self, pixels_per_micron: float) -> float:
        return calculate_diameter_pixels(
            pixels_per_micron=pixels_per_micron,
            effective_diameter_um=self.effective_diameter_um,
            projected_area_um2=self.projected_area_um2,
        )


@dataclass(frozen=True)
class EnsembleProfile:
    name: str
    version: str
    roles: dict[str, RoleSpec]
    processes_overlap_threshold: float
    cell_body_overlap_threshold: float
    combined_flow_role: str = "complete_cell"


@dataclass
class EnsembleSegmentationResult:
    masks: np.ndarray
    component_masks: dict[str, np.ndarray]
    component_paths: dict[str, str]
    combined_path: str
    role_diameters_px: dict[str, float]
    role_statistics: dict[str, dict[str, Any]]
    profile_name: str
    profile_version: str
    combined_flow_role: str
    merge_thresholds: dict[str, float]

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": "ensemble",
            "profile": self.profile_name,
            "profile_version": self.profile_version,
            "component_outputs": self.component_paths,
            "combined_output": self.combined_path,
            "role_diameters_px": self.role_diameters_px,
            "mask_statistics": self.role_statistics,
            "merge_thresholds": self.merge_thresholds,
            "combined_flow_role": self.combined_flow_role,
        }


def _parse_assets(payload: Mapping[str, Any]) -> dict[str, ModelAsset]:
    return {
        str(name): ModelAsset(name=str(name), **dict(asset))
        for name, asset in dict(payload.get("assets", {})).items()
    }


def _parse_profile(payload: Mapping[str, Any]) -> EnsembleProfile:
    roles_payload = payload.get("roles")
    if not isinstance(roles_payload, Mapping) or set(roles_payload) != set(ROLE_ORDER):
        raise ValueError(
            "An ensemble profile must define exactly complete_cell, processes, and cell_body roles"
        )

    roles: dict[str, RoleSpec] = {}
    for role in ROLE_ORDER:
        spec = dict(roles_payload[role])
        effective_diameter = spec.get("effective_diameter_um")
        projected_area = spec.get("projected_area_um2")
        # Validate now, rather than after a long model download.
        calculate_diameter_pixels(
            pixels_per_micron=1.0,
            effective_diameter_um=effective_diameter,
            projected_area_um2=projected_area,
        )
        asset = spec.get("asset")
        model_path = spec.get("model_path")
        if not asset and not model_path:
            raise ValueError(f"Ensemble role '{role}' requires asset or model_path")
        roles[role] = RoleSpec(
            role=role,
            asset=str(asset) if asset else None,
            model_path=str(model_path) if model_path else None,
            effective_diameter_um=(
                _positive_float(effective_diameter, f"{role}.effective_diameter_um")
                if effective_diameter is not None
                else None
            ),
            projected_area_um2=(
                _positive_float(projected_area, f"{role}.projected_area_um2")
                if projected_area is not None
                else None
            ),
            cellpose=dict(spec.get("cellpose", {})),
        )

    merge = dict(payload.get("merge", {}))
    return EnsembleProfile(
        name=str(payload.get("name", DEFAULT_PROFILE_NAME)),
        version=str(payload.get("version", "1")),
        roles=roles,
        processes_overlap_threshold=float(merge.get("processes_overlap_threshold", 0.15)),
        cell_body_overlap_threshold=float(merge.get("cell_body_overlap_threshold", 0.35)),
        combined_flow_role=str(payload.get("combined_flow_role", "complete_cell")),
    )


def load_ensemble_profile(
    profile_name: str = DEFAULT_PROFILE_NAME,
    config_path: Optional[str] = None,
) -> tuple[EnsembleProfile, dict[str, ModelAsset]]:
    """Load the packaged default profile or a complete custom JSON profile."""

    manifest = _read_package_json("model_manifest.json")
    assets = _parse_assets(manifest)
    if config_path is not None:
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    elif profile_name == DEFAULT_PROFILE_NAME:
        payload = _read_package_json("cp3-three-part-v1.json")
    else:
        raise ValueError(f"Unknown ensemble profile: {profile_name}")
    profile = _parse_profile(payload)
    for role in profile.roles.values():
        if role.asset is not None and role.asset not in assets:
            raise ValueError(f"Unknown model asset '{role.asset}' for role '{role.role}'")
    return profile, assets


def default_model_cache_dir() -> Path:
    configured = os.environ.get("ASTROGLIAL_MODEL_CACHE_DIR")
    if configured:
        return Path(configured)
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "astroglial-morphology" / "models"
    return Path.home() / ".cache" / "astroglial-morphology" / "models"


class ModelAssetResolver:
    """Resolve versioned model assets into a checksum-verified local cache."""

    def __init__(self, assets: Mapping[str, ModelAsset], cache_dir: Optional[str] = None):
        self.assets = dict(assets)
        self.cache_dir = Path(cache_dir) if cache_dir else default_model_cache_dir()

    @staticmethod
    def _checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _target(self, asset: ModelAsset) -> Path:
        return self.cache_dir / asset.version / asset.filename

    def resolve(self, asset_name: str) -> Path:
        if asset_name not in self.assets:
            raise ValueError(f"Unknown model asset: {asset_name}")
        asset = self.assets[asset_name]
        target = self._target(asset)
        if target.exists() and self._checksum(target).lower() == asset.sha256.lower():
            return target
        if target.exists():
            logger.warning("Removing checksum-mismatched cached model: %s", target)
            target.unlink()

        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=f"{target.name}.", suffix=".part", delete=False
            ) as handle:
                temp_path = Path(handle.name)
                with urlopen(asset.url) as response:
                    shutil.copyfileobj(response, handle)
            if self._checksum(temp_path).lower() != asset.sha256.lower():
                raise ValueError(
                    f"Checksum verification failed for {asset.name}; expected {asset.sha256}"
                )
            os.replace(temp_path, target)
            return target
        except Exception as exc:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(
                f"Could not download model '{asset.name}' from {asset.url}. "
                f"Check network access or provide a model cache with --model-cache-dir. {exc}"
            ) from exc

    def prefetch(self, profile: EnsembleProfile) -> dict[str, Path]:
        return {
            role: self.resolve(spec.asset)
            for role, spec in profile.roles.items()
            if spec.asset is not None
        }


class ThreeModelEnsembleSegmenter:
    """Run the three Cellpose roles and merge their masks in memory."""

    def __init__(
        self,
        *,
        profile: EnsembleProfile,
        assets: Mapping[str, ModelAsset],
        pixels_per_micron: float,
        gpu: bool = False,
        model_cache_dir: Optional[str] = None,
        model_factory: Callable[..., Any] = CellposeModel,
        save_function: Callable[..., Any] = masks_flows_to_seg,
    ) -> None:
        self.profile = profile
        self.pixels_per_micron = _positive_float(pixels_per_micron, "pixels_per_micron")
        self.gpu = gpu
        self.model_factory = model_factory
        self.save_function = save_function
        self.resolver = ModelAssetResolver(assets, model_cache_dir)

    def _model_path(self, spec: RoleSpec) -> str:
        if spec.model_path is not None:
            path = Path(spec.model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model path for {spec.role} does not exist: {path}")
            return str(path)
        if spec.asset is None:
            raise RuntimeError(f"No model source configured for {spec.role}")
        return str(self.resolver.resolve(spec.asset))

    def segment_img(
        self,
        image: np.ndarray,
        save_file_name: str,
        **image_eval_kwargs: Any,
    ) -> EnsembleSegmentationResult:
        component_masks: dict[str, np.ndarray] = {}
        component_paths: dict[str, str] = {}
        component_flows: dict[str, Any] = {}
        diameters: dict[str, float] = {}

        for role in ROLE_ORDER:
            spec = self.profile.roles[role]
            diameter = spec.diameter_pixels(self.pixels_per_micron)
            model = self.model_factory(gpu=self.gpu, pretrained_model=self._model_path(spec))
            params = dict(spec.cellpose)
            params.update(image_eval_kwargs)
            params["diameter"] = diameter
            logger.info("Segmenting ensemble role %s with parameters: %s", role, params)
            masks, flows, _ = model.eval(x=image, **params)
            masks = np.asarray(masks)
            component_base = f"{save_file_name}_{role}"
            self.save_function(image, masks, flows, component_base, diams=diameter)
            component_masks[role] = masks
            component_paths[role] = f"{component_base}_seg.npy"
            component_flows[role] = flows
            diameters[role] = diameter

        combined = merge_ensemble_masks(
            component_masks["complete_cell"],
            component_masks["processes"],
            component_masks["cell_body"],
            processes_overlap_threshold=self.profile.processes_overlap_threshold,
            cell_body_overlap_threshold=self.profile.cell_body_overlap_threshold,
        )
        combined_flows = component_flows[self.profile.combined_flow_role]
        self.save_function(
            image,
            combined,
            combined_flows,
            save_file_name,
            diams=diameters[self.profile.combined_flow_role],
        )

        role_statistics = {
            role: _mask_statistics(masks, self.pixels_per_micron)
            for role, masks in component_masks.items()
        }
        role_statistics["combined"] = _mask_statistics(combined, self.pixels_per_micron)
        return EnsembleSegmentationResult(
            masks=combined,
            component_masks=component_masks,
            component_paths=component_paths,
            combined_path=f"{save_file_name}_seg.npy",
            role_diameters_px=diameters,
            role_statistics=role_statistics,
            profile_name=self.profile.name,
            profile_version=self.profile.version,
            combined_flow_role=self.profile.combined_flow_role,
            merge_thresholds={
                "processes_overlap_threshold": self.profile.processes_overlap_threshold,
                "cell_body_overlap_threshold": self.profile.cell_body_overlap_threshold,
            },
        )
