"""Typed pipeline configuration.

`configs/default.yaml` holds every default. A run-specific YAML only needs the keys it
changes; it is deep-merged on top of the defaults and validated here, so typos fail loudly.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Strict):
    data_root: Path = Path("data/runs")


class VideoConfig(_Strict):
    proxy_fps: float = 30.0
    proxy_height: int = 1080
    rotate: Literal[0, 90, 180, 270] = 0
    crf: int = 18
    gop: int = 15


class TrackingConfig(_Strict):
    backend: Literal["sam3", "sam2"] = "sam3"
    model_ids: dict[str, str] = Field(
        default_factory=lambda: {"sam3": "facebook/sam3", "sam2": "facebook/sam2.1-hiera-small"}
    )
    device: str = "auto"
    presence_logit_threshold: float = 0.0
    min_area_frac: float = 2e-5
    max_area_change: float = 4.0
    min_appearance_sim: float = 0.55
    motion_gate_sigmas: float = 6.0
    patience: int = 3
    keep_memory_frames: int = 16
    backfill: bool = True


class ReconConfig(_Strict):
    backend: Literal["wat3r", "vggt"] = "wat3r"
    model_ids: dict[str, str] = Field(
        default_factory=lambda: {"wat3r": "lsxi77777/Wat3R", "vggt": "facebook/VGGT-1B"}
    )
    device: str = "auto"
    frame_range: tuple[int, int] | None = None
    keyframe_fps: float = 3.0
    sharpness_window: bool = True
    chunk_size: int = 32
    chunk_overlap: int = 8
    image_size: int = 518
    conf_percentile: float = 50.0
    max_depth_percentile: float = 95.0
    mask_fish: bool = True
    drop_above_camera: float | None = 0.02
    camera_height_percentile: float = 50.0
    voxel_size: float = 0.002
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    stereo_scale_mode: Literal["joint", "independent", "global"] = "joint"
    stereo_right_depth: bool = True
    cache_dir: str = "~/.cache/reeftrack"


class FusionConfig(_Strict):
    mask_erode_px: int = 2
    depth_stat: Literal["median", "mean"] = "median"
    max_interp_gap_s: float = 2.0
    smooth_window: int = 5


DEFAULT_BEHAVIOURS = [
    "Calm Swimming", "Foraging", "Near Other Picasso triggerfish", "Sheltering", "Stationary",
    "Cleaner fish interaction", "Chasing", "Erratic Swimming", "Escaping", "Fighting",
    "Housekeeping", "Scratching",
]


class AnnotationConfig(_Strict):
    behaviours: list[str] = Field(default_factory=lambda: list(DEFAULT_BEHAVIOURS))
    annotator: str = ""  # pre-fill the annotator name (else asked once and remembered per computer)


class Config(_Strict):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    recon: ReconConfig = Field(default_factory=ReconConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)

    def digest(self) -> str:
        """Short hash of the full config, stored with every output for provenance."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:10]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into a copy of `base` (override wins)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def drop_unknown_keys(raw: dict[str, Any], model: type[BaseModel] = None, prefix: str = "") -> list[str]:
    """Remove keys the current schema doesn't know (in place); returns the dropped dotted keys.

    Used for config snapshots saved inside old runs, so a run stays loadable after the code
    evolves. User-supplied override files remain strict (typos must fail loudly).
    """
    model = model or Config
    dropped = []
    for key in list(raw):
        field = model.model_fields.get(key)
        if field is None:
            dropped.append(prefix + key)
            del raw[key]
        elif isinstance(raw[key], dict) and isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel):
            dropped += drop_unknown_keys(raw[key], field.annotation, f"{prefix}{key}.")
    return dropped


def load_config(*overrides: str | Path | dict[str, Any] | None) -> Config:
    """Load defaults, then apply each override (YAML path or dict) in order."""
    raw: dict[str, Any] = yaml.safe_load(DEFAULT_CONFIG.read_text()) if DEFAULT_CONFIG.exists() else {}
    for ov in overrides:
        if ov is None:
            continue
        if isinstance(ov, dict):
            raw = deep_merge(raw, ov)
        else:
            raw = deep_merge(raw, yaml.safe_load(Path(ov).read_text()) or {})
    return Config.model_validate(raw)
