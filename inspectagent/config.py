"""Runtime configuration (environment variables) and tunable defaults.

All knobs live here so the agent loop, evaluation and deployment read the same values.
Environment variables are prefixed with ``INSPECTAGENT_``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(f"INSPECTAGENT_{name}", default)


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


# Categories of the VisA dataset used by the project (see scripts/prepare_visa.py).
CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4", "macaroni1", "cashew")

# Physical scale assumption used to report defect sizes in millimetres:
# scene pixels per millimetre for each category (documented in the technical report).
PX_PER_MM = {"pcb1": 20.0, "pcb2": 20.0, "pcb3": 20.0, "pcb4": 20.0, "macaroni1": 25.0, "cashew": 15.0}


@dataclass
class CameraConfig:
    sensor_long_side: int = 768  # captured image long side (px); short side follows scene aspect
    max_zoom: float = 4.0
    min_ev: float = -3.0
    max_ev: float = 3.0
    # simulated timing model (ms)
    t_exposure: float = 60.0
    t_readout: float = 25.0
    t_autofocus: float = 250.0
    t_light_switch: float = 60.0
    t_move_per_unit: float = 120.0  # gantry move per unit of normalized distance
    t_zoom_per_step: float = 40.0  # per unit of zoom change


@dataclass
class QualityConfig:
    target_luma: float = 118.0
    min_luma: float = 60.0
    max_luma: float = 190.0
    max_clipped_frac: float = 0.02
    max_crushed_frac: float = 0.05
    min_sharpness_ratio: float = 0.55  # vs clean capture at same view
    max_noise_sigma: float = 7.0  # on 0..255 scale
    max_glare_frac: float = 0.004


@dataclass
class DetectorConfig:
    template_long_side: int = 1280  # resolution of the statistical template (scene px are rescaled)
    sigma_floor: float = 2.0  # Lab units added in quadrature to per-pixel std
    zmap_blur_sigma: float = 2.0  # px at capture resolution
    topk_frac: float = 0.0005  # image score = mean of top-k fraction of z-map
    topk_min: int = 25
    min_inlier_ratio: float = 0.25
    min_inliers: int = 20
    min_ncc: float = 0.45
    min_valid_frac: float = 0.6
    min_region_area: int = 12  # capture px
    max_regions: int = 5
    # deep detector (PatchCore-lite on cv2.dnn)
    deep_input_long_side: int = 448
    deep_coreset_size: int = 8000
    deep_train_images: int = 200


@dataclass
class PolicyConfig:
    """Human-control policy: what the agent may decide on its own."""

    auto_pass_max_prob: float = 0.15  # may auto-PASS only if fused P(anomalous) <= this
    auto_fail_min_prob: float = 0.90  # may auto-FAIL only if fused P(anomalous) >= this
    max_steps: int = 14
    max_captures: int = 8
    max_deep_calls: int = 2
    max_sim_time_ms: float = 5000.0
    require_reliable_evidence: bool = True  # auto decisions need a reliable detector result
    scrap_requires_human: bool = True  # automatic FAILs go to quarantine; scrapping needs a person


@dataclass
class Settings:
    camera: CameraConfig = field(default_factory=CameraConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    models_dir: Path = field(default_factory=lambda: Path(_env("MODELS_DIR", str(REPO_ROOT / "models"))))
    samples_dir: Path = field(default_factory=lambda: Path(_env("SAMPLES_DIR", str(REPO_ROOT / "samples"))))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(REPO_ROOT / "data"))))
    results_dir: Path = field(default_factory=lambda: Path(_env("RESULTS_DIR", str(REPO_ROOT / "results"))))

    storage: str = field(default_factory=lambda: _env("STORAGE", "local"))  # local | aws
    local_storage_dir: Path = field(default_factory=lambda: Path(_env("LOCAL_STORAGE_DIR", str(REPO_ROOT / "runs"))))
    s3_bucket: str = field(default_factory=lambda: _env("S3_BUCKET", ""))
    ddb_table: str = field(default_factory=lambda: _env("DDB_TABLE", ""))
    aws_region: str = field(default_factory=lambda: os.environ.get("AWS_REGION", _env("AWS_REGION", "us-east-1")))

    planner_default: str = field(default_factory=lambda: _env("PLANNER", "rule"))  # rule | bedrock
    bedrock_model_id: str = field(default_factory=lambda: _env("BEDROCK_MODEL_ID", ""))
    bedrock_max_tokens: int = field(default_factory=lambda: _env_int("BEDROCK_MAX_TOKENS", 1024))

    cv_threads: int = field(default_factory=lambda: _env_int("CV_THREADS", 0))  # 0 = OpenCV default
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings()
    return _SETTINGS


def reset_settings() -> None:
    """Re-read environment variables (used by tests)."""
    global _SETTINGS
    _SETTINGS = None
