"""Shared data contracts for InspectAgent.

Every module (vision tools, agent loop, planners, MCP server, web API, evaluation)
exchanges these types. Keep them JSON-serializable through ``to_dict()``; numpy
arrays (images, anomaly maps) are deliberately excluded from serialization and are
referenced by artifact ids instead.

Coordinate conventions
----------------------
* **Scene frame**: pixel grid of the full-resolution part image (the "physical part"),
  x to the right, y down, size ``(W_s, H_s)``.
* **View**: ``View(cx, cy, zoom)``; ``cx, cy`` are normalized scene coordinates of the
  view centre in ``[0, 1]``; ``zoom >= 1``. The view window in scene pixels has size
  ``(W_s / zoom, H_s / zoom)`` and is shifted (not shrunk) to stay inside the scene.
* **Capture frame**: pixel grid of a captured image (sensor resolution). The commanded
  mapping scene -> capture is the affine ``Capture.scene_to_capture()``; the simulator
  additionally applies an unknown pose jitter which detectors must estimate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

LightMode = Literal["ring", "diffuse", "low_angle"]
VerdictLabel = Literal["PASS", "FAIL", "NEEDS_REVIEW"]
RoutingAction = Literal["route_pass", "route_quarantine", "route_scrap", "hold_for_review"]
DetectorName = Literal["fast", "deep"]

QUALITY_ISSUES = ("blurred", "underexposed", "overexposed", "noisy", "glare")


def _round(obj: Any, nd: int = 4) -> Any:
    """Recursively round floats (and numpy scalars) for compact, stable JSON."""
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), nd)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, nd) for v in obj]
    return obj


@dataclass(frozen=True)
class View:
    cx: float = 0.5
    cy: float = 0.5
    zoom: float = 1.0

    def to_dict(self) -> dict:
        return _round(asdict(self))


@dataclass(frozen=True)
class CaptureSettings:
    """What the agent controls when it takes a picture."""

    view: View = field(default_factory=View)
    exposure_ev: float = 0.0  # exposure compensation / digital gain, clamped to [-3, 3]
    light: LightMode = "ring"  # ring (bright, may glare) | diffuse (dimmer, no glare) | low_angle (relief)
    autofocus: bool = False  # run autofocus before this capture (+time)

    def to_dict(self) -> dict:
        return _round(
            {
                "view": self.view.to_dict(),
                "exposure_ev": self.exposure_ev,
                "light": self.light,
                "autofocus": self.autofocus,
            }
        )


@dataclass
class NuisanceConditions:
    """Hidden per-part world state used ONLY by the camera simulator.

    The agent never sees these values; it must infer them from images.
    """

    ambient_ev: float = 0.0  # illumination offset of the scene (EV stops)
    defocus_sigma: float = 0.0  # Gaussian blur sigma (sensor px) when autofocus is off
    noise_sigma: float = 1.0  # read-noise std on the 0..255 scale at unit gain
    glare_strength: float = 0.0  # 0..1 specular glare intensity under ring light
    glare_seed: int = 0  # placement of glare blobs (scene coordinates)
    rot_deg: float = 0.0  # pose jitter: rotation in degrees
    shift_x: float = 0.0  # pose jitter: translation as fraction of scene width
    shift_y: float = 0.0  # pose jitter: translation as fraction of scene height
    jpeg_quality: int = 92  # sensor/ISP compression
    preset: str = "custom"  # name of the regime/preset that produced these values

    def to_dict(self) -> dict:
        return _round(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "NuisanceConditions":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


@dataclass
class Capture:
    capture_id: str
    index: int  # 0-based order within the inspection session
    settings: CaptureSettings
    image: np.ndarray  # BGR uint8, sensor resolution (excluded from to_dict)
    window: tuple[float, float, float, float]  # x0, y0, w, h of the view window in scene px
    scene_size: tuple[int, int]  # (W_s, H_s)
    sim_time_ms: float  # simulated wall-clock cost of this capture (motion + AF + exposure)
    oracle: bool = False  # True for clean, degradation-free captures (evaluation upper bound)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h

    def scene_to_capture(self) -> np.ndarray:
        """2x3 affine mapping scene px -> capture px for the *commanded* view."""
        x0, y0, ww, wh = self.window
        w, h = self.size
        sx, sy = w / ww, h / wh
        return np.array([[sx, 0.0, -x0 * sx], [0.0, sy, -y0 * sy]], dtype=np.float64)

    def capture_to_scene(self) -> np.ndarray:
        """2x3 affine mapping capture px -> scene px for the commanded view."""
        x0, y0, ww, wh = self.window
        w, h = self.size
        return np.array([[ww / w, 0.0, x0], [0.0, wh / h, y0]], dtype=np.float64)

    def to_dict(self) -> dict:
        w, h = self.size
        return _round(
            {
                "capture_id": self.capture_id,
                "index": self.index,
                "settings": self.settings.to_dict(),
                "window": list(self.window),
                "scene_size": list(self.scene_size),
                "size": [w, h],
                "sim_time_ms": self.sim_time_ms,
                "oracle": self.oracle,
            }
        )


@dataclass
class QualityReport:
    capture_id: str
    sharpness: float  # variance of Laplacian (gray, capture res)
    sharpness_ratio: float | None  # sharpness / expected sharpness of a clean capture at this view
    mean_luma: float  # 0..255
    clipped_frac: float  # fraction of pixels >= 250
    crushed_frac: float  # fraction of pixels <= 5
    noise_sigma: float  # estimated noise std on 0..255 scale
    glare_frac: float  # fraction of frame covered by specular glare blobs
    glare_boxes: list[list[int]]  # [x, y, w, h] in capture px
    issues: list[str]  # subset of QUALITY_ISSUES
    suggested_exposure_delta: float  # EV change that would bring mean luma to target
    ok: bool  # no blocking issues
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return _round(asdict(self))


@dataclass
class Region:
    region_id: str
    capture_id: str
    source: str  # "fast" | "deep" | "zoom"
    bbox_capture: tuple[int, int, int, int]  # x, y, w, h in capture px
    bbox_scene: tuple[float, float, float, float]  # x, y, w, h in scene px
    peak: float  # peak value of the anomaly map inside the region
    mean: float  # mean anomaly value inside the region
    area_px: int  # area in capture px
    area_scene_px: float  # area converted to scene px^2

    def center_norm(self, scene_size: tuple[int, int]) -> tuple[float, float]:
        x, y, w, h = self.bbox_scene
        return (x + w / 2) / scene_size[0], (y + h / 2) / scene_size[1]

    def to_dict(self) -> dict:
        return _round(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "Region":
        return cls(
            region_id=d["region_id"],
            capture_id=d["capture_id"],
            source=d["source"],
            bbox_capture=tuple(int(v) for v in d["bbox_capture"]),
            bbox_scene=tuple(float(v) for v in d["bbox_scene"]),
            peak=float(d["peak"]),
            mean=float(d["mean"]),
            area_px=int(d["area_px"]),
            area_scene_px=float(d["area_scene_px"]),
        )


@dataclass
class AnomalyResult:
    capture_id: str
    detector: str  # "fast" | "deep"
    score: float  # raw image-level score (detector specific)
    prob: float  # calibrated P(anomalous) in [0, 1]
    regions: list[Region]
    reliable: bool  # False when registration failed / valid area too small / bad input
    registration: dict  # {"method", "ok", "inliers", "inlier_ratio", "ncc", "dx", "dy", "rot_deg", "scale"}
    latency_ms: float
    notes: list[str] = field(default_factory=list)
    anomaly_map: np.ndarray | None = None  # float32, capture size (excluded from to_dict)
    valid_mask: np.ndarray | None = None  # uint8 0/255, capture size (excluded from to_dict)

    def to_dict(self) -> dict:
        return _round(
            {
                "capture_id": self.capture_id,
                "detector": self.detector,
                "score": self.score,
                "prob": self.prob,
                "regions": [r.to_dict() for r in self.regions],
                "reliable": self.reliable,
                "registration": self.registration,
                "latency_ms": self.latency_ms,
                "notes": list(self.notes),
            }
        )


@dataclass
class Decision:
    """One planner decision: which tool to call next and why."""

    step: int
    tool: str
    args: dict
    rationale: str  # human-readable link between visual evidence and this action
    evidence: list[str] = field(default_factory=list)  # ids of captures/results/regions that triggered it
    planner: str = "rule"  # "rule" | "bedrock:<model-id>" | "mcp-client" | "policy"

    def to_dict(self) -> dict:
        return _round(asdict(self))


@dataclass
class Step:
    step: int
    decision: Decision
    observation: dict  # JSON summary returned by the tool
    artifacts: list[str] = field(default_factory=list)  # artifact ids (JPEG images) produced by the tool
    latency_ms: float = 0.0  # compute time of the tool call
    sim_time_ms: float = 0.0  # simulated physical time (camera motion, AF, exposure)
    overridden_by_policy: bool = False  # True when the human-control policy replaced the planner's decision

    def to_dict(self) -> dict:
        return _round(
            {
                "step": self.step,
                "decision": self.decision.to_dict(),
                "observation": self.observation,
                "artifacts": list(self.artifacts),
                "latency_ms": self.latency_ms,
                "sim_time_ms": self.sim_time_ms,
                "overridden_by_policy": self.overridden_by_policy,
            }
        )


@dataclass
class Verdict:
    label: str  # VerdictLabel
    confidence: float  # 0..1 confidence in the label
    rationale: str
    defect_regions: list[Region] = field(default_factory=list)
    action: str = "hold_for_review"  # RoutingAction executed on the line
    review_id: str | None = None
    prob_anomalous: float | None = None  # final fused P(anomalous)

    def to_dict(self) -> dict:
        return _round(
            {
                "label": self.label,
                "confidence": self.confidence,
                "rationale": self.rationale,
                "defect_regions": [r.to_dict() for r in self.defect_regions],
                "action": self.action,
                "review_id": self.review_id,
                "prob_anomalous": self.prob_anomalous,
            }
        )


@dataclass
class PartSpec:
    """A part presented to the inspection cell."""

    part_id: str
    category: str
    image_path: str  # full-resolution scene image
    label: int | None = None  # ground truth: 0 normal, 1 anomalous, None unknown
    mask_path: str | None = None  # ground-truth defect mask (scene frame), if known
    conditions: NuisanceConditions = field(default_factory=NuisanceConditions)

    def to_dict(self) -> dict:
        return _round(
            {
                "part_id": self.part_id,
                "category": self.category,
                "image_path": self.image_path,
                "label": self.label,
                "mask_path": self.mask_path,
                "conditions": self.conditions.to_dict(),
            }
        )


@dataclass
class Trace:
    trace_id: str
    part_id: str
    category: str
    planner: str
    started_at: str  # ISO-8601 UTC
    steps: list[Step] = field(default_factory=list)
    verdict: Verdict | None = None
    budget: dict = field(default_factory=dict)
    totals: dict = field(default_factory=dict)  # n_steps, n_captures, compute_ms, sim_time_ms, tool_counts, ...
    ground_truth: dict | None = None  # {"label": 0|1} when known
    conditions: dict | None = None  # hidden nuisance conditions (revealed only in traces for analysis)
    cv_info: dict = field(default_factory=dict)  # OpenCV version/build/arch (see vision.cvinfo)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return _round(
            {
                "trace_id": self.trace_id,
                "part_id": self.part_id,
                "category": self.category,
                "planner": self.planner,
                "started_at": self.started_at,
                "steps": [s.to_dict() for s in self.steps],
                "verdict": self.verdict.to_dict() if self.verdict else None,
                "budget": self.budget,
                "totals": self.totals,
                "ground_truth": self.ground_truth,
                "conditions": self.conditions,
                "cv_info": self.cv_info,
                "errors": list(self.errors),
            }
        )
