"""Inspection session: the agent's working memory for one part.

Holds captures, quality reports, detector results, candidate regions, zoom inspections,
measurements, JPEG artifacts, budget counters and the evidence-fusion rule
(:meth:`InspectionSession.fused_prob`). Tools (``agent/tools.py``) mutate a session; planners read
:meth:`InspectionSession.planner_state` (compact JSON, never arrays).

Vision functions are reached through a :class:`VisionBackend` so the agent can run against the
real OpenCV stack (default) or lightweight fakes in tests.

Evidence-fusion rule (documented here, used by policy, planners and verdicts)
-----------------------------------------------------------------------------
Let ``R`` be the detector results on *full views* (zoom <= 1.05, fast or deep). Only reliable ones
are eligible (if no result is reliable at all, every full-view result is eligible and the fused
value is flagged ``reliable=False``). Every region ``j`` of a result ``r`` has a share
``s_j = w_j / sum(w)`` with ``w_j = peak_j * sqrt(area_j)`` (the region ranking score).

A zoom inspection of a region ends ``confirmed`` (reliable, no *blocking* quality issue, zoom
``p >= 0.60`` and the anomaly is at the candidate location), ``refuted`` (reliable, strictly good
quality, zoom ``p < 0.30``) or ``inconclusive``. Moderate noise (``vision.quality.blocking_quality_issues``
tolerates ``noisy`` up to 14) may confirm but never refute: noise lowers sensitivity, so a noisy zoom
cannot vouch for a clean part. Overlapping regions of the same source inherit conclusive statuses.

    contrib(r) = p_r * (1 - (1 - LAMBDA) * S_same(r) - (1 - LAMBDA_X) * S_cross(r))   LAMBDA = 0.20, LAMBDA_X = 0.50
    positive zoom evidence z: a *confirmed* inspection (see above)  ->  contrib(z) = p_z
    fused = max(contrib over eligible full-view results and positive zoom evidence)

``S_same``/``S_cross`` sum the shares of the refuted regions of ``r`` whose refuting zoom ran the same /
a different detector family than ``r`` (inspect_region always runs the fast detector, so a deep-only
detection refuted by a fast zoom is a cross-detector refutation).

Why this rule: fast and deep detectors look at the *same* pixels, so their errors are strongly
correlated; noisy-OR would double count that evidence and inflate false rejects, while ``max`` is
conservative for escapes (any reliable detector that is sure keeps the part out of PASS). A
refutation removes the share of the image-level evidence that the refuted region explained but
keeps ``LAMBDA`` of it, because a single zoom view can miss an anomaly the full view saw (residual
mis-localisation, the anomaly lying just outside the zoom window). That argument (correlated errors)
holds when a detector re-checks its own candidate; a zoom of the *other* detector may simply be blind
to the defect (the agent escalated to deep because fast could not see it), so cross-detector
refutations keep ``LAMBDA_X`` = 0.5: one fast zoom cannot move a mid-band deep detection (p <= 0.75)
into the auto-PASS band. A confirmed zoom raises the fused probability through its own
(high-resolution) calibrated ``p``.

Unlocalized zoom alarms (guard auto-PASS)
-----------------------------------------
A reliable zoom result with ``p >= 0.60`` whose anomaly is *not* at the inspected candidate's location
(inspection ``inconclusive``) is not confirmation of that candidate and does not enter the fusion, but
it is evidence that something on the part looks anomalous: :meth:`InspectionSession.unlocalized_zoom_alarms`
lists it, its own regions become new zoom candidates for the rule planner, and the policy refuses an
automatic PASS until those regions are resolved (refuted, or confirmed - which then drives a FAIL that
reports the zoom region, not the candidate the zoom showed clean).

Contested evidence (guards auto-PASS)
-------------------------------------
A full-view result that is *excluded* from the fusion (unreliable while other evidence is reliable)
still carries information when its registration succeeded: it is unreliable for another reason
(too small a valid area, an image-quality issue) but its map is aligned. When such a result calls
the part anomalous (``p >= CONTESTED_ALARM_PROB``) with at least one region no zoom has refuted (or
with no region at all), while a reliable result on the *same capture* is what makes the fused
probability low, two detectors looked at the same pixels and disagree - typically the fast detector
sees a small defect that the alignment-free deep detector (coarser, stride-8 features) misses. Such
a result *contests* a PASS: :meth:`InspectionSession.contested_results` lists it, the rule planner
zooms into its regions to resolve the conflict, and the human-control policy refuses an automatic
PASS while it stands (the part goes to a human instead of escaping silently). Not contested:
results that failed registration (their maps carry no information) and results superseded by
reliable evidence from another, better capture (e.g. a re-capture after a glare fix).
"""

from __future__ import annotations

import inspect
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

from inspectagent.config import CameraConfig, PolicyConfig, QualityConfig, Settings, get_settings
from inspectagent.models_io import CategoryBundle
from inspectagent.vision.quality import blocking_quality_issues
from inspectagent.types import (
    AnomalyResult,
    Capture,
    CaptureSettings,
    NuisanceConditions,
    PartSpec,
    QualityReport,
    Region,
    View,
    _round,
)

# ---------------------------------------------------------------------------------------------
# Evidence / inspection constants (agent-level; detector thresholds live in config.py)
# ---------------------------------------------------------------------------------------------
FULL_VIEW_MAX_ZOOM = 1.05  # a view covers the whole part only at zoom ~1
ZOOM_CONFIRM_PROB = 0.60  # zoom p at/above this (reliable, good quality, localized) confirms a candidate
ZOOM_REFUTE_PROB = 0.30  # zoom p below this (reliable, good quality) refutes a candidate
REFUTATION_RESIDUAL = 0.20  # LAMBDA: share of refuted evidence that is kept (zoom can miss things)
CROSS_REFUTATION_RESIDUAL = 0.50  # LAMBDA when the refuting zoom ran a different detector family than the evidence
CONTESTED_ALARM_PROB = 0.50  # aligned-but-degraded full-view result at/above this contests an auto-PASS
LOCALIZATION_PEAK_FRAC = 0.60  # candidate area must hold >= this fraction of the zoom-map maximum
REGION_MATCH_MARGIN_FRAC = 0.03  # box-matching slack as a fraction of the scene diagonal
ARTIFACT_LONG_SIDE = 768
ARTIFACT_JPEG_QUALITY = 85

COUNTER_KEYS = (
    "steps",
    "captures",
    "corrective_captures",
    "fast_calls",
    "deep_calls",
    "zoom_inspections",
    "measurements",
    "reviews",
    "errors",
)


class SessionError(ValueError):
    """Invalid request against the session state (unknown id, unavailable detector...)."""


def zoom_bucket(zoom: float) -> str:
    """Calibration bucket of a zoom level: z1 (<1.5), z2 (<2.5), z3 (<3.5), z4."""
    if zoom < 1.5:
        return "z1"
    if zoom < 2.5:
        return "z2"
    if zoom < 3.5:
        return "z3"
    return "z4"


# ---------------------------------------------------------------------------------------------
# Vision backend (dependency injection point)
# ---------------------------------------------------------------------------------------------
def _local_imread(path: str) -> np.ndarray:
    """Unicode-safe image read (np.fromfile + imdecode), used when vision.imageio is unavailable."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode image: {path}")
    return img


def _to_bgr_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def basic_heatmap_overlay(bgr: np.ndarray, amap: np.ndarray, vmax: float | None = None, alpha: float = 0.45) -> np.ndarray:
    """Minimal INFERNO heat-map overlay (used only when ``vision.overlay`` is not importable)."""
    base = _to_bgr_u8(bgr)
    a = np.nan_to_num(amap.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if a.shape[:2] != base.shape[:2]:
        a = cv2.resize(a, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    top = float(vmax) if vmax else max(float(a.max()), 1e-6)
    norm = np.clip(a / top, 0.0, 1.0)
    col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    blend = cv2.addWeighted(base, 1.0 - alpha, col, alpha, 0.0)
    mask = (norm > 0.15)[..., None]
    return np.where(mask, blend, base).astype(np.uint8)


def basic_draw_regions(bgr: np.ndarray, regions: list[Region], color: tuple[int, int, int] = (0, 0, 255), label: bool = True) -> np.ndarray:
    """Minimal region boxes (fallback renderer). Text is drawn on uint8 only (OpenCV 5 rule)."""
    out = _to_bgr_u8(bgr).copy()
    for r in regions:
        x, y, w, h = (int(v) for v in r.bbox_capture)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        if label:
            cv2.putText(out, r.region_id, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def basic_draw_quality(bgr: np.ndarray, q: QualityReport) -> np.ndarray:
    """Minimal quality overlay: glare boxes and issue badges (fallback renderer)."""
    out = _to_bgr_u8(bgr).copy()
    for x, y, w, h in q.glare_boxes:
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)
    text = "OK" if q.ok else ",".join(q.issues)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0) if q.ok else (0, 0, 255), 2, cv2.LINE_AA)
    return out


@dataclass
class VisionBackend:
    """The vision functions the agent uses; swap for fakes in tests."""

    make_camera: Callable[[np.ndarray, NuisanceConditions, CameraConfig, int], Any]
    assess_quality: Callable[[Capture, float | None, QualityConfig], QualityReport]
    measure_region: Callable[[Region, Capture, np.ndarray | None, float], dict]
    heatmap_overlay: Callable[..., np.ndarray] = basic_heatmap_overlay
    draw_regions: Callable[..., np.ndarray] = basic_draw_regions
    draw_quality: Callable[..., np.ndarray] = basic_draw_quality
    imread: Callable[[str], np.ndarray] = _local_imread


def default_backend() -> VisionBackend:
    """Backend wired to ``inspectagent.vision`` (camera simulator, quality, regions, overlay, imageio)."""
    from inspectagent.vision.camera import VirtualCamera
    from inspectagent.vision.quality import assess_quality
    from inspectagent.vision.regions import measure_region

    def make_camera(scene: np.ndarray, cond: NuisanceConditions, cfg: CameraConfig, seed: int) -> Any:
        return VirtualCamera(scene, cond, cfg, seed=seed)

    backend = VisionBackend(make_camera=make_camera, assess_quality=assess_quality, measure_region=measure_region)
    try:
        from inspectagent.vision import overlay

        backend.heatmap_overlay = overlay.heatmap_overlay
        backend.draw_regions = overlay.draw_regions
        backend.draw_quality = overlay.draw_quality
    except ImportError:  # keep the minimal renderers above
        pass
    try:
        from inspectagent.vision.imageio import imread

        backend.imread = imread
    except ImportError:
        pass
    return backend


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------
@dataclass
class ResultInfo:
    """Book-keeping for one detector result (``det_N``)."""

    result_id: str
    capture_id: str
    detector: str  # fast | deep
    view_kind: str  # full | zoom
    parent_region: str | None = None  # region being inspected (inspect_region) or None
    region_ids: list[str] = field(default_factory=list)


@dataclass
class Inspection:
    """Outcome of one ``inspect_region`` call."""

    region_id: str
    capture_id: str
    quality_id: str
    result_id: str
    zoom: float
    prob: float
    reliable: bool
    quality_ok: bool
    quality_issues: list[str]
    region_peak: float
    map_max: float
    overlapping_regions: list[str]
    status: str  # confirmed | refuted | inconclusive
    reason: str

    @property
    def confirmed(self) -> bool:
        return self.status == "confirmed"

    def to_dict(self) -> dict:
        return _round(asdict(self))


def classify_inspection(
    prob: float,
    reliable: bool,
    quality_ok: bool,
    localized: bool,
    quality_issues: list[str] | None = None,
    blocking: list[str] | None = None,
) -> tuple[str, str]:
    """(status, reason) of a zoom inspection; see the module docstring for the rule.

    ``blocking`` are the quality issues that invalidate the zoom view
    (``vision.quality.blocking_quality_issues``; default: every issue when ``quality_ok`` is False).
    Confirmation needs no blocking issue (moderate noise only lowers sensitivity, it does not create
    alarms); refutation needs *strictly* good quality, because a refutation lowers the fused probability
    and a noise-desensitised zoom must not let a defect escape.
    """
    if blocking is None:
        blocking = [] if quality_ok else list(quality_issues or ["quality"])
    if blocking:
        return "inconclusive", f"zoom quality issues {quality_issues or blocking}"
    if not reliable:
        return "inconclusive", "zoom result unreliable (registration/valid area)"
    if prob >= ZOOM_CONFIRM_PROB:
        if localized:
            return "confirmed", f"zoom p={prob:.3f}>={ZOOM_CONFIRM_PROB:.2f} at the candidate location"
        return "inconclusive", f"zoom p={prob:.3f}>={ZOOM_CONFIRM_PROB:.2f} but the anomaly is not at the candidate location"
    if prob < ZOOM_REFUTE_PROB:
        if quality_ok:
            return "refuted", f"zoom p={prob:.3f}<{ZOOM_REFUTE_PROB:.2f} with good quality"
        return (
            "inconclusive",
            f"zoom p={prob:.3f}<{ZOOM_REFUTE_PROB:.2f} but tolerated quality issues {quality_issues or []} lower the "
            "sensitivity: a degraded zoom cannot refute",
        )
    return "inconclusive", f"zoom p={prob:.3f} in [{ZOOM_REFUTE_PROB:.2f}, {ZOOM_CONFIRM_PROB:.2f})"


def region_weight(r: Region) -> float:
    """Ranking score of a region (same ordering as ``regions.extract_regions``)."""
    return max(float(r.peak), 0.0) * math.sqrt(max(float(r.area_px), 0.0))


def _accepts_keyword(fn: Callable[..., Any], name: str) -> bool:
    """True when ``fn`` accepts keyword ``name`` (or ``**kwargs``); fakes may implement older signatures."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is inspect.Parameter.VAR_KEYWORD for p in params)


def _uniq(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ---------------------------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------------------------
class InspectionSession:
    """State of one inspection (one part presented to the cell)."""

    def __init__(
        self,
        part: PartSpec,
        bundle: CategoryBundle,
        settings: Settings | None = None,
        seed: int = 0,
        scene: np.ndarray | None = None,
        *,
        backend: VisionBackend | None = None,
        session_id: str | None = None,
    ) -> None:
        self.settings: Settings = settings or get_settings()
        self.policy_cfg: PolicyConfig = self.settings.policy  # the loop replaces it with the active policy's cfg
        self.part = part
        self.bundle = bundle
        self.seed = int(seed)
        self.session_id: str = session_id or uuid.uuid4().hex[:12]
        self.vision: VisionBackend = backend or default_backend()
        self.scene: np.ndarray = scene if scene is not None else self.vision.imread(part.image_path)
        self.camera = self.vision.make_camera(self.scene, part.conditions, self.settings.camera, self.seed)

        self.captures: dict[str, Capture] = {}
        self.quality: dict[str, QualityReport] = {}  # capture_id -> report
        self.quality_ids: dict[str, str] = {}  # capture_id -> "q_N"
        self.results: dict[str, AnomalyResult] = {}  # "det_N" -> result
        self.result_info: dict[str, ResultInfo] = {}
        self.regions: dict[str, Region] = {}
        self.region_result: dict[str, str] = {}  # region_id -> det id
        self.inspections: dict[str, list[Inspection]] = {}  # region_id -> inspections
        self.measurements: dict[str, dict] = {}
        self.artifacts: dict[str, bytes] = {}
        self.artifact_kinds: dict[str, str] = {}
        self.proposed: dict | None = None
        self.review: dict | None = None

        self.sim_time_ms: float = 0.0
        self.compute_ms: float = 0.0
        self.planner_ms: float = 0.0
        self.render_ms: float = 0.0  # simulator rendering time (excluded from compute: a real camera does it)
        self.last_tool_ms: float = 0.0
        self.counters: dict[str, int] = {k: 0 for k in COUNTER_KEYS}
        self.history: list[dict] = []
        self.last_observation: dict | None = None
        self.last_settings: CaptureSettings | None = None
        self.planner_memory: dict[str, Any] = {}  # free-form per-planner memory (e.g. LLM conversation)
        self._ids: dict[str, int] = {}

    # ------------------------------------------------------------------ ids & artifacts
    def new_id(self, kind: str) -> str:
        """Deterministic per-session id: ``cap_1``, ``q_1``, ``det_1``, ``art_1``, ``rev_1`` ..."""
        n = self._ids.get(kind, 0) + 1
        self._ids[kind] = n
        return f"{kind}_{n}"

    def add_artifact(self, img: np.ndarray, kind: str) -> str:
        """Store ``img`` as a JPEG artifact (long side <= 768 px) and return its id."""
        im = _to_bgr_u8(img)
        h, w = im.shape[:2]
        if max(h, w) > ARTIFACT_LONG_SIDE:
            s = ARTIFACT_LONG_SIDE / max(h, w)
            im = cv2.resize(im, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), ARTIFACT_JPEG_QUALITY])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        aid = self.new_id("art")
        self.artifacts[aid] = buf.tobytes()
        self.artifact_kinds[aid] = kind
        return aid

    # ------------------------------------------------------------------ geometry helpers
    @property
    def scene_size(self) -> tuple[int, int]:
        ss = getattr(self.camera, "scene_size", None)
        if ss is not None:
            return int(ss[0]), int(ss[1])
        h, w = self.scene.shape[:2]
        return w, h

    def is_full_view(self, capture_id: str) -> bool:
        return float(self.captures[capture_id].settings.view.zoom) <= FULL_VIEW_MAX_ZOOM

    def _match_margin(self) -> float:
        w, h = self.scene_size
        return REGION_MATCH_MARGIN_FRAC * math.hypot(w, h)

    def boxes_match(self, a: tuple[float, float, float, float], b: tuple[float, float, float, float], extra: float = 0.0) -> bool:
        """True when two scene boxes overlap after expanding by the matching margin."""
        m = self._match_margin() + extra
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return (ax - m < bx + bw) and (bx - m < ax + aw) and (ay - m < by + bh) and (by - m < ay + ah)

    # ------------------------------------------------------------------ requirements
    def require_capture(self, capture_id: str) -> Capture:
        if capture_id not in self.captures:
            raise SessionError(f"unknown capture_id {capture_id!r}; known: {list(self.captures)[-8:]}")
        return self.captures[capture_id]

    def require_region(self, region_id: str) -> Region:
        if region_id not in self.regions:
            raise SessionError(f"unknown region_id {region_id!r}; known: {list(self.regions)[-10:]}")
        return self.regions[region_id]

    @property
    def deep_available(self) -> bool:
        return getattr(self.bundle, "deep", None) is not None

    # ------------------------------------------------------------------ primitives used by tools
    def reference_sharpness(self, zoom: float) -> float | None:
        rs = getattr(self.bundle, "reference_sharpness", None) or {}
        v = rs.get(zoom_bucket(float(zoom)))
        return float(v) if v else None

    def reference_exposure(self, zoom: float) -> dict | None:
        """Exposure statistics of clean normal captures at this zoom bucket (``bundle.meta["reference_exposure"]``).

        ``{"mean_lin", "clipped_frac", "crushed_frac", ...}`` written by ``scripts/build_models.py``; None when the
        bundle has none (quality then uses the absolute exposure limits).
        """
        meta = getattr(self.bundle, "meta", None)
        table = meta.get("reference_exposure") if isinstance(meta, dict) else None
        if not isinstance(table, dict):
            return None
        v = table.get(zoom_bucket(float(zoom)))
        try:
            return dict(v) if isinstance(v, dict) and float(v.get("mean_lin", 0.0)) > 0.0 else None
        except (TypeError, ValueError):
            return None

    def clamp_settings(self, s: CaptureSettings) -> CaptureSettings:
        cc = self.settings.camera
        v = s.view
        view = View(
            cx=float(min(max(v.cx, 0.0), 1.0)),
            cy=float(min(max(v.cy, 0.0), 1.0)),
            zoom=float(min(max(v.zoom, 1.0), cc.max_zoom)),
        )
        ev = float(min(max(s.exposure_ev, cc.min_ev), cc.max_ev))
        light = s.light if s.light in ("ring", "diffuse", "low_angle") else "ring"
        return CaptureSettings(view=view, exposure_ev=ev, light=light, autofocus=bool(s.autofocus))

    def estimate_capture_ms(self, settings: CaptureSettings) -> float:
        """Simulated time of a capture with ``settings`` from the current camera state."""
        return float(self.camera.estimate_time_ms(self.last_settings, self.clamp_settings(settings)))

    def take_capture(self, settings: CaptureSettings) -> Capture:
        """Capture with the (clamped) settings; updates sim time, counters and camera state."""
        s = self.clamp_settings(settings)
        cid = self.new_id("cap")
        t_sim = self.estimate_capture_ms(s)
        t0 = time.perf_counter()
        cap = self.camera.capture(s, cid, len(self.captures))
        self.render_ms += (time.perf_counter() - t0) * 1000.0
        cap.sim_time_ms = t_sim
        self.captures[cid] = cap
        self.sim_time_ms += t_sim
        self.counters["captures"] += 1
        self.last_settings = s
        return cap

    def run_quality(self, capture_id: str) -> tuple[str, QualityReport]:
        cap = self.require_capture(capture_id)
        zoom = cap.settings.view.zoom
        ref_exp = self.reference_exposure(zoom)
        if ref_exp is not None and _accepts_keyword(self.vision.assess_quality, "reference_exposure"):
            q = self.vision.assess_quality(cap, self.reference_sharpness(zoom), self.settings.quality, reference_exposure=ref_exp)
        else:
            q = self.vision.assess_quality(cap, self.reference_sharpness(zoom), self.settings.quality)
        qid = self.new_id("q")
        self.quality[capture_id] = q
        self.quality_ids[capture_id] = qid
        return qid, q

    def run_detector(self, capture_id: str, detector: str = "fast", parent_region: str | None = None) -> str:
        """Run the fast or deep detector on a capture; registers result + regions; returns ``det_N``."""
        cap = self.require_capture(capture_id)
        if detector not in ("fast", "deep"):
            raise SessionError(f"unknown detector {detector!r}")
        if detector == "deep" and not self.deep_available:
            raise SessionError(f"deep detector not available for category {self.part.category!r}")
        det_id = self.new_id("det")
        n = det_id.split("_", 1)[1]
        prefix = f"reg_{n}_"
        if detector == "deep":
            self.counters["deep_calls"] += 1
            res = self.bundle.deep.predict(cap, id_prefix=prefix, quality=self.quality.get(capture_id))
        else:
            self.counters["fast_calls"] += 1
            res = self.bundle.stat.predict(cap, quality=self.quality.get(capture_id), id_prefix=prefix)
        view_kind = "full" if self.is_full_view(capture_id) else "zoom"
        res.capture_id = capture_id
        ids = []
        for k, r in enumerate(res.regions, start=1):
            r.region_id = f"{prefix}{k}"  # deterministic ids whatever the detector produced
            r.capture_id = capture_id
            r.source = "zoom" if view_kind == "zoom" else detector
            self.regions[r.region_id] = r
            self.region_result[r.region_id] = det_id
            ids.append(r.region_id)
        self.results[det_id] = res
        self.result_info[det_id] = ResultInfo(det_id, capture_id, detector, view_kind, parent_region, ids)
        return det_id

    def localize(self, region_id: str, result_id: str) -> tuple[float, float, list[str]]:
        """(region_peak, map_max, overlapping zoom regions) of a candidate inside a zoom result."""
        region = self.require_region(region_id)
        res = self.results[result_id]
        cap = self.captures[res.capture_id]
        overlapping = [r.region_id for r in res.regions if self.boxes_match(region.bbox_scene, r.bbox_scene)]
        amap = res.anomaly_map
        if amap is None or amap.size == 0:
            peak = max((r.peak for r in res.regions if r.region_id in overlapping), default=0.0)
            return float(peak), float(max((r.peak for r in res.regions), default=0.0)), overlapping
        a = np.nan_to_num(amap.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        ah, aw = a.shape[:2]
        cw, ch = cap.size
        sx, sy = aw / float(cw), ah / float(ch)
        x, y, w, h = region.bbox_scene
        m = self._match_margin() + 0.5 * max(w, h)
        M = cap.scene_to_capture()
        p0 = M @ np.array([x - m, y - m, 1.0])
        p1 = M @ np.array([x + w + m, y + h + m, 1.0])
        x0 = int(np.clip(np.floor(min(p0[0], p1[0]) * sx), 0, aw))
        x1 = int(np.clip(np.ceil(max(p0[0], p1[0]) * sx), 0, aw))
        y0 = int(np.clip(np.floor(min(p0[1], p1[1]) * sy), 0, ah))
        y1 = int(np.clip(np.ceil(max(p0[1], p1[1]) * sy), 0, ah))
        peak = float(a[y0:y1, x0:x1].max()) if (x1 > x0 and y1 > y0) else 0.0
        valid = res.valid_mask
        if valid is not None and valid.shape[:2] == a.shape[:2] and bool((valid > 0).any()):
            map_max = float(a[valid > 0].max())
        else:
            map_max = float(a.max())
        return peak, map_max, overlapping

    def open_review(self, reason: str, proposed_label: str, confidence: float, evidence: list[str] | None = None, source: str = "planner") -> str:
        """Create (or replace) the human-review ticket of this session; returns its id."""
        rid = f"rev_{self.session_id}_{self.new_id('rev').split('_', 1)[1]}"
        p, ev = self.fused_prob()
        self.review = {
            "review_id": rid,
            "status": "pending",
            "reason": str(reason),
            "proposed_label": proposed_label,
            "confidence": float(min(max(confidence, 0.0), 1.0)),
            "fused_prob": p,
            "evidence": _uniq(list(evidence or []) + ev),
            "source": source,
        }
        self.counters["reviews"] += 1
        return rid

    # ------------------------------------------------------------------ queries
    def result_ids(self, view: str | None = None, detector: str | None = None, capture_id: str | None = None) -> list[str]:
        out = []
        for rid, info in self.result_info.items():
            if view is not None and info.view_kind != view:
                continue
            if detector is not None and info.detector != detector:
                continue
            if capture_id is not None and info.capture_id != capture_id:
                continue
            out.append(rid)
        return out

    def best_full_view(self) -> str | None:
        """Best full-view capture: latest with ok quality, else fewest issues (ties -> latest)."""
        full = [cid for cid in self.captures if self.is_full_view(cid)]
        if not full:
            return None
        assessed = [cid for cid in full if cid in self.quality]
        if not assessed:
            return full[-1]
        ok = [cid for cid in assessed if self.quality[cid].ok]
        if ok:
            return ok[-1]
        return min(assessed, key=lambda c: (len(self.quality[c].issues), -self.captures[c].index))

    def last_good_settings(self) -> CaptureSettings:
        """Settings of the latest capture with ok quality (else the best-quality one, else defaults)."""
        assessed = [cid for cid in self.captures if cid in self.quality]
        if not assessed:
            return self.last_settings or CaptureSettings()
        ok = [cid for cid in assessed if self.quality[cid].ok]
        if ok:
            return self.captures[ok[-1]].settings
        best = min(assessed, key=lambda c: (len(self.quality[c].issues), -self.captures[c].index))
        return self.captures[best].settings

    def region_status_detail(self, region_id: str) -> tuple[str, list[str]]:
        """(status, ids of the zoom results that decided it).

        Own inspections first; conclusive inspections of overlapping regions of the same source are
        inherited (``confirmed`` beats ``refuted``); ``inconclusive`` is never inherited.
        """
        region = self.regions.get(region_id)
        own = self.inspections.get(region_id, [])
        confirmed = [i.result_id for i in own if i.status == "confirmed"]
        refuted = [i.result_id for i in own if i.status == "refuted"]
        if region is not None:
            for other_id, lst in self.inspections.items():
                if other_id == region_id:
                    continue
                other = self.regions.get(other_id)
                if other is None or other.source != region.source:
                    continue
                if not self.boxes_match(region.bbox_scene, other.bbox_scene):
                    continue
                confirmed += [i.result_id for i in lst if i.status == "confirmed"]
                refuted += [i.result_id for i in lst if i.status == "refuted"]
        if confirmed:
            return "confirmed", _uniq(confirmed)
        if refuted:
            return "refuted", _uniq(refuted)
        if own:
            return "inconclusive", [i.result_id for i in own]
        return "uninspected", []

    def region_status(self, region_id: str) -> str:
        return self.region_status_detail(region_id)[0]

    def location_inspected(self, region_id: str) -> bool:
        """True when another region at the same scene location (any source) was already zoomed into."""
        region = self.regions.get(region_id)
        if region is None:
            return False
        for other_id, lst in self.inspections.items():
            if other_id == region_id or not lst:
                continue
            other = self.regions.get(other_id)
            if other is not None and self.boxes_match(region.bbox_scene, other.bbox_scene):
                return True
        return False

    def inspection_for_result(self, result_id: str) -> Inspection | None:
        """The ``inspect_region`` record whose zoom detector result is ``result_id`` (None for direct detections)."""
        for lst in self.inspections.values():
            for insp in lst:
                if insp.result_id == result_id:
                    return insp
        return None

    def _zoom_alarm_candidate(self, result_id: str) -> bool:
        """Zoom result that is reliable, has no blocking quality issue and p >= ZOOM_CONFIRM_PROB."""
        info = self.result_info[result_id]
        res = self.results[result_id]
        if info.view_kind != "zoom" or not res.reliable or res.prob < ZOOM_CONFIRM_PROB:
            return False
        q = self.quality.get(info.capture_id)
        return q is None or not blocking_quality_issues(q)

    def zoom_positive(self, result_id: str) -> bool:
        """Positive zoom evidence: a *confirmed* zoom inspection (reliable, no blocking quality issue,
        p >= ZOOM_CONFIRM_PROB and the anomaly at the inspected candidate's location; SPEC 3.1 "confirmed
        zoom results"). A high zoom p elsewhere in the window is an unlocalized alarm, not confirmation."""
        if not self._zoom_alarm_candidate(result_id):
            return False
        insp = self.inspection_for_result(result_id)
        return insp is not None and insp.status == "confirmed"

    def unlocalized_zoom_alarms(self) -> list[str]:
        """Zoom results that flag the part (reliable, no blocking quality issue, p >= ZOOM_CONFIRM_PROB) without
        confirming the inspected candidate - the anomaly is elsewhere in the zoom window (or the zoom was a
        direct detection without a candidate). They block an automatic PASS until every one of their regions
        has been resolved by its own zoom inspection (refuted, or confirmed - a confirmation is positive
        evidence by itself); an alarm without regions stands until a human decides."""
        out: list[str] = []
        for rid in self.result_ids(view="zoom"):
            if not self._zoom_alarm_candidate(rid) or self.zoom_positive(rid):
                continue
            regs = self.results[rid].regions
            resolved = bool(regs) and all(
                any(i.status in ("refuted", "confirmed") for i in self.inspections.get(r.region_id, [])) for r in regs
            )
            if not resolved:
                out.append(rid)
        return out

    def unlocalized_alarm_regions(self) -> list[Region]:
        """Not yet inspected regions of :meth:`unlocalized_zoom_alarms` (new zoom candidates), ranked."""
        regs = [
            r
            for rid in self.unlocalized_zoom_alarms()
            for r in self.results[rid].regions
            if not self.inspections.get(r.region_id)
        ]
        return sorted(regs, key=lambda r: (-region_weight(r), r.region_id))

    def has_reliable_evidence(self, label: str = "PASS") -> bool:
        """PASS needs a reliable full-view result; FAIL also accepts positive zoom evidence."""
        if any(self.results[r].reliable for r in self.result_ids(view="full")):
            return True
        if label == "FAIL":
            return any(self.zoom_positive(r) for r in self.result_ids(view="zoom"))
        return False

    def _shares(self, regions: list[Region]) -> dict[str, float]:
        if not regions:
            return {}
        w = [region_weight(r) for r in regions]
        tot = sum(w)
        if tot <= 0:
            return {r.region_id: 1.0 / len(regions) for r in regions}
        return {r.region_id: wi / tot for r, wi in zip(regions, w)}

    def fusion_report(self) -> dict:
        """Full breakdown of the evidence-fusion rule (see module docstring)."""
        lam = REFUTATION_RESIDUAL
        full_ids = self.result_ids(view="full")
        zoom_ids = self.result_ids(view="zoom")
        reliable_full = [r for r in full_ids if self.results[r].reliable]
        positive_zoom = [r for r in zoom_ids if self.zoom_positive(r)]
        any_reliable = bool(reliable_full or positive_zoom)
        eligible = reliable_full if any_reliable else full_ids
        terms: list[dict] = []
        for rid in eligible:
            res = self.results[rid]
            det = self.result_info[rid].detector
            shares = self._shares(res.regions)
            share_same = share_cross = 0.0
            refuted_by: list[str] = []
            confirmed_by: list[str] = []
            for r in res.regions:
                st, by = self.region_status_detail(r.region_id)
                if st == "refuted":
                    refuted_by += by
                    if any(self.result_info[b].detector == det for b in by if b in self.result_info):
                        share_same += shares[r.region_id]
                    else:  # only another detector family looked: it may be blind to this defect
                        share_cross += shares[r.region_id]
                elif st == "confirmed":
                    confirmed_by += by
            share_same = min(share_same, 1.0)
            share_cross = min(share_cross, 1.0 - share_same)
            refuted_share = share_same + share_cross
            contrib = float(res.prob) * (1.0 - (1.0 - lam) * share_same - (1.0 - CROSS_REFUTATION_RESIDUAL) * share_cross)
            terms.append(
                {
                    "id": rid,
                    "kind": f"full/{det}",
                    "p": float(res.prob),
                    "reliable": bool(res.reliable),
                    "refuted_share": refuted_share,
                    "refuted_share_cross": share_cross,
                    "refuted_by": _uniq(refuted_by),
                    "confirmed_by": _uniq(confirmed_by),
                    "contrib": contrib,
                }
            )
        for rid in positive_zoom:
            res = self.results[rid]
            terms.append(
                {
                    "id": rid,
                    "kind": f"zoom/{self.result_info[rid].detector}",
                    "p": float(res.prob),
                    "reliable": True,
                    "refuted_share": 0.0,
                    "refuted_share_cross": 0.0,
                    "refuted_by": [],
                    "confirmed_by": [],
                    "contrib": float(res.prob),
                    "parent_region": self.result_info[rid].parent_region,
                }
            )
        if terms:
            best = max(terms, key=lambda t: t["contrib"])
            prob = best["contrib"]
            evidence = _uniq([best["id"]] + best["refuted_by"] + [t["id"] for t in terms if t is not best])
            parts = [f"{best['id']} ({best['kind']}) p={best['p']:.3f}"]
            same = best["refuted_share"] - best["refuted_share_cross"]
            if same > 0:
                parts.append(f"x (1-{1 - lam:.1f}*{same:.2f} refuted share by {','.join(best['refuted_by'])})")
            if best["refuted_share_cross"] > 0:
                parts.append(
                    f"x (1-{1 - CROSS_REFUTATION_RESIDUAL:.1f}*{best['refuted_share_cross']:.2f} share refuted by another "
                    f"detector family {','.join(best['refuted_by'])})"
                )
            others = [f"{t['id']}={t['contrib']:.3f}" for t in terms if t is not best]
            explain = f"fused p={prob:.3f} <- " + " ".join(parts) + (f"; other terms: {', '.join(others)}" if others else "")
            if not any_reliable:
                explain += " [no reliable evidence]"
        elif self.results:
            bid = max(self.results, key=lambda r: self.results[r].prob)
            prob = float(self.results[bid].prob)
            evidence = [bid]
            explain = f"fused p={prob:.3f} <- {bid} (no full-view or positive zoom evidence; raw max) [no reliable evidence]"
        else:
            prob, evidence = 0.5, []
            explain = "fused p=0.500 (no detector evidence yet: uninformative prior)"
        return {
            "prob": float(min(max(prob, 0.0), 1.0)),
            "evidence": evidence,
            "reliable": any_reliable,
            "terms": terms,
            "rule": (
                "max(full-view reliable results x (1-0.8*same-detector refuted share-0.5*cross-detector refuted share), "
                "confirmed zoom inspections p>=0.60)"
            ),
            "explain": explain,
        }

    def fused_prob(self) -> tuple[float, list[str]]:
        """Fused P(anomalous) and the evidence ids that determined it (argmax first)."""
        rep = self.fusion_report()
        return rep["prob"], rep["evidence"]

    def contested_results(self) -> list[str]:
        """Full-view results that contest an auto-PASS (see "Contested evidence" in the module docstring)."""
        terms = self.fusion_report()["terms"]
        eligible = {t["id"] for t in terms}
        reliable_caps = {self.results[t["id"]].capture_id for t in terms if t["reliable"] and t["id"] in self.results}
        out: list[str] = []
        for rid in self.result_ids(view="full"):
            res = self.results[rid]
            if rid in eligible or res.reliable or float(res.prob) < CONTESTED_ALARM_PROB:
                continue
            if not bool((res.registration or {}).get("ok")):
                continue  # misregistered: the map carries no information
            if res.capture_id not in reliable_caps:
                continue  # superseded by reliable evidence from another capture
            if not res.regions or any(self.region_status(r.region_id) != "refuted" for r in res.regions):
                out.append(rid)
        return out

    def _unreliable_reason(self, result_id: str) -> str:
        res = self.results[result_id]
        reasons = [n for n in res.notes if n.startswith(("valid area", "quality issues"))]
        if not reasons:
            q = self.quality.get(res.capture_id)
            if q is not None and not q.ok:
                reasons = [f"quality issues: {','.join(q.issues)}"]
        return "; ".join(reasons) or "unreliable"

    def contested_regions(self) -> list[Region]:
        """Unrefuted regions of :meth:`contested_results`, ranked by peak*sqrt(area) (ties by id)."""
        regs = [
            r for rid in self.contested_results() for r in self.results[rid].regions if self.region_status(r.region_id) != "refuted"
        ]
        return sorted(regs, key=lambda r: (-region_weight(r), r.region_id))

    def contested_note(self, result_ids: list[str] | None = None) -> str:
        """Human-readable summary of contested evidence (empty string when there is none)."""
        ids = self.contested_results() if result_ids is None else result_ids
        parts = []
        for rid in ids:
            res = self.results[rid]
            open_regs = [r.region_id for r in res.regions if self.region_status(r.region_id) != "refuted"]
            others = [
                o for o in self.result_ids(view="full", capture_id=res.capture_id) if o != rid and self.results[o].reliable
            ]
            vs = ", ".join(f"{o} ({self.result_info[o].detector}) p={float(self.results[o].prob):.3f}" for o in others)
            parts.append(
                f"{rid} ({self.result_info[rid].detector} on {res.capture_id}, registration ok, {self._unreliable_reason(rid)}) "
                f"p={float(res.prob):.3f} with "
                + (f"unrefuted regions {open_regs}" if open_regs else "no localized region")
                + (f" vs reliable {vs} on the same capture" if vs else "")
            )
        return "; ".join(parts)

    def full_view_regions(self, include_refuted: bool = False) -> list[Region]:
        """Regions of eligible full-view results, ranked by peak*sqrt(area) (ties by id)."""
        full_ids = self.result_ids(view="full")
        reliable = [r for r in full_ids if self.results[r].reliable]
        eligible = reliable if reliable else full_ids
        regs = [reg for rid in eligible for reg in self.results[rid].regions]
        if not include_refuted:
            regs = [r for r in regs if self.region_status(r.region_id) != "refuted"]
        return sorted(regs, key=lambda r: (-region_weight(r), r.region_id))

    def confirmed_regions(self) -> list[Region]:
        """Zoom-confirmed regions of any full-view result (a confirmation is reliable evidence by itself,
        so regions of contested, quality-degraded results count too) plus regions of zoom results that
        were confirmed by their *own* inspection (a candidate found by an unlocalized zoom alarm), ranked
        like ``full_view_regions``."""
        regs = [
            reg
            for rid in self.result_ids(view="full")
            for reg in self.results[rid].regions
            if self.region_status(reg.region_id) == "confirmed"
        ]
        regs += [
            reg
            for rid in self.result_ids(view="zoom")
            for reg in self.results[rid].regions
            if any(i.status == "confirmed" for i in self.inspections.get(reg.region_id, []))
        ]
        return sorted(regs, key=lambda r: (-region_weight(r), r.region_id))

    def defect_regions(self, region_ids: list[str] | None = None) -> list[Region]:
        """Regions to report on a FAIL / review verdict: requested ids, else confirmed regions, else the
        regions of unlocalized zoom alarms (where a zoom actually saw the anomaly) and then the strongest
        full-view candidates."""
        if region_ids:
            regs = [self.regions[r] for r in region_ids if r in self.regions]
            if regs:
                return regs
        conf = self.confirmed_regions()
        if conf:
            return conf
        alarm_regs = [r for rid in self.unlocalized_zoom_alarms() for r in self.results[rid].regions]
        alarm_regs = sorted(alarm_regs, key=lambda r: (-region_weight(r), r.region_id))
        out: list[Region] = []
        for r in alarm_regs + self.full_view_regions():
            if r.region_id not in {o.region_id for o in out}:
                out.append(r)
        return out[:3]

    # ------------------------------------------------------------------ planner view
    def budget_state(self) -> dict:
        pol = self.policy_cfg
        return {
            "max_steps": pol.max_steps,
            "max_captures": pol.max_captures,
            "max_deep_calls": pol.max_deep_calls,
            "max_sim_time_ms": pol.max_sim_time_ms,
            "steps_used": self.counters["steps"],
            "captures_used": self.counters["captures"],
            "deep_calls_used": self.counters["deep_calls"],
            "sim_time_ms": self.sim_time_ms,
            "steps_left": pol.max_steps - self.counters["steps"],
            "captures_left": pol.max_captures - self.counters["captures"],
            "deep_calls_left": pol.max_deep_calls - self.counters["deep_calls"],
            "sim_time_left_ms": pol.max_sim_time_ms - self.sim_time_ms,
        }

    def quality_summary(self, capture_id: str) -> dict | None:
        q = self.quality.get(capture_id)
        if q is None:
            return None
        return {
            "quality_id": self.quality_ids.get(capture_id),
            "ok": q.ok,
            "issues": list(q.issues),
            "sharpness": q.sharpness,
            "sharpness_ratio": q.sharpness_ratio,
            "mean_luma": q.mean_luma,
            "clipped_frac": q.clipped_frac,
            "crushed_frac": q.crushed_frac,
            "noise_sigma": q.noise_sigma,
            "glare_frac": q.glare_frac,
            "suggested_exposure_delta": q.suggested_exposure_delta,
        }

    def result_summary(self, result_id: str) -> dict:
        res = self.results[result_id]
        info = self.result_info[result_id]
        reg = res.registration or {}
        ss = self.scene_size
        return {
            "result_id": result_id,
            "capture_id": info.capture_id,
            "detector": info.detector,
            "view": info.view_kind,
            "parent_region": info.parent_region,
            "prob": float(res.prob),
            "score": float(res.score),
            "reliable": bool(res.reliable),
            "registration": {k: reg.get(k) for k in ("method", "ok", "ncc", "inliers", "inlier_ratio") if k in reg},
            "regions": [
                {
                    "region_id": r.region_id,
                    "peak": r.peak,
                    "mean": r.mean,
                    "area_px": r.area_px,
                    "bbox_scene": list(r.bbox_scene),
                    "center": list(r.center_norm(ss)),
                }
                for r in res.regions
            ],
            "notes": list(res.notes)[:6],
        }

    def planner_state(self) -> dict:
        """Compact JSON history for planners (no arrays, no images)."""
        pol = self.policy_cfg
        fusion = self.fusion_report()
        ss = self.scene_size
        caps = []
        for cid, cap in self.captures.items():
            s = cap.settings
            caps.append(
                {
                    "capture_id": cid,
                    "view": s.view.to_dict(),
                    "exposure_ev": s.exposure_ev,
                    "light": s.light,
                    "autofocus": s.autofocus,
                    "full_view": self.is_full_view(cid),
                    "sim_time_ms": cap.sim_time_ms,
                    "quality": self.quality_summary(cid),
                }
            )
        results = []
        for rid, info in self.result_info.items():
            res = self.results[rid]
            results.append(
                {
                    "result_id": rid,
                    "capture_id": info.capture_id,
                    "detector": info.detector,
                    "view": info.view_kind,
                    "prob": res.prob,
                    "score": res.score,
                    "reliable": res.reliable,
                    "registration_ok": (res.registration or {}).get("ok"),
                    "ncc": (res.registration or {}).get("ncc"),
                    "region_ids": list(info.region_ids),
                    "parent_region": info.parent_region,
                }
            )
        regions = []
        for rid, r in self.regions.items():
            regions.append(
                {
                    "region_id": rid,
                    "result_id": self.region_result.get(rid),
                    "source": r.source,
                    "peak": r.peak,
                    "area_px": r.area_px,
                    "center": list(r.center_norm(ss)),
                    "status": self.region_status(rid),
                    "inspections": len(self.inspections.get(rid, [])),
                    "measured": rid in self.measurements,
                }
            )
        meas = {
            k: {kk: v.get(kk) for kk in ("area_mm2", "length_mm", "width_mm", "type_guess") if kk in v}
            for k, v in self.measurements.items()
        }
        return _round(
            {
                "session_id": self.session_id,
                "part_id": self.part.part_id,
                "category": self.part.category,
                "budget": self.budget_state(),
                "thresholds": {
                    "auto_pass_max_prob": pol.auto_pass_max_prob,
                    "auto_fail_min_prob": pol.auto_fail_min_prob,
                    "zoom_confirm_prob": ZOOM_CONFIRM_PROB,
                    "zoom_refute_prob": ZOOM_REFUTE_PROB,
                    "contested_alarm_prob": CONTESTED_ALARM_PROB,
                    "quality": asdict(self.settings.quality),
                },
                "deep_available": self.deep_available,
                "captures": caps,
                "results": results,
                "regions": regions,
                "inspections": [i.to_dict() for lst in self.inspections.values() for i in lst],
                "measurements": meas,
                "fused_prob": fusion["prob"],
                "fused_evidence": fusion["evidence"],
                "fusion_explain": fusion["explain"],
                "contested_evidence": self.contested_note(),
                "unresolved_candidates": [
                    r.region_id for r in self.full_view_regions() if self.region_status(r.region_id) in ("uninspected", "inconclusive")
                ],
                "unlocalized_zoom_alarms": self.unlocalized_zoom_alarms(),
                "has_reliable_evidence": {
                    "PASS": self.has_reliable_evidence("PASS"),
                    "FAIL": self.has_reliable_evidence("FAIL"),
                },
                "proposed": self.proposed,
                "review": self.review,
                "errors": self.counters["errors"],
                "last_observation": self.last_observation,
                "history": self.history[-12:],
            }
        )
