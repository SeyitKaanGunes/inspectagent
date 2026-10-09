"""Lightweight fakes for agent-core tests (independent of detector quality).

* :class:`FakeCamera` renders a crop of the scene at a small sensor size (fast) with a simple
  brightness model; timing follows ``CameraConfig`` like the real simulator.
* :class:`FakeVision` builds a :class:`VisionBackend` whose ``assess_quality`` is an *oracle* over the
  hidden nuisance conditions + capture settings (glare under ring light, defocus without AF,
  exposure from ambient + EV + light gain, gain-induced noise) using ``QualityConfig`` thresholds.
* :class:`FakeStat` / :class:`FakeDeep` implement the detectors' ``predict``/``prob`` API from a
  list of :class:`Spot` (scene boxes with the probabilities a detector would report on full and
  zoom views), so planner/policy/loop behaviour can be tested deterministically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from inspectagent.agent.session import VisionBackend
from inspectagent.config import CameraConfig, QualityConfig
from inspectagent.models_io import CategoryBundle
from inspectagent.types import AnomalyResult, Capture, CaptureSettings, NuisanceConditions, PartSpec, QualityReport, Region, View

LIGHT_GAIN = {"ring": 1.0, "diffuse": 0.6, "low_angle": 0.5}
SCENE_SIZE = (600, 400)


def make_scene(seed: int = 0) -> np.ndarray:
    """Small synthetic 'board' (BGR uint8, 600x400)."""
    rng = np.random.default_rng(seed)
    w, h = SCENE_SIZE
    img = np.full((h, w, 3), (40, 90, 32), np.uint8)
    for _ in range(25):
        x, y = int(rng.integers(10, w - 60)), int(rng.integers(10, h - 40))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(10, 50)), y + int(rng.integers(5, 30))), (170, 190, 205), -1)
    return img


def preset_conditions(preset: str = "clean") -> NuisanceConditions:
    """Deterministic conditions mimicking the camera presets (strong, unambiguous degradations)."""
    if preset == "clean":
        return NuisanceConditions(noise_sigma=0.0, preset="clean")
    if preset == "glare":
        return NuisanceConditions(glare_strength=0.7, noise_sigma=1.0, preset="glare")
    if preset == "defocus":
        return NuisanceConditions(defocus_sigma=3.5, noise_sigma=1.0, preset="defocus")
    if preset == "dark":
        return NuisanceConditions(ambient_ev=-2.5, noise_sigma=1.0, preset="dark")
    if preset == "bright":
        return NuisanceConditions(ambient_ev=1.8, noise_sigma=1.0, preset="bright")
    if preset == "harsh":
        return NuisanceConditions(glare_strength=0.8, defocus_sigma=3.0, ambient_ev=-1.0, noise_sigma=2.0, preset="harsh")
    raise ValueError(preset)


class FakeCamera:
    """Deterministic camera stand-in with the VirtualCamera interface."""

    def __init__(self, scene: np.ndarray, conditions: NuisanceConditions, cfg: CameraConfig | None = None, seed: int = 0, sensor_long_side: int = 160) -> None:
        self.scene = scene
        self.conditions = conditions
        self.cfg = cfg or CameraConfig()
        self.seed = seed
        h, w = scene.shape[:2]
        self.scene_size = (w, h)
        s = sensor_long_side / max(w, h)
        self.sensor_size = (max(1, round(w * s)), max(1, round(h * s)))

    def view_window(self, view: View) -> tuple[float, float, float, float]:
        W, H = self.scene_size
        ww, wh = W / view.zoom, H / view.zoom
        x0 = min(max(view.cx * W - ww / 2, 0.0), W - ww)
        y0 = min(max(view.cy * H - wh / 2, 0.0), H - wh)
        return float(x0), float(y0), float(ww), float(wh)

    def _render(self, window: tuple[float, float, float, float], gain: float) -> np.ndarray:
        x0, y0, ww, wh = window
        crop = self.scene[int(y0) : int(math.ceil(y0 + wh)), int(x0) : int(math.ceil(x0 + ww))]
        img = cv2.resize(crop, self.sensor_size, interpolation=cv2.INTER_AREA).astype(np.float32)
        return np.clip(img * gain, 0, 255).astype(np.uint8)

    def capture(self, settings: CaptureSettings, capture_id: str, index: int) -> Capture:
        c = self.conditions
        eff = c.ambient_ev + settings.exposure_ev + math.log2(LIGHT_GAIN[settings.light])
        win = self.view_window(settings.view)
        img = self._render(win, 2.0 ** (eff / 2.2))
        return Capture(capture_id, index, settings, img, win, self.scene_size, self.estimate_time_ms(None, settings))

    def clean_capture(self, view: View, capture_id: str, index: int) -> Capture:
        win = self.view_window(view)
        return Capture(capture_id, index, CaptureSettings(view=view), self._render(win, 1.0), win, self.scene_size, 0.0, oracle=True)

    def estimate_time_ms(self, prev: CaptureSettings | None, settings: CaptureSettings) -> float:
        cfg = self.cfg
        p = prev or CaptureSettings()
        t = cfg.t_exposure + cfg.t_readout
        if settings.autofocus:
            t += cfg.t_autofocus
        if prev is not None and prev.light != settings.light:
            t += cfg.t_light_switch
        t += cfg.t_move_per_unit * math.hypot(settings.view.cx - p.view.cx, settings.view.cy - p.view.cy)
        t += cfg.t_zoom_per_step * abs(settings.view.zoom - p.view.zoom)
        return float(t)


class FakeVision:
    """Factory of a fake :class:`VisionBackend` (quality oracle + simple measurement)."""

    def __init__(self, sensor_long_side: int = 160, glare_min_zoom: float = 1.0) -> None:
        self.sensor_long_side = sensor_long_side
        self.glare_min_zoom = glare_min_zoom  # glare only visible at/above this zoom (zoom-specific specular spot)
        self.camera: FakeCamera | None = None
        self.quality_calls: list[str] = []

    def make_camera(self, scene: np.ndarray, cond: NuisanceConditions, cfg: CameraConfig, seed: int) -> FakeCamera:
        self.camera = FakeCamera(scene, cond, cfg, seed, self.sensor_long_side)
        return self.camera

    def assess_quality(self, capture: Capture, reference_sharpness: float | None = None, cfg: QualityConfig | None = None) -> QualityReport:
        assert self.camera is not None
        cfg = cfg or QualityConfig()
        c = self.camera.conditions
        s = capture.settings
        self.quality_calls.append(capture.capture_id)
        eff = c.ambient_ev + s.exposure_ev + math.log2(LIGHT_GAIN[s.light])
        target_lin = (cfg.target_luma / 255.0) ** 2.2
        mean_lin = min(target_lin * 2.0**eff, 1.0)
        mean_luma = 255.0 * mean_lin ** (1 / 2.2)
        glare = s.light == "ring" and c.glare_strength > 0 and s.view.zoom >= self.glare_min_zoom
        glare_frac = 0.02 + 0.02 * c.glare_strength if glare else 0.0
        clipped = 0.05 if (glare or mean_luma > 200) else 0.0
        crushed = 0.10 if mean_luma < 40 else 0.0
        sigma = c.defocus_sigma if not s.autofocus else 0.35
        ratio = 1.0 / (1.0 + 0.25 * sigma**2)
        noise = c.noise_sigma * 2.0 ** max(s.exposure_ev, 0.0)
        issues = []
        if ratio < cfg.min_sharpness_ratio:
            issues.append("blurred")
        if mean_luma < cfg.min_luma or crushed > cfg.max_crushed_frac:
            issues.append("underexposed")
        if mean_luma > cfg.max_luma or clipped > cfg.max_clipped_frac:
            issues.append("overexposed")
        if noise > cfg.max_noise_sigma:
            issues.append("noisy")
        if glare_frac > cfg.max_glare_frac:
            issues.append("glare")
        suggested = float(min(max(math.log2(target_lin / max(mean_lin, 1e-6)), -3.0), 3.0))
        return QualityReport(
            capture_id=capture.capture_id,
            sharpness=200.0 * ratio,
            sharpness_ratio=ratio,
            mean_luma=mean_luma,
            clipped_frac=clipped,
            crushed_frac=crushed,
            noise_sigma=noise,
            glare_frac=glare_frac,
            glare_boxes=[[10, 10, 20, 12]] if glare else [],
            issues=issues,
            suggested_exposure_delta=suggested,
            ok=not issues,
        )

    def measure_region(self, region: Region, capture: Capture, amap: np.ndarray | None, px_per_mm: float) -> dict:
        x, y, w, h = region.bbox_scene
        return {
            "area_mm2": region.area_scene_px / px_per_mm**2,
            "length_mm": max(w, h) / px_per_mm,
            "width_mm": min(w, h) / px_per_mm,
            "elongation": max(w, h) / max(min(w, h), 1e-6),
            "orientation_deg": 0.0,
            "contrast": float(region.peak),
            "type_guess": "spot",
        }

    def backend(self) -> VisionBackend:
        return VisionBackend(make_camera=self.make_camera, assess_quality=self.assess_quality, measure_region=self.measure_region)


@dataclass
class Spot:
    """A location the fake detectors react to: probabilities on full / zoom views (fast) and deep."""

    bbox: tuple[float, float, float, float]  # scene px (x, y, w, h)
    p_full: float
    p_zoom: float
    p_deep: float | None = None
    peak: float = 8.0


def _intersects(b: tuple[float, float, float, float], win: tuple[float, float, float, float]) -> bool:
    x, y, w, h = b
    x0, y0, ww, wh = win
    return x < x0 + ww and x0 < x + w and y < y0 + wh and y0 < y + h


def _make_result(capture: Capture, detector: str, prob: float, spots_p: list[tuple[Spot, float]], reliable: bool, reg_ok: bool, id_prefix: str, region_min_prob: float) -> AnomalyResult:
    w, h = capture.size
    M = capture.scene_to_capture()
    amap = np.zeros((h, w), np.float32)
    regions = []
    k = 0
    for sp, p in spots_p:
        if p < region_min_prob:
            continue
        x, y, bw, bh = sp.bbox
        p0 = M @ np.array([x, y, 1.0])
        p1 = M @ np.array([x + bw, y + bh, 1.0])
        cx0, cy0 = int(np.clip(p0[0], 0, w - 1)), int(np.clip(p0[1], 0, h - 1))
        cx1, cy1 = int(np.clip(math.ceil(p1[0]), cx0 + 1, w)), int(np.clip(math.ceil(p1[1]), cy0 + 1, h))
        peak = sp.peak * p
        amap[cy0:cy1, cx0:cx1] = np.maximum(amap[cy0:cy1, cx0:cx1], peak)
        k += 1
        area = (cx1 - cx0) * (cy1 - cy0)
        regions.append(
            Region(
                region_id=f"{id_prefix}{k}",
                capture_id=capture.capture_id,
                source=detector,
                bbox_capture=(cx0, cy0, cx1 - cx0, cy1 - cy0),
                bbox_scene=(float(x), float(y), float(bw), float(bh)),
                peak=float(peak),
                mean=float(0.8 * peak),
                area_px=int(area),
                area_scene_px=float(bw * bh),
            )
        )
    return AnomalyResult(
        capture_id=capture.capture_id,
        detector=detector,
        score=float(prob * 10.0),
        prob=float(prob),
        regions=regions,
        reliable=reliable,
        registration={"method": "fake", "ok": reg_ok, "inliers": 100 if reg_ok else 0, "inlier_ratio": 0.8 if reg_ok else 0.0, "ncc": 0.9 if reg_ok else 0.12, "dx": 0.0, "dy": 0.0, "rot_deg": 0.0, "scale": 1.0},
        latency_ms=1.0,
        notes=[],
        anomaly_map=amap,
        valid_mask=np.full((h, w), 255, np.uint8),
    )


@dataclass
class FakeStat:
    """Fast-detector stand-in (registration-based: unreliable when ``register_ok`` is False or quality bad)."""

    spots: list[Spot] = field(default_factory=list)
    base_full: float = 0.03
    base_zoom: float = 0.03
    register_ok: bool = True
    zoom_register_ok: bool = True
    region_min_prob: float = 0.1
    calls: list[str] = field(default_factory=list)

    def predict(self, capture: Capture, quality: QualityReport | None = None, id_prefix: str = "") -> AnomalyResult:
        self.calls.append(capture.capture_id)
        full = capture.settings.view.zoom <= 1.05
        vis = [sp for sp in self.spots if _intersects(sp.bbox, capture.window)]
        sp_p = [(sp, sp.p_full if full else sp.p_zoom) for sp in vis]
        prob = max([self.base_full if full else self.base_zoom] + [p for _, p in sp_p])
        reg_ok = self.register_ok if full else self.zoom_register_ok
        reliable = reg_ok and (quality is None or quality.ok)
        return _make_result(capture, "fast", prob, sp_p, reliable, reg_ok, id_prefix, self.region_min_prob)

    def prob(self, score: float, zoom: float) -> float:
        return float(min(max(score / 10.0, 0.0), 1.0))


@dataclass
class FakeDeep:
    """Deep-detector stand-in (alignment-free: always reliable; the quality report is accepted and ignored)."""

    spots: list[Spot] = field(default_factory=list)
    base: float = 0.03
    region_min_prob: float = 0.1
    calls: list[str] = field(default_factory=list)

    def predict(self, capture: Capture, id_prefix: str = "", quality: QualityReport | None = None) -> AnomalyResult:
        self.calls.append(capture.capture_id)
        vis = [sp for sp in self.spots if _intersects(sp.bbox, capture.window)]
        sp_p = [(sp, sp.p_deep if sp.p_deep is not None else sp.p_full) for sp in vis]
        prob = max([self.base] + [p for _, p in sp_p])
        return _make_result(capture, "deep", prob, sp_p, True, True, id_prefix, self.region_min_prob)


def make_bundle(spots: list[Spot] | None = None, with_deep: bool = True, deep_base: float = 0.03, **stat_kwargs) -> CategoryBundle:
    spots = list(spots or [])
    return CategoryBundle(
        category="pcb1",
        stat=FakeStat(spots=spots, **stat_kwargs),
        deep=FakeDeep(spots=spots, base=deep_base) if with_deep else None,
        meta={"fake": True},
        reference_sharpness={"z1": 200.0, "z2": 200.0, "z3": 200.0, "z4": 200.0},
    )


def make_part(preset: str = "clean", part_id: str = "fake_part", label: int | None = None) -> PartSpec:
    return PartSpec(part_id=part_id, category="pcb1", image_path="", label=label, conditions=preset_conditions(preset))


def make_session(preset: str = "clean", spots: list[Spot] | None = None, with_deep: bool = True, settings=None, vision: FakeVision | None = None, **stat_kwargs):
    """(session, vision) with fakes; the scene is passed in, so no file IO happens."""
    from inspectagent.agent.session import InspectionSession

    vision = vision or FakeVision()
    bundle = make_bundle(spots, with_deep=with_deep, **stat_kwargs)
    s = InspectionSession(make_part(preset), bundle, settings, seed=0, scene=make_scene(), backend=vision.backend(), session_id="test")
    return s, vision


def run_fake(preset: str = "clean", spots: list[Spot] | None = None, with_deep: bool = True, planner=None, policy=None, settings=None, label: int | None = None, vision: FakeVision | None = None, on_step=None, **stat_kwargs):
    """Run the real loop against the fakes; returns (trace, session)."""
    from inspectagent.agent.loop import run_inspection

    vision = vision or FakeVision()
    bundle = make_bundle(spots, with_deep=with_deep, **stat_kwargs)
    return run_inspection(
        make_part(preset, label=label),
        bundle,
        planner=planner,
        policy=policy,
        settings=settings,
        seed=0,
        scene=make_scene(),
        backend=vision.backend(),
        session_id="test",
        on_step=on_step,
    )


# Typical spots (scene px; scene is 600x400)
DEFECT = Spot(bbox=(380.0, 240.0, 24.0, 16.0), p_full=0.62, p_zoom=0.97)
FALSE_ALARM = Spot(bbox=(150.0, 100.0, 20.0, 20.0), p_full=0.55, p_zoom=0.04, p_deep=0.05)
