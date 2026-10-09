"""Tests for the fast statistical template detector (inspectagent.vision.anomaly_stat).

Offline: synthetic boards from tests/synth.py; captures from the camera simulator
(inspectagent.vision.camera.VirtualCamera) when importable, otherwise from a minimal local
stand-in with the same geometry and exposure model.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from inspectagent.config import DetectorConfig
from inspectagent.types import AnomalyResult, Capture, CaptureSettings, NuisanceConditions, QualityReport, View
from inspectagent.vision.anomaly_stat import (
    ZOOM_BUCKETS,
    StatTemplateModel,
    capture_zoom,
    pseudo_capture,
    view_window,
    zoom_bucket,
)
from inspectagent.vision.registration import warp_points
from tests.synth import DEFECT_KINDS, inject_defect, make_board

try:  # the real simulator (written by another module owner)
    from inspectagent.vision.camera import VirtualCamera, make_conditions
except ImportError:  # pragma: no cover - exercised only while camera.py is missing
    VirtualCamera = None
    make_conditions = None

N_FIT = 18


# --------------------------------------------------------------------------------------
# capture helpers
# --------------------------------------------------------------------------------------
class _LocalCamera:
    """Minimal stand-in: pose jitter + crop/resample + linear-light exposure gain + JPEG."""

    def __init__(self, scene: np.ndarray, cond: NuisanceConditions, seed: int = 0):
        h, w = scene.shape[:2]
        self.scene_size = (w, h)
        m = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), cond.rot_deg, 1.0)
        m[0, 2] += cond.shift_x * w
        m[1, 2] += cond.shift_y * h
        self.pose_matrix = m
        self.stage = cv2.warpAffine(scene, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        self.cond = cond

    def capture(self, settings: CaptureSettings, capture_id: str, index: int) -> Capture:
        cap = pseudo_capture(self.stage, settings.view, capture_id=capture_id, index=index)
        lin = (cap.image.astype(np.float32) / 255.0) ** 2.2 * 2.0 ** (settings.exposure_ev + self.cond.ambient_ev)
        img = np.clip(lin ** (1 / 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1], cv2.IMREAD_COLOR)
        cap.image, cap.settings, cap.oracle = img, settings, False
        return cap

    def true_scene_to_capture(self, capture: Capture) -> np.ndarray:
        a = np.vstack([capture.scene_to_capture(), [0.0, 0.0, 1.0]])
        return (a @ np.vstack([self.pose_matrix, [0.0, 0.0, 1.0]]))[:2]


def _camera(scene: np.ndarray, preset: str, seed: int):
    if VirtualCamera is not None:
        return VirtualCamera(scene, make_conditions(preset, np.random.default_rng(seed)), seed=seed)
    rng = np.random.default_rng(seed)
    cond = NuisanceConditions(noise_sigma=0.0)
    if preset != "clean":
        cond.rot_deg, cond.shift_x, cond.shift_y = rng.uniform(-1.5, 1.5), rng.uniform(-0.015, 0.015), rng.uniform(-0.015, 0.015)
    return _LocalCamera(scene, cond, seed)


def _shoot(scene: np.ndarray, view: View, ev: float = 0.0, preset: str = "clean", seed: int = 0):
    cam = _camera(scene, preset, seed)
    return cam.capture(CaptureSettings(view=view, exposure_ev=ev), f"cap_{seed}", 0), cam


def _auroc(neg: list[float], pos: list[float]) -> float:
    n = np.asarray(neg)[None, :]
    p = np.asarray(pos)[:, None]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def _defect_center(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(xs.mean() / mask.shape[1]), float(ys.mean() / mask.shape[0])


def _overlap(a: tuple, b: tuple) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model() -> StatTemplateModel:
    return StatTemplateModel.fit("synth", [make_board(i) for i in range(N_FIT)], log=None)


@pytest.fixture(scope="module")
def heldout_normals() -> list[np.ndarray]:
    return [make_board(100 + i) for i in range(6)]


@pytest.fixture(scope="module")
def clean_normal_results(model, heldout_normals) -> list[AnomalyResult]:
    """Fast results on clean full-view captures of the held-out normals (shared by tests)."""
    return [model.predict(_shoot(s, View(), seed=i)[0]) for i, s in enumerate(heldout_normals)]


@pytest.fixture(scope="module")
def defects() -> list[tuple[np.ndarray, np.ndarray, str]]:
    out = []
    for j in range(8):  # two of each kind
        kind = DEFECT_KINDS[j % len(DEFECT_KINDS)]
        img, mask = inject_defect(make_board(200 + j), kind, seed=j)
        out.append((img, mask, kind))
    return out


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------
def test_zoom_buckets_and_view_window() -> None:
    assert [zoom_bucket(z) for z in (1.0, 1.49, 1.5, 2.49, 2.5, 3.49, 3.5, 4.0)] == [
        "z1", "z1", "z2", "z2", "z3", "z3", "z4", "z4"
    ]
    assert view_window((900, 600), View(0.5, 0.5, 1.0)) == (0.0, 0.0, 900.0, 600.0)
    x0, y0, ww, wh = view_window((900, 600), View(0.98, 0.02, 3.0))  # shifted, not shrunk
    assert (ww, wh) == (300.0, 200.0) and x0 == pytest.approx(600.0) and y0 == 0.0
    cap = pseudo_capture(make_board(1), View(0.3, 0.6, 2.5))
    assert capture_zoom(cap) == pytest.approx(2.5)
    assert cap.size == (768, 512)


@pytest.mark.skipif(VirtualCamera is None, reason="camera simulator not available")
def test_pseudo_capture_matches_simulator_geometry() -> None:
    scene = make_board(4)
    cam = VirtualCamera(scene, make_conditions("clean", np.random.default_rng(0)), seed=0)
    for v in (View(), View(0.35, 0.6, 2.0), View(0.8, 0.25, 3.7)):
        a = pseudo_capture(scene, v).image.astype(np.float32)
        b = cam.clean_capture(v, "c", 0).image.astype(np.float32)
        assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.99


# --------------------------------------------------------------------------------------
# fitting / rendering
# --------------------------------------------------------------------------------------
def test_fit_produces_consistent_template(model: StatTemplateModel) -> None:
    wt, ht = model.template_size
    assert max(wt, ht) == DetectorConfig().template_long_side
    assert model.scene_size == (900, 600)
    assert model.scale == pytest.approx(wt / 900.0)
    assert model.mu_lab.shape == (ht, wt, 3) and model.sigma_lab.shape == (ht, wt, 3)
    assert model.ref_gray.dtype == np.uint8 and model.ref_gray.shape == (ht, wt)
    assert float(model.sigma_lab.min()) >= DetectorConfig().sigma_floor * 0.99
    assert (model.valid > 0).mean() > 0.95
    assert model.meta["n_used"] == N_FIT and not model.meta["failed"]
    assert model.calib["source"] == "provisional_holdout"
    assert set(model.calib["buckets"]) == set(ZOOM_BUCKETS)
    # float16-exact arrays (lossless save/load)
    assert np.array_equal(model.mu_lab, model.mu_lab.astype(np.float16).astype(np.float32))


@pytest.mark.parametrize("view", [View(), View(0.3, 0.7, 2.5), View(0.9, 0.1, 4.0), View(0.62, 0.45, 1.3)])
def test_expected_render_aligns_with_commanded_view(model: StatTemplateModel, view: View) -> None:
    """template px -> scene px -> commanded capture px: the render must overlay the capture."""
    # the medoid normal defines the template pose: its capture must overlay without registration
    cap = pseudo_capture(make_board(model.meta["medoid"]), view)
    exp = model.render_expected(cap)
    g = cv2.cvtColor(cap.image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = exp["valid"]
    assert m.mean() > 0.9
    assert np.corrcoef(g[m], exp["gray"][m].astype(np.float32))[0, 1] > 0.96
    _, _, _, reg = model.score_capture(cap)
    assert reg.ok and abs(reg.dx) < 1.0 and abs(reg.dy) < 1.0 and abs(reg.rot_deg) < 0.3
    # another (held-out) instance of the design: residual pose stays small
    _, _, _, reg = model.score_capture(pseudo_capture(make_board(150), view))
    assert reg.ok and abs(reg.dx) < 2.0 and abs(reg.dy) < 2.0 and abs(reg.rot_deg) < 0.3


# --------------------------------------------------------------------------------------
# detection quality
# --------------------------------------------------------------------------------------
def test_separates_defects_from_normals_clean(model, clean_normal_results, defects) -> None:
    neg = [r.score for r in clean_normal_results]
    assert all(r.reliable and r.prob < 0.5 for r in clean_normal_results)
    pos_by_kind: dict[str, list[float]] = {}
    for j, (img, _m, kind) in enumerate(defects):
        pos_by_kind.setdefault(kind, []).append(model.predict(_shoot(img, View(), seed=50 + j)[0]).score)
    pos = [s for v in pos_by_kind.values() for s in v]
    assert _auroc(neg, pos) == 1.0
    for kind, scores in pos_by_kind.items():
        assert min(scores) > 2.0 * max(neg), (kind, scores, neg)


@pytest.mark.parametrize("ev", [1.0, -1.0])
def test_exposure_change_is_not_an_anomaly(model, heldout_normals, clean_normal_results, defects, ev: float) -> None:
    base = clean_normal_results[:4]
    shifted = [model.predict(_shoot(s, View(), ev, seed=i)[0]) for i, s in enumerate(heldout_normals[:4])]
    dfx = [model.predict(_shoot(img, View(), ev, seed=50 + j)[0]) for j, (img, _m, _k) in enumerate(defects[:4])]
    for b, s in zip(base, shifted):
        assert s.score < 1.6 * b.score + 0.5  # exposure must not inflate the anomaly score
        assert s.prob < 0.5 and s.reliable
        assert not s.regions
    assert all(r.registration["ok"] for r in dfx)
    assert _auroc([r.score for r in shifted], [r.score for r in dfx]) == 1.0
    assert min(r.prob for r in dfx) > 0.9


def test_zoom3_on_defect_separates_and_localizes(model, heldout_normals, defects) -> None:
    neg, pos = [], []
    for j, (img, mask, kind) in enumerate(defects[:4]):
        cx, cy = _defect_center(mask)
        view = View(cx, cy, 3.0)
        cap, cam = _shoot(img, view, preset="nominal", seed=70 + j)
        res = model.predict(cap, id_prefix=f"reg_{j}_")
        pos.append(res.score)
        neg.append(model.predict(_shoot(heldout_normals[j], view, preset="nominal", seed=j)[0]).score)
        assert res.reliable and res.regions, kind
        top = res.regions[0]
        assert top.region_id == f"reg_{j}_1"
        ys, xs = np.nonzero(mask)
        pts = np.stack([xs, ys], 1).astype(float)
        # bbox_scene is in the commanded (stage) frame = part frame moved by the pose jitter
        st = warp_points(cam.pose_matrix, pts)
        gt_stage = (st[:, 0].min(), st[:, 1].min(), np.ptp(st[:, 0]) + 1, np.ptp(st[:, 1]) + 1)
        assert _overlap(top.bbox_scene, gt_stage) > 0, (kind, top.bbox_scene, gt_stage)
        cp = warp_points(cam.true_scene_to_capture(cap), pts)
        gt_cap = (cp[:, 0].min(), cp[:, 1].min(), np.ptp(cp[:, 0]) + 1, np.ptp(cp[:, 1]) + 1)
        assert _overlap(top.bbox_capture, gt_cap) > 0, (kind, top.bbox_capture, gt_cap)
        # the anomaly map is in the capture frame: its peak lies on the defect
        yy, xx = np.unravel_index(int(np.argmax(res.anomaly_map)), res.anomaly_map.shape)
        assert gt_cap[0] - 12 <= xx <= gt_cap[0] + gt_cap[2] + 12 and gt_cap[1] - 12 <= yy <= gt_cap[1] + gt_cap[3] + 12
    assert _auroc(neg, pos) == 1.0


def test_full_view_region_maps_back_to_scene(model, defects) -> None:
    img, mask, kind = defects[2]  # a spot
    cap, cam = _shoot(img, View(), preset="nominal", seed=91)
    res = model.predict(cap)
    assert res.regions and res.regions[0].region_id == f"{cap.capture_id}_r1"
    ys, xs = np.nonzero(mask)
    st = warp_points(cam.pose_matrix, np.stack([xs, ys], 1).astype(float))
    gt_stage = (st[:, 0].min(), st[:, 1].min(), np.ptp(st[:, 0]) + 1, np.ptp(st[:, 1]) + 1)
    assert _overlap(res.regions[0].bbox_scene, gt_stage) > 0
    # a follow-up zoom on the region centre sees the defect again
    cx, cy = res.regions[0].center_norm(cap.scene_size)
    zres = model.predict(cam.capture(CaptureSettings(view=View(cx, cy, 3.0)), "cap_z", 1))
    assert zres.prob > 0.9 and zres.regions


# --------------------------------------------------------------------------------------
# contract / reliability
# --------------------------------------------------------------------------------------
def test_predict_output_contract(model) -> None:
    cap, _ = _shoot(make_board(120), View(0.4, 0.4, 2.0), seed=3)
    res = model.predict(cap, id_prefix="reg_9_")
    assert isinstance(res, AnomalyResult)
    assert res.detector == "fast" and res.capture_id == cap.capture_id
    assert 0.0 <= res.prob <= 1.0 and res.reliable
    assert res.anomaly_map.shape == cap.image.shape[:2] and res.anomaly_map.dtype == np.float32
    assert res.valid_mask.dtype == np.uint8 and set(np.unique(res.valid_mask)) <= {0, 255}
    assert set(res.registration) >= {"method", "ok", "inliers", "inlier_ratio", "ncc", "dx", "dy", "rot_deg", "scale"}
    assert res.latency_ms > 0 and isinstance(res.notes, list)
    d = res.to_dict()
    assert "anomaly_map" not in d and isinstance(d["regions"], list)


def test_unreliable_when_registration_fails_or_quality_bad(model) -> None:
    rng = np.random.default_rng(0)
    junk = Capture("junk", 0, CaptureSettings(), cv2.GaussianBlur(rng.integers(0, 255, (512, 768, 3), dtype=np.uint8), (0, 0), 3),
                   (0.0, 0.0, 900.0, 600.0), (900, 600), 0.0)
    res = model.predict(junk)
    assert not res.reliable and not res.registration["ok"]
    assert any("registration failed" in n for n in res.notes)

    cap, _ = _shoot(make_board(121), View(), seed=4)
    q = QualityReport(cap.capture_id, 100.0, 1.0, 120.0, 0.0, 0.0, 2.0, 0.01, [[100, 100, 60, 40]], ["glare"], 0.0, False)
    res = model.predict(cap, quality=q)
    assert not res.reliable
    assert any("glare boxes masked" in n for n in res.notes)
    assert res.valid_mask[120, 130] == 0  # inside the (dilated) glare box


def test_moderate_noise_does_not_make_fast_result_unreliable(model) -> None:
    """``noisy`` up to NOISE_TOLERANCE_SIGMA is absorbed by the noise-aware sigma; heavier noise blocks."""
    from inspectagent.vision.quality import NOISE_TOLERANCE_SIGMA

    cap, _ = _shoot(make_board(122), View(), seed=5)

    def noisy(sigma: float) -> QualityReport:
        return QualityReport(cap.capture_id, 100.0, 1.0, 120.0, 0.0, 0.0, sigma, 0.0, [], ["noisy"], 0.0, False)

    res = model.predict(cap, quality=noisy(8.0))
    assert res.reliable and res.prob < 0.5
    assert any(n.startswith("quality issues tolerated: noisy") for n in res.notes)
    res = model.predict(cap, quality=noisy(NOISE_TOLERANCE_SIGMA + 1.0))
    assert not res.reliable and any(n == "quality issues: noisy" for n in res.notes)


@pytest.mark.skipif(VirtualCamera is None, reason="camera simulator not available")
def test_gain_and_jpeg_chroma_noise_do_not_inflate_normal_scores(model, heldout_normals) -> None:
    """Regression: the a/b residual noise (JPEG 4:2:0 chroma) was underestimated ~4x by the pixel-scale
    Immerkaer estimator, so gain-compensated dark captures and noisy normals scored as anomalous."""

    def median_score(noise: float, ambient: float, ev: float) -> tuple[float, np.ndarray]:
        scores, noise_est = [], None
        for i, scene in enumerate(heldout_normals[:5]):
            cond = NuisanceConditions(ambient_ev=ambient, noise_sigma=noise, defocus_sigma=0.3, rot_deg=0.6, shift_x=0.005,
                                      shift_y=-0.004, jpeg_quality=92, preset="test")
            cap = VirtualCamera(scene, cond, seed=i).capture(CaptureSettings(exposure_ev=ev), f"c{i}", 0)
            r = model._analyze(cap)
            scores.append(r["score"])
            noise_est = r["noise"] if noise_est is None else noise_est
        return float(np.median(scores)), noise_est

    sd = float(model.calib["buckets"]["z1"]["sd"])
    base, noise_base = median_score(1.5, 0.0, 0.0)
    dark, noise_dark = median_score(1.5, -2.0, 2.0)
    loud, _ = median_score(4.0, 0.0, 0.0)
    assert dark - base <= 1.5 * sd, (base, dark, sd)
    assert loud - base <= 1.5 * sd, (base, loud, sd)
    # the chroma noise estimate follows the gain (it was stuck near 0.3-0.4 Lab units before)
    assert noise_dark[1] > 2.0 * noise_base[1] and noise_dark[2] > 2.0 * noise_base[2]
    assert min(noise_base[1], noise_base[2]) > 0.8


# --------------------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------------------
def test_calibration_buckets_logistic_and_prob() -> None:
    m = StatTemplateModel(
        "c", (900, 600), 1.0, np.zeros((20, 30, 3), np.float32), np.ones((20, 30, 3), np.float32), np.zeros((20, 30), np.uint8)
    )
    assert m.calib["source"] == "default"
    rng = np.random.default_rng(1)
    normals = {"z1": list(rng.normal(2.5, 0.2, 30)), "z3": list(rng.normal(3.0, 0.4, 20)) + [40.0]}  # one gross outlier
    m.calibrate(normals)
    b = m.calib["buckets"]
    assert b["z1"]["mu"] == pytest.approx(2.5, abs=0.15) and b["z3"]["dropped"] == 1
    assert b["z2"]["filled_from"] in ("z1", "z3") and b["z4"]["filled_from"] == "z3"
    assert m.calib["region_thr"]["z1"] == pytest.approx(b["z1"]["region_thr"]) and m.region_threshold(1.0) >= 4.0
    assert m.calib["logistic"] == {"a": 1.6, "b": -4.0, "fitted": False, "n": 0}
    assert m.prob(2.5, 1.0) == pytest.approx(1 / (1 + np.exp(4.0)), rel=0.2)
    labeled = [(float(s), 0, "z1") for s in rng.normal(2.5, 0.2, 40)] + [(float(s), 1, 1.0) for s in rng.normal(4.5, 1.0, 40)]
    m.calibrate(normals, labeled)
    lg = m.calib["logistic"]
    assert lg["fitted"] and lg["a"] > 0 and lg["n"] == 80
    probs = [m.prob(s, 1.0) for s in (2.0, 2.5, 3.0, 4.0, 6.0)]
    assert all(p2 > p1 for p1, p2 in zip(probs, probs[1:])) and probs[0] < 0.2 and probs[-1] > 0.9
    assert m.prob(3.5, 3.0) < m.prob(3.5, 1.0)  # bucket-specific standardisation
    m.set_region_thresholds({"z2": 5.5, 3.2: 6.5})
    assert m.region_threshold(2.0) == 5.5 and m.calib["buckets"]["z3"]["region_thr"] == 6.5
    assert m.calib["logistic"]["fitted"]  # other calibration values are kept
    with pytest.raises(ValueError):
        m.calibrate({})


def test_scene_size_mismatch_and_resampled_scenes(model) -> None:
    # the same part imaged at 2x the fit resolution: the commanded mapping is rescaled
    big = cv2.resize(make_board(model.meta["medoid"]), (1800, 1200), interpolation=cv2.INTER_CUBIC)
    res = model.predict(pseudo_capture(big, View(0.4, 0.6, 2.0)))
    reg = res.registration
    assert res.reliable and reg["ok"] and abs(reg["dx"]) < 2.0 and abs(reg["dy"]) < 2.0
    assert res.prob < 0.5
    # pseudo_capture: a half-resolution image standing for the full-size scene
    full = make_board(3)
    half = cv2.resize(full, (450, 300), interpolation=cv2.INTER_AREA)
    a = pseudo_capture(half, View(0.6, 0.4, 1.5), scene_size=(900, 600))
    b = pseudo_capture(full, View(0.6, 0.4, 1.5))
    assert a.window == b.window and a.scene_size == b.scene_size
    assert np.corrcoef(a.image.astype(np.float32).ravel(), b.image.astype(np.float32).ravel())[0, 1] > 0.97


# --------------------------------------------------------------------------------------
# persistence / latency
# --------------------------------------------------------------------------------------
def test_save_load_round_trip(model, tmp_path: Path) -> None:
    p = tmp_path / "sub dir" / "stat.npz"  # space in the path on purpose
    model.save(p)
    with np.load(p, allow_pickle=False) as z:
        assert z["mu_lab"].dtype == np.float16 and z["sigma_lab"].dtype == np.float16
    m2 = StatTemplateModel.load(p)
    assert m2.category == model.category and m2.scene_size == model.scene_size and m2.scale == model.scale
    assert np.array_equal(m2.mu_lab, model.mu_lab) and np.array_equal(m2.sigma_lab, model.sigma_lab)
    assert np.array_equal(m2.ref_gray, model.ref_gray) and np.array_equal(m2.valid, model.valid)
    assert m2.calib == model.calib and m2.meta == model.meta
    cap, _ = _shoot(make_board(130), View(0.55, 0.5, 2.0), seed=5)
    assert m2.predict(cap).score == pytest.approx(model.predict(cap).score, rel=1e-6, abs=1e-6)


def test_predict_latency_768(model) -> None:
    cap, _ = _shoot(make_board(131), View(), preset="nominal", seed=6)
    model.predict(cap)  # warm caches
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        model.predict(cap)
        times.append((time.perf_counter() - t0) * 1000.0)
    assert cap.image.shape[1] == 768
    # target is < 150 ms on the dev box; allow headroom for loaded CI machines
    assert min(times) < 300.0, times
