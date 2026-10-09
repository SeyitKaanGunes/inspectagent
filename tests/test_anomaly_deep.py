"""Tests for the cv2.dnn PatchCore detector (SPEC §2.7).

Pure-numpy parts (preprocessing, feature post-processing, coreset, NN search, calibration, persistence)
always run. Tests that need the exported backbone skip cleanly when ``models/backbone_resnet18.onnx`` is
missing (generate it with ``scripts/export_backbone.py``). Everything is offline and deterministic.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from inspectagent.config import CameraConfig, DetectorConfig
from inspectagent.types import AnomalyResult, Capture, CaptureSettings, View
from inspectagent.vision import anomaly_deep as ad
from inspectagent.vision.anomaly_deep import PatchCoreDNN, border_margin, interior_max
from tests.synth import DEFECT_KINDS, inject_defect, make_board

ONNX_PATH = ad.default_onnx_path()
MISSING_ONNX = Path(__file__).resolve().parent / "__no_such_backbone__.onnx"


# ------------------------------------------------------------------------------------------- helpers
def _sensor_size(w: int, h: int) -> tuple[int, int]:
    s = CameraConfig().sensor_long_side / max(w, h)
    return int(round(w * s)), int(round(h * s))


def full_capture(scene: np.ndarray, cid: str = "cap_1") -> Capture:
    """Full-view capture of a scene at sensor resolution (commanded window = whole scene)."""
    h, w = scene.shape[:2]
    img = cv2.resize(scene, _sensor_size(w, h), interpolation=cv2.INTER_AREA)
    return Capture(cid, 0, CaptureSettings(), img, (0.0, 0.0, float(w), float(h)), (w, h), 0.0)


def zoom_capture(scene: np.ndarray, cx: float, cy: float, zoom: float, cid: str = "cap_z") -> Capture:
    """Zoomed capture: window of size scene/zoom centred on (cx, cy) (shifted inside), upsampled to sensor."""
    h, w = scene.shape[:2]
    ww, wh = w / zoom, h / zoom
    x0 = float(np.clip(cx * w - ww / 2, 0, w - ww))
    y0 = float(np.clip(cy * h - wh / 2, 0, h - wh))
    crop = scene[int(round(y0)) : int(round(y0 + wh)), int(round(x0)) : int(round(x0 + ww))]
    img = cv2.resize(crop, _sensor_size(w, h), interpolation=cv2.INTER_CUBIC)
    settings = CaptureSettings(view=View(cx, cy, zoom))
    return Capture(cid, 1, settings, img, (x0, y0, ww, wh), (w, h), 0.0)


def naive_greedy(x: np.ndarray, k: int, proj_dim: int | None, seed: int) -> np.ndarray:
    """Reference farthest-point sampling with the same projection/start as ``greedy_coreset``."""
    rng = np.random.default_rng(seed)
    if proj_dim is not None and proj_dim < x.shape[1]:
        r = (rng.standard_normal((x.shape[1], proj_dim)) / math.sqrt(proj_dim)).astype(np.float32)
        z = x.astype(np.float32) @ r
    else:
        z = x.astype(np.float32)
    first = int(rng.integers(len(z)))
    sel = [first]
    md = np.linalg.norm(z - z[first], axis=1)
    for _ in range(1, k):
        s = int(np.argmax(md))
        sel.append(s)
        md = np.minimum(md, np.linalg.norm(z - z[s], axis=1))
    return np.array(sel)


@pytest.fixture(scope="module")
def onnx_path() -> Path:
    if not ONNX_PATH.is_file():
        pytest.skip(f"{ONNX_PATH} missing - run scripts/export_backbone.py")
    return ONNX_PATH


@pytest.fixture(scope="module")
def fitted(onnx_path: Path) -> PatchCoreDNN:
    """Detector fitted on 16 synthetic normals, small coreset, no augmentation (spec-pure, camera-independent)."""
    normals = [make_board(i) for i in range(16)]
    return PatchCoreDNN.fit("synth", normals, onnx_path, coreset_size=1000, seed=0, augment=None, log=lambda s: None)


@pytest.fixture(scope="module")
def eval_set(fitted: PatchCoreDNN) -> dict:
    normals = [fitted.predict(full_capture(make_board(100 + i), f"n{i}")) for i in range(6)]
    defects = []
    for j in range(8):
        img, mask = inject_defect(make_board(200 + j), DEFECT_KINDS[j % len(DEFECT_KINDS)], seed=j)
        cap = full_capture(img, f"d{j}")
        mask_c = cv2.resize(mask, cap.size, interpolation=cv2.INTER_NEAREST)
        defects.append((fitted.predict(cap, id_prefix=f"reg_{j}_"), mask_c, cap))
    return {"normals": normals, "defects": defects}


# ------------------------------------------------------------------------------- pure numpy helpers
def test_input_size_for() -> None:
    assert ad.input_size_for(900, 600, 448) == (448, 304)
    assert ad.input_size_for(600, 900, 448) == (304, 448)
    for w, h in [(1500, 1000), (640, 480), (100, 3000), (37, 41)]:
        iw, ih = ad.input_size_for(w, h, 448)
        assert iw % 16 == 0 and ih % 16 == 0 and min(iw, ih) >= 32 and max(iw, ih) == 448
    with pytest.raises(ValueError):
        ad.input_size_for(0, 10)


def test_bilinear_resize_matches_cv2() -> None:
    rng = np.random.default_rng(0)
    x = rng.random((19, 28, 256), dtype=np.float32)
    for out_h, out_w in [(38, 56), (25, 33), (7, 9)]:
        mine = ad.resize_bilinear_hwc(x, out_h, out_w)
        assert mine.shape == (out_h, out_w, 256)
        for c0 in (0, 120, 253):  # cv2 handles <= 4 channels: compare 3-channel slices
            ref = cv2.resize(np.ascontiguousarray(x[..., c0 : c0 + 3]), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            assert np.abs(mine[..., c0 : c0 + 3] - ref).max() < 1e-5


def test_avg_pool_matches_cv2_blur() -> None:
    rng = np.random.default_rng(1)
    for h, w in [(38, 56), (2, 5), (1, 4), (3, 1)]:
        x = rng.random((h, w, 6), dtype=np.float32)
        mine = ad._avg_pool3(x)
        for c0 in (0, 3):
            ref = cv2.blur(np.ascontiguousarray(x[..., c0 : c0 + 3]), (3, 3), borderType=cv2.BORDER_REFLECT_101)
            assert np.abs(mine[..., c0 : c0 + 3] - ref.reshape(h, w, 3)).max() < 1e-5, (h, w)


def test_patch_features_from_layers_matches_reference() -> None:
    rng = np.random.default_rng(2)
    l2 = rng.random((1, 128, 38, 56), dtype=np.float32)
    l3 = rng.random((1, 256, 19, 28), dtype=np.float32)
    f = ad.patch_features_from_layers(l2, l3)
    assert f.shape == (38, 56, ad.FEATURE_DIM) and f.dtype == np.float32
    f2 = l2[0].transpose(1, 2, 0)
    up = ad.resize_bilinear_hwc(np.ascontiguousarray(l3[0].transpose(1, 2, 0)), 38, 56)
    ref = ad._avg_pool3(np.concatenate([f2, up], axis=2))
    assert np.abs(f - ref).max() < 1e-5
    with pytest.raises(ValueError):
        ad.patch_features_from_layers(l2[0], l3)


def test_make_blob_matches_manual_normalisation() -> None:
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (304, 448, 3), dtype=np.uint8)
    blob = ad.make_blob(img, (448, 304))
    rgb = img[..., ::-1].astype(np.float32) / 255.0
    ref = (rgb - np.array(ad.IMAGENET_MEAN_RGB, np.float32)) / np.array(ad.IMAGENET_STD_RGB, np.float32)
    assert blob.shape == (1, 3, 304, 448) and blob.dtype == np.float32
    assert np.abs(blob[0].transpose(1, 2, 0) - ref).max() < 1e-4
    gray = ad.make_blob(img[..., 0], (224, 160))  # gray input is expanded to 3 channels
    assert gray.shape == (1, 3, 160, 224)


def test_nn_distances_matches_bruteforce() -> None:
    rng = np.random.default_rng(4)
    q = rng.standard_normal((300, 384)).astype(np.float32)
    bank = rng.standard_normal((257, 384)).astype(np.float32)
    ref = np.sqrt(((q[:, None, :].astype(np.float64) - bank[None]) ** 2).sum(-1))
    for max_elems in (1000, 1 << 25):  # forces many chunks / single block
        d = ad.nn_distances(q, bank, max_elems=max_elems)
        assert np.allclose(d, ref.min(1), rtol=1e-4, atol=1e-3)
    exclude = np.zeros(len(bank), bool)
    exclude[::2] = True
    d_ex = ad.nn_distances(q, bank, exclude=exclude, max_elems=5000)
    assert np.allclose(d_ex, ref[:, ~exclude].min(1), rtol=1e-4, atol=1e-3)
    with pytest.raises(ValueError):
        ad.nn_distances(q, bank, exclude=np.ones(len(bank), bool))
    with pytest.raises(ValueError):
        ad.nn_distances(q, bank[:, :10])


@pytest.mark.parametrize("n,dim,k,proj", [(1500, 16, 150, None), (2500, 384, 200, 128), (600, 4, 599, None)])
def test_greedy_coreset_matches_naive_greedy(n: int, dim: int, k: int, proj: int | None) -> None:
    x = np.random.default_rng(5).standard_normal((n, dim)).astype(np.float32)
    sel = ad.greedy_coreset(x, k, proj, seed=3)
    ref = naive_greedy(x, k, proj, seed=3)
    assert sel.dtype == np.int64 and len(sel) == k and len(np.unique(sel)) == k
    assert np.array_equal(sel, ref)


def test_greedy_coreset_edge_cases() -> None:
    rng = np.random.default_rng(6)
    x = rng.standard_normal((400, 8)).astype(np.float32)
    assert np.array_equal(ad.greedy_coreset(x, 400), np.arange(400))
    assert np.array_equal(ad.greedy_coreset(x, 1000), np.arange(400))
    assert ad.greedy_coreset(x, 0).size == 0
    a, b = ad.greedy_coreset(x, 50, seed=1), ad.greedy_coreset(x, 50, seed=1)
    assert np.array_equal(a, b)  # deterministic
    dup = np.repeat(rng.standard_normal((30, 4)).astype(np.float32), 5, axis=0)
    s = ad.greedy_coreset(dup, 100, None)
    assert len(s) == 30 and len(np.unique(dup[s], axis=0)) == 30  # stops once only duplicates remain
    with pytest.raises(ValueError):
        ad.greedy_coreset(np.zeros((0, 4), np.float32), 3)
    # k-center property: covering radius beats a random subset of the same size
    pts = rng.standard_normal((3000, 16)).astype(np.float32)
    cs = ad.greedy_coreset(pts, 100, None)
    rnd = rng.choice(3000, 100, replace=False)
    assert ad.nn_distances(pts, pts[cs]).max() < ad.nn_distances(pts, pts[rnd]).max()


def test_constructor_validation() -> None:
    with pytest.raises(ValueError):
        PatchCoreDNN(MISSING_ONNX, (450, 304))
    with pytest.raises(ValueError):
        PatchCoreDNN(MISSING_ONNX, (16, 304))
    with pytest.raises(ValueError):
        PatchCoreDNN(MISSING_ONNX, (448, 304), engine="gpu")
    m = PatchCoreDNN(MISSING_ONNX, (448, 304))
    assert not m.is_fitted
    with pytest.raises(RuntimeError):
        m.predict(full_capture(make_board(0)))


def _manual_model(seed: int = 0) -> PatchCoreDNN:
    m = PatchCoreDNN(MISSING_ONNX, (448, 304))
    m.category, m.scene_size = "synth", (900, 600)
    m._set_bank(np.abs(np.random.default_rng(seed).standard_normal((64, ad.FEATURE_DIM))).astype(np.float32))
    return m


def test_calibrate_and_prob() -> None:
    m = _manual_model()
    with pytest.raises(ValueError):
        m.calibrate([])  # uncalibrated model needs normal scores
    normal = list(np.random.default_rng(7).normal(1.0, 0.1, 40))
    m.calibrate(normal)
    c = m.calib
    assert c["source"] == "calibrate" and (c["a"], c["b"]) == ad.DEFAULT_LOGISTIC
    assert abs(c["mu"] - np.mean(normal)) < 1e-9 and c["region_thr"] == pytest.approx(c["mu"] + 2 * c["sd"])
    assert m.prob(c["mu"]) < 0.05 < 0.5 < m.prob(c["mu"] + 6 * c["sd"])
    probs = [m.prob(s) for s in np.linspace(0.5, 2.0, 20)]
    assert all(p2 >= p1 for p1, p2 in zip(probs, probs[1:]))
    labeled = [(float(v), 0) for v in normal[:20]] + [(float(v), 1) for v in np.random.default_rng(8).normal(1.6, 0.1, 20)]
    m.calibrate(normal, labeled)
    assert m.calib["source"] == "calibrate+logistic" and m.calib["a"] > 0
    assert m.prob(1.6) > 0.9 and m.prob(1.0) < 0.1
    m.calibrate(normal, [(1.0, 0), (1.1, 0), (0.9, 0), (1.05, 0)])  # single class -> fallback logistic
    assert (m.calib["a"], m.calib["b"]) == ad.DEFAULT_LOGISTIC
    mu_before = m.calib["mu"]
    m.calibrate([float("nan")], labeled)  # no finite normal score: keep mu/sd, still fit the logistic
    assert m.calib["mu"] == mu_before and m.calib["source"] == "calibrate+logistic"
    # region threshold: normal level gates, half-maximum term follows strong peaks
    assert m.region_threshold(m.calib["mu"]) == pytest.approx(m.calib["region_thr"])
    assert m.region_threshold(10.0) == pytest.approx(m.calib["mu"] + 0.5 * (10.0 - m.calib["mu"]))


def test_logistic_fit_stays_finite_on_separable_calibration() -> None:
    """Perfectly separable labeled sets must give a graded logistic, not a step function."""
    rng = np.random.default_rng(3)
    z = np.r_[rng.normal(0.0, 1.0, 8), rng.normal(12.0, 3.0, 8)]
    y = np.r_[np.zeros(8), np.ones(8)]
    a, b = ad._fit_logistic(z, y)
    assert 0.0 < a < 5.0 and -15.0 < b < 0.0
    p = [ad._sigmoid(a * v + b) for v in (0.0, 3.0, 12.0)]
    assert p[0] < 0.05 and 1e-4 < p[1] < 0.9 and p[2] > 0.9  # mildly elevated scores are not 1e-26
    m = _manual_model(4)
    m.calibrate(list(rng.normal(2.0, 0.5, 16)), [(float(2.0 + 0.5 * v), int(t)) for v, t in zip(z, y)])
    assert m.calib["source"] == "calibrate+logistic" and m.calib["a"] < 5.0


def test_save_load_roundtrip_without_backbone(tmp_path: Path) -> None:
    m = _manual_model(1)
    m.calibrate([1.0, 1.2, 0.9, 1.1])
    m.fit_info = {"n_images": 3}
    path = tmp_path / "sub dir" / "deep.npz"  # space in path, parent created
    m.save(path)
    assert path.is_file() and not path.with_name("deep.npz.tmp").exists()
    m2 = PatchCoreDNN.load(path, MISSING_ONNX)
    assert m2.memory_bank.dtype == np.float16 and np.array_equal(m2.memory_bank, m.memory_bank)
    assert m2.calib == m.calib and m2.input_size == m.input_size and m2.scene_size == m.scene_size
    assert m2.category == "synth" and m2.fit_info == {"n_images": 3}
    bad = tmp_path / "bad.npz"
    np.savez(bad, memory_bank=m.memory_bank, meta=np.array(json.dumps({"format": "other"})))
    with pytest.raises(ValueError):
        PatchCoreDNN.load(bad, MISSING_ONNX)


def test_predict_without_backbone_is_unreliable() -> None:
    m = _manual_model(2)
    m.calibrate([1.0, 1.1])
    res = m.predict(full_capture(make_board(0)))
    assert isinstance(res, AnomalyResult) and res.detector == "deep"
    assert res.reliable is False and res.regions == [] and "deep inference failed" in res.notes[0]
    json.dumps(res.to_dict())


# ----------------------------------------------------------------------------------- needs the ONNX
def test_features_shape_engine_and_dynamic_input(onnx_path: Path) -> None:
    m = PatchCoreDNN(onnx_path, (448, 304))
    info = m.warmup()
    assert info["engine"] in ("new", "auto", "classic") and info["dynamic"] is True
    f = m.features(make_board(0))
    assert f.shape == (38, 56, 384) and f.dtype == np.float32
    assert np.isfinite(f).all() and f.min() >= 0.0 and f.max() > 0.0  # post-ReLU features
    g = m.features(make_board(0), size=(224, 160))  # dynamic H/W
    assert g.shape == (20, 28, 384)
    assert np.array_equal(m.features(make_board(0)), f)  # deterministic
    gray = m.features(cv2.cvtColor(make_board(0), cv2.COLOR_BGR2GRAY))
    assert gray.shape == f.shape


def test_engines_agree(onnx_path: Path) -> None:
    img = make_board(3)
    new = PatchCoreDNN(onnx_path, (224, 160), engine="new").features(img)
    classic = PatchCoreDNN(onnx_path, (224, 160), engine="classic").features(img)
    assert np.abs(new - classic).max() < 1e-3


def test_fit_builds_coreset_bank(fitted: PatchCoreDNN, onnx_path: Path) -> None:
    assert fitted.memory_bank.shape == (1000, 384) and fitted.memory_bank.dtype == np.float16
    assert fitted.input_size == (448, 304) and fitted.scene_size == (900, 600)
    info = fitted.fit_info
    assert info["n_images"] == 16 and info["n_bank_images"] == 16 and info["augment"] == "none"
    assert info["calib_mode"] == "loo" and fitted.calib["source"] == "fit_loo"  # < 3 held-out images
    assert info["n_sampled_patches"] == 16 * round(0.2 * 38 * 56)
    assert fitted.onnx_sha256 and len(fitted.onnx_sha256) == 64
    assert math.isfinite(fitted.calib["region_thr"]) and fitted.calib["sd"] > 0


def test_detects_injected_defects(eval_set: dict) -> None:
    normal_scores = [r.score for r in eval_set["normals"]]
    defect_scores = [r.score for r, _, _ in eval_set["defects"]]
    assert min(defect_scores) > max(normal_scores), (normal_scores, defect_scores)
    assert max(r.prob for r in eval_set["normals"]) < 0.15
    assert np.mean([r.prob for r, _, _ in eval_set["defects"]]) > 0.8
    assert all(not r.regions for r in eval_set["normals"])


def test_defect_localisation(eval_set: dict) -> None:
    for res, mask, cap in eval_set["defects"]:
        amap = res.anomaly_map
        assert amap.shape == mask.shape
        y, x = np.unravel_index(int(np.argmax(amap)), amap.shape)
        near = cv2.dilate(mask, np.ones((17, 17), np.uint8))  # one stride-8 patch at 448 ~ 14 capture px
        assert near[y, x] > 0, f"argmax ({x},{y}) outside defect"
        assert res.regions, "no region on a defective part"
        top = res.regions[0]
        bx, by, bw, bh = top.bbox_capture
        assert mask[by : by + bh, bx : bx + bw].any(), "top region misses the defect"
        assert top.source == "deep" and top.region_id.startswith(res.capture_id.replace("d", "reg_") + "_")
        # pixel-level separation: mean map inside the defect well above the rest
        assert amap[mask > 0].mean() > 1.5 * amap[mask == 0].mean()
        sx, sy, sw, sh = top.bbox_scene  # scene box = capture box scaled by the commanded window
        c2s = cap.capture_to_scene()
        assert sx == pytest.approx(bx * c2s[0, 0] + c2s[0, 2]) and sw == pytest.approx(bw * c2s[0, 0])


def test_predict_result_contract(eval_set: dict) -> None:
    res, _, cap = eval_set["defects"][0]
    w, h = cap.size
    assert res.detector == "deep" and res.reliable is True and res.capture_id == cap.capture_id
    assert res.anomaly_map.shape == (h, w) and res.anomaly_map.dtype == np.float32
    m = border_margin(w, h)
    assert res.valid_mask.shape == (h, w) and m >= 8
    assert (res.valid_mask[m : h - m, m : w - m] == 255).all() and (res.valid_mask[:m] == 0).all() and (res.valid_mask[:, -m:] == 0).all()
    assert res.registration["ok"] is True and res.registration["method"].startswith("none")
    assert res.score == pytest.approx(float(res.anomaly_map[res.valid_mask > 0].max())) and 0.0 <= res.prob <= 1.0
    assert any(n.startswith("engine=") for n in res.notes) and res.latency_ms > 0
    d = json.loads(json.dumps(res.to_dict()))
    assert d["detector"] == "deep" and len(d["regions"]) == len(res.regions)


def test_zoom_capture_is_scale_consistent(fitted: PatchCoreDNN) -> None:
    img, mask = inject_defect(make_board(250), "spot", seed=3)
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean() / mask.shape[1], ys.mean() / mask.shape[0]
    zd = zoom_capture(img, cx, cy, 2.0, "zd")
    zn = zoom_capture(make_board(251), cx, cy, 2.0, "zn")
    assert fitted.input_size_for_capture(zd) == (224, 160)  # half the scene -> half the input
    rd, rn = fitted.predict(zd), fitted.predict(zn)
    assert rd.anomaly_map.shape == (zd.size[1], zd.size[0])
    assert "input=224x160" in " ".join(rd.notes)
    # at half the input the zero-padding border dominates: zoom scores do not separate normals from defects
    # (measured), so zoomed deep results are never reliable evidence
    assert not rd.reliable and not rn.reliable and rd.prob == rn.prob == pytest.approx(0.5)


@pytest.mark.parametrize("zoom", [2.0, 3.0, 4.0])
def test_zoomed_deep_result_is_not_reliable_evidence(fitted: PatchCoreDNN, zoom: float) -> None:
    """Regression: deep results on zoomed views were scored with the full-view calibration (normal parts
    reached p=1.0) and reported reliable, so a zoom of a normal part could drive an automatic FAIL."""
    from inspectagent.agent.session import ZOOM_CONFIRM_PROB

    for j, (cx, cy) in enumerate(((0.5, 0.5), (0.3, 0.7))):
        res = fitted.predict(zoom_capture(make_board(270 + j), cx, cy, zoom, f"zn{j}"), id_prefix="reg_9_")
        assert res.reliable is False and res.prob < ZOOM_CONFIRM_PROB and res.regions == []
        assert any(n.startswith("zoomed view") and "not reliable" in n for n in res.notes), res.notes
    full = fitted.predict(full_capture(make_board(272), "fn"))
    assert full.reliable is True  # full views are unaffected


def test_border_margin_excludes_jitter_edge_from_score_and_regions(fitted: PatchCoreDNN) -> None:
    """Regression: pose-jitter edge streaks dominated the deep image score (max over the whole map)."""
    camera = pytest.importorskip("inspectagent.vision.camera")
    from inspectagent.types import NuisanceConditions

    cond = NuisanceConditions(rot_deg=2.5, shift_x=0.025, shift_y=-0.025, noise_sigma=0.0, jpeg_quality=95, preset="test")
    cap = camera.VirtualCamera(make_board(280), cond, seed=1).capture(CaptureSettings(), "jit", 0)
    res = fitted.predict(cap, id_prefix="reg_8_")
    w, h = cap.size
    m = border_margin(w, h)
    amap = res.anomaly_map
    assert res.score == pytest.approx(interior_max(amap))
    assert res.score <= float(amap.max())
    for r in res.regions:  # regions never live entirely inside the excluded margin
        x, y, bw, bh = r.bbox_capture
        assert x + bw > m and y + bh > m and x < w - m and y < h - m
    assert border_margin(768, 512) >= int(2 * 4.0) and border_margin(40, 30) * 4 < 30


def test_save_load_roundtrip_predict(fitted: PatchCoreDNN, onnx_path: Path, tmp_path: Path) -> None:
    path = tmp_path / "deep.npz"
    fitted.save(path)
    loaded = PatchCoreDNN.load(path, onnx_path)
    assert np.array_equal(loaded.memory_bank, fitted.memory_bank) and loaded.calib == fitted.calib
    assert loaded.fit_info == fitted.fit_info and loaded.onnx_sha256 == fitted.onnx_sha256
    img, _ = inject_defect(make_board(260), "solder", seed=4)
    cap = full_capture(img, "rt")
    a, b = fitted.predict(cap), loaded.predict(cap)
    assert b.score == pytest.approx(a.score, rel=1e-6) and b.prob == pytest.approx(a.prob, rel=1e-6)
    assert [r.bbox_capture for r in b.regions] == [r.bbox_capture for r in a.regions]


def test_calibrate_on_held_out_scores(fitted: PatchCoreDNN, onnx_path: Path, tmp_path: Path) -> None:
    path = tmp_path / "clone.npz"
    fitted.save(path)
    m = PatchCoreDNN.load(path, onnx_path)  # clone: the module fixture stays untouched
    normal = [m.predict(full_capture(make_board(300 + i))).score for i in range(4)]
    labeled = [(s, 0) for s in normal]
    for j in range(4):
        img, _ = inject_defect(make_board(320 + j), DEFECT_KINDS[j], seed=10 + j)
        labeled.append((m.predict(full_capture(img)).score, 1))
    m.calibrate(normal, labeled)
    assert m.calib["source"] == "calibrate+logistic" and m.calib["n_normal"] == 4
    assert max(m.prob(s) for s, y in labeled if y == 0) < 0.5 < min(m.prob(s) for s, y in labeled if y == 1)


def test_fit_holdout_mode_and_lazy_iterable(onnx_path: Path) -> None:
    cfg = DetectorConfig(deep_train_images=12)
    boards = (make_board(i) for i in itertools.count())  # infinite: fit must stop at deep_train_images
    m = PatchCoreDNN.fit(
        "synth", boards, onnx_path, cfg=cfg, input_size=(224, 160), coreset_size=300,
        holdout_every=4, augment=None, log=lambda s: None,
    )
    info = m.fit_info
    assert info["n_images"] == 12 and info["n_bank_images"] == 9 and info["calib_mode"] == "holdout"
    assert m.calib["source"] == "fit_holdout" and m.calib["n_normal"] == 3
    assert m.memory_bank.shape == (300, 384)
    with pytest.raises(ValueError):
        PatchCoreDNN.fit("synth", [make_board(0)], onnx_path, augment=None, log=lambda s: None)
    with pytest.raises(ValueError):
        PatchCoreDNN.fit("synth", [make_board(0)] * 3, onnx_path, augment="flip", log=lambda s: None)


def test_custom_augment_callable(onnx_path: Path) -> None:
    calls = []

    def flip(img: np.ndarray, index: int, rng: np.random.Generator) -> list[np.ndarray]:
        calls.append(index)
        return [img[:, ::-1]]

    m = PatchCoreDNN.fit(
        "synth", [make_board(i) for i in range(4)], onnx_path, input_size=(224, 160),
        coreset_size=200, augment=flip, log=lambda s: None,
    )
    assert calls == [0, 1, 2, 3] and m.fit_info["n_views"] == 8 and m.fit_info["augment"] == "flip"


def test_camera_augmentation_reduces_nominal_gap(fitted: PatchCoreDNN, onnx_path: Path) -> None:
    """Default fit renders each normal through the virtual camera; normal captures under nominal
    conditions (pose jitter, noise, JPEG) then score much closer to clean ones than with a bank of clean
    scenes only (``fitted``, augment=None)."""
    camera = pytest.importorskip("inspectagent.vision.camera")
    aug = PatchCoreDNN.fit("synth", [make_board(i) for i in range(16)], onnx_path, coreset_size=1000, log=lambda s: None)
    assert aug.fit_info["augment"].startswith("camera") and aug.fit_info["n_views"] == 32

    def gap(model: PatchCoreDNN) -> float:
        nominal, clean = [], []
        for j in range(3):
            scene = make_board(400 + j)
            cond = camera.make_conditions("nominal", np.random.default_rng(40 + j))
            cond.glare_strength = 0.0
            cap = camera.VirtualCamera(scene, cond, seed=j).capture(CaptureSettings(), f"nom{j}", 0)
            nominal.append(model.predict(cap).score)
            clean.append(model.predict(full_capture(scene)).score)
        return float(np.mean(nominal) / np.mean(clean))

    g_plain, g_aug = gap(fitted), gap(aug)
    assert g_aug < g_plain and g_aug < 1.4, (g_plain, g_aug)


def test_quality_report_gates_deep_reliability(fitted: PatchCoreDNN) -> None:
    """Deep is alignment-free but not quality-proof: blocking quality issues make it unreliable."""
    from inspectagent.types import QualityReport

    cap = full_capture(make_board(130), "q1")

    def q(issues: list[str], noise: float = 2.0) -> QualityReport:
        return QualityReport(cap.capture_id, 50.0, 1.0, 120.0, 0.0, 0.0, noise, 0.0, [], issues, 0.0, not issues)

    assert fitted.predict(cap).reliable and fitted.predict(cap, quality=q([])).reliable
    tolerated = fitted.predict(cap, quality=q(["noisy"], 8.0))
    assert tolerated.reliable and "quality issues tolerated: noisy" in tolerated.notes
    for issue in ("glare", "blurred", "overexposed"):
        res = fitted.predict(cap, quality=q([issue]))
        assert not res.reliable and f"quality issues: {issue}" in res.notes
        assert res.score == pytest.approx(tolerated.score)  # the map itself is unchanged
