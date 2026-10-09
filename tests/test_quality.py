"""Image-quality assessment on simulator captures of synthetic boards (vision.quality)."""

from __future__ import annotations

import json
import math
from functools import lru_cache

import cv2
import numpy as np
import pytest

from inspectagent.config import QualityConfig
from inspectagent.types import QUALITY_ISSUES, Capture, CaptureSettings, NuisanceConditions, View
from inspectagent.vision.camera import VirtualCamera, make_conditions
from inspectagent.vision.quality import (
    absolute_sharpness_threshold,
    assess_quality,
    detect_glare,
    estimate_noise_sigma,
    glare_mask,
    laplacian_sharpness,
)
from tests.synth import make_board

SEEDS = range(6)
EXPECTED = {
    "clean": set(),
    "nominal": set(),
    "dark": {"underexposed"},
    "bright": {"overexposed"},
    "defocus": {"blurred"},
    "glare": {"glare"},
}


@lru_cache(maxsize=None)
def _board(seed: int) -> np.ndarray:
    img = make_board(seed)
    img.setflags(write=False)  # shared between tests; VirtualCamera copies its scene
    return img


def _camera(preset: str, seed: int) -> VirtualCamera:
    return VirtualCamera(_board(seed), make_conditions(preset, np.random.default_rng(1000 + seed)), seed=seed)


def _ref(cam: VirtualCamera, view: View = View()) -> float:
    return laplacian_sharpness(cam.clean_capture(view, "ref", 99).image)


def _as_capture(img: np.ndarray, cid: str = "x") -> Capture:
    h, w = img.shape[:2]
    return Capture(cid, 0, CaptureSettings(), img, (0.0, 0.0, float(w), float(h)), (w, h), 0.0)


# ---------------------------------------------------------------------------------- primitives
def test_laplacian_sharpness_properties(board):
    s0 = laplacian_sharpness(board)
    assert s0 > 0
    assert laplacian_sharpness(cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)) == pytest.approx(s0)
    assert laplacian_sharpness(cv2.GaussianBlur(board, (0, 0), 3.0)) < 0.3 * s0
    # invariant to global gain (exposure) and to a smooth additive veil
    darker = np.clip(board.astype(np.float32) * 0.5, 0, 255).astype(np.uint8)
    assert laplacian_sharpness(darker) == pytest.approx(s0, rel=0.1)
    yy, xx = np.mgrid[0 : board.shape[0], 0 : board.shape[1]]
    veil = 40.0 * np.exp(-((xx - 450.0) ** 2 + (yy - 300.0) ** 2) / (2 * 200.0**2))
    veiled = np.clip(board.astype(np.float32) + veil[..., None], 0, 255).astype(np.uint8)
    assert laplacian_sharpness(veiled) == pytest.approx(s0, rel=0.15)
    assert laplacian_sharpness(np.zeros((8, 8), np.uint8)) == 0.0


def test_absolute_threshold_decreases_with_magnification():
    assert absolute_sharpness_threshold(0.5) == absolute_sharpness_threshold(1.0)
    assert absolute_sharpness_threshold(1.0) > absolute_sharpness_threshold(2.0) > absolute_sharpness_threshold(4.0) > 0


def test_noise_estimator_tracks_added_noise():
    rng = np.random.default_rng(0)
    flat = np.full((300, 400), 120.0, np.float32)
    for sigma in (2.0, 5.0, 10.0):
        noisy = np.clip(flat + rng.normal(0, sigma, flat.shape), 0, 255).astype(np.uint8)
        assert estimate_noise_sigma(noisy) == pytest.approx(sigma, rel=0.2)


def test_noise_estimator_ignores_structure(board):
    clean = VirtualCamera(board, NuisanceConditions(noise_sigma=0.0)).clean_capture(View(), "o", 0).image
    assert estimate_noise_sigma(clean) < 1.5  # edges of a sharp, noise-free board are not "noise"
    noise = np.random.default_rng(1).normal(0, 8, clean.shape[:2])[..., None]  # same in all channels
    noisy = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    assert estimate_noise_sigma(noisy) == pytest.approx(8.0, rel=0.25)


# ---------------------------------------------------------------------------------- glare
def test_glare_detected_on_ring_light_glare_and_masked(board):
    cam = VirtualCamera(board, make_conditions("glare", np.random.default_rng(4)), seed=4)
    img = cam.capture(CaptureSettings(), "c", 0).image
    frac, boxes = detect_glare(img)
    assert frac > QualityConfig().max_glare_frac and boxes
    for b in cam.glare_blobs:  # every simulated blob centre is inside a reported box
        u, v = b.cx * img.shape[1] / board.shape[1], b.cy * img.shape[0] / board.shape[0]
        assert any(x <= u <= x + w and y <= v <= y + h for x, y, w, h in boxes)
    mask = glare_mask(img)
    assert mask.dtype == np.uint8 and set(np.unique(mask)) <= {0, 255}
    assert np.count_nonzero(mask) / mask.size == pytest.approx(frac)


def test_glare_ignores_bright_pads_silkscreen_and_blocks(board):
    # overexposure clips white pads/silkscreen/pins, at every zoom level: not glare
    for seed in range(4):
        cam = _camera("bright", seed)
        for i, view in enumerate([View(), View(0.6, 0.4, 2.0), View(0.6, 0.4, 3.0), View(0.5, 0.5, 4.0), View(0.3, 0.3, 4.0)]):
            assert detect_glare(cam.capture(CaptureSettings(view=view), "c", i).image)[0] == 0.0, (seed, view)
    # a large, hard-edged white block (connector/silkscreen area) has no specular halo
    block = board.copy()
    cv2.rectangle(block, (300, 200), (420, 290), (255, 255, 255), -1)
    cv2.circle(block, (650, 420), 45, (252, 252, 252), -1, cv2.LINE_AA)
    assert detect_glare(block) == (0.0, [])
    # clean / nominal boards
    for seed in SEEDS:
        for preset in ("clean", "nominal"):
            assert detect_glare(_camera(preset, seed).capture(CaptureSettings(), "c", 0).image)[0] == 0.0


def test_glare_needs_soft_halo():
    img = np.full((512, 768, 3), (30, 90, 40), np.uint8)
    yy, xx = np.mgrid[0:512, 0:768].astype(np.float32)
    d = np.sqrt((xx - 380) ** 2 + (yy - 250) ** 2) / 50.0
    g = np.exp(-(np.maximum(d - 1.0, 0) / 0.5) ** 2)  # plateau + Gaussian halo (white light)
    lin = ((img.astype(np.float32) / 255.0) ** 2.2) + 1.6 * g[..., None]
    soft = (np.clip(lin, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)
    frac, boxes = detect_glare(soft)
    assert frac > 0.004 and len(boxes) == 1
    hard = img.copy()
    cv2.circle(hard, (380, 250), 62, (255, 255, 255), -1)
    assert detect_glare(hard)[0] == 0.0


# ---------------------------------------------------------------------------------- per preset
@pytest.mark.parametrize("with_reference", [True, False], ids=["reference", "absolute"])
@pytest.mark.parametrize("preset", list(EXPECTED))
def test_each_preset_triggers_its_issue(preset, with_reference):
    for seed in SEEDS:
        cam = _camera(preset, seed)
        cap = cam.capture(CaptureSettings(), f"cap_{seed}", 0)
        q = assess_quality(cap, _ref(cam) if with_reference else None)
        assert set(q.issues) <= set(QUALITY_ISSUES)
        if EXPECTED[preset]:
            assert EXPECTED[preset] <= set(q.issues), (preset, seed, q.to_dict())
            # a single degradation is not misattributed to blur/glare/exposure
            others = set(q.issues) - EXPECTED[preset] - {"noisy"}
            assert not others, (preset, seed, q.issues)
        else:
            assert q.ok and q.issues == [], (preset, seed, q.to_dict())


def test_report_fields(board):
    cam = _camera("glare", 1)
    cap = cam.capture(CaptureSettings(), "cap_7", 0)
    q = assess_quality(cap, reference_sharpness=_ref(cam))
    assert q.capture_id == "cap_7" and q.ok == (not q.issues)
    assert q.sharpness_ratio == pytest.approx(q.sharpness / _ref(cam))
    assert 0 <= q.clipped_frac <= 1 and 0 <= q.crushed_frac <= 1 and 0 <= q.glare_frac <= 1
    assert all(len(b) == 4 and all(isinstance(v, int) for v in b) for b in q.glare_boxes)
    assert -3.0 <= q.suggested_exposure_delta <= 3.0 and q.latency_ms > 0
    json.dumps(q.to_dict())
    q2 = assess_quality(cap, reference_sharpness=0.0)
    assert q2.sharpness_ratio is None  # unusable reference -> absolute heuristic


def test_suggested_exposure_closes_the_loop():
    cfg = QualityConfig()
    for preset, sign, issue in (("dark", 1, "underexposed"), ("bright", -1, "overexposed")):
        for seed in range(3):
            cam = _camera(preset, seed)
            q1 = assess_quality(cam.capture(CaptureSettings(), "c1", 0), _ref(cam))
            assert issue in q1.issues and math.copysign(1, q1.suggested_exposure_delta) == sign
            q2 = assess_quality(cam.capture(CaptureSettings(exposure_ev=q1.suggested_exposure_delta), "c2", 1), _ref(cam))
            assert issue not in q2.issues, (preset, seed, q1.to_dict(), q2.to_dict())
            if preset == "dark":
                assert abs(q2.mean_luma - cfg.target_luma) < abs(q1.mean_luma - cfg.target_luma)


def test_corrective_actions_clear_glare_and_blur():
    for seed in range(3):
        cam = _camera("glare", seed)
        ref = _ref(cam)
        q1 = assess_quality(cam.capture(CaptureSettings(), "c1", 0), ref)
        q2 = assess_quality(cam.capture(CaptureSettings(light="diffuse", exposure_ev=0.75), "c2", 1), ref)
        assert "glare" in q1.issues and q2.glare_frac == 0.0 and "glare" not in q2.issues
        cam = _camera("defocus", seed)
        ref = _ref(cam)
        q1 = assess_quality(cam.capture(CaptureSettings(), "c1", 0), ref)
        q2 = assess_quality(cam.capture(CaptureSettings(autofocus=True), "c2", 1), ref)
        assert "blurred" in q1.issues and q1.sharpness_ratio < 0.55
        assert "blurred" not in q2.issues and q2.sharpness_ratio > 0.8


@pytest.mark.parametrize("view", [View(0.4, 0.6, 2.0), View(0.6, 0.4, 3.0), View(0.5, 0.5, 4.0)])
def test_zoomed_views(view):
    for seed in range(3):
        nominal = _camera("nominal", seed)
        q = assess_quality(nominal.capture(CaptureSettings(view=view), "c", 0), _ref(nominal, view))
        assert "blurred" not in q.issues and "glare" not in q.issues and "overexposed" not in q.issues
        defocus = _camera("defocus", seed)
        cap = defocus.capture(CaptureSettings(view=view), "c", 0)
        assert "blurred" in assess_quality(cap, _ref(defocus, view)).issues
        assert "blurred" in assess_quality(cap, None).issues


def test_glare_pixels_do_not_drive_exposure():
    for seed in SEEDS:
        q = assess_quality(_camera("glare", seed).capture(CaptureSettings(), "c", 0))
        assert "overexposed" not in q.issues and q.suggested_exposure_delta > 0  # board itself is dim


def _bright_object_on_dark(seed: int = 0) -> np.ndarray:
    """A pcb4/cashew-like scene: a bright, textured part with glossy highlights on a dark tray."""
    rng = np.random.default_rng(seed)
    h, w = 600, 900
    img = np.full((h, w, 3), 30, np.uint8)
    obj = np.zeros((h, w), np.uint8)
    cv2.ellipse(obj, (450, 300), (235, 130), 12.0, 0, 360, 255, -1)
    tex = cv2.GaussianBlur(rng.normal(0, 20, (h, w)).astype(np.float32), (0, 0), 4)
    part = np.clip(np.array([150, 186, 214], np.float32)[None, None, :] + tex[..., None], 0, 255).astype(np.uint8)
    img[obj > 0] = part[obj > 0]
    for _ in range(14):  # small glossy ridges near saturation (not glare: hard-edged, small)
        c = (int(rng.integers(300, 600)), int(rng.integers(220, 380)))
        cv2.circle(img, c, int(rng.integers(4, 9)), (244, 247, 250), -1, cv2.LINE_AA)
    return img


def _exposure_chain(cam: VirtualCamera, reference: dict | None, max_captures: int = 4) -> list[tuple[float, list[str], float]]:
    """Apply the suggested exposure delta until no exposure issue remains: [(ev, exposure issues, delta)]."""
    ev, seq = 0.0, []
    for i in range(max_captures):
        q = assess_quality(cam.capture(CaptureSettings(exposure_ev=ev), f"c{i}", i), None, reference_exposure=reference)
        issues = [x for x in q.issues if x in ("underexposed", "overexposed")]
        seq.append((ev, issues, q.suggested_exposure_delta))
        if not issues or abs(q.suggested_exposure_delta) < 0.05:
            break
        ev = float(np.clip(ev + q.suggested_exposure_delta, -3.0, 3.0))
    return seq


def test_bright_part_on_dark_background_exposure_is_feasible_and_converges():
    """Regression: whole-frame luma limits made dark-background parts unfixable and the corrections oscillated."""
    from inspectagent.vision.quality import exposure_stats, reference_exposure_from

    scene = _bright_object_on_dark()
    clean = VirtualCamera(scene, NuisanceConditions(noise_sigma=0.0, preset="clean"), seed=0)
    ref = reference_exposure_from([exposure_stats(clean.clean_capture(View(), "ref", 0).image)])
    assert ref is not None and ref["mean_lin"] > 0
    # the part's own normal appearance at nominal exposure is acceptable against its reference
    q0 = assess_quality(clean.capture(CaptureSettings(), "c0", 0), None, reference_exposure=ref)
    assert not {"underexposed", "overexposed"} & set(q0.issues), q0.to_dict()
    # absolute regime: never brightened past the highlights, no under/over flip-flop
    seq_abs = _exposure_chain(VirtualCamera(scene, NuisanceConditions(noise_sigma=1.0), seed=1), None)
    assert not seq_abs[-1][1] and len(seq_abs) <= 3, seq_abs
    flips = [i for i in range(1, len(seq_abs)) if seq_abs[i][1] and seq_abs[i - 1][1] and seq_abs[i][1] != seq_abs[i - 1][1]]
    assert len(flips) <= 1, seq_abs
    # a real exposure error (ambient +-2 EV) is flagged against the reference and fixed within 2 corrections
    for amb, issue in ((-2.0, "underexposed"), (2.0, "overexposed")):
        cam = VirtualCamera(scene, NuisanceConditions(ambient_ev=amb, noise_sigma=1.0), seed=2)
        seq = _exposure_chain(cam, ref)
        assert issue in seq[0][1], (amb, seq)
        assert not seq[-1][1] and len(seq) <= 3, (amb, seq)


def test_positive_exposure_delta_respects_highlight_headroom():
    from inspectagent.vision.quality import exposure_stats

    scene = _bright_object_on_dark(3)
    cam = VirtualCamera(scene, NuisanceConditions(ambient_ev=-0.4, noise_sigma=1.0), seed=3)
    cap = cam.capture(CaptureSettings(), "c", 0)
    q = assess_quality(cap)
    headroom = math.log2(((249 / 255.0) ** 2.2) / exposure_stats(cap.image)["p99_lin"])
    assert q.suggested_exposure_delta <= max(headroom, 0.0) + 1e-6
    q2 = assess_quality(cam.capture(CaptureSettings(exposure_ev=q.suggested_exposure_delta), "c2", 1))
    assert q2.clipped_frac <= max(q.clipped_frac, 0.0) + QualityConfig().max_clipped_frac


def test_gray_and_tiny_inputs():
    gray = cv2.cvtColor(_board(0), cv2.COLOR_BGR2GRAY)
    q = assess_quality(_as_capture(gray))
    assert isinstance(q.ok, bool)
    tiny = assess_quality(_as_capture(np.full((6, 6, 3), 128, np.uint8)))
    assert tiny.sharpness == 0.0 and "blurred" in tiny.issues
