"""Active camera simulator: geometry, physics invariants, determinism, timing (vision.camera)."""

from __future__ import annotations

import math
import time
from dataclasses import replace

import cv2
import numpy as np
import pytest

from inspectagent.config import CameraConfig
from inspectagent.types import CaptureSettings, NuisanceConditions, View
from inspectagent.vision.camera import (
    AF_RESIDUAL_SIGMA,
    LIGHT_GAIN,
    PRESETS,
    VirtualCamera,
    linear_to_srgb,
    make_conditions,
    srgb_to_linear,
)
from inspectagent.vision.quality import detect_glare, laplacian_sharpness
from tests.synth import BOARD_SIZE, make_board

W, H = BOARD_SIZE
CLEAN = NuisanceConditions(noise_sigma=0.0, preset="clean")


def mean_luma(img: np.ndarray) -> float:
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


def dot_scene(x: int, y: int) -> np.ndarray:
    img = np.full((H, W, 3), 40, np.uint8)
    cv2.circle(img, (x, y), 4, (255, 255, 255), -1)
    return img


def centroid(img: np.ndarray) -> tuple[float, float]:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g = np.clip(g - 100.0, 0, None)
    m = cv2.moments(g)
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def apply(m: np.ndarray, x: float, y: float) -> tuple[float, float]:
    return float(m[0, 0] * x + m[0, 1] * y + m[0, 2]), float(m[1, 0] * x + m[1, 1] * y + m[1, 2])


# ---------------------------------------------------------------------------------- geometry
def test_sensor_size_follows_scene_aspect(board):
    cam = VirtualCamera(board, CLEAN)
    assert cam.scene_size == (W, H)
    assert cam.sensor_size == (768, 512)
    portrait = VirtualCamera(np.zeros((800, 400, 3), np.uint8), CLEAN, CameraConfig(sensor_long_side=300))
    assert portrait.sensor_size == (150, 300)
    gray = VirtualCamera(cv2.cvtColor(board, cv2.COLOR_BGR2GRAY), CLEAN)
    assert gray.capture(CaptureSettings(), "g", 0).image.shape == (512, 768, 3)


def test_view_window_geometry_and_clamping(board):
    cam = VirtualCamera(board, CLEAN)
    assert cam.view_window(View()) == pytest.approx((0.0, 0.0, W, H))
    assert cam.view_window(View(0.5, 0.5, 2.0)) == pytest.approx((W / 4, H / 4, W / 2, H / 2))
    # shifted, not shrunk, at the borders
    assert cam.view_window(View(0.0, 0.0, 2.0)) == pytest.approx((0.0, 0.0, W / 2, H / 2))
    assert cam.view_window(View(1.0, 1.0, 4.0)) == pytest.approx((W - W / 4, H - H / 4, W / 4, H / 4))
    assert cam.view_window(View(0.9, 0.1, 3.0)) == pytest.approx((W - W / 3, 0.0, W / 3, H / 3))
    # zoom and centre are clamped to the valid range
    assert cam.view_window(View(0.5, 0.5, 0.3)) == pytest.approx((0.0, 0.0, W, H))
    assert cam.view_window(View(0.5, 0.5, 10.0)) == pytest.approx(cam.view_window(View(0.5, 0.5, 4.0)))
    assert cam.view_window(View(-3.0, 7.0, 2.0)) == pytest.approx(cam.view_window(View(0.0, 1.0, 2.0)))
    for v in (View(0.2, 0.8, 1.7), View(0.95, 0.05, 3.3)):
        x0, y0, ww, wh = cam.view_window(v)
        assert 0 <= x0 <= W - ww + 1e-9 and 0 <= y0 <= H - wh + 1e-9


@pytest.mark.parametrize("view", [View(), View(0.3, 0.6, 2.0), View(1.0, 0.0, 4.0), View(0.47, 0.52, 3.1)])
def test_scene_capture_mappings_are_inverse(board, view):
    cap = VirtualCamera(board, CLEAN).capture(CaptureSettings(view=view), "c", 0)
    a = np.vstack([cap.scene_to_capture(), [0, 0, 1]])
    b = np.vstack([cap.capture_to_scene(), [0, 0, 1]])
    assert np.allclose(a @ b, np.eye(3), atol=1e-9) and np.allclose(b @ a, np.eye(3), atol=1e-9)
    assert cap.window == pytest.approx(VirtualCamera(board, CLEAN).view_window(view))
    assert cap.scene_size == (W, H) and cap.size == (768, 512)


@pytest.mark.parametrize("view", [View(), View(0.4, 0.45, 2.0), View(0.35, 0.4, 3.7)])
def test_rendered_geometry_matches_scene_to_capture(view):
    x, y = 350, 260
    cam = VirtualCamera(dot_scene(x, y), CLEAN)
    cap = cam.clean_capture(view, "o", 0)
    assert cap.oracle
    ex, ey = apply(cap.scene_to_capture(), x, y)
    cx, cy = centroid(cap.image)
    assert abs(cx - ex) < 0.6 and abs(cy - ey) < 0.6


def test_pose_jitter_is_fixed_and_reported():
    x, y = 420, 250
    cond = NuisanceConditions(noise_sigma=0.0, rot_deg=2.0, shift_x=0.02, shift_y=-0.015)
    cam = VirtualCamera(dot_scene(x, y), cond)
    for i, view in enumerate([View(), View(0.5, 0.45, 2.5)]):
        cap = cam.clean_capture(view, "o", i)
        tx, ty = apply(cam.true_scene_to_capture(cap), x, y)
        cx, cy = centroid(cap.image)
        assert abs(cx - tx) < 0.7 and abs(cy - ty) < 0.7
        nx, ny = apply(cap.scene_to_capture(), x, y)  # commanded mapping ignores the jitter
        assert math.hypot(cx - nx, cy - ny) > 3.0
    # oracle and degraded capture share the same pose
    deg = VirtualCamera(dot_scene(x, y), replace(cond, noise_sigma=1.0)).capture(CaptureSettings(), "c", 0)
    assert np.allclose(centroid(deg.image), centroid(cam.clean_capture(View(), "o", 0).image), atol=0.6)


# ---------------------------------------------------------------------------------- photometry
def test_srgb_linear_roundtrip():
    x = np.arange(256, dtype=np.uint8).reshape(16, 16)
    assert np.array_equal(linear_to_srgb(srgb_to_linear(x)), x)
    assert linear_to_srgb(np.array([[-0.5, 2.0]], np.float32)).tolist() == [[0, 255]]


def test_exposure_is_monotonic(board):
    cam = VirtualCamera(board, make_conditions("nominal", np.random.default_rng(1)), seed=1)
    lumas = [mean_luma(cam.capture(CaptureSettings(exposure_ev=ev), "c", i).image) for i, ev in enumerate([-2, -1, 0, 1, 2])]
    assert all(b > a + 3 for a, b in zip(lumas, lumas[1:])), lumas


def test_ambient_and_light_gain_ordering(board):
    lumas = {}
    for amb in (-2.0, 0.0, 1.5):
        cam = VirtualCamera(board, NuisanceConditions(noise_sigma=0.0, ambient_ev=amb))
        lumas[amb] = mean_luma(cam.capture(CaptureSettings(), "c", 0).image)
    assert lumas[-2.0] < lumas[0.0] < lumas[1.5]
    cam = VirtualCamera(board, CLEAN)
    by_light = {lm: mean_luma(cam.capture(CaptureSettings(light=lm), "c", 0).image) for lm in LIGHT_GAIN}
    assert by_light["ring"] > by_light["diffuse"] > by_light["low_angle"]
    # diffuse is ~0.74 EV dimmer than ring: +log2(1/0.6) EV compensates it closely
    comp = mean_luma(cam.capture(CaptureSettings(light="diffuse", exposure_ev=math.log2(1 / 0.6)), "c", 0).image)
    assert comp == pytest.approx(by_light["ring"], abs=1.5)


def test_low_angle_adds_directional_relief(board):
    cam = VirtualCamera(board, CLEAN)
    ev = math.log2(1 / 0.5)  # compensate the light gain so only the relief differs
    ring = cv2.cvtColor(cam.capture(CaptureSettings(light="ring"), "c", 0).image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    low = cv2.cvtColor(cam.capture(CaptureSettings(light="low_angle", exposure_ev=ev), "c", 0).image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = low - ring
    gx = cv2.Sobel(cv2.GaussianBlur(ring, (0, 0), 1.0), cv2.CV_32F, 1, 0)
    corr = float(np.corrcoef(diff.ravel(), gx.ravel())[0, 1])
    assert corr > 0.3, corr  # shading follows the horizontal gradient (light from one side)


def test_clean_capture_has_no_degradations(board):
    cond = NuisanceConditions(ambient_ev=-2.0, defocus_sigma=4.0, noise_sigma=5.0, glare_strength=0.9, glare_seed=3)
    cam = VirtualCamera(board, cond)
    oracle = cam.clean_capture(View(0.5, 0.5, 2.0), "o", 0)
    ref = VirtualCamera(board, CLEAN).clean_capture(View(0.5, 0.5, 2.0), "o", 0)
    assert oracle.oracle and np.array_equal(oracle.image, ref.image)
    deg = cam.capture(CaptureSettings(view=View(0.5, 0.5, 2.0)), "c", 0)
    assert not deg.oracle and not np.array_equal(deg.image, oracle.image)


def test_clean_preset_capture_is_close_to_oracle(board):
    cam = VirtualCamera(board, make_conditions("clean", np.random.default_rng(0)))
    cap, oracle = cam.capture(CaptureSettings(), "c", 0), cam.clean_capture(View(), "o", 0)
    assert float(np.abs(cap.image.astype(np.int16) - oracle.image.astype(np.int16)).mean()) < 2.0


def test_noise_grows_with_digital_gain(board):
    cam = VirtualCamera(board, NuisanceConditions(noise_sigma=2.0, ambient_ev=-2.0, jpeg_quality=100))

    def noise_std(ev: float) -> float:
        # two captures differ only by their (index-seeded) noise realisation
        a = cam.capture(CaptureSettings(exposure_ev=ev), "a", 1).image.astype(np.float32)
        b = cam.capture(CaptureSettings(exposure_ev=ev), "b", 2).image.astype(np.float32)
        return float((a - b).std() / math.sqrt(2.0))

    n0, n2 = noise_std(0.0), noise_std(2.0)
    assert n0 > 0.5
    assert n2 > 1.4 * n0  # x4 linear gain -> ~x4**(1/2.2) in sRGB


def test_glare_only_under_ring_light_and_consistent_across_views(board):
    cond = make_conditions("glare", np.random.default_rng(7))
    cam = VirtualCamera(board, cond, seed=3)
    assert 1 <= len(cam.glare_blobs) <= 3
    diag = math.hypot(W, H)
    for b in cam.glare_blobs:
        assert 0.03 * diag - 1e-6 <= b.radius <= 0.08 * diag + 1e-6
        assert 0 <= b.cx <= W and 0 <= b.cy <= H
    ring = cam.capture(CaptureSettings(), "c", 0)
    frac_ring, _ = detect_glare(ring.image)
    assert frac_ring > 0.004
    for i, lm in enumerate(("diffuse", "low_angle")):
        img = cam.capture(CaptureSettings(light=lm, exposure_ev=0.75), "c", i + 1).image
        assert detect_glare(img)[0] == 0.0
        assert float((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) >= 250).mean()) < 0.002
    # the same blob appears at the same scene location in a zoomed view
    blob = cam.glare_blobs[0]
    for k, zoom in enumerate((2.0, 3.5)):
        cap = cam.capture(CaptureSettings(view=View(blob.cx / W, blob.cy / H, zoom)), "z", 10 + k)
        u, v = apply(cap.scene_to_capture(), blob.cx, blob.cy)
        assert cv2.cvtColor(cap.image, cv2.COLOR_BGR2GRAY)[int(v), int(u)] >= 245
    u, v = apply(ring.scene_to_capture(), blob.cx, blob.cy)
    assert cv2.cvtColor(ring.image, cv2.COLOR_BGR2GRAY)[int(v), int(u)] >= 245


def test_glare_blobs_follow_edited_conditions(board):
    cond = NuisanceConditions(noise_sigma=0.0, glare_seed=5)
    cam = VirtualCamera(board, cond)
    assert cam.glare_blobs == ()
    cam.conditions.glare_strength = 0.8
    assert len(cam.glare_blobs) >= 1
    assert detect_glare(cam.capture(CaptureSettings(), "c", 0).image)[0] > 0.004


def test_autofocus_restores_sharpness(board):
    cond = make_conditions("defocus", np.random.default_rng(2))
    cam = VirtualCamera(board, cond, seed=2)
    for view in (View(), View(0.4, 0.5, 2.5)):
        ref = laplacian_sharpness(cam.clean_capture(view, "o", 0).image)
        blurred = laplacian_sharpness(cam.capture(CaptureSettings(view=view), "c", 1).image)
        focused = laplacian_sharpness(cam.capture(CaptureSettings(view=view, autofocus=True), "c", 2).image)
        assert blurred < 0.5 * ref
        assert focused > 0.8 * ref
        assert focused > 2.0 * blurred
    assert AF_RESIDUAL_SIGMA == pytest.approx(0.35)


# ---------------------------------------------------------------------------------- determinism
def test_capture_is_deterministic(board):
    cond = make_conditions("harsh", np.random.default_rng(11))
    s = CaptureSettings(view=View(0.3, 0.6, 2.2), exposure_ev=0.7, light="ring")
    a = VirtualCamera(board, cond, seed=4).capture(s, "a", 3)
    b = VirtualCamera(board, cond, seed=4).capture(s, "b", 3)
    assert np.array_equal(a.image, b.image)
    assert not np.array_equal(a.image, VirtualCamera(board, cond, seed=4).capture(s, "c", 4).image)
    assert not np.array_equal(a.image, VirtualCamera(board, cond, seed=5).capture(s, "d", 3).image)


def test_make_conditions_presets():
    assert PRESETS == ("clean", "nominal", "dark", "bright", "defocus", "glare", "harsh")
    for p in PRESETS:
        a = make_conditions(p, np.random.default_rng(3))
        b = make_conditions(p, np.random.default_rng(3))
        assert a == b and a.preset == p
    c = make_conditions("clean", np.random.default_rng(0))
    assert (c.ambient_ev, c.defocus_sigma, c.noise_sigma, c.glare_strength, c.rot_deg, c.shift_x, c.shift_y) == (0, 0, 0, 0, 0, 0, 0)
    rng = np.random.default_rng(123)
    glare_on = 0
    for _ in range(300):
        n = make_conditions("nominal", rng)
        assert abs(n.ambient_ev) <= 0.3 and 0 <= n.defocus_sigma <= 0.6 and 1 <= n.noise_sigma <= 2
        assert abs(n.rot_deg) <= 1.5 and abs(n.shift_x) <= 0.015 and abs(n.shift_y) <= 0.015
        assert n.glare_strength <= 0.2
        glare_on += n.glare_strength > 0
        d = make_conditions("dark", rng)
        assert -2.4 <= d.ambient_ev <= -1.6 and d.glare_strength == 0
        br = make_conditions("bright", rng)
        assert 1.2 <= br.ambient_ev <= 1.8
        df = make_conditions("defocus", rng)
        assert 2.5 <= df.defocus_sigma <= 4.5
        g = make_conditions("glare", rng)
        assert 0.5 <= g.glare_strength <= 0.9
        for x in (d, br, df, g):
            assert abs(x.rot_deg) <= 2.5 and abs(x.shift_x) <= 0.025 and abs(x.shift_y) <= 0.025
        h = make_conditions("harsh", rng)
        strong = [h.ambient_ev <= -1.6, h.ambient_ev >= 1.2, h.defocus_sigma >= 2.5, h.glare_strength >= 0.5]
        assert 1 <= sum(strong) <= 2
    assert 0.12 < glare_on / 300 < 0.30
    with pytest.raises(ValueError):
        make_conditions("foggy", np.random.default_rng(0))
    assert make_conditions("dark", 5) == make_conditions("dark", np.random.default_rng(5))


def test_settings_are_clamped_and_validated(board):
    cam = VirtualCamera(board, CLEAN)
    cap = cam.capture(CaptureSettings(view=View(1.4, -0.2, 9.0), exposure_ev=7.0), "c", 0)
    assert cap.settings.exposure_ev == 3.0 and cap.settings.view == View(1.0, 0.0, 4.0)
    with pytest.raises(ValueError):
        cam.capture(CaptureSettings(light="laser"), "c", 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        VirtualCamera(np.zeros((10, 10, 2), np.uint8), CLEAN)


# ---------------------------------------------------------------------------------- timing
def test_time_model(board):
    cfg = CameraConfig()
    cam = VirtualCamera(board, CLEAN, cfg)
    base = cfg.t_exposure + cfg.t_readout
    s0 = CaptureSettings()
    assert cam.estimate_time_ms(None, s0) == pytest.approx(base)
    assert cam.estimate_time_ms(s0, s0) == pytest.approx(base)
    moved = CaptureSettings(view=View(0.8, 0.5, 1.0))  # zoom 1: window cannot move
    assert cam.estimate_time_ms(s0, moved) == pytest.approx(base)
    z2 = CaptureSettings(view=View(0.5, 0.5, 2.0))
    assert cam.estimate_time_ms(s0, z2) == pytest.approx(base + cfg.t_zoom_per_step * 1.0)
    a, b = CaptureSettings(view=View(0.2, 0.2, 3.0)), CaptureSettings(view=View(0.8, 0.7, 3.0))
    dist = math.hypot(*(np.subtract(cam.window_center_norm(b.view), cam.window_center_norm(a.view))))
    assert cam.estimate_time_ms(a, b) == pytest.approx(base + cfg.t_move_per_unit * dist)
    assert cam.estimate_time_ms(s0, CaptureSettings(light="diffuse")) == pytest.approx(base + cfg.t_light_switch)
    assert cam.estimate_time_ms(s0, CaptureSettings(autofocus=True)) == pytest.approx(base + cfg.t_autofocus)
    # motion, zoom and light switch overlap; autofocus follows positioning
    combo = CaptureSettings(view=View(0.8, 0.7, 3.0), light="diffuse", autofocus=True)
    t_move = cfg.t_move_per_unit * math.hypot(*(np.subtract(cam.window_center_norm(combo.view), (0.5, 0.5))))
    expected = max(t_move, cfg.t_zoom_per_step * 2.0, cfg.t_light_switch) + cfg.t_autofocus + base
    assert cam.estimate_time_ms(s0, combo) == pytest.approx(expected)
    # exposure compensation is a digital gain: no extra time
    assert cam.estimate_time_ms(s0, CaptureSettings(exposure_ev=2.5)) == pytest.approx(base)


def test_capture_tracks_motion_state(board):
    cam = VirtualCamera(board, CLEAN)
    far = CaptureSettings(view=View(0.9, 0.9, 4.0), autofocus=True)
    c1 = cam.capture(far, "c1", 0)
    assert c1.sim_time_ms == pytest.approx(cam.estimate_time_ms(None, far))
    oracle = cam.clean_capture(View(), "o", 1)  # oracle peeks do not move the camera
    assert oracle.sim_time_ms == pytest.approx(cam.estimate_time_ms(far, CaptureSettings()))
    c2 = cam.capture(far, "c2", 2)
    assert c2.sim_time_ms == pytest.approx(cam.estimate_time_ms(far, far))
    cam.reset_motion()
    assert cam.capture(far, "c3", 3).sim_time_ms == pytest.approx(c1.sim_time_ms)


def test_capture_is_fast(board):
    """768 px sensor capture < 40 ms (best of 10, up to 3 rounds to ride out machine load)."""
    cam = VirtualCamera(board, make_conditions("glare", np.random.default_rng(1)), seed=0)
    settings = [
        CaptureSettings(),
        CaptureSettings(view=View(0.3, 0.4, 2.5), autofocus=True),
        CaptureSettings(view=View(0.7, 0.6, 4.0), light="low_angle", exposure_ev=1.0),
    ]
    for s in settings:
        cam.capture(s, "warm", 0)
        best = float("inf")
        for _round in range(3):
            for i in range(10):
                t0 = time.perf_counter()
                cam.capture(s, "t", i)
                best = min(best, (time.perf_counter() - t0) * 1000.0)
            if best < 40.0:
                break
        assert best < 40.0, (s, best)
