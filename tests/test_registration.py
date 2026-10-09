"""Tests for inspectagent.vision.registration (offline, synthetic boards)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from inspectagent.config import DetectorConfig
from inspectagent.vision import registration as R
from inspectagent.vision.registration import (
    RegistrationResult,
    compose_affine,
    decompose_affine,
    invert_affine,
    register,
    warp_points,
)
from tests.synth import make_board


def _gray_board(seed: int = 3, size: tuple[int, int] | None = (768, 512)) -> np.ndarray:
    img = make_board(seed)
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _jitter(w: int, h: int, rot_deg: float, tx: float, ty: float) -> np.ndarray:
    J = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), rot_deg, 1.0)
    J[0, 2] += tx
    J[1, 2] += ty
    return J


def _max_point_error(M_est: np.ndarray, M_true: np.ndarray, w: int, h: int) -> float:
    pts = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1], [w / 2, h / 2]], float)
    return float(np.abs(warp_points(M_est, pts) - warp_points(M_true, pts)).max())


@pytest.mark.parametrize(
    "rot_deg,tx,ty",
    [(2.0, 10.0, -7.0), (-1.3, -15.5, 4.25), (0.0, 3.0, 2.0), (4.5, 25.0, 18.0)],
)
def test_recovers_known_rotation_and_shift(rot_deg: float, tx: float, ty: float) -> None:
    fixed = _gray_board()
    h, w = fixed.shape
    J = _jitter(w, h, rot_deg, tx, ty)  # fixed -> moving
    moving = cv2.warpAffine(fixed, J, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    res = register(moving, fixed)
    assert isinstance(res, RegistrationResult)
    assert res.ok and res.method in ("orb", "orb+ecc")
    assert res.ncc > 0.9
    M_true = invert_affine(J)  # moving -> fixed
    assert _max_point_error(res.M, M_true, w, h) < 1.0
    _, _, rot_true, _ = decompose_affine(M_true)
    assert abs(res.rot_deg - rot_true) < 0.3
    assert abs(res.scale - 1.0) < 0.01


def test_robust_to_exposure_change_noise_and_mask() -> None:
    fixed = _gray_board(seed=5)
    h, w = fixed.shape
    J = _jitter(w, h, -2.2, 12.0, 9.0)
    moving = cv2.warpAffine(fixed, J, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rng = np.random.default_rng(0)
    lin = (moving.astype(np.float32) / 255.0) ** 2.2 * 2.0  # +1 EV in linear light
    noisy = np.clip(lin ** (1 / 2.2) * 255.0 + rng.normal(0, 3.0, lin.shape), 0, 255).astype(np.uint8)
    valid = np.zeros((h, w), np.uint8)
    valid[30 : h - 30, 40 : w - 40] = 255
    res = register(noisy, fixed, valid_fixed=valid)
    assert res.ok
    assert _max_point_error(res.M, invert_affine(J), w, h) < 1.0


def test_fallback_chain_when_orb_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    fixed = _gray_board(seed=8)
    h, w = fixed.shape
    J = np.array([[1.0, 0.0, 9.4], [0.0, 1.0, -6.3]])
    moving = cv2.warpAffine(fixed, J, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    monkeypatch.setattr(R, "_orb_stage", lambda *a, **k: (None, 0, 0.0, "forced failure"))
    res = register(moving, fixed)
    assert res.ok
    assert res.method in ("ecc", "phase", "phase+ecc")
    assert res.inliers == 0
    assert _max_point_error(res.M, invert_affine(J), w, h) < 1.0


@pytest.mark.parametrize("force_fallback", [False, True], ids=["chain", "orb_failed"])
@pytest.mark.parametrize("rot_deg,corner", [(2.0, 0.05), (-2.0, 0.95), (2.4, 0.95), (-2.4, 0.05)])
def test_corner_zoom_view_with_rotation_is_not_misregistered(monkeypatch, force_fallback, rot_deg, corner) -> None:
    """Regression: translation-only phase correlation (and ECC from it) was accepted at NCC >= 0.45 with a
    7-30 px residual at the corners of zoomed views under pose rotation -> false alarms on normal parts."""
    from inspectagent.types import CaptureSettings, NuisanceConditions, View
    from inspectagent.vision.anomaly_stat import pseudo_capture
    from inspectagent.vision.camera import VirtualCamera

    board = make_board(40)
    cond = NuisanceConditions(rot_deg=rot_deg, shift_x=0.012, shift_y=-0.01, noise_sigma=1.5, defocus_sigma=0.3, preset="test")
    cam = VirtualCamera(board, cond, seed=3)
    view = View(corner, corner, 4.0)
    cap = cam.capture(CaptureSettings(view=view), "c", 0)
    fixed = pseudo_capture(board, view, capture_id="ref").image  # commanded (jitter-free) view of the same part
    if force_fallback:
        monkeypatch.setattr(R, "_orb_stage", lambda *a, **k: (None, 0, 0.0, "forced failure"))
    res = register(cap.image, fixed)
    truth = compose_affine(cap.scene_to_capture(), invert_affine(cam.true_scene_to_capture(cap)))
    w, h = cap.size
    err = _max_point_error(res.M, truth, w, h)
    assert err <= 8.0 or not res.ok, (res.method, res.ncc, err)
    if force_fallback:
        assert res.method in ("ecc", "phase", "phase+ecc")


def test_rotation_grid_phase_recovers_rotation_without_features(monkeypatch) -> None:
    fixed = _gray_board(seed=6)
    h, w = fixed.shape
    J = _jitter(w, h, 2.5, 14.0, -9.0)
    moving = cv2.warpAffine(fixed, J, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    monkeypatch.setattr(R, "_orb_stage", lambda *a, **k: (None, 0, 0.0, "forced failure"))
    monkeypatch.setattr(R._Level, "ecc", lambda self, M, motion: None)  # ECC unavailable: phase alone
    res = register(moving, fixed)
    assert res.ok and res.method == "phase"
    assert _max_point_error(res.M, invert_affine(J), w, h) < 4.0  # 0.5 deg grid: <= 0.25 deg residual


def test_degenerate_input_does_not_raise() -> None:
    blank = np.full((256, 320), 128, np.uint8)
    res = register(blank, blank.copy())
    assert isinstance(res, RegistrationResult)
    assert not res.ok
    assert res.M.shape == (2, 3)


def test_accepts_bgr_and_float_inputs_and_moving_size_differs() -> None:
    board = make_board(11)
    fixed_bgr = cv2.resize(board, (640, 427), interpolation=cv2.INTER_AREA)
    moving = fixed_bgr[10:400, 20:600].astype(np.float32) / 255.0  # crop: moving px + (20, 10) = fixed px
    res = register(moving, fixed_bgr)
    assert res.ok
    expected = np.array([[1.0, 0.0, 20.0], [0.0, 1.0, 10.0]])
    assert _max_point_error(res.M, expected, 580, 390) < 1.0


def test_to_dict_and_affine_helpers() -> None:
    res = register(_gray_board(seed=2), _gray_board(seed=2))
    d = res.to_dict()
    assert set(d) == {"method", "ok", "inliers", "inlier_ratio", "ncc", "dx", "dy", "rot_deg", "scale"}
    assert "M" in res.to_dict(include_M=True)
    assert abs(d["dx"]) < 0.5 and abs(d["dy"]) < 0.5

    A = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]])
    B = cv2.getRotationMatrix2D((10.0, 20.0), 30.0, 1.5)
    p = np.array([[3.0, 4.0], [100.0, -7.0]])
    np.testing.assert_allclose(warp_points(compose_affine(A, B), p), warp_points(A, warp_points(B, p)))
    np.testing.assert_allclose(warp_points(compose_affine(B, invert_affine(B)), p), p, atol=1e-9)
    _, _, rot, scale = decompose_affine(B)
    assert abs(rot + 30.0) < 1e-9 and abs(scale - 1.5) < 1e-9  # y-down frame: CCW-displayed = negative


def test_respects_detector_config_thresholds() -> None:
    fixed = _gray_board(seed=4)
    strict = DetectorConfig(min_ncc=0.9999)
    res = register(cv2.GaussianBlur(fixed, (0, 0), 2.0), fixed, cfg=strict)
    assert not res.ok  # blurred copy cannot reach NCC 0.9999
    assert register(cv2.GaussianBlur(fixed, (0, 0), 2.0), fixed).ok
