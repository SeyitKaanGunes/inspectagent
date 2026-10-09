"""Tests for inspectagent.vision.regions (synthetic anomaly maps, no model needed)."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from inspectagent.types import Capture, CaptureSettings, Region, View
from inspectagent.vision.regions import TYPE_GUESSES, extract_regions, measure_region

W, H = 768, 512
SCENE = (1536, 1024)


def _capture(window: tuple[float, float, float, float] = (0.0, 0.0, 1536.0, 1024.0), image: np.ndarray | None = None) -> Capture:
    img = image if image is not None else np.full((H, W, 3), 40, np.uint8)
    zoom = SCENE[0] / window[2]
    return Capture(
        capture_id="cap_7",
        index=0,
        settings=CaptureSettings(view=View(0.5, 0.5, zoom)),
        image=img,
        window=window,
        scene_size=SCENE,
        sim_time_ms=0.0,
    )


def _blob_map() -> np.ndarray:
    amap = np.zeros((H, W), np.float32)
    cv2.circle(amap, (200, 150), 12, 9.0, -1)  # strong, medium
    cv2.circle(amap, (600, 400), 25, 6.0, -1)  # weaker but larger
    cv2.circle(amap, (400, 300), 2, 20.0, -1)  # tiny (area < min_area after opening)
    return amap


def test_extract_regions_ranking_ids_and_boxes() -> None:
    cap = _capture()
    regs = extract_regions(_blob_map(), 4.0, cap, None, "fast", min_area=60, max_regions=5, id_prefix="reg_3_")
    assert [r.region_id for r in regs] == ["reg_3_1", "reg_3_2"]
    big, small = regs  # 6*sqrt(area~1960) > 9*sqrt(area~450)
    assert big.peak == pytest.approx(6.0) and small.peak == pytest.approx(9.0)
    x, y, w, h = big.bbox_capture
    assert abs(x + w / 2 - 600) <= 1.5 and abs(y + h / 2 - 400) <= 1.5 and 48 <= w <= 53
    # identity window scale 2: scene box = 2x capture box
    sx, sy, sw, sh = big.bbox_scene
    assert sx == pytest.approx(2 * x, abs=1e-6) and sw == pytest.approx(2 * w, abs=1e-6)
    assert big.area_scene_px == pytest.approx(4 * big.area_px, rel=1e-6)
    assert big.source == "fast" and big.capture_id == "cap_7"
    assert isinstance(Region.from_dict(big.to_dict()), Region)


def test_extract_regions_limits_and_threshold() -> None:
    cap = _capture()
    amap = _blob_map()
    assert extract_regions(amap, 50.0, cap, None, "fast", 12, 5, "r") == []
    assert len(extract_regions(amap, 4.0, cap, None, "fast", 12, 1, "r")) == 1
    assert extract_regions(amap, float("nan"), cap, None, "fast", 12, 5, "r") == []
    regs = extract_regions(amap, 4.0, cap, None, "fast", 12, 5, "")
    assert regs[0].region_id == "cap_7_r1"  # default prefix


def test_extract_regions_maps_through_residual_and_zoom_window() -> None:
    # zoomed view: window of 384x256 scene px at (700, 300) -> 2 capture px per scene px
    window = (700.0, 300.0, 384.0, 256.0)
    cap = _capture(window)
    amap = np.zeros((H, W), np.float32)
    cv2.rectangle(amap, (300, 200), (339, 229), 10.0, -1)  # 40x30 px block in the *registered* frame
    residual = np.array([[1.0, 0.0, 15.0], [0.0, 1.0, -8.0]])  # capture px -> registered px
    (reg,) = extract_regions(amap, 5.0, cap, residual, "fast", 12, 5, "z_")
    x, y, w, h = reg.bbox_capture
    assert (x, y) == (285, 208) and (w, h) == (40, 30)  # shifted back by inverse(residual)
    sx, sy, sw, sh = reg.bbox_scene
    assert sx == pytest.approx(700 + 285 / 2.0) and sy == pytest.approx(300 + 208 / 2.0)
    assert sw == pytest.approx(20.0) and sh == pytest.approx(15.0)
    cx, cy = reg.center_norm(SCENE)
    assert cx == pytest.approx((700 + (285 + 20) / 2.0) / SCENE[0])
    assert cy == pytest.approx((300 + (208 + 15) / 2.0) / SCENE[1])


def test_morphology_merges_and_cleans() -> None:
    cap = _capture()
    amap = np.zeros((H, W), np.float32)
    amap[100:140, 100:120] = 8.0
    amap[100:140, 122:142] = 8.0  # 2-px gap -> closed by the 5x5 closing
    amap[300, 300] = 50.0  # single-pixel spike -> removed by the 3x3 opening
    regs = extract_regions(amap, 4.0, cap, None, "fast", 12, 5, "m")
    assert len(regs) == 1
    assert regs[0].bbox_capture[2] >= 42


def _measure_capture(draw) -> tuple[Capture, np.ndarray]:
    img = np.full((H, W, 3), 60, np.uint8)
    amap = np.zeros((H, W), np.float32)
    draw(img, amap)
    amap = cv2.GaussianBlur(amap, (0, 0), 1.0)
    return _capture(image=img), amap


@pytest.mark.parametrize("angle_deg", [0.0, 30.0, 120.0])
def test_measure_scratch(angle_deg: float) -> None:
    L = 120
    c = np.array([380.0, 260.0])
    d = np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))])
    p1, p2 = tuple(int(v) for v in c - d * L / 2), tuple(int(v) for v in c + d * L / 2)

    def draw(img, amap):
        cv2.line(img, p1, p2, (220, 220, 220), 3)
        cv2.line(amap, p1, p2, 12.0, 3)

    cap, amap = _measure_capture(draw)
    (reg,) = extract_regions(amap, 3.0, cap, None, "fast", 12, 5, "s")
    m = measure_region(reg, cap, amap, px_per_mm=20.0)
    assert set(m) == {"area_mm2", "length_mm", "width_mm", "elongation", "orientation_deg", "contrast", "type_guess"}
    assert m["type_guess"] == "scratch"
    assert m["elongation"] > 8
    diff = abs((m["orientation_deg"] - angle_deg + 90) % 180 - 90)
    assert diff < 4.0
    # length in scene px = 2x capture px (window 1536 px over 768 px); 20 px/mm
    assert m["length_mm"] == pytest.approx(2 * L / 20.0, rel=0.12)
    assert m["contrast"] > 100  # bright scratch on dark background


def test_measure_spot_and_missing_block() -> None:
    def spot(img, amap):
        cv2.circle(img, (200, 200), 6, (10, 10, 10), -1)
        cv2.circle(amap, (200, 200), 6, 9.0, -1)

    cap, amap = _measure_capture(spot)
    (reg,) = extract_regions(amap, 3.0, cap, None, "fast", 12, 5, "p")
    m = measure_region(reg, cap, amap, px_per_mm=20.0)
    assert m["type_guess"] == "spot"
    assert m["contrast"] < -20  # dark spot
    assert m["elongation"] < 1.5

    def block(img, amap):
        cv2.rectangle(img, (300, 150), (380, 210), (120, 160, 90), -1)
        cv2.rectangle(amap, (300, 150), (380, 210), 7.0, -1)

    cap, amap = _measure_capture(block)
    (reg,) = extract_regions(amap, 3.0, cap, None, "fast", 12, 5, "b")
    m = measure_region(reg, cap, amap, px_per_mm=20.0)
    assert m["type_guess"] == "missing_or_extra"
    assert m["area_mm2"] == pytest.approx(81 * 61 * 4 / 400.0, rel=0.15)


def test_measure_without_anomaly_map_and_degenerate_box() -> None:
    def spot(img, amap):
        cv2.circle(img, (500, 120), 8, (230, 230, 230), -1)

    cap, _ = _measure_capture(spot)
    reg = Region("q1", "cap_7", "fast", (480, 100, 40, 40), (960.0, 200.0, 80.0, 80.0), 5.0, 3.0, 300, 1200.0)
    m = measure_region(reg, cap, None, px_per_mm=20.0)
    assert m["type_guess"] in TYPE_GUESSES
    assert m["contrast"] > 50 and m["area_mm2"] > 0
    off = Region("q2", "cap_7", "fast", (W + 10, H + 10, 5, 5), (0.0, 0.0, 1.0, 1.0), 1.0, 1.0, 1, 1.0)
    assert measure_region(off, cap, None, 20.0)["area_mm2"] == 0.0
