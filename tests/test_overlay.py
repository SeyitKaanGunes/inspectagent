"""Overlay rendering (vision.overlay)."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from inspectagent.types import QualityReport, Region
from inspectagent.vision.imageio import decode_image
from inspectagent.vision.overlay import draw_quality, draw_regions, heatmap_overlay, jpeg_data_url, thumbnail


@pytest.fixture
def img(board):
    return cv2.resize(board, (768, 512), interpolation=cv2.INTER_AREA)


def _region(rid: str = "reg_1_1", box=(100, 120, 60, 40), peak: float = 7.5) -> Region:
    return Region(rid, "cap_1", "fast", box, (117.0, 140.0, 70.0, 47.0), peak, 3.0, box[2] * box[3], 3290.0)


def _quality(issues: list[str], boxes: list[list[int]] | None = None) -> QualityReport:
    return QualityReport(
        capture_id="cap_1", sharpness=40.0, sharpness_ratio=0.42, mean_luma=55.0, clipped_frac=0.001,
        crushed_frac=0.0, noise_sigma=3.2, glare_frac=0.031 if boxes else 0.0, glare_boxes=boxes or [],
        issues=issues, suggested_exposure_delta=1.2, ok=not issues,
    )


def test_heatmap_overlay_colors_only_hot_areas(img):
    orig = img.copy()
    amap = np.zeros((128, 192), np.float32)
    amap[40:60, 100:130] = 10.0
    out = heatmap_overlay(img, amap)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(img, orig)  # input untouched
    hot = np.zeros(img.shape[:2], bool)
    hot[180:220, 420:500] = True  # inside the (4x upscaled) hot block
    cold = np.zeros(img.shape[:2], bool)
    cold[:100, :200] = True
    diff = np.abs(out.astype(np.int16) - img.astype(np.int16)).sum(axis=2)
    assert diff[hot].mean() > 20 and diff[cold].max() == 0


def test_heatmap_overlay_edge_cases(img):
    assert np.array_equal(heatmap_overlay(img, np.zeros(img.shape[:2], np.float32)), img)
    nanmap = np.full(img.shape[:2], np.nan, np.float32)
    assert np.array_equal(heatmap_overlay(img, nanmap), img)
    amap = np.zeros(img.shape[:2], np.float32)
    amap[200:260, 300:380] = 1.0
    strong = heatmap_overlay(img, amap, vmax=1.0, alpha=0.9)
    weak = heatmap_overlay(img, amap, vmax=10.0, alpha=0.9)  # 0.1 of vmax -> below the mask threshold
    assert not np.array_equal(strong, img) and np.array_equal(weak, img)
    gray_out = heatmap_overlay(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), amap)
    assert gray_out.shape == img.shape


def test_draw_regions(img):
    reg = _region()
    out = draw_regions(img, [reg], color=(0, 0, 255), label=False)
    assert out.shape == img.shape and not np.array_equal(out, img)
    assert tuple(out[140, 100]) == (0, 0, 255)  # left edge of the box
    labelled = draw_regions(img, [reg])
    assert not np.array_equal(labelled, out)
    as_dict = draw_regions(img, [reg.to_dict()], label=False)
    assert np.array_equal(as_dict, out)
    edge = draw_regions(img, [_region("reg_9", (-20, 490, 900, 80), 1.0), _region("reg_8", (5, 0, 30, 20), 2.0)])
    assert edge.shape == img.shape
    assert np.array_equal(draw_regions(img, []), img)


def test_draw_quality(img):
    q_bad = _quality(["blurred", "glare"], boxes=[[300, 200, 120, 90]])
    out = draw_quality(img, q_bad)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert tuple(out[250, 300]) == (0, 255, 255)  # glare box outline
    q_ok = _quality([])
    ok_out = draw_quality(img, q_ok)
    assert not np.array_equal(ok_out, img) and not np.array_equal(ok_out, out)
    assert np.array_equal(draw_quality(img, q_ok.to_dict()), ok_out)
    small = draw_quality(cv2.resize(img, (160, 100)), _quality(list(("blurred", "underexposed", "noisy", "glare"))))
    assert small.shape == (100, 160, 3)


def test_thumbnail(img):
    t = thumbnail(img, 384)
    assert t.shape == (256, 384, 3)
    assert thumbnail(img, 2000).shape == img.shape  # never upscales
    portrait = thumbnail(np.zeros((900, 300), np.uint8), 300)
    assert portrait.shape == (300, 100, 3)


def test_jpeg_data_url(img):
    url = jpeg_data_url(img, quality=80, long_side=200)
    assert url.startswith("data:image/jpeg;base64,")
    dec = decode_image(base64.b64decode(url.split(",", 1)[1]))
    assert max(dec.shape[:2]) == 200
