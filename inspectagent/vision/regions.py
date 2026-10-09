"""Candidate defect regions from anomaly maps, and their physical measurements (SPEC §2.8).

Frames
------
``extract_regions`` accepts a map either in the capture frame (``residual_M=None``) or in a
*registered* frame (e.g. the commanded-view frame of the statistical detector) together
with ``residual_M``, the 2x3 affine mapping **capture px -> map px**.  Boxes are mapped to
capture px through ``inverse(residual_M)`` and then to scene px through
``Capture.capture_to_scene()``; ``bbox_scene`` is therefore expressed in the commanded
(stage) scene frame, i.e. the frame in which ``View``/window coordinates are defined, so a
follow-up capture centred on ``Region.center_norm()`` looks at the same physical spot.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from inspectagent.types import Capture, Region
from inspectagent.vision.registration import invert_affine, warp_points

__all__ = ["extract_regions", "measure_region", "TYPE_GUESSES"]

TYPE_GUESSES = ("scratch", "spot", "missing_or_extra", "contamination", "unknown")

_OPEN_K = np.ones((3, 3), np.uint8)
_CLOSE_K = np.ones((5, 5), np.uint8)


def _box_corners(x: float, y: float, w: float, h: float) -> np.ndarray:
    return np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=np.float64)


def _bbox_of(points: np.ndarray) -> tuple[float, float, float, float]:
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def extract_regions(
    amap: np.ndarray,
    thr: float,
    capture: Capture,
    residual_M: np.ndarray | None,
    source: str,
    min_area: int,
    max_regions: int,
    id_prefix: str,
) -> list[Region]:
    """Threshold -> open 3x3 -> close 5x5 -> connected components -> top-N by ``peak*sqrt(area)``.

    ``amap`` is float (any scale) with the same size as the capture; ``residual_M`` maps
    capture px -> ``amap`` px (None when ``amap`` is already in the capture frame).
    Region ids are ``f"{id_prefix}{rank}"`` (rank starts at 1; ``id_prefix`` defaults to
    ``f"{capture.capture_id}_r"`` when empty).
    """
    a = np.asarray(amap, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError("amap must be a 2-D array")
    if not np.isfinite(thr):
        return []
    prefix = id_prefix or f"{capture.capture_id}_r"
    binary = (a > float(thr)).astype(np.uint8)
    if not binary.any():
        return []
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _OPEN_K)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _CLOSE_K)
    n, labels, stats, _cent = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return []

    cw, ch = capture.size
    to_capture = invert_affine(residual_M) if residual_M is not None else None
    det_map_to_cap = abs(float(np.linalg.det(to_capture[:, :2]))) if to_capture is not None else 1.0
    c2s = capture.capture_to_scene()
    cap_px_area_scene = abs(float(np.linalg.det(c2s[:, :2])))

    # per-label peak / mean / count in one pass (labels are dense ints)
    flat_l = labels.ravel()
    flat_a = a.ravel()
    sel = flat_l > 0
    lab = flat_l[sel]
    vals = flat_a[sel]
    counts = np.bincount(lab, minlength=n).astype(np.float64)
    sums = np.bincount(lab, weights=vals, minlength=n)
    peaks = np.full(n, -np.inf)
    np.maximum.at(peaks, lab, vals)

    cands = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area) or counts[i] <= 0:
            continue
        peak = float(peaks[i])
        mean = float(sums[i] / counts[i])
        cands.append((peak * math.sqrt(area), i, peak, mean, area))
    cands.sort(key=lambda t: (-t[0], t[1]))

    regions: list[Region] = []
    for rank, (_rk, i, peak, mean, area) in enumerate(cands[: max(0, int(max_regions))], start=1):
        x, y, w, h = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        corners = _box_corners(x, y, w, h)
        cap_pts = warp_points(to_capture, corners) if to_capture is not None else corners
        cap_pts[:, 0] = np.clip(cap_pts[:, 0], 0.0, float(cw))
        cap_pts[:, 1] = np.clip(cap_pts[:, 1], 0.0, float(ch))
        bx0, by0, bw, bh = _bbox_of(cap_pts)
        ix0, iy0 = int(math.floor(bx0)), int(math.floor(by0))
        ix1, iy1 = int(math.ceil(bx0 + bw)), int(math.ceil(by0 + bh))
        ix0, iy0 = max(0, min(ix0, cw - 1)), max(0, min(iy0, ch - 1))
        ix1, iy1 = max(ix0 + 1, min(ix1, cw)), max(iy0 + 1, min(iy1, ch))
        scene_box = _bbox_of(warp_points(c2s, cap_pts))
        area_cap = float(area) * det_map_to_cap
        regions.append(
            Region(
                region_id=f"{prefix}{rank}",
                capture_id=capture.capture_id,
                source=source,
                bbox_capture=(ix0, iy0, ix1 - ix0, iy1 - iy0),
                bbox_scene=tuple(float(v) for v in scene_box),
                peak=peak,
                mean=mean,
                area_px=int(round(area_cap)),
                area_scene_px=float(area_cap * cap_px_area_scene),
            )
        )
    return regions


def _component_mask(roi_map: np.ndarray) -> np.ndarray:
    """Half-maximum component (uint8 0/1) of ``roi_map`` that contains its maximum."""
    peak = float(np.max(roi_map)) if roi_map.size else 0.0
    if not np.isfinite(peak) or peak <= 0:
        return np.zeros(roi_map.shape, np.uint8)
    binary = (roi_map >= 0.5 * peak).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return binary
    yx = np.unravel_index(int(np.argmax(roi_map)), roi_map.shape)
    lab = int(labels[yx])
    if lab == 0:
        lab = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == lab).astype(np.uint8)


def _contrast_mask(gray_roi: np.ndarray) -> np.ndarray:
    """Fallback segmentation without an anomaly map: Otsu on |gray - local median|."""
    g = gray_roi.astype(np.uint8)
    k = max(3, (min(g.shape) // 2) | 1)
    k = min(k, 31)
    bg = cv2.medianBlur(g, k if k % 2 == 1 else k + 1)
    diff = cv2.absdiff(g, bg)
    if int(diff.max()) == 0:
        return np.zeros(g.shape, np.uint8)
    _, binary = cv2.threshold(diff, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _component_mask(binary.astype(np.float32) * diff.astype(np.float32))


def _guess_type(area_mm2: float, length_mm: float, width_mm: float, elongation: float, fill: float) -> str:
    """Heuristic defect type from shape (documented rules, scene-scale based).

    * ``scratch``: elongated and thin (elongation >= 3 and width <= 1 mm).
    * ``spot``: small compact blob (area <= 1.5 mm^2, elongation < 2.5).
    * ``missing_or_extra``: large compact, box-like blob (area > 4 mm^2, fill >= 0.6, elongation < 3)
      — a missing or extra component/feature.
    * ``contamination``: medium/large irregular blob (fill < 0.6 and area > 0.5 mm^2).
    * ``unknown``: anything else.
    """
    if elongation >= 3.0 and width_mm <= 1.0:
        return "scratch"
    if area_mm2 <= 1.5 and elongation < 2.5:
        return "spot"
    if area_mm2 > 4.0 and fill >= 0.6 and elongation < 3.0:
        return "missing_or_extra"
    if fill < 0.6 and area_mm2 > 0.5:
        return "contamination"
    return "unknown"


def measure_region(region: Region, capture: Capture, amap: np.ndarray | None, px_per_mm: float) -> dict:
    """Physical size/shape of a region (``minAreaRect`` of the thresholded component in its bbox).

    ``amap`` must be in the capture frame (as ``AnomalyResult.anomaly_map``); when None the
    component is segmented from local contrast of the capture. ``px_per_mm`` is in scene px.
    Returns ``area_mm2, length_mm, width_mm, elongation, orientation_deg`` (long axis, scene
    frame, [0, 180)), ``contrast`` (mean gray inside minus surrounding ring, 0..255 scale,
    signed) and ``type_guess``.
    """
    img = capture.image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.astype(np.uint8)
    H, W = gray.shape[:2]
    x, y, w, h = (int(v) for v in region.bbox_capture)
    pad = 2
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    empty = {
        "area_mm2": 0.0,
        "length_mm": 0.0,
        "width_mm": 0.0,
        "elongation": 1.0,
        "orientation_deg": 0.0,
        "contrast": 0.0,
        "type_guess": "unknown",
    }
    if x1 <= x0 or y1 <= y0:
        return empty
    if amap is not None:
        roi = np.asarray(amap, dtype=np.float32)[y0:y1, x0:x1]
        comp = _component_mask(roi)
    else:
        comp = _contrast_mask(gray[y0:y1, x0:x1])
    if int(comp.sum()) == 0:
        return empty

    ys, xs = np.nonzero(comp)
    pts = np.stack([xs + x0, ys + y0], axis=1).astype(np.float32)
    if len(pts) >= 3:
        (_cx, _cy), (rw, rh), ang = cv2.minAreaRect(pts)
    else:
        rw, rh, ang = float(np.ptp(xs) + 1), float(np.ptp(ys) + 1), 0.0
    # rectangle through pixel centres -> pixel extent (+1 px per axis)
    rw, rh = float(rw) + 1.0, float(rh) + 1.0
    # long-axis direction vector in capture px
    theta = math.radians(ang) if rw >= rh else math.radians(ang + 90.0)
    long_px, short_px = max(rw, rh), min(rw, rh)
    c2s = capture.capture_to_scene()
    kx, ky = float(c2s[0, 0]), float(c2s[1, 1])
    u = np.array([math.cos(theta) * kx, math.sin(theta) * ky])
    v = np.array([-math.sin(theta) * kx, math.cos(theta) * ky])
    length_scene = long_px * float(np.linalg.norm(u))
    width_scene = short_px * float(np.linalg.norm(v))
    orient = math.degrees(math.atan2(u[1], u[0])) % 180.0
    area_scene = float(len(pts)) * abs(kx * ky)
    ppm = float(px_per_mm) if px_per_mm and px_per_mm > 0 else 1.0
    area_mm2 = area_scene / (ppm * ppm)
    length_mm = length_scene / ppm
    width_mm = width_scene / ppm
    elongation = length_scene / max(width_scene, 1e-6)
    fill = float(len(pts)) / max(rw * rh, 1.0)

    # contrast: component vs. a ring around it (capture gray)
    full = np.zeros((H, W), np.uint8)
    full[y0:y1, x0:x1] = comp
    ring = cv2.dilate(full, np.ones((7, 7), np.uint8), iterations=1)
    ring = (ring > 0) & (full == 0)
    inside = gray[full > 0].astype(np.float64)
    outside = gray[ring].astype(np.float64)
    contrast = float(inside.mean() - outside.mean()) if outside.size and inside.size else 0.0

    return {
        "area_mm2": float(area_mm2),
        "length_mm": float(length_mm),
        "width_mm": float(width_mm),
        "elongation": float(elongation),
        "orientation_deg": float(orient),
        "contrast": contrast,
        "type_guess": _guess_type(area_mm2, length_mm, width_mm, elongation, min(fill, 1.0)),
    }
