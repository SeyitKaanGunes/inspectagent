"""Visual evidence rendering: heatmaps, region boxes, quality badges, thumbnails, JPEG/data URLs.

All functions return new BGR uint8 images and never modify their inputs. Text is drawn only
on uint8 images (OpenCV 5 asserts on text drawing for other depths).
"""

from __future__ import annotations

import base64
from collections.abc import Iterable

import cv2
import numpy as np

from inspectagent.types import QualityReport, Region
from inspectagent.vision.imageio import encode_jpeg, to_uint8

__all__ = [
    "heatmap_overlay",
    "draw_regions",
    "draw_quality",
    "thumbnail",
    "encode_jpeg",
    "jpeg_data_url",
    "ISSUE_COLORS",
]

ISSUE_COLORS: dict[str, tuple[int, int, int]] = {
    "blurred": (0, 140, 255),  # orange
    "underexposed": (200, 90, 20),  # blue
    "overexposed": (40, 200, 255),  # amber
    "noisy": (180, 60, 180),  # purple
    "glare": (0, 255, 255),  # yellow
}
_OK_COLOR = (60, 170, 60)
_GLARE_BOX_COLOR = (0, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _bgr_u8(img: np.ndarray) -> np.ndarray:
    """Copy of ``img`` as contiguous BGR uint8."""
    arr = to_uint8(np.asarray(img))
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return cv2.cvtColor(arr[..., 0], cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected a gray/BGR image, got shape {arr.shape}")
    return np.ascontiguousarray(arr).copy()


def _style(img: np.ndarray) -> tuple[int, float, int]:
    """(line thickness, font scale, text thickness) scaled to the image size."""
    long_side = max(img.shape[:2])
    thick = max(1, int(round(long_side / 400.0)))
    scale = max(0.4, long_side / 1400.0)
    return thick, scale, max(1, int(round(scale * 1.6)))


def _text_color_for(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = bg
    return (0, 0, 0) if (0.114 * b + 0.587 * g + 0.299 * r) > 140 else (255, 255, 255)


def _label(img: np.ndarray, text: str, org: tuple[int, int], bg: tuple[int, int, int], scale: float, tt: int) -> int:
    """Draw ``text`` on a filled box with top-left corner ``org``; returns the box width."""
    (tw, th), base = cv2.getTextSize(text, _FONT, scale, tt)
    h, w = img.shape[:2]
    pad = max(2, int(round(3 * scale)))
    x = int(min(max(0, org[0]), max(0, w - tw - 2 * pad)))
    y = int(min(max(0, org[1]), max(0, h - th - base - 2 * pad)))
    cv2.rectangle(img, (x, y), (x + tw + 2 * pad, y + th + base + 2 * pad), bg, -1)
    cv2.putText(img, text, (x + pad, y + pad + th), _FONT, scale, _text_color_for(bg), tt, cv2.LINE_AA)
    return tw + 2 * pad


def heatmap_overlay(
    bgr: np.ndarray, amap: np.ndarray, vmax: float | None = None, alpha: float = 0.45
) -> np.ndarray:
    """Blend an anomaly map onto the image with ``COLORMAP_INFERNO``.

    The map is resized to the image if needed and normalized to ``[0, vmax]`` (default: the
    99.5th percentile of positive values). Values below 20 % of ``vmax`` are left
    uncoloured, with a smooth ramp to full ``alpha`` at 35 %.
    """
    base = _bgr_u8(bgr)
    h, w = base.shape[:2]
    a = np.nan_to_num(np.asarray(amap, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if a.ndim == 3:
        a = a.max(axis=2)
    if a.shape != (h, w):
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    a = np.maximum(a, 0.0)
    if vmax is None:
        pos = a[a > 0]
        vmax = float(np.percentile(pos, 99.5)) if pos.size else 0.0
    vmax = float(vmax)
    if not np.isfinite(vmax) or vmax <= 1e-12:
        return base
    norm = np.clip(a / vmax, 0.0, 1.0)
    colored = cv2.applyColorMap((norm * 255.0 + 0.5).astype(np.uint8), cv2.COLORMAP_INFERNO)
    lo, hi = 0.20, 0.35
    t = np.clip((norm - lo) / (hi - lo), 0.0, 1.0)
    weight = (float(np.clip(alpha, 0.0, 1.0)) * t * t * (3.0 - 2.0 * t))[..., None]
    out = base.astype(np.float32) * (1.0 - weight) + colored.astype(np.float32) * weight
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _as_region(r: Region | dict) -> Region:
    return r if isinstance(r, Region) else Region.from_dict(r)


def draw_regions(
    bgr: np.ndarray,
    regions: Iterable[Region | dict],
    color: tuple[int, int, int] = (0, 0, 255),
    label: bool = True,
) -> np.ndarray:
    """Draw region boxes (``bbox_capture``) with optional ``"<region_id> <peak>"`` labels."""
    img = _bgr_u8(bgr)
    h, w = img.shape[:2]
    thick, scale, tt = _style(img)
    color = tuple(int(c) for c in color)
    for reg in regions:
        r = _as_region(reg)
        x, y, bw, bh = (int(round(v)) for v in r.bbox_capture)
        x0, y0 = max(0, min(w - 1, x)), max(0, min(h - 1, y))
        x1, y1 = max(0, min(w - 1, x + max(1, bw) - 1)), max(0, min(h - 1, y + max(1, bh) - 1))
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thick, cv2.LINE_AA)
        if label:
            text = f"{r.region_id} {r.peak:.2f}"
            (_, th), base = cv2.getTextSize(text, _FONT, scale, tt)
            box_h = th + base + 2 * max(2, int(round(3 * scale)))
            ty = y0 - box_h if y0 - box_h >= 0 else min(h - box_h, y1 + 1)
            _label(img, text, (x0, ty), color, scale, tt)
    return img


def draw_quality(bgr: np.ndarray, q: QualityReport | dict) -> np.ndarray:
    """Glare boxes + issue badges (top-left) + a metrics strip (bottom) for a quality report."""
    if isinstance(q, dict):
        q = QualityReport(**{k: q[k] for k in QualityReport.__dataclass_fields__ if k in q})
    img = _bgr_u8(bgr)
    h, w = img.shape[:2]
    thick, scale, tt = _style(img)
    for box in q.glare_boxes:
        x, y, bw, bh = (int(v) for v in box)
        cv2.rectangle(img, (x, y), (x + bw - 1, y + bh - 1), _GLARE_BOX_COLOR, thick + 1, cv2.LINE_AA)
    margin = max(4, int(round(6 * scale)))
    x = margin
    badges = [(i.upper(), ISSUE_COLORS.get(i, (0, 0, 255))) for i in q.issues] or [("QUALITY OK", _OK_COLOR)]
    (_, th), base = cv2.getTextSize("Ag", _FONT, scale, tt)
    row_h = th + base + 2 * max(2, int(round(3 * scale))) + margin // 2
    y = margin
    for text, col in badges:
        (tw, _), _ = cv2.getTextSize(text, _FONT, scale, tt)
        if x + tw + 2 * margin > w and x > margin:
            x, y = margin, y + row_h
        x += _label(img, text, (x, y), col, scale, tt) + margin // 2
    ratio = "n/a" if q.sharpness_ratio is None else f"{q.sharpness_ratio:.2f}x"
    strip = (
        f"luma {q.mean_luma:.0f}  sharp {q.sharpness:.1f} ({ratio})  noise {q.noise_sigma:.1f}  "
        f"glare {100.0 * q.glare_frac:.2f}%  clip {100.0 * q.clipped_frac:.1f}%  dEV {q.suggested_exposure_delta:+.2f}"
    )
    s_scale = scale * 0.85
    (sw, sh), sbase = cv2.getTextSize(strip, _FONT, s_scale, tt)
    if sw > w - 2 * margin:
        s_scale *= (w - 2 * margin) / max(sw, 1)
        (sw, sh), sbase = cv2.getTextSize(strip, _FONT, s_scale, tt)
    y0 = h - sh - sbase - 2 * margin
    roi = img[max(0, y0) : h, :]
    roi[:] = (roi.astype(np.float32) * 0.4).astype(np.uint8)
    cv2.putText(img, strip, (margin, h - sbase - margin), _FONT, s_scale, (255, 255, 255), tt, cv2.LINE_AA)
    return img


def thumbnail(bgr: np.ndarray, long_side: int = 384) -> np.ndarray:
    """Downscale (``INTER_AREA``) so the long side is at most ``long_side``; never upscales."""
    img = _bgr_u8(bgr)
    h, w = img.shape[:2]
    long_side = max(1, int(long_side))
    if max(h, w) <= long_side:
        return img
    s = long_side / float(max(h, w))
    size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def jpeg_data_url(img: np.ndarray, quality: int = 85, long_side: int | None = None) -> str:
    """``data:image/jpeg;base64,...`` for embedding an image in JSON (optionally thumbnailed)."""
    src = thumbnail(img, long_side) if long_side else img
    return "data:image/jpeg;base64," + base64.b64encode(encode_jpeg(src, quality)).decode("ascii")
