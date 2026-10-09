"""Image-quality assessment: the visual evidence that drives corrective re-captures.

``assess_quality(capture)`` measures sharpness, exposure, noise and specular glare and
turns them into the issues of :data:`inspectagent.types.QUALITY_ISSUES` plus a suggested
exposure correction (in linear light).

Design notes (calibrated on ``tests/synth`` boards through the camera simulator):

* **Sharpness** is a noise-robust, contrast-normalized Laplacian measure: the variance of
  the Laplacian of the Gaussian-smoothed gray (σ=2) divided by the variance of a mid-band
  difference of Gaussians (σ=3 vs 9), ×1000. A raw ``var(Laplacian)`` is dominated by
  sensor noise and scales with exposure, so a dark or noisy-but-sharp frame would look
  "sharper" or "blurrier" than it is. The band ratio is invariant to gain and to smooth
  additive veils (weak glare) and drops strongly under defocus. Reference values passed as
  ``reference_sharpness`` must be computed with :func:`laplacian_sharpness` (e.g.
  ``assess_quality(clean_capture).sharpness``).
* **Glare** = saturated (V >= 245) low-saturation (S <= 40) blobs that are large relative
  to the frame, compact, and surrounded by a bright *halo gradient* (specular light falls
  off smoothly; a clipped pad or silkscreen glyph has a sharp edge to a darker
  background). This keeps bright pads/silkscreen/overexposed metal from being reported as
  glare.
* **Exposure**: a pixel is *clipped* when any channel is >= 250 (information is lost in that
  channel) and *crushed* when all channels are <= 5. Exposure decisions and the suggested EV
  ignore detected glare pixels (a specular highlight is not an exposure problem — metering
  ignores it); ``mean_luma``, ``clipped_frac`` and ``crushed_frac`` in the report are
  whole-frame values. Two regimes:

  - **reference-relative** (``reference_exposure`` given: ``{"mean_lin", "clipped_frac",
    "crushed_frac"}`` of clean captures of the category at this zoom bucket, stored in the model
    bundle by ``scripts/build_models.py``, see :func:`exposure_stats`): ``ev_off =
    log2(mean_lin / ref.mean_lin)``; *underexposed* when ``ev_off < -REF_EV_TOLERANCE`` or the
    crushed fraction exceeds the reference by more than ``max_crushed_frac``; *overexposed* when
    ``ev_off > +REF_EV_TOLERANCE`` or the clipped fraction exceeds the reference by more than
    ``max_clipped_frac``; ``suggested_exposure_delta = -ev_off``. A part that is bright on a dark
    background (VisA cashew/pcb4/pcb1) is judged against its own normal appearance, not against a
    whole-frame mean that no exposure can satisfy.
  - **absolute** (no reference): mean luma in ``[min_luma, max_luma]`` and clipped/crushed fractions
    below the limits; ``suggested_exposure_delta = log2(target_lin / mean_lin)`` (linear light).
    A low mean is reported as *underexposed* only when the highlights leave at least
    ``MIN_UNDER_HEADROOM_EV`` of headroom - a frame whose bright parts are already near saturation
    cannot be brightened (dark background around a bright object) and is the best exposure available.

  In both regimes, when highlights are clipped beyond the allowance the delta is additionally capped
  at ``-(b + 0.5·log2(1 + excess/max_clipped))`` (>= -2; ``b = 0.5`` absolute, ``0`` against a
  reference, whose mean stays informative), because the mean of a clipped frame understates the
  overexposure; and a *positive* delta is always capped by the highlight
  headroom ``log2(lin(249) / p99_lin)`` (99th percentile of the glare-excluded linear max channel), so
  a brightening step never creates new clipping and corrections cannot oscillate between under- and
  overexposed. The delta is clamped to ±3.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping

import cv2
import numpy as np

from inspectagent.config import QualityConfig
from inspectagent.types import Capture, QualityReport

__all__ = [
    "assess_quality",
    "laplacian_sharpness",
    "estimate_noise_sigma",
    "detect_glare",
    "glare_mask",
    "absolute_sharpness_threshold",
    "blocking_quality_issues",
    "exposure_stats",
    "reference_exposure_from",
    "GLARE_V_MIN",
    "GLARE_S_MAX",
    "NOISE_TOLERANCE_SIGMA",
    "REF_EV_TOLERANCE",
    "MIN_UNDER_HEADROOM_EV",
]

GAMMA = 2.2

# --- detector reliability ---
# Sensor noise up to this estimated sigma (0..255 scale, 2x the default QualityConfig.max_noise_sigma)
# does not invalidate a detector result: the fast detector inflates its per-pixel sigma by the noise
# measured on the capture's own residual (noise lowers sensitivity, it does not create false alarms).
# Above it results are treated as unreliable (checked on synthetic dark/glare captures, noise 7-15;
# the deep detector's patch features drift under heavier noise).
NOISE_TOLERANCE_SIGMA = 14.0


def blocking_quality_issues(quality: QualityReport) -> list[str]:
    """Quality issues that make a detector result on this capture unreliable.

    Every issue blocks except ``noisy`` with ``noise_sigma <= NOISE_TOLERANCE_SIGMA``: glare, clipping,
    crushed shadows and blur destroy or distort what the detectors compare, moderate noise does not.
    The rule planner accepts moderate noise it cannot correct (e.g. after the +0.75 EV compensation of
    the glare fix), so detector results on such captures must stay usable.
    """
    return [i for i in quality.issues if not (i == "noisy" and float(quality.noise_sigma) <= NOISE_TOLERANCE_SIGMA)]

# --- sharpness (band-ratio Laplacian) ---
SHARP_HI_SIGMA = 2.0
SHARP_MID_SIGMAS = (3.0, 9.0)
SHARP_SCALE = 1000.0
# Reference-free heuristic (used only when no reference sharpness is known), threshold
# ABS_SHARP_BASE / max(1, m) ** ABS_SHARP_EXP for magnification m (sensor px per scene px).
# Measured: synthetic boards score ~57 (m<=1), ~25 (m=1.7), ~13 (m=2.6-3.4) when sharp and
# <= 11 / 8.6 / 5.7 with defocus sigma >= 2.5 px; sharp VisA frames (m~0.55) score 20-110.
# It is content dependent (e.g. defocused VisA cashew can reach ~23) - prefer a reference.
ABS_SHARP_BASE = 17.0
ABS_SHARP_EXP = 0.75

# --- glare ---
GLARE_V_MIN = 245
GLARE_S_MAX = 40
GLARE_MIN_AREA_FRAC = 0.004  # core must cover >= 0.4 % of the frame (pads/glyphs are smaller)
GLARE_MIN_AREA_PX = 64
GLARE_MIN_SOLIDITY = 0.75  # specular blobs are compact; traces/text are not
GLARE_MIN_HALO = 0.25  # (inner ring - background ring) / (255 - background ring), min channel
GLARE_RING_FRAC = 0.25  # inner ring width k = 0.25 * equivalent core radius ...
GLARE_RING_MAX = 40.0  # ... capped (px)
GLARE_RING_SKIP_PX = 5.0  # skip the first max(5 px, 0.2 k) outside the core (edge anti-aliasing)
GLARE_RING_SKIP_FRAC = 0.2
GLARE_HUGE_FRAC = 0.25  # a core this large is glare even when no halo is visible in-frame

# --- exposure ---
CLIP_LEVEL = 250  # a pixel is clipped when any channel is >= this (information lost)
CRUSH_LEVEL = 5  # a pixel is crushed when all channels are <= this
_LUT_LIN = (np.arange(256, dtype=np.float64) / 255.0) ** GAMMA
REF_EV_TOLERANCE = 0.75  # |log2(mean_lin / reference mean_lin)| above this is an exposure error (reference regime)
HEADROOM_LEVEL = 249  # brightening may lift the 99th-percentile highlight up to this 8-bit level
HIGHLIGHT_PERCENTILE = 99.0
MIN_UNDER_HEADROOM_EV = 0.25  # absolute regime: a dark mean is an issue only if it can be brightened by this much
_EXPOSURE_KEYS = ("mean_lin", "clipped_frac", "crushed_frac")

# --- noise (Immerkær with edge masking) ---
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
NOISE_EDGE_PERCENTILE = 60.0


def _to_gray(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr if arr.dtype == np.uint8 else np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGRA2GRAY)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"expected a gray or BGR image, got shape {arr.shape}")
    return arr


def laplacian_sharpness(gray: np.ndarray) -> float:
    """Noise-robust, contrast-normalized sharpness (x1000): var(LoG sigma=2) / var(DoG sigma 3-9).

    Accepts gray or BGR images. The high band is the Laplacian of the sigma=2 Gaussian-smoothed
    gray at full resolution; the mid band is a sigma=3 vs sigma=9 difference of Gaussians,
    evaluated on the 2x pyramid level for speed (sigma 1.5 vs 4.5 there). Invariant to global
    gain and to smooth additive veils; drops strongly with defocus. Returns 0.0 for tiny images.
    """
    g = _to_gray(gray).astype(np.float32)
    if g.shape[0] < 16 or g.shape[1] < 16:
        return 0.0
    hi = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), SHARP_HI_SIGMA), cv2.CV_32F)
    half = cv2.pyrDown(g)
    s1, s2 = SHARP_MID_SIGMAS[0] / 2.0, SHARP_MID_SIGMAS[1] / 2.0
    mid = cv2.subtract(cv2.GaussianBlur(half, (0, 0), s1), cv2.GaussianBlur(half, (0, 0), s2))
    v_hi = float(cv2.meanStdDev(hi)[1][0, 0]) ** 2
    v_mid = float(cv2.meanStdDev(mid)[1][0, 0]) ** 2
    return SHARP_SCALE * v_hi / (v_mid + 1.0)


def absolute_sharpness_threshold(magnification: float) -> float:
    """Reference-free blur threshold for a frame sampled at ``magnification`` sensor px / scene px."""
    return ABS_SHARP_BASE / max(1.0, float(magnification)) ** ABS_SHARP_EXP


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer (1996) fast noise estimate (0..255 scale), restricted to low-gradient pixels.

    Edges inflate the plain estimator on structured scenes, so pixels above the 60th
    gradient-magnitude percentile and clipped/crushed pixels (whose noise is truncated)
    are excluded.
    """
    g8 = _to_gray(gray)
    if g8.dtype != np.uint8:
        g8 = np.clip(g8, 0, 255).astype(np.uint8)
    h, w = g8.shape
    if h < 8 or w < 8:
        return 0.0
    g = g8.astype(np.float32)
    resp = cv2.absdiff(cv2.filter2D(g, cv2.CV_32F, _IMMERKAER, borderType=cv2.BORDER_REFLECT), 0.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.add(cv2.absdiff(gx, 0.0), cv2.absdiff(gy, 0.0))  # L1 gradient magnitude
    valid = cv2.inRange(g8, 6, 249)
    valid[0, :] = 0
    valid[-1, :] = 0
    valid[:, 0] = 0
    valid[:, -1] = 0
    if cv2.countNonZero(valid) < 64:
        valid[1:-1, 1:-1] = 255
    sub = mag[::3, ::3][valid[::3, ::3] > 0]
    thr = float(np.percentile(sub, NOISE_EDGE_PERCENTILE)) if sub.size else float("inf")
    sel = cv2.bitwise_and(valid, (mag <= thr).astype(np.uint8) * 255)
    if cv2.countNonZero(sel) < 64:
        sel = valid
    mean_resp = float(cv2.mean(resp, mask=sel)[0])
    return float(math.sqrt(math.pi / 2.0) * mean_resp / 6.0)


def _glare_analysis(bgr: np.ndarray) -> tuple[float, list[list[int]], np.ndarray]:
    """Return ``(glare_frac, boxes, mask)`` where ``mask`` (uint8 0/255) marks accepted glare cores.

    Candidate cores: V >= 245 and S <= 40, opened 3x3, area >= 0.4 % of the frame,
    solidity >= 0.75. Halo test on the *minimum channel* (white specular light raises all
    channels; coloured bright structures such as copper traces do not): mean of the ring
    (d_a, k] px outside the core vs. the background ring (1.5k, 3k], with k = 0.25 * r_eq and
    d_a = max(5 px, 0.2 * k) (skips the anti-aliased edge of a sharply bounded bright object
    such as a pad, while a specular halo is still bright there).
    """
    img = np.asarray(bgr)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    frame = float(h * w)
    out_mask = np.zeros((h, w), np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    core = cv2.bitwise_and(
        cv2.inRange(hsv[..., 2], GLARE_V_MIN, 255), cv2.inRange(hsv[..., 1], 0, GLARE_S_MAX)
    )
    min_area = max(GLARE_MIN_AREA_PX, GLARE_MIN_AREA_FRAC * frame)
    if cv2.countNonZero(core) < min_area:
        return 0.0, [], out_mask
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    minc: np.ndarray | None = None
    boxes: list[list[int]] = []
    total = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        r_eq = math.sqrt(area / math.pi)
        k = float(min(GLARE_RING_MAX, max(GLARE_RING_SKIP_PX + 4.0, GLARE_RING_FRAC * r_eq)))
        d_a = max(GLARE_RING_SKIP_PX, GLARE_RING_SKIP_FRAC * k)
        pad = int(math.ceil(3 * k)) + 1
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        comp = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
        huge = area >= GLARE_HUGE_FRAC * frame
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        hull_area = float(cv2.contourArea(cv2.convexHull(np.vstack(contours))))
        solidity = area / hull_area if hull_area > 0 else 0.0
        if solidity < GLARE_MIN_SOLIDITY and not huge:
            continue
        dist = cv2.distanceTransform(1 - comp, cv2.DIST_L2, 3)
        inner = (dist > d_a) & (dist <= k)
        outer = (dist > 1.5 * k) & (dist <= 3 * k)
        expected_inner = 2.0 * math.pi * (r_eq + 0.5 * (k + d_a)) * (k - d_a)
        if int(inner.sum()) < 0.25 * expected_inner or int(outer.sum()) < 8:
            accepted = huge
        else:
            if minc is None:
                minc = cv2.min(cv2.min(img[..., 0], img[..., 1]), img[..., 2]).astype(np.float32)
            roi = minc[y0:y1, x0:x1]
            m_in = float(roi[inner].mean())
            m_out = float(roi[outer].mean())
            halo = (m_in - m_out) / max(255.0 - m_out, 1.0)
            accepted = halo >= GLARE_MIN_HALO or huge
        if not accepted:
            continue
        out_mask[y0:y1, x0:x1][comp > 0] = 255
        boxes.append([x, y, bw, bh])
        total += area
    boxes.sort(key=lambda b: -b[2] * b[3])
    return float(total / frame), boxes, out_mask


def detect_glare(bgr: np.ndarray) -> tuple[float, list[list[int]]]:
    """Specular glare: ``(fraction of frame covered by glare cores, [[x, y, w, h], ...])``."""
    frac, boxes, _ = _glare_analysis(bgr)
    return frac, boxes


def glare_mask(bgr: np.ndarray) -> np.ndarray:
    """uint8 0/255 mask of the accepted glare cores (capture px)."""
    return _glare_analysis(bgr)[2]


def _measure_exposure(img: np.ndarray, gray: np.ndarray, glare_boxes: list, gmask: np.ndarray) -> dict:
    """Glare-excluded exposure statistics of a BGR uint8 frame (see module docstring)."""
    maxc = cv2.max(cv2.max(img[..., 0], img[..., 1]), img[..., 2])
    keep = None
    if glare_boxes:  # exposure statistics without specular highlights (and their immediate halo)
        k = max(3, int(round(0.02 * max(gray.shape))))
        excl = cv2.dilate(gmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
        keep = cv2.bitwise_not(excl)
        if cv2.countNonZero(keep) < 0.05 * gray.size:
            keep = None
    hist = cv2.calcHist([gray], [0], keep, [256], [0, 256]).ravel().astype(np.float64)
    n_sel = max(1.0, float(hist.sum()))
    hmax = cv2.calcHist([maxc], [0], keep, [256], [0, 256]).ravel().astype(np.float64)
    cum = np.cumsum(hmax)
    p_level = int(np.searchsorted(cum, HIGHLIGHT_PERCENTILE / 100.0 * max(cum[-1], 1.0)))
    if keep is None:
        eff_clipped = float(np.count_nonzero(maxc >= CLIP_LEVEL)) / maxc.size
        eff_crushed = float(np.count_nonzero(maxc <= CRUSH_LEVEL)) / maxc.size
    else:
        eff_clipped = float(np.count_nonzero((maxc >= CLIP_LEVEL) & (keep > 0))) / n_sel
        eff_crushed = float(np.count_nonzero((maxc <= CRUSH_LEVEL) & (keep > 0))) / n_sel
    return {
        "eff_mean": float((hist * np.arange(256)).sum() / n_sel),
        "mean_lin": float((hist * _LUT_LIN).sum() / n_sel),
        "clipped_frac": eff_clipped,
        "crushed_frac": eff_crushed,
        "p99_lin": float(_LUT_LIN[min(255, max(0, p_level))]),
    }


def exposure_stats(image: np.ndarray) -> dict:
    """Glare-excluded exposure statistics of an image: ``{"mean_lin", "clipped_frac", "crushed_frac", "p99_lin"}``.

    The same statistics :func:`assess_quality` judges; computed on clean captures of normal parts they
    form the per-zoom ``reference_exposure`` of a category (see :func:`reference_exposure_from`).
    """
    img = np.asarray(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _frac, boxes, gmask = _glare_analysis(img)
    m = _measure_exposure(img, gray, boxes, gmask)
    return {k: m[k] for k in (*_EXPOSURE_KEYS, "p99_lin")}


def reference_exposure_from(stats: Iterable[Mapping[str, float]]) -> dict | None:
    """Median of :func:`exposure_stats` dicts (clean captures of normals at one zoom bucket); None if empty."""
    rows = [s for s in stats if s and float(s.get("mean_lin", 0.0)) > 0.0]
    if not rows:
        return None
    out = {k: float(np.median([float(r.get(k, 0.0)) for r in rows])) for k in (*_EXPOSURE_KEYS, "p99_lin")}
    out["n"] = len(rows)
    return out


def _valid_reference(ref: Mapping | None) -> Mapping | None:
    if not isinstance(ref, Mapping):
        return None
    try:
        return ref if float(ref.get("mean_lin", 0.0)) > 1e-6 else None
    except (TypeError, ValueError):
        return None


def _magnification(capture: Capture) -> float:
    try:
        x0, y0, ww, wh = capture.window
        w, h = capture.size
        return float(min(w / ww, h / wh)) if ww > 0 and wh > 0 else 1.0
    except Exception:
        return 1.0


def assess_quality(
    capture: Capture,
    reference_sharpness: float | None = None,
    cfg: QualityConfig | None = None,
    reference_exposure: Mapping[str, float] | None = None,
) -> QualityReport:
    """Measure the capture and list blocking quality issues (see module docstring).

    ``reference_exposure`` (optional, ``{"mean_lin", "clipped_frac", "crushed_frac"}`` of clean
    captures of the category at this zoom bucket) switches exposure judgement to the
    reference-relative regime.
    """
    t0 = time.perf_counter()
    cfg = cfg or QualityConfig()
    img = np.asarray(capture.image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sharpness = laplacian_sharpness(gray)
    ratio: float | None = None
    if reference_sharpness is not None and float(reference_sharpness) > 0:
        ratio = sharpness / float(reference_sharpness)

    mean_luma = float(gray.mean())
    maxc = cv2.max(cv2.max(img[..., 0], img[..., 1]), img[..., 2])
    clipped_frac = float(np.count_nonzero(maxc >= CLIP_LEVEL)) / maxc.size
    crushed_frac = float(np.count_nonzero(maxc <= CRUSH_LEVEL)) / maxc.size
    noise_sigma = estimate_noise_sigma(gray)
    glare_frac, glare_boxes, gmask = _glare_analysis(img)

    ex = _measure_exposure(img, gray, glare_boxes, gmask)
    eff_mean, mean_lin = ex["eff_mean"], ex["mean_lin"]
    eff_clipped, eff_crushed = ex["clipped_frac"], ex["crushed_frac"]
    headroom = math.log2(_LUT_LIN[HEADROOM_LEVEL] / max(ex["p99_lin"], 1e-6))
    ref = _valid_reference(reference_exposure)
    if ref is not None:
        ev_off = math.log2(max(mean_lin, 1e-6) / float(ref["mean_lin"]))
        clip_excess = eff_clipped - float(ref.get("clipped_frac", 0.0) or 0.0)
        crush_excess = eff_crushed - float(ref.get("crushed_frac", 0.0) or 0.0)
        under = ev_off < -REF_EV_TOLERANCE or crush_excess > cfg.max_crushed_frac
        over = ev_off > REF_EV_TOLERANCE or clip_excess > cfg.max_clipped_frac
        delta = -ev_off
    else:
        target_lin = (cfg.target_luma / 255.0) ** GAMMA
        clip_excess = eff_clipped
        delta = math.log2(target_lin / max(mean_lin, 1e-6))
        # a dark mean that cannot be lifted without clipping the highlights is the best exposure available
        under = (eff_mean < cfg.min_luma and headroom >= MIN_UNDER_HEADROOM_EV) or eff_crushed > cfg.max_crushed_frac
        over = eff_mean > cfg.max_luma or eff_clipped > cfg.max_clipped_frac
    if clip_excess > cfg.max_clipped_frac:
        # highlights are lost: the mean says little about how far above saturation they are,
        # so step down proportionally to the excess clipping (never brighten a clipped frame);
        # against a reference the mean is still informative, so the minimum step is smaller
        excess = clip_excess / cfg.max_clipped_frac
        base = 0.0 if ref is not None else 0.5
        delta = min(delta, -min(2.0, base + 0.5 * math.log2(1.0 + excess)))
    if delta > 0.0:
        delta = min(delta, max(headroom, 0.0))  # never brighten past the highlight headroom (no oscillation)
    delta = float(min(3.0, max(-3.0, delta)))

    issues: list[str] = []
    if ratio is not None:
        blurred = ratio < cfg.min_sharpness_ratio
    else:
        blurred = sharpness < absolute_sharpness_threshold(_magnification(capture))
    if blurred:
        issues.append("blurred")
    if under:
        issues.append("underexposed")
    if over:
        issues.append("overexposed")
    if noise_sigma > cfg.max_noise_sigma:
        issues.append("noisy")
    if glare_frac > cfg.max_glare_frac:
        issues.append("glare")

    return QualityReport(
        capture_id=capture.capture_id,
        sharpness=float(sharpness),
        sharpness_ratio=None if ratio is None else float(ratio),
        mean_luma=mean_luma,
        clipped_frac=clipped_frac,
        crushed_frac=crushed_frac,
        noise_sigma=float(noise_sigma),
        glare_frac=float(glare_frac),
        glare_boxes=glare_boxes,
        issues=issues,
        suggested_exposure_delta=delta,
        ok=not issues,
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )
