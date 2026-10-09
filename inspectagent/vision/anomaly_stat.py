"""StatTemplateModel — the fast, registration-based statistical template detector (SPEC §2.6).

Frames and transforms
---------------------
* **scene px**: pixel grid of the full-resolution part image (``scene_size = (W_s, H_s)``).
* **template px**: the scene rescaled so that its long side is ``template_long_side``; the
  template is built with ``cv2.resize`` (pixel-centre convention), hence
  ``x_s = (x_t + 0.5) * W_s / W_t - 0.5`` (:meth:`StatTemplateModel.template_to_scene`).
  The template lives in the pose of the *medoid* normal; ``scale = W_t / W_s``.
* **commanded capture px**: ``Capture.scene_to_capture()`` — what the camera was asked to see
  (``cv2.warpAffine`` pixel convention, identical to the camera simulator).
* **capture px**: the actual image. It differs from the commanded frame by an unknown residual
  pose (part placement + camera pose jitter) that :func:`register` estimates as ``M``
  (capture px -> commanded px).

Prediction pipeline (any commanded view, any exposure / light)
--------------------------------------------------------------
1. Render the expected template stack (Lab mean, Lab std, reference gray, validity) into the
   commanded capture grid through ``template -> scene -> capture`` (area pre-filter when
   minifying, cubic when magnifying; cached per view geometry).
2. Register the capture gray to the expected gray (``M``: capture -> commanded frame).
3. Warp the capture into the commanded frame; large fully-saturated plateaus (specular glare,
   blown highlights) carry no information and are removed from the valid mask.
4. Photometric normalisation: (a) a smooth, positive, achromatic illumination field (weak
   glare / halos, larger than any small or thin defect) is estimated in linear light against a
   glare-insensitive gain (inter-channel differences cancel white light) and removed;
   (b) PSF matching — the defocus blur of the capture relative to the template is estimated
   (least-squares search over Gaussian sigmas on the 0.5x level) and applied to the expected
   mean; for ``low_angle`` light a fitted relief-shading term is applied as well; (c) per Lab
   channel robust 2-pass trimmed weighted LSQ ``capture ~ a * mu + b`` (``a`` clamped to
   [0.3, 3]; saturated px excluded); (d) the prediction is passed through the sRGB gamut
   (clip-aware), so highlights clipped by an exposure change are predicted, not flagged.
5. Noise-aware sigma ``sqrt(sigma^2 + (1.2 * noise_c)^2 + (eps * |grad mu_c|)^2)``: ``noise_c``
   is a robust estimate of the residual noise of Lab channel ``c`` at the scale the z-map sees
   (L: MAD of the Immerkaer Laplacian; a/b: also the MAD of the high-passed residual, because JPEG
   4:2:0 chroma subsampling makes chroma noise spatially correlated - see :func:`_robust_noise`);
   the last term models the sub-pixel registration/resampling uncertainty on template edges
   (``eps = REG_UNCERT_PX * max(1, magnification)``).
6. ``z = max_c |x_c - pred_c| / (a_c * sigma_eff_c)`` (= ``|x'-mu|/sigma_eff`` with
   ``x' = (x - b) / a``), Gaussian-blurred, zeroed outside the eroded valid mask.
7. Score = mean of the top-k z values (``k = max(topk_min, topk_frac * N_valid)``).
8. Regions on the z-map, mapped to capture px through ``inverse(M)`` and then to (commanded,
   i.e. stage) scene px through ``capture_to_scene`` (see :mod:`inspectagent.vision.regions`).
"""

from __future__ import annotations

import io
import json
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from inspectagent.config import CameraConfig, DetectorConfig
from inspectagent.types import AnomalyResult, Capture, CaptureSettings, QualityReport, View
from inspectagent.vision.quality import NOISE_TOLERANCE_SIGMA, blocking_quality_issues
from inspectagent.vision.registration import RegistrationResult, compose_affine, invert_affine, register
from inspectagent.vision.regions import extract_regions

__all__ = [
    "StatTemplateModel",
    "ZOOM_BUCKETS",
    "DEFAULT_LOGISTIC",
    "zoom_bucket",
    "capture_zoom",
    "view_window",
    "sensor_size_for",
    "pseudo_capture",
]

ZOOM_BUCKETS = ("z1", "z2", "z3", "z4")
DEFAULT_LOGISTIC = {"a": 1.6, "b": -4.0}
FORMAT_VERSION = 1

# --- template ---------------------------------------------------------------------------
SIGMA_SMOOTH = 1.5  # Gaussian smoothing of the per-pixel std (template px)
# --- photometric / PSF normalisation -------------------------------------------------------
PHOTO_GAIN_RANGE = (0.3, 3.0)
PHOTO_TRIM = 2.5  # trimmed LSQ keeps |r| <= PHOTO_TRIM * robust sigma
PHOTO_MAX_SAMPLES = 16_000
MIN_CHANNEL_STD = 1.0  # Lab units; below this a channel gain is not identifiable
SATURATED_HI = 250
SATURATED_LO = 4
BLUR_GRID = (0.0, 0.6, 1.2, 2.0, 3.0, 4.5)  # candidate PSF sigmas (capture px)
BLUR_APPLY_MIN = 0.3  # smaller estimated PSF sigmas are ignored
GAMMA = 2.2  # transfer curve used for linear-light operations (matches the camera model)
GLARE_R_FRAC = 0.02  # opening radius = GLARE_R_FRAC * scene diagonal (scene px)
GLARE_MIN_LIN = 0.01  # minimal peak (linear light) of an illumination field worth removing
PLATEAU_MIN_FRAC = 0.5  # fraction of saturated px in a block for it to count as a plateau
# --- noise-aware sigma / z-map -----------------------------------------------------------
NOISE_GAIN = 1.2
CHROMA_NOISE_HP_SIGMA = 6.0  # px: a/b noise is measured on resid - Gaussian(resid, 6 px) (see _robust_noise)
REG_UNCERT_PX = 0.35  # registration/resampling uncertainty (capture px at <= 1x magnification)
VALID_ERODE_PX = 3
GLARE_DILATE_PX = 5
MAGNIFY_CUBIC_ABOVE = 1.15
# --- calibration ---------------------------------------------------------------------------
SD_REL_FLOOR = 0.12  # sd >= SD_REL_FLOOR * |mu| (scores vary with light mode / exposure)
SD_ABS_FLOOR = 0.10
REGION_K = 3.0
REGION_PEAK_RATIO = 1.5  # z-map peaks of normals run ~1.2-1.5x their top-k score
REGION_THR_MIN = 4.0
DEFAULT_BUCKET = {"mu": 3.0, "sd": 0.75, "region_thr": 7.9}  # only when nothing was calibrated

_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
_LUT_TO_LIN = ((np.arange(256, dtype=np.float64) / 255.0) ** GAMMA).astype(np.float32)
_LEVEL_CACHE_MAX = 6
_RENDER_CACHE_MAX = 4

# Provisional-calibration views (cx, cy, zoom, light, exposure_ev, ambient_ev or None): the operating
# envelope of the agent (ring light at EV 0, diffuse light with the glare-remedy EV compensation, and a
# dimmed scene compensated by digital gain - the noisier captures the agent accepts after a dark fix).
_PROVISIONAL_VIEWS = (
    (0.5, 0.5, 1.0, "ring", 0.0, None),
    (0.5, 0.5, 1.0, "ring", 1.75, -1.75),
    (0.42, 0.42, 1.25, "diffuse", 0.75, None),
    (0.58, 0.58, 1.25, "ring", 0.0, None),
    (0.3, 0.35, 2.0, "ring", 0.0, None),
    (0.7, 0.65, 2.0, "diffuse", 0.75, None),
    (0.25, 0.7, 3.0, "ring", 0.0, None),
    (0.75, 0.3, 3.0, "diffuse", 0.75, None),
    (0.4, 0.4, 4.0, "ring", 0.0, None),
    (0.6, 0.62, 4.0, "diffuse", 0.75, None),
)


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------
def zoom_bucket(zoom: float) -> str:
    """``"z1"`` (zoom < 1.5), ``"z2"`` (< 2.5), ``"z3"`` (< 3.5), else ``"z4"``."""
    z = float(zoom)
    if z < 1.5:
        return "z1"
    if z < 2.5:
        return "z2"
    if z < 3.5:
        return "z3"
    return "z4"


def capture_zoom(capture: Capture) -> float:
    """Effective zoom of a capture (scene width / window width; falls back to the settings)."""
    try:
        ww = float(capture.window[2])
        if ww > 0:
            return float(capture.scene_size[0]) / ww
    except (TypeError, IndexError, ValueError):
        pass
    return float(capture.settings.view.zoom)


def view_window(scene_size: tuple[int, int], view: View) -> tuple[float, float, float, float]:
    """Window ``(x0, y0, w, h)`` in scene px: size ``scene / zoom``, shifted (not shrunk) into the scene."""
    W, H = float(scene_size[0]), float(scene_size[1])
    z = max(1.0, float(view.zoom))
    ww, wh = W / z, H / z
    x0 = min(max(float(view.cx) * W - ww / 2.0, 0.0), W - ww)
    y0 = min(max(float(view.cy) * H - wh / 2.0, 0.0), H - wh)
    return (x0, y0, ww, wh)


def sensor_size_for(scene_size: tuple[int, int], long_side: int) -> tuple[int, int]:
    """Sensor ``(w, h)`` with the given long side and the scene aspect ratio."""
    W, H = scene_size
    if W >= H:
        return int(long_side), max(1, int(round(long_side * H / W)))
    return max(1, int(round(long_side * W / H))), int(long_side)


def _resize_map(src_size: tuple[int, int], dst_size: tuple[int, int]) -> np.ndarray:
    """2x3 map dst px -> src px for ``cv2.resize(src, dst_size)`` (sizes are (w, h))."""
    kx = src_size[0] / dst_size[0]
    ky = src_size[1] / dst_size[1]
    return np.array([[kx, 0.0, 0.5 * kx - 0.5], [0.0, ky, 0.5 * ky - 0.5]], dtype=np.float64)


def _render(
    src: np.ndarray, dst_to_src: np.ndarray, dsize: tuple[int, int], border: int = cv2.BORDER_REPLICATE
) -> np.ndarray:
    """Sample ``src`` (<= 4 channels) on the ``dsize`` grid through ``dst_to_src``.

    Area pre-filter when minifying (> 1.15x), cubic when magnifying, bilinear otherwise.
    """
    M = np.asarray(dst_to_src, dtype=np.float64)
    k = max(abs(M[0, 0]) + abs(M[0, 1]), abs(M[1, 0]) + abs(M[1, 1]))  # src px per dst px
    interp = cv2.INTER_LINEAR
    if k > 1.15:
        sh, sw = src.shape[:2]
        rs = (max(4, int(round(sw / k))), max(4, int(round(sh / k))))
        small = cv2.resize(src, rs, interpolation=cv2.INTER_AREA)
        M = compose_affine(invert_affine(_resize_map((sw, sh), rs)), M)
        src = small
    elif k < 1.0 / MAGNIFY_CUBIC_ABOVE:
        interp = cv2.INTER_CUBIC
    return cv2.warpAffine(src, M, dsize, flags=interp | cv2.WARP_INVERSE_MAP, borderMode=border)


def pseudo_capture(
    scene_bgr: np.ndarray,
    view: View | None = None,
    long_side: int | None = None,
    capture_id: str = "pseudo",
    index: int = 0,
    scene_size: tuple[int, int] | None = None,
) -> Capture:
    """Clean, jitter-free capture of ``scene_bgr`` at ``view`` (geometry of SPEC §2.3 steps 1-2).

    Used for the provisional calibration and as an offline stand-in for the camera.
    ``scene_size`` lets a re-sampled image stand for a scene of a different pixel size.
    """
    view = view or View()
    img = _as_bgr(scene_bgr)
    ih, iw = img.shape[:2]
    ss = (int(scene_size[0]), int(scene_size[1])) if scene_size else (iw, ih)
    w, h = sensor_size_for(ss, int(long_side or CameraConfig().sensor_long_side))
    cap = Capture(
        capture_id=capture_id,
        index=index,
        settings=CaptureSettings(view=view),
        image=np.zeros((h, w, 3), np.uint8),
        window=view_window(ss, view),
        scene_size=ss,
        sim_time_ms=0.0,
        oracle=True,
    )
    c2s = cap.capture_to_scene()
    if ss != (iw, ih):  # scene px -> image px (resize convention)
        c2s = compose_affine(invert_affine(_resize_map(ss, (iw, ih))), c2s)
    cap.image = _render(img, c2s, (w, h))
    return cap


# --------------------------------------------------------------------------------------
# image / statistics helpers
# --------------------------------------------------------------------------------------
def _as_bgr(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype != np.uint8:
        f = a.astype(np.float32)
        if f.size and float(np.nanmax(f)) <= 1.0 + 1e-6:
            f = f * 255.0
        a = np.clip(f + 0.5, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    elif a.ndim == 3 and a.shape[2] == 4:
        a = cv2.cvtColor(a, cv2.COLOR_BGRA2BGR)
    elif a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"expected a BGR image, got shape {a.shape}")
    return np.ascontiguousarray(a)


def _u8_to_lab(bgr_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.multiply(bgr_u8, np.float32(1.0 / 255.0), dtype=np.float32), cv2.COLOR_BGR2Lab)


def _lab_to_gray_u8(lab: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(np.ascontiguousarray(lab, dtype=np.float32), cv2.COLOR_Lab2BGR)
    return np.clip(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _stride_for(mask: np.ndarray, max_samples: int) -> int:
    """Grid stride so that a strided view of ``mask`` holds about ``max_samples`` true px."""
    n = int(np.count_nonzero(mask))
    return max(1, int(math.ceil(math.sqrt(n / float(max_samples))))) if n > max_samples else 1


def _robust_linear_fit(
    ref: np.ndarray, obs: np.ndarray, w: np.ndarray | None, a_fallback: float | None = None
) -> tuple[float, float]:
    """Weighted 2-pass trimmed LSQ ``obs ~ a * ref + b`` with ``a`` clamped to PHOTO_GAIN_RANGE."""
    ref = ref.astype(np.float64)
    obs = obs.astype(np.float64)
    w = np.ones(ref.shape, np.float64) if w is None else w.astype(np.float64)
    if ref.size < 16:
        return 1.0, (float(np.median(obs - ref)) if ref.size else 0.0)
    keep = np.ones(ref.shape, bool)
    a, b = 1.0, 0.0
    for _ in range(3):  # initial fit + 2 trimmed passes
        ww = w[keep]
        sw = float(ww.sum())
        if sw <= 0 or int(keep.sum()) < 16:
            break
        X, Y = ref[keep], obs[keep]
        mx = float((ww * X).sum() / sw)
        my = float((ww * Y).sum() / sw)
        dx = X - mx
        vx = float((ww * dx * dx).sum() / sw)
        if vx < MIN_CHANNEL_STD**2:
            a = float(a_fallback) if a_fallback is not None else 1.0
        else:
            a = float((ww * dx * (Y - my)).sum() / sw / vx)
        a = float(np.clip(a, *PHOTO_GAIN_RANGE))
        b = my - a * mx
        r = np.abs(obs - (a * ref + b))
        s = max(1.4826 * float(np.median(r[keep])), 0.5)
        keep = r <= PHOTO_TRIM * s
    return a, b


def _photometric_fit(
    mu: np.ndarray, x: np.ndarray, fit_mask: np.ndarray, weight: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-Lab-channel gains/offsets so that ``x ~ a * mu + b`` on ``fit_mask`` (strided sample)."""
    a = np.ones(3, np.float64)
    b = np.zeros(3, np.float64)
    s = _stride_for(fit_mask, PHOTO_MAX_SAMPLES)
    m = fit_mask[::s, ::s]
    if int(np.count_nonzero(m)) < 16:
        return a, b
    mu_s = mu[::s, ::s][m]
    x_s = x[::s, ::s][m]
    w_s = weight[::s, ::s][m] if weight is not None else None
    for c in range(3):
        a[c], b[c] = _robust_linear_fit(
            mu_s[:, c], x_s[:, c], None if w_s is None else w_s[:, c], a_fallback=None if c == 0 else float(a[0])
        )
    return a, b


def _robust_noise(resid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-channel noise std of the Lab residual ``resid`` on ``mask``, at the scale the z-map sees.

    L: MAD of the Immerkaer Laplacian (pixel-scale estimator; luma noise is white).
    a/b: ``max(Immerkaer, MAD of the high-passed residual)``. JPEG 4:2:0 chroma subsampling turns
    chroma noise into 2x2 blobs that the 3x3 Laplacian barely sees (measured: a/b estimated 4-6x too
    low, the error growing with gain), so the a/b noise is also measured as the robust spread of
    ``resid - smooth(resid)`` (normalised Gaussian of sigma ``CHROMA_NOISE_HP_SIGMA`` px on ``mask``):
    the slow component (illumination / photometric-fit residue) is removed, everything up to a few
    px - the scale of the blurred z-map - is kept. The MAD stays robust to sparse defects.
    """
    out = np.zeros(resid.shape[2], np.float64)
    s = _stride_for(mask, 24_000)
    m = mask[::s, ::s]
    if int(np.count_nonzero(m)) < 64:
        return out
    r = np.ascontiguousarray(resid, dtype=np.float32)
    f = cv2.filter2D(r, cv2.CV_32F, _IMMERKAER, borderType=cv2.BORDER_REFLECT)
    vals = np.abs(f[::s, ::s][m])
    out = 1.4826 * np.median(vals, axis=0).astype(np.float64) / 6.0
    if r.ndim == 3 and r.shape[2] >= 3:
        mf = mask.astype(np.float32)
        den = cv2.GaussianBlur(mf, (0, 0), CHROMA_NOISE_HP_SIGMA, borderType=cv2.BORDER_REPLICATE)
        np.maximum(den, 1e-3, out=den)
        for c in (1, 2):
            ch = np.ascontiguousarray(r[..., c])
            smooth = cv2.divide(cv2.GaussianBlur(cv2.multiply(ch, mf), (0, 0), CHROMA_NOISE_HP_SIGMA, borderType=cv2.BORDER_REPLICATE), den)
            hp = cv2.subtract(ch, smooth)[::s, ::s][m].astype(np.float64)
            mad = 1.4826 * float(np.median(np.abs(hp - np.median(hp))))
            out[c] = max(float(out[c]), mad)
    return out


def _estimate_blur(ref_gray: np.ndarray, cap_gray: np.ndarray, mask: np.ndarray) -> float:
    """Gaussian PSF sigma (capture px) that best maps ``ref_gray`` onto ``cap_gray``.

    Least-squares search over ``BLUR_GRID`` (with a linear photometric fit per candidate) on
    the 0.5x level, refined by a parabola through the best three candidates.
    """
    h, w = ref_gray.shape[:2]
    hs = (max(8, w // 2), max(8, h // 2))
    r = cv2.resize(ref_gray.astype(np.float32), hs, interpolation=cv2.INTER_AREA)
    c = cv2.resize(cap_gray.astype(np.float32), hs, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.uint8), hs, interpolation=cv2.INTER_NEAREST) > 0
    s = _stride_for(m, 12_000)
    ms = m[::s, ::s]
    if int(np.count_nonzero(ms)) < 256:
        return 0.0
    y = c[::s, ::s][ms].astype(np.float64)
    errs = []
    for sig in BLUR_GRID:
        e = cv2.GaussianBlur(r, (0, 0), sig / 2.0) if sig > 0 else r
        x = e[::s, ::s][ms].astype(np.float64)
        xm, ym = x.mean(), y.mean()
        vx = float(((x - xm) ** 2).mean())
        a = float(((x - xm) * (y - ym)).mean() / vx) if vx > 1e-9 else 0.0
        errs.append(float(((y - ym - a * (x - xm)) ** 2).mean()))
    i = int(np.argmin(errs))
    if 0 < i < len(BLUR_GRID) - 1:
        x0, x1, x2 = BLUR_GRID[i - 1 : i + 2]
        y0, y1, y2 = errs[i - 1 : i + 2]
        den = (x0 - x1) * (x0 - x2) * (x1 - x2)
        A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den if den else 0.0
        B = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den if den else 0.0
        if A > 0:
            return float(np.clip(-B / (2 * A), x0, x2))
    return float(BLUR_GRID[i])


def _relief_shading(
    gray: np.ndarray, x_lin: np.ndarray, mu_lin: np.ndarray, mask: np.ndarray, px_per_scene: float
) -> np.ndarray | None:
    """Multiplicative low-angle relief shading ``1 + c * R`` of the expected image (or None).

    ``R`` is the normalised horizontal derivative of the (scene-scale smoothed) expected gray,
    i.e. the directional shading a grazing light produces on edges; ``c`` is fitted by robust
    least squares of the linear-light luminance residual on ``mu_Y * R``.
    """
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), max(0.5, float(px_per_scene)))
    r = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    scale = float(np.percentile(np.abs(r[::4, ::4]), 99.0))
    if scale <= 1e-6:
        return None
    r = np.clip(r / scale, -1.0, 1.0)
    wy = (0.0722, 0.7152, 0.2126)  # BGR luminance weights
    yx = cv2.transform(x_lin, np.array([wy], np.float64))
    ym = cv2.transform(mu_lin, np.array([wy], np.float64))
    st = _stride_for(mask, PHOTO_MAX_SAMPLES)
    m = mask[::st, ::st]
    if int(np.count_nonzero(m)) < 64:
        return None
    X = ym[::st, ::st][m].astype(np.float64)
    Y = yx[::st, ::st][m].astype(np.float64)
    Rs = r[::st, ::st][m].astype(np.float64)
    keep = np.ones(X.shape, bool)
    gain, c = 1.0, 0.0
    for _ in range(3):  # Y ~ gain * X + c' * X * R, trimmed
        A = np.stack([X[keep], X[keep] * Rs[keep]], 1)
        sol, *_ = np.linalg.lstsq(A, Y[keep], rcond=None)
        gain, cp = float(sol[0]), float(sol[1])
        res = np.abs(Y - gain * X - cp * X * Rs)
        keep = res <= 2.5 * max(1.4826 * float(np.median(res[keep])), 1e-4)
    if gain <= 0:
        return None
    c = float(np.clip(cp / gain, -1.0, 1.0))
    if abs(c) < 0.02:
        return None
    return (1.0 + c * r).astype(np.float32)


def _glare_free_gain(mu_lin: np.ndarray, x_lin: np.ndarray, mask: np.ndarray) -> float | None:
    """Global linear-light gain ``g`` with ``x_lin ~ g * mu_lin + white light``.

    Fitted on inter-channel differences (G-R, G-B), which additive white light (glare)
    cancels: robust through-origin LSQ with two trimmed passes. None for achromatic scenes.
    """
    s = _stride_for(mask, PHOTO_MAX_SAMPLES)
    m = mask[::s, ::s]
    if int(np.count_nonzero(m)) < 64:
        return None
    M = mu_lin[::s, ::s][m].astype(np.float64)
    X = x_lin[::s, ::s][m].astype(np.float64)
    dm = np.concatenate([M[:, 1] - M[:, 2], M[:, 1] - M[:, 0]])
    dx = np.concatenate([X[:, 1] - X[:, 2], X[:, 1] - X[:, 0]])
    if float(np.sqrt(np.mean(dm * dm))) < 0.01:
        return None
    keep = np.ones(dm.shape, bool)
    g = 1.0
    for _ in range(3):
        den = float((dm[keep] ** 2).sum())
        if den <= 1e-12 or int(keep.sum()) < 32:
            return None
        g = float((dm[keep] * dx[keep]).sum() / den)
        r = np.abs(dx - g * dm)
        sc = max(1.4826 * float(np.median(r[keep])), 1e-4)
        keep = r <= 2.5 * sc
    return float(np.clip(g, 0.02, 50.0)) if np.isfinite(g) and g > 0 else None


def _illumination_field(x_lin: np.ndarray, pred_lin: np.ndarray, valid: np.ndarray, radius: float) -> np.ndarray | None:
    """Smooth, positive, achromatic additive light field (weak glare / halos) in linear light.

    ``min_c(x_lin - pred_lin)`` is positive only where light was *added* to every channel.
    On a 0.25x grid a median filter (window ``2*radius+1``) rejects edge/misregistration
    outliers and thin or small structures (scratches, spots, small blobs); a grey opening with
    the same square removes any remaining bright structure narrower than that; a light Gaussian
    keeps the field smooth. Returns None when the field is negligible (< GLARE_MIN_LIN).
    """
    h, w = valid.shape
    diff = cv2.subtract(x_lin, pred_lin)
    ch = cv2.split(diff)
    amin = cv2.min(cv2.min(ch[0], ch[1]), ch[2])
    vf = valid.astype(np.float32)
    amin = cv2.multiply(amin, vf)
    q = 4
    ssz = (max(4, w // q), max(4, h // q))
    frac = cv2.resize(vf, ssz, interpolation=cv2.INTER_AREA)
    small = np.divide(
        cv2.resize(amin, ssz, interpolation=cv2.INTER_AREA), frac, out=np.zeros(frac.shape, np.float32), where=frac > 0.25
    )
    if float(small.max()) < GLARE_MIN_LIN:
        return None
    rs = max(1, int(round(radius / q)))
    ksz = min(2 * rs + 1, 255)
    holes = frac <= 0.25
    if holes.any():  # fill invalid cells (saturation, borders) from valid neighbours, not with zeros
        wv = (~holes).astype(np.float32)
        num = cv2.GaussianBlur(small * wv, (0, 0), max(1.0, rs / 2.0), borderType=cv2.BORDER_REPLICATE)
        den = cv2.GaussianBlur(wv, (0, 0), max(1.0, rs / 2.0), borderType=cv2.BORDER_REPLICATE)
        small = np.where(holes, np.divide(num, den, out=np.zeros_like(num), where=den > 1e-3), small).astype(np.float32)
    # median on an 8-bit quantisation of [-0.1, 0.9] (OpenCV's large-window median needs uint8)
    q8 = np.clip((small + 0.1) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    med = cv2.medianBlur(q8, ksz).astype(np.float32) / 255.0 - 0.1
    op = cv2.morphologyEx(med, cv2.MORPH_OPEN, np.ones((ksz, ksz), np.uint8), borderType=cv2.BORDER_REPLICATE)
    g0 = cv2.GaussianBlur(np.maximum(op, 0.0), (0, 0), max(1.0, rs / 3.0))
    if float(g0.max()) < GLARE_MIN_LIN:
        return None
    # robust refinement passes (inlier-weighted normalised convolution at 1/2 and 1/4 of the
    # scale) recover peaked tops and border ramps that the median/opening flatten
    vs = cv2.resize(valid.astype(np.uint8), small.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    support = cv2.dilate((g0 >= 0.5 * GLARE_MIN_LIN).astype(np.uint8), np.ones((ksz, ksz), np.uint8)) > 0
    sel = support & vs
    for div in (2.0, 4.0):
        if int(np.count_nonzero(sel)) < 16:
            break
        r = small - g0
        tau = max(0.02, 3.0 * 1.4826 * float(np.median(np.abs(r[sel]))))
        wgt = (sel & (np.abs(r) < tau)).astype(np.float32)
        sig = max(1.0, rs / div)
        num = cv2.GaussianBlur(r * wgt, (0, 0), sig, borderType=cv2.BORDER_REPLICATE)
        den = cv2.GaussianBlur(wgt, (0, 0), sig, borderType=cv2.BORDER_REPLICATE)
        corr = np.divide(num, den, out=np.zeros_like(num), where=den > 0.05)
        g0 = np.where(support, np.maximum(g0 + corr, 0.0), g0).astype(np.float32)
    if float(g0.max()) < GLARE_MIN_LIN:
        return None
    return cv2.resize(g0, (w, h), interpolation=cv2.INTER_LINEAR)


def _saturated_plateaus(bgr: np.ndarray, radius: float) -> np.ndarray:
    """Bool mask of fully saturated areas wider than ``2*radius`` in both directions (0.25x grid)."""
    h, w = bgr.shape[:2]
    ch = cv2.split(bgr)
    vmin = cv2.min(cv2.min(ch[0], ch[1]), ch[2])
    sat = (vmin >= SATURATED_HI).astype(np.float32)
    if not sat.any():
        return np.zeros((h, w), bool)
    q = 4
    small = cv2.resize(sat, (max(4, w // q), max(4, h // q)), interpolation=cv2.INTER_AREA) >= PLATEAU_MIN_FRAC
    rs = max(1, int(round(radius / q)))
    k = np.ones((2 * rs + 1, 2 * rs + 1), np.uint8)
    op = cv2.morphologyEx(small.astype(np.uint8), cv2.MORPH_OPEN, k)
    if not op.any():
        return np.zeros((h, w), bool)
    op = cv2.dilate(op, np.ones((3, 3), np.uint8))
    return cv2.resize(op, (w, h), interpolation=cv2.INTER_NEAREST) > 0


def _lin_to_lab(lin: np.ndarray) -> np.ndarray:
    enc = cv2.pow(np.maximum(lin, 0.0), 1.0 / GAMMA)
    return cv2.cvtColor(np.minimum(enc, 1.0).astype(np.float32), cv2.COLOR_BGR2Lab)


def _lab_to_lin(lab: np.ndarray) -> np.ndarray:
    return cv2.pow(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), GAMMA)


def _sigmoid(t: float) -> float:
    t = float(np.clip(t, -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(-t))


def _fit_logistic(s: np.ndarray, y: np.ndarray, l2_a: float = 0.5, l2_b: float = 1e-3) -> tuple[float, float]:
    """L2-regularised 1-D logistic regression ``P(y=1) = sigmoid(a*s + b)`` (Newton's method)."""
    s = np.asarray(s, np.float64)
    y = np.asarray(y, np.float64)
    a, b = DEFAULT_LOGISTIC["a"], DEFAULT_LOGISTIC["b"]
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-np.clip(a * s + b, -60, 60)))
        g_a = float(((p - y) * s).sum() + l2_a * a)
        g_b = float((p - y).sum() + l2_b * b)
        wv = p * (1.0 - p)
        h_aa = float((wv * s * s).sum() + l2_a)
        h_ab = float((wv * s).sum())
        h_bb = float(wv.sum() + l2_b)
        det = h_aa * h_bb - h_ab * h_ab
        if det <= 1e-12:
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        step = 1.0 if max(abs(da), abs(db)) <= 5 else 5.0 / max(abs(da), abs(db))
        a -= step * da
        b -= step * db
        if abs(da) < 1e-9 and abs(db) < 1e-9:
            break
    return float(a), float(b)


def _as_bucket(key: object) -> str:
    """Bucket name from ``"z1".."z4"`` or a zoom value (number or numeric string)."""
    if isinstance(key, str) and key in ZOOM_BUCKETS:
        return key
    try:
        return zoom_bucket(float(key))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown zoom bucket {key!r}") from exc


def _nearest_bucket(available: Iterable[str], bucket: str) -> str:
    i = ZOOM_BUCKETS.index(bucket)
    return min(available, key=lambda k: (abs(ZOOM_BUCKETS.index(k) - i), ZOOM_BUCKETS.index(k)))


def _jsonable(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class _MomentAccumulator:
    """Masked per-pixel running mean / variance in float64 (shifted-data algorithm).

    Samples are accumulated relative to a fixed reference image (the medoid), so the sums of
    squares stay small and the float64 moments are as stable as Welford's recurrence while
    using OpenCV's masked ``accumulate``/``accumulateSquare`` kernels.
    """

    def __init__(self, ref: np.ndarray):
        self.ref = np.ascontiguousarray(ref, dtype=np.float32)
        h, w = ref.shape[:2]
        self.n = np.zeros((h, w), np.float64)
        self.s1 = np.zeros(ref.shape, np.float64)
        self.s2 = np.zeros(ref.shape, np.float64)
        self.count = 0

    def update(self, x: np.ndarray, mask: np.ndarray) -> None:
        """Add sample ``x`` on the pixels where ``mask > 0``."""
        m = (mask > 0).astype(np.uint8)
        d = cv2.subtract(np.asarray(x, np.float32), self.ref)
        cv2.accumulate(d, self.s1, mask=m)
        cv2.accumulateSquare(d, self.s2, mask=m)
        self.n += m
        self.count += 1

    def moments(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(count, mean, unbiased variance); mean = ref where count == 0."""
        n3 = np.maximum(self.n, 1.0)[..., None]
        mean_d = self.s1 / n3
        var = np.maximum(self.s2 - self.s1 * mean_d, 0.0) / np.maximum(self.n - 1.0, 1.0)[..., None]
        return self.n, self.ref.astype(np.float64) + mean_d, var


# --------------------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------------------
class StatTemplateModel:
    """Registration-based per-pixel Lab Gaussian template (the fast detector).

    Attributes (SPEC §2.6): ``category``, ``scene_size`` (W_s, H_s), ``scale`` (template px =
    scene px * scale), ``mu_lab``/``sigma_lab`` (H_t, W_t, 3) float32 (float16-exact),
    ``ref_gray`` (H_t, W_t) uint8, ``valid`` (H_t, W_t) uint8 0/255, ``calib`` (dict) and
    ``meta`` (fit statistics). Thread-safe for concurrent ``predict`` calls.
    """

    def __init__(
        self,
        category: str,
        scene_size: tuple[int, int],
        scale: float,
        mu_lab: np.ndarray,
        sigma_lab: np.ndarray,
        ref_gray: np.ndarray,
        valid: np.ndarray | None = None,
        calib: dict | None = None,
        cfg: DetectorConfig | None = None,
        meta: dict | None = None,
    ):
        self.category = str(category)
        self.scene_size = (int(scene_size[0]), int(scene_size[1]))
        self.scale = float(scale)
        self.mu_lab = np.ascontiguousarray(mu_lab, dtype=np.float32)
        self.sigma_lab = np.ascontiguousarray(sigma_lab, dtype=np.float32)
        self.ref_gray = np.ascontiguousarray(ref_gray, dtype=np.uint8)
        ht, wt = self.mu_lab.shape[:2]
        if self.mu_lab.ndim != 3 or self.mu_lab.shape[2] != 3:
            raise ValueError("mu_lab must be (H, W, 3)")
        if self.sigma_lab.shape != self.mu_lab.shape or self.ref_gray.shape != (ht, wt):
            raise ValueError("mu_lab, sigma_lab and ref_gray must share the template size")
        self.valid = (
            np.full((ht, wt), 255, np.uint8) if valid is None else np.where(np.asarray(valid) > 0, 255, 0).astype(np.uint8)
        )
        self.cfg = cfg or DetectorConfig()
        self.calib = calib if calib is not None else self._default_calib()
        self.meta = dict(meta or {})
        self._stack = np.dstack(
            [self.mu_lab, self.sigma_lab, self.ref_gray.astype(np.float32), (self.valid > 0).astype(np.float32)]
        ).astype(np.float32)
        self._levels: OrderedDict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._renders: OrderedDict[tuple, dict] = OrderedDict()
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- geometry
    @property
    def template_size(self) -> tuple[int, int]:
        """(W_t, H_t)."""
        return self.mu_lab.shape[1], self.mu_lab.shape[0]

    def template_to_scene(self) -> np.ndarray:
        """2x3 map template px -> scene px (``cv2.resize`` convention, per-axis exact)."""
        return _resize_map(self.scene_size, self.template_size)

    def model_to_capture_scene(self, capture: Capture) -> np.ndarray:
        """2x3 map model scene px -> the capture's scene px (identity unless the part image size differs)."""
        cw, ch = (float(v) for v in capture.scene_size)
        return np.array([[cw / self.scene_size[0], 0.0, 0.0], [0.0, ch / self.scene_size[1], 0.0]], dtype=np.float64)

    def _glare_radius(self, capture: Capture) -> float:
        """Opening radius (capture px) separating illumination fields from defects."""
        w, _h = capture.size
        px_per_scene = w / max(float(capture.window[2]), 1e-6)
        return GLARE_R_FRAC * math.hypot(*self.scene_size) * px_per_scene

    # ---------------------------------------------------------------- calibration
    @staticmethod
    def _default_calib() -> dict:
        return {
            "buckets": {b: dict(DEFAULT_BUCKET, n=0) for b in ZOOM_BUCKETS},
            "region_thr": {b: DEFAULT_BUCKET["region_thr"] for b in ZOOM_BUCKETS},
            "logistic": dict(DEFAULT_LOGISTIC, fitted=False, n=0),
            "source": "default",
        }

    def _bucket_stats(self, bucket: str) -> dict:
        buckets = self.calib.get("buckets") or {}
        if bucket in buckets:
            return buckets[bucket]
        if buckets:
            return buckets[_nearest_bucket(buckets, bucket)]
        return dict(DEFAULT_BUCKET)

    def region_threshold(self, zoom: float) -> float:
        """z-map threshold used for region extraction at this zoom (``calib["region_thr"][bucket]``)."""
        b = zoom_bucket(zoom)
        thr = (self.calib.get("region_thr") or {}).get(b)
        if thr is None:
            thr = self._bucket_stats(b).get("region_thr", DEFAULT_BUCKET["region_thr"])
        return float(thr)

    @staticmethod
    def _bucket_entry(vals: np.ndarray) -> dict:
        dropped = 0
        if vals.size >= 8:  # drop gross outliers (e.g. failed captures) before the moments
            med = float(np.median(vals))
            mad = 1.4826 * float(np.median(np.abs(vals - med)))
            keep = vals <= med + 6.0 * max(mad, 1e-6)
            dropped = int((~keep).sum())
            vals = vals[keep]
        mu = float(vals.mean())
        sd = float(vals.std(ddof=1)) if vals.size >= 2 else 0.0
        sd = max(sd, SD_REL_FLOOR * abs(mu), SD_ABS_FLOOR)
        if vals.size < 3:
            sd = max(sd, 0.2 * abs(mu))
        return {
            "mu": mu,
            "sd": sd,
            "region_thr": max(REGION_THR_MIN, REGION_PEAK_RATIO * (mu + REGION_K * sd)),
            "n": int(vals.size),
            "dropped": dropped,
        }

    def calibrate(
        self, normal_scores_by_zoom: dict[str, list[float]], labeled: list[tuple[float, int, str]] | None = None
    ) -> None:
        """Per-zoom-bucket normal statistics (+ region thresholds) and a global logistic.

        ``normal_scores_by_zoom`` maps bucket names (``"z1".."z4"``, or zoom values) to raw
        scores of *normal* captures; per bucket ``mu``/``sd`` (robust to gross outliers, with
        floors) and ``region_thr = max(4, 1.5 * (mu + 3 sd))`` are stored. ``labeled`` holds
        ``(score, label 0/1, bucket-or-zoom)`` samples for the logistic on standardized scores;
        without both classes the fallback ``a=1.6, b=-4.0`` is used. Buckets without data
        inherit from the nearest calibrated bucket.
        """
        pooled: dict[str, list[float]] = {}
        for key, scores in (normal_scores_by_zoom or {}).items():
            vals = [float(v) for v in scores if v is not None and np.isfinite(float(v))]
            if vals:
                pooled.setdefault(_as_bucket(key), []).extend(vals)
        if not pooled:
            raise ValueError("calibrate() needs at least one finite normal score")
        out = {b: self._bucket_entry(np.asarray(v, np.float64)) for b, v in pooled.items()}
        for b in ZOOM_BUCKETS:
            if b not in out:
                src = _nearest_bucket(list(pooled), b)
                out[b] = dict(out[src], n=0, filled_from=src)
        logistic = dict(DEFAULT_LOGISTIC, fitted=False, n=0)
        if labeled:
            s, y = [], []
            for item in labeled:
                st = out[_as_bucket(item[2])]
                s.append((float(item[0]) - st["mu"]) / st["sd"])
                y.append(1 if int(item[1]) else 0)
            y_arr = np.asarray(y)
            if 0 < int(y_arr.sum()) < len(y_arr):
                a, bb = _fit_logistic(np.asarray(s), y_arr)
                if a > 0 and np.isfinite(a) and np.isfinite(bb):
                    logistic = {"a": a, "b": bb, "fitted": True, "n": int(len(y_arr))}
        self.calib = {
            "buckets": {b: out[b] for b in ZOOM_BUCKETS},
            "region_thr": {b: float(out[b]["region_thr"]) for b in ZOOM_BUCKETS},
            "logistic": logistic,
            "source": "calibrated",
        }

    def set_region_thresholds(self, thr_by_bucket: dict[str, float]) -> None:
        """Override per-bucket region thresholds (e.g. from pooled normal pixel z-scores).

        Keys are bucket names or zoom values; ``calib["region_thr"]`` and the bucket entries
        are updated, other calibration values are kept.
        """
        rt = dict(self.calib.get("region_thr") or {})
        buckets = self.calib.setdefault("buckets", {})
        for key, val in (thr_by_bucket or {}).items():
            b = _as_bucket(key)
            rt[b] = float(val)
            if isinstance(buckets.get(b), dict):
                buckets[b]["region_thr"] = float(val)
        self.calib["region_thr"] = rt

    def prob(self, score: float, zoom: float) -> float:
        """Calibrated P(anomalous) = ``sigmoid(a * (score - mu_b) / sd_b + b)`` for the zoom bucket."""
        st = self._bucket_stats(zoom_bucket(zoom))
        lg = self.calib.get("logistic") or DEFAULT_LOGISTIC
        sd = max(float(st.get("sd", 1.0)), 1e-6)
        s = (float(score) - float(st.get("mu", 0.0))) / sd
        return _sigmoid(float(lg.get("a", DEFAULT_LOGISTIC["a"])) * s + float(lg.get("b", DEFAULT_LOGISTIC["b"])))

    # ---------------------------------------------------------------- rendering
    def _level(self, f_x: float, f_y: float) -> tuple[np.ndarray, np.ndarray]:
        """Template stack for magnification (f_x, f_y): area-filtered copy when minifying (cached)."""
        wt, ht = self.template_size
        if max(f_x, f_y) >= 1.0 / 1.15:
            return self._stack, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        key = (max(4, int(round(wt * f_x))), max(4, int(round(ht * f_y))))
        with self._lock:
            hit = self._levels.get(key)
            if hit is not None:
                self._levels.move_to_end(key)
                return hit
        # cv2.resize handles <= 4 channels: resample the 8-channel stack in two halves
        small = np.dstack(
            [cv2.resize(np.ascontiguousarray(self._stack[..., i : i + 4]), key, interpolation=cv2.INTER_AREA) for i in (0, 4)]
        )
        hit = (small, _resize_map((wt, ht), key))  # level px -> template px
        with self._lock:
            self._levels[key] = hit
            while len(self._levels) > _LEVEL_CACHE_MAX:
                self._levels.popitem(last=False)
        return hit

    def render_expected(self, capture: Capture) -> dict:
        """Expected appearance in the *commanded* capture grid (cached per view geometry).

        Returns ``mu`` (h,w,3 Lab), ``sigma`` (h,w,3), ``grad2`` (h,w,3: squared gradient
        magnitude of ``mu`` per capture px), ``gray`` (h,w uint8), ``valid`` (h,w bool),
        ``T2C`` (2x3 template px -> commanded capture px) and ``mag`` (capture px per template
        px). The arrays are shared through the cache: treat them as read-only.
        """
        w, h = capture.size
        t2c = compose_affine(capture.scene_to_capture(), self.model_to_capture_scene(capture), self.template_to_scene())
        key = (w, h) + tuple(round(float(v), 5) for v in t2c.ravel())
        with self._lock:
            hit = self._renders.get(key)
            if hit is not None:
                self._renders.move_to_end(key)
                return hit
        mag = float(max(abs(t2c[0, 0]), abs(t2c[1, 1])))
        stack, l2t = self._level(abs(t2c[0, 0]), abs(t2c[1, 1]))
        l2c = compose_affine(t2c, l2t)
        if mag > MAGNIFY_CUBIC_ABOVE:  # cubic warps support <= 4 channels
            out = np.dstack(
                [
                    cv2.warpAffine(
                        np.ascontiguousarray(stack[..., i : i + 4]),
                        l2c,
                        (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                    for i in (0, 4)
                ]
            )
        else:
            out = cv2.warpAffine(stack, l2c, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        inside = cv2.warpAffine(
            np.full(stack.shape[:2], 255, np.uint8), l2c, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
        )
        mu = np.ascontiguousarray(out[..., 0:3])
        sigma = np.maximum(np.ascontiguousarray(out[..., 3:6]), 1e-3)
        res = {
            "mu": mu,
            "sigma": sigma,
            "sigma2": cv2.multiply(sigma, sigma),
            "inv_var": cv2.divide(np.ones_like(sigma), cv2.multiply(sigma, sigma)),
            "grad2": self._grad2(mu),
            "mu_lin": _lab_to_lin(mu),
            "gray": np.clip(out[..., 6] + 0.5, 0, 255).astype(np.uint8),
            "valid": (out[..., 7] >= 0.99) & (inside > 0),
            "T2C": t2c,
            "mag": mag,
        }
        for v in res.values():
            if isinstance(v, np.ndarray):
                v.setflags(write=False)
        with self._lock:
            self._renders[key] = res
            while len(self._renders) > _RENDER_CACHE_MAX:
                self._renders.popitem(last=False)
        return res

    @staticmethod
    def _grad2(mu: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(mu, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
        gy = cv2.Sobel(mu, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
        return cv2.add(cv2.multiply(gx, gx), cv2.multiply(gy, gy))

    # ---------------------------------------------------------------- scoring core
    def _analyze(self, capture: Capture, quality: QualityReport | None = None) -> dict:
        """Steps 1-7 of the pipeline; returns score, z-map/valid (commanded frame) and diagnostics."""
        cfg = self.cfg
        img = _as_bgr(capture.image)
        h, w = img.shape[:2]
        notes: list[str] = []
        # (1) expected appearance, (2) residual registration
        exp = self.render_expected(capture)
        reg = register(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), exp["gray"], exp["valid"].astype(np.uint8) * 255, cfg)
        M = reg.M
        # (3) capture -> commanded frame; validity
        warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        cover = cv2.warpAffine(
            np.full((h, w), 255, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
        )
        valid = exp["valid"] & (cover > 0)
        radius = self._glare_radius(capture)
        plateau = _saturated_plateaus(warped, radius)
        if plateau.any():
            n_pl = int(np.count_nonzero(valid & plateau))
            valid &= ~plateau
            notes.append(f"saturated plateau masked: {n_pl} px")
        if quality is not None and quality.glare_boxes:
            gmask = np.zeros((h, w), np.uint8)
            for bx, by, bw, bh in quality.glare_boxes:
                cv2.rectangle(gmask, (int(bx), int(by)), (int(bx + bw), int(by + bh)), 255, -1)
            gmask = cv2.dilate(gmask, np.ones((2 * GLARE_DILATE_PX + 1,) * 2, np.uint8))
            gmask = cv2.warpAffine(gmask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            n_gl = int(np.count_nonzero(valid & (gmask > 0)))
            valid &= gmask == 0
            notes.append(f"glare boxes masked: {len(quality.glare_boxes)} box(es), {n_gl} px")
        valid_frac = float(np.count_nonzero(valid)) / float(h * w)

        ch = cv2.split(warped)
        vmax = cv2.max(cv2.max(ch[0], ch[1]), ch[2])
        fit_mask = valid & (vmax < SATURATED_HI) & (vmax > SATURATED_LO)
        if int(np.count_nonzero(fit_mask)) < 256:
            fit_mask = valid
        inv_var = exp["inv_var"]
        x_lin = cv2.LUT(warped, _LUT_TO_LIN)
        light_ok = valid & (vmax < SATURATED_HI)
        # (4c, first) smooth achromatic illumination field (weak glare) in linear light, estimated
        # against a glare-insensitive gain (channel differences cancel additive white light)
        field = None
        g_lin = _glare_free_gain(exp["mu_lin"], x_lin, fit_mask)
        if g_lin is not None:
            field = _illumination_field(x_lin, cv2.multiply(exp["mu_lin"], (g_lin, g_lin, g_lin, 0.0)), light_ok, radius)
        if field is not None:
            x_lin = cv2.subtract(x_lin, cv2.merge([field, field, field]))
            x = _lin_to_lab(x_lin)
            src_gray = cv2.cvtColor(np.minimum(cv2.pow(np.maximum(x_lin, 0.0), 1.0 / GAMMA), 1.0), cv2.COLOR_BGR2GRAY) * 255.0
        else:
            x = _u8_to_lab(warped)
            src_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # (4a) PSF matching (+ low-angle relief shading), (4b) global photometric fit
        blur = _estimate_blur(exp["gray"], src_gray, fit_mask)
        mu, grad2 = self._blurred(exp, blur)
        if capture.settings.light == "low_angle":
            mu_lin = _lab_to_lin(mu)
            shade = _relief_shading(exp["gray"], x_lin, mu_lin, fit_mask, capture.size[0] / max(float(capture.window[2]), 1e-6))
            if shade is not None:
                mu = _lin_to_lab(cv2.multiply(mu_lin, cv2.merge([shade, shade, shade])))
                notes.append("low-angle relief shading modelled")
        a, b = _photometric_fit(mu, x, fit_mask, inv_var)
        pred = self._predict_lab(mu, a, b)
        if field is None and g_lin is None:
            # achromatic scene: fall back to a field relative to the Lab prediction, then refit
            field = _illumination_field(x_lin, _lab_to_lin(pred), light_ok, radius)
            if field is not None:
                x = _lin_to_lab(cv2.subtract(x_lin, cv2.merge([field, field, field])))
                a, b = _photometric_fit(mu, x, fit_mask, inv_var)
                pred = self._predict_lab(mu, a, b)
        illum_peak = 0.0
        if field is not None:
            illum_peak = float(field.max())
            notes.append(f"illumination field removed (peak +{illum_peak:.3f} linear)")
        inv_a = tuple(float(1.0 / v) for v in a) + (0.0,)
        d = cv2.multiply(cv2.subtract(x, pred), inv_a)  # residual in template units: x' - mu

        # (5) noise-aware sigma (+ registration uncertainty on template edges)
        inner = cv2.erode(valid.astype(np.uint8), np.ones((2 * VALID_ERODE_PX + 1,) * 2, np.uint8)) > 0
        noise = _robust_noise(d, inner)
        n2 = np.array([(NOISE_GAIN * v) ** 2 for v in noise], np.float32)
        if field is None:
            var = cv2.add(exp["sigma2"], tuple(float(v) for v in n2) + (0.0,))
        else:
            # removing added light expands the contrast (and the noise) of the affected pixels by
            # dL/dY ratio ((Y + G) / Y)^(2/3); scale the noise term per pixel accordingly
            y_t = np.maximum(((pred[..., 0] + 16.0) / 116.0) ** 3, 0.01).astype(np.float32)
            amp2 = np.clip(((y_t + field) / y_t) ** (4.0 / 3.0), 1.0, 100.0).astype(np.float32)
            var = exp["sigma2"] + amp2[..., None] * n2
        eps = REG_UNCERT_PX * max(1.0, exp["mag"])
        if eps > 0:
            var = cv2.scaleAdd(grad2, eps * eps, var)
        # (6) z-map
        c0, c1, c2 = cv2.split(cv2.divide(cv2.multiply(d, d), var))
        z = cv2.sqrt(cv2.max(cv2.max(c0, c1), c2))
        z[~valid] = 0.0
        z = cv2.GaussianBlur(z, (0, 0), cfg.zmap_blur_sigma)
        z[~inner] = 0.0
        # (7) top-k score
        n_valid = int(np.count_nonzero(inner))
        if n_valid > 0:
            k = int(min(n_valid, max(cfg.topk_min, int(cfg.topk_frac * n_valid))))
            vals = z[inner]
            score = float(np.partition(vals, n_valid - k)[n_valid - k :].mean())
        else:
            score = 0.0
        return {
            "score": score,
            "zmap": z,
            "valid": inner,
            "reg": reg,
            "valid_frac": valid_frac,
            "gain": a,
            "offset": b,
            "noise": noise,
            "blur": blur,
            "illum_peak": illum_peak,
            "notes": notes,
        }

    @staticmethod
    def _blurred(exp: dict, blur: float) -> tuple[np.ndarray, np.ndarray]:
        """Expected mean matched to the capture PSF, and the (sharp, hence conservative) edge term."""
        if blur < BLUR_APPLY_MIN:
            return exp["mu"], exp["grad2"]
        return cv2.GaussianBlur(exp["mu"], (0, 0), blur), exp["grad2"]

    @staticmethod
    def _predict_lab(mu: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """``a * mu + b`` per channel, passed through the sRGB gamut (clip-aware prediction)."""
        lin = np.zeros((3, 4), np.float64)
        lin[[0, 1, 2], [0, 1, 2]] = a
        lin[:, 3] = b
        pred = cv2.transform(mu, lin)
        return cv2.cvtColor(cv2.cvtColor(pred, cv2.COLOR_Lab2BGR), cv2.COLOR_BGR2Lab)

    def score_capture(self, capture: Capture) -> tuple[float, np.ndarray, np.ndarray, RegistrationResult]:
        """(score, zmap, valid, reg); ``zmap``/``valid`` are in the registered (commanded) frame.

        ``reg.M`` maps capture px -> that frame (warp with ``WARP_INVERSE_MAP`` to bring maps back).
        """
        r = self._analyze(capture)
        return r["score"], r["zmap"], r["valid"], r["reg"]

    def predict(self, capture: Capture, quality: QualityReport | None = None, id_prefix: str = "") -> AnomalyResult:
        """Fast-detector result for any commanded view.

        ``anomaly_map``/``valid_mask`` are in the capture frame; regions carry capture-px boxes
        and commanded-frame scene boxes. ``reliable = reg.ok and valid_frac >= min_valid_frac
        and no blocking quality issue`` (:func:`blocking_quality_issues`: every issue except moderate
        ``noisy``, which the noise-aware sigma absorbs); glare boxes of ``quality`` are excluded from scoring.
        """
        t0 = time.perf_counter()
        cfg = self.cfg
        r = self._analyze(capture, quality)
        reg: RegistrationResult = r["reg"]
        zoom = capture_zoom(capture)
        bucket = zoom_bucket(zoom)
        thr = self.region_threshold(zoom)
        regions = extract_regions(r["zmap"], thr, capture, reg.M, "fast", cfg.min_region_area, cfg.max_regions, id_prefix)
        w, h = capture.size
        amap = cv2.warpAffine(
            r["zmap"], reg.M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT
        )
        vmask = cv2.warpAffine(
            r["valid"].astype(np.uint8) * 255,
            reg.M,
            (w, h),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
        )
        score = float(r["score"])
        notes = list(r["notes"])
        reliable = True
        if not reg.ok:
            reliable = False
            notes.append(f"registration failed (method={reg.method}, ncc={reg.ncc:.2f} < {cfg.min_ncc:.2f})")
        if r["valid_frac"] < cfg.min_valid_frac:
            reliable = False
            notes.append(f"valid area {r['valid_frac']:.2f} < {cfg.min_valid_frac:.2f}")
        if quality is not None and not quality.ok:
            blocking = blocking_quality_issues(quality)
            if blocking:
                reliable = False
                notes.append("quality issues: " + ",".join(quality.issues))
            else:
                notes.append(
                    f"quality issues tolerated: {','.join(quality.issues)} (noise_sigma={quality.noise_sigma:.1f}<="
                    f"{NOISE_TOLERANCE_SIGMA:.1f}; the noise-aware sigma absorbs it)"
                )
        g = r["gain"]
        notes.append(
            f"zoom={zoom:.2f} ({bucket}) region_thr={thr:.2f} gain L/a/b={g[0]:.2f}/{g[1]:.2f}/{g[2]:.2f} "
            f"psf_sigma={r['blur']:.2f} noise_L={r['noise'][0]:.2f} valid={r['valid_frac']:.2f} "
            f"calib={self.calib.get('source', '?')}"
        )
        return AnomalyResult(
            capture_id=capture.capture_id,
            detector="fast",
            score=score,
            prob=self.prob(score, zoom),
            regions=regions,
            reliable=bool(reliable),
            registration=reg.to_dict(),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            notes=notes,
            anomaly_map=amap.astype(np.float32, copy=False),
            valid_mask=vmask,
        )

    # ---------------------------------------------------------------- fitting
    @classmethod
    def fit(
        cls,
        category: str,
        normal_images: list[np.ndarray] | Iterable[np.ndarray],
        cfg: DetectorConfig | None = None,
        log: Callable[..., object] | None = print,
        *,
        holdout: int | None = None,
        sensor_long_side: int | None = None,
    ) -> "StatTemplateModel":
        """Build the template from normal scene images (BGR uint8).

        Medoid (downscaled gray closest to the median image) -> register every normal to it
        (failures dropped) -> photometric normalisation to the medoid -> masked float64 Lab
        mean/std (shifted-data accumulation, Welford-equivalent stability) ->
        ``sigma = sqrt(std^2 + sigma_floor^2)`` smoothed (sigma 1.5) -> float16-exact arrays
        (so save/load is lossless).

        A provisional calibration is computed on ``holdout`` normals (default ``n // 8``
        clamped to 2..6, when n >= 10) captured at several views (camera simulator, nominal
        conditions without glare; clean pseudo captures if the simulator is unavailable) and
        scored against the template built *without* them (they are added afterwards); with
        fewer normals in-sample scores with a widened spread are used. Run :meth:`calibrate`
        on real captures to replace it. Pass a list or lazy ``Sequence`` to avoid
        materialising a generator.
        """
        cfg = cfg or DetectorConfig()
        say = log if log is not None else (lambda *a, **k: None)
        t0 = time.perf_counter()
        if isinstance(normal_images, np.ndarray):
            seq: Sequence = list(normal_images) if normal_images.ndim == 4 else [normal_images]
        elif isinstance(normal_images, Sequence):
            seq = normal_images
        else:
            seq = list(normal_images)
        n = len(seq)
        if n < 2:
            raise ValueError("StatTemplateModel.fit needs at least 2 normal images")
        first = _as_bgr(seq[0])
        hs, ws = first.shape[:2]
        scene_size = (ws, hs)
        scale = cfg.template_long_side / float(max(ws, hs))
        wt, ht = int(round(ws * scale)), int(round(hs * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC

        def load(i: int) -> np.ndarray:
            img = first if i == 0 else _as_bgr(seq[i])
            if img.shape[:2] != (hs, ws):
                img = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_AREA)
            return img

        def to_tmpl(img: np.ndarray) -> np.ndarray:
            return cv2.resize(img, (wt, ht), interpolation=interp)

        # 1) medoid on downscaled gray
        sw, sh = sensor_size_for(scene_size, 160)
        smalls = np.stack(
            [cv2.resize(cv2.cvtColor(load(i), cv2.COLOR_BGR2GRAY), (sw, sh), interpolation=cv2.INTER_AREA) for i in range(n)]
        ).astype(np.float32)
        medoid = int(np.argmin(((smalls - np.median(smalls, axis=0)) ** 2).mean(axis=(1, 2))))
        del smalls

        # 2) deterministic, evenly spaced holdout (never the medoid)
        if holdout is None:
            h_n = min(6, max(2, n // 8)) if n >= 10 else 0
        else:
            h_n = max(0, min(int(holdout), n - 2))
        others = [i for i in range(n) if i != medoid]
        hold = sorted({others[int(j)] for j in np.linspace(0, len(others) - 1, h_n).round()}) if h_n > 0 else []
        hold_set = set(hold)

        # 3) register to the medoid, normalise photometry, accumulate
        fixed = to_tmpl(load(medoid))
        fixed_gray = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
        fixed_lab = _u8_to_lab(fixed)
        acc = _MomentAccumulator(fixed_lab)
        pending: list[tuple[np.ndarray, np.ndarray]] = []
        failed: list[int] = []
        ones = np.full((ht, wt), 255, np.uint8)
        k3 = np.ones((3, 3), np.uint8)
        for i in range(n):
            if i == medoid:
                lab_n, cover = fixed_lab, ones
            else:
                img_t = to_tmpl(load(i))
                reg = register(cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY), fixed_gray, None, cfg)
                if not reg.ok:
                    failed.append(i)
                    say(f"[stat:{category}] normal {i}: registration failed ({reg.method}, ncc={reg.ncc:.2f}) - dropped")
                    continue
                warped = cv2.warpAffine(img_t, reg.M, (wt, ht), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                cover = cv2.erode(
                    cv2.warpAffine(ones, reg.M, (wt, ht), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT), k3
                )
                lab = _u8_to_lab(warped)
                chs = cv2.split(warped)
                vmax = cv2.max(cv2.max(chs[0], chs[1]), chs[2])
                fit_mask = (cover > 0) & (vmax < SATURATED_HI) & (vmax > SATURATED_LO)
                a, b = _photometric_fit(fixed_lab, lab, fit_mask, None)
                offs = tuple(float(v) for v in b) + (0.0,)
                lab_n = cv2.multiply(cv2.subtract(lab, offs), tuple(float(1.0 / v) for v in a) + (0.0,))
            if i in hold_set:
                pending.append((lab_n, cover))
            else:
                acc.update(lab_n, cover)
        n_used = acc.count + len(pending)
        if n_used < 2:
            raise ValueError(f"StatTemplateModel.fit: only {n_used} normal(s) registered to the medoid")

        def finalize() -> "StatTemplateModel":
            cnt, mean, var = acc.moments()
            need = max(2, int(math.ceil(0.5 * acc.count)))
            std = np.sqrt(var)
            sigma = np.sqrt(std * std + cfg.sigma_floor**2).astype(np.float32)
            sigma = cv2.GaussianBlur(sigma, (0, 0), SIGMA_SMOOTH)
            mu = mean.astype(np.float32)
            valid = cv2.erode((cnt >= need).astype(np.uint8) * 255, k3, iterations=2)
            sigma = np.where((valid > 0)[..., None], sigma, np.float32(cfg.sigma_floor * 5.0))
            mu = mu.astype(np.float16).astype(np.float32)  # float16-exact -> lossless save/load
            sigma = sigma.astype(np.float16).astype(np.float32)
            return cls(
                category=category,
                scene_size=scene_size,
                scale=wt / float(ws),
                mu_lab=mu,
                sigma_lab=sigma,
                ref_gray=_lab_to_gray_u8(mu),
                valid=valid,
                calib=None,
                cfg=cfg,
            )

        ls = int(sensor_long_side or CameraConfig().sensor_long_side)
        if pending:
            interim = finalize()
            calib = interim._provisional_calibration([load(i) for i in hold], ls, source="provisional_holdout")
            for lab_n, cover in pending:
                acc.update(lab_n, cover)
            model = finalize()
        else:
            model = finalize()
            sample = [i for i in others if i not in failed][:3] or [medoid]
            calib = model._provisional_calibration([load(i) for i in sample], ls, source="provisional_in_sample")
        model.calib = calib
        model.meta = {
            "n_normals": n,
            "n_used": int(acc.count),
            "failed": failed,
            "medoid": medoid,
            "holdout": hold,
            "fit_seconds": round(time.perf_counter() - t0, 3),
            "template_size": [wt, ht],
        }
        say(
            f"[stat:{category}] fitted on {acc.count}/{n} normals (medoid={medoid}, holdout={len(hold)}, "
            f"failed={len(failed)}) template={wt}x{ht} in {model.meta['fit_seconds']:.1f}s"
        )
        return model

    def _provisional_calibration(self, scenes: list[np.ndarray], long_side: int, source: str) -> dict:
        """Calibration from captures of ``scenes`` at the provisional views.

        Captures come from the camera simulator under ``nominal`` conditions without glare
        (seeded, deterministic) so the normal-score spread reflects typical capture noise,
        blur, JPEG, exposure and pose jitter; clean pseudo captures are the fallback.
        """
        try:
            from inspectagent.vision.camera import VirtualCamera, make_conditions
        except ImportError:  # camera simulator unavailable: clean geometry-only captures
            VirtualCamera = make_conditions = None  # type: ignore[assignment]
        by_bucket: dict[str, list[float]] = {}
        cams: list = []
        for si, scene in enumerate(scenes):
            cam = None
            if VirtualCamera is not None:
                cond = make_conditions("nominal", np.random.default_rng(9_000 + si))
                cond.glare_strength = 0.0
                cam = VirtualCamera(scene, cond, CameraConfig(sensor_long_side=long_side), seed=9_000 + si)
            cams.append(cam)
        # views outermost so each expected render is computed once and reused from the cache
        for vi, (cx, cy, z, light, ev, ambient) in enumerate(_PROVISIONAL_VIEWS):
            for si, (scene, cam) in enumerate(zip(scenes, cams)):
                if cam is not None:
                    st = CaptureSettings(view=View(cx, cy, z), exposure_ev=ev, light=light)
                    base_ambient = cam.conditions.ambient_ev
                    if ambient is not None:
                        cam.conditions.ambient_ev = float(ambient)
                    try:
                        cap = cam.capture(st, f"cal_{si}_{vi}", vi)
                    finally:
                        cam.conditions.ambient_ev = base_ambient
                elif ambient is not None:
                    continue  # the clean fallback cannot render a gain-compensated dark capture
                else:
                    cap = pseudo_capture(scene, View(cx, cy, z), long_side, capture_id=f"cal_{si}_{vi}")
                by_bucket.setdefault(zoom_bucket(z), []).append(float(self._analyze(cap)["score"]))
        self.calibrate(by_bucket)
        calib = self.calib
        if source == "provisional_in_sample":  # in-sample scores are optimistic: widen the spread
            for b, st in calib["buckets"].items():
                st["sd"] = max(st["sd"], 0.25 * abs(st["mu"]))
                st["region_thr"] = max(REGION_THR_MIN, REGION_PEAK_RATIO * (st["mu"] + REGION_K * st["sd"]))
                calib["region_thr"][b] = st["region_thr"]
        calib["source"] = source
        return calib

    # ---------------------------------------------------------------- persistence
    def save(self, path: Path | str) -> None:
        """Write an ``.npz`` (float16 Lab arrays, uint8 gray/valid, JSON calib/config/meta)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            format_version=np.array(FORMAT_VERSION, np.int32),
            category=np.array(self.category),
            scene_size=np.array(self.scene_size, np.int32),
            scale=np.array(self.scale, np.float64),
            mu_lab=self.mu_lab.astype(np.float16),
            sigma_lab=self.sigma_lab.astype(np.float16),
            ref_gray=self.ref_gray,
            valid=self.valid,
            calib_json=np.array(json.dumps(_jsonable(self.calib))),
            cfg_json=np.array(json.dumps(asdict(self.cfg))),
            meta_json=np.array(json.dumps(_jsonable(self.meta))),
        )
        p.write_bytes(buf.getvalue())

    @classmethod
    def load(cls, path: Path | str) -> "StatTemplateModel":
        """Inverse of :meth:`save` (no pickle; unicode-safe paths)."""
        data = Path(path).read_bytes()
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            version = int(z["format_version"])
            if version > FORMAT_VERSION:
                raise ValueError(f"unsupported StatTemplateModel format version {version}")
            cfg_d = json.loads(str(z["cfg_json"]))
            known = {k: v for k, v in cfg_d.items() if k in DetectorConfig.__dataclass_fields__}
            return cls(
                category=str(z["category"]),
                scene_size=tuple(int(v) for v in z["scene_size"]),
                scale=float(z["scale"]),
                mu_lab=z["mu_lab"].astype(np.float32),
                sigma_lab=z["sigma_lab"].astype(np.float32),
                ref_gray=z["ref_gray"].astype(np.uint8),
                valid=z["valid"],
                calib=json.loads(str(z["calib_json"])),
                cfg=DetectorConfig(**known),
                meta=json.loads(str(z["meta_json"])),
            )
