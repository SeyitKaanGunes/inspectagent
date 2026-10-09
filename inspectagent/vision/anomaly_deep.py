"""PatchCore-lite deep anomaly detector on ``cv2.dnn`` (SPEC §2.7).

Pipeline
--------
1. **Preprocess** - resize the BGR image to the network input size ``(W, H)`` (multiples of 16;
   ``INTER_AREA`` when shrinking so high-resolution scenes and sensor captures alias identically), then
   ``cv2.dnn.blobFromImageWithParams`` (BGR->RGB, ImageNet mean/std, NCHW float32).
2. **Backbone** - torchvision ResNet18 (ImageNet weights) exported to ONNX by ``scripts/export_backbone.py``
   with dynamic H/W; ``net.forward(["layer2", "layer3"])`` returns the stride-8 (128 ch) and stride-16
   (256 ch) maps. Inference uses OpenCV only - torch is a dev-time dependency of the export script.
3. **Patch features** - ``concat(layer2, bilinear_upsample(layer3))`` -> 3x3 average pooling ->
   ``(h, w, 384)``. Done in numpy: OpenCV 5's ``resize``/``blur`` reject >4-channel float maps.
4. **Memory bank** (``fit``) - at most ``deep_train_images`` normals (by default each plus one
   camera-rendered ``nominal`` capture of it, see ``fit(augment=...)``), a seeded random 20 % of each
   view's patches, greedy k-center coreset of ``deep_coreset_size`` (selection in a 128-d Gaussian random
   projection, exact triangle-inequality pruning, chunked distance updates), stored as float16.
5. **Inference** - 1-NN Euclidean distance of every patch to the bank (chunked matmul), map upsampled to
   the capture size, Gaussian sigma=4, image score = max of the smoothed map *inside the frame interior*:
   a border margin of ``BORDER_MARGIN_FRAC`` of the long side (>= 2 map sigmas, ~31 px at 768 px, about two
   stride-8 patches) is excluded from the score and from the regions (``valid_mask`` marks it 0). Pose
   jitter rotates/shifts near-edge content outside the range the bank learned and the camera fills the
   uncovered edge with replicated streaks; the CNN zero padding adds its own edge response. Measured on
   synthetic boards with harsh-range jitter, image AUROC 0.75 with the full map vs 0.97-1.0 with a
   24-32 px margin. Calibration (``fit``) and :meth:`PatchCoreDNN.score_image` use the same interior max.
6. **Calibration** - ``prob = sigmoid(a * (score - mu) / sd + b)``; ``fit`` initialises ``mu``/``sd`` from
   training images held out of the bank (every 10th; leave-one-image-out for tiny sets), ``calibrate``
   replaces them with held-out normal captures and (optionally) fits ``a``/``b`` on labeled data.

DNN engine
----------
Engines are probed in the order ``ENGINE_NEW -> ENGINE_AUTO -> ENGINE_CLASSIC`` (first one that loads the
model and returns correctly shaped outputs wins). On the dev box (i7-13700H, OpenCV 5.0.0) all three load
the dynamic-shape model and match torch (max abs diff ~7e-6 NEW/AUTO, ~3e-5 CLASSIC); forward at 448x320
takes ~31 ms with ENGINE_NEW vs ~49 ms with ENGINE_CLASSIC, so NEW is the default. The model is loaded by
path because the in-memory ``readNetFromONNX(buffer, engine)`` overload ignores ``engine`` in 5.0.0
(non-ASCII paths fall back to it, i.e. to the new engine). The chosen engine is reported in
``PatchCoreDNN.engine`` and in every result's notes.

Zoomed captures
---------------
The bank and the calibration are built from full views. A capture whose window covers a fraction ``f`` of
the scene is fed at ``f * input_size`` (rounded to multiples of 16) so the physical scale of the features
matches the bank, but at such small inputs (224x160 at zoom 2, 112x80 at zoom 4) most patches sit in the
receptive field of the zero-padded border and normal parts score far above the full-view calibration
(measured: clean normals p=1.0 at zoom 2-4 and higher scores than real defects). A deep result on a zoomed
view is therefore returned with ``reliable=False``, ``prob=0.5`` (uninformative; the raw full-view-calibrated
p is kept in the notes) and no regions: it can neither confirm a candidate nor drive the fused probability.
The deep detector is alignment-free, which is why the agent escalates to it on the *full view* when
registration fails.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from inspectagent.config import CameraConfig, DetectorConfig, get_settings
from inspectagent.types import AnomalyResult, Capture, QualityReport, Region
from inspectagent.vision.quality import blocking_quality_issues

__all__ = [
    "AugmentFn",
    "BACKBONE_FILENAME",
    "FEATURE_DIM",
    "IMAGENET_MEAN_RGB",
    "IMAGENET_STD_RGB",
    "INPUT_NAME",
    "OUTPUT_NAMES",
    "PatchCoreDNN",
    "border_margin",
    "camera_augmenter",
    "interior_mask",
    "interior_max",
    "default_onnx_path",
    "greedy_coreset",
    "input_size_for",
    "make_blob",
    "nn_distances",
    "patch_features_from_layers",
    "resize_bilinear_hwc",
]

BACKBONE_FILENAME = "backbone_resnet18.onnx"
INPUT_NAME = "input"
OUTPUT_NAMES = ("layer2", "layer3")
LAYER_CHANNELS = (128, 256)
LAYER_STRIDES = (8, 16)
FEATURE_DIM = 384
SIZE_MULTIPLE = 16  # input sizes must be multiples of the deepest stride
MIN_INPUT_SIDE = 32
IMAGENET_MEAN_RGB = (0.485, 0.456, 0.406)
IMAGENET_STD_RGB = (0.229, 0.224, 0.225)

MAP_SIGMA = 4.0  # Gaussian smoothing of the anomaly map, capture px (SPEC §2.7)
BORDER_MARGIN_FRAC = 0.04  # frame border excluded from score/regions (fraction of the long side, >= 2 MAP_SIGMA)
ZOOM_UNCALIBRATED_PROB = 0.5  # reported prob of a (full-view calibrated) result on a zoomed view
SAMPLE_FRAC = 0.2  # random patch subsample per training image before the coreset
PROJ_DIM = 128  # random projection used only for coreset *selection*
REGION_SD_K = 2.0  # region_thr = mu + REGION_SD_K * sd of normal image scores
REGION_REL = 0.5  # per-image region threshold also >= mu + REGION_REL * (score - mu) (half maximum)
DEFAULT_LOGISTIC = (1.6, -4.0)  # (a, b) fallback when no labeled calibration data exist
HOLDOUT_EVERY = 10  # every 10th training image is held out of the bank for the default calibration
MIN_HOLDOUT = 3  # fewer held-out images than this -> leave-one-image-out calibration instead
CALIB_IMAGES = 32  # max images scored for the default calibration
SD_FLOOR_REL = 0.05  # calibration sd >= 5 % of mu (guards tiny / homogeneous calibration sets)
KNN_MAX_ELEMS = 1 << 25  # distance-matrix block (float32 elements = 128 MiB) for the NN search
FORMAT = "inspectagent.patchcore_dnn/1"
# coreset: pruned (row-gather) update when candidates <= this fraction of points, else contiguous dense
# update; 0.03 measured best-ish for N=76k, k in {1000, 8000} on the dev box (~3x faster than naive at k=8000)
_CORESET_SPARSE_FRAC = 0.03

ENGINE_ORDER = ("new", "auto", "classic")
_ENGINE_ATTR = {"new": "ENGINE_NEW", "auto": "ENGINE_AUTO", "classic": "ENGINE_CLASSIC", "ort": "ENGINE_ORT"}

_NO_REGISTRATION = {
    "method": "none(alignment-free)",
    "ok": True,
    "inliers": 0,
    "inlier_ratio": 0.0,
    "ncc": 0.0,
    "dx": 0.0,
    "dy": 0.0,
    "rot_deg": 0.0,
    "scale": 1.0,
}


# --------------------------------------------------------------------------------------------- helpers
def default_onnx_path() -> Path:
    """``<models_dir>/backbone_resnet18.onnx`` (shared by all categories)."""
    return Path(get_settings().models_dir) / BACKBONE_FILENAME


def _round_to_multiple(v: float, multiple: int = SIZE_MULTIPLE, minimum: int = MIN_INPUT_SIDE) -> int:
    return max(minimum, int(round(v / multiple)) * multiple)


def input_size_for(width: int, height: int, long_side: int = 448) -> tuple[int, int]:
    """Network input ``(W, H)`` keeping the aspect ratio, long side ~``long_side``, multiples of 16."""
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size {width}x{height}")
    s = long_side / max(width, height)
    return _round_to_multiple(width * s), _round_to_multiple(height * s)


def _as_bgr_u8(img: np.ndarray) -> np.ndarray:
    """Coerce gray/BGRA/non-uint8 images to contiguous BGR uint8 (float input is assumed 0..255)."""
    a = np.asarray(img)
    if a.ndim not in (2, 3) or a.size == 0:
        raise ValueError(f"expected an HxW or HxWxC image, got shape {a.shape}")
    if a.dtype != np.uint8:
        if a.dtype == np.uint16:
            a = (a // 257).astype(np.uint8)
        else:
            a = np.clip(np.nan_to_num(a.astype(np.float32)), 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    elif a.shape[2] == 1:
        a = cv2.cvtColor(a[..., 0], cv2.COLOR_GRAY2BGR)
    elif a.shape[2] == 4:
        a = cv2.cvtColor(a, cv2.COLOR_BGRA2BGR)
    elif a.shape[2] != 3:
        raise ValueError(f"unsupported channel count {a.shape[2]}")
    return np.ascontiguousarray(a)


def _blob_params(size: tuple[int, int]) -> "cv2.dnn.Image2BlobParams":
    p = cv2.dnn.Image2BlobParams()
    mean = [255.0 * m for m in IMAGENET_MEAN_RGB]
    scale = [1.0 / (255.0 * s) for s in IMAGENET_STD_RGB]
    p.mean = (mean[0], mean[1], mean[2], 0.0)  # RGB order: applied after swapRB
    p.scalefactor = (scale[0], scale[1], scale[2], 0.0)
    p.size = (int(size[0]), int(size[1]))
    p.swapRB = True
    p.ddepth = cv2.CV_32F
    layout = getattr(cv2, "DNN_LAYOUT_NCHW", getattr(cv2.dnn, "DNN_LAYOUT_NCHW", None))
    if layout is not None:
        p.datalayout = layout
    p.paddingmode = cv2.dnn.DNN_PMODE_NULL
    return p


def make_blob(bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """ImageNet-normalised NCHW float32 blob ``(1, 3, H, W)`` of ``bgr`` resized to ``size=(W, H)``."""
    img = _as_bgr_u8(bgr)
    w, h = int(size[0]), int(size[1])
    if (img.shape[1], img.shape[0]) != (w, h):
        shrink = img.shape[1] > w or img.shape[0] > h
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    return cv2.dnn.blobFromImageWithParams(img, _blob_params((w, h)))


def _interp_matrix(n_out: int, n_in: int) -> np.ndarray:
    """1-D bilinear interpolation matrix (half-pixel centres, edge clamp) = cv2 INTER_LINEAR."""
    a = np.zeros((n_out, n_in), np.float32)
    if n_in == 1:
        a[:, 0] = 1.0
        return a
    s = (np.arange(n_out, dtype=np.float64) + 0.5) * (n_in / n_out) - 0.5
    s = np.clip(s, 0.0, n_in - 1)
    i0 = np.floor(s).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_in - 1)
    w1 = (s - i0).astype(np.float32)
    rows = np.arange(n_out)
    np.add.at(a, (rows, i0), 1.0 - w1)
    np.add.at(a, (rows, i1), w1)
    return a


def resize_bilinear_hwc(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize of an ``(h, w, C)`` float map for any C (matches cv2 INTER_LINEAR / torch
    ``align_corners=False``)."""
    h, w, c = x.shape
    if (h, w) == (out_h, out_w):
        return x.astype(np.float32, copy=False)
    ah = _interp_matrix(out_h, h)
    aw = _interp_matrix(out_w, w)
    t = (ah @ x.reshape(h, w * c).astype(np.float32, copy=False)).reshape(out_h, w, c)
    t = np.ascontiguousarray(t.transpose(1, 0, 2)).reshape(w, out_h * c)
    return np.ascontiguousarray((aw @ t).reshape(out_w, out_h, c).transpose(1, 0, 2))


def _upsample2x_rows(x: np.ndarray, out: np.ndarray) -> None:
    """Exact 2x bilinear upsampling along axis 0 into ``out`` (half-pixel centres, edge clamp)."""
    ev, od = out[0::2], out[1::2]
    np.multiply(x, np.float32(0.75), out=ev)
    od[...] = ev
    q = x * np.float32(0.25)
    ev[1:] += q[:-1]
    ev[0] += q[0]
    od[:-1] += q[1:]
    od[-1] += q[-1]


def _box3_axis(x: np.ndarray, axis: int) -> np.ndarray:
    """Un-normalised 3-tap box sum along axis 0 or 1 with reflect-101 borders (edge if length 1)."""
    s = x.copy()
    v, sv = (x, s) if axis == 0 else (x.transpose(1, 0, 2), s.transpose(1, 0, 2))
    if v.shape[0] >= 2:
        sv[1:] += v[:-1]
        sv[:-1] += v[1:]
        sv[0] += v[1]
        sv[-1] += v[-2]
    else:
        sv *= np.float32(3.0)
    return s


def _avg_pool3(x: np.ndarray) -> np.ndarray:
    """3x3 stride-1 mean filter of an ``(h, w, C)`` map (reflect-101 borders, = cv2.blur), separable."""
    s = _box3_axis(_box3_axis(np.asarray(x, np.float32), 0), 1)
    s *= np.float32(1.0 / 9.0)
    return s


def patch_features_from_layers(layer2: np.ndarray, layer3: np.ndarray) -> np.ndarray:
    """PatchCore patch features ``(h, w, 384)`` from NCHW ``layer2`` (stride 8) and ``layer3`` (stride 16)."""
    if layer2.ndim != 4 or layer3.ndim != 4:
        raise ValueError(f"expected NCHW outputs, got {layer2.shape} and {layer3.shape}")
    c2, h, w = layer2.shape[1:]
    c3, h3, w3 = layer3.shape[1:]
    feat = np.empty((h, w, c2 + c3), np.float32)
    feat[..., :c2] = layer2[0].transpose(1, 2, 0)
    f3 = np.ascontiguousarray(layer3[0].transpose(1, 2, 0), dtype=np.float32)
    if (h, w) == (2 * h3, 2 * w3):  # always the case for inputs that are multiples of 16
        tmp = np.empty((h, w3, c3), np.float32)
        _upsample2x_rows(f3, tmp)
        _upsample2x_rows(tmp.transpose(1, 0, 2), feat[..., c2:].transpose(1, 0, 2))
    else:
        feat[..., c2:] = resize_bilinear_hwc(f3, h, w)
    return _avg_pool3(feat)


def nn_distances(
    queries: np.ndarray,
    bank: np.ndarray,
    bank_sqnorm: np.ndarray | None = None,
    exclude: np.ndarray | None = None,
    max_elems: int = KNN_MAX_ELEMS,
) -> np.ndarray:
    """Euclidean distance of each query row to its nearest ``bank`` row (chunked matmul, float32).

    ``exclude`` is an optional boolean mask over bank rows that must not be matched (used for
    leave-one-image-out calibration).
    """
    q = np.ascontiguousarray(queries, dtype=np.float32)
    b = np.ascontiguousarray(bank, dtype=np.float32)
    if q.ndim != 2 or b.ndim != 2 or q.shape[1] != b.shape[1]:
        raise ValueError(f"shape mismatch: queries {q.shape} vs bank {b.shape}")
    if b.shape[0] == 0:
        raise ValueError("empty memory bank")
    bn = np.einsum("ij,ij->i", b, b) if bank_sqnorm is None else np.asarray(bank_sqnorm, np.float32)
    half = (np.float32(0.5) * bn).astype(np.float32)
    if exclude is not None:
        exclude = np.asarray(exclude, bool)
        if exclude.all():
            raise ValueError("exclude mask removes every bank entry")
        half = np.where(exclude, np.float32(np.inf), half).astype(np.float32)
    qn = np.einsum("ij,ij->i", q, q)
    qt = np.ascontiguousarray(q.T)  # (D, P)
    # min_j |q - b_j|^2 = |q|^2 + 2 * min_j (|b_j|^2 / 2 - q.b_j); chunk along the bank axis with a running
    # min. Large (bank x queries) GEMM blocks are much faster with OpenBLAS than many small ones.
    acc = np.full(q.shape[0], np.inf, np.float32)
    rows = max(1, int(max_elems) // max(1, q.shape[0]))
    for s in range(0, b.shape[0], rows):
        d = b[s : s + rows] @ qt  # (r, P)
        np.subtract(half[s : s + rows, None], d, out=d)
        np.minimum(acc, d.min(axis=0), out=acc)
    out = qn + np.float32(2.0) * acc
    np.maximum(out, 0.0, out=out)
    return np.sqrt(out, out=out)


def greedy_coreset(
    x: np.ndarray,
    k: int,
    proj_dim: int | None = PROJ_DIM,
    seed: int = 0,
    chunk: int = 16384,
) -> np.ndarray:
    """Indices of a greedy k-center (farthest-point) coreset of the rows of ``x``.

    Selection runs in a seeded Gaussian random projection to ``proj_dim`` dims (``None`` = original
    space). Each step adds the point ``s`` farthest from the current centres and updates every point's
    distance to its nearest centre. Exact pruning: a point ``p`` whose nearest centre ``c_p`` satisfies
    ``d(c_p, s) >= 2 d(p, c_p)`` cannot get closer to ``s`` (triangle inequality), so only the remaining
    candidates are updated (rows gathered in ``chunk`` blocks); while candidates exceed
    ``_CORESET_SPARSE_FRAC`` of the points a contiguous dense update is cheaper and used instead. The
    result equals naive greedy up to floating-point ties (verified in the tests).
    """
    x = np.asarray(x)
    n = x.shape[0]
    if x.ndim != 2 or n == 0:
        raise ValueError(f"expected a non-empty 2-D array, got shape {x.shape}")
    if k >= n:
        return np.arange(n, dtype=np.int64)
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if proj_dim is not None and proj_dim < x.shape[1]:
        r = (rng.standard_normal((x.shape[1], proj_dim)) / math.sqrt(proj_dim)).astype(np.float32)
        z = np.empty((n, proj_dim), np.float32)
        for s in range(0, n, chunk):
            z[s : s + chunk] = x[s : s + chunk].astype(np.float32, copy=False) @ r
    else:
        z = np.ascontiguousarray(x, dtype=np.float32)
    sq = np.einsum("ij,ij->i", z, z)
    # einsum (single-threaded SIMD) beats multi-threaded BLAS on the many small/medium mat-vecs below;
    # BLAS is kept for the full dense passes.
    slack_mul, slack_add = np.float32(2.0 * 1.0001), np.float32(1e-6)  # fp-rounding slack on the bound

    def dense_dist(c: np.ndarray, c_sq: float) -> np.ndarray:
        v = z @ c  # one BLAS mat-vec over all points (output is only n floats)
        v *= np.float32(-2.0)
        v += sq
        v += np.float32(c_sq)
        np.maximum(v, 0.0, out=v)
        return np.sqrt(v, out=v)

    sel = np.empty(k, np.int64)
    centers = np.empty((k, z.shape[1]), np.float32)
    csq = np.empty(k, np.float32)
    first = int(rng.integers(n))
    sel[0], centers[0], csq[0] = first, z[first], sq[first]
    min_d = dense_dist(z[first], float(sq[first]))
    min_d[first] = 0.0
    assign = np.zeros(n, np.int32)
    bound: np.ndarray | None = None  # 2 * d(p, c_p) with slack; built when switching to sparse mode
    count = 1
    for i in range(1, k):
        s = int(np.argmax(min_d))
        if min_d[s] <= 0.0:  # every remaining point duplicates a centre
            break
        c = z[s]
        c_sq = float(sq[s])
        sel[i], centers[i], csq[i] = s, c, c_sq
        cand = None
        if bound is not None or i % 8 == 0:  # sparse mode, or periodic test whether pruning pays off
            dcc = np.sqrt(np.maximum(csq[:i] - 2.0 * np.einsum("ij,j->i", centers[:i], c) + c_sq, 0.0))
            b = bound if bound is not None else min_d * slack_mul + slack_add
            cand = np.flatnonzero(dcc[assign] < b)
            if cand.size <= _CORESET_SPARSE_FRAC * n:
                bound = b  # (enter or stay in) sparse mode
            else:
                cand = None  # too many candidates this step: dense update
        if cand is None:
            d = dense_dist(c, c_sq)
            better = d < min_d
            np.minimum(min_d, d, out=min_d)
            assign[better] = i
            if bound is not None:
                np.multiply(min_d, slack_mul, out=bound)
                bound += slack_add
        else:
            du = np.empty(cand.size, np.float32)
            for a in range(0, cand.size, chunk):
                j = cand[a : a + chunk]
                du[a : a + chunk] = np.sqrt(np.maximum(sq[j] - 2.0 * np.einsum("ij,j->i", z[j], c) + c_sq, 0.0))
            better = du < min_d[cand]
            upd, du = cand[better], du[better]
            min_d[upd] = du
            assign[upd] = i
            bound[upd] = du * slack_mul + slack_add
        min_d[s], assign[s] = 0.0, i
        if bound is not None:
            bound[s] = slack_add
        count = i + 1
    return sel[:count]


def _sigmoid(v: float) -> float:
    v = float(np.clip(v, -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(-v))


LOGISTIC_L2_A = 0.5  # ridge on the slope (same as anomaly_stat): keeps separable calibration sets finite
LOGISTIC_L2_B = 1e-3  # weak ridge on the intercept
LOGISTIC_MAX_STEP = 5.0  # Newton step clamp (damping)


def _fit_logistic(
    z: np.ndarray, y: np.ndarray, l2_a: float = LOGISTIC_L2_A, l2_b: float = LOGISTIC_L2_B, iters: int = 100
) -> tuple[float, float]:
    """1-D logistic regression ``P(y=1|z) = sigmoid(a z + b)`` by damped, L2-regularised Newton.

    Calibration sets are small and often perfectly separable (normal vs. clearly defective parts);
    an (almost) unregularised fit then diverges to a step function (observed: ``a=180, b=-800``,
    i.e. probabilities of 1e-26 for mildly elevated scores). The slope ridge ``l2_a`` (shared with
    ``anomaly_stat``) keeps the fit finite and the probabilities graded.
    """
    z = np.asarray(z, np.float64)
    y = np.asarray(y, np.float64)
    theta = np.array(DEFAULT_LOGISTIC, dtype=np.float64)
    xm = np.stack([z, np.ones_like(z)], axis=1)
    reg = np.diag([l2_a, l2_b])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(xm @ theta, -60, 60)))
        g = xm.T @ (p - y) + reg @ theta
        h = (xm * (p * (1 - p))[:, None]).T @ xm + reg
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            break
        big = float(np.max(np.abs(step)))
        if big > LOGISTIC_MAX_STEP:
            step *= LOGISTIC_MAX_STEP / big
        theta -= step
        if big < 1e-9:
            break
    return float(theta[0]), float(theta[1])


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def border_margin(width: int, height: int) -> int:
    """Border (px) excluded from the deep score/regions of a ``width x height`` map (interior stays non-empty)."""
    m = max(int(math.ceil(2.0 * MAP_SIGMA)), int(round(BORDER_MARGIN_FRAC * max(width, height))))
    return int(max(0, min(m, (min(width, height) - 1) // 4)))


def interior_mask(width: int, height: int) -> np.ndarray:
    """uint8 0/255 mask of the scored interior of a ``width x height`` map (see :func:`border_margin`)."""
    m = border_margin(width, height)
    valid = np.zeros((height, width), np.uint8)
    valid[m : height - m, m : width - m] = 255
    return valid


def interior_max(amap: np.ndarray) -> float:
    """Image score: maximum of the anomaly map inside the interior mask."""
    h, w = amap.shape[:2]
    m = border_margin(w, h)
    return float(amap[m : h - m, m : w - m].max())


def _window_fraction(capture: Capture) -> tuple[float, float]:
    """Fraction of the scene (x, y) covered by the capture's commanded window."""
    sw, sh = capture.scene_size
    _, _, ww, wh = capture.window
    return min(1.0, ww / max(float(sw), 1e-9)), min(1.0, wh / max(float(sh), 1e-9))


def _ref_capture_size(scene_size: tuple[int, int]) -> tuple[int, int]:
    """Full-view capture size of the camera for a scene (anomaly-map resolution used for calibration)."""
    w, h = scene_size
    s = CameraConfig().sensor_long_side / max(w, h)
    return max(1, int(round(w * s))), max(1, int(round(h * s)))


def _extract_regions(
    amap: np.ndarray, thr: float, capture: Capture, min_area: int, max_regions: int, id_prefix: str
) -> tuple[list[Region], str | None]:
    """Regions via ``vision.regions.extract_regions``; a spec-equivalent local fallback if it is absent."""
    try:
        from inspectagent.vision.regions import extract_regions
    except ImportError:
        extract_regions = None
    if extract_regions is not None:
        regions = extract_regions(amap, thr, capture, None, "deep", min_area, max_regions, id_prefix)
        return list(regions), None
    prefix = id_prefix or f"{capture.capture_id}_r"  # same id convention as vision.regions
    mask = (amap > thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    c2s = capture.capture_to_scene()
    sx, sy = float(c2s[0, 0]), float(c2s[1, 1])
    cands = []
    for lab in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[lab])
        if area < min_area:
            continue
        sub = amap[y : y + h, x : x + w][labels[y : y + h, x : x + w] == lab]
        peak, mean = float(sub.max()), float(sub.mean())
        cands.append((peak * math.sqrt(area), x, y, w, h, area, peak, mean))
    cands.sort(key=lambda t: -t[0])
    regions = []
    for k, (_, x, y, w, h, area, peak, mean) in enumerate(cands[:max_regions]):
        regions.append(
            Region(
                region_id=f"{prefix}{k + 1}",
                capture_id=capture.capture_id,
                source="deep",
                bbox_capture=(x, y, w, h),
                bbox_scene=(x * sx + float(c2s[0, 2]), y * sy + float(c2s[1, 2]), w * sx, h * sy),
                peak=peak,
                mean=mean,
                area_px=area,
                area_scene_px=float(area * sx * sy),
            )
        )
    return regions, "regions: vision.regions unavailable, used built-in fallback"


# -------------------------------------------------------------------------------------- augmentation
AugmentFn = Callable[[np.ndarray, int, np.random.Generator], list[np.ndarray]]


def camera_augmenter(presets: tuple[str, ...] = ("nominal",), copies: int = 1, glare: bool = False) -> AugmentFn:
    """Augmenter rendering each normal through ``vision.camera.VirtualCamera`` (full view, default settings).

    ``copies`` captures per image, presets cycled; conditions and camera seed are drawn from the ``rng``
    passed by :meth:`PatchCoreDNN.fit` (deterministic). Glare is disabled by default: it is a nuisance the
    agent removes by switching light, not normal appearance the detector should learn.
    """
    if copies < 1 or not presets:
        raise ValueError("camera_augmenter needs copies >= 1 and at least one preset")
    from inspectagent.types import CaptureSettings
    from inspectagent.vision.camera import VirtualCamera, make_conditions

    def augment(img: np.ndarray, index: int, rng: np.random.Generator) -> list[np.ndarray]:
        views = []
        for c in range(copies):
            cond = make_conditions(presets[c % len(presets)], rng)
            if not glare:
                cond.glare_strength = 0.0
            cam = VirtualCamera(img, cond, seed=int(rng.integers(0, 2**31 - 1)))
            views.append(cam.capture(CaptureSettings(), f"aug_{index}_{c}", 0).image)
        return views

    return augment


def _resolve_augment(augment: "AugmentFn | str | None", log: Callable[[str], None]) -> tuple[AugmentFn | None, str]:
    """Map the ``fit(augment=...)`` argument to (callable or None, description)."""
    if augment is None or augment == "none":
        return None, "none"
    if augment == "camera":
        try:
            return camera_augmenter(), "camera(nominal x1, no glare)"
        except ImportError as e:
            log(f"[deep] camera augmentation unavailable ({e}); fitting on the given images only")
            return None, "none(camera unavailable)"
    if callable(augment):
        return augment, getattr(augment, "__name__", "custom")
    raise ValueError(f"augment must be 'camera', None or a callable, got {augment!r}")


# ----------------------------------------------------------------------------------------- net cache
@dataclass
class _NetHandle:
    net: "cv2.dnn.Net"
    engine: str
    dynamic: bool
    lock: threading.Lock
    load_ms: float
    tried: list[str]


_NET_CACHE: dict[tuple, _NetHandle] = {}
_NET_CACHE_LOCK = threading.Lock()


def _check_outputs(outs: tuple, w: int, h: int) -> bool:
    if len(outs) != 2:
        return False
    exp = [(1, c, h // s, w // s) for c, s in zip(LAYER_CHANNELS, LAYER_STRIDES)]
    return all(tuple(o.shape) == e for o, e in zip(outs, exp))


def _load_net(onnx_path: Path, engine: str | None, probe_size: tuple[int, int]) -> _NetHandle:
    """Load (or reuse) the backbone with the first working engine; nets are shared per file+engine."""
    onnx_path = Path(onnx_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"backbone ONNX not found: {onnx_path} (run scripts/export_backbone.py)")
    st = onnx_path.stat()
    order = (engine,) if engine else ENGINE_ORDER
    key = (str(onnx_path.resolve()), st.st_size, st.st_mtime_ns, order)
    with _NET_CACHE_LOCK:
        if key in _NET_CACHE:
            return _NET_CACHE[key]
        # The path overload honours ``engine``; the in-memory overload (needed only for non-ASCII paths,
        # which cv2 cannot open on Windows) ignores it in OpenCV 5.0.0 and always runs the new engine.
        path_str = str(onnx_path.resolve())
        source: str | np.ndarray = path_str
        tried: list[str] = []
        if not path_str.isascii():
            source = np.frombuffer(onnx_path.read_bytes(), np.uint8)
            tried.append(f"non-ASCII path: in-memory load, engine fixed to 'new' (requested {list(order)})")
            order = ("new",)
        pw, ph = probe_size
        rng = np.random.default_rng(0)
        for name in order:
            attr = _ENGINE_ATTR.get(name)
            if attr is None or not hasattr(cv2.dnn, attr):
                tried.append(f"{name}: not available in this OpenCV build")
                continue
            t0 = time.perf_counter()
            try:
                net = cv2.dnn.readNetFromONNX(source, getattr(cv2.dnn, attr))
                net.setInput(rng.standard_normal((1, 3, ph, pw)).astype(np.float32))
                outs = net.forward(list(OUTPUT_NAMES))
                if not _check_outputs(outs, pw, ph):
                    tried.append(f"{name}: unexpected output shapes {[o.shape for o in outs]}")
                    continue
                load_ms = (time.perf_counter() - t0) * 1e3
                # dynamic H/W support: a second, different size must give proportionally sized outputs
                qw = _round_to_multiple(pw * 0.5)
                qh = _round_to_multiple(ph * 0.75)
                try:
                    net.setInput(np.zeros((1, 3, qh, qw), np.float32))
                    dynamic = _check_outputs(net.forward(list(OUTPUT_NAMES)), qw, qh)
                except cv2.error:
                    dynamic = False
            except Exception as e:  # cv2.error / SystemError from a broken build: record and try the next
                msg = str(e).strip().splitlines()
                tried.append(f"{name}: {type(e).__name__}: {(msg[-1] if msg else '')[:200]}")
                continue
            handle = _NetHandle(net, name, dynamic, threading.Lock(), load_ms, tried + [f"{name}: ok"])
            _NET_CACHE[key] = handle
            return handle
    raise RuntimeError(f"no cv2.dnn engine could run {onnx_path.name}: " + "; ".join(tried))


# ---------------------------------------------------------------------------------------------- model
class PatchCoreDNN:
    """Alignment-free PatchCore-lite detector (ResNet18 layer2+layer3 features via ``cv2.dnn``)."""

    def __init__(self, onnx_path: Path, input_size: tuple[int, int], engine: str | None = None) -> None:
        w, h = int(input_size[0]), int(input_size[1])
        if w % SIZE_MULTIPLE or h % SIZE_MULTIPLE or min(w, h) < MIN_INPUT_SIDE:
            raise ValueError(f"input_size must be multiples of {SIZE_MULTIPLE} and >= {MIN_INPUT_SIDE}, got {input_size}")
        if engine is not None and engine not in _ENGINE_ATTR:
            raise ValueError(f"engine must be one of {sorted(_ENGINE_ATTR)}, got {engine!r}")
        self.onnx_path = Path(onnx_path)
        self.input_size: tuple[int, int] = (w, h)
        self.requested_engine = engine
        self.category: str = ""
        self.scene_size: tuple[int, int] | None = None
        self.cfg = DetectorConfig()
        self.zoom_consistent = True
        self.memory_bank: np.ndarray | None = None  # (N, 384) float16
        self.calib: dict = {
            "mu": 0.0,
            "sd": 1.0,
            "a": DEFAULT_LOGISTIC[0],
            "b": DEFAULT_LOGISTIC[1],
            "region_thr": float("inf"),
            "n_normal": 0,
            "n_labeled": 0,
            "source": "uncalibrated",
        }
        self.fit_info: dict = {}
        self.onnx_sha256: str | None = None
        self.last_timing: dict[str, float] = {}
        self._bank32: np.ndarray | None = None
        self._bank_sqn: np.ndarray | None = None
        self._net: _NetHandle | None = None

    # ------------------------------------------------------------------ network
    def _handle(self) -> _NetHandle:
        if self._net is None:
            self._net = _load_net(self.onnx_path, self.requested_engine, self.input_size)
        return self._net

    @property
    def engine(self) -> str:
        """Name of the cv2.dnn engine in use (loads the network on first access)."""
        return self._handle().engine

    @property
    def dynamic_input(self) -> bool:
        """True when the ONNX graph accepts arbitrary H/W (multiples of 16)."""
        return self._handle().dynamic

    def warmup(self) -> dict:
        """Load the network and run one forward pass; returns engine info."""
        h = self._handle()
        self.features(np.zeros((self.input_size[1], self.input_size[0], 3), np.uint8))
        return {"engine": h.engine, "dynamic": h.dynamic, "load_ms": round(h.load_ms, 2), "tried": list(h.tried)}

    def features(self, bgr: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
        """Patch features ``(H/8, W/8, 384)`` float32 of ``bgr`` resized to ``size`` (default ``input_size``)."""
        h = self._handle()
        size = self.input_size if size is None else (int(size[0]), int(size[1]))
        if size != self.input_size:
            if size[0] % SIZE_MULTIPLE or size[1] % SIZE_MULTIPLE or min(size) < MIN_INPUT_SIDE:
                raise ValueError(f"size must be multiples of {SIZE_MULTIPLE} and >= {MIN_INPUT_SIDE}, got {size}")
            if not h.dynamic:
                size = self.input_size  # fixed-shape export: always run at the export size
        blob = make_blob(bgr, size)
        with h.lock:  # cv2.dnn.Net is not thread-safe
            h.net.setInput(blob)
            outs = h.net.forward(list(OUTPUT_NAMES))
            l2, l3 = np.array(outs[0], copy=True), np.array(outs[1], copy=True)
        if not _check_outputs((l2, l3), size[0], size[1]):
            raise RuntimeError(f"backbone returned unexpected shapes {l2.shape}, {l3.shape} for input {size}")
        return patch_features_from_layers(l2, l3)

    # ------------------------------------------------------------------ bank / scoring
    @property
    def is_fitted(self) -> bool:
        return self.memory_bank is not None and len(self.memory_bank) > 0

    def _set_bank(self, bank: np.ndarray) -> None:
        bank = np.asarray(bank)
        if bank.ndim != 2 or bank.shape[1] != FEATURE_DIM or bank.shape[0] == 0:
            raise ValueError(f"memory bank must be (N, {FEATURE_DIM}), got {bank.shape}")
        self.memory_bank = bank.astype(np.float16)
        self._bank32 = self.memory_bank.astype(np.float32)
        self._bank_sqn = np.einsum("ij,ij->i", self._bank32, self._bank32)

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("PatchCoreDNN has no memory bank: call fit() or load() first")

    def input_size_for_capture(self, capture: Capture) -> tuple[int, int]:
        """Input size keeping the bank's physical scale for zoomed captures (see module docstring)."""
        if not self.zoom_consistent or not self.dynamic_input:
            return self.input_size
        fx, fy = _window_fraction(capture)
        if fx >= 0.999 and fy >= 0.999:
            return self.input_size
        return _round_to_multiple(self.input_size[0] * fx), _round_to_multiple(self.input_size[1] * fy)

    def _finalize_map(self, dist: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
        """Upsample a patch-distance map to ``out_size=(W, H)`` and smooth (Gaussian sigma=4)."""
        amap = cv2.resize(dist.astype(np.float32), (int(out_size[0]), int(out_size[1])), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(amap, (0, 0), MAP_SIGMA, borderType=cv2.BORDER_REFLECT)

    def _compute_map(
        self, bgr: np.ndarray, out_size: tuple[int, int] | None, size: tuple[int, int] | None
    ) -> tuple[np.ndarray, dict[str, float]]:
        self._require_fitted()
        t0 = time.perf_counter()
        f = self.features(bgr, size)
        t1 = time.perf_counter()
        d = nn_distances(f.reshape(-1, FEATURE_DIM), self._bank32, self._bank_sqn).reshape(f.shape[:2])
        t2 = time.perf_counter()
        if out_size is None:
            out_size = (bgr.shape[1], bgr.shape[0])
        amap = self._finalize_map(d, out_size)
        t3 = time.perf_counter()
        timing = {
            "features_ms": (t1 - t0) * 1e3,
            "knn_ms": (t2 - t1) * 1e3,
            "post_ms": (t3 - t2) * 1e3,
            "total_ms": (t3 - t0) * 1e3,
        }
        self.last_timing = timing  # informational only (models are shared between threads)
        return amap, timing

    def anomaly_map(
        self, bgr: np.ndarray, out_size: tuple[int, int] | None = None, size: tuple[int, int] | None = None
    ) -> np.ndarray:
        """Smoothed anomaly map of a raw image at ``out_size`` (default: image size), float32."""
        return self._compute_map(bgr, out_size, size)[0]

    def score_image(self, bgr: np.ndarray) -> float:
        """Raw image score (interior max of the smoothed map at the camera's full-view resolution)."""
        ref = _ref_capture_size((bgr.shape[1], bgr.shape[0]))
        return interior_max(self.anomaly_map(bgr, out_size=ref))

    def predict(self, capture: Capture, id_prefix: str = "", quality: QualityReport | None = None) -> AnomalyResult:
        """Deep anomaly result for a capture: map at capture size, regions, calibrated probability.

        ``score`` is the interior max of the map (the border margin, see :func:`border_margin`, is excluded
        from score and regions and marked 0 in ``valid_mask``). ``reliable`` is True when inference succeeded
        (the detector is alignment-free), the capture is a *full view* (results on zoomed views are not covered
        by the full-view calibration: ``reliable=False``, ``prob=0.5``, no regions - see the module docstring)
        and, when the capture's ``quality`` report is given, it has no blocking issue
        (``quality.blocking_quality_issues``: glare, clipping, crushed shadows, blur or heavy noise also shift the
        patch features - specular glare in particular scores as an anomaly).
        """
        self._require_fitted()
        t0 = time.perf_counter()
        w, h = capture.size
        notes: list[str] = []
        try:
            size = self.input_size_for_capture(capture)
            amap, t = self._compute_map(capture.image, (w, h), size)
        except (cv2.error, RuntimeError, ValueError, OSError) as e:
            return AnomalyResult(
                capture_id=capture.capture_id,
                detector="deep",
                score=0.0,
                prob=0.5,
                regions=[],
                reliable=False,
                registration=dict(_NO_REGISTRATION),
                latency_ms=(time.perf_counter() - t0) * 1e3,
                notes=[f"deep inference failed: {type(e).__name__}: {str(e).strip()[:300]}"],
            )
        valid = interior_mask(w, h)
        score = float(amap[valid > 0].max())
        prob = self.prob(score)
        fx, fy = _window_fraction(capture)
        zoomed = fx < 0.999 or fy < 0.999
        thr = self.region_threshold(score)
        regions: list[Region] = []
        if zoomed:
            notes.append(
                f"zoomed view (window {fx:.2f}x{fy:.2f} of the scene): deep calibration is full-view only -> "
                f"result not reliable, prob reported as {ZOOM_UNCALIBRATED_PROB:.1f} (raw full-view-calibrated p={prob:.3f}), no regions"
            )
            prob = ZOOM_UNCALIBRATED_PROB
        elif math.isfinite(thr):
            scored = np.where(valid > 0, amap, np.float32(min(float(amap.min()), thr - 1.0))).astype(np.float32)
            regions, note = _extract_regions(scored, thr, capture, self.cfg.min_region_area, self.cfg.max_regions, id_prefix)
            if note:
                notes.append(note)
        else:
            notes.append("uncalibrated: no region threshold")
        notes.append(
            f"engine={self._handle().engine} input={size[0]}x{size[1]} features={t['features_ms']:.1f}ms "
            f"knn={t['knn_ms']:.1f}ms bank={len(self.memory_bank)} border_margin={border_margin(w, h)}px"
        )
        if str(self.calib.get("source", "")).startswith(("uncalibrated", "fit_")):
            notes.append(f"calibration={self.calib.get('source')}")
        reliable = not zoomed
        if quality is not None and not quality.ok:
            blocking = blocking_quality_issues(quality)
            if blocking:
                reliable = False
                notes.append("quality issues: " + ",".join(quality.issues))
            else:
                notes.append("quality issues tolerated: " + ",".join(quality.issues))
        return AnomalyResult(
            capture_id=capture.capture_id,
            detector="deep",
            score=score,
            prob=prob,
            regions=regions,
            reliable=reliable,
            registration=dict(_NO_REGISTRATION),
            latency_ms=(time.perf_counter() - t0) * 1e3,
            notes=notes,
            anomaly_map=amap,
            valid_mask=valid,
        )

    # ------------------------------------------------------------------ calibration
    def calibrate(self, normal_scores: list[float], labeled: list[tuple[float, int]] | None = None) -> None:
        """Set ``mu``/``sd``/``region_thr`` from normal scores; fit the logistic on labeled (score, label) pairs.

        Without labels (or with a single class / a non-increasing fit) ``a, b`` fall back to (1.6, -4.0).
        With no finite normal score the current ``mu``/``sd`` (e.g. from ``fit``) are kept.
        """
        s = np.asarray([float(v) for v in normal_scores if np.isfinite(v)], np.float64)
        if s.size:
            mu = float(s.mean())
            sd = float(s.std(ddof=1)) if s.size > 1 else 0.0
            sd = max(sd, SD_FLOOR_REL * abs(mu), 1e-6)
        elif self.calib.get("source") != "uncalibrated":
            mu, sd = float(self.calib["mu"]), float(self.calib["sd"])
        else:
            raise ValueError("calibrate() needs at least one finite normal score for an uncalibrated model")
        a, b = DEFAULT_LOGISTIC
        n_lab = 0
        if labeled:
            arr = np.asarray([(float(v), int(y)) for v, y in labeled if np.isfinite(v)], np.float64).reshape(-1, 2)
            n_lab = len(arr)
            ys = arr[:, 1]
            if n_lab >= 4 and ys.min() == 0 and ys.max() == 1:
                fa, fb = _fit_logistic((arr[:, 0] - mu) / sd, (ys > 0).astype(np.float64))
                if fa > 0 and np.isfinite(fa) and np.isfinite(fb):
                    a, b = fa, fb
        self.calib = {
            "mu": mu,
            "sd": sd,
            "a": float(a),
            "b": float(b),
            "region_thr": mu + REGION_SD_K * sd,
            "n_normal": int(s.size) if s.size else int(self.calib.get("n_normal", 0)),
            "n_labeled": int(n_lab),
            "source": "calibrate" if (a, b) == DEFAULT_LOGISTIC else "calibrate+logistic",
        }

    def region_threshold(self, score: float) -> float:
        """Map threshold for regions: ``max(region_thr, mu + REGION_REL * (score - mu))``.

        ``region_thr`` (normal-score level) decides *whether* a region exists; the half-maximum term keeps
        regions on the core of strong peaks instead of the smoothed halo around them.
        """
        base = float(self.calib.get("region_thr", float("inf")))
        if not math.isfinite(base):
            return base
        return max(base, float(self.calib["mu"]) + REGION_REL * (float(score) - float(self.calib["mu"])))

    def prob(self, score: float) -> float:
        """Calibrated P(anomalous) = sigmoid(a * (score - mu) / sd + b)."""
        c = self.calib
        return _sigmoid(c["a"] * (float(score) - c["mu"]) / c["sd"] + c["b"])

    # ------------------------------------------------------------------ training
    @classmethod
    def fit(
        cls,
        category: str,
        normal_images: Iterable[np.ndarray],
        onnx_path: Path,
        cfg: DetectorConfig | None = None,
        log: Callable[[str], None] = print,
        *,
        input_size: tuple[int, int] | None = None,
        seed: int = 0,
        engine: str | None = None,
        coreset_size: int | None = None,
        sample_frac: float = SAMPLE_FRAC,
        holdout_every: int = HOLDOUT_EVERY,
        calib_images: int = CALIB_IMAGES,
        augment: "AugmentFn | str | None" = "camera",
    ) -> "PatchCoreDNN":
        """Build the memory bank from normal images (greedy coreset) plus a default calibration.

        At most ``cfg.deep_train_images`` images are consumed (an iterable is read lazily). ``augment``
        adds extra views of every normal to the bank: ``"camera"`` (default) = one full-view capture
        rendered by ``vision.camera.VirtualCamera`` under ``nominal`` conditions with glare disabled
        (pose jitter, noise, JPEG, slight defocus/EV - the imaging variation the detector sees at
        inspection time; a bank of clean scenes alone flags such normal captures as anomalous), ``None``
        = no augmentation, or a callable ``(image, index, rng) -> list[image]``. If the camera module
        cannot be imported, ``"camera"`` degrades to no augmentation (logged).

        Every ``holdout_every``-th image (with its views; up to ``calib_images`` images) is held out of
        the bank and scored for an unbiased default calibration (``source="fit_holdout"``). With fewer
        than ``MIN_HOLDOUT`` held-out images they are folded back into the bank and training images are
        scored leave-one-image-out instead (``source="fit_loo"``; conservative: removing an image's
        coreset centres leaves holes, so its scores run high). Either way, :meth:`calibrate` on held-out
        captures should replace it.
        """
        cfg = cfg or DetectorConfig()
        if not 0.0 < sample_frac <= 1.0:
            raise ValueError("sample_frac must be in (0, 1]")
        aug_fn, aug_desc = _resolve_augment(augment, log)
        rng = np.random.default_rng(seed)
        aug_rng = np.random.default_rng([seed, 0xA06])  # separate stream: augmentation never shifts sampling
        t_start = time.perf_counter()
        model: PatchCoreDNN | None = None
        feats: list[np.ndarray] = []
        srcs: list[np.ndarray] = []
        train_cache: list[tuple[int, np.ndarray]] = []  # (image index, float16 view features) for LOO
        holdout: list[tuple[int, np.ndarray]] = []
        fmap_shape: tuple[int, int] = (0, 0)
        n_img = n_views = 0
        n_holdout_img = 0
        cached_imgs: set[int] = set()
        t_feat = t_aug = 0.0

        def add_to_bank(i: int, flat: np.ndarray) -> None:
            n_take = max(1, int(round(sample_frac * flat.shape[0])))
            idx = np.sort(rng.choice(flat.shape[0], n_take, replace=False))
            feats.append(flat[idx].astype(np.float32, copy=False))
            srcs.append(np.full(n_take, i, np.int32))
            if i in cached_imgs or len(cached_imgs) < calib_images:
                cached_imgs.add(i)
                train_cache.append((i, flat.astype(np.float16)))

        for img in normal_images:
            if n_img >= cfg.deep_train_images:
                break
            img = _as_bgr_u8(img)
            if model is None:
                hh, ww = img.shape[:2]
                size = input_size or input_size_for(ww, hh, cfg.deep_input_long_side)
                model = cls(onnx_path, size, engine=engine)
                model.category, model.scene_size, model.cfg = category, (ww, hh), cfg
            views = [img]
            if aug_fn is not None:
                t0 = time.perf_counter()
                views += [_as_bgr_u8(v) for v in aug_fn(img, n_img, aug_rng)]
                t_aug += time.perf_counter() - t0
            is_holdout = (
                holdout_every > 1 and n_img % holdout_every == holdout_every - 1 and n_holdout_img < calib_images
            )
            n_holdout_img += int(is_holdout)
            for view in views:
                t0 = time.perf_counter()
                f = model.features(view)
                t_feat += time.perf_counter() - t0
                fmap_shape = f.shape[:2]
                flat = f.reshape(-1, FEATURE_DIM)
                if is_holdout:
                    holdout.append((n_img, flat.astype(np.float16)))
                else:
                    add_to_bank(n_img, flat)
                n_views += 1
            n_img += 1
        if model is None or n_img < 2:
            raise ValueError(f"PatchCoreDNN.fit needs at least 2 normal images, got {n_img}")
        mode = "holdout" if n_holdout_img >= MIN_HOLDOUT else "loo"
        if mode == "loo":
            for i, flat16 in holdout:
                add_to_bank(i, flat16.astype(np.float32))
            holdout, n_holdout_img = [], 0
        x = np.concatenate(feats)
        src = np.concatenate(srcs)
        del feats
        k = min(int(coreset_size or cfg.deep_coreset_size), x.shape[0])
        t0 = time.perf_counter()
        sel = greedy_coreset(x, k, PROJ_DIM, seed=seed)
        t_core = time.perf_counter() - t0
        model._set_bank(x[sel])
        bank_src = src[sel]
        n_bank_img = n_img - n_holdout_img
        log(
            f"[deep:{category}] {n_bank_img} bank images ({n_views} views incl. held-out, augment={aug_desc}), "
            f"{x.shape[0]} sampled patches -> coreset {len(sel)} (features {t_feat:.1f}s, augment {t_aug:.1f}s, "
            f"coreset {t_core:.1f}s, engine={model.engine})"
        )
        ref_size = _ref_capture_size(model.scene_size)
        scores = []
        for j, flat16 in holdout if mode == "holdout" else train_cache:
            exclude = None if mode == "holdout" else (bank_src == j)
            if exclude is not None and exclude.all():
                continue
            d = nn_distances(flat16.astype(np.float32), model._bank32, model._bank_sqn, exclude=exclude)
            scores.append(interior_max(model._finalize_map(d.reshape(fmap_shape), ref_size)))
        model.calibrate(scores)
        model.calib["source"] = f"fit_{mode}"
        model.onnx_sha256 = _file_sha256(model.onnx_path)
        model.fit_info = {
            "n_images": n_img,
            "n_bank_images": n_bank_img,
            "n_views": n_views,
            "augment": aug_desc,
            "n_sampled_patches": int(x.shape[0]),
            "coreset_size": int(len(sel)),
            "feature_map": [int(fmap_shape[0]), int(fmap_shape[1])],
            "sample_frac": float(sample_frac),
            "seed": int(seed),
            "calib_mode": mode,
            "n_calib_scores": len(scores),
            "features_seconds": round(t_feat, 3),
            "augment_seconds": round(t_aug, 3),
            "coreset_seconds": round(t_core, 3),
            "fit_seconds": round(time.perf_counter() - t_start, 3),
        }
        log(
            f"[deep:{category}] default calibration ({mode}, n={len(scores)}): mu={model.calib['mu']:.4f} "
            f"sd={model.calib['sd']:.4f} region_thr={model.calib['region_thr']:.4f}"
        )
        return model

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        """Write the model to ``path`` (npz: float16 bank + JSON metadata), atomically."""
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "format": FORMAT,
            "category": self.category,
            "input_size": list(self.input_size),
            "scene_size": list(self.scene_size) if self.scene_size else None,
            "feature_dim": FEATURE_DIM,
            "calib": self.calib,
            "fit_info": self.fit_info,
            "onnx_name": self.onnx_path.name,
            "onnx_sha256": self.onnx_sha256,
            "zoom_consistent": self.zoom_consistent,
            "min_region_area": self.cfg.min_region_area,
            "max_regions": self.cfg.max_regions,
        }
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, memory_bank=self.memory_bank.astype(np.float16), meta=np.array(json.dumps(meta)))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path, onnx_path: Path, engine: str | None = None) -> "PatchCoreDNN":
        """Load a model written by :meth:`save`; the backbone is loaded lazily on first use."""
        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            bank = z["memory_bank"]
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path}: unsupported model format {meta.get('format')!r}")
        if int(meta.get("feature_dim", -1)) != FEATURE_DIM:
            raise ValueError(f"{path}: feature_dim {meta.get('feature_dim')} != {FEATURE_DIM}")
        model = cls(onnx_path, tuple(meta["input_size"]), engine=engine)
        model.category = meta.get("category", "")
        model.scene_size = tuple(meta["scene_size"]) if meta.get("scene_size") else None
        model.calib = dict(meta["calib"])
        model.fit_info = dict(meta.get("fit_info", {}))
        model.zoom_consistent = bool(meta.get("zoom_consistent", True))
        model.cfg.min_region_area = int(meta.get("min_region_area", model.cfg.min_region_area))
        model.cfg.max_regions = int(meta.get("max_regions", model.cfg.max_regions))
        model.onnx_sha256 = meta.get("onnx_sha256")
        model._set_bank(bank)
        onnx_path = Path(onnx_path)
        if model.onnx_sha256 and onnx_path.is_file() and _file_sha256(onnx_path) != model.onnx_sha256:
            warnings.warn(
                f"{onnx_path.name} differs from the backbone used to fit {path} (sha256 mismatch); "
                "features may be incompatible - refit or re-export with the same weights",
                RuntimeWarning,
                stacklevel=2,
            )
        return model

    def __repr__(self) -> str:
        n = 0 if self.memory_bank is None else len(self.memory_bank)
        return f"PatchCoreDNN(category={self.category!r}, input_size={self.input_size}, bank={n})"
