"""Image registration: ORB/affine -> ECC refinement -> ECC-only -> phase-correlation fallback chain.

``register(moving, fixed)`` estimates a 2x3 affine ``M`` that maps **moving** pixel
coordinates to **fixed** pixel coordinates, i.e. ``cv2.warpAffine(moving, M, fixed_size)``
renders the moving image in the fixed frame.  Pixel coordinates follow the OpenCV
``warpAffine`` convention (integer coordinates are pixel centres).

Chain (SPEC §2.5):

1. CLAHE-normalise both images (robust to exposure / light changes).
2. ORB (4000 features) + brute-force Hamming kNN matching with Lowe ratio 0.8.
3. ``estimateAffinePartial2D`` (RANSAC, 3 px) -> accepted if inliers >= ``min_inliers``,
   inlier ratio >= ``min_inlier_ratio``, |rotation| <= 8 deg and scale in [0.85, 1.15].
4. ECC refinement (Euclidean, or affine when the similarity has a noticeable scale) on a
   0.5x pyramid level; kept only when it does not lower the NCC. A *strong* ORB solution
   (>= 2 x ``min_inliers`` inliers) with ``ncc >= min_ncc`` is returned right away.
5. Otherwise (ORB failed, or a weak, near-threshold ORB fit that may be a wrong local solution)
   the alignment-based candidates are computed as well: ECC alone from the identity, and a
   **rotation-grid phase correlation** (moving level rotated over -3..3 deg in 0.5 deg steps,
   ``phaseCorrelate`` per angle, best NCC), refined by ECC only when the refinement moves the
   corners by <= 12 px and does not lower the NCC. Translation-only phase correlation used to
   leave the pose rotation as a 7-30 px residual at the corners of zoomed views that still
   passed the NCC test (false alarms); ECC started from it diverged.
6. The candidate with the highest normalised cross-correlation (NCC, computed on the 0.5x
   level, warped moving vs. fixed inside the valid area) wins; ``ok = ncc >= min_ncc``.

ORB features of the fixed image are cached by content hash (the commanded-view reference
renders repeat across parts), which makes repeated registrations to the same view cheaper.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import cv2
import numpy as np

from inspectagent.config import DetectorConfig

__all__ = ["RegistrationResult", "register", "compose_affine", "invert_affine", "warp_points", "decompose_affine"]

ORB_FEATURES = 4000
RATIO_TEST = 0.8
RANSAC_THRESHOLD_PX = 3.0
MAX_ROT_DEG = 8.0
SCALE_RANGE = (0.85, 1.15)
ECC_LEVEL_SCALE = 0.5
ECC_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-5)
_KEYPOINT_MASK_ERODE_PX = 15
_FIXED_CACHE_MAX = 8
PHASE_ROT_ANGLES = tuple(float(a) for a in np.arange(-3.0, 3.01, 0.5))  # deg: covers the pose-jitter range
STRONG_ORB_FACTOR = 2  # ORB solutions with < STRONG_ORB_FACTOR * min_inliers inliers are cross-checked
FALLBACK_ECC_MAX_DRIFT_PX = 12.0  # ECC refining a phase solution may move the corners at most this much


# --------------------------------------------------------------------------------------
# small affine helpers (shared with anomaly_stat / regions)
# --------------------------------------------------------------------------------------
def _to3(M: np.ndarray) -> np.ndarray:
    out = np.eye(3, dtype=np.float64)
    out[:2, :] = np.asarray(M, dtype=np.float64).reshape(2, 3)
    return out


def compose_affine(*mats: np.ndarray) -> np.ndarray:
    """Compose 2x3 affines: ``compose_affine(A, B, C)`` maps x -> A(B(C(x)))."""
    out = np.eye(3, dtype=np.float64)
    for m in mats:
        out = out @ _to3(m)
    return out[:2, :].copy()


def invert_affine(M: np.ndarray) -> np.ndarray:
    """Inverse of a 2x3 affine (float64)."""
    return cv2.invertAffineTransform(np.asarray(M, dtype=np.float64).reshape(2, 3))


def warp_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine to an (N, 2) array of points."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    M = np.asarray(M, dtype=np.float64).reshape(2, 3)
    return p @ M[:, :2].T + M[:, 2]


def decompose_affine(M: np.ndarray, size: tuple[int, int] | None = None) -> tuple[float, float, float, float]:
    """(dx, dy, rot_deg, scale) of a 2x3 affine.

    ``dx, dy`` is the displacement of the image centre (``size=(w, h)``) or of the origin
    when ``size`` is None; ``rot_deg`` is the rotation angle in the y-down pixel frame
    (``atan2`` of the averaged rotation part); ``scale`` is ``sqrt(|det A|)``.
    """
    M = np.asarray(M, dtype=np.float64).reshape(2, 3)
    rot = math.degrees(math.atan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1]))
    scale = math.sqrt(abs(float(np.linalg.det(M[:, :2]))))
    if size is None:
        dx, dy = float(M[0, 2]), float(M[1, 2])
    else:
        c = np.array([[(size[0] - 1) / 2.0, (size[1] - 1) / 2.0]])
        d = warp_points(M, c)[0] - c[0]
        dx, dy = float(d[0]), float(d[1])
    return dx, dy, rot, scale


# --------------------------------------------------------------------------------------
# result type
# --------------------------------------------------------------------------------------
@dataclass
class RegistrationResult:
    """Outcome of :func:`register`. ``M`` maps MOVING (capture) px -> FIXED (reference) px.

    ``inliers``/``inlier_ratio`` always describe the ORB stage (0 when ORB found nothing),
    ``method`` names the stage that produced ``M`` (``orb``, ``orb+ecc``, ``ecc``, ``phase``,
    ``phase+ecc`` or ``identity``); ``dx, dy`` is the displacement of the moving image
    centre, ``rot_deg``/``scale`` the rotation/scale of ``M``.
    """

    ok: bool
    method: str
    M: np.ndarray = field(repr=False)
    inliers: int = 0
    inlier_ratio: float = 0.0
    ncc: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    rot_deg: float = 0.0
    scale: float = 1.0

    def to_dict(self, include_M: bool = False) -> dict:
        """JSON-friendly summary (``M`` only when ``include_M``)."""
        d = {
            "method": self.method,
            "ok": bool(self.ok),
            "inliers": int(self.inliers),
            "inlier_ratio": round(float(self.inlier_ratio), 4),
            "ncc": round(float(self.ncc), 4),
            "dx": round(float(self.dx), 3),
            "dy": round(float(self.dy), 3),
            "rot_deg": round(float(self.rot_deg), 4),
            "scale": round(float(self.scale), 5),
        }
        if include_M:
            d["M"] = [[round(float(v), 6) for v in row] for row in np.asarray(self.M).reshape(2, 3)]
        return d


# --------------------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------------------
_TLS = threading.local()


def _clahe() -> cv2.CLAHE:
    c = getattr(_TLS, "clahe", None)
    if c is None:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        _TLS.clahe = c
    return c


def _orb() -> cv2.ORB:
    return cv2.ORB_create(nfeatures=ORB_FEATURES, scaleFactor=1.2, nlevels=8, fastThreshold=12)


def _float_to_u8(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    hi = float(np.nanmax(a)) if a.size else 1.0
    if hi <= 1.0 + 1e-6:  # assume [0, 1]
        a = a * 255.0
    return np.clip(a + 0.5, 0, 255).astype(np.uint8)


def _as_gray_u8(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 3:
        if a.shape[2] == 1:
            a = a[..., 0]
        else:
            if a.dtype != np.uint8:
                a = _float_to_u8(a)
            a = cv2.cvtColor(np.ascontiguousarray(a[..., :3]), cv2.COLOR_BGR2GRAY)
    if a.dtype != np.uint8:
        a = _float_to_u8(a)
    return np.ascontiguousarray(a)


def _mask_u8(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.full(shape, 255, np.uint8)
    m = np.asarray(mask)
    if m.shape[:2] != shape:
        raise ValueError(f"valid mask shape {m.shape[:2]} != image shape {shape}")
    if m.dtype == np.uint8:
        return cv2.compare(m, 0, cv2.CMP_GT)
    return np.where(m > 0, 255, 0).astype(np.uint8)


def _level_map(src_size: tuple[int, int], dst_size: tuple[int, int]) -> np.ndarray:
    """2x3 map full-res px -> ``cv2.resize``-d level px (pixel-centre convention); sizes (w, h)."""
    sx = dst_size[0] / src_size[0]
    sy = dst_size[1] / src_size[1]
    return np.array([[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5]], dtype=np.float64)


class _Level:
    """0.5x working level shared by the ECC, NCC and phase-correlation stages."""

    def __init__(self, mov: np.ndarray, fix: np.ndarray, mov_n: np.ndarray, fix_n: np.ndarray, valid_fixed: np.ndarray):
        hm, wm = mov.shape[:2]
        hf, wf = fix.shape[:2]
        self.sm = (max(8, int(round(wm * ECC_LEVEL_SCALE))), max(8, int(round(hm * ECC_LEVEL_SCALE))))
        self.sf = (max(8, int(round(wf * ECC_LEVEL_SCALE))), max(8, int(round(hf * ECC_LEVEL_SCALE))))
        self.Dm = _level_map((wm, hm), self.sm)
        self.Df = _level_map((wf, hf), self.sf)
        self.mov = cv2.resize(mov, self.sm, interpolation=cv2.INTER_AREA)
        self.fix = cv2.resize(fix, self.sf, interpolation=cv2.INTER_AREA).astype(np.float32)
        self.mov_n = cv2.resize(mov_n, self.sm, interpolation=cv2.INTER_AREA).astype(np.float32)
        self.fix_n = cv2.resize(fix_n, self.sf, interpolation=cv2.INTER_AREA).astype(np.float32)
        self.valid = cv2.erode(cv2.resize(valid_fixed, self.sf, interpolation=cv2.INTER_NEAREST), np.ones((3, 3), np.uint8))
        self.ones_m = np.full(self.mov.shape[:2], 255, np.uint8)

    def to_level(self, M: np.ndarray) -> np.ndarray:
        """Full-res moving->fixed affine expressed between the level grids."""
        return compose_affine(self.Df, M, invert_affine(self.Dm))

    def to_full(self, A: np.ndarray) -> np.ndarray:
        """Level moving->fixed affine expressed between the full-res grids."""
        return compose_affine(invert_affine(self.Df), A, self.Dm)

    def ncc(self, M: np.ndarray) -> float:
        """NCC of the warped moving gray vs. the fixed gray over the jointly valid area."""
        A = self.to_level(M)
        w, h = self.sf
        warped = cv2.warpAffine(self.mov, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cover = cv2.warpAffine(self.ones_m, A, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        m = (cover > 0) & (self.valid > 0)
        if int(np.count_nonzero(m)) < 64:
            return 0.0
        a = warped[m].astype(np.float64)
        b = self.fix[m].astype(np.float64)
        a -= a.mean()
        b -= b.mean()
        den = math.sqrt(float(a @ a) * float(b @ b))
        return float(a @ b / den) if den > 1e-9 else 0.0

    def ecc(self, M_init: np.ndarray, motion: int) -> np.ndarray | None:
        """ECC refinement on this level; ``M_init`` and the result map moving -> fixed (full res)."""
        W = invert_affine(self.to_level(M_init)).astype(np.float32)  # ECC warp maps fixed -> moving
        try:
            _, W = cv2.findTransformECCWithMask(self.fix_n, self.mov_n, self.valid, self.ones_m, W, motion, ECC_CRITERIA, 5)
        except cv2.error:
            return None
        if not np.all(np.isfinite(W)):
            return None
        return self.to_full(invert_affine(W.astype(np.float64)))

    def _fixed_for_phase(self, h: int, w: int) -> np.ndarray:
        a = self.fix_n[:h, :w]
        vf = self.valid[:h, :w] > 0
        if vf.any() and not vf.all():
            a = np.where(vf, a, float(a[vf].mean())).astype(np.float32)
        return np.ascontiguousarray(a)

    def phase(self) -> np.ndarray | None:
        """Translation-only estimate via phase correlation (Hanning window)."""
        h = min(self.mov_n.shape[0], self.fix_n.shape[0])
        w = min(self.mov_n.shape[1], self.fix_n.shape[1])
        if h < 16 or w < 16:
            return None
        a = self._fixed_for_phase(h, w)
        b = self.mov_n[:h, :w]
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        (sx, sy), _resp = cv2.phaseCorrelate(a, np.ascontiguousarray(b), win)
        if not (np.isfinite(sx) and np.isfinite(sy)):
            return None
        # phaseCorrelate returns the shift s of b (moving) relative to a (fixed): b(x) ~ a(x - s)
        return self.to_full(np.array([[1.0, 0.0, -sx], [0.0, 1.0, -sy]], dtype=np.float64))

    def phase_rot(self, size_m: tuple[int, int], angles: tuple[float, ...] | np.ndarray | None = None) -> tuple[np.ndarray, float] | None:
        """Rotation-grid phase correlation: ``(M, ncc)`` of the best angle (None for degenerate input).

        For every angle ``a`` of the grid the moving level is rotated about its centre (reflect border),
        phase-correlated against the fixed level and composed back; the candidate with the highest NCC
        wins. Translation-only phase correlation leaves the pose rotation (up to a few degrees) as a
        residual that reaches tens of px at the far corners of a zoomed view while the NCC still looks
        acceptable; the grid removes it (rotation step 0.5 deg -> <= 0.25 deg residual, refined by ECC).
        """
        h = min(self.mov_n.shape[0], self.fix_n.shape[0])
        w = min(self.mov_n.shape[1], self.fix_n.shape[1])
        if h < 16 or w < 16:
            return None
        grid = PHASE_ROT_ANGLES if angles is None else angles
        a = self._fixed_for_phase(h, w)
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        wm, hm = size_m
        centre = ((wm - 1) / 2.0, (hm - 1) / 2.0)
        lw, lh = self.mov_n.shape[1], self.mov_n.shape[0]
        best: tuple[np.ndarray, float] | None = None
        for ang in grid:
            Rm = cv2.getRotationMatrix2D(centre, float(ang), 1.0).astype(np.float64)  # full-res moving -> rotated moving
            Rl = compose_affine(self.Dm, Rm, invert_affine(self.Dm))
            rot = cv2.warpAffine(self.mov_n, Rl, (lw, lh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            (sx, sy), _resp = cv2.phaseCorrelate(a, np.ascontiguousarray(rot[:h, :w]), win)
            if not (np.isfinite(sx) and np.isfinite(sy)):
                continue
            M = compose_affine(self.to_full(np.array([[1.0, 0.0, -sx], [0.0, 1.0, -sy]], dtype=np.float64)), Rm)
            score = self.ncc(M)
            if best is None or score > best[1]:
                best = (M, score)
        return best


_FIXED_CACHE: OrderedDict[bytes, tuple[np.ndarray, np.ndarray | None]] = OrderedDict()
_FIXED_LOCK = threading.Lock()


def _fixed_features(fix_n: np.ndarray, kmask: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """ORB keypoints (as points) + descriptors of the fixed image, cached by content hash."""
    hsh = hashlib.blake2b(digest_size=16)
    hsh.update(np.asarray(fix_n.shape, np.int64).tobytes())
    hsh.update(fix_n.tobytes())
    hsh.update(kmask.tobytes())
    key = hsh.digest()
    with _FIXED_LOCK:
        hit = _FIXED_CACHE.get(key)
        if hit is not None:
            _FIXED_CACHE.move_to_end(key)
            return hit
    kp, des = _orb().detectAndCompute(fix_n, kmask)
    pts = np.float32([k.pt for k in kp]).reshape(-1, 2) if kp else np.zeros((0, 2), np.float32)
    val = (pts, des)
    with _FIXED_LOCK:
        _FIXED_CACHE[key] = val
        while len(_FIXED_CACHE) > _FIXED_CACHE_MAX:
            _FIXED_CACHE.popitem(last=False)
    return val


def _orb_stage(mov_n: np.ndarray, fix_n: np.ndarray, valid_fixed: np.ndarray, cfg: DetectorConfig):
    """ORB + ratio test + RANSAC similarity. Returns (M | None, inliers, ratio, reason)."""
    kmask = cv2.erode(valid_fixed, np.ones((3, 3), np.uint8), iterations=_KEYPOINT_MASK_ERODE_PX)
    if cv2.countNonZero(kmask) < 0.05 * kmask.size:
        kmask = valid_fixed
    kp_m, des_m = _orb().detectAndCompute(mov_n, None)
    pts_f, des_f = _fixed_features(fix_n, kmask)
    if des_m is None or des_f is None or len(kp_m) < 4 or len(pts_f) < 4:
        return None, 0, 0.0, "too few keypoints"
    knn = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(des_m, des_f, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < RATIO_TEST * p[1].distance]
    if len(good) < max(4, cfg.min_inliers // 2):
        return None, 0, 0.0, f"too few matches ({len(good)})"
    src = np.float32([kp_m[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    dst = pts_f[np.fromiter((g.trainIdx for g in good), dtype=np.int64, count=len(good))].reshape(-1, 1, 2)
    M, inl = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD_PX,
        maxIters=3000,
        confidence=0.995,
        refineIters=10,
    )
    if M is None or inl is None:
        return None, 0, 0.0, "RANSAC failed"
    n_in = int(inl.sum())
    ratio = n_in / float(len(good))
    _, _, rot, scale = decompose_affine(M)
    if n_in < cfg.min_inliers or ratio < cfg.min_inlier_ratio:
        return None, n_in, ratio, f"weak geometry (inliers={n_in}, ratio={ratio:.2f})"
    if abs(rot) > MAX_ROT_DEG or not (SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]):
        return None, n_in, ratio, f"implausible transform (rot={rot:.1f}, scale={scale:.3f})"
    return M.astype(np.float64), n_in, ratio, "ok"


def _motion_for(M: np.ndarray) -> int:
    _, _, _, s = decompose_affine(M)
    return cv2.MOTION_EUCLIDEAN if abs(s - 1.0) < 0.01 else cv2.MOTION_AFFINE


def _plausible(M: np.ndarray) -> bool:
    _, _, rot, s = decompose_affine(M)
    return abs(rot) <= MAX_ROT_DEG and SCALE_RANGE[0] <= s <= SCALE_RANGE[1]


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------
def register(
    moving_gray: np.ndarray,
    fixed_gray: np.ndarray,
    valid_fixed: np.ndarray | None = None,
    cfg: DetectorConfig | None = None,
) -> RegistrationResult:
    """Register ``moving_gray`` onto ``fixed_gray``; ``M`` maps moving px -> fixed px.

    ``valid_fixed`` (optional, uint8/bool, fixed size) marks the fixed pixels that carry
    real content (keypoints, ECC and NCC are restricted to it). Inputs may be gray or BGR,
    uint8 or float. Never raises for degenerate content: returns ``ok=False`` instead.
    """
    cfg = cfg or DetectorConfig()
    mov = _as_gray_u8(moving_gray)
    fix = _as_gray_u8(fixed_gray)
    vf = _mask_u8(valid_fixed, fix.shape[:2])
    clahe = _clahe()
    mov_n = clahe.apply(mov)
    fix_n = clahe.apply(fix)
    lv = _Level(mov, fix, mov_n, fix_n, vf)
    size_m = (mov.shape[1], mov.shape[0])

    def result(M: np.ndarray, method: str, ncc: float, n_in: int, ratio: float) -> RegistrationResult:
        dx, dy, rot, s = decompose_affine(M, size_m)
        return RegistrationResult(
            ok=bool(ncc >= cfg.min_ncc),
            method=method,
            M=np.asarray(M, dtype=np.float64).reshape(2, 3),
            inliers=int(n_in),
            inlier_ratio=float(ratio),
            ncc=float(ncc),
            dx=dx,
            dy=dy,
            rot_deg=rot,
            scale=s,
        )

    candidates: list[tuple[np.ndarray, str, float]] = []
    corners = np.array([[0, 0], [size_m[0] - 1, 0], [0, size_m[1] - 1], [size_m[0] - 1, size_m[1] - 1]], float)

    def drift_px(A: np.ndarray, B: np.ndarray) -> float:
        return float(np.max(np.linalg.norm(warp_points(A, corners) - warp_points(B, corners), axis=1)))

    # 1) ORB + RANSAC (+ ECC refinement)
    M_orb, n_in, ratio, _reason = _orb_stage(mov_n, fix_n, vf, cfg)
    if M_orb is not None:
        ncc_orb = lv.ncc(M_orb)
        best = (M_orb, "orb", ncc_orb)
        motions = [_motion_for(M_orb)]
        if motions[0] != cv2.MOTION_EUCLIDEAN:
            motions.append(cv2.MOTION_EUCLIDEAN)  # rigid is the physical prior for a fixed camera
        for motion in motions:
            M_ref = lv.ecc(M_orb, motion)
            if M_ref is None or not _plausible(M_ref):
                continue
            # reject refinements that drift far away from the feature-based solution
            if drift_px(M_ref, M_orb) <= 4.0 * RANSAC_THRESHOLD_PX:
                ncc_ref = lv.ncc(M_ref)
                if ncc_ref >= best[2] - 1e-3:
                    best = (M_ref, "orb+ecc", ncc_ref)
        strong = n_in >= STRONG_ORB_FACTOR * cfg.min_inliers
        if best[2] >= cfg.min_ncc and strong:
            return result(best[0], best[1], best[2], n_in, ratio)
        # a near-threshold feature solution can be a wrong local fit: cross-check it against the
        # alignment-based candidates below and keep whichever explains the image best (max NCC)
        candidates.append(best)

    # 2) ECC alone from the identity (kept as a candidate; never trusted without the phase cross-check)
    ident = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    M_ecc = lv.ecc(ident, cv2.MOTION_EUCLIDEAN)
    if M_ecc is not None and _plausible(M_ecc):
        candidates.append((M_ecc, "ecc", lv.ncc(M_ecc)))

    # 3) rotation-grid phase correlation, refined by ECC only when the refinement stays close and helps
    pr = lv.phase_rot(size_m)
    if pr is not None:
        M_ph, ncc_ph = pr
        candidates.append((M_ph, "phase", ncc_ph))
        M_phr = lv.ecc(M_ph, cv2.MOTION_EUCLIDEAN)
        if M_phr is not None and _plausible(M_phr) and drift_px(M_phr, M_ph) <= FALLBACK_ECC_MAX_DRIFT_PX:
            ncc_phr = lv.ncc(M_phr)
            if ncc_phr >= ncc_ph - 1e-3:
                candidates.append((M_phr, "phase+ecc", ncc_phr))

    if not candidates:
        return result(ident, "identity", lv.ncc(ident), n_in, ratio)
    M_best, method, ncc_best = max(candidates, key=lambda c: c[2])
    return result(M_best, method, ncc_best, n_in, ratio)
