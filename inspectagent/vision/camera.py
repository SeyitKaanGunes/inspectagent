"""Active camera simulator — the physical "world" the agent acts on.

A VisA image (or a synthetic board) is the physical part lying on the stage. The virtual
camera has a fixed sensor resolution and a controllable view (pan ``cx, cy`` + optical
``zoom``), exposure compensation (digital gain), light mode (``ring`` / ``diffuse`` /
``low_angle``) and autofocus. Hidden per-part :class:`NuisanceConditions` degrade the
captures; the agent never sees them and must infer them from images.

Image formation (SPEC §2.3), float32 in linear light:

1. pose jitter (fixed per camera instance, precomputed): rotation about the scene centre
   + translation, ``BORDER_REPLICATE``;
2. crop the commanded view window and resample it to the sensor grid (``INTER_AREA`` when
   downscaling, ``INTER_CUBIC`` when upscaling) — geometry follows
   :meth:`Capture.scene_to_capture` exactly (``warpAffine`` pixel convention);
3. sRGB → linear, × light gain × ``2**ambient_ev``; low-angle relief shading; ring-light
   specular glare blobs (fixed in scene/stage coordinates, consistent across views);
4. digital gain ``2**exposure_ev``;
5. defocus blur (``defocus_sigma``, or the 0.35 px autofocus residual);
6. signal-dependent noise, RNG seeded by ``(seed, index)``;
7. clip, linear → sRGB, uint8, JPEG round-trip.

The camera interface (``capture(settings)``, ``estimate_time_ms``) is what a real
PTZ/industrial camera driver would implement; only this module is simulated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

from inspectagent.config import CameraConfig
from inspectagent.types import Capture, CaptureSettings, NuisanceConditions, View

__all__ = [
    "PRESETS",
    "LIGHT_GAIN",
    "GlareBlob",
    "VirtualCamera",
    "make_conditions",
    "srgb_to_linear",
    "linear_to_srgb",
]

PRESETS = ("clean", "nominal", "dark", "bright", "defocus", "glare", "harsh")
LIGHT_GAIN = {"ring": 1.0, "diffuse": 0.6, "low_angle": 0.5}

GAMMA = 2.2
AF_RESIDUAL_SIGMA = 0.35  # px, residual blur after autofocus
GLARE_GAIN = 2.5  # linear intensity of a glare plateau per unit glare_strength
GLARE_HALO_WIDTH = 0.5  # Gaussian halo width beyond the plateau, in units of the blob radius
GLARE_EXTENT = 1.0 + 3.0 * GLARE_HALO_WIDTH  # normalized radius beyond which glare is negligible
RELIEF_GAIN = 0.6
RELIEF_CLIP = 0.6
SHOT_NOISE_COEF = 0.012
_MIN_BLUR_SIGMA = 0.05
_GLARE_GRID = 4  # coarse-grid step (sensor px) for evaluating glare profiles

_LUT_SRGB_TO_LIN = ((np.arange(256, dtype=np.float64) / 255.0) ** GAMMA).astype(np.float32)


def srgb_to_linear(img_u8: np.ndarray) -> np.ndarray:
    """uint8 sRGB-ish (gamma 2.2) image → float32 linear light in [0, 1]."""
    return cv2.LUT(np.ascontiguousarray(img_u8, dtype=np.uint8), _LUT_SRGB_TO_LIN)


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    """float linear light (any range; clipped to [0, 1]) → uint8 gamma-2.2 image."""
    # clamp to a tiny positive floor: negatives would give NaN and exact zeros hit a ~6x slower
    # path in cv2.pow; 1e-7 maps to 0.17/255 and still rounds to 0
    x = np.maximum(np.asarray(lin, dtype=np.float32), np.float32(1e-7))
    x = cv2.pow(x, 1.0 / GAMMA)
    return cv2.convertScaleAbs(x, alpha=255.0)  # rounds; values > 1 saturate at 255


@dataclass(frozen=True)
class GlareBlob:
    """One specular highlight, in (jittered) scene/stage pixel coordinates.

    The profile is a saturating elliptical plateau (normalized radius ``d <= 1``) with a
    smooth Gaussian halo ``exp(-((d - 1) / GLARE_HALO_WIDTH)**2)`` outside it.
    """

    cx: float
    cy: float
    a: float  # semi-major axis of the plateau (scene px)
    b: float  # semi-minor axis of the plateau (scene px)
    theta: float  # orientation of the major axis (radians)

    @property
    def radius(self) -> float:
        """Equivalent (area-preserving) plateau radius in scene px."""
        return math.sqrt(self.a * self.b)

    def to_dict(self) -> dict:
        return {"cx": self.cx, "cy": self.cy, "a": self.a, "b": self.b, "theta": self.theta}


def _make_glare_blobs(seed: int, scene_size: tuple[int, int]) -> tuple[GlareBlob, ...]:
    """1–3 blobs, equivalent radius 3–8 % of the scene diagonal, fully inside the scene."""
    w, h = scene_size
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    diag = math.hypot(w, h)
    blobs = []
    for _ in range(int(rng.integers(1, 4))):
        r = float(rng.uniform(0.03, 0.08)) * diag
        elong = float(rng.uniform(1.0, 1.8))
        a, b = r * math.sqrt(elong), r / math.sqrt(elong)
        theta = float(rng.uniform(0.0, math.pi))
        # half extents of the rotated halo ellipse (so the whole blob stays inside the scene)
        ea, eb = a * GLARE_EXTENT, b * GLARE_EXTENT
        hx = math.sqrt((ea * math.cos(theta)) ** 2 + (eb * math.sin(theta)) ** 2)
        hy = math.sqrt((ea * math.sin(theta)) ** 2 + (eb * math.cos(theta)) ** 2)
        lo_x, hi_x = (hx, w - hx) if w > 2 * hx else (w / 2, w / 2)
        lo_y, hi_y = (hy, h - hy) if h > 2 * hy else (h / 2, h / 2)
        cx = float(rng.uniform(lo_x, hi_x))
        cy = float(rng.uniform(lo_y, hi_y))
        blobs.append(GlareBlob(cx, cy, a, b, theta))
    return tuple(blobs)


def _uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def make_conditions(preset: str, rng: np.random.Generator | int | None = None) -> NuisanceConditions:
    """Sample hidden nuisance conditions for a preset (deterministic for a given RNG state).

    ``clean``: no degradation (no read noise, no pose jitter). ``nominal``: small random
    variation. ``dark``/``bright``/``defocus``/``glare``: one strong degradation on top of
    nominal-level variation. ``harsh``: 1–2 strong degradations combined at random (never
    dark + bright). Non-clean presets other than ``nominal`` use pose jitter up to 2.5°
    rotation and 2.5 % shift.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; expected one of {PRESETS}")
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    glare_seed = int(rng.integers(0, 2**31 - 1))
    if preset == "clean":
        return NuisanceConditions(
            ambient_ev=0.0,
            defocus_sigma=0.0,
            noise_sigma=0.0,
            glare_strength=0.0,
            glare_seed=glare_seed,
            rot_deg=0.0,
            shift_x=0.0,
            shift_y=0.0,
            jpeg_quality=95,
            preset="clean",
        )

    pose_rot, pose_shift = (1.5, 0.015) if preset == "nominal" else (2.5, 0.025)
    rot = _uniform(rng, -pose_rot, pose_rot)
    shift_x = _uniform(rng, -pose_shift, pose_shift)
    shift_y = _uniform(rng, -pose_shift, pose_shift)
    ambient = _uniform(rng, -0.3, 0.3)
    defocus = _uniform(rng, 0.0, 0.6)
    noise = _uniform(rng, 1.0, 2.0)
    jpeg = int(rng.integers(88, 96))
    glare_draw = float(rng.random())
    glare_small = _uniform(rng, 0.05, 0.2)
    glare = glare_small if (preset == "nominal" and glare_draw >= 0.8) else 0.0

    strong: tuple[str, ...]
    if preset == "harsh":
        options = ("dark", "bright", "defocus", "glare")
        k = int(rng.integers(1, 3))
        while True:
            pick = tuple(sorted(rng.choice(len(options), size=k, replace=False).tolist()))
            strong = tuple(options[i] for i in pick)
            if not ("dark" in strong and "bright" in strong):
                break
    elif preset == "nominal":
        strong = ()
    else:
        strong = (preset,)

    for kind in strong:
        if kind == "dark":
            ambient = _uniform(rng, -2.4, -1.6)
        elif kind == "bright":
            ambient = _uniform(rng, 1.2, 1.8)
        elif kind == "defocus":
            defocus = _uniform(rng, 2.5, 4.5)
        elif kind == "glare":
            glare = _uniform(rng, 0.5, 0.9)

    return NuisanceConditions(
        ambient_ev=ambient,
        defocus_sigma=defocus,
        noise_sigma=noise,
        glare_strength=glare,
        glare_seed=glare_seed,
        rot_deg=rot,
        shift_x=shift_x,
        shift_y=shift_y,
        jpeg_quality=jpeg,
        preset=preset,
    )


def _as_bgr_u8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr if arr.dtype == np.uint8 else np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"scene must be a BGR (H, W, 3) image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        if a.size and float(a.max()) <= 1.0:
            a *= 255.0
        arr = np.clip(a + 0.5, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


class VirtualCamera:
    """Deterministic active-camera simulator over a fixed scene image.

    Attributes: ``scene_size`` = (W_s, H_s); ``sensor_size`` = (W, H) with the long side
    equal to ``cfg.sensor_long_side`` and the scene aspect ratio; ``pose_matrix`` = 2x3
    affine scene px → stage px (the hidden pose jitter).
    """

    def __init__(
        self,
        scene_bgr: np.ndarray,
        conditions: NuisanceConditions,
        cfg: CameraConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg or CameraConfig()
        self.conditions = conditions
        self.seed = int(seed)
        scene = _as_bgr_u8(scene_bgr)
        h_s, w_s = scene.shape[:2]
        if w_s < 2 or h_s < 2:
            raise ValueError("scene too small")
        self.scene_size: tuple[int, int] = (int(w_s), int(h_s))
        long_side = int(self.cfg.sensor_long_side)
        if w_s >= h_s:
            self.sensor_size: tuple[int, int] = (long_side, max(1, int(round(long_side * h_s / w_s))))
        else:
            self.sensor_size = (max(1, int(round(long_side * w_s / h_s))), long_side)

        # 1. pose jitter, precomputed once
        c = conditions
        m = cv2.getRotationMatrix2D(((w_s - 1) / 2.0, (h_s - 1) / 2.0), float(c.rot_deg), 1.0)
        m[0, 2] += float(c.shift_x) * w_s
        m[1, 2] += float(c.shift_y) * h_s
        self.pose_matrix: np.ndarray = m.astype(np.float64)
        if abs(c.rot_deg) < 1e-9 and abs(c.shift_x) < 1e-12 and abs(c.shift_y) < 1e-12:
            self._stage = scene.copy()
        else:
            self._stage = cv2.warpAffine(
                scene, m, (w_s, h_s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
        self._relief: np.ndarray | None = None  # lazily computed low-angle relief map (stage frame)
        self._glare_cache: tuple[int, tuple[GlareBlob, ...]] | None = None
        self._last_settings: CaptureSettings | None = None

    @property
    def glare_blobs(self) -> tuple[GlareBlob, ...]:
        """Specular blobs (stage px) for the current ``conditions.glare_seed`` (empty if no glare).

        Computed lazily so that conditions edited after construction (tests, what-if tools)
        are honoured; the pose jitter, in contrast, is fixed at construction.
        """
        if self.conditions.glare_strength <= 0:
            return ()
        seed = int(self.conditions.glare_seed)
        if self._glare_cache is None or self._glare_cache[0] != seed:
            self._glare_cache = (seed, _make_glare_blobs(seed, self.scene_size))
        return self._glare_cache[1]

    # ------------------------------------------------------------------ geometry
    def _clamp_view(self, view: View) -> View:
        zoom = float(min(max(float(view.zoom), 1.0), float(self.cfg.max_zoom)))
        cx = float(min(max(float(view.cx), 0.0), 1.0))
        cy = float(min(max(float(view.cy), 0.0), 1.0))
        return View(cx=cx, cy=cy, zoom=zoom)

    def view_window(self, view: View) -> tuple[float, float, float, float]:
        """``(x0, y0, w, h)`` of the view window in scene px, shifted (not shrunk) to stay inside."""
        v = self._clamp_view(view)
        w_s, h_s = self.scene_size
        ww, wh = w_s / v.zoom, h_s / v.zoom
        x0 = min(max(v.cx * w_s - ww / 2.0, 0.0), w_s - ww)
        y0 = min(max(v.cy * h_s - wh / 2.0, 0.0), h_s - wh)
        return (float(x0), float(y0), float(ww), float(wh))

    def window_center_norm(self, view: View) -> tuple[float, float]:
        """Normalized centre of the *effective* (clamped) view window."""
        x0, y0, ww, wh = self.view_window(view)
        return ((x0 + ww / 2.0) / self.scene_size[0], (y0 + wh / 2.0) / self.scene_size[1])

    def true_scene_to_capture(self, capture: Capture) -> np.ndarray:
        """2x3 affine scene px → capture px *including* the hidden pose jitter (ground truth)."""
        a = np.vstack([capture.scene_to_capture(), [0.0, 0.0, 1.0]])
        p = np.vstack([self.pose_matrix, [0.0, 0.0, 1.0]])
        return (a @ p)[:2]

    def _resample(self, src: np.ndarray, window: tuple[float, float, float, float], upscale_interp: int) -> np.ndarray:
        """Resample the stage-frame window onto the sensor grid.

        Pixel convention matches ``cv2.warpAffine`` with ``Capture.scene_to_capture()``:
        capture px ``u`` samples stage px ``u / sx + x0``.
        """
        x0, y0, ww, wh = window
        sw, sh = self.sensor_size
        sx, sy = sw / ww, sh / wh
        flags_inv = cv2.WARP_INVERSE_MAP
        if sx >= 1.0 - 1e-9 and sy >= 1.0 - 1e-9:
            m = np.array([[1.0 / sx, 0.0, x0], [0.0, 1.0 / sy, y0]], dtype=np.float64)
            return cv2.warpAffine(src, m, (sw, sh), flags=upscale_interp | flags_inv, borderMode=cv2.BORDER_REPLICATE)
        # downscale: integer crop (+margin) -> INTER_AREA resize -> exact sub-pixel placement
        h_src, w_src = src.shape[:2]
        ix0 = max(0, int(math.floor(x0)) - 1)
        iy0 = max(0, int(math.floor(y0)) - 1)
        ix1 = min(w_src, int(math.ceil(x0 + ww)) + 2)
        iy1 = min(h_src, int(math.ceil(y0 + wh)) + 2)
        crop = src[iy0:iy1, ix0:ix1]
        cw, ch = ix1 - ix0, iy1 - iy0
        rw, rh = max(1, int(round(cw * sx))), max(1, int(round(ch * sy)))
        small = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_AREA)
        srx, sry = rw / cw, rh / ch
        # cv2.resize maps crop px c -> small px (c + 0.5) * s - 0.5
        m = np.array(
            [
                [srx / sx, 0.0, (x0 - ix0 + 0.5) * srx - 0.5],
                [0.0, sry / sy, (y0 - iy0 + 0.5) * sry - 0.5],
            ],
            dtype=np.float64,
        )
        return cv2.warpAffine(small, m, (sw, sh), flags=cv2.INTER_LINEAR | flags_inv, borderMode=cv2.BORDER_REPLICATE)

    # ------------------------------------------------------------------ photometry helpers
    def _relief_map(self) -> np.ndarray:
        """Normalized horizontal relief (Sobel-x of the stage gray), clipped to ±RELIEF_CLIP."""
        if self._relief is None:
            gray = cv2.cvtColor(self._stage, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0)
            scale = float(np.percentile(np.abs(gx), 99.0))
            scale = max(scale, 1e-4)
            self._relief = np.clip(gx / scale, -RELIEF_CLIP, RELIEF_CLIP).astype(np.float32)
        return self._relief

    def _add_glare(self, lin: np.ndarray, window: tuple[float, float, float, float], strength: float) -> None:
        """Add the ring-light specular blobs (stage coordinates) to ``lin`` in place.

        The smooth profile is evaluated on a coarse grid (every ~4 sensor px) inside each
        blob's bounding box and bilinearly upsampled — exact at the grid nodes, and the halo
        is >= 10 px wide, so the interpolation error is negligible.
        """
        if strength <= 0 or not self.glare_blobs:
            return
        x0, y0, ww, wh = window
        sw, sh = self.sensor_size
        sx, sy = sw / ww, sh / wh
        peak = float(strength) * GLARE_GAIN
        for blob in self.glare_blobs:
            ea, eb = blob.a * GLARE_EXTENT, blob.b * GLARE_EXTENT
            ct, st = math.cos(blob.theta), math.sin(blob.theta)
            hx = math.sqrt((ea * ct) ** 2 + (eb * st) ** 2)
            hy = math.sqrt((ea * st) ** 2 + (eb * ct) ** 2)
            u0 = max(0, int(math.floor((blob.cx - hx - x0) * sx)))
            u1 = min(sw, int(math.ceil((blob.cx + hx - x0) * sx)) + 1)
            v0 = max(0, int(math.floor((blob.cy - hy - y0) * sy)))
            v1 = min(sh, int(math.ceil((blob.cy + hy - y0) * sy)) + 1)
            nu, nv = u1 - u0, v1 - v0
            if nu <= 0 or nv <= 0:
                continue
            gu = nu if nu < 16 else max(4, nu // _GLARE_GRID)
            gv = nv if nv < 16 else max(4, nv // _GLARE_GRID)
            fu, fv = nu / gu, nv / gv
            # coarse node j sits at bbox px (j + 0.5) * f - 0.5 (cv2.resize convention)
            us = u0 + (np.arange(gu, dtype=np.float64) + 0.5) * fu - 0.5
            vs = v0 + (np.arange(gv, dtype=np.float64) + 0.5) * fv - 0.5
            dx = (us / sx + (x0 - blob.cx)).astype(np.float32)[None, :]
            dy = (vs / sy + (y0 - blob.cy)).astype(np.float32)[:, None]
            p = (dx * ct + dy * st) / blob.a
            q = (dy * ct - dx * st) / blob.b
            excess = np.maximum(np.sqrt(p * p + q * q) - 1.0, 0.0) / GLARE_HALO_WIDTH
            g = (np.exp(-(excess * excess)) * peak).astype(np.float32)
            if (gu, gv) != (nu, nv):
                g = cv2.resize(g, (nu, nv), interpolation=cv2.INTER_LINEAR)
            lin[v0:v1, u0:u1] += g[..., None]

    # ------------------------------------------------------------------ capture
    def _effective_settings(self, settings: CaptureSettings) -> CaptureSettings:
        if settings.light not in LIGHT_GAIN:
            raise ValueError(f"unknown light mode {settings.light!r}; expected one of {tuple(LIGHT_GAIN)}")
        ev = float(min(max(float(settings.exposure_ev), self.cfg.min_ev), self.cfg.max_ev))
        return replace(settings, view=self._clamp_view(settings.view), exposure_ev=ev, autofocus=bool(settings.autofocus))

    def _noise_seed(self, index: int) -> int:
        """31-bit noise seed (C int) derived from ``(seed, index)`` (stable across platforms)."""
        ss = np.random.SeedSequence([self.seed & 0xFFFFFFFF, int(index) & 0xFFFFFFFF, 0x1A5E])
        return (int(ss.generate_state(1, dtype=np.uint32)[0]) & 0x7FFFFFFF) or 1

    def capture(self, settings: CaptureSettings, capture_id: str, index: int) -> Capture:
        """Take a (degraded) picture with the given settings; deterministic in ``(seed, index)``."""
        s = self._effective_settings(settings)
        c = self.conditions
        window = self.view_window(s.view)
        sim_ms = self.estimate_time_ms(self._last_settings, s)

        img8 = self._resample(self._stage, window, cv2.INTER_CUBIC)

        # 3. illumination (global light gain folded into the sRGB->linear LUT)
        gain_light = LIGHT_GAIN[s.light] * (2.0 ** float(c.ambient_ev))
        lin = cv2.LUT(img8, (_LUT_SRGB_TO_LIN * np.float32(gain_light)).astype(np.float32))
        if s.light == "low_angle":
            relief = self._resample(self._relief_map(), window, cv2.INTER_LINEAR)
            shade = cv2.addWeighted(relief, RELIEF_GAIN, relief, 0.0, 1.0)  # 1 + 0.6 * relief
            lin = cv2.multiply(lin, cv2.merge([shade, shade, shade]))
        if s.light == "ring" and c.glare_strength > 0:
            self._add_glare(lin, window, float(c.glare_strength))

        # 5. defocus (optical blur; commutes with the linear digital gain of step 4)
        sigma = AF_RESIDUAL_SIGMA if s.autofocus else float(c.defocus_sigma)
        if sigma > _MIN_BLUR_SIGMA:
            lin = cv2.GaussianBlur(lin, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)

        # 4 + 6. digital gain and noise: std = gain * (read/255 + k * sqrt(signal_before_gain))
        gain = 2.0 ** s.exposure_ev
        std = cv2.addWeighted(cv2.sqrt(lin), gain * SHOT_NOISE_COEF, lin, 0.0, gain * float(c.noise_sigma) / 255.0)
        noise = np.empty(lin.shape, dtype=np.float32)
        cv2.setRNGSeed(self._noise_seed(index))  # thread-local OpenCV RNG -> deterministic per (seed, index)
        cv2.randn(noise.reshape(lin.shape[0], -1), 0.0, 1.0)
        out = cv2.scaleAdd(lin, gain, cv2.multiply(noise, std))

        # 7. sensor output: clip, gamma, 8-bit, JPEG
        img = linear_to_srgb(out)
        q = int(min(100, max(1, c.jpeg_quality)))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if dec is not None:
                img = dec

        self._last_settings = s
        return Capture(
            capture_id=capture_id,
            index=int(index),
            settings=s,
            image=img,
            window=window,
            scene_size=self.scene_size,
            sim_time_ms=float(sim_ms),
            oracle=False,
        )

    def clean_capture(self, view: View, capture_id: str, index: int) -> Capture:
        """Oracle capture: same pose jitter and view geometry, no photometric/blur/noise/glare/JPEG.

        Does not change the camera's motion state (oracle peeks are not physical moves).
        """
        s = self._effective_settings(CaptureSettings(view=view))
        window = self.view_window(s.view)
        img = self._resample(self._stage, window, cv2.INTER_CUBIC)
        return Capture(
            capture_id=capture_id,
            index=int(index),
            settings=s,
            image=img,
            window=window,
            scene_size=self.scene_size,
            sim_time_ms=float(self.estimate_time_ms(self._last_settings, s)),
            oracle=True,
        )

    # ------------------------------------------------------------------ timing
    def estimate_time_ms(self, prev: CaptureSettings | None, settings: CaptureSettings) -> float:
        """Simulated wall-clock cost of taking ``settings`` after ``prev`` (``None`` = home pose).

        Gantry motion (distance between effective view centres), zoom travel and a light
        switch run concurrently; autofocus runs after positioning; then exposure + readout
        (EV is a digital gain, so integration time is constant).
        """
        cfg = self.cfg
        p = prev if prev is not None else CaptureSettings()
        c0 = self.window_center_norm(p.view)
        c1 = self.window_center_norm(settings.view)
        dist = math.hypot(c1[0] - c0[0], c1[1] - c0[1])
        z0 = self._clamp_view(p.view).zoom
        z1 = self._clamp_view(settings.view).zoom
        t_move = cfg.t_move_per_unit * dist
        t_zoom = cfg.t_zoom_per_step * abs(z1 - z0)
        t_light = cfg.t_light_switch if settings.light != p.light else 0.0
        t_af = cfg.t_autofocus if settings.autofocus else 0.0
        return float(max(t_move, t_zoom, t_light) + t_af + cfg.t_exposure + cfg.t_readout)

    def reset_motion(self) -> None:
        """Forget the last pose (next capture is timed from the home pose)."""
        self._last_settings = None
