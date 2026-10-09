"""Fit and calibrate the per-category detector bundles under ``models/<category>/``.

Usage (after ``scripts/prepare_visa.py``; deep model needs ``models/backbone_resnet18.onnx``
from ``scripts/export_backbone.py``)::

    python scripts/build_models.py                         # all categories, parallel
    python scripts/build_models.py --categories pcb1 --no-deep --max-fit 100

Per category (one process per category):

1. **Fast detector** - ``StatTemplateModel.fit`` on up to ``--max-fit`` (default 300)
   normals of the ``fit`` split (deterministic seeded subset).
2. **Reference sharpness and exposure** per zoom bucket - median ``quality.laplacian_sharpness`` and
   median ``quality.exposure_stats`` (glare-excluded linear mean, clipped / crushed fractions) of oracle
   clean captures (``VirtualCamera.clean_capture``) of ``val_normal`` parts at zoom 1 and 2/3/4; the
   exposure reference lets ``assess_quality`` judge a part against its own normal appearance (bright
   parts on dark backgrounds) instead of whole-frame limits.
3. **Normal-score calibration** - ``val_normal`` parts rendered by ``VirtualCamera`` under the
   ``clean`` and ``nominal`` presets: the full view (zoom 1) plus ``--zoom-views`` random
   zoomed views at zoom 2, 3 and 4 each (centres drawn half uniformly, half weighted by
   image texture, where candidate regions appear); plus full views of the corrected captures the
   agent accepts (diffuse light at +0.75 EV, and a dimmed scene compensated by +1.5..+2 EV gain) so
   the normal-score spread covers the noise those corrections add. Only captures whose registration
   succeeded are used. Gives per-bucket score ``mu/sd`` and the region threshold =
   99.9th percentile of the pooled normal pixel z-scores of the bucket (captures with
   outlier image scores, > median + 6 MAD, excluded; floor 3.0).
4. **Logistic calibration** on the ``calib`` split: single-shot fast scores of one default
   full-view capture under the ``nominal`` preset.
5. **Deep detector** (unless ``--no-deep`` or the ONNX backbone is missing) -
   ``PatchCoreDNN.fit`` on <= 200 fit normals, calibrated on the zoom-1 ``val_normal``
   captures (normal scores) and the ``calib`` captures (logistic).
6. ``models_io.save_bundle`` + ``models/<category>/meta.json`` (counts, timings, OpenCV
   version, calibration values, calibration-split AUROC diagnostics) + ``build.log``.

All randomness is seeded from ``--seed`` and the part ids (``eval_utils.stable_seed``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inspectagent.config import CATEGORIES, get_settings  # noqa: E402
from inspectagent.eval_utils import (  # noqa: E402
    auroc,
    describe,
    nan_to_none,
    part_id_from_relpath,
    split_fraction,
    stable_seed,
    stratified_subsample,
    zoom_bucket,
)

BUCKETS = ("z1", "z2", "z3", "z4")
CALIB_ZOOMS = (2.0, 3.0, 4.0)
REGION_PERCENTILE = 99.9
REGION_THR_MIN = 3.0  # same floor as StatTemplateModel (z units)
PIXELS_PER_CAPTURE = 20000
SHARPNESS_PARTS = 24
# normal-score calibration regimes: the operating envelope of the agent (ring EV 0 under clean / nominal
# conditions, plus the corrected captures it accepts: diffuse +0.75 EV after glare, gain-compensated dark scenes)
CALIB_REGIMES = ("clean", "nominal", "diffuse", "dark_comp")
CALIB_REGIME_PRESET = {"clean": "clean", "nominal": "nominal", "diffuse": "nominal", "dark_comp": "nominal"}


# --------------------------------------------------------------------------- helpers


class _Logger:
    """Prefix + timestamp logger writing to stdout and ``models/<category>/build.log``."""

    def __init__(self, category: str, path: Path) -> None:
        self.category = category
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("w", encoding="utf-8")
        self._t0 = time.perf_counter()

    def __call__(self, msg: str) -> None:
        line = f"[{self.category} +{time.perf_counter() - self._t0:7.1f}s] {msg}"
        print(line, flush=True)
        self._f.write(line + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def _imread(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    from inspectagent.vision.imageio import imread

    return imread(path, flags)


class LazyImages(Sequence):
    """Read-on-access image sequence (keeps hundreds of full-resolution normals out of RAM)."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = list(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):  # type: ignore[override]
        if isinstance(i, slice):
            return LazyImages(self.paths[i])
        return _imread(self.paths[i])


def texture_weights(scene_bgr: np.ndarray, grid: tuple[int, int] = (64, 48)) -> np.ndarray:
    """Probability grid (rows x cols) proportional to local gradient energy of the scene."""
    gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (grid[0] * 4, grid[1] * 4), interpolation=cv2.INTER_AREA).astype(np.float32)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), 3.0)
    cell = cv2.resize(mag, grid, interpolation=cv2.INTER_AREA).astype(np.float64)
    total = cell.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full((grid[1], grid[0]), 1.0 / (grid[0] * grid[1]))
    return cell / total


def sample_views(weights: np.ndarray, rng: np.random.Generator, zooms: tuple[float, ...], per_zoom: int) -> list:
    """Zoomed views: centres half uniform, half texture-weighted, kept inside the scene."""
    from inspectagent.types import View

    rows, cols = weights.shape
    flat = weights.ravel()
    views = []
    for z in zooms:
        for _ in range(per_zoom):
            if rng.random() < 0.5:
                cx, cy = float(rng.random()), float(rng.random())
            else:
                k = int(rng.choice(flat.size, p=flat))
                r, c = divmod(k, cols)
                cx, cy = (c + float(rng.random())) / cols, (r + float(rng.random())) / rows
            half = 0.5 / z
            views.append(View(cx=float(np.clip(cx, half, 1 - half)), cy=float(np.clip(cy, half, 1 - half)), zoom=float(z)))
    return views


def apply_region_thresholds(stat: Any, thr_by_bucket: dict[str, float]) -> None:
    """Store per-bucket region thresholds in ``stat.calib`` (SPEC 2.6: ``calib["region_thr"][bucket]``).

    Uses a dedicated setter when the model provides one; otherwise writes both the
    top-level ``region_thr`` mapping and, when present, the per-bucket entries.
    """
    if not thr_by_bucket:
        return
    setter = getattr(stat, "set_region_thresholds", None)
    if callable(setter):
        setter(dict(thr_by_bucket))
        return
    calib = stat.calib if isinstance(getattr(stat, "calib", None), dict) else {}
    rt = calib.get("region_thr")
    rt = dict(rt) if isinstance(rt, dict) else {}
    rt.update({b: float(v) for b, v in thr_by_bucket.items()})
    calib["region_thr"] = rt
    for container_key in ("buckets", "zoom"):
        cont = calib.get(container_key)
        if isinstance(cont, dict):
            for b, v in thr_by_bucket.items():
                if isinstance(cont.get(b), dict):
                    cont[b]["region_thr"] = float(v)
    for b, v in thr_by_bucket.items():
        if isinstance(calib.get(b), dict):
            calib[b]["region_thr"] = float(v)
    stat.calib = calib


def region_thresholds(
    scores: dict[str, list[float]], zpix: dict[str, list[np.ndarray]], percentile: float = REGION_PERCENTILE
) -> tuple[dict[str, float], dict[str, int]]:
    """Per-bucket region threshold = ``percentile`` of pooled normal pixel z-scores.

    Captures whose image score is a gross outlier (> median + 6 MAD, the same rule
    ``StatTemplateModel.calibrate`` applies to scores - typically residual misregistration)
    are excluded so a handful of bad captures cannot dominate the pooled tail. Thresholds are
    floored at ``REGION_THR_MIN`` (3.0 z) and empty buckets inherit from the nearest one.
    Returns ``(thresholds, excluded captures per bucket)``.
    """
    out: dict[str, float] = {}
    excluded: dict[str, int] = {}
    for b in BUCKETS:
        sc = np.asarray(scores.get(b, []), np.float64)
        px = zpix.get(b, [])
        if sc.size == 0 or not px:
            continue
        keep = np.ones(sc.size, bool)
        if sc.size >= 8:
            med = float(np.median(sc))
            mad = 1.4826 * float(np.median(np.abs(sc - med)))
            keep = sc <= med + 6.0 * max(mad, 1e-6)
        excluded[b] = int((~keep).sum())
        pooled = np.concatenate([p for p, k in zip(px, keep) if k]) if keep.any() else np.concatenate(px)
        out[b] = max(REGION_THR_MIN, float(np.percentile(pooled, percentile)))
    return _fill_missing_buckets(out), excluded


def _fill_missing_buckets(values: dict[str, float]) -> dict[str, float]:
    """Copy the nearest available bucket's value into empty buckets (z1 < z2 < z3 < z4)."""
    out = dict(values)
    have = [b for b in BUCKETS if b in out and math.isfinite(out[b])]
    if not have:
        return out
    for b in BUCKETS:
        if b not in out or not math.isfinite(out[b]):
            i = BUCKETS.index(b)
            nearest = min(have, key=lambda h: abs(BUCKETS.index(h) - i))
            out[b] = out[nearest]
    return out


def _subset(paths: list[str], n: int | None, seed: int, tag: str, category: str) -> list[str]:
    """Deterministic seeded subset of at most ``n`` paths (input order kept)."""
    if n is None or n <= 0 or n >= len(paths):
        return list(paths)
    first, _ = split_fraction(paths, n / len(paths), stable_seed(seed, category, tag))
    return first[:n]


# --------------------------------------------------------------------------- per-category build


def build_category(job: dict) -> dict:
    """Fit + calibrate + save one category bundle; returns a short status dict (runs in a worker)."""
    category: str = job["category"]
    models_dir = Path(job["models_dir"])
    out_dir = models_dir / category
    log = _Logger(category, out_dir / "build.log")
    try:
        return _build(job, log)
    except Exception as exc:  # report, do not kill the other workers
        log(f"FAILED: {exc!r}\n{traceback.format_exc()}")
        return {"category": category, "ok": False, "error": repr(exc)}
    finally:
        log.close()


def _build(job: dict, log: Callable[[str], None]) -> dict:
    from inspectagent.models_io import CategoryBundle, save_bundle
    from inspectagent.types import CaptureSettings, View
    from inspectagent.vision.anomaly_stat import StatTemplateModel
    from inspectagent.vision.camera import VirtualCamera, make_conditions
    from inspectagent.vision.quality import exposure_stats, laplacian_sharpness, reference_exposure_from

    category: str = job["category"]
    seed: int = int(job["seed"])
    threads = int(job.get("threads", 0))
    if threads > 0:
        cv2.setNumThreads(threads)
    settings = get_settings()
    det_cfg = settings.detector
    cam_cfg = settings.camera
    data_root = Path(job["data_root"])
    models_dir = Path(job["models_dir"])
    split = json.loads(Path(job["splits"]).read_text(encoding="utf-8"))[category]
    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    fit_paths = _subset(list(split["fit"]), job["max_fit"], seed, "fit-subset", category)
    val_paths = _subset(list(split["val_normal"]), job.get("max_val"), seed, "val-subset", category)
    calib_items = list(split["calib"])
    if job.get("max_calib"):
        calib_items = stratified_subsample(
            calib_items, [int(d["label"]) for d in calib_items], int(job["max_calib"]), stable_seed(seed, category, "calib-subset")
        )
    log(f"fit={len(fit_paths)}/{len(split['fit'])} val_normal={len(val_paths)} calib={len(calib_items)} threads={threads}")

    # 1. fast detector ---------------------------------------------------------------------------
    normals = LazyImages([data_root / p for p in fit_paths])
    t0 = time.perf_counter()
    stat = StatTemplateModel.fit(category, normals, cfg=det_cfg, log=log)
    timings["stat_fit_s"] = time.perf_counter() - t0
    log(f"stat fit done in {timings['stat_fit_s']:.1f}s")

    # 5a. deep detector fit (before calibration so both share the same captures) -------------------
    onnx_path = Path(job["onnx"])
    deep = None
    deep_status = "disabled (--no-deep)" if job["no_deep"] else "missing backbone"
    n_deep_fit = 0
    if not job["no_deep"] and onnx_path.is_file():
        from inspectagent.vision.anomaly_deep import PatchCoreDNN

        t0 = time.perf_counter()
        deep_imgs = normals[: det_cfg.deep_train_images]
        n_deep_fit = len(deep_imgs)
        deep = PatchCoreDNN.fit(category, deep_imgs, onnx_path, cfg=det_cfg, log=log, seed=seed)
        timings["deep_fit_s"] = time.perf_counter() - t0
        deep_status = "fitted"
        log(f"deep fit done in {timings['deep_fit_s']:.1f}s on {n_deep_fit} normals")
    elif not job["no_deep"]:
        log(f"deep detector skipped: {onnx_path} not found")
    del normals

    # 2+3. reference sharpness and normal-score calibration ------------------------------------------
    t0 = time.perf_counter()
    sharp: dict[str, list[float]] = {b: [] for b in BUCKETS}
    expo: dict[str, list[dict]] = {b: [] for b in BUCKETS}
    scores: dict[str, list[float]] = {b: [] for b in BUCKETS}
    zpix: dict[str, list[np.ndarray]] = {b: [] for b in BUCKETS}
    deep_normal: list[float] = []
    n_caps = n_reg_fail = 0
    for i, rel in enumerate(val_paths):
        pid = part_id_from_relpath(rel)
        scene = _imread(data_root / rel)
        weights = texture_weights(scene)
        if i < SHARPNESS_PARTS:
            cond = make_conditions("clean", np.random.default_rng(stable_seed(seed, pid, "sharp-cond")))
            cam = VirtualCamera(scene, cond, cam_cfg, seed=stable_seed(seed, pid, "sharp-cam"))
            vrng = np.random.default_rng(stable_seed(seed, pid, "sharp-views"))
            views = [View(0.5, 0.5, 1.0)] + sample_views(weights, vrng, CALIB_ZOOMS, 1)
            for k, view in enumerate(views):
                cap = cam.clean_capture(view, f"sharp_{k}", k)
                gray = cv2.cvtColor(cap.image, cv2.COLOR_BGR2GRAY)
                sharp[zoom_bucket(view.zoom)].append(float(laplacian_sharpness(gray)))
                expo[zoom_bucket(view.zoom)].append(exposure_stats(cap.image))
        for preset in CALIB_REGIMES:
            cond = make_conditions(CALIB_REGIME_PRESET[preset], np.random.default_rng(stable_seed(seed, pid, preset, "calib-cond")))
            ev, light = 0.0, "ring"
            if preset == "diffuse":  # the glare remedy: diffuse light with +0.75 EV compensation
                ev, light = 0.75, "diffuse"
            elif preset == "dark_comp":  # dim scene compensated by digital gain (noisier normals)
                cond.ambient_ev = float(np.random.default_rng(stable_seed(seed, pid, "dark-amb")).uniform(-2.0, -1.5))
                ev = -cond.ambient_ev
            cam = VirtualCamera(scene, cond, cam_cfg, seed=stable_seed(seed, pid, preset, "calib-cam"))
            vrng = np.random.default_rng(stable_seed(seed, pid, preset, "calib-views"))
            n_zoom = int(job["zoom_views"]) if preset in ("clean", "nominal") else 0  # corrected regimes: full view only
            views = [View(0.5, 0.5, 1.0)] + sample_views(weights, vrng, CALIB_ZOOMS, n_zoom)
            prng = np.random.default_rng(stable_seed(seed, pid, preset, "calib-pixels"))
            for k, view in enumerate(views):
                cap = cam.capture(CaptureSettings(view=view, exposure_ev=ev, light=light, autofocus=False), f"cal_{k}", k)
                n_caps += 1
                score, zmap, valid, reg = stat.score_capture(cap)
                if deep is not None and view.zoom < 1.5:
                    deep_normal.append(float(deep.predict(cap).score))
                if not bool(getattr(reg, "ok", True)) or not math.isfinite(float(score)):
                    n_reg_fail += 1
                    continue
                b = zoom_bucket(view.zoom)
                scores[b].append(float(score))
                vals = np.asarray(zmap, np.float32)[np.asarray(valid) > 0] if valid is not None else np.asarray(zmap).ravel()
                vals = vals[np.isfinite(vals)]
                if vals.size > PIXELS_PER_CAPTURE:
                    vals = prng.choice(vals, PIXELS_PER_CAPTURE, replace=False)
                zpix[b].append(vals.astype(np.float32))
        if (i + 1) % 10 == 0:
            log(f"normal calibration {i + 1}/{len(val_paths)} parts ({n_caps} captures, {n_reg_fail} registration failures)")
    timings["normal_calib_s"] = time.perf_counter() - t0

    ref_sharp = _fill_missing_buckets({b: float(np.median(v)) for b, v in sharp.items() if v})
    ref_expo = {b: r for b, r in ((b, reference_exposure_from(v)) for b, v in expo.items()) if r is not None}
    for b in BUCKETS:  # buckets without samples inherit the nearest measured one
        if b not in ref_expo and ref_expo:
            src = min(ref_expo, key=lambda k: abs(BUCKETS.index(k) - BUCKETS.index(b)))
            ref_expo[b] = dict(ref_expo[src], filled_from=src)
    region_thr, n_outlier_caps = region_thresholds(scores, zpix)
    log(f"reference sharpness {json.dumps({k: round(v, 1) for k, v in ref_sharp.items()})}")
    log(f"reference exposure mean_lin {json.dumps({k: round(v['mean_lin'], 4) for k, v in ref_expo.items()})}")
    log(f"region thresholds p{REGION_PERCENTILE} {json.dumps({k: round(v, 3) for k, v in region_thr.items()})}")

    # 4. labeled calibration (calib split, single shot, nominal preset) ----------------------------
    t0 = time.perf_counter()
    labeled_stat: list[tuple[float, int, str]] = []
    labeled_deep: list[tuple[float, int]] = []
    calib_scores_all: list[tuple[float, int, bool]] = []
    for item in calib_items:
        pid = part_id_from_relpath(item["image"])
        scene = _imread(data_root / item["image"])
        cond = make_conditions("nominal", np.random.default_rng(stable_seed(seed, pid, "labeled-cond")))
        cam = VirtualCamera(scene, cond, cam_cfg, seed=stable_seed(seed, pid, "labeled-cam"))
        cap = cam.capture(CaptureSettings(), "cal_0", 0)
        score, _zmap, _valid, reg = stat.score_capture(cap)
        ok = bool(getattr(reg, "ok", True)) and math.isfinite(float(score))
        calib_scores_all.append((float(score), int(item["label"]), ok))
        if ok:
            labeled_stat.append((float(score), int(item["label"]), "z1"))
        if deep is not None:
            labeled_deep.append((float(deep.predict(cap).score), int(item["label"])))
    timings["labeled_calib_s"] = time.perf_counter() - t0

    stat.calibrate({b: v for b, v in scores.items() if v}, labeled_stat or None)
    score_based_thr = {str(k): float(v) for k, v in (stat.calib.get("region_thr") or {}).items()}
    mode = str(job.get("region_thr_mode", "pixel"))
    if mode == "score":
        applied_thr = dict(score_based_thr)
    elif mode == "max":
        applied_thr = {b: max(region_thr.get(b, 0.0), score_based_thr.get(b, 0.0)) for b in BUCKETS}
    else:
        applied_thr = dict(region_thr)
    apply_region_thresholds(stat, applied_thr)
    log(f"region thresholds applied ({mode}) {json.dumps({k: round(v, 3) for k, v in applied_thr.items()})}")
    if deep is not None:
        t0 = time.perf_counter()
        deep.calibrate(deep_normal, labeled_deep or None)
        timings["deep_calib_s"] = time.perf_counter() - t0

    # diagnostics (calibration split, in-sample for the logistic; eval numbers come from evaluate.py)
    diag = {
        "calib_auroc_fast_reliable": auroc([s for s, _, ok in calib_scores_all if ok], [y for _, y, ok in calib_scores_all if ok]),
        "calib_registration_fail_frac": (sum(1 for *_, ok in calib_scores_all if not ok) / len(calib_scores_all)) if calib_scores_all else float("nan"),
        "calib_auroc_deep": auroc([s for s, _ in labeled_deep], [y for _, y in labeled_deep]) if labeled_deep else float("nan"),
        "normal_registration_fail_frac": (n_reg_fail / n_caps) if n_caps else float("nan"),
        "normal_scores": {b: describe(v) for b, v in scores.items()},
        "deep_normal_scores": describe(deep_normal),
        "sharpness_samples": {b: describe(v) for b, v in sharp.items()},
    }
    log(
        f"calib AUROC fast={diag['calib_auroc_fast_reliable']:.3f} deep={diag['calib_auroc_deep']:.3f} "
        f"reg-fail normal={diag['normal_registration_fail_frac']:.3f} calib={diag['calib_registration_fail_frac']:.3f}"
    )

    timings["total_s"] = time.perf_counter() - t_all
    meta = {
        "category": category,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "builder": "scripts/build_models.py",
        "seed": seed,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "arch": platform.machine(),
        "counts": {
            "fit_available": len(split["fit"]),
            "fit_used": len(fit_paths),
            "deep_fit_used": n_deep_fit,
            "val_normal_used": len(val_paths),
            "normal_calib_captures": n_caps,
            "normal_calib_registration_failures": n_reg_fail,
            "normal_calib_scores_per_bucket": {b: len(v) for b, v in scores.items()},
            "calib_used": len(calib_items),
            "calib_reliable": len(labeled_stat),
            "calib_pos": sum(1 for it in calib_items if int(it["label"]) == 1),
            "calib_neg": sum(1 for it in calib_items if int(it["label"]) == 0),
        },
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
        "calibration": {
            "stat": _jsonable(getattr(stat, "calib", {})),
            "deep": _jsonable(getattr(deep, "calib", {})) if deep is not None else None,
            "region_thr": applied_thr,
            "region_thr_mode": mode,
            "region_thr_pixel_p999": region_thr,
            "region_percentile": REGION_PERCENTILE,
            "region_thr_score_based": score_based_thr,
            "region_thr_excluded_outlier_captures": n_outlier_caps,
            "presets": list(CALIB_REGIMES),
            "zooms": [1.0, *CALIB_ZOOMS],
            "zoom_views_per_level": int(job["zoom_views"]),
        },
        "reference_sharpness": ref_sharp,
        "reference_exposure": ref_expo,
        "diagnostics": diag,
        "deep_status": deep_status,
        "detector_config": asdict(det_cfg),
        "camera_config": asdict(cam_cfg),
    }
    meta = nan_to_none(meta)
    bundle = CategoryBundle(category=category, stat=stat, deep=deep, meta=meta, reference_sharpness=ref_sharp)
    save_bundle(bundle, models_dir)
    _merge_meta(models_dir / category / "meta.json", meta)
    log(f"saved bundle to {models_dir / category} in {timings['total_s']:.1f}s total")
    return {"category": category, "ok": True, "timings_s": meta["timings_s"], "deep": deep_status, "diagnostics": {
        k: diag[k] for k in ("calib_auroc_fast_reliable", "calib_auroc_deep", "normal_registration_fail_frac")
    }}


def _jsonable(obj: Any) -> Any:
    """JSON-safe copy (numpy -> Python, NaN -> None)."""
    return nan_to_none(json.loads(json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))))


def _merge_meta(path: Path, meta: dict) -> None:
    """Ensure meta.json holds our keys while keeping anything ``save_bundle`` wrote itself."""
    existing: dict = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    merged = {**existing, **meta}
    path.write_text(json.dumps(nan_to_none(merged), indent=2, allow_nan=False), encoding="utf-8")


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    ap.add_argument("--splits", type=Path, default=settings.data_dir / "splits.json")
    ap.add_argument("--data-root", type=Path, default=settings.data_dir / "visa")
    ap.add_argument("--models-dir", type=Path, default=settings.models_dir)
    ap.add_argument("--onnx", type=Path, default=settings.models_dir / "backbone_resnet18.onnx")
    ap.add_argument("--no-deep", action="store_true", help="skip the PatchCoreDNN detector")
    ap.add_argument("--max-fit", type=int, default=300, help="max fit normals for the statistical template")
    ap.add_argument("--max-val", type=int, default=0, help="max val_normal parts for calibration (0 = all)")
    ap.add_argument("--max-calib", type=int, default=0, help="max calib parts for the logistic (0 = all)")
    ap.add_argument("--zoom-views", type=int, default=1, help="random views per zoom level (2,3,4) per part and preset")
    ap.add_argument(
        "--region-thr-mode",
        choices=("pixel", "score", "max"),
        default="pixel",
        help="region threshold per zoom bucket: pixel = p99.9 of normal pixel z (default); "
        "score = StatTemplateModel's mu + 3 sd of normal image scores; max = the stricter of both",
    )
    ap.add_argument("--workers", type=int, default=0, help="parallel categories (0 = min(#categories, cpu/3))")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if not args.splits.is_file():
        raise SystemExit(f"{args.splits} not found - run scripts/prepare_visa.py first")
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    cats = [c for c in args.categories if c in splits and not c.startswith("_")]
    missing = [c for c in args.categories if c not in cats]
    if missing:
        print(f"[warn] not in {args.splits}: {missing}", flush=True)
    if not cats:
        raise SystemExit("no categories to build")
    cpu = os.cpu_count() or 4
    workers = args.workers or max(1, min(len(cats), cpu // 3))
    workers = max(1, min(workers, len(cats)))
    threads = max(1, cpu // workers)
    jobs = [
        {
            "category": c,
            "seed": args.seed,
            "threads": threads,
            "splits": str(args.splits),
            "data_root": str(args.data_root),
            "models_dir": str(args.models_dir),
            "onnx": str(args.onnx),
            "no_deep": bool(args.no_deep),
            "max_fit": args.max_fit,
            "max_val": args.max_val or None,
            "max_calib": args.max_calib or None,
            "zoom_views": max(1, args.zoom_views),
            "region_thr_mode": args.region_thr_mode,
        }
        for c in cats
    ]
    print(f"[build] {len(cats)} categories, {workers} workers x {threads} OpenCV threads, deep={'off' if args.no_deep else args.onnx.is_file()}", flush=True)
    t0 = time.perf_counter()
    results: list[dict] = []
    if workers == 1:
        results = [build_category(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
            futs = {ex.submit(build_category, j): j["category"] for j in jobs}
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as exc:  # worker crashed hard (e.g. out of memory)
                    results.append({"category": futs[fut], "ok": False, "error": repr(exc)})
    print(f"[build] finished in {time.perf_counter() - t0:.1f}s", flush=True)
    for r in sorted(results, key=lambda r: cats.index(r["category"])):
        status = "ok" if r.get("ok") else f"FAILED {r.get('error')}"
        print(f"  {r['category']:<10} {status} {json.dumps(nan_to_none(r.get('diagnostics', {})))}", flush=True)
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
