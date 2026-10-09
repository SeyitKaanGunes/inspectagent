"""FastAPI web API + static single-page UI for InspectAgent (SPEC §5).

Endpoints (JSON unless noted)::

    GET  /healthz                                   liveness + version + OpenCV build info
    GET  /api/samples                               curated demo parts from samples/samples.json
    GET  /api/samples/{id}/image[?full=1]           JPEG thumbnail (default) or full image
    POST /api/inspect                               JSON {sample_id, preset, planner, seed} or multipart upload
    GET  /api/traces?limit=20                       recent trace summaries
    GET  /api/traces/{trace_id}                     stored trace (+ artifact URLs)
    GET  /api/traces/{trace_id}/artifacts/{art_id}  stored JPEG artifact
    GET  /api/reviews?status=pending|resolved|rescanned|all
    POST /api/reviews/{review_id}/decision          {decision: approve_fail|override_pass|rescan, reviewer, note}
    GET  /api/metrics                               eval summary tables + figures + live counters
    GET  /api/figures/{name}                        evaluation figure image
    GET  /api/system                                OpenCV / runtime / deployment / non-secret settings
    GET  /api/bench?n=10                            in-process micro-benchmark (scripts.bench_cv.run_bench)
    GET  /                                          static UI (static/index.html)

Heavy collaborators (agent loop, model bundles, camera presets, planners) are reached through small
module-level proxy functions (``run_inspection``, ``load_bundle``, ``available_categories``,
``make_conditions``, ``make_planner``) that import lazily at call time, so the API starts even when optional
modules are missing and tests can monkeypatch them on this module.
"""

from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import importlib
import json
import logging
import math
import os
import platform
import secrets
import struct
import sys
import threading
import time
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from inspectagent import __version__
from inspectagent.api.storage import (
    REVIEW_STATUSES,
    ConflictError,
    NotFoundError,
    StorageError,
    append_audit,
    make_storage,
    utc_now,
    validate_id,
    verify_audit,
)
from inspectagent.config import CATEGORIES, REPO_ROOT, Settings, get_settings
from inspectagent.types import NuisanceConditions, PartSpec

log = logging.getLogger("inspectagent.api")

__all__ = [
    "create_app",
    "app",
    "ApiService",
    "ApiError",
    "SampleCatalog",
    "encode_artifacts",
    "eval_tables",
    "run_inspection",
    "load_bundle",
    "available_categories",
    "make_conditions",
    "make_planner",
    "cv_info",
]

# ------------------------------------------------------------------------------------------ constants

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # uploaded image size limit
MAX_UPLOAD_PIXELS = 50_000_000  # decompression-bomb guard (checked from the header before decoding)
MIN_UPLOAD_SIDE = 64
ARTIFACT_MAX_SIDE = 512  # artifact thumbnails in the inspect response
RESPONSE_BUDGET_BYTES = 5_000_000  # Lambda Function URL payload limit is 6 MB
GALLERY_THUMB_SIDE = 384
MAX_ASPECT_MISMATCH = 1.25  # uploads whose aspect differs more than this from the category model are rejected
DEFAULT_PRESETS = ("clean", "nominal", "dark", "bright", "defocus", "glare", "harsh")
PLANNERS = ("rule", "bedrock")
DECISIONS = ("approve_fail", "override_pass", "rescan")
STATIC_DIR = Path(__file__).resolve().parent / "static"
FIGURE_EXTS = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml", ".webp": "image/webp"}
_SEED_MAX = 2**31 - 1


# ------------------------------------------------------------------------------------------- logging


def configure_logging(level: str | int = "INFO") -> None:
    """Make ``inspectagent.*`` JSON log lines visible (CloudWatch picks up stdout in Lambda)."""
    pkg = logging.getLogger("inspectagent")
    try:
        pkg.setLevel(level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO))
    except (TypeError, ValueError):
        pkg.setLevel(logging.INFO)
    if not logging.getLogger().handlers and not any(getattr(h, "_inspectagent", False) for h in pkg.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._inspectagent = True  # type: ignore[attr-defined]
        pkg.addHandler(handler)


def log_event(evt: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one structured JSON log line ``{"evt": ..., ...}``."""
    if log.isEnabledFor(level):
        log.log(level, json.dumps({"evt": evt, "ts": utc_now(), **fields}, default=str, ensure_ascii=False))


# ------------------------------------------------------------------------------------------- errors


class ApiError(HTTPException):
    """HTTP error raised by the service layer (status code + human-readable detail)."""

    def __init__(self, status_code: int, detail: Any) -> None:
        super().__init__(status_code=status_code, detail=detail)


# ---------------------------------------------------------------------------- lazy collaborator proxies


def run_inspection(
    part: PartSpec,
    bundle: Any = None,
    planner: Any = None,
    policy: Any = None,
    settings: Settings | None = None,
    seed: int = 0,
    on_step: Any = None,
    scene: np.ndarray | None = None,
) -> tuple[Any, Any]:
    """Proxy to :func:`inspectagent.agent.loop.run_inspection` (SPEC §3.6), imported on first use."""
    from inspectagent.agent.loop import run_inspection as _impl

    return _impl(
        part, bundle=bundle, planner=planner, policy=policy, settings=settings, seed=seed, on_step=on_step, scene=scene
    )


def load_bundle(category: str, models_dir: Path | None = None, with_deep: bool = True) -> Any:
    """Proxy to :func:`inspectagent.models_io.load_bundle` (cached per process by models_io)."""
    from inspectagent.models_io import load_bundle as _impl

    return _impl(category, models_dir=models_dir, with_deep=with_deep)


def available_categories(models_dir: Path | None = None) -> list[str]:
    """Categories with a model bundle on disk ([] when models_io is not installed)."""
    try:
        from inspectagent.models_io import available_categories as _impl
    except ImportError as exc:
        log_event("models_io_unavailable", logging.WARNING, error=str(exc))
        return []
    try:
        return sorted(str(c) for c in _impl(models_dir=models_dir))
    except Exception as exc:  # unreadable models dir must not break the API
        log_event("available_categories_failed", logging.WARNING, error=str(exc))
        return []


def make_conditions(preset: str, rng: np.random.Generator) -> NuisanceConditions:
    """Proxy to :func:`inspectagent.vision.camera.make_conditions` (hidden nuisance state for a preset)."""
    from inspectagent.vision.camera import make_conditions as _impl

    return _impl(preset, rng)


def presets() -> tuple[str, ...]:
    """Camera condition presets (from vision.camera.PRESETS, SPEC default if not importable)."""
    try:
        from inspectagent.vision.camera import PRESETS

        return tuple(PRESETS)
    except ImportError:
        return DEFAULT_PRESETS


def make_planner(name: str, settings: Settings) -> Any:
    """Instantiate the planner selected by name (``rule`` | ``bedrock``)."""
    try:
        if name == "rule":
            from inspectagent.agent.planner_rule import RulePlanner

            return RulePlanner()
        if name == "bedrock":
            if not settings.bedrock_model_id:
                raise ApiError(400, "bedrock planner not configured: set INSPECTAGENT_BEDROCK_MODEL_ID")
            from inspectagent.agent.planner_bedrock import BedrockPlanner
            from inspectagent.agent.planner_rule import RulePlanner

            return BedrockPlanner(
                settings.bedrock_model_id,
                region=settings.aws_region,
                max_tokens=settings.bedrock_max_tokens,
                fallback=RulePlanner(),
                settings=settings,
            )
    except ImportError as exc:
        raise ApiError(503, f"planner {name!r} unavailable: {exc}") from exc
    raise ApiError(422, f"unknown planner {name!r}; expected one of {list(PLANNERS)}")


_CV_INFO: dict | None = None
_CV_INFO_LOCK = threading.Lock()


def _fallback_cv_info() -> dict:
    """Minimal OpenCV build introspection used when vision.cvinfo is not available."""
    build = cv2.getBuildInformation()
    evidence = [ln.strip() for ln in build.splitlines() if "cool" in ln.lower() or "cloud-optimized" in ln.lower()]
    env_backend = os.environ.get("INSPECTAGENT_CV_BACKEND", "").lower()

    def _field(prefix: str) -> str:
        for ln in build.splitlines():
            if ln.strip().startswith(prefix):
                return ln.split(":", 1)[1].strip()
        return ""

    engines = [n for n in ("ENGINE_AUTO", "ENGINE_CLASSIC", "ENGINE_NEW", "ENGINE_ORT") if hasattr(cv2.dnn, n)]
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        runtime = "lambda"
    elif Path("/.dockerenv").exists():
        runtime = "container"
    else:
        runtime = "local"
    return {
        "opencv_version": cv2.__version__,
        "build_backend": "cool" if env_backend == "cool" or evidence else "stock",
        "backend_evidence": "; ".join(evidence) or (f"env INSPECTAGENT_CV_BACKEND={env_backend}" if env_backend else ""),
        "arch": platform.machine(),
        "cpu_baseline": _field("Baseline"),
        "cpu_dispatch": _field("Dispatched code generation"),
        "parallel_framework": _field("Parallel framework"),
        "threads": cv2.getNumThreads(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "dnn_engines": engines,
        "runtime": runtime,
    }


def cv_info() -> dict:
    """OpenCV build/arch info (vision.cvinfo.cv_info when available), cached per process."""
    global _CV_INFO
    with _CV_INFO_LOCK:
        if _CV_INFO is None:
            try:
                from inspectagent.vision.cvinfo import cv_info as _impl

                _CV_INFO = dict(_impl())
            except Exception as exc:  # ImportError or introspection failure -> local fallback
                log_event("cvinfo_fallback", logging.DEBUG, error=str(exc))
                _CV_INFO = _fallback_cv_info()
        return dict(_CV_INFO)


# --------------------------------------------------------------------------------------- image helpers


def _decode_image(data: bytes, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Decode encoded image bytes (None when undecodable)."""
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    try:
        img = cv2.imdecode(arr, flags)
    except cv2.error:
        return None
    return img


def _encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def _read_image_file(path: Path) -> np.ndarray:
    """Read an image from a (possibly non-ASCII) path without cv2.imread."""
    try:
        from inspectagent.vision.imageio import imread as _imread

        return _imread(path)
    except ImportError:
        pass
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise ValueError(f"cannot decode image {Path(path).name}")
    return img


def _fit_long_side(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= long_side:
        return img
    s = long_side / float(m)
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def sniff_image(data: bytes) -> tuple[str | None, tuple[int, int] | None]:
    """Identify JPEG/PNG by magic bytes and read (W, H) from the header without decoding."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        if len(data) >= 24 and data[12:16] == b"IHDR":
            w, h = struct.unpack(">II", data[16:24])
            return "png", (int(w), int(h))
        return "png", None
    if data[:3] == b"\xff\xd8\xff":
        sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
        i, n = 2, len(data)
        while i + 4 <= n:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker == 0xFF:
                i += 1
                continue
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            if marker in (0xD9, 0xDA):
                break
            seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
            if marker in sof and i + 9 <= n:
                h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                return "jpeg", (int(w), int(h))
            i += 2 + seg_len
        return "jpeg", None
    return None, None


def encode_artifacts(
    artifacts: dict[str, bytes],
    order: list[str],
    budget_bytes: int,
    max_side: int = ARTIFACT_MAX_SIDE,
) -> tuple[dict[str, str], list[str]]:
    """Encode artifacts as JPEG data URLs (long side <= ``max_side``) within ``budget_bytes`` of text.

    Progressively shrinks/recompresses all thumbnails when over budget and, as a last resort, drops the
    oldest artifacts (the newest evidence is kept). Returns ``(data_urls, notes)``.
    """
    notes: list[str] = []
    decoded: dict[str, tuple[np.ndarray, bytes]] = {}
    for aid in order:
        raw = artifacts.get(aid)
        img = _decode_image(raw) if raw else None
        if img is None:
            notes.append(f"artifact {aid} is not a decodable image and was omitted")
            continue
        decoded[aid] = (img, raw)
    if not decoded:
        return {}, notes

    ladder = [(max_side, 82), (min(max_side, 384), 72), (min(max_side, 320), 64), (min(max_side, 256), 56), (min(max_side, 192), 48)]
    base = {aid: _fit_long_side(img, max_side) for aid, (img, _) in decoded.items()}
    encoded: dict[str, str] = {}
    total = 0
    for level, (side, quality) in enumerate(ladder):
        encoded = {}
        for aid, img in base.items():
            src_img, raw = decoded[aid]
            if level == 0 and max(src_img.shape[:2]) <= max_side and raw[:3] == b"\xff\xd8\xff":
                encoded[aid] = _data_url(raw)  # already a small JPEG: pass through untouched
            else:
                encoded[aid] = _data_url(_encode_jpeg(_fit_long_side(img, side), quality))
        total = sum(len(v) + len(k) + 8 for k, v in encoded.items())
        if total <= budget_bytes:
            if level:
                notes.append(f"artifact thumbnails reduced to {side}px / q{quality} to fit the response budget")
            return encoded, notes
    keys = list(encoded)
    dropped = 0
    while keys and total > budget_bytes:
        k = keys.pop(0)
        total -= len(encoded[k]) + len(k) + 8
        dropped += 1
    if dropped:
        notes.append(f"{dropped} oldest artifact(s) omitted to keep the response under {budget_bytes} bytes")
    return {k: encoded[k] for k in keys}, notes


# ------------------------------------------------------------------------------------ eval summaries


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def _flatten_row(d: dict, prefix: str = "") -> dict:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if _is_scalar(v):
            out[key] = v
        elif isinstance(v, dict) and all(_is_scalar(x) for x in v.values()) and len(v) <= 16:
            out.update(_flatten_row(v, f"{key}."))
        elif isinstance(v, list) and all(_is_scalar(x) for x in v):
            out[key] = ", ".join("" if x is None else str(x) for x in v[:12]) + (" …" if len(v) > 12 else "")
        elif isinstance(v, (list, dict)):
            out[key] = f"[{len(v)} items]"
        else:
            out[key] = str(v)
    return out


def _is_pure(d: Any) -> bool:
    """Non-empty dict whose values are scalars or lists of scalars."""
    if not isinstance(d, dict) or not d:
        return False
    return all(_is_scalar(v) or (isinstance(v, list) and all(_is_scalar(x) for x in v)) for v in d.values())


def _is_record(d: Any) -> bool:
    """Like :func:`_is_pure` but also allows one level of nested scalar dicts (flattened as ``a.b``)."""
    if not isinstance(d, dict) or not d:
        return False
    return all(
        _is_scalar(v)
        or (isinstance(v, list) and all(_is_scalar(x) for x in v))
        or (isinstance(v, dict) and all(_is_scalar(x) for x in v.values()))
        for v in d.values()
    )


def _table(title: str, rows: list[dict], max_rows: int, max_cols: int = 24) -> dict:
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    cols = cols[:max_cols]
    return {"title": title or "summary", "columns": cols, "rows": [[r.get(c) for c in cols] for r in rows[:max_rows]]}


def eval_tables(summary: Any, max_rows: int = 400) -> list[dict]:
    """Turn an arbitrary evaluation-summary JSON into display tables ``{title, columns, rows}``.

    Lists of records become tables; dicts of records (``{system: {metric: v}}``) become tables with a key
    column; two-level dicts (``{category: {system: {metric: v}}}``) get two key columns; scalars are
    collected into an "Overview" key/value table. Deeper structures are walked recursively, preferring
    the interpretation with flat (pure scalar) records.
    """
    tables: list[dict] = []
    scalars: list[list[Any]] = []

    def walk(node: Any, path: str, depth: int) -> None:
        if depth > 6:
            return
        if isinstance(node, list):
            if node and all(isinstance(x, dict) for x in node):
                tables.append(_table(path, [_flatten_row(x) for x in node], max_rows))
            elif all(_is_scalar(x) for x in node):
                if path:
                    scalars.append([path, ", ".join(str(x) for x in node[:20])])
            else:
                for i, x in enumerate(node[:50]):
                    walk(x, f"{path}[{i}]", depth + 1)
            return
        if not isinstance(node, dict):
            if path:
                scalars.append([path, node])
            return
        vals = list(node.values())
        if vals and all(_is_pure(v) for v in vals):
            tables.append(_table(path, [{"key": k, **_flatten_row(v)} for k, v in node.items()], max_rows))
            return
        if vals and all(isinstance(v, dict) and v and all(_is_pure(x) for x in v.values()) for v in vals):
            rows = [{"key": k, "key2": k2, **_flatten_row(v2)} for k, v in node.items() for k2, v2 in v.items()]
            tables.append(_table(path, rows, max_rows))
            return
        if vals and all(_is_record(v) for v in vals):
            tables.append(_table(path, [{"key": k, **_flatten_row(v)} for k, v in node.items()], max_rows))
            return
        for k, v in node.items():
            p = f"{path}.{k}" if path else str(k)
            if _is_scalar(v):
                scalars.append([p, v])
            else:
                walk(v, p, depth + 1)

    walk(summary, "", 0)
    if scalars:
        tables.insert(0, {"title": "Overview", "columns": ["metric", "value"], "rows": scalars[:max_rows]})
    return tables


def _jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays, dataclasses and paths into JSON-safe values."""

    def default(o: Any) -> Any:
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist() if o.size <= 4096 else f"<array {o.shape} {o.dtype}>"
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        if isinstance(o, (Path, bytes)):
            return str(o) if isinstance(o, Path) else f"<{len(o)} bytes>"
        if hasattr(o, "to_dict"):
            return o.to_dict()
        return str(o)

    return json.loads(json.dumps(obj, default=default))


# ------------------------------------------------------------------------------------------- samples


def _first(d: dict, *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] is not None and d[k] != "":
            return d[k]
    return None


def _norm_label(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return "anomaly" if v else "normal"
    if isinstance(v, (int, float)):
        return "anomaly" if float(v) >= 0.5 else "normal"
    s = str(v).strip().lower()
    if s in {"anomaly", "anomalous", "abnormal", "defect", "defective", "bad", "ng", "fail", "1", "true"}:
        return "anomaly"
    if s in {"normal", "good", "ok", "pass", "0", "false"}:
        return "normal"
    return None


class SampleCatalog:
    """Curated demo parts loaded from ``<samples_dir>/samples.json`` (reloaded when the file changes).

    Accepted formats: a list of entries, ``{"samples": [...]}`` or ``{id: entry}``. Entry keys (aliases):
    ``id|sample_id|part_id|name``, ``category|object``, ``label`` (normal/anomaly or 0/1),
    ``image|image_path|path|file``, optional ``mask|mask_path``, ``title``, ``description``, ``defect_type``.
    Relative paths resolve against the samples dir (then the repo root). Entries whose image is missing are
    skipped (logged).
    """

    def __init__(self, samples_dir: Path) -> None:
        self.dir = Path(samples_dir)
        self._lock = threading.Lock()
        self._sig: tuple[float, int] | None = None
        self._items: "OrderedDict[str, dict]" = OrderedDict()
        self._thumbs: "OrderedDict[tuple[str, int], bytes]" = OrderedDict()

    @property
    def index_path(self) -> Path:
        return self.dir / "samples.json"

    def _resolve(self, value: Any) -> Path | None:
        if not value:
            return None
        p = Path(str(value))
        candidates = [p] if p.is_absolute() else [self.dir / p, REPO_ROOT / p]
        for c in candidates:
            if c.is_file():
                return c.resolve()
        return None

    def _load(self) -> None:
        path = self.index_path
        try:
            st = path.stat()
        except FileNotFoundError:
            self._items, self._sig = OrderedDict(), None
            return
        sig = (st.st_mtime, st.st_size)
        if sig == self._sig:
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log_event("samples_unreadable", logging.ERROR, error=str(exc))
            self._items, self._sig = OrderedDict(), sig
            return
        if isinstance(raw, dict) and isinstance(raw.get("samples"), list):
            entries = raw["samples"]
        elif isinstance(raw, dict):
            entries = [dict(v, id=v.get("id", k)) for k, v in raw.items() if isinstance(v, dict)]
        elif isinstance(raw, list):
            entries = raw
        else:
            entries = []
        items: "OrderedDict[str, dict]" = OrderedDict()
        for e in entries:
            if not isinstance(e, dict):
                continue
            image = self._resolve(_first(e, "image", "image_path", "path", "file", "filename", "img"))
            if image is None:
                log_event("sample_image_missing", logging.WARNING, entry=str(_first(e, "id", "sample_id", "name")))
                continue
            sid = str(_first(e, "id", "sample_id", "part_id", "name") or image.stem)
            try:
                validate_id(sid, "sample id")
            except StorageError:
                log_event("sample_id_invalid", logging.WARNING, id=sid)
                continue
            label = _norm_label(_first(e, "label", "gt", "ground_truth", "anomaly", "defective", "is_anomaly"))
            items[sid] = {
                "id": sid,
                "category": str(_first(e, "category", "object", "class") or "unknown"),
                "label": label,
                "defective": None if label is None else label == "anomaly",
                "title": _first(e, "title", "caption"),
                "description": _first(e, "description", "note", "notes"),
                "defect_type": _first(e, "defect_type", "defect", "anomaly_type"),
                "suggested_preset": _first(e, "suggested_preset", "preset"),
                "defect_area_frac": _first(e, "defect_area_frac"),
                "image_path": image,
                "mask_path": self._resolve(_first(e, "mask", "mask_path")),
                "thumb_path": self._resolve(_first(e, "thumb", "thumbnail", "thumb_path")),
            }
        self._items, self._sig = items, sig
        self._thumbs.clear()

    def list(self) -> list[dict]:
        with self._lock:
            self._load()
            return [dict(v) for v in self._items.values()]

    def get(self, sample_id: str) -> dict | None:
        with self._lock:
            self._load()
            item = self._items.get(sample_id)
            return dict(item) if item else None

    def image_bytes(self, sample: dict, full: bool) -> bytes:
        """JPEG bytes of the sample (gallery thumbnail by default, full resolution with ``full=True``)."""
        path: Path = sample["image_path"]
        if full:
            raw = path.read_bytes()
            if raw[:3] == b"\xff\xd8\xff":
                return raw
            return _encode_jpeg(_read_image_file(path), 92)
        key = (sample["id"], int(path.stat().st_mtime_ns))
        with self._lock:
            cached = self._thumbs.get(key)
            if cached is not None:
                self._thumbs.move_to_end(key)
                return cached
        src = sample.get("thumb_path") or path
        thumb = _encode_jpeg(_fit_long_side(_read_image_file(src), GALLERY_THUMB_SIDE), 84)
        with self._lock:
            self._thumbs[key] = thumb
            while len(self._thumbs) > 256:
                self._thumbs.popitem(last=False)
        return thumb


# ------------------------------------------------------------------------------------- live counters


class LiveCounters:
    """In-process counters since start (per Lambda container / uvicorn worker)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = utc_now()
        self.inspections = 0
        self.errors = 0
        self.reviews_created = 0
        self.rescans = 0
        self.by_label: Counter[str] = Counter()
        self.by_action: Counter[str] = Counter()
        self.by_preset: Counter[str] = Counter()
        self.by_planner: Counter[str] = Counter()
        self.tool_counts: Counter[str] = Counter()
        self.decisions: Counter[str] = Counter()
        self._sum = {"cycle_time_ms": 0.0, "compute_ms": 0.0, "sim_time_ms": 0.0, "n_steps": 0.0, "n_captures": 0.0, "wall_ms": 0.0}

    def record_inspection(self, trace: dict, preset: str, planner: str, wall_ms: float) -> None:
        verdict = trace.get("verdict") or {}
        totals = trace.get("totals") or {}
        with self._lock:
            self.inspections += 1
            self.by_label[str(verdict.get("label"))] += 1
            self.by_action[str(verdict.get("action"))] += 1
            self.by_preset[preset] += 1
            self.by_planner[planner] += 1
            for step in trace.get("steps") or []:
                self.tool_counts[str((step.get("decision") or {}).get("tool"))] += 1
            for k in ("cycle_time_ms", "compute_ms", "sim_time_ms", "n_steps", "n_captures"):
                v = totals.get(k)
                if isinstance(v, (int, float)) and math.isfinite(v):
                    self._sum[k] += float(v)
            self._sum["wall_ms"] += wall_ms

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def record_review(self) -> None:
        with self._lock:
            self.reviews_created += 1

    def record_decision(self, decision: str) -> None:
        with self._lock:
            self.decisions[decision] += 1
            if decision == "rescan":
                self.rescans += 1

    def snapshot(self) -> dict:
        with self._lock:
            n = max(1, self.inspections)
            return {
                "since": self.started_at,
                "inspections": self.inspections,
                "errors": self.errors,
                "reviews_created": self.reviews_created,
                "rescans": self.rescans,
                "by_label": dict(self.by_label),
                "by_action": dict(self.by_action),
                "by_preset": dict(self.by_preset),
                "by_planner": dict(self.by_planner),
                "tool_counts": dict(self.tool_counts),
                "decisions": dict(self.decisions),
                "means": {k: (round(v / n, 2) if self.inspections else None) for k, v in self._sum.items()},
            }


# ----------------------------------------------------------------------------------- request models


class InspectRequest(BaseModel):
    """JSON body of ``POST /api/inspect``."""

    sample_id: str = Field(min_length=1, max_length=128)
    preset: str = "nominal"
    planner: Literal["rule", "bedrock"] = "rule"
    seed: int = Field(0, ge=0, le=_SEED_MAX)


class DecisionRequest(BaseModel):
    """Body of ``POST /api/reviews/{review_id}/decision``."""

    decision: Literal["approve_fail", "override_pass", "rescan"]
    reviewer: str = Field("operator", min_length=1, max_length=64)
    note: str = Field("", max_length=2000)


# ------------------------------------------------------------------------------------------ service


def _artifact_url(trace_id: str, art_id: str) -> str:
    return f"/api/traces/{quote(trace_id, safe='')}/artifacts/{quote(art_id, safe='')}"


def _ordered_artifact_ids(trace: dict, artifacts: dict[str, bytes]) -> list[str]:
    order: list[str] = []
    for step in trace.get("steps") or []:
        for aid in step.get("artifacts") or []:
            if aid in artifacts and aid not in order:
                order.append(aid)
    order.extend(a for a in artifacts if a not in order)
    return order


class ApiService:
    """Framework-independent implementation of the API behaviour (used by the FastAPI routes)."""

    def __init__(self, settings: Settings, storage: Any) -> None:
        self.settings = settings
        self.storage = storage
        self.catalog = SampleCatalog(settings.samples_dir)
        self.counters = LiveCounters()
        self._run_lock = threading.Lock()  # cv2.dnn nets / shared bundles are not re-entrant
        self._eval_cache: tuple[tuple[float, int], dict | None] | None = None

    # -- validation helpers --------------------------------------------------------------------
    def _check_preset(self, preset: str) -> str:
        allowed = presets()
        if preset not in allowed:
            raise ApiError(422, f"unknown preset {preset!r}; expected one of {list(allowed)}")
        return preset

    def _conditions(self, preset: str, seed: int) -> NuisanceConditions:
        try:
            cond = make_conditions(preset, np.random.default_rng(int(seed)))
        except ImportError as exc:
            raise ApiError(503, f"camera simulator unavailable: {exc}") from exc
        except ValueError as exc:
            raise ApiError(422, str(exc)) from exc
        if getattr(cond, "preset", "custom") in ("", "custom", None):
            try:
                cond.preset = preset
            except (AttributeError, dataclasses.FrozenInstanceError):
                pass
        return cond

    def _bundle(self, category: str) -> Any:
        cats = available_categories(self.settings.models_dir)
        if category not in cats:
            raise ApiError(503, f"no model bundle available for category {category!r} (available: {cats})")
        try:
            return load_bundle(category, models_dir=self.settings.models_dir, with_deep=True)
        except ImportError as exc:
            raise ApiError(503, f"model loader unavailable: {exc}") from exc
        except Exception as exc:
            log_event("bundle_deep_load_failed", logging.WARNING, category=category, error=str(exc))
            try:
                return load_bundle(category, models_dir=self.settings.models_dir, with_deep=False)
            except Exception as exc2:
                raise ApiError(503, f"model bundle for {category!r} failed to load: {exc2}") from exc2

    @staticmethod
    def _expected_scene_size(bundle: Any) -> tuple[int, int] | None:
        size = getattr(getattr(bundle, "stat", None), "scene_size", None)
        if size is None:
            meta = getattr(bundle, "meta", None)
            size = meta.get("scene_size") if isinstance(meta, dict) else None
        try:
            w, h = int(size[0]), int(size[1])
        except (TypeError, ValueError, IndexError):
            return None
        return (w, h) if w > 0 and h > 0 else None

    def _conform_scene(self, scene: np.ndarray, bundle: Any, category: str, notes: list[str]) -> np.ndarray:
        """Resize the scene to the category model's scene size (aspect must roughly match)."""
        expected = self._expected_scene_size(bundle)
        h, w = scene.shape[:2]
        if expected is None or expected == (w, h):
            return scene
        ew, eh = expected
        ratio = (w / h) / (ew / eh)
        if max(ratio, 1.0 / ratio) > MAX_ASPECT_MISMATCH:
            raise ApiError(
                422,
                f"image aspect ratio {w / h:.3f} is incompatible with the {category!r} model (expected ~{ew / eh:.3f})",
            )
        interp = cv2.INTER_AREA if w * h > ew * eh else cv2.INTER_CUBIC
        notes.append(f"image resized from {w}x{h} to {ew}x{eh} to match the {category} model geometry")
        return cv2.resize(scene, (ew, eh), interpolation=interp)

    # -- samples ------------------------------------------------------------------------------
    def list_samples(self) -> dict:
        cats = set(available_categories(self.settings.models_dir))
        out = []
        for s in self.catalog.list():
            out.append(
                {
                    "id": s["id"],
                    "category": s["category"],
                    "label": s["label"],
                    "defective": s["defective"],
                    "title": s["title"],
                    "description": s["description"],
                    "defect_type": s["defect_type"],
                    "suggested_preset": s["suggested_preset"],
                    "defect_area_frac": s["defect_area_frac"],
                    "thumb_url": f"/api/samples/{quote(s['id'], safe='')}/image",
                    "image_url": f"/api/samples/{quote(s['id'], safe='')}/image?full=1",
                    "model_available": s["category"] in cats,
                }
            )
        return {"samples": out, "count": len(out), "categories_with_models": sorted(cats)}

    def sample_image(self, sample_id: str, full: bool) -> bytes:
        sample = self.catalog.get(sample_id)
        if sample is None:
            raise ApiError(404, f"unknown sample {sample_id!r}")
        try:
            return self.catalog.image_bytes(sample, full)
        except (OSError, ValueError) as exc:
            raise ApiError(500, f"cannot read sample image: {exc}") from exc

    def _sample_part(self, sample: dict, preset: str, seed: int) -> tuple[PartSpec, np.ndarray]:
        label = sample.get("label")
        part = PartSpec(
            part_id=sample["id"],
            category=sample["category"],
            image_path=str(sample["image_path"]),
            label=None if label is None else int(label == "anomaly"),
            mask_path=str(sample["mask_path"]) if sample.get("mask_path") else None,
            conditions=self._conditions(preset, seed),
        )
        try:
            scene = _read_image_file(sample["image_path"])
        except (OSError, ValueError) as exc:
            raise ApiError(500, f"cannot read sample image: {exc}") from exc
        return part, scene

    # -- inspection ---------------------------------------------------------------------------
    def inspect_sample(self, sample_id: str, preset: str = "nominal", planner: str = "rule", seed: int = 0) -> dict:
        self._check_preset(preset)
        sample = self.catalog.get(sample_id)
        if sample is None:
            raise ApiError(404, f"unknown sample {sample_id!r}")
        part, scene = self._sample_part(sample, preset, seed)
        request_info = {
            "source": "sample",
            "sample_id": sample_id,
            "category": sample["category"],
            "preset": preset,
            "planner": planner,
            "seed": int(seed),
        }
        return self._execute(part=part, scene=scene, request_info=request_info, planner_name=planner, seed=seed)

    def validate_upload(self, data: bytes, category: str) -> np.ndarray:
        """Validate an uploaded image (type, size, dimensions, category) and decode it."""
        if len(data) > MAX_UPLOAD_BYTES:
            raise ApiError(413, f"image too large: {len(data)} bytes (limit {MAX_UPLOAD_BYTES})")
        if not data:
            raise ApiError(422, "empty upload")
        fmt, dims = sniff_image(data)
        if fmt is None:
            raise ApiError(415, "unsupported image type: only JPEG and PNG are accepted")
        if dims is not None and dims[0] * dims[1] > MAX_UPLOAD_PIXELS:
            raise ApiError(413, f"image dimensions {dims[0]}x{dims[1]} exceed {MAX_UPLOAD_PIXELS} pixels")
        cats = available_categories(self.settings.models_dir)
        if category not in cats:
            raise ApiError(422, f"category {category!r} has no model bundle (available: {cats})")
        img = _decode_image(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ApiError(422, f"could not decode the {fmt.upper()} image")
        h, w = img.shape[:2]
        if min(h, w) < MIN_UPLOAD_SIDE:
            raise ApiError(422, f"image too small: {w}x{h} (minimum side {MIN_UPLOAD_SIDE}px)")
        return img

    def inspect_upload(
        self,
        data: bytes,
        filename: str,
        category: str,
        preset: str = "nominal",
        planner: str = "rule",
        seed: int = 0,
    ) -> dict:
        self._check_preset(preset)
        if planner not in PLANNERS:
            raise ApiError(422, f"unknown planner {planner!r}; expected one of {list(PLANNERS)}")
        scene = self.validate_upload(data, category)
        digest = hashlib.sha256(data).hexdigest()
        safe_name = Path(filename or "upload").name[:96]
        part = PartSpec(
            part_id=f"upload-{digest[:12]}",
            category=category,
            image_path=f"upload:{safe_name}",
            label=None,
            conditions=self._conditions(preset, seed),
        )
        request_info = {
            "source": "upload",
            "filename": safe_name,
            "sha256": digest,
            "category": category,
            "preset": preset,
            "planner": planner,
            "seed": int(seed),
        }
        return self._execute(part=part, scene=scene, request_info=request_info, planner_name=planner, seed=seed)

    def _unique_trace_id(self, trace_id: Any) -> str:
        tid = str(trace_id or "")
        try:
            validate_id(tid, "trace id")
        except StorageError:
            tid = f"tr-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{secrets.token_hex(3)}"
        candidate = tid
        for k in range(2, 10_000):
            try:
                if not self.storage.trace_exists(candidate):
                    return candidate
            except StorageError as exc:
                log_event("trace_exists_failed", logging.WARNING, error=str(exc))
                return candidate
            candidate = f"{tid}-r{k}"
        return f"{tid}-{secrets.token_hex(4)}"

    @staticmethod
    def _evidence(trace: dict, limit: int = 6) -> list[dict]:
        out: list[dict] = []
        tid = trace["trace_id"]
        for step in reversed(trace.get("steps") or []):
            tool = (step.get("decision") or {}).get("tool")
            for aid in reversed(step.get("artifacts") or []):
                out.append({"artifact_id": aid, "step": step.get("step"), "tool": tool, "url": _artifact_url(tid, aid)})
                if len(out) >= limit:
                    return list(reversed(out))
        return list(reversed(out))

    def _new_review(self, trace: dict, request_info: dict, parent_review_id: str | None) -> dict | None:
        verdict = trace.get("verdict") or {}
        label, action = verdict.get("label"), verdict.get("action")
        if label == "NEEDS_REVIEW" or action == "hold_for_review":
            kind = "needs_review"
        elif label == "FAIL" and action == "route_quarantine":
            kind = "scrap_approval"
        else:
            return None
        req_args: dict = {}
        for step in reversed(trace.get("steps") or []):
            dec = step.get("decision") or {}
            if dec.get("tool") == "request_human_review":
                req_args = dec.get("args") or {}
                break
        now = utc_now()
        review = {
            "review_id": f"rv-{trace['trace_id']}"[:128],
            "trace_id": trace["trace_id"],
            "kind": kind,
            "status": "pending",
            "created_at": now,
            "updated_at": now,
            "part_id": trace.get("part_id"),
            "category": trace.get("category"),
            "planner": trace.get("planner"),
            "verdict_label": label,
            "proposed_label": req_args.get("proposed_label") or ("FAIL" if label == "FAIL" else None),
            "confidence": verdict.get("confidence"),
            "prob_anomalous": verdict.get("prob_anomalous"),
            "reason": str(req_args.get("reason") or verdict.get("rationale") or ""),
            "agent_review_id": verdict.get("review_id"),
            "routing_action": action,
            "defect_regions": verdict.get("defect_regions") or [],
            "evidence": self._evidence(trace),
            "request": request_info,
            "parent_review_id": parent_review_id,
            "decision": None,
            "final_action": None,
            "final_label": None,
            "rescan_trace_id": None,
            "version": 1,
            "audit": [],
        }
        append_audit(
            review,
            {
                "event": "created",
                "actor": f"agent:{trace.get('planner')}",
                "kind": kind,
                "action": action,
                "verdict": label,
                "prob_anomalous": verdict.get("prob_anomalous"),
                "ts": now,
            },
        )
        return review

    def _execute(
        self,
        *,
        part: PartSpec,
        scene: np.ndarray,
        request_info: dict,
        planner_name: str,
        seed: int,
        links: dict | None = None,
        parent_review_id: str | None = None,
    ) -> dict:
        notes: list[str] = []
        bundle = self._bundle(part.category)
        scene = self._conform_scene(scene, bundle, part.category, notes)
        planner = make_planner(planner_name, self.settings)
        t0 = time.perf_counter()
        try:
            with self._run_lock:
                trace_obj, session = run_inspection(
                    part, bundle=bundle, planner=planner, settings=self.settings, seed=int(seed), scene=scene
                )
        except HTTPException:
            raise
        except Exception as exc:
            self.counters.record_error()
            log_event("inspection_failed", logging.ERROR, part_id=part.part_id, error=f"{type(exc).__name__}: {exc}")
            raise ApiError(500, f"inspection failed: {type(exc).__name__}: {exc}") from exc
        wall_ms = (time.perf_counter() - t0) * 1000.0

        trace = copy.deepcopy(trace_obj.to_dict() if hasattr(trace_obj, "to_dict") else dict(trace_obj))
        original_id = trace.get("trace_id")
        trace["trace_id"] = self._unique_trace_id(original_id)
        if trace["trace_id"] != original_id:
            trace["original_trace_id"] = original_id
        tid = trace["trace_id"]
        trace["request"] = request_info
        trace["links"] = dict(links or {})
        trace["api"] = {"version": __version__, "wall_ms": round(wall_ms, 1), "storage": self.storage.kind}

        artifacts = {str(k): bytes(v) for k, v in (getattr(session, "artifacts", None) or {}).items()}
        order = _ordered_artifact_ids(trace, artifacts)

        review = self._new_review(trace, request_info, parent_review_id)
        if review is not None:
            trace["review_ref"] = review["review_id"]

        persisted = True
        try:
            for aid in order:
                self.storage.put_artifact(f"{tid}/{aid}", artifacts[aid])
            if request_info.get("source") == "upload":
                self.storage.put_artifact(f"{tid}/input", _encode_jpeg(scene, 95))
            self.storage.put_trace(trace)
            if review is not None:
                self.storage.put_review(review)
                self.counters.record_review()
        except (StorageError, OSError, ValueError) as exc:
            persisted = False
            notes.append(f"storage error, results were not persisted: {exc}")
            log_event("persist_failed", logging.ERROR, trace_id=tid, error=str(exc))

        response: dict[str, Any] = {
            "trace": trace,
            "artifacts": {},
            "artifact_urls": {aid: _artifact_url(tid, aid) for aid in order} if persisted else {},
            "review": review,
            "persisted": persisted,
            "notes": notes,
        }
        base_len = len(json.dumps(response, default=str))
        budget = RESPONSE_BUDGET_BYTES - base_len - 65_536
        data_urls, art_notes = encode_artifacts(artifacts, order, budget)
        response["artifacts"] = data_urls
        notes.extend(art_notes)

        self.counters.record_inspection(trace, request_info.get("preset", "?"), planner_name, wall_ms)
        verdict = trace.get("verdict") or {}
        log_event(
            "inspection",
            trace_id=tid,
            part_id=trace.get("part_id"),
            category=trace.get("category"),
            preset=request_info.get("preset"),
            planner=planner_name,
            label=verdict.get("label"),
            action=verdict.get("action"),
            prob=verdict.get("prob_anomalous"),
            n_steps=(trace.get("totals") or {}).get("n_steps"),
            wall_ms=round(wall_ms, 1),
            review_id=review["review_id"] if review else None,
            persisted=persisted,
        )
        return response

    # -- traces -------------------------------------------------------------------------------
    def get_trace(self, trace_id: str) -> dict:
        try:
            validate_id(trace_id, "trace id")
        except StorageError as exc:
            raise ApiError(404, str(exc)) from exc
        trace = self.storage.get_trace(trace_id)
        if trace is None:
            raise ApiError(404, f"trace {trace_id!r} not found")
        arts: list[str] = []
        for step in trace.get("steps") or []:
            for aid in step.get("artifacts") or []:
                if aid not in arts:
                    arts.append(aid)
        trace["artifact_urls"] = {aid: _artifact_url(trace_id, aid) for aid in arts}
        return trace

    def get_artifact(self, trace_id: str, art_id: str) -> bytes:
        try:
            data = self.storage.get_artifact(f"{validate_id(trace_id)}/{validate_id(art_id)}")
        except StorageError as exc:
            if isinstance(exc, NotFoundError) or "invalid" in str(exc):
                raise ApiError(404, str(exc)) from exc
            raise
        if data is None:
            raise ApiError(404, f"artifact {trace_id}/{art_id} not found")
        return data

    # -- reviews ------------------------------------------------------------------------------
    def list_reviews(self, status: str = "pending", limit: int = 100) -> dict:
        if status != "all" and status not in REVIEW_STATUSES:
            raise ApiError(422, f"unknown status {status!r}; expected one of {[*REVIEW_STATUSES, 'all']}")
        items = self.storage.list_reviews(status=None if status == "all" else status, limit=limit)
        for item in items:
            item["audit_valid"] = verify_audit(item)
        return {"reviews": items, "count": len(items), "status": status}

    def _rescan_inputs(self, review: dict, new_seed: int) -> tuple[PartSpec, np.ndarray, dict]:
        req = dict(review.get("request") or {})
        src = req.get("source")
        if src == "sample":
            sample = self.catalog.get(str(req.get("sample_id")))
            if sample is None:
                raise ApiError(409, f"sample {req.get('sample_id')!r} is no longer available for a rescan")
            part, scene = self._sample_part(sample, "clean", new_seed)
        elif src == "upload":
            data = self.storage.get_artifact(f"{review['trace_id']}/input")
            scene = _decode_image(data) if data else None
            if scene is None:
                raise ApiError(409, "the uploaded input image is no longer stored; cannot rescan")
            part = PartSpec(
                part_id=str(review.get("part_id") or f"upload-{req.get('sha256', '')[:12]}"),
                category=str(req.get("category") or review.get("category")),
                image_path=f"upload:{req.get('filename', 'upload')}",
                label=None,
                conditions=self._conditions("clean", new_seed),
            )
        else:
            raise ApiError(409, "review has no reproducible request; cannot rescan")
        req.update({"preset": "clean", "seed": new_seed, "rescan_of": review["trace_id"], "rescan_review_id": review["review_id"]})
        return part, scene, req

    def decide(self, review_id: str, decision: str, reviewer: str = "operator", note: str = "") -> dict:
        try:
            validate_id(review_id, "review id")
        except StorageError as exc:
            raise ApiError(404, str(exc)) from exc
        if decision not in DECISIONS:
            raise ApiError(422, f"unknown decision {decision!r}; expected one of {list(DECISIONS)}")
        review = self.storage.get_review(review_id)
        if review is None:
            raise ApiError(404, f"review {review_id!r} not found")
        if review.get("status") != "pending":
            raise ApiError(409, f"review {review_id!r} is already {review.get('status')}")

        rescan: dict | None = None
        if decision == "approve_fail":
            action, final_label, new_status = "route_scrap", "FAIL", "resolved"
        elif decision == "override_pass":
            action, final_label, new_status = "route_pass", "PASS", "resolved"
        else:
            req = review.get("request") or {}
            new_seed = (int(req.get("seed", 0)) + 1) % (_SEED_MAX + 1)
            part, scene, req_info = self._rescan_inputs(review, new_seed)
            rescan = self._execute(
                part=part,
                scene=scene,
                request_info=req_info,
                planner_name=str(req.get("planner") or "rule"),
                seed=new_seed,
                links={"rescan_of": review["trace_id"], "review_id": review_id},
                parent_review_id=review_id,
            )
            verdict = rescan["trace"].get("verdict") or {}
            action = str(verdict.get("action") or "hold_for_review")
            final_label = verdict.get("label")
            new_status = "rescanned"

        now = utc_now()
        decision_rec = {"decision": decision, "reviewer": reviewer, "note": note, "ts": now}
        patch = {
            "status": new_status,
            "decision": decision_rec,
            "final_action": action,
            "final_label": final_label,
            "rescan_trace_id": rescan["trace"]["trace_id"] if rescan else None,
        }
        event = {
            "event": "decision",
            "actor": reviewer,
            "decision": decision,
            "note": note,
            "action": action,
            "final_label": final_label,
            "previous_status": "pending",
            "new_status": new_status,
            "rescan_trace_id": patch["rescan_trace_id"],
            "ts": now,
        }
        try:
            updated = self.storage.update_review(review_id, patch, audit_event=event, expect_status="pending")
        except NotFoundError as exc:
            raise ApiError(404, str(exc)) from exc
        except ConflictError as exc:
            raise ApiError(409, str(exc)) from exc

        try:
            original = self.storage.get_trace(review["trace_id"])
            if original is not None:
                original["human_decision"] = {"review_id": review_id, **decision_rec, "action": action}
                if rescan:
                    links = original.setdefault("links", {})
                    links.setdefault("rescanned_by", [])
                    links["rescanned_by"].append(rescan["trace"]["trace_id"])
                self.storage.put_trace(original)
        except StorageError as exc:
            log_event("trace_link_failed", logging.WARNING, trace_id=review.get("trace_id"), error=str(exc))

        updated["audit_valid"] = verify_audit(updated)
        self.counters.record_decision(decision)
        log_event(
            "review_decision",
            review_id=review_id,
            trace_id=review.get("trace_id"),
            decision=decision,
            reviewer=reviewer,
            action=action,
            rescan_trace_id=patch["rescan_trace_id"],
        )
        return {"review": updated, "action": action, "final_label": final_label, "rescan": rescan}

    # -- metrics / figures / system / bench -----------------------------------------------------
    def _eval_summary(self) -> tuple[dict | None, str | None]:
        path = Path(self.settings.results_dir) / "eval_summary.json"
        try:
            st = path.stat()
        except FileNotFoundError:
            return None, None
        sig = (st.st_mtime, st.st_size)
        if self._eval_cache is None or self._eval_cache[0] != sig:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                log_event("eval_summary_unreadable", logging.WARNING, error=str(exc))
                data = None
            self._eval_cache = (sig, data)
        mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
        return self._eval_cache[1], mtime

    def _figures(self) -> list[dict]:
        fig_dir = Path(self.settings.results_dir) / "figures"
        if not fig_dir.is_dir():
            return []
        out = []
        for p in sorted(fig_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in FIGURE_EXTS:
                title = p.stem.replace("_", " ").replace("-", " ").strip().capitalize()
                out.append({"name": p.name, "title": title, "url": f"/api/figures/{quote(p.name, safe='')}"})
        return out

    def figure(self, name: str) -> tuple[bytes, str]:
        for fig in self._figures():
            if fig["name"] == name:
                path = Path(self.settings.results_dir) / "figures" / name
                return path.read_bytes(), FIGURE_EXTS[path.suffix.lower()]
        raise ApiError(404, f"figure {name!r} not found")

    def metrics(self) -> dict:
        summary, generated_at = self._eval_summary()
        store: dict[str, Any] = {}
        try:
            pending = self.storage.list_reviews(status="pending", limit=500)
            recent = self.storage.list_traces(limit=200)
            store = {
                "pending_reviews": len(pending),
                "recent_traces": len(recent),
                "recent_by_label": dict(Counter(str(t.get("label")) for t in recent)),
                "recent_by_action": dict(Counter(str(t.get("action")) for t in recent)),
            }
        except StorageError as exc:
            store = {"error": str(exc)}
        raw_ok = summary is not None and len(json.dumps(summary, default=str)) <= 300_000
        return {
            "eval": {
                "available": summary is not None,
                "source": "results/eval_summary.json",
                "generated_at": generated_at,
                "tables": eval_tables(summary) if summary is not None else [],
                "summary": summary if raw_ok else None,
            },
            "figures": self._figures(),
            "live": self.counters.snapshot(),
            "store": store,
        }

    def system(self) -> dict:
        s = self.settings
        env = os.environ
        storage_desc = {k: v for k, v in self.storage.describe().items() if k in ("kind", "bucket", "table", "region")}
        cats = available_categories(s.models_dir)
        return {
            "version": __version__,
            "cv": cv_info(),
            "runtime": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(terse=True),
                "machine": platform.machine(),
                "cpu_count": os.cpu_count(),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
            },
            "deployment": {
                "lambda": bool(env.get("AWS_LAMBDA_FUNCTION_NAME")),
                "function_name": env.get("AWS_LAMBDA_FUNCTION_NAME"),
                "function_version": env.get("AWS_LAMBDA_FUNCTION_VERSION"),
                "memory_mb": env.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE"),
                "region": env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or s.aws_region,
                "execution_env": env.get("AWS_EXECUTION_ENV"),
                "arch": platform.machine(),
                "graviton": platform.machine().lower() in ("aarch64", "arm64"),
                "deploy_arch": env.get("INSPECTAGENT_DEPLOY_ARCH"),
                "stack_name": env.get("INSPECTAGENT_STACK_NAME"),
                "image_tag": env.get("INSPECTAGENT_IMAGE_TAG"),
                "git_sha": env.get("INSPECTAGENT_GIT_SHA"),
            },
            "settings": {
                "storage": storage_desc,
                "planner_default": s.planner_default,
                "bedrock_model_id": s.bedrock_model_id or None,
                "bedrock_configured": bool(s.bedrock_model_id),
                "cv_threads": s.cv_threads,
                "log_level": s.log_level,
                "camera": dataclasses.asdict(s.camera),
                "quality": dataclasses.asdict(s.quality),
                "detector": dataclasses.asdict(s.detector),
                "policy": dataclasses.asdict(s.policy),
            },
            "models": {"available_categories": cats, "known_categories": list(CATEGORIES)},
            "presets": list(presets()),
            "planners": [
                {"name": "rule", "available": True},
                {"name": "bedrock", "available": bool(s.bedrock_model_id), "model_id": s.bedrock_model_id or None},
            ],
            "samples": len(self.catalog.list()),
            "limits": {
                "max_upload_bytes": MAX_UPLOAD_BYTES,
                "artifact_max_side": ARTIFACT_MAX_SIDE,
                "response_budget_bytes": RESPONSE_BUDGET_BYTES,
            },
            "started_at": self.counters.started_at,
        }

    def bench(self, n: int) -> dict:
        try:
            mod = importlib.import_module("scripts.bench_cv")
        except ImportError:
            if str(REPO_ROOT) not in sys.path:
                sys.path.append(str(REPO_ROOT))
            try:
                mod = importlib.import_module("scripts.bench_cv")
            except ImportError as exc:
                raise ApiError(501, f"benchmark module unavailable: {exc}") from exc
        run_bench = getattr(mod, "run_bench", None)
        if not callable(run_bench):
            raise ApiError(501, "scripts.bench_cv.run_bench is not available")
        t0 = time.perf_counter()
        with self._run_lock:
            try:
                result = run_bench(n=n)
            except Exception as exc:
                log_event("bench_failed", logging.ERROR, error=f"{type(exc).__name__}: {exc}")
                raise ApiError(500, f"benchmark failed: {type(exc).__name__}: {exc}") from exc
        wall = (time.perf_counter() - t0) * 1000.0
        log_event("bench", n=n, wall_ms=round(wall, 1), arch=platform.machine())
        return {"n": n, "wall_ms": round(wall, 1), "arch": platform.machine(), "cv": cv_info(), "result": _jsonable(result)}


# ------------------------------------------------------------------------------------------ app factory


def _no_store(resp: Response) -> Response:
    resp.headers["Cache-Control"] = "no-store"
    return resp


def create_app(settings: Settings | None = None, storage: Any | None = None) -> FastAPI:
    """Build the FastAPI application (one independent service/storage per app instance)."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    storage = storage if storage is not None else make_storage(settings)
    service = ApiService(settings, storage)

    app = FastAPI(
        title="InspectAgent",
        version=__version__,
        description="Agentic visual inspection with OpenCV 5 on AWS Graviton",
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.state.storage = storage
    app.state.service = service
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False)

    @app.middleware("http")
    async def _timing(request: Request, call_next: Any) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000.0
        response.headers["X-Response-Time-ms"] = f"{ms:.1f}"
        path = request.url.path
        if path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"  # revalidate (ETag) so UI updates ship immediately
        level = logging.DEBUG if path in ("/healthz",) or path.startswith("/static/") else logging.INFO
        log_event("http", level, method=request.method, path=path, status=response.status_code, ms=round(ms, 1))
        return response

    @app.exception_handler(NotFoundError)
    async def _not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ConflictError)
    async def _conflict(_: Request, exc: ConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(StorageError)
    async def _storage(_: Request, exc: StorageError) -> JSONResponse:
        log_event("storage_error", logging.ERROR, error=str(exc))
        return JSONResponse(status_code=503, content={"detail": f"storage unavailable: {exc}"})

    # -- health / samples ---------------------------------------------------------------------
    @app.get("/healthz", tags=["system"])
    def healthz() -> dict:
        return {"ok": True, "version": __version__, "cv": cv_info()}

    @app.get("/api/samples", tags=["samples"])
    def samples() -> dict:
        return service.list_samples()

    @app.get("/api/samples/{sample_id}/image", tags=["samples"])
    def sample_image(sample_id: str, full: bool = False) -> Response:
        data = service.sample_image(sample_id, full)
        return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})

    # -- inspect ------------------------------------------------------------------------------
    inspect_schema = InspectRequest.model_json_schema()

    @app.post(
        "/api/inspect",
        tags=["inspection"],
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {"schema": inspect_schema},
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["file", "category"],
                            "properties": {
                                "file": {"type": "string", "format": "binary"},
                                "category": {"type": "string"},
                                "preset": {"type": "string", "default": "nominal"},
                                "planner": {"type": "string", "enum": list(PLANNERS), "default": "rule"},
                                "seed": {"type": "integer", "default": 0},
                            },
                        }
                    },
                },
            }
        },
    )
    async def inspect(request: Request) -> JSONResponse:
        ctype = request.headers.get("content-type", "").lower()
        if ctype.startswith("multipart/form-data"):
            try:
                clen = int(request.headers.get("content-length") or 0)
            except ValueError:
                clen = 0
            if clen > MAX_UPLOAD_BYTES + 512 * 1024:
                raise ApiError(413, f"request too large: {clen} bytes (image limit {MAX_UPLOAD_BYTES})")
            form = await request.form(max_files=1, max_fields=8)
            try:
                upload = form.get("file")
                if upload is None or not hasattr(upload, "read"):
                    raise ApiError(422, "multipart upload requires a 'file' field")
                data = await upload.read(MAX_UPLOAD_BYTES + 1)
                if len(data) > MAX_UPLOAD_BYTES:
                    raise ApiError(413, f"image too large (limit {MAX_UPLOAD_BYTES} bytes)")
                ct = (getattr(upload, "content_type", None) or "").lower()
                if ct and ct not in ("image/jpeg", "image/jpg", "image/pjpeg", "image/png", "application/octet-stream"):
                    raise ApiError(415, f"unsupported content type {ct!r}: only JPEG and PNG are accepted")
                category = str(form.get("category") or "").strip()
                if not category:
                    raise ApiError(422, "multipart upload requires a 'category' field")
                preset = str(form.get("preset") or "nominal").strip()
                planner = str(form.get("planner") or "rule").strip()
                try:
                    seed = int(str(form.get("seed") or "0").strip())
                except ValueError as exc:
                    raise ApiError(422, "seed must be an integer") from exc
                if not 0 <= seed <= _SEED_MAX:
                    raise ApiError(422, f"seed must be in [0, {_SEED_MAX}]")
                filename = getattr(upload, "filename", None) or "upload"
            finally:
                await form.close()
            result = await run_in_threadpool(
                service.inspect_upload, data, filename, category, preset, planner, seed
            )
            return _no_store(JSONResponse(result))
        if ctype.startswith("application/json") or not ctype:
            try:
                body = await request.json()
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                raise ApiError(422, f"invalid JSON body: {exc}") from exc
            try:
                req = InspectRequest.model_validate(body)
            except ValidationError as exc:
                raise ApiError(422, json.loads(exc.json(include_url=False, include_context=False))) from exc
            result = await run_in_threadpool(service.inspect_sample, req.sample_id, req.preset, req.planner, req.seed)
            return _no_store(JSONResponse(result))
        raise ApiError(415, "use application/json or multipart/form-data")

    # -- traces -------------------------------------------------------------------------------
    @app.get("/api/traces", tags=["traces"])
    def traces(limit: int = Query(20, ge=1, le=200)) -> dict:
        items = storage.list_traces(limit=limit)
        return {"traces": items, "count": len(items)}

    @app.get("/api/traces/{trace_id}", tags=["traces"])
    def trace(trace_id: str) -> dict:
        return service.get_trace(trace_id)

    @app.get("/api/traces/{trace_id}/artifacts/{art_id}", tags=["traces"])
    def artifact(trace_id: str, art_id: str) -> Response:
        data = service.get_artifact(trace_id, art_id)
        return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    # -- reviews ------------------------------------------------------------------------------
    @app.get("/api/reviews", tags=["reviews"])
    def reviews(status: str = "pending", limit: int = Query(100, ge=1, le=500)) -> dict:
        return service.list_reviews(status=status, limit=limit)

    @app.post("/api/reviews/{review_id}/decision", tags=["reviews"])
    def decision(review_id: str, body: DecisionRequest) -> dict:
        return service.decide(review_id, body.decision, body.reviewer.strip() or "operator", body.note)

    # -- metrics / system / bench ---------------------------------------------------------------
    @app.get("/api/metrics", tags=["system"])
    def metrics() -> dict:
        return service.metrics()

    @app.get("/api/figures/{name}", tags=["system"])
    def figure(name: str) -> Response:
        data, media = service.figure(name)
        return Response(content=data, media_type=media, headers={"Cache-Control": "public, max-age=600"})

    @app.get("/api/system", tags=["system"])
    def system() -> dict:
        return service.system()

    @app.get("/api/bench", tags=["system"])
    def bench(n: int = Query(10, ge=1, le=100)) -> dict:
        return service.bench(n)

    # -- static UI ----------------------------------------------------------------------------
    app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> Response:
        page = STATIC_DIR / "index.html"
        if not page.is_file():
            return JSONResponse(status_code=404, content={"detail": "UI not bundled"})
        return FileResponse(page, media_type="text/html", headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
