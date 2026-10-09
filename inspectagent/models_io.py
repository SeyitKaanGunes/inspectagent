"""Per-category model bundles on disk: ``models/<category>/{stat.npz, deep.npz, meta.json}``.

The deep detector's ONNX backbone (``models/backbone_resnet18.onnx``) is shared by all categories.
Vision classes are imported lazily so this module (and the agent layer) can be imported even when
only part of the vision stack is installed.
"""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from inspectagent.config import CATEGORIES, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from inspectagent.vision.anomaly_deep import PatchCoreDNN
    from inspectagent.vision.anomaly_stat import StatTemplateModel

log = logging.getLogger(__name__)

STAT_FILE = "stat.npz"
DEEP_FILE = "deep.npz"
META_FILE = "meta.json"
BACKBONE_FILE = "backbone_resnet18.onnx"


@dataclass
class CategoryBundle:
    """Everything the agent needs for one product category."""

    category: str
    stat: "StatTemplateModel"
    deep: "PatchCoreDNN | None" = None
    meta: dict = field(default_factory=dict)
    reference_sharpness: dict[str, float] = field(default_factory=dict)  # per zoom bucket "z1".."z4"

    @property
    def has_deep(self) -> bool:
        return self.deep is not None


def _models_root(models_dir: str | Path | None) -> Path:
    return Path(models_dir) if models_dir is not None else Path(get_settings().models_dir)


def bundle_dir(category: str, models_dir: str | Path | None = None) -> Path:
    """Directory holding the bundle of ``category`` (``models/<category>/``)."""
    return _models_root(models_dir) / category


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def save_bundle(b: CategoryBundle, models_dir: str | Path | None = None) -> None:
    """Write ``stat.npz`` (+ ``deep.npz`` when present) and ``meta.json``; invalidates the load cache."""
    d = bundle_dir(b.category, models_dir)
    d.mkdir(parents=True, exist_ok=True)
    b.stat.save(d / STAT_FILE)
    if b.deep is not None:
        b.deep.save(d / DEEP_FILE)
    meta = dict(b.meta)
    meta.update(
        {
            "category": b.category,
            "reference_sharpness": {k: float(v) for k, v in b.reference_sharpness.items()},
            "has_deep": b.deep is not None,
        }
    )
    meta.setdefault("backbone", BACKBONE_FILE)
    (d / META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    clear_bundle_cache()


def _load_stat(path: Path) -> "StatTemplateModel":
    from inspectagent.vision.anomaly_stat import StatTemplateModel

    return StatTemplateModel.load(path)


def _load_deep(path: Path, onnx_path: Path) -> "PatchCoreDNN":
    from inspectagent.vision.anomaly_deep import PatchCoreDNN

    return PatchCoreDNN.load(path, onnx_path)


def _find_backbone(root: Path, category_dir: Path, name: str) -> Path | None:
    for cand in (root / name, category_dir / name):
        if cand.is_file():
            return cand
    return None


@functools.lru_cache(maxsize=16)
def _load_bundle_cached(category: str, models_dir: str, with_deep: bool) -> CategoryBundle:
    root = Path(models_dir)
    d = root / category
    stat_path = d / STAT_FILE
    if not stat_path.is_file():
        raise FileNotFoundError(
            f"no model bundle for category {category!r}: {stat_path} is missing (run scripts/build_models.py)"
        )
    meta: dict = {}
    meta_path = d / META_FILE
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ref = {str(k): float(v) for k, v in (meta.get("reference_sharpness") or {}).items()}
    stat = _load_stat(stat_path)
    deep = None
    if with_deep:
        deep_path = d / DEEP_FILE
        onnx = _find_backbone(root, d, str(meta.get("backbone", BACKBONE_FILE)))
        if not deep_path.is_file():
            meta["deep_load_status"] = "missing deep.npz"
        elif onnx is None:
            meta["deep_load_status"] = f"missing backbone {meta.get('backbone', BACKBONE_FILE)}"
        else:
            try:
                deep = _load_deep(deep_path, onnx)
                meta["deep_load_status"] = "loaded"
            except Exception as exc:  # corrupt file / incompatible OpenCV build: run without the deep detector
                log.warning("deep model for %s could not be loaded: %s", category, exc)
                meta["deep_load_status"] = f"load error: {type(exc).__name__}: {exc}"
    else:
        meta["deep_load_status"] = "disabled"
    return CategoryBundle(category=category, stat=stat, deep=deep, meta=meta, reference_sharpness=ref)


def load_bundle(category: str, models_dir: str | Path | None = None, with_deep: bool = True) -> CategoryBundle:
    """Load (and cache per process) the bundle of ``category``.

    The cache key is ``(category, resolved models_dir, with_deep)``. Bundles are shared between
    sessions and must be treated as read-only.
    """
    root = str(_models_root(models_dir).resolve())
    return _load_bundle_cached(category, root, bool(with_deep))


def clear_bundle_cache() -> None:
    """Forget cached bundles (after re-building models or in tests)."""
    _load_bundle_cached.cache_clear()


load_bundle.cache_clear = clear_bundle_cache  # type: ignore[attr-defined]  # lru_cache-compatible API


def available_categories(models_dir: str | Path | None = None) -> list[str]:
    """Categories with a ``stat.npz`` bundle, in ``config.CATEGORIES`` order first, then alphabetical."""
    root = _models_root(models_dir)
    if not root.is_dir():
        return []
    found = {p.name for p in root.iterdir() if p.is_dir() and (p / STAT_FILE).is_file()}
    ordered = [c for c in CATEGORIES if c in found]
    return ordered + sorted(found - set(ordered))
