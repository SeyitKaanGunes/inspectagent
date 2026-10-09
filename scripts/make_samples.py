"""Curate demo parts (VisA images, CC BY 4.0) into ``samples/`` for the UI, API and MCP server.

Usage (after ``scripts/prepare_visa.py``)::

    python scripts/make_samples.py [--categories pcb1 cashew] [--per-category-normal 2 --per-category-anomaly 2]

For each category, picks parts from the held-out **eval** split (never from the data
used to fit or calibrate models):

* 2 normal parts (seeded random choice), and
* 2 anomalous parts whose defects are *clearly visible but not trivial*: defect area
  between the 25th and 90th percentile of the category (and at least 0.02 % of the
  image), ranked by a local-contrast visibility score; the chosen ones sit near the
  70th visibility percentile, with distinct defect types when possible.

Writes full-resolution JPEG (quality 92) to ``samples/<category>/<id>.jpg``, binary
masks (0/255 PNG) to ``samples/<category>/<id>_mask.png``, ``samples/samples.json``
and ``samples/ATTRIBUTION.md``.

``samples.json`` is a list of objects::

    {"id", "category", "label" (0 normal | 1 anomalous), "label_name", "image", "mask"
     (paths relative to samples/, forward slashes; mask null for normals),
     "suggested_preset", "note", "defect_type", "defect_area_frac", "width", "height",
     "source" (VisA-relative path), "split": "eval"}
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inspectagent.config import CATEGORIES  # noqa: E402
from inspectagent.eval_utils import part_id_from_relpath, stable_seed  # noqa: E402

try:  # unicode-safe I/O (the project path contains a space / Turkish locale)
    from inspectagent.vision.imageio import imread, imwrite  # noqa: E402
except ImportError:  # pragma: no cover - fallback while modules are being integrated

    def imread(path, flags=cv2.IMREAD_COLOR):  # type: ignore[no-redef]
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), flags)
        if img is None:
            raise ValueError(f"cannot decode {path}")
        return img

    def imwrite(path, img, params=None):  # type: ignore[no-redef]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(Path(path).suffix or ".png", img, params or [])
        if not ok:
            raise ValueError(f"cannot encode {path}")
        buf.tofile(str(path))


JPEG_QUALITY = 92
DEGRADED_CYCLE = ("glare", "dark", "defocus", "bright")
PRESET_HINT = {
    "nominal": "nominal line conditions",
    "glare": "specular glare under ring light - the agent should switch to diffuse light",
    "dark": "under-exposed scene - the agent should raise exposure before detecting",
    "defocus": "out-of-focus camera - the agent should trigger autofocus",
    "bright": "over-exposed scene - the agent should lower exposure",
}

ATTRIBUTION = """# Sample images - attribution

The images and masks in this folder are a small curated subset of the **VisA (Visual
Anomaly) dataset** by Amazon, released under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

* Dataset: VisA, `VisA_20220922.tar` - <https://github.com/amazon-science/spot-diff>
* Paper: Y. Zou, J. Jeong, L. Pemula, D. Zhang, O. Dabeer. *SPot-the-Difference
  Self-Supervised Pre-training for Anomaly Detection and Segmentation.* ECCV 2022.
* Copyright (c) Amazon.com, Inc. or its affiliates; licensed under CC BY 4.0.

Changes made by InspectAgent: images were selected from the held-out evaluation split
(`data/splits.json`), re-encoded as JPEG (quality {q}) at full resolution, and the
ground-truth masks were binarized to 0/255 PNG. File names were changed to
`<category>_<normal|anomaly>_<original stem>`. The original VisA path of every file is
recorded in `samples.json` (`source`).

The same attribution applies to the extracted dataset under `data/visa/` (not
redistributed with this repository) and to any figure or trace derived from it.
"""


def _log(msg: str) -> None:
    print(msg, flush=True)


def load_defect_types(data_root: Path, category: str) -> dict[str, str]:
    """image path -> defect type text from VisA ``<category>/image_anno.csv`` (empty if missing)."""
    path = data_root / category / "image_anno.csv"
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            img = (row.get("image") or "").strip().replace("\\", "/")
            lab = (row.get("label") or "").strip()
            if img and lab and lab.lower() != "normal":
                out[img] = lab
    return out


def defect_stats(img_bgr: np.ndarray, mask: np.ndarray) -> dict:
    """Area fraction, connected components and a local-contrast visibility score of a defect."""
    m = (mask > 0).astype(np.uint8)
    area = int(m.sum())
    h, w = m.shape[:2]
    if area == 0:
        return {"area_frac": 0.0, "components": 0, "visibility": 0.0}
    n_cc, _ = cv2.connectedComponents(m)
    k = max(7, int(0.02 * float(np.hypot(h, w))) | 1)
    ring = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) & (1 - m)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    inside = lab[m > 0]
    around = lab[ring > 0] if ring.any() else lab.reshape(-1, 3)
    mu = around.mean(axis=0)
    d_in = np.linalg.norm(inside - mu, axis=1).mean()
    d_ring = np.linalg.norm(around - mu, axis=1).mean()
    return {"area_frac": area / float(h * w), "components": int(n_cc - 1), "visibility": float(d_in / (d_ring + 1.0))}


def choose_anomalies(cands: list[dict], n: int, rng: np.random.Generator) -> list[dict]:
    """Pick ``n`` clearly-visible-but-not-trivial anomalies (see module docstring)."""
    if len(cands) <= n:
        return list(cands)
    areas = np.array([c["area_frac"] for c in cands])
    lo, hi = np.percentile(areas, 25), np.percentile(areas, 90)
    pool = [c for c in cands if lo <= c["area_frac"] <= hi and c["area_frac"] >= 2e-4]
    if len(pool) < n:
        pool = [c for c in cands if c["area_frac"] > 0] or list(cands)
    vis = np.array([c["visibility"] for c in pool])
    order = np.argsort(np.argsort(vis, kind="mergesort"), kind="mergesort")
    pct = order / max(1, len(pool) - 1)
    jitter = rng.uniform(0, 1e-6, len(pool))  # deterministic tie-break
    ranked = [pool[i] for i in np.argsort(np.abs(pct - 0.7) + jitter, kind="mergesort")]
    chosen: list[dict] = []
    for c in ranked:  # prefer distinct defect types
        if len(chosen) >= n:
            break
        if all(c["defect_type"] != x["defect_type"] for x in chosen):
            chosen.append(c)
    for c in ranked:
        if len(chosen) >= n:
            break
        if c not in chosen:
            chosen.append(c)
    return chosen


def _note(category: str, label: int, preset: str, defect_type: str, area_frac: float) -> str:
    hint = PRESET_HINT.get(preset, preset)
    if label == 0:
        return f"Defect-free VisA {category} part. Suggested preset '{preset}': {hint}."
    kind = " + ".join(t.strip() for t in defect_type.split(",") if t.strip()) if defect_type else "visible"
    return (
        f"VisA {category} part with a {kind} defect covering {100 * area_frac:.2f} % of the image. "
        f"Suggested preset '{preset}': {hint}."
    )


def curate_category(
    category: str, ci: int, split: dict, data_root: Path, out_dir: Path, n_normal: int, n_anomaly: int, seed: int
) -> list[dict]:
    """Select, copy and describe the demo parts of one category."""
    rng = np.random.default_rng(stable_seed(seed, category, "samples"))
    types = load_defect_types(data_root, category)
    normals = sorted((d for d in split["eval"] if int(d["label"]) == 0), key=lambda d: d["image"])
    anomalies = sorted((d for d in split["eval"] if int(d["label"]) == 1 and d.get("mask")), key=lambda d: d["image"])

    picked_normals = [normals[i] for i in sorted(rng.choice(len(normals), size=min(n_normal, len(normals)), replace=False))]
    cands = []
    for d in anomalies:
        img = imread(data_root / d["image"], cv2.IMREAD_REDUCED_COLOR_2)
        mask = imread(data_root / d["mask"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        st = defect_stats(img, mask)
        cands.append({**d, **st, "defect_type": types.get(d["image"], "")})
    picked_anoms = choose_anomalies(cands, n_anomaly, rng)

    cat_dir = out_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    degraded = DEGRADED_CYCLE[ci % len(DEGRADED_CYCLE)]
    degraded_anom = DEGRADED_CYCLE[(ci + 1) % len(DEGRADED_CYCLE)]
    for j, d in enumerate(picked_normals):
        entries.append(_write_entry(category, d, 0, "nominal" if j == 0 else degraded, data_root, out_dir, {}))
    for j, c in enumerate(picked_anoms):
        entries.append(_write_entry(category, c, 1, "nominal" if j == 0 else degraded_anom, data_root, out_dir, c))

    keep = {Path(e["image"]).name for e in entries} | {Path(e["mask"]).name for e in entries if e["mask"]}
    pattern = re.compile(rf"^{re.escape(category)}_(normal|anomaly)_.+\.(jpg|png)$")
    for f in cat_dir.iterdir():  # drop stale outputs of earlier runs (only files this script names)
        if f.is_file() and pattern.match(f.name) and f.name not in keep:
            f.unlink()
    return entries


def _write_entry(category: str, d: dict, label: int, preset: str, data_root: Path, out_dir: Path, stats: dict) -> dict:
    sid = part_id_from_relpath(d["image"])
    img = imread(data_root / d["image"], cv2.IMREAD_COLOR)
    rel_img = f"{category}/{sid}.jpg"
    imwrite(out_dir / rel_img, img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    rel_mask = None
    area_frac = 0.0
    if label == 1 and d.get("mask"):
        mask = imread(data_root / d["mask"], cv2.IMREAD_GRAYSCALE)
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        binm = np.where(mask > 0, 255, 0).astype(np.uint8)
        area_frac = float((binm > 0).mean())
        rel_mask = f"{category}/{sid}_mask.png"
        imwrite(out_dir / rel_mask, binm)
    defect_type = stats.get("defect_type", "") if label == 1 else ""
    return {
        "id": sid,
        "category": category,
        "label": label,
        "label_name": "anomaly" if label == 1 else "normal",
        "image": rel_img,
        "mask": rel_mask,
        "suggested_preset": preset,
        "note": _note(category, label, preset, defect_type, area_frac),
        "defect_type": defect_type or None,
        "defect_area_frac": round(area_frac, 6),
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "source": d["image"],
        "split": "eval",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=Path, default=REPO_ROOT / "data" / "splits.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "visa")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "samples")
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    ap.add_argument("--per-category-normal", type=int, default=2)
    ap.add_argument("--per-category-anomaly", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if not args.splits.is_file():
        raise SystemExit(f"{args.splits} not found - run scripts/prepare_visa.py first")
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    for ci, cat in enumerate(CATEGORIES):
        if cat not in args.categories:
            continue
        if cat not in splits:
            _log(f"[warn] {cat} not in {args.splits}; skipped")
            continue
        es = curate_category(
            cat, ci, splits[cat], args.data_root, args.out, args.per_category_normal, args.per_category_anomaly, args.seed
        )
        entries.extend(es)
        for e in es:
            _log(f"[{cat}] {e['id']:<24} label={e['label']} preset={e['suggested_preset']:<8} {e['defect_type'] or ''}")
    # keep entries of categories not regenerated in this run
    old_path = args.out / "samples.json"
    if old_path.is_file():
        try:
            old = json.loads(old_path.read_text(encoding="utf-8"))
            done = {e["category"] for e in entries}
            entries = [e for e in old if e.get("category") not in done] + entries
        except (json.JSONDecodeError, OSError, TypeError, KeyError):
            pass
    order = {c: i for i, c in enumerate(CATEGORIES)}
    entries.sort(key=lambda e: (order.get(e["category"], 99), -int(e["label"] == 0), e["id"]))
    old_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    (args.out / "ATTRIBUTION.md").write_text(ATTRIBUTION.format(q=JPEG_QUALITY), encoding="utf-8")
    _log(f"[samples] wrote {len(entries)} entries to {old_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
