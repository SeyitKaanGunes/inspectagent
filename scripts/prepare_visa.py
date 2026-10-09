"""Extract the VisA categories used by InspectAgent and build the deterministic data split.

Usage (from the repository root)::

    python scripts/prepare_visa.py                 # extract + split (tar must be complete)
    python scripts/prepare_visa.py --wait          # poll the tar size every 60 s until complete
    python scripts/prepare_visa.py --split-only    # re-create data/splits.json from data/visa

* Stream-extracts only ``config.CATEGORIES`` and ``split_csv/1cls.csv`` from
  ``data/raw/VisA_20220922.tar`` into ``data/visa/`` using ``tarfile`` with
  ``filter="data"`` (safe extraction: no absolute paths, no ``..``, no device files,
  no links escaping the destination). The tar itself is opened read-only.
* Builds ``data/splits.json`` (seed 0, SPEC section 7): per category
  ``fit`` / ``val_normal`` (VisA 1cls *train* normals, 90 % / 10 %) and
  ``calib`` / ``eval`` (VisA *test*, stratified by label, 40 % / 60 %).
  Paths are relative to ``data/visa`` with forward slashes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inspectagent.config import CATEGORIES  # noqa: E402
from inspectagent.eval_utils import split_fraction, stable_seed, stratified_split  # noqa: E402

EXPECTED_TAR_SIZE = 1_929_840_640
SPLIT_CSV = "split_csv/1cls.csv"
SPLITS_VERSION = 1
FIT_FRAC = 0.9  # of VisA 1cls train normals -> fit (rest -> val_normal)
CALIB_FRAC = 0.4  # of VisA 1cls test (stratified) -> calib (rest -> eval)


def _log(msg: str) -> None:
    print(msg, flush=True)


def wait_for_tar(tar_path: Path, expected: int, wait: bool, poll_s: float = 60.0, timeout_s: float = 6 * 3600) -> None:
    """Return when the tar has the expected size; poll if ``wait`` else raise."""
    t0 = time.monotonic()
    while True:
        size = tar_path.stat().st_size if tar_path.exists() else -1
        if size == expected:
            return
        if not wait:
            raise SystemExit(
                f"{tar_path} has {size} bytes, expected {expected} (still downloading?). Use --wait to poll."
            )
        if time.monotonic() - t0 > timeout_s:
            raise SystemExit(f"timed out waiting for {tar_path} ({size}/{expected} bytes)")
        _log(f"[wait] {size}/{expected} bytes ({100.0 * max(size, 0) / expected:.1f} %), next check in {poll_s:.0f} s")
        time.sleep(poll_s)


def _wanted(name: str, categories: tuple[str, ...]) -> str | None:
    """Normalized relative path if the tar member should be extracted, else None."""
    while name.startswith("./"):
        name = name[2:]
    parts = [x for x in PurePosixPath(name).parts if x not in ("", ".", "/")]
    if not parts:
        return None
    rel = "/".join(parts)
    if rel == SPLIT_CSV:
        return rel
    if parts[0] in categories and len(parts) > 1:
        return rel
    return None


def extract(tar_path: Path, dest: Path, categories: tuple[str, ...], force: bool = False) -> dict:
    """Stream-extract the wanted members (regular files only) with ``filter='data'``."""
    marker = dest / ".extracted.json"
    tar_size = tar_path.stat().st_size
    if marker.exists() and not force:
        try:
            info = json.loads(marker.read_text(encoding="utf-8"))
            if info.get("tar_size") == tar_size and sorted(info.get("categories", [])) == sorted(categories):
                _log(f"[extract] already extracted ({info.get('files')} files) - use --force to redo")
                return info
        except (json.JSONDecodeError, OSError):
            pass
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    n_files = n_skipped = n_bytes = 0
    per_cat: dict[str, int] = {}
    with tarfile.open(tar_path, mode="r|") as tar:
        for member in tar:
            if not member.isfile():
                continue
            rel = _wanted(member.name, categories)
            if rel is None:
                continue
            target = dest / rel
            if target.exists() and target.stat().st_size == member.size and not force:
                n_skipped += 1
            else:
                member.name = rel
                tar.extract(member, path=dest, filter="data")
                n_bytes += member.size
            n_files += 1
            top = rel.split("/", 1)[0]
            per_cat[top] = per_cat.get(top, 0) + 1
            if n_files % 2000 == 0:
                _log(f"[extract] {n_files} files ({n_bytes / 1e6:.0f} MB written, {n_skipped} already present)")
    info = {
        "tar": tar_path.name,
        "tar_size": tar_size,
        "categories": list(categories),
        "files": n_files,
        "skipped_existing": n_skipped,
        "bytes_written": n_bytes,
        "per_top_level": per_cat,
        "seconds": round(time.monotonic() - t0, 1),
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    marker.write_text(json.dumps(info, indent=2), encoding="utf-8")
    _log(f"[extract] done: {n_files} files ({n_skipped} already present) in {info['seconds']} s")
    return info


def read_split_csv(csv_path: Path, categories: tuple[str, ...]) -> list[dict]:
    """Rows of VisA ``1cls.csv`` (object,split,label,image,mask) restricted to ``categories``."""
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"object", "split", "label", "image", "mask"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{csv_path}: missing columns {sorted(missing)}")
        for r in reader:
            obj = (r["object"] or "").strip()
            if obj not in categories:
                continue
            rows.append(
                {
                    "object": obj,
                    "split": (r["split"] or "").strip().lower(),
                    "label": (r["label"] or "").strip().lower(),
                    "image": (r["image"] or "").strip().replace("\\", "/"),
                    "mask": (r["mask"] or "").strip().replace("\\", "/"),
                }
            )
    return rows


def build_splits(rows: list[dict], categories: tuple[str, ...], seed: int = 0) -> dict:
    """Deterministic per-category split (see module docstring)."""
    out: dict = {}
    for cat in categories:
        cat_rows = [r for r in rows if r["object"] == cat]
        train_normals = sorted(r["image"] for r in cat_rows if r["split"] == "train" and r["label"] == "normal")
        test = sorted(
            (
                {
                    "image": r["image"],
                    "label": 1 if r["label"] == "anomaly" else 0,
                    "mask": (r["mask"] or None) if r["label"] == "anomaly" else None,
                }
                for r in cat_rows
                if r["split"] == "test" and r["label"] in ("normal", "anomaly")
            ),
            key=lambda d: d["image"],
        )
        fit, val_normal = split_fraction(train_normals, FIT_FRAC, stable_seed(seed, cat, "fit"))
        calib, ev = stratified_split(test, [d["label"] for d in test], CALIB_FRAC, stable_seed(seed, cat, "calib"))
        out[cat] = {"fit": fit, "val_normal": val_normal, "calib": calib, "eval": ev}
    return out


def verify_files(splits: dict, root: Path) -> dict[str, int]:
    """Count referenced image/mask files missing under ``root`` per category."""
    missing: dict[str, int] = {}
    for cat, s in splits.items():
        if cat.startswith("_"):
            continue
        paths = list(s["fit"]) + list(s["val_normal"])
        for item in list(s["calib"]) + list(s["eval"]):
            paths.append(item["image"])
            if item["mask"]:
                paths.append(item["mask"])
        missing[cat] = sum(1 for p in paths if not (root / p).is_file())
    return missing


def print_counts(splits: dict) -> None:
    """Pretty table of split sizes."""
    header = f"{'category':<10} {'fit':>5} {'val_n':>6} {'calib':>6} {'(n/a)':>9} {'eval':>5} {'(n/a)':>9}"
    _log(header)
    _log("-" * len(header))
    tot = [0, 0, 0, 0, 0, 0, 0, 0]
    for cat, s in splits.items():
        if cat.startswith("_"):
            continue
        cn = sum(1 for d in s["calib"] if d["label"] == 0)
        ca = len(s["calib"]) - cn
        en = sum(1 for d in s["eval"] if d["label"] == 0)
        ea = len(s["eval"]) - en
        vals = [len(s["fit"]), len(s["val_normal"]), len(s["calib"]), cn, ca, len(s["eval"]), en, ea]
        tot = [a + b for a, b in zip(tot, vals)]
        _log(f"{cat:<10} {vals[0]:>5} {vals[1]:>6} {vals[2]:>6} {f'{cn}/{ca}':>9} {vals[5]:>5} {f'{en}/{ea}':>9}")
    _log("-" * len(header))
    _log(f"{'TOTAL':<10} {tot[0]:>5} {tot[1]:>6} {tot[2]:>6} {f'{tot[3]}/{tot[4]}':>9} {tot[5]:>5} {f'{tot[6]}/{tot[7]}':>9}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar", type=Path, default=REPO_ROOT / "data" / "raw" / "VisA_20220922.tar")
    ap.add_argument("--dest", type=Path, default=REPO_ROOT / "data" / "visa")
    ap.add_argument("--splits", type=Path, default=REPO_ROOT / "data" / "splits.json")
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expected-size", type=int, default=EXPECTED_TAR_SIZE)
    ap.add_argument("--wait", action="store_true", help="poll the tar size every --poll seconds until complete")
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--force", action="store_true", help="re-extract even if files exist")
    ap.add_argument("--split-only", action="store_true", help="skip extraction, rebuild splits.json from --dest")
    args = ap.parse_args(argv)

    categories = tuple(args.categories)
    unknown = [c for c in categories if c not in CATEGORIES]
    if unknown:
        _log(f"[warn] categories not in config.CATEGORIES: {unknown}")

    extract_info: dict = {}
    if not args.split_only:
        wait_for_tar(args.tar, args.expected_size, args.wait, args.poll)
        extract_info = extract(args.tar, args.dest, categories, force=args.force)

    csv_path = args.dest / SPLIT_CSV
    if not csv_path.is_file():
        raise SystemExit(f"{csv_path} not found - run without --split-only first")
    rows = read_split_csv(csv_path, categories)
    splits = build_splits(rows, categories, seed=args.seed)
    missing = verify_files(splits, args.dest)
    splits_out = {
        "_meta": {
            "version": SPLITS_VERSION,
            "seed": args.seed,
            "root": "data/visa",
            "source": f"VisA ({args.tar.name}), split_csv/1cls.csv",
            "license": "VisA dataset (Amazon), CC BY 4.0",
            "rule": "fit/val_normal = 90/10 of 1cls train normals; calib/eval = stratified 40/60 of 1cls test",
            "categories": list(categories),
            "missing_files": missing,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        **splits,
    }
    args.splits.parent.mkdir(parents=True, exist_ok=True)
    args.splits.write_text(json.dumps(splits_out, indent=1), encoding="utf-8")
    _log(f"[splits] wrote {args.splits}")
    print_counts(splits)
    if any(missing.values()):
        _log(f"[warn] missing files referenced by the split: {missing}")
    if extract_info:
        _log(f"[extract] {extract_info.get('files')} files under {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
