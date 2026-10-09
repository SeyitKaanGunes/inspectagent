"""Latency micro-benchmark of InspectAgent's core OpenCV operations (x86-vs-Graviton evidence).

Times, on one part: virtual-camera capture, ``assess_quality``, ``register``, ``StatTemplateModel.predict``,
``PatchCoreDNN.predict`` (when the bundle has a deep model) and a full ``run_inspection`` with the rule
planner. The part is the first bundled sample (``samples/samples.json``) whose category has a model bundle;
without bundles it falls back to a deterministic synthetic PCB scene and a quickly fitted statistical
template (reported as ``source.kind == "synthetic"``). Reports median / p90 milliseconds and ``cv_info()``.

Used in-process by ``GET /api/bench?n=..`` and from the command line::

    python scripts/bench_cv.py --n 20                     # -> results/bench_<arch>_<backend>.json
    python scripts/bench_cv.py --n 20 --out results/bench_aarch64_stock.json --category pcb1
"""

from __future__ import annotations

import argparse
import functools
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):  # executed as a script: make the repository importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from inspectagent.config import REPO_ROOT, DetectorConfig, Settings, get_settings
from inspectagent.types import CaptureSettings, PartSpec, View

SCHEMA = "inspectagent.bench/1"
SYNTH_SIZE = (900, 600)  # (W, H) of the synthetic fallback scene
SYNTH_NORMALS = 4  # normal boards used to fit the fallback template (fit cost grows with n)
SYNTH_TEMPLATE_LONG_SIDE = 512  # small template: the fallback must stay fast (also on Lambda)

__all__ = ["run_bench", "synthetic_board", "summarize_ms", "main"]


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def summarize_ms(samples: list[float]) -> dict:
    """Median / p90 / mean / min / max (ms, rounded) of a list of durations."""
    if not samples:
        raise ValueError("no samples")
    a = np.asarray(samples, dtype=np.float64)
    return {
        "n": int(a.size),
        "median_ms": round(float(np.median(a)), 3),
        "p90_ms": round(float(np.percentile(a, 90)), 3),
        "mean_ms": round(float(a.mean()), 3),
        "min_ms": round(float(a.min()), 3),
        "max_ms": round(float(a.max()), 3),
    }


def _time_op(fn: Callable[[int], Any], n: int, warmup: int) -> tuple[dict, Any]:
    """Call ``fn(i)`` ``warmup`` times untimed (i < 0) then ``n`` times timed; returns (summary, last result)."""
    last: Any = None
    for w in range(warmup):
        last = fn(-1 - w)
    samples: list[float] = []
    for i in range(n):
        t0 = time.perf_counter()
        last = fn(i)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return summarize_ms(samples), last


def synthetic_board(seed: int = 0, size: tuple[int, int] = SYNTH_SIZE, design_seed: int = 11) -> np.ndarray:
    """Deterministic PCB-like scene (BGR uint8): fixed layout from ``design_seed``, per-instance
    photometric / placement variation from ``seed`` (so several instances can train a template)."""
    w, h = size
    design = np.random.default_rng(design_seed)
    rng = np.random.default_rng(50_000 + seed)
    base = np.array([40, 95, 32], np.float32) + rng.normal(0, 2.5, 3).astype(np.float32)
    canvas = np.empty((h, w, 3), np.float32)
    canvas[:] = base
    canvas += cv2.GaussianBlur(rng.normal(0, 5, (h, w)).astype(np.float32), (0, 0), 3)[..., None]
    img = np.clip(canvas, 0, 255).astype(np.uint8)  # text/shape drawing needs 8-bit images in OpenCV 5
    for _ in range(28):  # copper traces
        x1, y1 = int(design.integers(30, w - 30)), int(design.integers(30, h - 30))
        horizontal = design.random() < 0.5
        x2 = int(design.integers(30, w - 30)) if horizontal else x1
        y2 = y1 if horizontal else int(design.integers(30, h - 30))
        t = int(design.integers(3, 7))
        j = rng.integers(-1, 2, 2)
        cv2.line(img, (x1 + int(j[0]), y1 + int(j[1])), (x2 + int(j[0]), y2 + int(j[1])), (60, 150, 200), t, cv2.LINE_AA)
    for _ in range(36):  # pads with drill holes
        x, y, r = int(design.integers(40, w - 40)), int(design.integers(40, h - 40)), int(design.integers(6, 12))
        cv2.circle(img, (x, y), r, (170, 190, 205), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), max(2, r // 3), (40, 40, 40), -1, cv2.LINE_AA)
    for k in range(5):  # chips with leads and markings
        cw, ch = int(design.integers(60, 130)), int(design.integers(40, 90))
        x, y = int(design.integers(30, w - cw - 30)), int(design.integers(30, h - ch - 30))
        cv2.rectangle(img, (x, y), (x + cw, y + ch), (28, 28, 30), -1)
        for px in range(x + 6, x + cw - 4, 10):
            cv2.rectangle(img, (px, y - 6), (px + 4, y), (190, 195, 200), -1)
            cv2.rectangle(img, (px, y + ch), (px + 4, y + ch + 6), (190, 195, 200), -1)
        cv2.putText(img, f"U{k + 1}", (x + 8, y + ch // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (205, 205, 205), 1, cv2.LINE_AA)
    cv2.putText(img, "BENCH-1", (24, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)
    out = img.astype(np.float32) * (1.0 + rng.normal(0, 0.015)) + rng.normal(0, 1.2, img.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _load_samples(samples_dir: Path) -> list[dict]:
    path = Path(samples_dir) / "samples.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("samples", [])
    return [d for d in data if isinstance(d, dict) and d.get("id") and d.get("category")]


def _sample_image_path(sample: dict, samples_dir: Path) -> Path | None:
    for key in ("image", "image_path", "path", "file"):
        if sample.get(key):
            p = Path(sample[key])
            return p if p.is_absolute() else Path(samples_dir) / p
    return None


def _reference_sharpness(cam: Any, zooms: tuple[float, ...] = (1.0, 3.0)) -> dict[str, float]:
    """Laplacian-variance sharpness of clean captures per zoom bucket (what bundles store)."""
    from inspectagent.vision.quality import laplacian_sharpness

    out: dict[str, float] = {}
    for i, z in enumerate(zooms):
        cap = cam.clean_capture(View(0.5, 0.5, z), f"ref_{i}", i)
        gray = cv2.cvtColor(cap.image, cv2.COLOR_BGR2GRAY)
        out[f"z{int(round(z))}"] = float(laplacian_sharpness(gray))
    return out


@functools.lru_cache(maxsize=1)
def _synthetic_bundle() -> tuple[Any, np.ndarray, float]:
    """(CategoryBundle, unseen normal scene, fit ms) for the no-bundle fallback; cached per process."""
    from inspectagent.models_io import CategoryBundle
    from inspectagent.types import NuisanceConditions
    from inspectagent.vision.anomaly_stat import StatTemplateModel
    from inspectagent.vision.camera import VirtualCamera

    normals = [synthetic_board(i) for i in range(SYNTH_NORMALS)]
    t0 = time.perf_counter()
    stat = StatTemplateModel.fit(
        "synthetic", normals, cfg=DetectorConfig(template_long_side=SYNTH_TEMPLATE_LONG_SIDE), log=None
    )
    fit_ms = (time.perf_counter() - t0) * 1000.0
    scene = synthetic_board(1000)
    ref = _reference_sharpness(VirtualCamera(scene, NuisanceConditions(noise_sigma=0.0, preset="clean"), seed=0))
    bundle = CategoryBundle(
        category="synthetic",
        stat=stat,
        deep=None,
        meta={"synthetic": True, "n_normals": SYNTH_NORMALS, "template_long_side": SYNTH_TEMPLATE_LONG_SIDE},
        reference_sharpness=ref,
    )
    return bundle, scene, fit_ms


def _resolve_target(category: str | None, settings: Settings, notes: list[str]) -> dict:
    """Pick the benchmark part: a bundled sample if possible, else the synthetic fallback."""
    from inspectagent.models_io import available_categories, load_bundle
    from inspectagent.vision.imageio import imread

    cats = available_categories(settings.models_dir)
    samples = _load_samples(settings.samples_dir)
    wanted = [s for s in samples if s["category"] == category] if category else [s for s in samples if s["category"] in cats]
    wanted.sort(key=lambda s: s.get("label") not in (0, "0", "normal"))  # stable sort: normal parts first
    if category and category not in cats:
        notes.append(f"no model bundle for category {category!r} under {settings.models_dir}")
    elif not samples:
        notes.append(f"no samples.json under {settings.samples_dir}")
    elif not wanted:
        notes.append("no bundled sample has a model bundle")
    for sample in wanted[:1]:
        img_path = _sample_image_path(sample, settings.samples_dir)
        try:
            if img_path is None:
                raise FileNotFoundError(f"sample {sample['id']} has no image path")
            t0 = time.perf_counter()
            bundle = load_bundle(sample["category"], settings.models_dir)
            load_ms = (time.perf_counter() - t0) * 1000.0
            scene = imread(img_path)
        except Exception as exc:  # noqa: BLE001 - fall back to the synthetic part, but say why
            notes.append(f"bundled sample {sample['id']} unusable ({type(exc).__name__}: {exc})")
            break
        label = sample.get("label")
        return {
            "kind": "bundle",
            "category": sample["category"],
            "sample_id": sample["id"],
            "image_path": str(img_path),
            "label": int(label) if isinstance(label, (int, np.integer)) else None,
            "scene": scene,
            "bundle": bundle,
            "setup_ms": {"load_bundle": round(load_ms, 1)},
        }
    bundle, scene, fit_ms = _synthetic_bundle()
    notes.append("synthetic fallback: deterministic PCB-like scene + statistical template fitted on "
                 f"{SYNTH_NORMALS} synthetic normals (no deep detector)")
    return {
        "kind": "synthetic",
        "category": bundle.category,
        "sample_id": "synthetic_board_1000",
        "image_path": "",
        "label": 0,
        "scene": scene,
        "bundle": bundle,
        "setup_ms": {"fit_stat_template": round(fit_ms, 1)},
    }


# ------------------------------------------------------------------------------------------------
# benchmark
# ------------------------------------------------------------------------------------------------
def run_bench(
    n: int = 10,
    category: str | None = None,
    *,
    seed: int = 0,
    preset: str = "nominal",
    warmup: int = 1,
    settings: Settings | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Time the core operations ``n`` times each; returns a JSON-serializable report.

    ``timings_ms[op]`` holds ``median_ms``/``p90_ms`` (plus mean/min/max). Each per-operation timing is
    preceded by ``warmup`` untimed calls; the full-loop timing reuses the warmed state.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    say = log or (lambda _msg: None)
    settings = settings or get_settings()
    from inspectagent.agent.loop import run_inspection
    from inspectagent.agent.planner_rule import RulePlanner
    from inspectagent.vision.camera import VirtualCamera, make_conditions
    from inspectagent.vision.cvinfo import cv_info
    from inspectagent.vision.quality import assess_quality
    from inspectagent.vision.registration import register

    t_start = time.perf_counter()
    notes: list[str] = []
    target = _resolve_target(category, settings, notes)
    bundle, scene = target["bundle"], target["scene"]
    conditions = make_conditions(preset, np.random.default_rng(seed))
    cam = VirtualCamera(scene, conditions, cfg=settings.camera, seed=seed)
    cap_settings = CaptureSettings()  # full view, EV 0, ring light, no autofocus (the agent's first shot)
    ref_sharp = (bundle.reference_sharpness or {}).get("z1")
    say(f"bench: {target['kind']} part {target['sample_id']} ({target['category']}), n={n}, preset={preset}")

    timings: dict[str, dict] = {}
    details: dict[str, Any] = {}

    timings["camera_capture"], cap = _time_op(lambda i: cam.capture(cap_settings, f"bench_cap_{i + 1}", i + 1), n, warmup)
    say(f"  camera_capture  median {timings['camera_capture']['median_ms']:.1f} ms")

    timings["assess_quality"], q = _time_op(lambda i: assess_quality(cap, ref_sharp, settings.quality), n, warmup)
    details["quality"] = {"ok": bool(q.ok), "issues": list(q.issues), "sharpness": round(float(q.sharpness), 2)}
    say(f"  assess_quality  median {timings['assess_quality']['median_ms']:.1f} ms")

    moving = cv2.cvtColor(cap.image, cv2.COLOR_BGR2GRAY)
    fixed = cv2.cvtColor(cam.clean_capture(cap_settings.view, "bench_ref", 0).image, cv2.COLOR_BGR2GRAY)
    timings["registration"], reg = _time_op(lambda i: register(moving, fixed, None, settings.detector), n, warmup)
    details["registration"] = {"ok": bool(reg.ok), "method": str(reg.method), "ncc": round(float(reg.ncc), 4)}
    say(f"  registration    median {timings['registration']['median_ms']:.1f} ms ({reg.method})")

    timings["stat_predict"], res = _time_op(lambda i: bundle.stat.predict(cap, q, id_prefix="bench"), n, warmup)
    details["stat_predict"] = {"prob": round(float(res.prob), 4), "reliable": bool(res.reliable), "n_regions": len(res.regions)}
    say(f"  stat_predict    median {timings['stat_predict']['median_ms']:.1f} ms")

    if bundle.deep is not None:
        timings["deep_predict"], dres = _time_op(lambda i: bundle.deep.predict(cap, id_prefix="bench"), n, warmup)
        details["deep_predict"] = {"prob": round(float(dres.prob), 4), "n_regions": len(dres.regions)}
        say(f"  deep_predict    median {timings['deep_predict']['median_ms']:.1f} ms")
    else:
        details["deep_predict"] = {"skipped": "bundle has no deep detector"}

    part = PartSpec(
        part_id=target["sample_id"],
        category=target["category"],
        image_path=target["image_path"],
        label=target["label"],
        conditions=conditions,
    )
    loops: list[dict] = []

    def one_loop(i: int) -> Any:
        trace, _session = run_inspection(part, bundle, planner=RulePlanner(), settings=settings, seed=seed + max(i, 0), scene=scene)
        tot = trace.totals or {}
        loops.append({
            "seed": seed + max(i, 0),
            "verdict": trace.verdict.label if trace.verdict else None,
            "n_steps": int(tot.get("n_steps", len(trace.steps))),
            "n_captures": int(tot.get("n_captures", 0)),
            "compute_ms": round(float(tot.get("compute_ms", 0.0)), 1),
            "sim_time_ms": round(float(tot.get("sim_time_ms", 0.0)), 1),
        })
        return trace

    timings["run_inspection"], _ = _time_op(one_loop, n, 0)
    details["run_inspection"] = {
        "planner": "rule",
        "median_steps": float(np.median([r["n_steps"] for r in loops])),
        "median_captures": float(np.median([r["n_captures"] for r in loops])),
        "median_sim_time_ms": float(np.median([r["sim_time_ms"] for r in loops])),
        "verdicts": {v: sum(1 for r in loops if r["verdict"] == v) for v in sorted({str(r["verdict"]) for r in loops})},
        "runs": loops,
    }
    say(f"  run_inspection  median {timings['run_inspection']['median_ms']:.1f} ms")

    info = cv_info()
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n": int(n),
        "warmup": int(warmup),
        "seed": int(seed),
        "preset": preset,
        "source": {
            "kind": target["kind"],
            "category": target["category"],
            "sample_id": target["sample_id"],
            "scene_size": [int(scene.shape[1]), int(scene.shape[0])],
            "sensor_size": [int(cam.sensor_size[0]), int(cam.sensor_size[1])],
            "has_deep": bundle.deep is not None,
        },
        "machine": {
            "arch": platform.machine(),
            "processor": platform.processor(),
            "system": platform.system(),
            "cpu_count": info.get("cpu_count"),
            "opencv_threads": int(cv2.getNumThreads()),
        },
        "cv": info,
        "setup_ms": target["setup_ms"],
        "timings_ms": timings,
        "details": details,
        "notes": notes,
        "wall_ms": round((time.perf_counter() - t_start) * 1000.0, 1),
    }


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def _norm_arch(arch: str) -> str:
    a = (arch or "unknown").lower()
    return {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}.get(a, a)


def default_out_path(report: dict) -> Path:
    cv = report.get("cv") or {}
    return REPO_ROOT / "results" / f"bench_{_norm_arch(cv.get('arch', ''))}_{cv.get('build_backend', 'stock')}.json"


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Benchmark InspectAgent's OpenCV operations and the full agent loop.")
    ap.add_argument("--n", type=int, default=10, help="timed repetitions per operation (default: %(default)s)")
    ap.add_argument("--category", default=None, help="category of the bundled sample to use (default: first with a bundle)")
    ap.add_argument("--preset", default="nominal", help="nuisance-condition preset of the virtual camera (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1, help="untimed calls before each operation (default: %(default)s)")
    ap.add_argument("--threads", type=int, default=None, help="OpenCV threads (default: Settings.cv_threads / OpenCV default)")
    ap.add_argument("--out", default=None, help="output JSON (default: results/bench_<arch>_<backend>.json)")
    args = ap.parse_args(argv)

    from inspectagent.vision.cvinfo import configure_threads

    configure_threads(args.threads)
    report = run_bench(n=args.n, category=args.category, seed=args.seed, preset=args.preset, warmup=args.warmup, log=print)
    out = Path(args.out) if args.out else default_out_path(report)
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    src, cv = report["source"], report["cv"]
    print(f"\nOpenCV {cv.get('opencv_version')} ({cv.get('build_backend')}) on {cv.get('arch')}, "
          f"{report['machine']['opencv_threads']} threads; part: {src['kind']} {src['sample_id']} {src['scene_size']}")
    print(f"{'operation':<16} {'median ms':>10} {'p90 ms':>10} {'n':>4}")
    for op, t in report["timings_ms"].items():
        print(f"{op:<16} {t['median_ms']:>10.2f} {t['p90_ms']:>10.2f} {t['n']:>4}")
    for note in report["notes"]:
        print(f"note: {note}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
