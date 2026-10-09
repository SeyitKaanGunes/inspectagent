"""End-to-end smoke test of the integrated InspectAgent stack on synthetic boards (offline).

No VisA data, no AWS, no network. Steps:

1. **Bundle** - ``StatTemplateModel.fit`` on 24 synthetic normal boards (``tests/synth.py``), then the
   ``scripts/build_models.py`` recipe in miniature: reference sharpness from oracle clean captures,
   normal-score calibration on held-out boards (clean + nominal presets, zoom 1 and 2/3/4 views,
   pixel-percentile region thresholds), logistic calibration on labeled single-shot captures. When
   ``models/backbone_resnet18.onnx`` exists a ``PatchCoreDNN`` (small coreset) is fitted and
   calibrated too. The bundle is written with ``models_io.save_bundle`` into a temporary models dir
   and loaded back with ``models_io.load_bundle``.
2. **Agent** - ``run_inspection`` (RulePlanner + HumanControlPolicy) on 3 normal + 3 defective boards
   for each preset (clean, nominal, dark, bright, defocus, glare, harsh); prints one row per run:
   preset, label, verdict, final prob, steps, tools used, cycle time.
3. **Web API** - ``create_app`` with ``LocalStorage`` in a temp dir, the temp models dir and a temp
   ``samples.json`` pointing at synthetic boards; ``POST /api/inspect``, a review decision,
   ``GET /api/system`` (plus samples, traces, artifacts, reviews, metrics, healthz).
4. **MCP** - the ``InspectAgentMCP`` tool functions for one session, driven step by step by the
   RulePlanner acting as an MCP client, then ``get_trace`` / ``get_artifact`` / ``run_inspection`` /
   ``close_session``.

Sanity checks (exit code 1 when one fails): every run ends with a terminal tool inside the
budget; traces serialize to strict JSON; clean defective boards are never PASSed and at most ~10 %
of all defective runs are PASSed; most clean normal boards PASS; the planner reacts to a detected
glare / blur / under-exposure with diffuse light / autofocus / positive EV; API and MCP calls
succeed.

Usage::

    python scripts/smoke_synthetic.py                 # full run (~2-4 min)
    python scripts/smoke_synthetic.py --no-deep --presets clean glare --per-preset 2
    python scripts/smoke_synthetic.py --keep --json smoke_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inspectagent.config import PolicyConfig, Settings  # noqa: E402
from inspectagent.eval_utils import stable_seed  # noqa: E402
from inspectagent.models_io import BACKBONE_FILE, CategoryBundle, clear_bundle_cache, load_bundle, save_bundle  # noqa: E402
from inspectagent.types import CaptureSettings, PartSpec, Trace, View  # noqa: E402
from tests.synth import DEFECT_KINDS, inject_defect, make_board  # noqa: E402

CATEGORY = "synthpcb"
ALL_PRESETS = ("clean", "nominal", "dark", "bright", "defocus", "glare", "harsh")
TERMINAL = ("finish", "request_human_review")
BUCKETS = ("z1", "z2", "z3", "z4")
CALIB_ZOOMS = (2.0, 3.0, 4.0)

# Board seeds (disjoint ranges): fit normals, calibration normals, labeled calibration, test parts.
FIT_SEED0, CAL_SEED0, LAB_SEED0, TEST_SEED0 = 0, 200, 300, 400


# --------------------------------------------------------------------------------------- helpers
def _log(msg: str) -> None:
    print(msg, flush=True)


def _strict_json(obj: Any) -> str:
    """Strict JSON (no NaN/inf); raises ``ValueError`` when the object is not serializable."""
    return json.dumps(obj, allow_nan=False, ensure_ascii=False)


@dataclass
class Checks:
    """Collects named pass/fail checks and prints them."""

    items: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.items.append((name, bool(ok), detail))
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.items)

    def report(self) -> None:
        _log("\n== sanity checks ==")
        for name, ok, detail in self.items:
            _log(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
        n_fail = sum(1 for _, ok, _ in self.items if not ok)
        _log(f"  {len(self.items) - n_fail}/{len(self.items)} checks passed")


def _defective_board(k: int) -> tuple[np.ndarray, str]:
    kind = DEFECT_KINDS[k % len(DEFECT_KINDS)]
    img, _mask = inject_defect(make_board(TEST_SEED0 + 50 + k), kind, seed=k)
    return img, kind


def _test_parts(per_preset: int) -> list[tuple[str, int, np.ndarray, str]]:
    """``(part_id, label, scene, kind)`` - the same boards are inspected under every preset."""
    parts: list[tuple[str, int, np.ndarray, str]] = []
    for k in range(per_preset):
        parts.append((f"synth_normal_{k}", 0, make_board(TEST_SEED0 + k), "normal"))
    for k in range(per_preset):
        img, kind = _defective_board(k)
        parts.append((f"synth_defect_{k}_{kind}", 1, img, kind))
    return parts


# ----------------------------------------------------------------------------------- 1. bundle
def build_bundle(models_dir: Path, settings: Settings, n_fit: int, use_deep: bool, coreset: int, seed: int) -> dict:
    """Fit + calibrate + save the synthetic bundle; returns build info."""
    from inspectagent.vision.anomaly_stat import StatTemplateModel
    from inspectagent.vision.camera import VirtualCamera, make_conditions
    from inspectagent.vision.quality import laplacian_sharpness
    from scripts.build_models import apply_region_thresholds, region_thresholds, sample_views, texture_weights

    info: dict[str, Any] = {"n_fit": n_fit, "timings_s": {}}
    cam_cfg = settings.camera
    t0 = time.perf_counter()
    normals = [make_board(FIT_SEED0 + i) for i in range(n_fit)]
    stat = StatTemplateModel.fit(CATEGORY, normals, cfg=settings.detector, log=None)
    info["timings_s"]["stat_fit"] = time.perf_counter() - t0
    _log(f"[bundle] StatTemplateModel fitted on {n_fit} boards in {info['timings_s']['stat_fit']:.1f}s "
         f"(template {stat.template_size}, provisional calib source={stat.calib.get('source')})")

    deep = None
    onnx = REPO_ROOT / "models" / BACKBONE_FILE
    if use_deep and onnx.is_file():
        from inspectagent.vision.anomaly_deep import PatchCoreDNN

        t0 = time.perf_counter()
        deep = PatchCoreDNN.fit(CATEGORY, normals, onnx, cfg=settings.detector, log=lambda m: None, seed=seed, coreset_size=coreset)
        info["timings_s"]["deep_fit"] = time.perf_counter() - t0
        _log(f"[bundle] PatchCoreDNN fitted (bank {deep.memory_bank.shape[0]}, engine {deep.engine}) in {info['timings_s']['deep_fit']:.1f}s")
    elif use_deep:
        _log(f"[bundle] {onnx} not found -> no deep detector")
    info["deep"] = deep is not None

    # reference sharpness (oracle clean captures) + normal-score calibration (held-out boards)
    t0 = time.perf_counter()
    sharp: dict[str, list[float]] = {b: [] for b in BUCKETS}
    scores: dict[str, list[float]] = {b: [] for b in BUCKETS}
    zpix: dict[str, list[np.ndarray]] = {b: [] for b in BUCKETS}
    deep_normal: list[float] = []
    n_caps = n_fail = 0
    for i in range(8):
        scene = make_board(CAL_SEED0 + i)
        weights = texture_weights(scene)
        vrng = np.random.default_rng(stable_seed(seed, "sharp", i))
        cam = VirtualCamera(scene, make_conditions("clean", np.random.default_rng(stable_seed(seed, "sharp-cond", i))), cam_cfg, seed=i)
        for k, view in enumerate([View(0.5, 0.5, 1.0)] + sample_views(weights, vrng, CALIB_ZOOMS, 1)):
            cap = cam.clean_capture(view, f"sharp_{k}", k)
            sharp[_bucket(view.zoom)].append(float(laplacian_sharpness(cap.image)))
        for preset in ("clean", "nominal"):
            cond = make_conditions(preset, np.random.default_rng(stable_seed(seed, "cal-cond", preset, i)))
            cam = VirtualCamera(scene, cond, cam_cfg, seed=stable_seed(seed, "cal-cam", preset, i))
            vrng = np.random.default_rng(stable_seed(seed, "cal-views", preset, i))
            prng = np.random.default_rng(stable_seed(seed, "cal-px", preset, i))
            for k, view in enumerate([View(0.5, 0.5, 1.0)] + sample_views(weights, vrng, CALIB_ZOOMS, 1)):
                cap = cam.capture(CaptureSettings(view=view), f"cal_{k}", k)
                n_caps += 1
                score, zmap, valid, reg = stat.score_capture(cap)
                if deep is not None and view.zoom < 1.5:
                    deep_normal.append(float(deep.predict(cap).score))
                if not reg.ok or not math.isfinite(float(score)):
                    n_fail += 1
                    continue
                b = _bucket(view.zoom)
                scores[b].append(float(score))
                vals = np.asarray(zmap, np.float32)[np.asarray(valid) > 0]
                vals = vals[np.isfinite(vals)]
                if vals.size > 20000:
                    vals = prng.choice(vals, 20000, replace=False)
                zpix[b].append(vals)
    ref_sharp = {b: float(np.median(v)) for b, v in sharp.items() if v}
    region_thr, _excluded = region_thresholds(scores, zpix)

    # labeled calibration: one default nominal full-view capture per board
    labeled_stat: list[tuple[float, int, str]] = []
    labeled_deep: list[tuple[float, int]] = []
    for j in range(16):
        label = j % 2
        if label:
            scene, _ = inject_defect(make_board(LAB_SEED0 + j), DEFECT_KINDS[(j // 2) % len(DEFECT_KINDS)], seed=100 + j)
        else:
            scene = make_board(LAB_SEED0 + j)
        cond = make_conditions("nominal", np.random.default_rng(stable_seed(seed, "lab-cond", j)))
        cond.glare_strength = 0.0  # logistic is fitted on detector separability, not on glare handling
        cam = VirtualCamera(scene, cond, cam_cfg, seed=stable_seed(seed, "lab-cam", j))
        cap = cam.capture(CaptureSettings(), "lab_0", 0)
        score, _z, _v, reg = stat.score_capture(cap)
        if reg.ok and math.isfinite(float(score)):
            labeled_stat.append((float(score), label, "z1"))
        if deep is not None:
            labeled_deep.append((float(deep.predict(cap).score), label))
    stat.calibrate({b: v for b, v in scores.items() if v}, labeled_stat or None)
    score_thr = {str(k): float(v) for k, v in (stat.calib.get("region_thr") or {}).items()}
    applied = {b: max(region_thr.get(b, 0.0), score_thr.get(b, 0.0)) for b in BUCKETS}
    apply_region_thresholds(stat, applied)
    if deep is not None:
        deep.calibrate(deep_normal, labeled_deep or None)
    info["timings_s"]["calibration"] = time.perf_counter() - t0
    info["calibration"] = {
        "normal_captures": n_caps,
        "registration_failures": n_fail,
        "scores_per_bucket": {b: len(v) for b, v in scores.items()},
        "labeled": len(labeled_stat),
        "logistic": stat.calib.get("logistic"),
        "region_thr": applied,
        "deep_calib": {k: deep.calib.get(k) for k in ("mu", "sd", "a", "b", "source")} if deep is not None else None,
    }
    _log(
        f"[bundle] calibration in {info['timings_s']['calibration']:.1f}s: {n_caps} normal captures "
        f"({n_fail} registration failures), {len(labeled_stat)} labeled, logistic={_fmt_dict(stat.calib.get('logistic'))}, "
        f"region_thr={_fmt_dict(applied)}, ref_sharpness={_fmt_dict(ref_sharp)}"
    )

    meta = {"builder": "scripts/smoke_synthetic.py", "synthetic": True, "scene_size": list(stat.scene_size), "calibration": info["calibration"]}
    save_bundle(CategoryBundle(CATEGORY, stat, deep, meta, ref_sharp), models_dir)
    if deep is not None:  # models_io looks for the shared backbone next to the category dirs
        shutil.copy2(onnx, models_dir / BACKBONE_FILE)
    info["files"] = sorted(p.name for p in (models_dir / CATEGORY).iterdir())
    return info


def _bucket(zoom: float) -> str:
    return "z1" if zoom < 1.5 else "z2" if zoom < 2.5 else "z3" if zoom < 3.5 else "z4"


def _fmt_dict(d: Any) -> str:
    if not isinstance(d, dict):
        return str(d)
    return "{" + ", ".join(f"{k}: {v:.3g}" if isinstance(v, float) else f"{k}: {v}" for k, v in d.items()) + "}"


# ------------------------------------------------------------------------------------ 2. agent
def _tool_token(step: Any) -> str:
    d = step.decision
    a = d.args or {}
    if d.tool == "capture_image":
        bits = []
        if a.get("light", "ring") != "ring":
            bits.append(str(a["light"]))
        if abs(float(a.get("exposure_ev", 0.0))) > 1e-6:
            bits.append(f"{float(a['exposure_ev']):+.2g}EV")
        if a.get("autofocus"):
            bits.append("AF")
        return "cap" + (f"[{','.join(bits)}]" if bits else "")
    if d.tool == "assess_quality":
        issues = step.observation.get("issues") or []
        return "q" + (f"({','.join(i[:4] for i in issues)})" if issues else "")
    if d.tool == "detect_anomalies":
        return a.get("detector", "fast")
    if d.tool == "inspect_region":
        st = step.observation.get("status", "?")
        return f"zoom:{st[:4]}"
    if d.tool == "measure_defect":
        return "meas"
    if d.tool == "finish":
        return f"finish({a.get('label')})"
    if d.tool == "request_human_review":
        return "review" + ("*" if step.overridden_by_policy else "")
    return d.tool


def _reaction(trace: Trace) -> dict:
    """Whether the planner applied the specified fix after the first full-view quality report."""
    steps = trace.steps
    q_first = next((s for s in steps if s.decision.tool == "assess_quality"), None)
    issues = list((q_first.observation.get("issues") or [])) if q_first else []
    caps = [s.decision.args for s in steps if s.decision.tool == "capture_image"]
    second = caps[1] if len(caps) > 1 else None
    out = {"first_issues": issues, "recaptured": second is not None}
    if second is not None:
        out["second_capture"] = {k: second.get(k) for k in ("exposure_ev", "light", "autofocus")}
    return out


def run_agent_matrix(bundle: CategoryBundle, settings: Settings, presets: list[str], per_preset: int, seed: int) -> list[dict]:
    """``run_inspection`` for every preset x test part; returns one row dict per run."""
    from inspectagent.agent.loop import run_inspection
    from inspectagent.agent.trace import trace_to_json
    from inspectagent.vision.camera import make_conditions

    parts = _test_parts(per_preset)
    rows: list[dict] = []
    for preset in presets:
        for k, (pid, label, scene, kind) in enumerate(parts):
            cond = make_conditions(preset, np.random.default_rng(stable_seed(seed, "test-cond", preset, k % per_preset)))
            part = PartSpec(part_id=f"{pid}_{preset}", category=CATEGORY, image_path="", label=label, conditions=cond)
            t0 = time.perf_counter()
            err = None
            try:
                trace, session = run_inspection(part, bundle, settings=settings, seed=stable_seed(seed, "run", preset, k), scene=scene)
            except Exception as exc:  # reported as a failed row; the smoke test continues
                err = f"{type(exc).__name__}: {exc}"
                rows.append({"preset": preset, "part_id": pid, "label": label, "kind": kind, "error": err})
                _log(f"  {preset:<8} {pid:<24} ERROR {err}")
                continue
            wall = (time.perf_counter() - t0) * 1000.0
            json_ok, json_err = True, ""
            try:
                _strict_json(json.loads(trace_to_json(trace)))
                _strict_json(trace.to_dict())
            except (TypeError, ValueError) as exc:
                json_ok, json_err = False, str(exc)
            pol = session.policy_cfg
            t = trace.totals
            within = (
                len(trace.steps) <= pol.max_steps
                and session.counters["captures"] <= pol.max_captures
                and session.counters["deep_calls"] <= pol.max_deep_calls
                and session.sim_time_ms <= pol.max_sim_time_ms
                and bool(trace.steps)
                and trace.steps[-1].decision.tool in TERMINAL
            )
            rows.append(
                {
                    "preset": preset,
                    "part_id": pid,
                    "label": label,
                    "kind": kind,
                    "verdict": trace.verdict.label,
                    "action": trace.verdict.action,
                    "prob": float(trace.verdict.prob_anomalous if trace.verdict.prob_anomalous is not None else float("nan")),
                    "steps": len(trace.steps),
                    "captures": t["n_captures"],
                    "deep": t["n_deep"],
                    "tools": " ".join(_tool_token(s) for s in trace.steps),
                    "cycle_ms": float(t["cycle_time_ms"]),
                    "sim_ms": float(t["sim_time_ms"]),
                    "compute_ms": float(t["compute_ms"]),
                    "wall_ms": wall,
                    "within_budget": within,
                    "json_ok": json_ok,
                    "json_error": json_err,
                    "overrides": t["n_overrides"],
                    "errors": list(trace.errors),
                    "reaction": _reaction(trace),
                }
            )
            r = rows[-1]
            _log(
                f"  {preset:<8} {('defect' if label else 'normal'):<6} {kind:<8} {r['verdict']:<12} p={r['prob']:.3f} "
                f"steps={r['steps']:>2} cycle={r['cycle_ms']:7.0f}ms  {r['tools']}"
            )
    return rows


def print_table(rows: list[dict]) -> None:
    _log("\n== agent runs ==")
    hdr = f"{'preset':<8} {'label':<6} {'kind':<8} {'verdict':<12} {'prob':>6} {'steps':>5} {'cycle ms':>9}  tools used"
    _log(hdr)
    _log("-" * len(hdr))
    for r in rows:
        if "error" in r:
            _log(f"{r['preset']:<8} {('defect' if r['label'] else 'normal'):<6} {r['kind']:<8} ERROR {r['error']}")
            continue
        _log(
            f"{r['preset']:<8} {('defect' if r['label'] else 'normal'):<6} {r['kind']:<8} {r['verdict']:<12} "
            f"{r['prob']:6.3f} {r['steps']:5d} {r['cycle_ms']:9.0f}  {r['tools']}"
        )
    _log("\n== verdicts by preset (normal: PASS/REVIEW/FAIL | defect: PASS/REVIEW/FAIL) ==")
    presets = list(dict.fromkeys(r["preset"] for r in rows))
    for p in presets:
        cells = []
        for label in (0, 1):
            c = Counter(r.get("verdict", "ERROR") for r in rows if r["preset"] == p and r["label"] == label)
            cells.append(f"{c.get('PASS', 0)}/{c.get('NEEDS_REVIEW', 0)}/{c.get('FAIL', 0)}" + (f" err={c['ERROR']}" if c.get("ERROR") else ""))
        ok_rows = [r for r in rows if r["preset"] == p and "error" not in r]
        mean_cycle = float(np.mean([r["cycle_ms"] for r in ok_rows])) if ok_rows else float("nan")
        mean_steps = float(np.mean([r["steps"] for r in ok_rows])) if ok_rows else float("nan")
        _log(f"  {p:<8} normal {cells[0]:<9} defect {cells[1]:<9} mean steps {mean_steps:4.1f}  mean cycle {mean_cycle:6.0f} ms")


def check_agent(rows: list[dict], checks: Checks, presets: list[str]) -> None:
    ok_rows = [r for r in rows if "error" not in r]
    checks.add("no run raised", len(ok_rows) == len(rows), f"{len(rows) - len(ok_rows)} errors")
    bad = [f"{r['preset']}/{r['part_id']}" for r in ok_rows if not r["within_budget"]]
    checks.add("every run terminates with a terminal tool within budget", not bad, ", ".join(bad))
    bad = [f"{r['preset']}/{r['part_id']}: {r['json_error']}" for r in ok_rows if not r["json_ok"]]
    checks.add("every trace serializes to strict JSON", not bad, "; ".join(bad))

    defect = [r for r in ok_rows if r["label"] == 1]
    escapes = [f"{r['preset']}/{r['part_id']}" for r in defect if r["verdict"] == "PASS"]
    limit = max(1, int(round(0.10 * len(defect))))
    checks.add(f"defective boards silently PASSed rarely (<= {limit} of {len(defect)})", len(escapes) <= limit, ", ".join(escapes) or "none")
    if "clean" in presets:
        clean_def = [r for r in defect if r["preset"] == "clean"]
        clean_esc = [r["part_id"] for r in clean_def if r["verdict"] == "PASS"]
        flagged = sum(1 for r in clean_def if r["verdict"] in ("FAIL", "NEEDS_REVIEW"))
        checks.add("clean defective boards are never PASSed", not clean_esc, f"FAIL/REVIEW {flagged}/{len(clean_def)}")
        clean_fail = sum(1 for r in clean_def if r["verdict"] == "FAIL")
        checks.add("clean defective boards mostly auto-FAIL", clean_fail * 2 > len(clean_def), f"FAIL {clean_fail}/{len(clean_def)}")
        clean_norm = [r for r in ok_rows if r["preset"] == "clean" and r["label"] == 0]
        n_pass = sum(1 for r in clean_norm if r["verdict"] == "PASS")
        checks.add("clean normal boards mostly PASS", n_pass * 3 >= 2 * len(clean_norm), f"PASS {n_pass}/{len(clean_norm)}")
    normals = [r for r in ok_rows if r["label"] == 0]
    fr = [f"{r['preset']}/{r['part_id']}" for r in normals if r["verdict"] == "FAIL"]
    fr_limit = max(1, int(round(0.10 * len(normals))))
    checks.add(f"normal boards rarely auto-FAILed (<= {fr_limit} of {len(normals)})", len(fr) <= fr_limit, ", ".join(fr) or "none")

    # planner reactions: issue detected on the first full view -> the specified fix on the recapture
    def reacted(preset: str, issue: str, fix: Callable[[dict], bool], what: str) -> None:
        runs = [r for r in ok_rows if r["preset"] == preset]
        if not runs:
            return
        detected = [r for r in runs if issue in r["reaction"]["first_issues"] or (issue == "glare" and "overexposed" in r["reaction"]["first_issues"])]
        fixed = [r for r in detected if r["reaction"].get("recaptured") and fix(r["reaction"]["second_capture"])]
        checks.add(
            f"{preset}: planner answers '{issue}' with {what}",
            bool(detected) and len(fixed) == len(detected),
            f"issue detected in {len(detected)}/{len(runs)} runs, fixed in {len(fixed)}/{len(detected)}",
        )

    reacted("glare", "glare", lambda s: s.get("light") == "diffuse", "diffuse light")
    reacted("defocus", "blurred", lambda s: bool(s.get("autofocus")), "autofocus")
    reacted("dark", "underexposed", lambda s: float(s.get("exposure_ev") or 0.0) > 0.5, "EV > +0.5")
    reacted("bright", "overexposed", lambda s: float(s.get("exposure_ev") or 0.0) < 0.0 or s.get("light") == "diffuse", "negative EV (or diffuse light)")


# -------------------------------------------------------------------------------------- 3. API
def write_samples(samples_dir: Path, per_kind: int = 2) -> list[dict]:
    """Synthetic sample images + ``samples.json`` (same schema as ``scripts/make_samples.py``)."""
    from inspectagent.vision.imageio import imwrite

    entries: list[dict] = []
    for k in range(per_kind):
        img = make_board(TEST_SEED0 + 100 + k)
        rel = f"{CATEGORY}/{CATEGORY}_normal_{k}.png"
        imwrite(samples_dir / rel, img)
        entries.append({"id": f"{CATEGORY}_normal_{k}", "category": CATEGORY, "label": 0, "label_name": "normal", "image": rel,
                        "mask": None, "suggested_preset": "nominal", "note": "synthetic normal board", "defect_type": None})
    for k in range(per_kind):
        kind = DEFECT_KINDS[k % len(DEFECT_KINDS)]
        img, mask = inject_defect(make_board(TEST_SEED0 + 150 + k), kind, seed=50 + k)
        rel = f"{CATEGORY}/{CATEGORY}_anomaly_{k}.png"
        mrel = f"{CATEGORY}/{CATEGORY}_anomaly_{k}_mask.png"
        imwrite(samples_dir / rel, img)
        imwrite(samples_dir / mrel, mask)
        entries.append({"id": f"{CATEGORY}_anomaly_{k}", "category": CATEGORY, "label": 1, "label_name": "anomaly", "image": rel,
                        "mask": mrel, "suggested_preset": "glare", "note": f"synthetic {kind}", "defect_type": kind})
    (samples_dir / "samples.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return entries


def run_api(settings: Settings, storage_dir: Path, checks: Checks, seed: int) -> dict:
    from fastapi.testclient import TestClient

    from inspectagent.api.app import create_app
    from inspectagent.api.storage import LocalStorage

    out: dict[str, Any] = {}
    app = create_app(settings=settings, storage=LocalStorage(storage_dir))
    client = TestClient(app)

    r = client.get("/healthz")
    checks.add("API GET /healthz", r.status_code == 200 and r.json().get("ok") is True, f"status {r.status_code}")
    r = client.get("/api/samples")
    body = r.json() if r.status_code == 200 else {}
    n_avail = sum(1 for s in body.get("samples", []) if s.get("model_available"))
    checks.add("API GET /api/samples lists the synthetic samples with a model", r.status_code == 200 and n_avail >= 4,
               f"status {r.status_code}, {n_avail} with model, categories {body.get('categories_with_models')}")
    if body.get("samples"):
        r = client.get(body["samples"][0]["thumb_url"])
        checks.add("API GET sample thumbnail", r.status_code == 200 and r.headers.get("content-type", "").startswith("image/jpeg"), f"status {r.status_code}")

    t0 = time.perf_counter()
    r = client.post("/api/inspect", json={"sample_id": f"{CATEGORY}_anomaly_0", "preset": "glare", "planner": "rule", "seed": seed})
    wall = (time.perf_counter() - t0) * 1000.0
    ok = r.status_code == 200
    detail = f"status {r.status_code}"
    resp = r.json() if ok else {}
    if not ok:
        detail += f": {r.text[:300]}"
    else:
        v = resp["trace"]["verdict"]
        detail += (f", verdict {v['label']} ({v['action']}) p={v['prob_anomalous']:.3f}, {len(resp['trace']['steps'])} steps, "
                   f"{len(resp['artifacts'])} artifacts, {len(r.content) / 1024:.0f} KiB, {wall:.0f} ms, review={bool(resp.get('review'))}")
    checks.add("API POST /api/inspect (defective sample, glare preset)", ok and resp.get("persisted") is True, detail)
    out["inspect_defect"] = {"status": r.status_code, "detail": detail}
    if ok:
        checks.add("API response < 5 MB", len(r.content) < 5_000_000, f"{len(r.content)} bytes")
        checks.add("API defective sample not PASSed", resp["trace"]["verdict"]["label"] != "PASS", resp["trace"]["verdict"]["label"])
        tid = resp["trace"]["trace_id"]
        r2 = client.get(f"/api/traces/{tid}")
        checks.add("API GET /api/traces/{id}", r2.status_code == 200 and r2.json()["trace_id"] == tid, f"status {r2.status_code}")
        if resp["artifact_urls"]:
            url = next(iter(resp["artifact_urls"].values()))
            r3 = client.get(url)
            checks.add("API GET stored artifact", r3.status_code == 200 and r3.content[:3] == b"\xff\xd8\xff", f"status {r3.status_code}")
        review = resp.get("review")
        if review is not None:
            decision = "approve_fail" if review.get("kind") == "scrap_approval" else "rescan"
            r4 = client.post(f"/api/reviews/{review['review_id']}/decision", json={"decision": decision, "reviewer": "smoke", "note": "smoke test"})
            dbody = r4.json() if r4.status_code == 200 else {}
            expected = "route_scrap" if decision == "approve_fail" else None
            dok = r4.status_code == 200 and dbody.get("review", {}).get("audit_valid") is True and (expected is None or dbody.get("action") == expected)
            checks.add(f"API POST /api/reviews/{{id}}/decision ({decision})", dok,
                       f"status {r4.status_code}, action {dbody.get('action')}, final {dbody.get('final_label')}")
            out["decision"] = {"decision": decision, "status": r4.status_code, "action": dbody.get("action")}
            r5 = client.post(f"/api/reviews/{review['review_id']}/decision", json={"decision": "override_pass"})
            checks.add("API second decision on the same review -> 409", r5.status_code == 409, f"status {r5.status_code}")
        else:
            checks.add("API inspect created a review item (FAIL -> scrap approval / NEEDS_REVIEW)", False, "no review item")

    # a normal sample with a degraded preset: any verdict is acceptable, the request must succeed
    r = client.post("/api/inspect", json={"sample_id": f"{CATEGORY}_normal_0", "preset": "dark", "planner": "rule", "seed": seed})
    nb = r.json() if r.status_code == 200 else {}
    nv = (nb.get("trace") or {}).get("verdict") or {}
    checks.add("API POST /api/inspect (normal sample, dark preset)", r.status_code == 200,
               f"status {r.status_code}, verdict {nv.get('label')} ({nv.get('action')}), review={bool(nb.get('review'))}")
    if nb.get("review") and nb["review"].get("kind") == "needs_review":
        r6 = client.post(f"/api/reviews/{nb['review']['review_id']}/decision", json={"decision": "rescan", "reviewer": "smoke"})
        rb = r6.json() if r6.status_code == 200 else {}
        rescan_v = ((rb.get("rescan") or {}).get("trace") or {}).get("verdict") or {}
        checks.add("API review decision 'rescan' re-runs the inspection (clean preset)", r6.status_code == 200 and bool(rescan_v),
                   f"status {r6.status_code}, rescan verdict {rescan_v.get('label')}")

    r = client.get("/api/reviews?status=all")
    checks.add("API GET /api/reviews", r.status_code == 200, f"status {r.status_code}, {r.json().get('count') if r.status_code == 200 else '?'} reviews")
    r = client.get("/api/system")
    sysb = r.json() if r.status_code == 200 else {}
    checks.add("API GET /api/system", r.status_code == 200 and CATEGORY in sysb.get("models", {}).get("available_categories", []),
               f"status {r.status_code}, opencv {sysb.get('cv', {}).get('opencv_version')}, arch {sysb.get('deployment', {}).get('arch')}")
    out["system"] = {k: sysb.get(k) for k in ("version", "models", "presets")}
    r = client.get("/api/metrics")
    live = r.json().get("live", {}) if r.status_code == 200 else {}
    checks.add("API GET /api/metrics", r.status_code == 200 and live.get("inspections", 0) >= 2, f"status {r.status_code}, live inspections {live.get('inspections')}")
    r = client.get("/api/traces?limit=10")
    checks.add("API GET /api/traces", r.status_code == 200, f"status {r.status_code}")
    return out


# -------------------------------------------------------------------------------------- 4. MCP
def run_mcp(settings: Settings, samples_json: Path, checks: Checks, seed: int) -> dict:
    import anyio

    from inspectagent.agent.planner_rule import RulePlanner
    from inspectagent.mcp_server import InspectAgentMCP

    app = InspectAgentMCP(settings=settings, samples_path=samples_json)
    out: dict[str, Any] = {}
    names = [t.name for t in app.tool_definitions()]
    checks.add("MCP tool list contains the registry tools", all(n in names for n in ("start_session", "capture_image", "finish", "run_inspection")), ", ".join(names))
    res = app.call("list_samples", {})
    checks.add("MCP list_samples", not res.is_error and res.data.get("count", 0) >= 4 and CATEGORY in res.data.get("model_categories", []),
               f"{res.data.get('count')} samples, model categories {res.data.get('model_categories')}")
    res = app.call("start_session", {"sample_id": f"{CATEGORY}_anomaly_1", "preset": "defocus", "seed": seed})
    if not checks.add("MCP start_session", not res.is_error and bool(res.data.get("session_id")), json.dumps(res.data)[:200]):
        return out
    sid = res.data["session_id"]
    ms = app._get(sid)  # the in-memory session (planner reads its state like an MCP client would read observations)
    session = ms.session
    planner = RulePlanner()
    tools_called: list[str] = []
    verdict = None
    pol: PolicyConfig = settings.policy
    for _ in range(pol.max_steps + 2):
        d = planner.next_action(session.planner_state(), session)
        args = dict(d.args)
        args["session_id"] = sid
        args["step_rationale"] = d.rationale[:1000]
        res = app.call(d.tool, args)
        tools_called.append(d.tool + ("(err)" if res.is_error else ""))
        # results must convert to MCP CallToolResult (JSON text + images)
        app.to_call_result(res)
        if res.data.get("session_closed"):
            verdict = res.data.get("verdict")
            break
    checks.add("MCP session driven by RulePlanner reaches a verdict within budget", verdict is not None and len(tools_called) <= pol.max_steps,
               f"{len(tools_called)} calls: {' '.join(tools_called)} -> {verdict and verdict.get('label')}")
    caps = [s.decision.args for s in ms.steps if s.decision.tool == "capture_image"]
    if any("blurred" in (s.observation.get("issues") or []) for s in ms.steps if s.decision.tool == "assess_quality"):
        checks.add("MCP defocus session: blur answered with autofocus", any(c.get("autofocus") for c in caps), str(caps))
    res = app.call("get_trace", {"session_id": sid})
    tr_ok = not res.is_error and res.data.get("verdict") is not None
    try:
        _strict_json(res.data)
    except (TypeError, ValueError):
        tr_ok = False
    checks.add("MCP get_trace (strict JSON, verdict, ground truth revealed after verdict)", tr_ok and res.data.get("ground_truth") == {"label": 1},
               f"{len(res.data.get('steps', []))} steps, gt {res.data.get('ground_truth')}")
    arts = [a for s in ms.steps for a in s.artifacts]
    if arts:
        res = app.call("get_artifact", {"session_id": sid, "artifact_id": arts[-1]})
        checks.add("MCP get_artifact returns an image", not res.is_error and len(res.images) == 1, f"{arts[-1]}")
    res = app.call("capture_image", {"session_id": sid})
    checks.add("MCP tool call on a finished session is rejected", res.is_error, res.data.get("error", "")[:120])
    res = app.call("close_session", {"session_id": sid})
    checks.add("MCP close_session", not res.is_error and res.data.get("closed") is True)
    t0 = time.perf_counter()
    result = anyio.run(app.acall, "run_inspection", {"sample_id": f"{CATEGORY}_normal_1", "preset": "glare", "seed": seed})
    wall = (time.perf_counter() - t0) * 1000.0
    sc = result.structured_content or {}
    v = sc.get("verdict") or {}
    checks.add("MCP run_inspection (async CallToolResult)", not result.is_error and bool(v),
               f"verdict {v.get('label')} p={v.get('prob_anomalous')}, {len(sc.get('steps', []))} steps, {wall:.0f} ms, {len(result.content)} content blocks")
    out.update({"session_tools": tools_called, "session_verdict": verdict, "run_inspection_verdict": v})
    return out


# ---------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-fit", type=int, default=24, help="synthetic normals for the template (default 24)")
    ap.add_argument("--presets", nargs="+", default=list(ALL_PRESETS), choices=ALL_PRESETS)
    ap.add_argument("--per-preset", type=int, default=3, help="normal and defective boards per preset (default 3 + 3)")
    ap.add_argument("--no-deep", action="store_true", help="do not fit the PatchCoreDNN detector")
    ap.add_argument("--coreset", type=int, default=2000, help="deep memory-bank size (default 2000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-api", action="store_true")
    ap.add_argument("--skip-mcp", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the temp dir (bundle, storage, samples)")
    ap.add_argument("--verbose", action="store_true", help="keep the JSON step/HTTP logs of the API app (log level INFO)")
    ap.add_argument("--json", type=Path, default=None, help="write the full report as JSON")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    work = Path(tempfile.mkdtemp(prefix="inspectagent_smoke_"))
    models_dir, samples_dir, storage_dir, results_dir = (work / n for n in ("models", "samples", "runs", "results"))
    for d in (models_dir, samples_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)
    settings = Settings(models_dir=models_dir, samples_dir=samples_dir, results_dir=results_dir, local_storage_dir=storage_dir,
                        storage="local", bedrock_model_id="", log_level="INFO" if args.verbose else "WARNING")
    _log(f"[smoke] work dir {work}")
    checks = Checks()
    report: dict[str, Any] = {"work_dir": str(work), "opencv": cv2.__version__}
    try:
        report["bundle"] = build_bundle(models_dir, settings, args.n_fit, not args.no_deep, args.coreset, args.seed)
        clear_bundle_cache()
        bundle = load_bundle(CATEGORY, models_dir)
        checks.add("bundle saved and loaded back via models_io", bundle.stat is not None and bundle.has_deep == report["bundle"]["deep"],
                   f"files {report['bundle']['files']}, deep_load_status {bundle.meta.get('deep_load_status')}")

        _log(f"\n[agent] run_inspection: presets {args.presets}, {args.per_preset} normal + {args.per_preset} defective boards each")
        rows = run_agent_matrix(bundle, settings, list(args.presets), args.per_preset, args.seed)
        report["runs"] = rows
        print_table(rows)
        check_agent(rows, checks, list(args.presets))

        if not args.skip_api or not args.skip_mcp:
            report["samples"] = write_samples(samples_dir)
        if not args.skip_api:
            _log("\n[api] FastAPI TestClient")
            report["api"] = run_api(settings, storage_dir, checks, args.seed)
        if not args.skip_mcp:
            _log("[mcp] InspectAgentMCP tool functions")
            report["mcp"] = run_mcp(settings, samples_dir / "samples.json", checks, args.seed)
    except Exception as exc:  # unexpected crash: report it as a failed check
        import traceback

        traceback.print_exc()
        checks.add("smoke test completed without an unexpected exception", False, f"{type(exc).__name__}: {exc}")
    finally:
        clear_bundle_cache()
        checks.report()
        report["checks"] = [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks.items]
        report["ok"] = checks.ok
        report["wall_s"] = round(time.perf_counter() - t_start, 1)
        _log(f"[smoke] {'OK' if checks.ok else 'FAILED'} in {report['wall_s']:.1f}s")
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            _log(f"[smoke] report written to {args.json}")
        if args.keep:
            _log(f"[smoke] kept {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    return 0 if checks.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
