"""Evaluate InspectAgent against single-shot baselines on the held-out VisA eval split (SPEC 7).

Usage (after ``prepare_visa.py`` and ``build_models.py``)::

    python scripts/evaluate.py                                   # everything, all cores
    python scripts/evaluate.py --categories pcb1 --limit 20 --systems single_fast agent_rule
    python scripts/evaluate.py --bedrock-model <model-or-profile-id> --bedrock-limit 30
    python scripts/evaluate.py --summarize-only                  # rebuild summary/traces/failures

Systems
    ``oracle_fast``  clean (degradation-free) full-view capture, fast detector, single shot
    ``single_fast``  one default capture (centre, zoom 1, EV 0, ring light, no AF), fast detector
    ``single_deep``  the same capture, deep detector
    ``agent_rule``   full agent loop with the deterministic RulePlanner + human-control policy
    ``agent_bedrock`` agent loop with the Amazon Bedrock planner (opt-in subset, needs AWS credentials)

Single-shot baselines are decided with the *same* human-control thresholds as the agent
(PASS iff reliable and p <= auto_pass_max_prob, FAIL iff reliable and p >= auto_fail_min_prob,
otherwise NEEDS_REVIEW), so automation/review rates are comparable.

Regimes ``nominal`` and ``harsh``: the hidden nuisance conditions of each part are drawn by
``make_conditions(regime, rng)`` with ``rng`` seeded from ``(part_id, regime)``; every system
sees the same conditions and camera seed for a given part and regime.

Outputs (``--out``, default ``results/``)
    eval_items.csv     one row per (part, regime, system); appended as results arrive (resumable)
    eval_summary.json  metrics per category x regime x system and overall (category "ALL")
    traces/            curated agent traces (quality->capture change, zoom confirm/refute,
                       deep escalation decided, human review) + index.json
    failures/          overlays of agent_rule escapes / false rejects + index.json
    cache/             all agent traces and pixel samples (git-ignored)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inspectagent.config import CATEGORIES, get_settings  # noqa: E402
from inspectagent.eval_utils import (  # noqa: E402
    FAIL,
    METRIC_DEFINITIONS,
    PASS,
    REVIEW,
    auroc,
    binary_metrics,
    bootstrap_ci,
    decide_three_way,
    decision_metrics,
    describe,
    evidence_action_flags,
    nan_to_none,
    normalize_decision,
    part_id_from_relpath,
    resample_curve,
    risk_coverage,
    sample_pixels,
    split_fraction,
    stable_seed,
    stratified_subsample,
    tool_histogram,
)

SYSTEMS = ("oracle_fast", "single_fast", "single_deep", "agent_rule", "agent_bedrock")
DEFAULT_SYSTEMS = ("oracle_fast", "single_fast", "single_deep", "agent_rule")
SINGLE_SHOT = {"oracle_fast": ("fast", True), "single_fast": ("fast", False), "single_deep": ("deep", False)}
REGIMES = ("nominal", "harsh")
RC_GRID = [round(x, 2) for x in np.linspace(0.0, 1.0, 51)]
PIXEL_MAX_POS = 20000
PIXEL_MAX_NEG = 20000

FIELDNAMES = [
    "part_id", "category", "label", "regime", "system", "decision", "prob", "score", "reliable",
    "n_steps", "n_captures", "n_deep", "compute_ms", "sim_time_ms", "cycle_time_ms", "wall_ms",
    "tool_counts", "ev_quality_recapture", "ev_zoom", "ev_zoom_confirmed", "ev_zoom_refuted", "ev_zoom_inconclusive",
    "ev_deep", "ev_review", "ev_review_visual", "ev_review_nonvisual", "ev_any", "policy_overrides", "planner",
    "n_fallback_steps", "fallback_share", "bedrock_calls", "bedrock_input_tokens", "bedrock_output_tokens",
    "bedrock_latency_ms", "trace_id", "trace_file", "failure_kind", "failure_overlay", "error", "seed", "image", "mask",
    "conditions",
]  # fmt: skip
EV_KEYS = ("quality_recapture", "zoom", "zoom_confirmed", "zoom_refuted", "zoom_inconclusive", "deep", "review", "review_visual",
           "review_nonvisual", "any")  # fmt: skip
_BOOL_FIELDS = {"reliable", *(f"ev_{k}" for k in EV_KEYS)}
_FLOAT_FIELDS = {"prob", "score", "compute_ms", "sim_time_ms", "cycle_time_ms", "wall_ms", "fallback_share", "bedrock_latency_ms"}
_INT_FIELDS = {"label", "n_steps", "n_captures", "n_deep", "policy_overrides", "seed", "n_fallback_steps", "bedrock_calls",
               "bedrock_input_tokens", "bedrock_output_tokens"}  # fmt: skip
# agent_bedrock rows whose every decision came from the model (no rule-planner fallback) are also summarised
# as this pseudo-system, so fallback-only runs (e.g. model access not granted) cannot pass as Bedrock results
BEDROCK_PURE = "agent_bedrock_no_fallback"


def _log(msg: str) -> None:
    print(msg, flush=True)


# =========================================================================== CSV I/O


def _to_bool(v: Any) -> bool | None:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _to_float(v: Any) -> float:
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _to_int(v: Any) -> int | None:
    f = _to_float(v)
    return int(f) if math.isfinite(f) else None


def parse_row(r: dict) -> dict:
    """Typed copy of a CSV/in-memory row (idempotent)."""
    out = dict(r)
    for k in _BOOL_FIELDS:
        if k in out:
            out[k] = _to_bool(out[k])
    for k in _FLOAT_FIELDS:
        if k in out:
            out[k] = _to_float(out[k])
    for k in _INT_FIELDS:
        if k in out:
            out[k] = _to_int(out[k])
    for k in ("error", "decision", "failure_overlay", "failure_kind", "trace_file"):
        if out.get(k) is None:
            out[k] = ""
    return out


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return "" if not math.isfinite(f) else repr(round(f, 6))
    if isinstance(v, (dict, list)):
        return json.dumps(v, separators=(",", ":"))
    return str(v)


def append_rows(path: Path, rows: Iterable[dict]) -> None:
    """Append rows to the CSV (header written for a new file; existing header must match)."""
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file() or path.stat().st_size == 0
    if not new:
        with path.open("r", encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), [])
        if header != FIELDNAMES:
            raise SystemExit(f"{path} has a different header; move it away or use --fresh")
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in FIELDNAMES})
        f.flush()


def read_rows(path: Path) -> list[dict]:
    """All rows of the CSV, typed (empty list if the file does not exist)."""
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [parse_row(r) for r in csv.DictReader(f)]


def _key(r: dict) -> tuple[str, str, str]:
    return (str(r["part_id"]), str(r["regime"]), str(r["system"]))


def latest_rows(rows: Iterable[dict]) -> list[dict]:
    """Keep the last row per (part, regime, system) - later attempts supersede earlier ones."""
    d: dict[tuple, dict] = {}
    for r in rows:
        d[_key(r)] = r
    return list(d.values())


def done_keys(rows: Iterable[dict], include_errors: bool = True) -> set[tuple[str, str, str]]:
    """Keys already evaluated (optionally treating error rows as not done)."""
    return {_key(r) for r in latest_rows(rows) if include_errors or not r.get("error")}


# =========================================================================== summary


def _flag_rate(rows: list[dict], key: str) -> float:
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    return float(np.mean([bool(v) for v in vals])) if vals else float("nan")


def group_metrics(rows: list[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    """All SPEC 7 metrics of one group of rows (single system)."""
    labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    decisions = [normalize_decision(r.get("decision") or None) for r in rows]
    probs = np.array([_to_float(r.get("prob")) for r in rows], dtype=np.float64)
    dm = decision_metrics(labels, decisions)
    out: dict[str, Any] = {**dm}
    out["auroc"] = auroc(probs, labels)
    finite = np.isfinite(probs)
    bm = binary_metrics(labels[finite], (probs[finite] >= 0.5).astype(np.int64))
    out.update({f"bin05_{k}": v for k, v in bm.items() if k in ("accuracy", "precision", "recall", "f1", "specificity")})

    dec = np.array(decisions)
    pos, neg = labels == 1, labels == 0
    ci: dict[str, list[float]] = {}
    ci["auroc"] = list(bootstrap_ci(auroc, [probs, labels], n_boot=n_boot, seed=seed))
    esc = (dec[pos] == PASS).astype(np.float64)
    frr = (dec[neg] == FAIL).astype(np.float64)
    rev = (dec == REVIEW).astype(np.float64)
    acc_auto_ok = ((dec == FAIL) & pos) | ((dec == PASS) & neg)
    auto = dec != REVIEW
    ci["escape_rate"] = list(bootstrap_ci(np.mean, [esc], n_boot=n_boot, seed=seed + 1, min_valid=1)) if esc.size else [float("nan")] * 2
    ci["false_reject_rate"] = list(bootstrap_ci(np.mean, [frr], n_boot=n_boot, seed=seed + 2, min_valid=1)) if frr.size else [float("nan")] * 2
    ci["review_rate"] = list(bootstrap_ci(np.mean, [rev], n_boot=n_boot, seed=seed + 3, min_valid=1)) if rev.size else [float("nan")] * 2
    ci["accuracy_auto"] = (
        list(bootstrap_ci(np.mean, [acc_auto_ok[auto].astype(np.float64)], n_boot=n_boot, seed=seed + 4, min_valid=1))
        if auto.any()
        else [float("nan")] * 2
    )
    out["ci95"] = ci

    for k in ("n_steps", "n_captures", "n_deep", "compute_ms", "sim_time_ms", "cycle_time_ms", "wall_ms"):
        d = describe(_to_float(r.get(k)) for r in rows)
        out[f"mean_{k}"] = d["mean"]
        if k in ("cycle_time_ms", "compute_ms", "n_steps"):
            out[f"median_{k}"] = d["median"]
            out[f"p90_{k}"] = d["p90"]

    tcs = []
    for r in rows:
        tc = r.get("tool_counts")
        if isinstance(tc, str) and tc.strip():
            try:
                tc = json.loads(tc)
            except json.JSONDecodeError:
                tc = None
        if isinstance(tc, dict):
            tcs.append(tc)
    out["tool_usage"] = tool_histogram(tcs) if tcs else None
    out["evidence_rates"] = {k: _flag_rate(rows, f"ev_{k}") for k in EV_KEYS}
    out["evidence_changed_action_rate"] = out["evidence_rates"]["any"]
    shares = [_to_float(r.get("fallback_share")) for r in rows]
    shares = [s for s in shares if math.isfinite(s)]
    if shares:
        out["mean_fallback_share"] = float(np.mean(shares))
        out["runs_with_fallback_rate"] = float(np.mean([s > 0 for s in shares]))
    for k in ("bedrock_calls", "bedrock_input_tokens", "bedrock_output_tokens", "bedrock_latency_ms"):
        vals = [_to_float(r.get(k)) for r in rows]
        vals = [v for v in vals if math.isfinite(v)]
        if vals:
            out[f"mean_{k}"] = float(np.mean(vals))
    rc = risk_coverage(probs, labels)
    out["aurc"] = rc["aurc"]
    reliable = [r.get("reliable") for r in rows if r.get("reliable") is not None]
    out["reliable_rate"] = float(np.mean(reliable)) if reliable else float("nan")
    return out


def summarize(rows: list[dict], pixel: dict | None = None, n_boot: int = 1000, seed: int = 0, meta: dict | None = None) -> dict:
    """Aggregate rows into the eval_summary.json structure (strict-JSON safe)."""
    rows = [parse_row(r) for r in latest_rows(rows)]
    ok_rows = [r for r in rows if not r.get("error") and r.get("label") is not None]
    errors = [r for r in rows if r.get("error")]
    # model-only Bedrock runs (no rule-planner fallback decision) as an extra pseudo-system
    ok_rows += [
        {**r, "system": BEDROCK_PURE}
        for r in ok_rows
        if r["system"] == "agent_bedrock" and math.isfinite(_to_float(r.get("fallback_share"))) and _to_float(r.get("fallback_share")) == 0.0
    ]
    cats = [c for c in CATEGORIES if any(r["category"] == c for r in ok_rows)]
    cats += sorted({r["category"] for r in ok_rows} - set(cats))
    regimes = [g for g in REGIMES if any(r["regime"] == g for r in ok_rows)]
    regimes += sorted({r["regime"] for r in ok_rows} - set(regimes))
    systems = [s for s in (*SYSTEMS, BEDROCK_PURE) if any(r["system"] == s for r in ok_rows)]
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in ok_rows:
        groups[(r["category"], r["regime"], r["system"])].append(r)
        groups[("ALL", r["regime"], r["system"])].append(r)
    results: dict[str, dict] = {}
    for cat in [*cats, "ALL"]:
        for regime in regimes:
            for system in systems:
                g = groups.get((cat, regime, system))
                if g:
                    seed_g = stable_seed(seed, cat, regime, system) % 100_000
                    results.setdefault(cat, {}).setdefault(regime, {})[system] = group_metrics(g, n_boot, seed_g)
    curves: dict[str, dict] = {}
    for regime in regimes:
        for system in systems:
            g = groups.get(("ALL", regime, system))
            if not g:
                continue
            probs = [_to_float(r.get("prob")) for r in g]
            labels = [int(r["label"]) for r in g]
            rc = risk_coverage(probs, labels)
            curves.setdefault(regime, {})[system] = {
                "coverage": RC_GRID,
                "accuracy": resample_curve(rc["coverage"], rc["accuracy"], RC_GRID),
                "aurc": rc["aurc"],
            }
    err_counts: dict[str, int] = defaultdict(int)
    for r in errors:
        err_counts[r["system"]] += 1
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": len(rows),
        "n_ok": len(ok_rows),
        "n_errors": len(errors),
        "errors_by_system": dict(err_counts),
        "error_examples": sorted({r["error"][:200] for r in errors})[:10],
        "categories": cats,
        "regimes": regimes,
        "systems": systems,
        "definitions": METRIC_DEFINITIONS,
        "meta": meta or {},
        "results": results,
        "overall": results.get("ALL", {}),
        "risk_coverage": curves,
        "pixel_auroc": pixel or {},
    }
    return nan_to_none(summary)


# =========================================================================== worker side

_BUNDLE: dict[str, Any] = {}


def _init_worker(threads: int) -> None:
    cv2.setNumThreads(max(1, int(threads)))
    try:
        from inspectagent.vision.cvinfo import configure_threads

        configure_threads(max(1, int(threads)))
    except Exception:
        pass


def _bundle(category: str, models_dir: str):
    """Load one category bundle per worker at a time (tasks arrive grouped by category)."""
    from inspectagent import models_io

    if category not in _BUNDLE:
        if _BUNDLE:
            _BUNDLE.clear()
            clear = getattr(models_io, "clear_bundle_cache", None) or getattr(models_io.load_bundle, "cache_clear", None)
            if callable(clear):
                clear()
        _BUNDLE[category] = models_io.load_bundle(category, Path(models_dir), with_deep=True)
    return _BUNDLE[category]


def _imread(path: Path | str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    from inspectagent.vision.imageio import imread

    return imread(path, flags)


def make_part(category: str, item: dict, regime: str, data_root: Path):
    """PartSpec with hidden conditions seeded from (part_id, regime); returns (part, seed)."""
    from inspectagent.types import PartSpec
    from inspectagent.vision.camera import make_conditions

    pid = part_id_from_relpath(item["image"])
    cond = make_conditions(regime, np.random.default_rng(stable_seed(pid, regime)))
    part = PartSpec(
        part_id=pid,
        category=category,
        image_path=str(data_root / item["image"]),
        label=int(item["label"]),
        mask_path=str(data_root / item["mask"]) if item.get("mask") else None,
        conditions=cond,
    )
    return part, stable_seed(pid, regime, "camera")


def _first(d: dict, *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def single_shot_fields(d: dict, pass_max: float, fail_min: float) -> dict:
    """Normalize a ``run_single_shot`` result dict into CSV fields (decision re-derived from prob)."""
    res = d.get("result") if isinstance(d.get("result"), dict) else {}
    prob = _first(d, "prob", "prob_anomalous", "final_prob")
    prob = _first(res, "prob") if prob is None else prob
    score = _first(d, "score")
    score = _first(res, "score") if score is None else score
    reliable = _first(d, "reliable")
    reliable = _first(res, "reliable") if reliable is None else reliable
    reliable = True if reliable is None else bool(reliable)
    compute = _to_float(_first(d, "compute_ms", "latency_ms"))
    if not math.isfinite(compute):
        compute = _to_float(_first(res, "latency_ms"))
    sim = _to_float(_first(d, "sim_time_ms"))
    cycle = _to_float(_first(d, "cycle_time_ms"))
    if not math.isfinite(cycle):
        cycle = (compute if math.isfinite(compute) else 0.0) + (sim if math.isfinite(sim) else 0.0)
    p = _to_float(prob)
    return {
        "decision": decide_three_way(p if math.isfinite(p) else None, reliable, pass_max, fail_min),
        "prob": p,
        "score": _to_float(score),
        "reliable": reliable,
        "n_steps": 1,
        "n_captures": 1,
        "n_deep": 1 if str(_first(d, "detector") or res.get("detector") or "") == "deep" else 0,
        "compute_ms": compute,
        "sim_time_ms": sim,
        "cycle_time_ms": cycle,
    }


def agent_fields(trace_d: dict) -> dict:
    """CSV fields of an agent run from ``Trace.to_dict()``."""
    v = trace_d.get("verdict") or {}
    totals = trace_d.get("totals") or {}
    prob = v.get("prob_anomalous")
    if prob is None:
        prob = totals.get("final_prob")
    steps = trace_d.get("steps") or []
    flags = evidence_action_flags(steps)
    compute = _to_float(totals.get("compute_ms"))
    sim = _to_float(totals.get("sim_time_ms"))
    cycle = _to_float(totals.get("cycle_time_ms"))
    if not math.isfinite(cycle) and (math.isfinite(compute) or math.isfinite(sim)):
        cycle = (compute if math.isfinite(compute) else 0.0) + (sim if math.isfinite(sim) else 0.0)
    # planner accounting from the steps themselves (works for cached traces without the newer totals)
    n_fallback = sum(1 for s in steps if "fallback" in str((s.get("decision") or {}).get("planner", "")))
    bedrock = totals.get("bedrock") if isinstance(totals.get("bedrock"), dict) else {}
    is_bedrock = str(trace_d.get("planner", "")) == "bedrock"
    return {
        "decision": normalize_decision(v.get("label")),
        "prob": _to_float(prob),
        "score": _to_float(prob),
        "reliable": None,
        "n_steps": _to_int(totals.get("n_steps", len(trace_d.get("steps") or []))),
        "n_captures": _to_int(totals.get("n_captures")),
        "n_deep": _to_int(totals.get("n_deep")),
        "compute_ms": compute,
        "sim_time_ms": sim,
        "cycle_time_ms": cycle,
        "tool_counts": totals.get("tool_counts") or {},
        **{f"ev_{k}": flags[k] for k in EV_KEYS},
        "policy_overrides": flags["policy_overrides"],
        "planner": trace_d.get("planner", ""),
        "n_fallback_steps": n_fallback,
        "fallback_share": (n_fallback / len(steps)) if steps else float("nan"),
        "bedrock_calls": _to_int(bedrock.get("calls", 0)) if is_bedrock else None,
        "bedrock_input_tokens": _to_int(bedrock.get("input_tokens", 0)) if is_bedrock else None,
        "bedrock_output_tokens": _to_int(bedrock.get("output_tokens", 0)) if is_bedrock else None,
        "bedrock_latency_ms": _to_float(bedrock.get("latency_ms", 0.0)) if is_bedrock else float("nan"),
        "trace_id": trace_d.get("trace_id", ""),
    }


def _heatmap(bgr: np.ndarray, amap: np.ndarray) -> np.ndarray:
    try:
        from inspectagent.vision.overlay import heatmap_overlay

        out = heatmap_overlay(bgr, amap)
        if out is not None and out.shape[:2] == bgr.shape[:2]:
            return np.ascontiguousarray(out[..., :3]).astype(np.uint8)
    except Exception:
        pass
    a = np.asarray(amap, np.float32)
    if a.shape[:2] != bgr.shape[:2]:
        a = cv2.resize(a, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    vmax = float(np.percentile(a, 99.9)) if a.size else 1.0
    vmax = vmax if vmax > 1e-6 else 1.0
    u8 = np.clip(a / vmax * 255.0, 0, 255).astype(np.uint8)
    col = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    alpha = (np.clip(a / vmax, 0, 1) * 0.6)[..., None]
    return (bgr.astype(np.float32) * (1 - alpha) + col.astype(np.float32) * alpha).astype(np.uint8)


def _put_lines(img: np.ndarray, lines: list[str], scale: float = 0.6) -> np.ndarray:
    """Caption bar below an image (uint8 only - OpenCV 5 text needs 8-bit)."""
    h_line = int(30 * scale / 0.6)
    bar = np.full((h_line * len(lines) + 12, img.shape[1], 3), 245, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(bar, t, (10, h_line * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def failure_overlay(session: Any, scene: np.ndarray, mask: np.ndarray | None, caption: list[str]) -> np.ndarray:
    """Side-by-side figure: scene with GT (green) + predicted regions (red) | decisive capture heatmap."""
    target_h = 540
    left = scene.copy()
    if mask is not None and mask.shape[:2] == scene.shape[:2]:
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(left, contours, -1, (0, 200, 0), max(2, scene.shape[1] // 300))
    regions = list(getattr(session, "regions", {}).values()) if hasattr(session, "regions") else []
    thick = max(2, scene.shape[1] // 400)
    for reg in regions:
        x, y, w, h = (int(round(v)) for v in reg.bbox_scene)
        cv2.rectangle(left, (x, y), (x + w, y + h), (0, 0, 255), thick)
    s = target_h / left.shape[0]
    left = cv2.resize(left, (int(round(left.shape[1] * s)), target_h), interpolation=cv2.INTER_AREA)

    right = None
    results = list(getattr(session, "results", {}).values())
    captures = getattr(session, "captures", {})
    for res in reversed(results):
        amap = getattr(res, "anomaly_map", None)
        cap = captures.get(res.capture_id)
        if amap is not None and cap is not None:
            right = _heatmap(cap.image, amap)
            for reg in res.regions:
                x, y, w, h = (int(v) for v in reg.bbox_capture)
                cv2.rectangle(right, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(right, f"{res.capture_id} {res.detector} p={res.prob:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            break
    if right is None and captures:
        right = list(captures.values())[-1].image.copy()
    if right is None:
        right = np.zeros_like(left)
    s = target_h / right.shape[0]
    right = cv2.resize(right, (int(round(right.shape[1] * s)), target_h), interpolation=cv2.INTER_AREA)
    gap = np.full((target_h, 8, 3), 255, np.uint8)
    return _put_lines(np.hstack([left, gap, right]), caption)


def run_part_task(task: dict) -> list[dict]:
    """Evaluate one (part, regime) for the requested systems; returns CSV rows."""
    from inspectagent.agent.loop import run_inspection, run_single_shot

    cfg = task["cfg"]
    settings = get_settings()
    pcfg = settings.policy
    category, item, regime = task["category"], task["item"], task["regime"]
    data_root = Path(cfg["data_root"])
    part, seed = make_part(category, item, regime, data_root)
    base = {
        "part_id": part.part_id,
        "category": category,
        "label": int(item["label"]),
        "regime": regime,
        "seed": seed,
        "image": item["image"],
        "mask": item.get("mask") or "",
        "conditions": part.conditions.to_dict(),
    }
    rows: list[dict] = []
    try:
        bundle = _bundle(category, cfg["models_dir"])
    except Exception as exc:
        return [{**base, "system": s, "error": f"bundle: {type(exc).__name__}: {exc}"} for s in task["systems"]]
    try:
        scene = _imread(part.image_path)  # decoded once, shared by every system of this part
    except Exception as exc:
        return [{**base, "system": s, "error": f"image: {type(exc).__name__}: {exc}"} for s in task["systems"]]
    for system in task["systems"]:
        row = {**base, "system": system}
        t0 = time.perf_counter()
        try:
            if system in SINGLE_SHOT:
                detector, oracle = SINGLE_SHOT[system]
                if detector == "deep" and getattr(bundle, "deep", None) is None:
                    raise RuntimeError("deep model unavailable for this category")
                d = run_single_shot(part, bundle, detector=detector, oracle=oracle, seed=seed, settings=settings, scene=scene)
                row.update(single_shot_fields(d, pcfg.auto_pass_max_prob, pcfg.auto_fail_min_prob))
            else:
                planner = None
                if system == "agent_bedrock":
                    from inspectagent.agent.planner_bedrock import BedrockPlanner

                    planner = BedrockPlanner(cfg["bedrock_model"], region=settings.aws_region, max_tokens=settings.bedrock_max_tokens)
                trace, session = run_inspection(part, bundle, planner=planner, settings=settings, seed=seed, scene=scene)
                td = trace.to_dict()
                row.update(agent_fields(td))
                tpath = Path(cfg["cache_dir"]) / "traces" / system / f"{part.part_id}__{regime}.json"
                _save_trace(trace, td, tpath)
                row["trace_file"] = str(tpath)
                if system == "agent_rule":
                    row.update(_maybe_failure(row, session, scene, part, Path(cfg["out_dir"])))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:500]
            if cfg.get("verbose"):
                traceback.print_exc()
        row["wall_ms"] = (time.perf_counter() - t0) * 1000.0
        rows.append(row)
    return rows


def _save_trace(trace: Any, td: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from inspectagent.agent.trace import save_trace

        save_trace(trace, path)
        if path.is_file():
            return
    except Exception:
        pass
    path.write_text(json.dumps(nan_to_none(td), indent=1), encoding="utf-8")


def _maybe_failure(row: dict, session: Any, scene: np.ndarray, part: Any, out_dir: Path) -> dict:
    """Write an overlay for escapes / false rejects of the agent; returns the CSV fields."""
    label, decision = int(row["label"]), row.get("decision")
    kind = "escape" if (label == 1 and decision == PASS) else "false_reject" if (label == 0 and decision == FAIL) else ""
    if not kind:
        return {}
    mask = None
    if part.mask_path:
        try:
            mask = _imread(part.mask_path, cv2.IMREAD_GRAYSCALE)
        except (FileNotFoundError, ValueError):
            mask = None
    p = row.get("prob")
    tools = ", ".join(f"{k}x{v}" for k, v in sorted((row.get("tool_counts") or {}).items()))
    caption = [
        f"{kind.upper()}: {part.part_id} | regime={row['regime']} | GT={'anomalous' if label else 'normal'} -> {decision} (p={p:.2f})",
        f"steps={row.get('n_steps')} captures={row.get('n_captures')} deep={row.get('n_deep')} | {tools}"[:150],
        "left: scene, green = ground-truth defect, red = predicted regions | right: last detector heatmap",
    ]
    img = failure_overlay(session, scene, mask, caption)
    rel = f"failures/{kind}__{part.part_id}__{row['regime']}.jpg"
    from inspectagent.vision.imageio import imwrite

    imwrite(out_dir / rel, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {"failure_kind": kind, "failure_overlay": rel}


def run_pixel_task(task: dict) -> dict:
    """Pixel-level samples of fast/deep maps on an oracle (clean, no pose jitter) full-view capture."""
    from inspectagent.types import NuisanceConditions, View
    from inspectagent.vision.camera import VirtualCamera

    cfg = task["cfg"]
    settings = get_settings()
    category, item = task["category"], task["item"]
    pid = part_id_from_relpath(item["image"])
    out: dict[str, Any] = {"part_id": pid, "category": category}
    try:
        bundle = _bundle(category, cfg["models_dir"])
        data_root = Path(cfg["data_root"])
        scene = _imread(data_root / item["image"])
        mask = _imread(data_root / item["mask"], cv2.IMREAD_GRAYSCALE)
        cond = NuisanceConditions(ambient_ev=0.0, defocus_sigma=0.0, noise_sigma=0.0, glare_strength=0.0, rot_deg=0.0, shift_x=0.0, shift_y=0.0, jpeg_quality=95, preset="oracle")
        cam = VirtualCamera(scene, cond, settings.camera, seed=stable_seed(pid, "pixel"))
        cap = cam.clean_capture(View(0.5, 0.5, 1.0), "px_0", 0)
        w, h = cap.size
        m = (mask > 0).astype(np.float32)
        mask_c = (cv2.warpAffine(m, cap.scene_to_capture(), (w, h), flags=cv2.INTER_LINEAR) >= 0.5).astype(np.uint8)
        for det in ("fast", "deep"):
            model = bundle.stat if det == "fast" else getattr(bundle, "deep", None)
            if model is None:
                continue
            res = model.predict(cap)
            amap = getattr(res, "anomaly_map", None)
            if amap is None:
                continue
            amap = np.asarray(amap, np.float32)
            if amap.shape[:2] != (h, w):
                amap = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)
            rng = np.random.default_rng(stable_seed(pid, det, "pixels"))
            s, y = sample_pixels(amap, mask_c, PIXEL_MAX_POS, PIXEL_MAX_NEG, rng)
            path = Path(cfg["cache_dir"]) / "pixel" / det / f"{pid}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, scores=s.astype(np.float32), labels=y.astype(np.uint8))
            out[det] = auroc(s, y)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def _dispatch(task: dict) -> tuple[str, Any]:
    if task["kind"] == "pixel":
        return "pixel", run_pixel_task(task)
    return "part", run_part_task(task)


# =========================================================================== main side


def aggregate_pixel(cache_dir: Path, parts_by_cat: dict[str, list[str]]) -> dict:
    """Pooled and mean-per-image pixel AUROC per category/detector from cached samples."""
    out: dict[str, dict] = {}
    pooled_all: dict[str, tuple[list, list]] = {"fast": ([], []), "deep": ([], [])}
    per_img_all: dict[str, list[float]] = {"fast": [], "deep": []}
    for cat, pids in parts_by_cat.items():
        for det in ("fast", "deep"):
            ss, yy, per = [], [], []
            for pid in pids:
                f = cache_dir / "pixel" / det / f"{pid}.npz"
                if not f.is_file():
                    continue
                with np.load(f) as z:
                    s, y = z["scores"], z["labels"]
                ss.append(s)
                yy.append(y)
                per.append(auroc(s, y))
            if not ss:
                continue
            fin = [v for v in per if math.isfinite(v)]
            out.setdefault(cat, {})[det] = {
                "pooled": auroc(np.concatenate(ss), np.concatenate(yy)),
                "mean_per_image": float(np.mean(fin)) if fin else float("nan"),
                "n_images": len(ss),
            }
            pooled_all[det][0].extend(ss)
            pooled_all[det][1].extend(yy)
            per_img_all[det].extend(fin)
    for det, (ss, yy) in pooled_all.items():
        if ss:
            out.setdefault("ALL", {})[det] = {
                "pooled": auroc(np.concatenate(ss), np.concatenate(yy)),
                "mean_per_image": float(np.mean(per_img_all[det])) if per_img_all[det] else float("nan"),
                "n_images": len(ss),
            }
    out["_note"] = (
        "oracle full-view capture (no degradation, no pose jitter); per image up to "
        f"{PIXEL_MAX_POS} defect + {PIXEL_MAX_NEG} normal pixels sampled uniformly"
    )
    return out


def _is_correct(r: dict) -> bool:
    return (int(r["label"]) == 1 and r["decision"] == FAIL) or (int(r["label"]) == 0 and r["decision"] == PASS)


CURATED_KINDS = (
    ("quality_changed_capture", "quality evidence (glare/blur/exposure) changed the capture settings"),
    ("zoom_confirmed_or_refuted", "a zoomed re-inspection confirmed or refuted a candidate region"),
    ("deep_escalation_decided", "escalation to the deep detector led to an automatic decision"),
    ("human_review", "the evidence stayed ambiguous, so the part was routed to a human reviewer"),
)


def curate_traces(rows: list[dict], out_dir: Path, system: str = "agent_rule") -> list[dict]:
    """Copy one example trace per category and kind into ``out/traces`` (+ index.json)."""
    tdir = out_dir / "traces"
    tdir.mkdir(parents=True, exist_ok=True)
    for f in tdir.glob("*__*__*.json"):
        f.unlink()
    rows = [r for r in latest_rows(rows) if r["system"] == system and not r.get("error") and r.get("trace_file")]
    index: list[dict] = []
    for cat in sorted({r["category"] for r in rows}, key=lambda c: (CATEGORIES.index(c) if c in CATEGORIES else 99, c)):
        cand_all = sorted((r for r in rows if r["category"] == cat and Path(r["trace_file"]).is_file()), key=lambda r: (r["part_id"], r["regime"]))
        used: set[tuple] = set()
        for kind, desc in CURATED_KINDS:
            if kind == "quality_changed_capture":
                cand = [r for r in cand_all if r.get("ev_quality_recapture")]
                key = lambda r: (not _is_correct(r), r["regime"] != "harsh")  # noqa: E731
            elif kind == "zoom_confirmed_or_refuted":
                cand = [r for r in cand_all if r.get("ev_zoom_confirmed") or r.get("ev_zoom_refuted")]
                key = lambda r: (not _is_correct(r), not r.get("ev_zoom_refuted"))  # noqa: E731
            elif kind == "deep_escalation_decided":
                cand = [r for r in cand_all if r.get("ev_deep") and r["decision"] in (PASS, FAIL)]
                key = lambda r: (not _is_correct(r),)  # noqa: E731
            else:
                cand = [r for r in cand_all if r["decision"] == REVIEW]
                key = lambda r: (r["regime"] != "harsh", int(r["label"]) == 0)  # noqa: E731
            cand.sort(key=key)
            pick = next((r for r in cand if _key(r) not in used), cand[0] if cand else None)
            if pick is None:
                continue
            used.add(_key(pick))
            name = f"{cat}__{kind}__{pick['part_id']}__{pick['regime']}.json"
            shutil.copyfile(pick["trace_file"], tdir / name)
            index.append(
                {
                    "file": f"traces/{name}",
                    "category": cat,
                    "kind": kind,
                    "description": desc,
                    "part_id": pick["part_id"],
                    "regime": pick["regime"],
                    "label": int(pick["label"]),
                    "decision": pick["decision"],
                    "prob": pick.get("prob"),
                    "correct": _is_correct(pick) if pick["decision"] != REVIEW else None,
                    "n_steps": pick.get("n_steps"),
                }
            )
    (tdir / "index.json").write_text(json.dumps(nan_to_none(index), indent=2), encoding="utf-8")
    return index


def index_failures(rows: list[dict], out_dir: Path) -> list[dict]:
    """results/failures/index.json from rows carrying a failure overlay."""
    fdir = out_dir / "failures"
    fdir.mkdir(parents=True, exist_ok=True)
    idx = []
    for r in sorted(latest_rows(rows), key=lambda r: (r["category"], r["part_id"], r["regime"])):
        rel = r.get("failure_overlay")
        if not rel or r.get("error") or not (out_dir / rel).is_file():
            continue
        p = r.get("prob")
        idx.append(
            {
                "file": rel,
                "kind": r.get("failure_kind"),
                "category": r["category"],
                "part_id": r["part_id"],
                "regime": r["regime"],
                "label": int(r["label"]),
                "decision": r["decision"],
                "prob": p,
                "caption": f"{r.get('failure_kind')}: {r['part_id']} ({r['regime']}), ground truth "
                f"{'anomalous' if int(r['label']) else 'normal'} decided {r['decision']} at P(anomalous)="
                f"{p:.2f} after {r.get('n_steps')} steps",
            }
        )
    (fdir / "index.json").write_text(json.dumps(nan_to_none(idx), indent=2), encoding="utf-8")
    return idx


def _aws_credentials_available() -> bool:
    try:
        import boto3

        return boto3.Session().get_credentials() is not None
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=list(REGIMES))
    ap.add_argument("--systems", nargs="+", default=None, choices=list(SYSTEMS))
    ap.add_argument("--limit", type=int, default=0, help="max eval parts per category (label-stratified, 0 = all)")
    ap.add_argument("--workers", type=int, default=0, help="worker processes (0 = cpu_count - 2)")
    ap.add_argument("--splits", type=Path, default=settings.data_dir / "splits.json")
    ap.add_argument("--data-root", type=Path, default=settings.data_dir / "visa")
    ap.add_argument("--models-dir", type=Path, default=settings.models_dir)
    ap.add_argument("--out", type=Path, default=settings.results_dir)
    ap.add_argument("--bedrock-model", default=settings.bedrock_model_id or "", help="Bedrock model / inference profile id")
    ap.add_argument("--bedrock-limit", type=int, default=30, help="max parts x regimes for agent_bedrock (cost cap)")
    ap.add_argument("--pixel-parts", type=int, default=20, help="anomalous eval parts per category for pixel AUROC (0 = skip)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--retry-errors", action="store_true", help="re-run rows that previously ended with an error")
    ap.add_argument("--fresh", action="store_true", help="start a new eval_items.csv (the old one is renamed .bak)")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    items_csv = out_dir / "eval_items.csv"
    if args.fresh and items_csv.is_file():
        bak = items_csv.with_suffix(f".{datetime.now().strftime('%Y%m%d-%H%M%S')}.bak")
        items_csv.rename(bak)
        _log(f"[eval] previous results moved to {bak.name}")

    systems = list(args.systems) if args.systems else list(DEFAULT_SYSTEMS) + (["agent_bedrock"] if args.bedrock_model else [])
    if "agent_bedrock" in systems and (not args.bedrock_model or not _aws_credentials_available()):
        _log("[eval] agent_bedrock skipped: needs --bedrock-model and AWS credentials")
        systems = [s for s in systems if s != "agent_bedrock"]

    if not args.splits.is_file():
        raise SystemExit(f"{args.splits} not found - run scripts/prepare_visa.py first")
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    cats = [c for c in args.categories if c in splits and not c.startswith("_")]
    eval_items: dict[str, list[dict]] = {}
    for c in cats:
        items = list(splits[c]["eval"])
        if args.limit and args.limit < len(items):
            items = stratified_subsample(items, [int(d["label"]) for d in items], args.limit, stable_seed(0, c, "limit"))
        eval_items[c] = items
    pixel_parts: dict[str, list[dict]] = {}
    for c in cats:
        anoms = [d for d in eval_items[c] if int(d["label"]) == 1 and d.get("mask")]
        if args.pixel_parts and len(anoms) > args.pixel_parts:
            anoms = split_fraction(anoms, args.pixel_parts / len(anoms), stable_seed(0, c, "pixel"))[0][: args.pixel_parts]
        pixel_parts[c] = anoms if args.pixel_parts else []

    cfg = {
        "data_root": str(args.data_root),
        "models_dir": str(args.models_dir),
        "out_dir": str(out_dir),
        "cache_dir": str(cache_dir),
        "bedrock_model": args.bedrock_model,
        "verbose": args.verbose,
    }
    t_start = time.perf_counter()
    if not args.summarize_only:
        if not args.models_dir.is_dir():
            raise SystemExit(f"{args.models_dir} not found - run scripts/build_models.py first")
        existing = read_rows(items_csv)
        done = done_keys(existing, include_errors=not args.retry_errors)
        pool_systems = [s for s in systems if s != "agent_bedrock"]
        tasks: list[dict] = []
        for c in cats:
            for item in eval_items[c]:
                pid = part_id_from_relpath(item["image"])
                for regime in args.regimes:
                    todo = [s for s in pool_systems if (pid, regime, s) not in done]
                    if todo:
                        tasks.append({"kind": "part", "category": c, "item": item, "regime": regime, "systems": todo, "cfg": cfg})
            for item in pixel_parts[c]:
                pid = part_id_from_relpath(item["image"])
                cached = all((cache_dir / "pixel" / d / f"{pid}.npz").is_file() for d in ("fast",))
                if not cached:
                    tasks.append({"kind": "pixel", "category": c, "item": item, "cfg": cfg})
        n_rows_todo = sum(len(t["systems"]) for t in tasks if t["kind"] == "part")
        cpu = os.cpu_count() or 4
        workers = args.workers or max(1, cpu - 2)
        _log(f"[eval] {len(cats)} categories, systems={pool_systems}, regimes={args.regimes}: {n_rows_todo} rows to run "
             f"({len(done)} done), {sum(1 for t in tasks if t['kind'] == 'pixel')} pixel tasks, {workers} workers")
        if tasks:
            _run_tasks(tasks, workers, items_csv)

        if "agent_bedrock" in systems:
            btasks = _bedrock_tasks(cats, eval_items, args.regimes, done, args.bedrock_limit, cfg)
            _log(f"[eval] agent_bedrock: {len(btasks)} runs (limit {args.bedrock_limit})")
            for i, t in enumerate(btasks, 1):
                rows = run_part_task(t)
                append_rows(items_csv, rows)
                _log(f"[bedrock {i}/{len(btasks)}] {rows[0]['part_id']} {t['regime']}: {rows[0].get('decision') or rows[0].get('error')}")

    rows = read_rows(items_csv)
    if not rows:
        _log("[eval] no rows - nothing to summarize")
        return 1
    pixel = aggregate_pixel(cache_dir, {c: [part_id_from_relpath(d["image"]) for d in pixel_parts[c]] for c in cats})
    meta = {
        "opencv_version": cv2.__version__,
        "policy": {
            "auto_pass_max_prob": settings.policy.auto_pass_max_prob,
            "auto_fail_min_prob": settings.policy.auto_fail_min_prob,
            "max_steps": settings.policy.max_steps,
            "max_captures": settings.policy.max_captures,
            "max_deep_calls": settings.policy.max_deep_calls,
        },
        "single_shot_rule": "PASS iff reliable and p<=auto_pass_max_prob; FAIL iff reliable and p>=auto_fail_min_prob; else NEEDS_REVIEW",
        "conditions_seed": "make_conditions(regime, default_rng(stable_seed(part_id, regime)))",
        "split": "data/splits.json eval split (60 % of VisA 1cls test, stratified, seed 0)",
        "n_boot": args.n_boot,
        "limit_per_category": args.limit or None,
        "eval_parts": {c: len(v) for c, v in eval_items.items()},
    }
    summary = summarize(rows, pixel=pixel, n_boot=args.n_boot, meta=meta)
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=1, allow_nan=False), encoding="utf-8")
    curated = curate_traces(rows, out_dir)
    failures = index_failures(rows, out_dir)
    _log(f"[eval] summary -> {out_dir / 'eval_summary.json'} ({summary['n_ok']} ok rows, {summary['n_errors']} errors); "
         f"{len(curated)} curated traces, {len(failures)} failure overlays; {time.perf_counter() - t_start:.0f}s")
    _print_table(summary)
    return 0


def _run_tasks(tasks: list[dict], workers: int, items_csv: Path) -> None:
    """Run tasks in a spawn pool, appending rows to the CSV as they complete."""
    n = len(tasks)
    t0 = time.perf_counter()
    done = 0
    n_err = 0

    def handle(kind: str, payload: Any) -> None:
        nonlocal done, n_err
        done += 1
        if kind == "part":
            append_rows(items_csv, payload)
            n_err += sum(1 for r in payload if r.get("error"))
            if any(r.get("error") for r in payload) and n_err <= 5:
                _log(f"[eval] error: {next(r['error'] for r in payload if r.get('error'))}")
        elif payload.get("error"):
            _log(f"[pixel] {payload['part_id']}: {payload['error']}")
        if done % 20 == 0 or done == n:
            el = time.perf_counter() - t0
            eta = el / done * (n - done)
            _log(f"[eval] {done}/{n} tasks, {n_err} row errors, {el:.0f}s elapsed, ETA {eta:.0f}s")

    if workers <= 1:
        _init_worker(max(1, (os.cpu_count() or 2) // 2))
        for t in tasks:
            handle(*_dispatch(t))
        return
    ctx = get_context("spawn")
    with ctx.Pool(processes=workers, initializer=_init_worker, initargs=(1,)) as pool:
        for kind, payload in pool.imap_unordered(_dispatch, tasks, chunksize=1):
            handle(kind, payload)


def _bedrock_tasks(cats: list[str], eval_items: dict, regimes: list[str], done: set, limit: int, cfg: dict) -> list[dict]:
    """Label-balanced round-robin subset over categories x regimes, capped at ``limit`` runs."""
    queues: list[list[dict]] = []
    for c in cats:
        for regime in regimes:
            items = sorted(eval_items[c], key=lambda d: stable_seed(d["image"], "bedrock"))
            pos = [d for d in items if int(d["label"]) == 1]
            neg = [d for d in items if int(d["label"]) == 0]
            inter = [x for pair in zip(pos, neg) for x in pair] + pos[len(neg):] + neg[len(pos):]
            queues.append([{"kind": "part", "category": c, "item": d, "regime": regime, "systems": ["agent_bedrock"], "cfg": cfg}
                           for d in inter if (part_id_from_relpath(d["image"]), regime, "agent_bedrock") not in done])
    out: list[dict] = []
    while len(out) < limit and any(queues):
        for q in queues:
            if q and len(out) < limit:
                out.append(q.pop(0))
    return out


def _print_table(summary: dict) -> None:
    res = summary.get("overall") or {}
    _log(f"{'regime':<8} {'system':<14} {'n':>5} {'AUROC':>6} {'escape':>7} {'f-rej':>6} {'review':>7} {'acc_auto':>8} {'cycle_ms':>9}")
    for regime, by_sys in res.items():
        for system, m in by_sys.items():
            def f(v: Any, w: int, p: int = 3) -> str:
                return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"

            _log(f"{regime:<8} {system:<14} {m['n']:>5} {f(m.get('auroc'), 6)} {f(m.get('escape_rate'), 7)} {f(m.get('false_reject_rate'), 6)} "
                 f"{f(m.get('review_rate'), 7)} {f(m.get('accuracy_auto'), 8)} {f(m.get('mean_cycle_time_ms'), 9, 0)}")


if __name__ == "__main__":
    raise SystemExit(main())
