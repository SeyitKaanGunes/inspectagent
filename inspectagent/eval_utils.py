"""Pure evaluation helpers for InspectAgent (no sklearn / scipy dependency).

Everything here is deterministic and side-effect free so it can be unit-tested with
small hand-checked examples (``tests/test_eval_utils.py``) and reused by
``scripts/prepare_visa.py``, ``scripts/build_models.py``, ``scripts/evaluate.py``,
``scripts/make_samples.py`` and ``scripts/make_figures.py``.

Decision vocabulary
-------------------
A system's per-part decision is one of ``"PASS"``, ``"FAIL"`` or ``"NEEDS_REVIEW"``.
Ground-truth labels are ``0`` (normal) and ``1`` (anomalous / defective).

Metric definitions (also exported as :data:`METRIC_DEFINITIONS`)
---------------------------------------------------------------
* escape rate        = #(defective -> PASS) / #defective
* false-reject rate  = #(normal -> FAIL) / #normal
* review rate        = #NEEDS_REVIEW / #parts ; automation rate = 1 - review rate
* accuracy_auto      = accuracy over automatically decided parts only (PASS/FAIL)
* accuracy_hitl      = accuracy when every NEEDS_REVIEW part is resolved correctly by a person
* accuracy / precision / recall / F1 ("operating point", line view): a part counts as
  *flagged* when it is not automatically passed (FAIL or NEEDS_REVIEW), i.e. it is held
  back from shipping. recall = 1 - escape rate.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

import numpy as np

PASS = "PASS"
FAIL = "FAIL"
REVIEW = "NEEDS_REVIEW"
DECISIONS = (PASS, FAIL, REVIEW)

METRIC_DEFINITIONS: dict[str, str] = {
    "auroc": "image-level ROC AUC of the score (agents: final fused P(anomalous)); Mann-Whitney rank statistic, ties averaged",
    "escape_rate": "#(defective -> PASS) / #defective",
    "false_reject_rate": "#(normal -> FAIL) / #normal",
    "review_rate": "#NEEDS_REVIEW / #parts",
    "automation_rate": "1 - review_rate",
    "accuracy_auto": "accuracy over automatically decided parts (PASS/FAIL only)",
    "accuracy_hitl": "accuracy assuming every NEEDS_REVIEW part is resolved correctly by the human reviewer",
    "accuracy": "line view: flagged = FAIL or NEEDS_REVIEW (held back); accuracy of flagged-vs-defective",
    "precision": "line view: #defective among flagged / #flagged",
    "recall": "line view: #flagged defective / #defective (= 1 - escape_rate)",
    "f1": "line view: harmonic mean of precision and recall",
    "bin05_*": "binary metrics of prob >= 0.5 (forced decision, no review band)",
    "ci95": "percentile bootstrap 95% confidence interval (resampling parts with replacement)",
}


# --------------------------------------------------------------------------- seeds & ids


def stable_seed(*parts: Any) -> int:
    """Platform-independent 31-bit seed derived from arbitrary parts (blake2b of their str)."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF


def part_id_from_relpath(rel_path: str) -> str:
    """Stable part id from a VisA-relative image path.

    ``"pcb1/Data/Images/Anomaly/012.JPG"`` -> ``"pcb1_anomaly_012"``. Normal and anomaly
    folders reuse file names in VisA, so the folder kind is part of the id.
    """
    p = PurePosixPath(str(rel_path).replace("\\", "/"))
    parts = [x for x in p.parts if x not in ("", ".", "/")]
    if not parts:
        raise ValueError(f"empty path: {rel_path!r}")
    category = parts[0].lower()
    lowered = [x.lower() for x in parts[1:-1]]
    kind = "anomaly" if "anomaly" in lowered else ("normal" if "normal" in lowered else "part")
    stem = p.stem.lower().replace(" ", "_")
    return f"{category}_{kind}_{stem}"


def zoom_bucket(zoom: float) -> str:
    """Zoom bucket used for calibration (SPEC 2.6): z1 (<1.5), z2 (<2.5), z3 (<3.5), z4."""
    z = float(zoom)
    if z < 1.5:
        return "z1"
    if z < 2.5:
        return "z2"
    if z < 3.5:
        return "z3"
    return "z4"


# --------------------------------------------------------------------------- basic helpers


def safe_div(num: float, den: float) -> float:
    """num / den, NaN when den == 0."""
    return float(num) / float(den) if den else float("nan")


def nan_to_none(obj: Any) -> Any:
    """Recursively replace NaN/inf floats by None (strict-JSON friendly) and numpy scalars by Python types."""
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [nan_to_none(v) for v in obj.tolist()]
    if isinstance(obj, Mapping):
        return {str(k): nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [nan_to_none(v) for v in obj]
    return obj


def normalize_decision(d: Any) -> str:
    """Map assorted decision encodings to PASS / FAIL / NEEDS_REVIEW.

    Accepts the canonical strings (case-insensitive), ``"REVIEW"``, ints/bools
    (1/True = FAIL, 0/False = PASS) and ``None``/empty (= NEEDS_REVIEW).
    """
    if d is None:
        return REVIEW
    if isinstance(d, (bool, np.bool_)):
        return FAIL if d else PASS
    if isinstance(d, (int, np.integer)):
        if int(d) == 1:
            return FAIL
        if int(d) == 0:
            return PASS
        raise ValueError(f"unknown decision {d!r}")
    s = str(d).strip().upper()
    if s in ("", "NONE", "NEEDS_REVIEW", "REVIEW", "HOLD", "HOLD_FOR_REVIEW"):
        return REVIEW
    if s in ("PASS", "OK", "ROUTE_PASS"):
        return PASS
    if s in ("FAIL", "NG", "ROUTE_QUARANTINE", "ROUTE_SCRAP"):
        return FAIL
    raise ValueError(f"unknown decision {d!r}")


def decide_three_way(prob: float | None, reliable: bool, pass_max: float, fail_min: float) -> str:
    """Human-control decision rule applied to single-shot baselines (same thresholds as the agent policy).

    PASS iff reliable and prob <= pass_max; FAIL iff reliable and prob >= fail_min; else NEEDS_REVIEW.
    """
    if prob is None or not math.isfinite(float(prob)) or not reliable:
        return REVIEW
    p = float(prob)
    if p <= pass_max:
        return PASS
    if p >= fail_min:
        return FAIL
    return REVIEW


# --------------------------------------------------------------------------- ranking / AUROC


def rankdata(x: Sequence[float] | np.ndarray) -> np.ndarray:
    """1-based ranks with ties replaced by their average rank (like scipy's 'average')."""
    a = np.asarray(x, dtype=np.float64).ravel()
    n = a.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    boundaries = np.flatnonzero(np.r_[True, sa[1:] != sa[:-1], True])
    counts = np.diff(boundaries)
    avg = boundaries[:-1] + (counts + 1) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.repeat(avg, counts)
    return ranks


def auroc(scores: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (ties count 1/2). NaN when a class is missing.

    Non-finite scores are dropped together with their labels.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(np.int64)
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and labels {y.shape} differ in shape")
    keep = np.isfinite(s)
    s, y = s[keep], y[keep]
    pos = y == 1
    n1 = int(pos.sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    u = r[pos].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def sample_pixels(
    amap: np.ndarray, mask: np.ndarray, max_pos: int, max_neg: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly subsample anomaly-map values inside (label 1) and outside (label 0) a GT mask."""
    a = np.asarray(amap, dtype=np.float32)
    m = np.asarray(mask)
    if a.shape[:2] != m.shape[:2]:
        raise ValueError(f"map {a.shape} and mask {m.shape} differ in shape")
    a = a.reshape(-1)
    m = (m.reshape(-1) > 0)
    pos = np.flatnonzero(m)
    neg = np.flatnonzero(~m)
    if pos.size > max_pos:
        pos = rng.choice(pos, size=max_pos, replace=False)
    if neg.size > max_neg:
        neg = rng.choice(neg, size=max_neg, replace=False)
    idx = np.concatenate([np.sort(pos), np.sort(neg)])
    lab = np.concatenate([np.ones(pos.size, np.uint8), np.zeros(neg.size, np.uint8)])
    return a[idx], lab


def pixel_auroc(
    maps: Iterable[np.ndarray],
    masks: Iterable[np.ndarray],
    max_pos: int = 20000,
    max_neg: int = 20000,
    seed: int = 0,
) -> dict:
    """Localization AUROC over images: pooled (per-image stratified pixel samples) and mean per image."""
    rng = np.random.default_rng(seed)
    all_s: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    per_image: list[float] = []
    for amap, mask in zip(maps, masks, strict=True):
        s, y = sample_pixels(amap, mask, max_pos, max_neg, rng)
        all_s.append(s)
        all_y.append(y)
        per_image.append(auroc(s, y))
    if not all_s:
        return {"pooled": float("nan"), "mean_per_image": float("nan"), "n_images": 0}
    pooled = auroc(np.concatenate(all_s), np.concatenate(all_y))
    finite = [v for v in per_image if math.isfinite(v)]
    return {
        "pooled": pooled,
        "mean_per_image": float(np.mean(finite)) if finite else float("nan"),
        "n_images": len(per_image),
    }


# --------------------------------------------------------------------------- decision metrics


def binary_metrics(labels: Sequence[int] | np.ndarray, preds: Sequence[int] | np.ndarray) -> dict:
    """Confusion counts and accuracy/precision/recall/F1/specificity for binary predictions (1 = defective)."""
    y = np.asarray(labels).ravel().astype(np.int64)
    p = np.asarray(preds).ravel().astype(np.int64)
    if y.shape != p.shape:
        raise ValueError("labels and preds differ in shape")
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if math.isfinite(precision) and math.isfinite(recall) else float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": safe_div(tp + tn, y.size),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": safe_div(tn, tn + fp),
    }


def confusion_counts(labels: Sequence[int] | np.ndarray, decisions: Sequence[Any]) -> dict[str, int]:
    """3-way confusion: tp (1->FAIL), fn (1->PASS), rp (1->REVIEW), tn (0->PASS), fp (0->FAIL), rn (0->REVIEW)."""
    y = np.asarray(labels).ravel().astype(np.int64)
    d = [normalize_decision(v) for v in decisions]
    if len(d) != y.size:
        raise ValueError("labels and decisions differ in length")
    out = {"tp": 0, "fn": 0, "rp": 0, "tn": 0, "fp": 0, "rn": 0}
    for lab, dec in zip(y.tolist(), d):
        if lab == 1:
            out["tp" if dec == FAIL else "fn" if dec == PASS else "rp"] += 1
        elif lab == 0:
            out["fp" if dec == FAIL else "tn" if dec == PASS else "rn"] += 1
        else:
            raise ValueError(f"label must be 0 or 1, got {lab}")
    return out


def decision_metrics(labels: Sequence[int] | np.ndarray, decisions: Sequence[Any]) -> dict:
    """Escape / false-reject / review / automation rates and accuracy variants (see module docstring)."""
    c = confusion_counts(labels, decisions)
    tp, fn, rp, tn, fp, rn = c["tp"], c["fn"], c["rp"], c["tn"], c["fp"], c["rn"]
    n_pos = tp + fn + rp
    n_neg = tn + fp + rn
    n = n_pos + n_neg
    n_rev = rp + rn
    n_auto = n - n_rev
    flagged_tp, flagged_fp = tp + rp, fp + rn
    precision = safe_div(flagged_tp, flagged_tp + flagged_fp)
    recall = safe_div(flagged_tp, n_pos)
    if math.isfinite(precision) and math.isfinite(recall) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")
    return {
        **c,
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_review": n_rev,
        "n_auto": n_auto,
        "escape_rate": safe_div(fn, n_pos),
        "false_reject_rate": safe_div(fp, n_neg),
        "review_rate": safe_div(n_rev, n),
        "automation_rate": safe_div(n_auto, n),
        "accuracy_auto": safe_div(tp + tn, n_auto),
        "accuracy_hitl": safe_div(tp + tn + n_rev, n),
        "accuracy": safe_div(flagged_tp + tn, n),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_auto": safe_div(tp, tp + fp),
        "recall_auto": safe_div(tp, n_pos),
    }


# --------------------------------------------------------------------------- risk-coverage


def risk_coverage(probs: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray, threshold: float = 0.5) -> dict:
    """Automation-vs-accuracy (risk-coverage) curve of a probabilistic classifier.

    Parts are auto-decided most-confident first (confidence = |p - threshold|, decision
    p >= threshold); a point is emitted at the end of each group of equal confidence.
    Returns ``coverage`` (fraction auto-decided), ``risk`` (error rate among them),
    ``accuracy`` (= 1 - risk), ``threshold_conf`` and ``aurc`` (area under the
    risk-coverage curve, step integration; lower is better).
    """
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(np.int64)
    keep = np.isfinite(p)
    p, y = p[keep], y[keep]
    n = p.size
    if n == 0:
        return {"coverage": [], "risk": [], "accuracy": [], "threshold_conf": [], "aurc": float("nan"), "n": 0}
    conf = np.round(np.abs(p - threshold), 12)  # float-equal confidences form one tie group
    pred = (p >= threshold).astype(np.int64)
    err = (pred != y).astype(np.float64)
    order = np.argsort(-conf, kind="mergesort")
    conf_s = conf[order]
    cum_err = np.cumsum(err[order])
    ends = np.flatnonzero(np.r_[conf_s[1:] != conf_s[:-1], True])  # last index of each tie group
    k = ends + 1
    coverage = k / n
    risk = cum_err[ends] / k
    prev_cov = np.r_[0.0, coverage[:-1]]
    aurc = float(np.sum(risk * (coverage - prev_cov)))
    return {
        "coverage": coverage.tolist(),
        "risk": risk.tolist(),
        "accuracy": (1.0 - risk).tolist(),
        "threshold_conf": conf_s[ends].tolist(),
        "aurc": aurc,
        "n": int(n),
    }


def resample_curve(x: Sequence[float], y: Sequence[float], grid: Sequence[float]) -> list[float]:
    """Linear interpolation of a monotone-x curve onto ``grid`` (NaN outside the x range)."""
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    g = np.asarray(grid, dtype=np.float64)
    if xs.size == 0:
        return [float("nan")] * g.size
    out = np.interp(g, xs, ys, left=np.nan, right=np.nan)
    return out.tolist()


# --------------------------------------------------------------------------- splits


def split_fraction(items: Sequence[Any], frac_first: float, seed: int) -> tuple[list[Any], list[Any]]:
    """Random split into (first, second) with round(frac_first * n) items in ``first``.

    Both outputs keep the input order (deterministic for a given seed and input order).
    """
    if not 0.0 <= frac_first <= 1.0:
        raise ValueError("frac_first must be in [0, 1]")
    n = len(items)
    k = int(math.floor(frac_first * n + 0.5))
    rng = np.random.default_rng(seed)
    chosen = set(rng.permutation(n)[:k].tolist())
    first = [items[i] for i in range(n) if i in chosen]
    second = [items[i] for i in range(n) if i not in chosen]
    return first, second


def stratified_split(
    items: Sequence[Any], labels: Sequence[Any], frac_first: float, seed: int
) -> tuple[list[Any], list[Any]]:
    """Per-label :func:`split_fraction` (classes visited in sorted order, one RNG stream).

    Each class contributes round(frac_first * n_class) items to ``first``; outputs keep the
    input order.
    """
    if len(items) != len(labels):
        raise ValueError("items and labels differ in length")
    if not 0.0 <= frac_first <= 1.0:
        raise ValueError("frac_first must be in [0, 1]")
    rng = np.random.default_rng(seed)
    chosen: set[int] = set()
    lab_list = list(labels)
    for cls in sorted(set(lab_list), key=lambda v: (str(type(v)), v)):
        idx = [i for i, v in enumerate(lab_list) if v == cls]
        k = int(math.floor(frac_first * len(idx) + 0.5))
        perm = rng.permutation(len(idx))[:k]
        chosen.update(idx[j] for j in perm.tolist())
    first = [items[i] for i in range(len(items)) if i in chosen]
    second = [items[i] for i in range(len(items)) if i not in chosen]
    return first, second


def stratified_subsample(items: Sequence[Any], labels: Sequence[Any], n: int, seed: int) -> list[Any]:
    """Deterministic label-proportional subsample of at most ``n`` items (input order kept)."""
    if n >= len(items):
        return list(items)
    if n <= 0:
        return []
    first, _ = stratified_split(items, labels, n / len(items), seed)
    if len(first) > n:  # rounding can overshoot by one per class
        first = split_fraction(first, n / len(first), seed + 1)[0]
    return first


# --------------------------------------------------------------------------- bootstrap


def bootstrap_ci(
    stat_fn: Callable[..., float],
    arrays: Sequence[Sequence[Any] | np.ndarray],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    min_valid: int = 20,
) -> tuple[float, float]:
    """Percentile bootstrap (1 - alpha) CI of ``stat_fn(*arrays)`` resampling rows with replacement.

    Resamples where the statistic is not finite (e.g. AUROC with one class) are skipped;
    returns (NaN, NaN) when fewer than ``min_valid`` resamples are finite.
    """
    arrs = [np.asarray(a) for a in arrays]
    if not arrs:
        raise ValueError("need at least one array")
    n = len(arrs[0])
    if any(len(a) != n for a in arrs):
        raise ValueError("arrays differ in length")
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    m = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = float(stat_fn(*[a[idx] for a in arrs]))
        if math.isfinite(v):
            vals[m] = v
            m += 1
    if m < min_valid:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals[:m], [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def rate_ci(successes: Sequence[bool] | np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """(rate, lo, hi) of a boolean vector with a bootstrap 95% CI."""
    v = np.asarray(successes, dtype=np.float64).ravel()
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = bootstrap_ci(lambda a: float(np.mean(a)), [v], n_boot=n_boot, seed=seed, min_valid=1)
    return float(v.mean()), lo, hi


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (closed form; NaN when n == 0)."""
    if n <= 0:
        return float("nan"), float("nan")
    phat = k / n
    den = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / den
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


# --------------------------------------------------------------------------- aggregation


def describe(values: Iterable[float]) -> dict:
    """mean / std / median / p10 / p90 / min / max of the finite values (NaN-filled when empty)."""
    v = np.asarray([float(x) for x in values if x is not None], dtype=np.float64)
    v = v[np.isfinite(v)]
    keys = ("mean", "std", "median", "p10", "p90", "min", "max")
    if v.size == 0:
        return {"n": 0, **{k: float("nan") for k in keys}}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=0)),
        "median": float(np.median(v)),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def tool_histogram(tool_counts: Iterable[Mapping[str, int] | None]) -> dict:
    """Aggregate per-run tool-call counts: ``{"total": {...}, "mean_per_run": {...}, "runs_using": {...}, "n_runs"}``."""
    total: dict[str, int] = {}
    using: dict[str, int] = {}
    n = 0
    for tc in tool_counts:
        n += 1
        for name, cnt in (tc or {}).items():
            c = int(cnt)
            total[name] = total.get(name, 0) + c
            if c > 0:
                using[name] = using.get(name, 0) + 1
    names = sorted(total)
    return {
        "n_runs": n,
        "total": {k: total[k] for k in names},
        "mean_per_run": {k: safe_div(total[k], n) for k in names},
        "runs_using": {k: safe_div(using.get(k, 0), n) for k in names},
    }


def evidence_action_flags(steps: Sequence[Mapping[str, Any]]) -> dict:
    """Did visual evidence change what the agent did? (computed from serialized trace steps).

    Same per-step definition as the loop's ``Trace.totals["evidence_driven_steps"]``
    (``inspectagent.agent.trace.classify_evidence_steps``):

    * ``quality_recapture``: a ``capture_image`` issued after a quality assessment that
      reported issues (or whose decision cites a quality report ``q_*``).
    * ``zoom`` / ``zoom_confirmed`` / ``zoom_refuted`` / ``zoom_inconclusive``: an ``inspect_region`` call
      and its outcome (from the observation's ``status``; an inconclusive zoom is never a refutation).
    * ``deep``: a ``detect_anomalies`` call with ``detector == "deep"`` (escalation).
    * ``review``: any ``request_human_review`` call; ``review_visual``: one triggered by the visual
      evidence (planner-initiated, or a policy refusal of an unsupported verdict); ``review_nonvisual``:
      one forced by the step/time/capture budget, tool errors, a planner failure or an invalid action.
    * ``any``: quality_recapture, zoom, deep or review_visual; ``policy_overrides``: steps replaced by the policy.
    """
    from inspectagent.agent.trace import classify_evidence_steps

    flags = {
        "quality_recapture": False,
        "zoom": False,
        "zoom_confirmed": False,
        "zoom_refuted": False,
        "zoom_inconclusive": False,
        "deep": False,
        "review": False,
        "review_visual": False,
        "review_nonvisual": False,
        "policy_overrides": sum(1 for s in steps if s.get("overridden_by_policy")),
    }
    for rec in classify_evidence_steps(steps):
        kind = rec["kind"]
        if kind in ("quality_recapture", "zoom", "deep"):
            flags[kind] = True
        if rec["zoom_status"] is not None:
            flags[f"zoom_{rec['zoom_status']}"] = True
        if kind in ("review_visual", "review_nonvisual"):
            flags["review"] = True
            flags[kind] = True
    flags["any"] = bool(flags["quality_recapture"] or flags["zoom"] or flags["deep"] or flags["review_visual"])
    return flags
