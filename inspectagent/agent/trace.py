"""Trace (de)serialization and structured JSON step logging.

Every agent step is emitted as ONE JSON line (``{"evt": "step", "trace_id": ..., ...}``) so the
logs are CloudWatch-Insights friendly when the service runs on AWS Lambda, and greppable locally.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from inspectagent.types import Decision, Region, Step, Trace, Verdict

LOGGER_NAME = "inspectagent.agent"

# Observation keys copied into the compact per-step log line (scalars only, plus these lists).
_LOG_LIST_KEYS = ("issues", "artifacts", "region_ids", "fused_evidence")


def jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` into plain JSON types.

    numpy scalars/arrays become Python numbers/lists, tuples become lists, ``Path`` becomes ``str``,
    dataclasses/objects with ``to_dict`` are expanded, and non-finite floats become ``None`` (so
    the output is strict JSON that browsers can parse).
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (bytes, bytearray)):
        return {"bytes": len(obj)}
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return jsonable(obj.to_dict())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return jsonable(dataclasses.asdict(obj))
    return str(obj)


def trace_to_dict(trace: Trace | dict) -> dict:
    """``Trace`` (or an already-serialized dict) -> plain JSON-safe dict."""
    d = trace.to_dict() if isinstance(trace, Trace) else trace
    return jsonable(d)


def trace_to_json(trace: Trace | dict, indent: int | None = None) -> str:
    """Serialize a trace to strict JSON (UTF-8 text, NaN-free)."""
    return json.dumps(trace_to_dict(trace), ensure_ascii=False, indent=indent, allow_nan=False)


def save_trace(trace: Trace | dict, path: str | Path) -> Path:
    """Write a trace as pretty JSON (creates parent directories). Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(trace_to_json(trace, indent=2), encoding="utf-8")
    return p


def load_trace(path: str | Path) -> dict:
    """Read a trace JSON file back as a dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _region(d: dict) -> Region:
    return Region.from_dict(d)


def trace_from_dict(d: dict) -> Trace:
    """Rebuild a ``Trace`` object (steps, decisions, verdict, regions) from its dict form."""
    steps: list[Step] = []
    for s in d.get("steps", []):
        dd = s["decision"]
        dec = Decision(
            step=int(dd["step"]),
            tool=dd["tool"],
            args=dict(dd.get("args", {})),
            rationale=dd.get("rationale", ""),
            evidence=list(dd.get("evidence", [])),
            planner=dd.get("planner", "rule"),
        )
        steps.append(
            Step(
                step=int(s["step"]),
                decision=dec,
                observation=dict(s.get("observation", {})),
                artifacts=list(s.get("artifacts", [])),
                latency_ms=float(s.get("latency_ms", 0.0)),
                sim_time_ms=float(s.get("sim_time_ms", 0.0)),
                overridden_by_policy=bool(s.get("overridden_by_policy", False)),
            )
        )
    verdict = None
    if d.get("verdict"):
        v = d["verdict"]
        verdict = Verdict(
            label=v["label"],
            confidence=float(v["confidence"]),
            rationale=v.get("rationale", ""),
            defect_regions=[_region(r) for r in v.get("defect_regions", [])],
            action=v.get("action", "hold_for_review"),
            review_id=v.get("review_id"),
            prob_anomalous=v.get("prob_anomalous"),
        )
    return Trace(
        trace_id=d["trace_id"],
        part_id=d["part_id"],
        category=d["category"],
        planner=d.get("planner", ""),
        started_at=d.get("started_at", ""),
        steps=steps,
        verdict=verdict,
        budget=dict(d.get("budget", {})),
        totals=dict(d.get("totals", {})),
        ground_truth=d.get("ground_truth"),
        conditions=d.get("conditions"),
        cv_info=dict(d.get("cv_info", {})),
        errors=list(d.get("errors", [])),
    )


# --------------------------------------------------------------------------------------------
# "visual evidence changed the action" - one definition shared by loop totals and evaluation
# --------------------------------------------------------------------------------------------
REVIEW_TRIGGERS = ("visual", "budget", "errors", "planner_error", "invalid")
_BUDGET_REASONS = ("step budget", "sim-time budget", "capture budget", "deep-call budget", "wall-clock budget")


def review_trigger(step: Mapping[str, Any]) -> str:
    """Why a ``request_human_review`` step happened: one of :data:`REVIEW_TRIGGERS`.

    Uses the structured ``observation["review_trigger"]`` written by the loop; for older traces it falls
    back to the decision: planner-initiated reviews are ``visual``, policy reviews are classified by the
    reason after the ``"policy: "`` prefix of the rationale.
    """
    obs = step.get("observation") or {}
    trig = obs.get("review_trigger") if isinstance(obs, Mapping) else None
    if trig in REVIEW_TRIGGERS:
        return str(trig)
    dec = step.get("decision") or {}
    if dec.get("planner") != "policy":
        return "visual"
    reason = str(dec.get("rationale") or "")
    reason = reason.split("policy:", 1)[1].strip() if "policy:" in reason else reason
    low = reason.lower()
    if low.startswith("planner error"):
        return "planner_error"
    if re.match(r"\d+ tool errors", low):
        return "errors"
    if low.startswith(("unknown tool", "finish with invalid label")):
        return "invalid"
    if low.startswith(_BUDGET_REASONS):
        return "budget"
    return "visual"


def classify_evidence_steps(steps: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Per serialized step: ``{"step", "kind", "zoom_status", "review_trigger"}`` where ``kind`` is the way
    visual evidence changed the action at that step, or None.

    * ``quality_recapture``: a ``capture_image`` after a quality report with issues, or citing a ``q_*`` id;
    * ``zoom``: an ``inspect_region`` call (``zoom_status`` = confirmed / refuted / inconclusive from the
      observation's ``status``; never inferred as refuted from ``confirmed: false``);
    * ``deep``: ``detect_anomalies`` with ``detector == "deep"`` (escalation);
    * ``review_visual`` / ``review_nonvisual``: a human review triggered by the visual evidence vs. by the
      budget, tool errors, a planner failure or an invalid action (see :func:`review_trigger`).
    """
    out: list[dict] = []
    seen_capture = False
    quality_issue_pending = False
    for i, s in enumerate(steps):
        dec = s.get("decision") or {}
        tool = dec.get("tool", "")
        args = dec.get("args") or {}
        obs = s.get("observation") or {}
        evidence = [str(e) for e in (dec.get("evidence") or [])]
        rec: dict[str, Any] = {"step": s.get("step", i + 1), "kind": None, "zoom_status": None, "review_trigger": None}
        if tool == "assess_quality":
            issues = obs.get("issues") or []
            if obs.get("ok") is False or len(issues) > 0:
                quality_issue_pending = True
        elif tool == "capture_image":
            if seen_capture and (quality_issue_pending or any(e.startswith("q_") for e in evidence)):
                rec["kind"] = "quality_recapture"
            seen_capture = True
            quality_issue_pending = False
        elif tool == "inspect_region":
            rec["kind"] = "zoom"
            status = obs.get("status")
            if status in ("confirmed", "refuted", "inconclusive"):
                rec["zoom_status"] = status
            elif obs.get("confirmed") is True:
                rec["zoom_status"] = "confirmed"
        elif tool == "detect_anomalies" and str(args.get("detector", "fast")) == "deep":
            rec["kind"] = "deep"
        elif tool == "request_human_review":
            trig = review_trigger(s)
            rec["review_trigger"] = trig
            rec["kind"] = "review_visual" if trig == "visual" else "review_nonvisual"
        out.append(rec)
    return out


EVIDENCE_KINDS = ("quality_recapture", "zoom", "deep", "review_visual")


def _compact_observation(obs: dict) -> dict:
    out: dict[str, Any] = {}
    for k, v in obs.items():
        if v is None or isinstance(v, (str, bool, int, float)):
            out[k] = v
        elif k in _LOG_LIST_KEYS and isinstance(v, list):
            out[k] = v
        elif k == "regions" and isinstance(v, list):
            out["n_regions"] = len(v)
    return out


def step_record(step: Step, trace_id: str) -> dict:
    """The structured record logged for one step (JSON-safe dict)."""
    d = step.decision
    return jsonable(
        {
            "evt": "step",
            "trace_id": trace_id,
            "step": step.step,
            "tool": d.tool,
            "args": d.args,
            "planner": d.planner,
            "overridden": step.overridden_by_policy,
            "evidence": d.evidence,
            "rationale": d.rationale,
            "latency_ms": round(float(step.latency_ms), 2),
            "sim_time_ms": round(float(step.sim_time_ms), 2),
            "artifacts": step.artifacts,
            "obs": _compact_observation(step.observation),
        }
    )


def log_step(logger: logging.Logger | None, step: Step, trace_id: str) -> dict:
    """Emit one JSON line for ``step`` at INFO level and return the record."""
    rec = step_record(step, trace_id)
    (logger or logging.getLogger(LOGGER_NAME)).info(json.dumps(rec, ensure_ascii=True, separators=(",", ":")))
    return rec


def log_event(logger: logging.Logger | None, evt: str, trace_id: str, **fields: Any) -> dict:
    """Emit one JSON line for a non-step event (``start``, ``verdict``, ``error``...)."""
    rec = jsonable({"evt": evt, "trace_id": trace_id, **fields})
    (logger or logging.getLogger(LOGGER_NAME)).info(json.dumps(rec, ensure_ascii=True, separators=(",", ":")))
    return rec


class JsonLineFormatter(logging.Formatter):
    """Pass JSON messages through unchanged; wrap anything else as ``{"evt": "log", ...}``."""

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if msg.startswith("{") and msg.endswith("}"):
            return msg
        rec = {"evt": "log", "level": record.levelname, "logger": record.name, "msg": msg}
        if record.exc_info:
            rec["exc"] = self.formatException(record.exc_info)
        return json.dumps(rec, ensure_ascii=True, separators=(",", ":"))


def configure_logging(level: str | int = "INFO", stream: TextIO | None = None) -> logging.Logger:
    """Attach a single JSON-line handler to the agent logger (idempotent)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level if isinstance(level, int) else str(level).upper())
    for h in list(logger.handlers):
        if getattr(h, "_inspectagent_json", False):
            logger.removeHandler(h)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonLineFormatter())
    handler._inspectagent_json = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.propagate = False
    return logger
