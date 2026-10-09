"""Agent tools: JSON-Schema specs + implementations over an :class:`InspectionSession`.

``ToolRegistry.specs()`` is the single source of tool definitions for the rule planner, the
Amazon Bedrock Converse planner (``toolSpec.inputSchema.json``) and the MCP server. Schemas use a
conservative JSON Schema (draft 2020-12) subset: ``type: object``, ``properties``, ``required``,
``additionalProperties: false``, numeric ``minimum``/``maximum``, ``enum``, ``maxLength``,
``maxItems`` and ``default`` annotations.

``ToolRegistry.execute`` validates arguments with ``jsonschema``; violations of numeric ranges,
``maxLength`` and ``maxItems`` are *clamped* (and reported in ``arg_notes``), every other violation
is an error. Tool exceptions never escape: they become ``{"error": ...}`` observations and are
counted in ``session.counters["errors"]`` (the policy forces a human review after two).
Observations are compact JSON (no arrays, no images; images are JPEG artifacts referenced by id).
"""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import cv2
import numpy as np
from jsonschema import Draft202012Validator

from inspectagent.config import PX_PER_MM, Settings, get_settings
from inspectagent.agent.session import (
    ZOOM_CONFIRM_PROB,
    ZOOM_REFUTE_PROB,
    InspectionSession,
    Inspection,
    LOCALIZATION_PEAK_FRAC,
    SessionError,
    classify_inspection,
)
from inspectagent.agent.trace import jsonable
from inspectagent.types import CaptureSettings, View
from inspectagent.vision.quality import blocking_quality_issues

log = logging.getLogger(__name__)

LIGHTS = ("ring", "diffuse", "low_angle")
TOOL_NAMES = (
    "capture_image",
    "assess_quality",
    "detect_anomalies",
    "inspect_region",
    "measure_defect",
    "request_human_review",
    "finish",
)
TERMINAL_TOOLS = frozenset({"request_human_review", "finish"})
CAPTURE_TOOLS = frozenset({"capture_image", "inspect_region"})
DEFAULT_INSPECT_ZOOM = 3.0
_CLAMPABLE = frozenset({"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "maxLength", "maxItems"})


class ToolArgumentError(ValueError):
    """Arguments do not satisfy the tool's input schema (after clamping)."""


@dataclass
class ToolSpec:
    """One tool: name, description, JSON-Schema input, implementation, terminal flag."""

    name: str
    description: str
    input_schema: dict
    fn: Callable[[InspectionSession, dict], dict]
    terminal: bool = False

    def to_spec(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": copy.deepcopy(self.input_schema)}


# ---------------------------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------------------------
def _num(lo: float, hi: float, desc: str, default: float | None = None) -> dict:
    d: dict[str, Any] = {"type": "number", "minimum": lo, "maximum": hi, "description": desc}
    if default is not None:
        d["default"] = default
    return d


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def build_schemas(settings: Settings | None = None) -> dict[str, dict]:
    """Input schemas of all tools (ranges taken from ``config.CameraConfig``)."""
    cc = (settings or get_settings()).camera
    zmax = float(cc.max_zoom)
    light = {"type": "string", "enum": list(LIGHTS), "description": "ring = bright, may cause specular glare; diffuse = 0.6x light, no glare; low_angle = 0.5x light, relief shading."}
    return {
        "capture_image": _obj(
            {
                "cx": _num(0.0, 1.0, "View centre x, normalized scene coordinates.", 0.5),
                "cy": _num(0.0, 1.0, "View centre y, normalized scene coordinates.", 0.5),
                "zoom": _num(1.0, zmax, "Optical zoom; 1 = whole part.", 1.0),
                "exposure_ev": _num(float(cc.min_ev), float(cc.max_ev), "Exposure compensation in EV stops.", 0.0),
                "light": {**light, "default": "ring"},
                "autofocus": {"type": "boolean", "description": "Run autofocus before the capture (+time).", "default": False},
            },
            [],
        ),
        "assess_quality": _obj(
            {"capture_id": {"type": "string", "minLength": 1, "maxLength": 64, "description": "Capture to assess, e.g. cap_1."}},
            ["capture_id"],
        ),
        "detect_anomalies": _obj(
            {
                "capture_id": {"type": "string", "minLength": 1, "maxLength": 64, "description": "Capture to analyse."},
                "detector": {
                    "type": "string",
                    "enum": ["fast", "deep"],
                    "default": "fast",
                    "description": "fast = registration-based statistical template (needs alignment); deep = alignment-free PatchCore on cv2.dnn (budgeted).",
                },
            },
            ["capture_id"],
        ),
        "inspect_region": _obj(
            {
                "region_id": {"type": "string", "minLength": 1, "maxLength": 64, "description": "Candidate region to zoom into, e.g. reg_1_1."},
                "zoom": _num(2.0, zmax, "Zoom factor of the confirmation view.", DEFAULT_INSPECT_ZOOM),
                "exposure_ev": _num(float(cc.min_ev), float(cc.max_ev), "Optional exposure override (default: last good settings)."),
                "light": {**light, "description": "Optional light override (default: last good settings)."},
                "autofocus": {"type": "boolean", "description": "Optional autofocus override (default: last good settings)."},
            },
            ["region_id"],
        ),
        "measure_defect": _obj(
            {"region_id": {"type": "string", "minLength": 1, "maxLength": 64, "description": "Region to measure (mm)."}},
            ["region_id"],
        ),
        "request_human_review": _obj(
            {
                "reason": {"type": "string", "minLength": 1, "maxLength": 2000, "description": "Evidence-based reason for escalation."},
                "proposed_label": {"type": "string", "enum": ["PASS", "FAIL"], "description": "What the agent would decide."},
                "confidence": _num(0.0, 1.0, "Confidence in the proposed label.", 0.5),
            },
            ["reason", "proposed_label"],
        ),
        "finish": _obj(
            {
                "label": {"type": "string", "enum": ["PASS", "FAIL"], "description": "Proposed verdict (gated by the human-control policy)."},
                "confidence": _num(0.0, 1.0, "Confidence in the label."),
                "rationale": {"type": "string", "minLength": 1, "maxLength": 4000, "description": "Cite evidence ids and numbers."},
                "region_ids": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 64},
                    "maxItems": 10,
                    "default": [],
                    "description": "Defect regions supporting a FAIL.",
                },
            },
            ["label", "confidence", "rationale"],
        ),
    }


DESCRIPTIONS = {
    "capture_image": (
        "Move the camera and take a picture. Returns capture_id, view window, simulated time and mean luma. "
        "Start with the default full view; recapture with corrected light/exposure/autofocus when assess_quality reports issues."
    ),
    "assess_quality": (
        "Measure image quality of a capture with OpenCV (Laplacian sharpness vs reference, exposure, clipping, noise, specular glare). "
        "Returns issues and a suggested exposure delta. Fix issues before trusting detectors."
    ),
    "detect_anomalies": (
        "Run an anomaly detector on a capture. Returns calibrated P(anomalous), reliability (registration), candidate regions with ids "
        "and the updated fused probability. Use deep when the fast detector is unreliable or the evidence stays ambiguous."
    ),
    "inspect_region": (
        "Zoom into a candidate region (default x3, last good settings), re-check quality and run the fast detector on the zoom view. "
        "Returns status confirmed/refuted/inconclusive, region_peak, zoom prob and the updated fused probability."
    ),
    "measure_defect": "Measure a defect region in millimetres (area, length, width, elongation, orientation, contrast, type guess).",
    "request_human_review": "Terminal. Hand the part to a human with the evidence and a proposed label (used when evidence is insufficient or ambiguous).",
    "finish": (
        "Terminal. Propose the final verdict. The human-control policy only accepts PASS when fused P(anomalous) <= auto_pass_max_prob "
        "and FAIL when >= auto_fail_min_prob, with reliable evidence; otherwise the part goes to human review."
    ),
}


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------
def _mean_luma(img: np.ndarray) -> float:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(img.mean())


def _safe_artifact(session: InspectionSession, render: Callable[[], np.ndarray], kind: str, obs: dict) -> str | None:
    """Render + store an artifact; rendering failures are reported, never fatal."""
    try:
        aid = session.add_artifact(render(), kind)
    except Exception as exc:  # artifact rendering must not break the inspection
        log.warning("artifact %s failed: %s", kind, exc)
        obs["artifact_error"] = f"{type(exc).__name__}: {exc}"
        return None
    obs["artifact_id"] = aid
    return aid


def _render_result(session: InspectionSession, result_id: str, highlight=None) -> np.ndarray:
    res = session.results[result_id]
    cap = session.captures[res.capture_id]
    img = cap.image
    if res.anomaly_map is not None:
        img = session.vision.heatmap_overlay(img, res.anomaly_map)
    if res.regions:
        img = session.vision.draw_regions(img, res.regions)
    if highlight is not None:
        img = session.vision.draw_regions(img, [highlight], color=(255, 255, 0), label=False)
    return img


def _fused(session: InspectionSession) -> dict:
    rep = session.fusion_report()
    return {"fused_prob": rep["prob"], "fused_evidence": rep["evidence"], "fusion": rep["explain"]}


def settings_for_call(session: InspectionSession, tool: str, args: dict) -> CaptureSettings | None:
    """Capture settings a capture-producing tool call would use (``None`` for other tools)."""
    if tool == "capture_image":
        return session.clamp_settings(
            CaptureSettings(
                view=View(float(args.get("cx", 0.5)), float(args.get("cy", 0.5)), float(args.get("zoom", 1.0))),
                exposure_ev=float(args.get("exposure_ev", 0.0)),
                light=args.get("light", "ring"),
                autofocus=bool(args.get("autofocus", False)),
            )
        )
    if tool == "inspect_region":
        region = session.require_region(str(args["region_id"]))
        base = session.last_good_settings()
        cx, cy = region.center_norm(session.scene_size)
        return session.clamp_settings(
            CaptureSettings(
                view=View(float(cx), float(cy), float(args.get("zoom", DEFAULT_INSPECT_ZOOM))),
                exposure_ev=float(args["exposure_ev"]) if args.get("exposure_ev") is not None else base.exposure_ev,
                light=args["light"] if args.get("light") is not None else base.light,
                autofocus=bool(args["autofocus"]) if args.get("autofocus") is not None else base.autofocus,
            )
        )
    return None


# ---------------------------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------------------------
def tool_capture_image(session: InspectionSession, a: dict) -> dict:
    cid_before = len(session.captures)
    s = settings_for_call(session, "capture_image", a)
    assert s is not None
    if cid_before > 0 and s.view.zoom <= 1.05:
        session.counters["corrective_captures"] += 1
    cap = session.take_capture(s)
    budget = session.budget_state()
    obs: dict[str, Any] = {
        "capture_id": cap.capture_id,
        "index": cap.index,
        "settings": cap.settings.to_dict(),
        "full_view": session.is_full_view(cap.capture_id),
        "window": list(cap.window),
        "size": list(cap.size),
        "sim_time_ms": cap.sim_time_ms,
        "sim_time_total_ms": session.sim_time_ms,
        "mean_luma": _mean_luma(cap.image),
        "captures_used": budget["captures_used"],
        "captures_left": budget["captures_left"],
    }
    _safe_artifact(session, lambda: cap.image, "capture", obs)
    return obs


def tool_assess_quality(session: InspectionSession, a: dict) -> dict:
    cid = a["capture_id"]
    cap = session.require_capture(cid)
    qid, q = session.run_quality(cid)
    qc = session.settings.quality
    d = q.to_dict()
    d["glare_boxes"] = d.get("glare_boxes", [])[:5]
    obs: dict[str, Any] = {
        "quality_id": qid,
        **d,
        "reference_sharpness": session.reference_sharpness(cap.settings.view.zoom),
        "thresholds": {
            "min_sharpness_ratio": qc.min_sharpness_ratio,
            "min_luma": qc.min_luma,
            "max_luma": qc.max_luma,
            "max_clipped_frac": qc.max_clipped_frac,
            "max_crushed_frac": qc.max_crushed_frac,
            "max_noise_sigma": qc.max_noise_sigma,
            "max_glare_frac": qc.max_glare_frac,
            "target_luma": qc.target_luma,
        },
    }
    _safe_artifact(session, lambda: session.vision.draw_quality(cap.image, q), "quality", obs)
    return obs


def tool_detect_anomalies(session: InspectionSession, a: dict) -> dict:
    det_id = session.run_detector(a["capture_id"], a.get("detector", "fast"))
    obs: dict[str, Any] = session.result_summary(det_id)
    obs["latency_ms"] = float(session.results[det_id].latency_ms)
    obs["deep_calls_used"] = session.counters["deep_calls"]
    obs.update(_fused(session))
    _safe_artifact(session, lambda: _render_result(session, det_id), "heatmap", obs)
    return obs


def tool_inspect_region(session: InspectionSession, a: dict) -> dict:
    rid = a["region_id"]
    region = session.require_region(rid)
    s = settings_for_call(session, "inspect_region", a)
    assert s is not None
    cap = session.take_capture(s)
    qid, q = session.run_quality(cap.capture_id)
    det_id = session.run_detector(cap.capture_id, "fast", parent_region=rid)
    res = session.results[det_id]
    peak, map_max, overlapping = session.localize(rid, det_id)
    localized = bool(overlapping) or (map_max > 0.0 and peak >= LOCALIZATION_PEAK_FRAC * map_max)
    # the same quality rule the detectors use: tolerated noise may confirm, only strictly good quality refutes
    blocking = blocking_quality_issues(q) if not q.ok else []
    status, reason = classify_inspection(float(res.prob), bool(res.reliable), bool(q.ok), localized, list(q.issues), blocking=blocking)
    insp = Inspection(
        region_id=rid,
        capture_id=cap.capture_id,
        quality_id=qid,
        result_id=det_id,
        zoom=float(cap.settings.view.zoom),
        prob=float(res.prob),
        reliable=bool(res.reliable),
        quality_ok=bool(q.ok),
        quality_issues=list(q.issues),
        region_peak=float(peak),
        map_max=float(map_max),
        overlapping_regions=list(overlapping),
        status=status,
        reason=reason,
    )
    session.inspections.setdefault(rid, []).append(insp)
    session.counters["zoom_inspections"] += 1
    obs: dict[str, Any] = {
        "region_id": rid,
        "zoom_capture_id": cap.capture_id,
        "settings": cap.settings.to_dict(),
        "sim_time_ms": cap.sim_time_ms,
        "quality": session.quality_summary(cap.capture_id),
        "result_id": det_id,
        "prob": float(res.prob),
        "reliable": bool(res.reliable),
        "registration": {k: (res.registration or {}).get(k) for k in ("method", "ok", "ncc")},
        "region_peak": float(peak),
        "map_max": float(map_max),
        "overlapping_regions": overlapping,
        "zoom_region_ids": [r.region_id for r in res.regions],
        "confirmed": status == "confirmed",
        "status": status,
        "reason": reason,
        "thresholds": {"zoom_confirm_prob": ZOOM_CONFIRM_PROB, "zoom_refute_prob": ZOOM_REFUTE_PROB},
    }
    obs.update(_fused(session))

    def render() -> np.ndarray:
        img = _render_result(session, det_id)
        # candidate box mapped into the zoom view (cyan)
        M = cap.scene_to_capture()
        x, y, w, h = region.bbox_scene
        p0 = M @ np.array([x, y, 1.0])
        p1 = M @ np.array([x + w, y + h, 1.0])
        img = np.ascontiguousarray(img.copy())
        cv2.rectangle(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (255, 255, 0), 2)
        return img

    _safe_artifact(session, render, "zoom", obs)
    return obs


def _measurement_target(session: InspectionSession, region_id: str):
    """Prefer the overlapping region of a confirming zoom view (higher resolution)."""
    for insp in reversed(session.inspections.get(region_id, [])):
        if insp.status == "confirmed" and insp.overlapping_regions:
            zr = session.regions.get(insp.overlapping_regions[0])
            if zr is not None:
                return zr
    return session.regions[region_id]


def tool_measure_defect(session: InspectionSession, a: dict) -> dict:
    rid = a["region_id"]
    session.require_region(rid)
    target = _measurement_target(session, rid)
    cap = session.captures[target.capture_id]
    res = session.results[session.region_result[target.region_id]]
    px_per_mm = float(PX_PER_MM.get(session.part.category, 20.0))
    m = dict(session.vision.measure_region(target, cap, res.anomaly_map, px_per_mm))
    session.measurements[rid] = m
    session.counters["measurements"] += 1
    obs: dict[str, Any] = {
        "region_id": rid,
        "measured_region_id": target.region_id,
        "capture_id": target.capture_id,
        "px_per_mm_scene": px_per_mm,
        **m,
    }

    def render() -> np.ndarray:
        x, y, w, h = (int(v) for v in target.bbox_capture)
        img = cap.image
        H, W = img.shape[:2]
        pad = max(16, int(0.75 * max(w, h)))
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
        crop = np.ascontiguousarray(img[y0:y1, x0:x1].copy())
        cv2.rectangle(crop, (x - x0, y - y0), (x - x0 + w, y - y0 + h), (0, 0, 255), 1)
        scale = max(1.0, 256.0 / max(1, max(crop.shape[:2])))
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return crop

    _safe_artifact(session, render, "measure", obs)
    return obs


def tool_request_human_review(session: InspectionSession, a: dict) -> dict:
    p, ev = session.fused_prob()
    rid = session.open_review(a["reason"], a["proposed_label"], float(a.get("confidence", 0.5)), ev, source="tool")
    return {
        "review_id": rid,
        "status": "pending",
        "proposed_label": a["proposed_label"],
        "confidence": float(a.get("confidence", 0.5)),
        "fused_prob": p,
        "fused_evidence": ev,
        "terminal": True,
    }


def tool_finish(session: InspectionSession, a: dict) -> dict:
    p, ev = session.fused_prob()
    requested = list(a.get("region_ids", []))
    known = [r for r in requested if r in session.regions]
    session.proposed = {
        "label": a["label"],
        "confidence": float(a["confidence"]),
        "rationale": a["rationale"],
        "region_ids": known,
        "fused_prob": p,
        "fused_evidence": ev,
    }
    obs: dict[str, Any] = {
        "proposed_label": a["label"],
        "confidence": float(a["confidence"]),
        "region_ids": known,
        "fused_prob": p,
        "fused_evidence": ev,
        "status": "proposed (subject to human-control policy gating)",
        "terminal": True,
    }
    unknown = [r for r in requested if r not in session.regions]
    if unknown:
        obs["unknown_region_ids"] = unknown
    return obs


_IMPLS: dict[str, Callable[[InspectionSession, dict], dict]] = {
    "capture_image": tool_capture_image,
    "assess_quality": tool_assess_quality,
    "detect_anomalies": tool_detect_anomalies,
    "inspect_region": tool_inspect_region,
    "measure_defect": tool_measure_defect,
    "request_human_review": tool_request_human_review,
    "finish": tool_finish,
}


# ---------------------------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------------------------
def _coerce(value: Any, sub: dict | None, key: str, notes: list[str]) -> Any:
    """Lenient coercion of common LLM argument mistakes (numeric strings, 'true', scalar for array)."""
    if sub is None:
        return value
    t = sub.get("type")
    if t == "number" and isinstance(value, str):
        try:
            v = float(value.strip())
        except ValueError:
            return value
        notes.append(f"{key}: coerced string {value!r} to number")
        return v
    if t == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0", "yes", "no"):
        notes.append(f"{key}: coerced string {value!r} to boolean")
        return value.strip().lower() in ("true", "1", "yes")
    if t == "array" and isinstance(value, str):
        notes.append(f"{key}: wrapped string into a list")
        return [value]
    return value


def _clamp(value: Any, sub: dict, key: str, notes: list[str]) -> Any:
    t = sub.get("type")
    if t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
        lo = sub.get("minimum", sub.get("exclusiveMinimum", -math.inf))
        hi = sub.get("maximum", sub.get("exclusiveMaximum", math.inf))
        v = float(min(max(float(value), lo), hi))
        if v != value:
            notes.append(f"{key}: clamped {value} to {v}")
        return v
    if t == "string" and isinstance(value, str) and "maxLength" in sub and len(value) > sub["maxLength"]:
        notes.append(f"{key}: truncated to {sub['maxLength']} characters")
        return value[: sub["maxLength"]]
    if t == "array" and isinstance(value, list):
        out = list(value)
        if "maxItems" in sub and len(out) > sub["maxItems"]:
            notes.append(f"{key}: truncated to {sub['maxItems']} items")
            out = out[: sub["maxItems"]]
        item = sub.get("items", {})
        if item.get("type") == "string" and "maxLength" in item:
            out = [s[: item["maxLength"]] if isinstance(s, str) else s for s in out]
        return out
    return value


class ToolRegistry:
    """Named tools with JSON-Schema validation, clamping and error capture."""

    def __init__(self, tools: Iterable[ToolSpec] = ()) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._validators: dict[str, Draft202012Validator] = {}
        for t in tools:
            self.register(t)

    def register(self, spec: ToolSpec) -> None:
        Draft202012Validator.check_schema(spec.input_schema)
        self._tools[spec.name] = spec
        self._validators[spec.name] = Draft202012Validator(spec.input_schema)

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def is_terminal(self, name: str) -> bool:
        t = self._tools.get(name)
        return bool(t and t.terminal)

    def specs(self) -> list[dict]:
        """``[{"name", "description", "input_schema"}]`` (deep copies)."""
        return [t.to_spec() for t in self._tools.values()]

    def bedrock_tool_config(self) -> dict:
        """Ready-to-use Bedrock Converse ``toolConfig``."""
        return {
            "tools": [
                {"toolSpec": {"name": s["name"], "description": s["description"], "inputSchema": {"json": s["input_schema"]}}}
                for s in self.specs()
            ]
        }

    def mcp_tools(self) -> list[dict]:
        """MCP-style tool listing (``inputSchema`` key)."""
        return [{"name": s["name"], "description": s["description"], "inputSchema": s["input_schema"]} for s in self.specs()]

    def validate(self, name: str, args: Any) -> tuple[dict, list[str]]:
        """Validate (jsonschema) then clamp; returns (clean args with defaults, notes)."""
        spec = self._tools.get(name)
        if spec is None:
            raise ToolArgumentError(f"unknown tool {name!r}; known tools: {', '.join(self._tools)}")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise ToolArgumentError(f"arguments of {name} must be a JSON object, got {type(args).__name__}")
        schema = spec.input_schema
        props: dict = schema.get("properties", {})
        required = set(schema.get("required", []))
        notes: list[str] = []
        clean: dict[str, Any] = {}
        for k, v in args.items():
            if v is None and k in props and k not in required:
                notes.append(f"{k}: null ignored")
                continue
            v = _coerce(v, props.get(k), k, notes)
            if isinstance(v, float) and not math.isfinite(v):
                raise ToolArgumentError(f"{name}: argument {k} must be a finite number")
            clean[k] = v
        validator = self._validators[name]
        hard: list[str] = []
        for err in validator.iter_errors(clean):
            path = list(err.absolute_path)
            key = path[0] if path else None
            if err.validator in _CLAMPABLE and isinstance(key, str) and key in props:
                clean[key] = _clamp(clean[key], props[key], key, notes)
            else:
                hard.append(f"{'/'.join(str(p) for p in path) or '<args>'}: {err.message}")
        if hard:
            raise ToolArgumentError(f"invalid arguments for {name}: " + "; ".join(sorted(hard)))
        remaining = [e.message for e in validator.iter_errors(clean)]
        if remaining:
            raise ToolArgumentError(f"invalid arguments for {name}: " + "; ".join(remaining))
        for k, sub in props.items():
            if k not in clean and "default" in sub:
                clean[k] = copy.deepcopy(sub["default"])
        return clean, notes

    def execute(self, session: InspectionSession, name: str, args: Any) -> dict:
        """Validate, clamp, run and record one tool call; never raises for tool failures."""
        t0 = time.perf_counter()
        render0 = session.render_ms
        before = set(session.artifacts)
        session.counters["steps"] += 1
        clean: Any = args
        try:
            clean, notes = self.validate(name, args)
            obs = self._tools[name].fn(session, clean)
            if notes:
                obs["arg_notes"] = notes
        except (ToolArgumentError, SessionError) as exc:
            session.counters["errors"] += 1
            obs = {"error": str(exc), "tool": name}
        except Exception as exc:  # tool failure -> observation, the loop/policy decide what next
            session.counters["errors"] += 1
            log.exception("tool %s failed", name)
            obs = {"error": f"{type(exc).__name__}: {exc}", "tool": name}
        obs["artifacts"] = [a for a in session.artifacts if a not in before]
        obs = jsonable(obs)
        dt = (time.perf_counter() - t0) * 1000.0 - (session.render_ms - render0)
        dt = max(dt, 0.0)
        session.compute_ms += dt
        session.last_tool_ms = dt
        session.last_observation = obs
        session.history.append(
            {"n": session.counters["steps"], "tool": name, "args": jsonable(clean), "error": obs.get("error")}
        )
        return obs


def default_registry(settings: Settings | None = None) -> ToolRegistry:
    """Registry with the seven InspectAgent tools."""
    schemas = build_schemas(settings)
    return ToolRegistry(
        ToolSpec(name, DESCRIPTIONS[name], schemas[name], _IMPLS[name], terminal=name in TERMINAL_TOOLS) for name in TOOL_NAMES
    )
