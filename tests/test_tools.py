"""Tool registry: JSON-Schema specs, validation + clamping, observations, artifacts, evidence fusion."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from jsonschema import Draft202012Validator

from inspectagent.agent.session import (
    REFUTATION_RESIDUAL,
    ZOOM_CONFIRM_PROB,
    ZOOM_REFUTE_PROB,
    classify_inspection,
    zoom_bucket,
)
from inspectagent.agent.tools import TERMINAL_TOOLS, TOOL_NAMES, ToolArgumentError, default_registry
from tests.agent_fakes import DEFECT, FALSE_ALARM, Spot, make_session

ALLOWED_KEYS = {"type", "properties", "required", "additionalProperties", "description", "minimum", "maximum", "enum", "default", "items", "maxItems", "minLength", "maxLength"}


def _walk_keys(schema: dict):
    yield from schema.keys()
    for sub in schema.get("properties", {}).values():
        yield from _walk_keys(sub)
    if isinstance(schema.get("items"), dict):
        yield from _walk_keys(schema["items"])


@pytest.fixture()
def reg():
    return default_registry()


def test_specs_are_valid_json_schema(reg):
    specs = reg.specs()
    assert [s["name"] for s in specs] == list(TOOL_NAMES)
    for s in specs:
        sch = s["input_schema"]
        Draft202012Validator.check_schema(sch)
        assert sch["type"] == "object"
        assert sch["additionalProperties"] is False
        assert isinstance(sch["required"], list)
        assert set(sch["required"]) <= set(sch["properties"])
        assert set(_walk_keys(sch)) <= ALLOWED_KEYS | set(sch["properties"])
        assert s["description"]
        json.dumps(s)  # plain JSON
    # numeric ranges from SPEC 3.1
    cap = next(s for s in specs if s["name"] == "capture_image")["input_schema"]["properties"]
    assert (cap["zoom"]["minimum"], cap["zoom"]["maximum"]) == (1.0, 4.0)
    assert (cap["exposure_ev"]["minimum"], cap["exposure_ev"]["maximum"]) == (-3.0, 3.0)
    assert cap["light"]["enum"] == ["ring", "diffuse", "low_angle"]
    insp = next(s for s in specs if s["name"] == "inspect_region")["input_schema"]["properties"]
    assert (insp["zoom"]["minimum"], insp["zoom"]["maximum"]) == (2.0, 4.0)


def test_bedrock_and_mcp_shapes(reg):
    cfg = reg.bedrock_tool_config()
    assert len(cfg["tools"]) == len(TOOL_NAMES)
    for t in cfg["tools"]:
        ts = t["toolSpec"]
        assert set(ts) == {"name", "description", "inputSchema"}
        assert ts["inputSchema"]["json"]["type"] == "object"
    for t in reg.mcp_tools():
        assert set(t) == {"name", "description", "inputSchema"}
    assert {n for n in TOOL_NAMES if reg.is_terminal(n)} == set(TERMINAL_TOOLS)


def test_validate_clamps_and_defaults(reg):
    args, notes = reg.validate("capture_image", {"cx": -0.2, "zoom": 7, "exposure_ev": "5"})
    assert args["cx"] == 0.0 and args["zoom"] == 4.0 and args["exposure_ev"] == 3.0
    assert args["cy"] == 0.5 and args["light"] == "ring" and args["autofocus"] is False
    assert any("clamped" in n for n in notes) and any("coerced" in n for n in notes)
    args, notes = reg.validate("finish", {"label": "PASS", "confidence": 1.7, "rationale": "x" * 5000, "region_ids": "reg_1_1"})
    assert args["confidence"] == 1.0 and len(args["rationale"]) == 4000 and args["region_ids"] == ["reg_1_1"]
    args, _ = reg.validate("inspect_region", {"region_id": "reg_1_1", "light": None})
    assert args == {"region_id": "reg_1_1", "zoom": 3.0}


@pytest.mark.parametrize(
    "tool,args",
    [
        ("capture_image", {"light": "laser"}),
        ("capture_image", {"bogus": 1}),
        ("capture_image", {"cx": True}),
        ("capture_image", {"cx": float("nan")}),
        ("assess_quality", {}),
        ("detect_anomalies", {"capture_id": "cap_1", "detector": "huge"}),
        ("finish", {"label": "MAYBE", "confidence": 0.5, "rationale": "r"}),
        ("request_human_review", {"reason": "", "proposed_label": "PASS"}),
    ],
)
def test_validate_rejects(reg, tool, args):
    with pytest.raises(ToolArgumentError):
        reg.validate(tool, args)


def test_execute_errors_become_observations(reg):
    s, _ = make_session()
    obs = reg.execute(s, "assess_quality", {"capture_id": "cap_9"})
    assert "error" in obs and "cap_9" in obs["error"]
    obs = reg.execute(s, "teleport", {})
    assert "unknown tool" in obs["error"]
    obs = reg.execute(s, "capture_image", {"light": "laser"})
    assert "error" in obs
    assert s.counters["errors"] == 3 and s.counters["steps"] == 3
    obs = reg.execute(s, "detect_anomalies", {"capture_id": "cap_1"})  # no capture yet
    assert "error" in obs


def test_execute_capture_quality_detect(reg):
    s, _ = make_session("glare", spots=[DEFECT])
    obs = reg.execute(s, "capture_image", {"zoom": 9})
    assert obs["capture_id"] == "cap_1" and obs["settings"]["view"]["zoom"] == 4.0
    assert obs["arg_notes"] and obs["artifact_id"] in s.artifacts and obs["artifacts"] == [obs["artifact_id"]]
    assert obs["sim_time_ms"] > 0 and s.sim_time_ms == pytest.approx(obs["sim_time_ms"])
    obs = reg.execute(s, "capture_image", {})
    q = reg.execute(s, "assess_quality", {"capture_id": obs["capture_id"]})
    assert q["quality_id"] == "q_1" and "glare" in q["issues"] and q["thresholds"]["max_glare_frac"] == pytest.approx(0.004)
    d = reg.execute(s, "detect_anomalies", {"capture_id": "cap_2"})
    assert d["result_id"] == "det_1" and d["view"] == "full" and d["detector"] == "fast"
    assert not d["reliable"]  # glare -> quality not ok -> registration-based detector unreliable
    assert [r["region_id"] for r in d["regions"]] == ["reg_1_1"]
    assert "fused_prob" in d and d["artifact_id"] in s.artifacts
    for o in (obs, q, d):
        json.dumps(o, allow_nan=False)
        assert not any(isinstance(v, np.ndarray) for v in o.values())
    jpeg = s.artifacts[d["artifact_id"]]
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[2] == 3


def test_deep_unavailable_is_error(reg):
    s, _ = make_session(with_deep=False)
    reg.execute(s, "capture_image", {})
    obs = reg.execute(s, "detect_anomalies", {"capture_id": "cap_1", "detector": "deep"})
    assert "not available" in obs["error"]


def test_inspect_measure_finish_review(reg):
    s, _ = make_session(spots=[DEFECT])
    reg.execute(s, "capture_image", {})
    reg.execute(s, "assess_quality", {"capture_id": "cap_1"})
    reg.execute(s, "detect_anomalies", {"capture_id": "cap_1"})
    obs = reg.execute(s, "inspect_region", {"region_id": "reg_1_1"})
    assert obs["status"] == "confirmed" and obs["confirmed"] is True
    assert obs["zoom_capture_id"] == "cap_2" and obs["settings"]["view"]["zoom"] == 3.0
    assert obs["prob"] >= ZOOM_CONFIRM_PROB and obs["region_peak"] > 0 and obs["overlapping_regions"]
    assert obs["fused_prob"] == pytest.approx(DEFECT.p_zoom)
    m = reg.execute(s, "measure_defect", {"region_id": "reg_1_1"})
    assert m["measured_region_id"] == obs["overlapping_regions"][0]  # measured on the zoom view
    assert m["area_mm2"] > 0 and s.measurements["reg_1_1"]["type_guess"] == "spot"
    f = reg.execute(s, "finish", {"label": "FAIL", "confidence": 0.9, "rationale": "r", "region_ids": ["reg_1_1", "reg_9_9"]})
    assert f["terminal"] and s.proposed["region_ids"] == ["reg_1_1"] and f["unknown_region_ids"] == ["reg_9_9"]
    r = reg.execute(s, "request_human_review", {"reason": "check", "proposed_label": "FAIL"})
    assert r["review_id"].startswith("rev_test_") and s.review["status"] == "pending" and s.review["confidence"] == 0.5


def test_inspect_unknown_region(reg):
    s, _ = make_session()
    obs = reg.execute(s, "inspect_region", {"region_id": "reg_1_1"})
    assert "unknown region_id" in obs["error"]


# ------------------------------------------------------------------ evidence fusion
def _detect_full(reg, s, detector="fast"):
    if not s.captures:
        reg.execute(s, "capture_image", {})
        reg.execute(s, "assess_quality", {"capture_id": "cap_1"})
    return reg.execute(s, "detect_anomalies", {"capture_id": "cap_1", "detector": detector})


def test_fusion_no_evidence_is_uninformative():
    s, _ = make_session()
    p, ev = s.fused_prob()
    assert p == 0.5 and ev == []
    assert not s.has_reliable_evidence("PASS") and not s.has_reliable_evidence("FAIL")


def test_refuted_zoom_lowers_fused_prob(reg):
    s, _ = make_session(spots=[FALSE_ALARM])
    _detect_full(reg, s)
    p0, ev0 = s.fused_prob()
    assert p0 == pytest.approx(FALSE_ALARM.p_full) and ev0 == ["det_1"]
    obs = reg.execute(s, "inspect_region", {"region_id": "reg_1_1"})
    assert obs["status"] == "refuted" and obs["prob"] < ZOOM_REFUTE_PROB
    p1, ev1 = s.fused_prob()
    assert p1 == pytest.approx(p0 * REFUTATION_RESIDUAL)  # single region -> whole share refuted
    assert ev1[:2] == ["det_1", "det_2"]
    assert "refuted share" in s.fusion_report()["explain"]


def test_confirmed_zoom_raises_fused_prob(reg):
    s, _ = make_session(spots=[DEFECT])
    _detect_full(reg, s)
    p0, _ = s.fused_prob()
    reg.execute(s, "inspect_region", {"region_id": "reg_1_1"})
    p1, ev1 = s.fused_prob()
    assert p1 > p0 and p1 == pytest.approx(DEFECT.p_zoom) and ev1[0] == "det_2"
    assert s.region_status("reg_1_1") == "confirmed"


def test_partial_refutation_uses_region_shares(reg):
    a = Spot((100.0, 80.0, 20.0, 20.0), p_full=0.6, p_zoom=0.04, peak=8.0)
    b = Spot((420.0, 280.0, 20.0, 20.0), p_full=0.5, p_zoom=0.05, peak=8.0)
    s, _ = make_session(spots=[a, b])
    d = _detect_full(reg, s)
    regs = {r["region_id"]: r for r in d["regions"]}
    w = {k: r["peak"] * np.sqrt(r["area_px"]) for k, r in regs.items()}
    top = max(w, key=w.get)
    reg.execute(s, "inspect_region", {"region_id": top})
    share = w[top] / sum(w.values())
    p, _ = s.fused_prob()
    assert p == pytest.approx(0.6 * (1 - (1 - REFUTATION_RESIDUAL) * share))
    other = next(k for k in regs if k != top)
    reg.execute(s, "inspect_region", {"region_id": other})
    assert s.fused_prob()[0] == pytest.approx(0.6 * REFUTATION_RESIDUAL)


def test_deep_combines_by_max_and_unreliable_fast_is_ignored(reg):
    spot = Spot((380.0, 240.0, 24.0, 16.0), p_full=0.7, p_zoom=0.9, p_deep=0.2)
    s, _ = make_session(spots=[spot], register_ok=False)
    _detect_full(reg, s)  # fast unreliable (registration failed)
    p, _ = s.fused_prob()
    rep = s.fusion_report()
    assert p == pytest.approx(0.7) and not rep["reliable"] and "[no reliable evidence]" in rep["explain"]
    _detect_full(reg, s, "deep")
    p, ev = s.fused_prob()
    assert p == pytest.approx(0.2) and ev == ["det_2"]  # only the reliable deep result counts
    assert s.has_reliable_evidence("PASS")
    # with a reliable fast result, deep combines by max (no noisy-OR double counting)
    s2, _ = make_session(spots=[Spot((380.0, 240.0, 24.0, 16.0), p_full=0.4, p_zoom=0.9, p_deep=0.6)])
    _detect_full(reg, s2)
    _detect_full(reg, s2, "deep")
    assert s2.fused_prob()[0] == pytest.approx(0.6)


def test_unreliable_or_bad_quality_zoom_is_inconclusive():
    assert classify_inspection(0.95, True, True, True)[0] == "confirmed"
    assert classify_inspection(0.95, True, True, False)[0] == "inconclusive"
    assert classify_inspection(0.05, True, True, False)[0] == "refuted"
    assert classify_inspection(0.05, False, True, True)[0] == "inconclusive"
    assert classify_inspection(0.05, True, False, True, ["glare"])[1].startswith("zoom quality issues")
    assert classify_inspection(0.45, True, True, True)[0] == "inconclusive"


def test_zoom_bucket():
    assert [zoom_bucket(z) for z in (1.0, 1.49, 1.5, 2.6, 3.5, 4.0)] == ["z1", "z1", "z2", "z3", "z4", "z4"]


def test_planner_state_is_compact_json(reg):
    s, _ = make_session(spots=[DEFECT])
    _detect_full(reg, s)
    reg.execute(s, "inspect_region", {"region_id": "reg_1_1"})
    st = s.planner_state()
    txt = json.dumps(st, allow_nan=False)
    assert len(txt) < 20000
    assert st["budget"]["captures_used"] == 2 and st["deep_available"] is True
    assert st["regions"][0]["status"] == "confirmed" and st["inspections"][0]["status"] == "confirmed"
    assert st["fused_evidence"][0] == "det_2" and st["history"][-1]["tool"] == "inspect_region"
