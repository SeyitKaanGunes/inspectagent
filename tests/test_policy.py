"""Human-control policy: guardrails, budget enforcement and final routing."""

from __future__ import annotations

import pytest

from inspectagent.agent.policy import HumanControlPolicy
from inspectagent.agent.tools import default_registry
from inspectagent.config import PolicyConfig
from inspectagent.types import Decision
from tests.agent_fakes import DEFECT, FALSE_ALARM, Spot, make_session

REG = default_registry()


def _dec(tool: str, args: dict | None = None, step: int = 1) -> Decision:
    return Decision(step=step, tool=tool, args=args or {}, rationale="test", evidence=["det_1"], planner="test")


def _session_with_prob(p_full: float, register_ok: bool = True, **kw):
    spots = [Spot((380.0, 240.0, 24.0, 16.0), p_full=p_full, p_zoom=0.97)] if p_full > 0.1 else []
    s, _ = make_session(spots=spots, register_ok=register_ok, **kw)
    REG.execute(s, "capture_image", {})
    REG.execute(s, "assess_quality", {"capture_id": "cap_1"})
    REG.execute(s, "detect_anomalies", {"capture_id": "cap_1"})
    return s


def test_unknown_tool_goes_to_review():
    s, _ = make_session()
    d, ov = HumanControlPolicy().vet(_dec("format_disk"), s)
    assert ov and d.tool == "request_human_review" and d.planner == "policy"
    assert "unknown tool" in d.rationale and d.args["proposed_label"] in ("PASS", "FAIL")
    # the replacement is always executable
    REG.validate(d.tool, d.args)


def test_review_is_always_allowed():
    s, _ = make_session()
    dec = _dec("request_human_review", {"reason": "r", "proposed_label": "PASS"})
    assert HumanControlPolicy().vet(dec, s) == (dec, False)


def test_finish_pass_needs_low_prob_and_reliable_evidence():
    pol = HumanControlPolicy()
    s = _session_with_prob(0.03)
    dec = _dec("finish", {"label": "PASS", "confidence": 0.97, "rationale": "clean"})
    assert pol.vet(dec, s) == (dec, False)

    s = _session_with_prob(0.4)
    d, ov = pol.vet(dec, s)
    assert ov and d.tool == "request_human_review" and d.args["proposed_label"] == "PASS"
    assert "0.400>auto_pass_max_prob=0.15" in d.rationale and "det_1" in d.evidence

    s = _session_with_prob(0.03, register_ok=False)  # low p but no reliable evidence
    d, ov = pol.vet(dec, s)
    assert ov and "without reliable" in d.rationale

    s, _ = make_session()  # no detection at all
    d, ov = pol.vet(dec, s)
    assert ov and d.tool == "request_human_review"


def test_finish_fail_needs_high_prob():
    pol = HumanControlPolicy()
    dec = _dec("finish", {"label": "FAIL", "confidence": 0.9, "rationale": "defect"})
    s = _session_with_prob(0.95)
    assert pol.vet(dec, s) == (dec, False)
    s = _session_with_prob(0.6)
    d, ov = pol.vet(dec, s)
    assert ov and d.args["proposed_label"] == "FAIL" and "0.600<auto_fail_min_prob=0.90" in d.rationale
    s = _session_with_prob(0.95, register_ok=False)
    d, ov = pol.vet(dec, s)
    assert ov and "without reliable evidence" in d.rationale


def test_finish_with_invalid_label_goes_to_review():
    s = _session_with_prob(0.03)
    d, ov = HumanControlPolicy().vet(_dec("finish", {"label": "MAYBE"}), s)
    assert ov and d.tool == "request_human_review"


def test_policy_respects_custom_thresholds():
    s = _session_with_prob(0.4, region_min_prob=0.5)  # image-level p only, no candidate region
    dec = _dec("finish", {"label": "PASS", "confidence": 0.6, "rationale": "r"})
    assert HumanControlPolicy(PolicyConfig(auto_pass_max_prob=0.5)).vet(dec, s) == (dec, False)


def test_pass_with_uninspected_candidate_is_blocked():
    """Regression: an LLM/MCP finish(PASS) passed vet with an un-inspected candidate the rule planner would zoom
    into (here the zoom would confirm a defect at p=0.97 that the full view diluted to p=0.12)."""
    pol = HumanControlPolicy()
    s, _ = make_session(spots=[Spot((380.0, 240.0, 24.0, 16.0), p_full=0.12, p_zoom=0.97)])
    for tool, args in (("capture_image", {}), ("assess_quality", {"capture_id": "cap_1"}), ("detect_anomalies", {"capture_id": "cap_1"})):
        REG.execute(s, tool, args)
    assert s.fused_prob()[0] == pytest.approx(0.12) and [r.region_id for r in s.full_view_regions()] == ["reg_1_1"]
    dec = _dec("finish", {"label": "PASS", "confidence": 0.88, "rationale": "p=0.12"}, step=4)
    d, ov = pol.vet(dec, s)
    assert ov and d.tool == "request_human_review" and "unresolved candidate(s) ['reg_1_1']" in d.rationale
    assert getattr(d, "review_trigger") == "visual"
    # the final gate refuses an unvetted PASS as well
    REG.execute(s, "finish", {"label": "PASS", "confidence": 0.88, "rationale": "p=0.12"})
    v = pol.final_verdict(s)
    assert v.label == "NEEDS_REVIEW" and "unresolved candidate" in v.rationale


def test_pass_allowed_when_zoom_candidates_are_exhausted_and_deep_corroborated():
    """The rule planner's escape hatch: 3 refuted zooms + deep on the full view -> PASS stays allowed."""
    spots = [
        Spot((100.0, 80.0, 20.0, 20.0), 0.12, 0.04, p_deep=0.04),
        Spot((300.0, 200.0, 20.0, 20.0), 0.11, 0.05, p_deep=0.04),
        Spot((480.0, 300.0, 20.0, 20.0), 0.105, 0.03, p_deep=0.04),
        Spot((500.0, 40.0, 20.0, 20.0), 0.101, 0.03, p_deep=0.04),
    ]
    pol = HumanControlPolicy()
    s, _ = make_session(spots=spots, region_min_prob=0.1)
    for tool, args in (("capture_image", {}), ("assess_quality", {"capture_id": "cap_1"}), ("detect_anomalies", {"capture_id": "cap_1"})):
        REG.execute(s, tool, args)
    regs = [r.region_id for r in s.full_view_regions()]
    assert len(regs) == 4
    dec = _dec("finish", {"label": "PASS", "confidence": 0.9, "rationale": "r"}, step=s.counters["steps"] + 1)
    for rid in regs[:3]:
        assert pol.vet(dec, s)[1] is True  # still a candidate to inspect
        REG.execute(s, "inspect_region", {"region_id": rid})
    dec.step = s.counters["steps"] + 1
    d, ov = pol.vet(dec, s)
    assert ov and "deep detector was not run" in d.rationale  # 3 zooms used, deep still possible
    REG.execute(s, "detect_anomalies", {"capture_id": "cap_1", "detector": "deep"})
    dec.step = s.counters["steps"] + 1
    assert pol.vet(dec, s) == (dec, False)


def test_unlocalized_zoom_alarm_blocks_pass():
    a = Spot((300.0, 200.0, 20.0, 20.0), p_full=0.12, p_zoom=0.05)
    b = Spot((370.0, 200.0, 20.0, 20.0), p_full=0.0, p_zoom=0.95)  # only visible on the zoom view
    s, _ = make_session(spots=[a, b])
    for tool, args in (("capture_image", {}), ("assess_quality", {"capture_id": "cap_1"}), ("detect_anomalies", {"capture_id": "cap_1"})):
        REG.execute(s, tool, args)
    obs = REG.execute(s, "inspect_region", {"region_id": "reg_1_1"})
    assert obs["status"] == "inconclusive" and "not at the candidate location" in obs["reason"]
    assert s.unlocalized_zoom_alarms() == ["det_2"] and not s.zoom_positive("det_2")
    assert s.fused_prob()[0] == pytest.approx(0.12)  # an unlocalized zoom is not confirmation
    d, ov = HumanControlPolicy().vet(_dec("finish", {"label": "PASS", "confidence": 0.9, "rationale": "r"}, step=5), s)
    assert ov and "unlocalized zoom alarm" in d.rationale


def test_wall_clock_deadline_forces_terminal_review():
    import time

    s, _ = make_session()
    pol = HumanControlPolicy(deadline=time.monotonic() + 2.0)  # inside the safety margin
    d, ov = pol.vet(_dec("capture_image", {}), s)
    assert ov and d.tool == "request_human_review" and "wall-clock budget" in d.rationale
    assert getattr(d, "review_trigger") == "budget"
    fin = _dec("request_human_review", {"reason": "r", "proposed_label": "PASS"})
    assert pol.vet(fin, s) == (fin, False)  # terminal actions stay allowed
    assert HumanControlPolicy(deadline=time.monotonic() + 600.0).vet(_dec("capture_image", {}), s)[1] is False


def test_capture_budget():
    pol = HumanControlPolicy(PolicyConfig(max_captures=1))
    s, _ = make_session()
    assert pol.vet(_dec("capture_image", {}), s)[1] is False
    REG.execute(s, "capture_image", {})
    d, ov = pol.vet(_dec("capture_image", {}, step=2), s)
    assert ov and "capture budget exhausted: 1/1" in d.rationale


def test_inspect_region_counts_as_capture():
    pol = HumanControlPolicy(PolicyConfig(max_captures=1))
    s = _session_with_prob(0.6)
    d, ov = pol.vet(_dec("inspect_region", {"region_id": "reg_1_1"}, step=4), s)
    assert ov and d.tool == "request_human_review"


def test_deep_budget():
    pol = HumanControlPolicy(PolicyConfig(max_deep_calls=1))
    s = _session_with_prob(0.5)
    dec = _dec("detect_anomalies", {"capture_id": "cap_1", "detector": "deep"}, step=4)
    assert pol.vet(dec, s)[1] is False
    REG.execute(s, "detect_anomalies", {"capture_id": "cap_1", "detector": "deep"})
    d, ov = pol.vet(_dec("detect_anomalies", {"capture_id": "cap_1", "detector": "deep"}, step=5), s)
    assert ov and "deep-call budget exhausted: 1/1" in d.rationale
    # fast calls are not limited by the deep budget
    assert pol.vet(_dec("detect_anomalies", {"capture_id": "cap_1", "detector": "fast"}, step=5), s)[1] is False


def test_sim_time_budget():
    pol = HumanControlPolicy(PolicyConfig(max_sim_time_ms=150.0))
    s, _ = make_session()
    REG.execute(s, "capture_image", {"autofocus": True})  # 60+25+250 ms > 150 ms
    d, ov = pol.vet(_dec("assess_quality", {"capture_id": "cap_1"}, step=2), s)
    assert ov and "sim-time budget exhausted" in d.rationale
    # a capture that would overrun the remaining time is refused before it happens
    s2, _ = make_session()
    pol2 = HumanControlPolicy(PolicyConfig(max_sim_time_ms=200.0))
    d, ov = pol2.vet(_dec("capture_image", {"autofocus": True}), s2)
    assert ov and "sim-time budget" in d.rationale


def test_last_step_must_be_terminal():
    pol = HumanControlPolicy(PolicyConfig(max_steps=3))
    s, _ = make_session()
    d, ov = pol.vet(_dec("capture_image", {}, step=3), s)
    assert ov and "step 3/3 must be terminal" in d.rationale
    fin = _dec("request_human_review", {"reason": "r", "proposed_label": "PASS"}, step=3)
    assert pol.vet(fin, s) == (fin, False)


def test_two_errors_force_review():
    pol = HumanControlPolicy()
    s, _ = make_session()
    REG.execute(s, "assess_quality", {"capture_id": "nope"})
    assert pol.vet(_dec("capture_image", {}, step=2), s)[1] is False
    REG.execute(s, "assess_quality", {"capture_id": "nope"})
    d, ov = pol.vet(_dec("capture_image", {}, step=3), s)
    assert ov and "2 tool errors" in d.rationale


# ------------------------------------------------------------------ final verdict
def test_final_verdict_pass_routes_pass():
    s = _session_with_prob(0.03)
    REG.execute(s, "finish", {"label": "PASS", "confidence": 0.97, "rationale": "clean part"})
    v = HumanControlPolicy().final_verdict(s)
    assert v.label == "PASS" and v.action == "route_pass" and v.review_id is None
    assert v.confidence == pytest.approx(0.97) and v.prob_anomalous == pytest.approx(0.03)
    assert "clean part" in v.rationale and "route_pass" in v.rationale


def test_final_verdict_fail_quarantine_or_scrap():
    s, _ = make_session(spots=[DEFECT])
    REG.execute(s, "capture_image", {})
    REG.execute(s, "assess_quality", {"capture_id": "cap_1"})
    REG.execute(s, "detect_anomalies", {"capture_id": "cap_1"})
    REG.execute(s, "inspect_region", {"region_id": "reg_1_1"})
    REG.execute(s, "finish", {"label": "FAIL", "confidence": 0.97, "rationale": "scratch", "region_ids": ["reg_1_1"]})
    v = HumanControlPolicy().final_verdict(s)
    assert v.label == "FAIL" and v.action == "route_quarantine" and "scrap requires a human" in v.rationale
    assert [r.region_id for r in v.defect_regions] == ["reg_1_1"]
    v2 = HumanControlPolicy(PolicyConfig(scrap_requires_human=False)).final_verdict(s)
    assert v2.action == "route_scrap"


def test_final_verdict_review_has_ticket():
    s = _session_with_prob(0.5)
    REG.execute(s, "request_human_review", {"reason": "ambiguous", "proposed_label": "FAIL", "confidence": 0.5})
    v = HumanControlPolicy().final_verdict(s)
    assert v.label == "NEEDS_REVIEW" and v.action == "hold_for_review"
    assert v.review_id == s.review["review_id"] and v.review_id.startswith("rev_")
    assert v.defect_regions  # proposed FAIL -> candidate regions are shown to the reviewer


def test_final_gate_rechecks_unvetted_finish():
    """A finish that bypassed vet (e.g. a direct MCP call without policy) is still gated."""
    s = _session_with_prob(0.5)
    REG.execute(s, "finish", {"label": "PASS", "confidence": 0.9, "rationale": "looks fine"})
    v = HumanControlPolicy().final_verdict(s)
    assert v.label == "NEEDS_REVIEW" and v.review_id is not None and "final gate" in v.rationale


def test_final_verdict_without_terminal_decision():
    s, _ = make_session()
    v = HumanControlPolicy().final_verdict(s)
    assert v.label == "NEEDS_REVIEW" and v.review_id is not None and "without a terminal decision" in v.rationale


def test_refutation_enables_auto_pass():
    """Visual evidence (a refuting zoom) changes what the policy allows."""
    pol = HumanControlPolicy()
    s, _ = make_session(spots=[FALSE_ALARM])
    REG.execute(s, "capture_image", {})
    REG.execute(s, "assess_quality", {"capture_id": "cap_1"})
    REG.execute(s, "detect_anomalies", {"capture_id": "cap_1"})
    dec = _dec("finish", {"label": "PASS", "confidence": 0.9, "rationale": "r"}, step=4)
    assert pol.vet(dec, s)[1] is True
    REG.execute(s, "inspect_region", {"region_id": "reg_1_1"})
    assert pol.vet(dec, s) == (dec, False)
