"""Contested evidence: an aligned-but-unreliable alarm contradicting a reliable result on the same capture.

Integration guardrail (session + policy + rule planner): when the fast detector flags a defect on a
capture where it is unreliable for a non-registration reason while the alignment-free deep detector
calls the same capture normal, the part must not be auto-PASSed until a zoom refutes the alarm.

The fakes model "unreliable for a non-registration reason" with ``FakeStat`` (``reliable =
register_ok and quality.ok``) on a full view whose quality report only says ``noisy`` (noise 8 <=
``NOISE_TOLERANCE_SIGMA``, so the planner may still run the deep detector on it).
"""

from __future__ import annotations

from inspectagent.agent.policy import HumanControlPolicy
from inspectagent.agent.session import CONTESTED_ALARM_PROB
from inspectagent.agent.tools import default_registry
from inspectagent.types import Decision, QualityReport
from inspectagent.vision.quality import NOISE_TOLERANCE_SIGMA, blocking_quality_issues
from tests.agent_fakes import FakeVision, Spot, make_session, run_fake

REG = default_registry()
BOX = (380.0, 240.0, 24.0, 16.0)


class _NoisyFullView(FakeVision):
    """Full views are slightly noisy (fast detector fake -> unreliable); zoom views are clean."""

    def assess_quality(self, capture, reference_sharpness=None, cfg=None) -> QualityReport:
        q = super().assess_quality(capture, reference_sharpness, cfg)
        if capture.settings.view.zoom <= 1.05:
            q.issues = ["noisy"]
            q.noise_sigma = 8.0
            q.ok = False
        return q


def _finish_pass() -> Decision:
    return Decision(step=1, tool="finish", args={"label": "PASS", "confidence": 0.95, "rationale": "deep says normal"},
                    rationale="test", evidence=[], planner="test")


def _contested_session(p_zoom: float = 0.04):
    s, _ = make_session(spots=[Spot(BOX, p_full=0.95, p_zoom=p_zoom, p_deep=0.05)], vision=_NoisyFullView())
    REG.execute(s, "capture_image", {})
    REG.execute(s, "assess_quality", {"capture_id": "cap_1"})
    REG.execute(s, "detect_anomalies", {"capture_id": "cap_1", "detector": "fast"})
    REG.execute(s, "detect_anomalies", {"capture_id": "cap_1", "detector": "deep"})
    return s


def test_blocking_quality_issues_tolerates_moderate_noise_only():
    def q(issues, noise):
        return QualityReport("c", 50.0, 1.0, 120.0, 0.0, 0.0, noise, 0.0, [], issues, 0.0, not issues)

    assert blocking_quality_issues(q(["noisy"], 8.0)) == []
    assert blocking_quality_issues(q(["noisy"], NOISE_TOLERANCE_SIGMA + 0.1)) == ["noisy"]
    assert blocking_quality_issues(q(["glare", "noisy"], 8.0)) == ["glare"]
    assert blocking_quality_issues(q(["blurred", "underexposed", "overexposed"], 1.0)) == ["blurred", "underexposed", "overexposed"]


def test_session_lists_contested_results():
    s = _contested_session()
    p, ev = s.fused_prob()
    assert p < 0.15 and ev[0] == "det_2"  # only the reliable deep result is eligible
    assert s.results["det_1"].prob >= CONTESTED_ALARM_PROB and not s.results["det_1"].reliable
    assert s.contested_results() == ["det_1"]
    assert [r.region_id for r in s.contested_regions()] == ["reg_1_1"]
    note = s.contested_note()
    assert "det_1 (fast on cap_1, registration ok" in note and "unrefuted regions ['reg_1_1']" in note
    assert "vs reliable det_2 (deep)" in note
    assert s.planner_state()["contested_evidence"] == note


def test_policy_blocks_pass_against_contested_evidence():
    s = _contested_session()
    pol = HumanControlPolicy()
    d, overridden = pol.vet(_finish_pass(), s)
    assert overridden and d.tool == "request_human_review" and d.planner == "policy"
    assert "conflicting evidence" in d.rationale and d.args["confidence"] == 0.5
    # the final gate refuses a PASS that bypassed vet() as well
    REG.execute(s, "finish", _finish_pass().args)
    v = pol.final_verdict(s)
    assert v.label == "NEEDS_REVIEW" and "conflicting evidence" in v.rationale


def test_refuting_zoom_clears_the_conflict():
    s = _contested_session(p_zoom=0.04)
    obs = REG.execute(s, "inspect_region", {"region_id": "reg_1_1"})
    assert obs["status"] == "refuted"
    assert s.contested_results() == []
    d, overridden = HumanControlPolicy().vet(_finish_pass(), s)
    assert not overridden and d.tool == "finish"


def test_superseded_alarm_is_not_contested():
    """A reliable result on a *different* (better) capture supersedes the degraded one."""
    s = _contested_session()
    s.results["det_2"].capture_id = "cap_other"  # pretend the deep evidence came from another capture
    assert s.contested_results() == []


def test_misregistered_alarm_is_not_contested():
    s = _contested_session()
    s.results["det_1"].registration["ok"] = False
    assert s.contested_results() == []


def test_rule_planner_resolves_conflict_by_zoom_then_passes():
    trace, s = run_fake("clean", spots=[Spot(BOX, p_full=0.95, p_zoom=0.04, p_deep=0.05)], vision=_NoisyFullView())
    tools = [st.decision.tool for st in trace.steps]
    assert tools == ["capture_image", "assess_quality", "detect_anomalies", "detect_anomalies", "inspect_region", "finish"]
    zoom = trace.steps[4].decision
    assert "conflicting evidence" in zoom.rationale and "det_1" in zoom.evidence and zoom.args["region_id"] == "reg_1_1"
    assert trace.verdict.label == "PASS" and trace.verdict.action == "route_pass"


def test_rule_planner_sends_unresolved_conflict_to_review():
    trace, s = run_fake("clean", spots=[Spot(BOX, p_full=0.95, p_zoom=0.45, p_deep=0.05)], vision=_NoisyFullView(), label=1)
    last = trace.steps[-1].decision
    assert "inspect_region" in [st.decision.tool for st in trace.steps]
    assert last.tool == "request_human_review" and "contradicted by det_1" in last.rationale
    assert last.args["confidence"] <= 0.5
    assert trace.verdict.label == "NEEDS_REVIEW"  # never a silent PASS
