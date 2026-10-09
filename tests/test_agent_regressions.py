"""Regression tests for agent-layer review findings (fusion, zoom confirmation, evidence metrics, planner usage)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inspectagent.agent.planner_rule import RulePlanner
from inspectagent.agent.session import CROSS_REFUTATION_RESIDUAL, REFUTATION_RESIDUAL, classify_inspection
from inspectagent.agent.tools import default_registry
from inspectagent.config import PolicyConfig, Settings
from inspectagent.types import Decision, QualityReport
from tests.agent_fakes import FakeVision, Spot, make_session, run_fake

REG = default_registry()
ROOT = Path(__file__).resolve().parent.parent


def _tools(trace) -> list[str]:
    return [st.decision.tool for st in trace.steps]


# ------------------------------------------------------------------ cross-detector refutation
@pytest.mark.parametrize("p_deep", [0.70, 0.74])
def test_fast_zoom_cannot_clear_a_deep_only_detection(p_deep: float) -> None:
    """Regression: fast unregistrable -> deep p=0.70 with a region -> one fast zoom (blind to the defect)
    refuted it with LAMBDA=0.2 (fused 0.14) and the part was auto-PASSed."""
    spot = Spot(bbox=(380.0, 240.0, 24.0, 16.0), p_full=0.05, p_zoom=0.05, p_deep=p_deep)
    trace, s = run_fake("clean", spots=[spot], register_ok=False, zoom_register_ok=True)
    assert "inspect_region" in _tools(trace)
    assert trace.verdict.action != "route_pass" and trace.verdict.label != "PASS", trace.verdict.rationale
    term = next(t for t in s.fusion_report()["terms"] if t["kind"] == "full/deep")
    assert term["refuted_share_cross"] == pytest.approx(1.0)
    assert term["contrib"] == pytest.approx(p_deep * CROSS_REFUTATION_RESIDUAL)


def test_same_detector_refutation_keeps_the_original_residual() -> None:
    s, _ = make_session(spots=[Spot((150.0, 100.0, 20.0, 20.0), p_full=0.55, p_zoom=0.04, p_deep=0.05)])
    for tool, args in (("capture_image", {}), ("assess_quality", {"capture_id": "cap_1"}), ("detect_anomalies", {"capture_id": "cap_1"})):
        REG.execute(s, tool, args)
    REG.execute(s, "inspect_region", {"region_id": "reg_1_1"})
    term = s.fusion_report()["terms"][0]
    assert term["refuted_share_cross"] == 0.0 and term["contrib"] == pytest.approx(0.55 * REFUTATION_RESIDUAL)


# ------------------------------------------------------------------ unlocalized zoom alarm
def test_unlocalized_zoom_hit_becomes_a_candidate_and_is_reported() -> None:
    """Regression: a zoom on A that saw an anomaly only at B forced auto-FAIL reporting/measuring A."""
    a = Spot(bbox=(300.0, 200.0, 20.0, 20.0), p_full=0.50, p_zoom=0.05)
    b = Spot(bbox=(370.0, 200.0, 20.0, 20.0), p_full=0.0, p_zoom=0.95)
    trace, s = run_fake("clean", spots=[a, b])
    v = trace.verdict
    reported = [r.region_id for r in v.defect_regions]
    assert not (v.label == "FAIL" and reported == ["reg_1_1"]), (v.label, reported, _tools(trace))
    assert v.label in ("FAIL", "NEEDS_REVIEW")
    inspected = [st.decision.args["region_id"] for st in trace.steps if st.decision.tool == "inspect_region"]
    assert inspected[:2] == ["reg_1_1", "reg_2_1"]  # the zoom's own region is re-centred and inspected
    if v.label == "FAIL":
        assert reported == ["reg_2_1"]
        meas = next(st for st in trace.steps if st.decision.tool == "measure_defect")
        assert meas.decision.args["region_id"] == "reg_2_1"
        x, y, w, h = v.defect_regions[0].bbox_scene
        assert x >= 340.0  # at B, not at A


# ------------------------------------------------------------------ zoom confirmation vs quality gate
def test_tolerated_noise_confirms_but_cannot_refute() -> None:
    assert classify_inspection(0.99, True, False, True, ["noisy"], blocking=[])[0] == "confirmed"
    status, reason = classify_inspection(0.05, True, False, True, ["noisy"], blocking=[])
    assert status == "inconclusive" and "cannot refute" in reason
    assert classify_inspection(0.99, True, False, True, ["noisy", "glare"], blocking=["glare"])[0] == "inconclusive"
    assert classify_inspection(0.05, True, True, True, [], blocking=[])[0] == "refuted"


class _DimZoomVision(FakeVision):
    """Zoom views inherit the full-view EV and come out underexposed (fixable by raising the EV)."""

    def assess_quality(self, capture, reference_sharpness=None, cfg=None) -> QualityReport:
        q = super().assess_quality(capture, reference_sharpness, cfg)
        if capture.settings.view.zoom > 1.05 and capture.settings.exposure_ev < 0.5:
            q.issues = sorted(set(q.issues) | {"underexposed"})
            q.mean_luma, q.suggested_exposure_delta, q.ok = 56.0, 0.8, False
        return q


def test_inconclusive_confirmation_zoom_with_fixable_quality_is_retried_before_fail() -> None:
    """Regression: the strong-evidence branch (p in [0.9, 0.98)) went to measure+FAIL after an inconclusive
    (underexposed) confirmation zoom without the retry that the ambiguous branch had."""
    spot = Spot((380.0, 240.0, 24.0, 16.0), p_full=0.93, p_zoom=0.97)
    trace, s = run_fake("clean", spots=[spot], vision=_DimZoomVision())
    insp = [st for st in trace.steps if st.decision.tool == "inspect_region"]
    assert len(insp) == 2
    assert insp[0].observation["status"] == "inconclusive" and "underexposed" in insp[0].observation["quality"]["issues"]
    assert insp[1].decision.args.get("exposure_ev") == pytest.approx(0.8)
    assert insp[1].observation["status"] == "confirmed"
    assert trace.verdict.label == "FAIL"


# ------------------------------------------------------------------ evidence metric definitions
class _RaisingAfterDetect:
    name = "raising_late"

    def __init__(self) -> None:
        self.rule = RulePlanner()

    def next_action(self, state, session):
        if session.counters["steps"] >= 3:
            raise RuntimeError("LLM down")
        return self.rule.next_action(state, session)


class _BadArgsAfterDetect(_RaisingAfterDetect):
    name = "badargs_late"

    def next_action(self, state, session):
        if session.counters["steps"] >= 3:
            return Decision(step=0, tool="assess_quality", args={"capture_id": 42}, rationale="bad", evidence=["det_1"], planner="x")
        return self.rule.next_action(state, session)


@pytest.mark.parametrize("planner_cls,trigger", [(_RaisingAfterDetect, "planner_error"), (_BadArgsAfterDetect, "errors")])
def test_non_visual_reviews_do_not_count_as_evidence_driven(planner_cls, trigger) -> None:
    """Regression: budget/tool-error/planner-error reviews carried fusion evidence ids and were counted as
    'visual evidence changed the action'."""
    trace, _ = run_fake("clean", planner=planner_cls())
    last = trace.steps[-1]
    assert last.decision.tool == "request_human_review" and last.observation["review_trigger"] == trigger
    assert trace.totals["evidence_changed_action"] is False and trace.totals["evidence_driven_steps"] == []


def test_step_budget_review_is_not_evidence_driven() -> None:
    pol = PolicyConfig(max_steps=3)
    from inspectagent.agent.policy import HumanControlPolicy

    class Looping:
        name = "loop"

        def next_action(self, state, session):
            return Decision(step=0, tool="capture_image", args={"cx": 0.5, "cy": 0.5, "exposure_ev": 0.1 * session.counters["steps"]},
                            rationale="again", evidence=["q_1"], planner="loop")

    trace, _ = run_fake("clean", planner=Looping(), policy=HumanControlPolicy(pol), settings=Settings(policy=pol))
    assert trace.steps[-1].observation["review_trigger"] == "budget"
    assert all(k != "review" for k in trace.totals["evidence_driven_steps"])
    assert trace.steps[-1].step not in trace.totals["evidence_driven_steps"]


def test_visual_review_by_the_planner_counts() -> None:
    trace, _ = run_fake("clean", register_ok=False, with_deep=False)
    assert trace.steps[-1].observation["review_trigger"] == "visual"
    assert trace.steps[-1].step in trace.totals["evidence_driven_steps"]


# ------------------------------------------------------------------ Bedrock accounting in the trace
def test_bedrock_fallback_and_usage_are_in_trace_totals() -> None:
    """Regression: a run where Bedrock was never reachable was indistinguishable from a model-planned run."""
    from tests.test_planner_bedrock import MODEL, StubBedrockClient, real_registry, throttle

    from inspectagent.agent.planner_bedrock import BedrockPlanner

    client = StubBedrockClient([throttle("AccessDeniedException", 403), throttle("AccessDeniedException", 403)])
    planner = BedrockPlanner(MODEL, client=client, registry=real_registry(), sleep=lambda s: None)
    trace, s = run_fake("clean", planner=planner)
    t = trace.totals
    assert trace.planner == "bedrock"
    assert t["n_fallback_decisions"] == len(trace.steps) and t["planner_counts"] == {"rule(fallback)": len(trace.steps)}
    assert t["bedrock"]["errors"] == 2 and t["bedrock"]["fallbacks"] == len(trace.steps) and "calls" not in t["bedrock"]
    from scripts.evaluate import agent_fields

    row = agent_fields(trace.to_dict())
    assert row["fallback_share"] == pytest.approx(1.0) and row["n_fallback_steps"] == len(trace.steps)


# ------------------------------------------------------------------ glare demo (real camera + quality)
def test_glare_demo_sample_ends_on_diffuse_light_under_rule_planner() -> None:
    """pcb1_normal_0768 under the 'glare' preset (seed 0): the glare detector misses the specular blob, the
    specular fallback (clipping-only overexposure under ring light) still switches to diffuse light."""
    img_path = ROOT / "samples" / "pcb1" / "pcb1_normal_0768.jpg"
    if not img_path.is_file():
        pytest.skip("curated sample image not available")
    from inspectagent.agent.loop import run_inspection
    from inspectagent.agent.session import default_backend
    from inspectagent.types import PartSpec
    from inspectagent.vision.camera import make_conditions
    from inspectagent.vision.imageio import imread
    from tests.agent_fakes import make_bundle

    scene = imread(img_path)
    cond = make_conditions("glare", np.random.default_rng(0))
    part = PartSpec("demo", "pcb1", str(img_path), label=0, conditions=cond)
    trace, s = run_inspection(part, make_bundle([]), scene=scene, seed=0, backend=default_backend(), session_id="glare_demo")
    full = [cid for cid in s.captures if s.is_full_view(cid)]
    assert len(full) >= 2, _tools(trace)
    first = s.quality[full[0]]
    assert "overexposed" in first.issues or "glare" in first.issues
    assert s.captures[full[-1]].settings.light == "diffuse"
    assert "glare" not in s.quality[full[-1]].issues
