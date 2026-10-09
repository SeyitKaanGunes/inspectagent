"""RulePlanner (SPEC 3.4): visual evidence drives the next action; rationales quote ids, numbers, thresholds."""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from inspectagent.agent.planner_rule import GLARE_EV_COMP, RulePlanner
from inspectagent.agent.policy import HumanControlPolicy
from inspectagent.agent.tools import TERMINAL_TOOLS, default_registry
from inspectagent.config import PolicyConfig, Settings
from inspectagent.types import QualityReport
from tests.agent_fakes import DEFECT, FALSE_ALARM, FakeVision, Spot, make_session, run_fake

REG = default_registry()
ID_RE = re.compile(r"\b(?:cap|q|det|reg|art)_\d+(?:_\d+)?\b")


def _next(s):
    return RulePlanner().next_action(s.planner_state(), s)


def _tools(trace):
    return [st.decision.tool for st in trace.steps]


def _corrective(trace):
    return [st for st in trace.steps if st.decision.tool == "capture_image" and st.decision.evidence]


def test_first_steps_capture_then_assess():
    s, _ = make_session()
    d = _next(s)
    assert d.tool == "capture_image" and d.planner == "rule" and d.step == 1
    assert d.args == {"cx": 0.5, "cy": 0.5, "zoom": 1.0, "exposure_ev": 0.0, "light": "ring", "autofocus": False}
    REG.execute(s, d.tool, d.args)
    d = _next(s)
    assert d.tool == "assess_quality" and d.args == {"capture_id": "cap_1"} and d.evidence == ["cap_1"]


def test_glare_switches_to_diffuse_and_compensates_ev():
    trace, s = run_fake("glare")
    fix = _corrective(trace)
    assert len(fix) == 1
    d = fix[0].decision
    assert d.args["light"] == "diffuse" and d.args["exposure_ev"] == pytest.approx(GLARE_EV_COMP)
    assert "q_1" in d.evidence and "cap_1" in d.evidence
    assert "glare_frac=0.0340>0.0040" in d.rationale and "diffuse" in d.rationale
    assert trace.verdict.label == "PASS"
    # the detector ran on the corrected capture, not the glary one
    det = next(st for st in trace.steps if st.decision.tool == "detect_anomalies")
    assert det.decision.args["capture_id"] == "cap_2"


def test_defocus_turns_on_autofocus():
    trace, _ = run_fake("defocus")
    d = _corrective(trace)[0].decision
    assert d.args["autofocus"] is True and d.args["light"] == "ring" and d.args["exposure_ev"] == 0.0
    assert re.search(r"sharpness_ratio=0\.\d+<0\.55 -> autofocus on", d.rationale)


def test_dark_raises_exposure_by_suggested_delta():
    trace, s = run_fake("dark")
    d = _corrective(trace)[0].decision
    q1 = s.quality["cap_1"]
    assert "underexposed" in q1.issues
    assert d.args["exposure_ev"] == pytest.approx(q1.suggested_exposure_delta, abs=1e-3) and d.args["exposure_ev"] > 2.0
    assert f"mean_luma={q1.mean_luma:.1f}" in d.rationale and "[60,190]" in d.rationale
    assert s.quality["cap_2"].ok and trace.verdict.label == "PASS"


def test_bright_lowers_exposure():
    trace, s = run_fake("bright")
    q1 = s.quality["cap_1"]
    assert "overexposed" in q1.issues and q1.mean_luma > 190
    d = _corrective(trace)[0].decision
    assert d.args["light"] == "ring"  # global overexposure (mean too high) is not mistaken for specular
    assert d.args["exposure_ev"] == pytest.approx(q1.suggested_exposure_delta, abs=1e-3) and d.args["exposure_ev"] < 0
    assert s.quality["cap_2"].ok and trace.verdict.label == "PASS"


def test_combined_fixes_in_one_recapture():
    trace, s = run_fake("harsh")
    fix = _corrective(trace)
    d = fix[0].decision
    assert d.args["light"] == "diffuse" and d.args["autofocus"] is True
    assert d.args["exposure_ev"] >= GLARE_EV_COMP  # glare compensation (+ dark ambient correction)
    assert "glare_frac" in d.rationale and "autofocus on" in d.rationale
    assert len(fix) <= 3 and trace.verdict.label in ("PASS", "NEEDS_REVIEW")


def test_noisy_high_gain_switches_to_ring_when_no_glare_seen():
    s, vision = make_session("clean")
    vision.camera.conditions.noise_sigma = 2.0
    vision.camera.conditions.ambient_ev = -1.3
    REG.execute(s, "capture_image", {"light": "diffuse", "exposure_ev": 2.0})
    REG.execute(s, "assess_quality", {"capture_id": "cap_1"})
    assert s.quality["cap_1"].issues == ["noisy"]
    d = _next(s)
    assert d.tool == "capture_image" and d.args["light"] == "ring"
    assert d.args["exposure_ev"] == pytest.approx(2.0 - math.log2(1 / 0.6), abs=1e-3)
    assert "noise_sigma=8.00>7.00" in d.rationale


def test_noise_at_low_gain_is_accepted_then_deep():
    s2, vision = make_session("clean")
    vision.camera.conditions.noise_sigma = 9.0  # sensor noise, not gain-induced
    pol = HumanControlPolicy()
    planner = RulePlanner()
    tools = []
    for i in range(1, 15):
        d = planner.next_action(s2.planner_state(), s2)
        d.step = i
        d, _ = pol.vet(d, s2)
        REG.execute(s2, d.tool, d.args)
        tools.append(d.tool)
        if d.tool in TERMINAL_TOOLS:
            break
    assert tools[:3] == ["capture_image", "assess_quality", "detect_anomalies"]  # no pointless recapture
    det_step = 3
    assert tools[det_step] == "detect_anomalies" and s2.result_info["det_2"].detector == "deep"
    assert tools[-1] == "finish" and s2.proposed["label"] == "PASS"


class _StubbornVision(FakeVision):
    """Quality that never becomes acceptable (exposure keeps drifting)."""

    def assess_quality(self, capture, reference_sharpness=None, cfg=None) -> QualityReport:
        q = super().assess_quality(capture, reference_sharpness, cfg)
        q.issues = ["underexposed"]
        q.ok = False
        q.mean_luma = 40.0
        q.suggested_exposure_delta = 0.4
        return q


def test_at_most_three_corrective_recaptures():
    trace, s = run_fake("clean", vision=_StubbornVision())
    fix = _corrective(trace)
    assert len(fix) == 3
    evs = [st.decision.args["exposure_ev"] for st in fix]
    assert evs == pytest.approx([0.4, 0.8, 1.2])
    assert "corrective recapture 3/3" in fix[-1].decision.rationale
    det = next(st for st in trace.steps if st.decision.tool == "detect_anomalies" and st.decision.args["detector"] == "fast")
    assert "accepted" in det.decision.rationale
    # fast is unreliable on bad quality; a deep result on the same (still underexposed) capture would be
    # unreliable too (vision.quality.blocking_quality_issues) -> no wasted deep call, human review
    assert not any(st.decision.args.get("detector") == "deep" for st in trace.steps if st.decision.tool == "detect_anomalies")
    last = trace.steps[-1].decision
    assert last.tool == "request_human_review" and "blocking quality issues ['underexposed']" in last.rationale
    assert trace.verdict.label == "NEEDS_REVIEW"
    assert trace.totals["n_steps"] <= 14


def test_ambiguous_candidate_confirmed_then_measured_then_fail():
    trace, s = run_fake("clean", spots=[DEFECT])
    assert _tools(trace) == ["capture_image", "assess_quality", "detect_anomalies", "inspect_region", "measure_defect", "finish"]
    insp = trace.steps[3].decision
    assert insp.args == {"region_id": "reg_1_1", "zoom": 3.0}
    assert "fused p=0.620" in insp.rationale and "(0.15, 0.90)" in insp.rationale and "reg_1_1" in insp.evidence
    meas = trace.steps[4].decision
    assert "det_2" in meas.rationale and "zoom p=0.970>=0.60" in meas.rationale
    fin = trace.steps[5].decision
    assert fin.args["label"] == "FAIL" and fin.args["region_ids"] == ["reg_1_1"]
    assert "area=" in fin.rationale and ">=0.90" in fin.rationale
    assert trace.verdict.label == "FAIL" and trace.verdict.action == "route_quarantine"


def test_false_alarm_refuted_then_pass():
    trace, s = run_fake("clean", spots=[FALSE_ALARM])
    assert _tools(trace) == ["capture_image", "assess_quality", "detect_anomalies", "inspect_region", "finish"]
    fin = trace.steps[-1].decision
    assert fin.args["label"] == "PASS" and "refuted share by det_2" in fin.rationale
    assert trace.verdict.label == "PASS" and trace.verdict.prob_anomalous == pytest.approx(0.11)


def test_strong_evidence_zooms_once_below_098_and_not_above():
    trace, _ = run_fake("clean", spots=[Spot((380.0, 240.0, 24.0, 16.0), 0.93, 0.97)])
    assert _tools(trace).count("inspect_region") == 1 and trace.verdict.label == "FAIL"
    assert "but <0.98" in trace.steps[3].decision.rationale
    trace, _ = run_fake("clean", spots=[Spot((380.0, 240.0, 24.0, 16.0), 0.99, 0.97)])
    assert "inspect_region" not in _tools(trace)
    assert _tools(trace)[-2:] == ["measure_defect", "finish"] and trace.verdict.label == "FAIL"


def test_strong_evidence_refuted_by_zoom_escalates_to_deep():
    spot = Spot((100.0, 80.0, 20.0, 20.0), p_full=0.95, p_zoom=0.04, p_deep=0.05)
    trace, s = run_fake("clean", spots=[spot])
    tools = _tools(trace)
    assert tools[3] == "inspect_region" and tools[4] == "detect_anomalies"
    assert trace.steps[4].decision.args["detector"] == "deep"
    assert "refuted share" in trace.steps[4].decision.rationale
    # deep agrees with the refutation -> fused p = max(0.95*0.2, 0.05) = 0.19 > 0.15 -> human review
    assert trace.verdict.label == "NEEDS_REVIEW" and trace.verdict.prob_anomalous == pytest.approx(0.19)


def test_registration_failure_escalates_to_deep():
    trace, _ = run_fake("clean", register_ok=False)
    d = trace.steps[3].decision
    assert d.tool == "detect_anomalies" and d.args["detector"] == "deep"
    assert "registration ok=False" in d.rationale and "ncc=0.12 (min 0.45)" in d.rationale and "det_1" in d.evidence
    assert trace.verdict.label == "PASS"


def test_registration_failure_without_deep_asks_human():
    trace, _ = run_fake("clean", register_ok=False, with_deep=False)
    d = trace.steps[-1].decision
    assert d.tool == "request_human_review" and d.planner == "rule"
    assert "deep detector not available" in d.rationale
    assert trace.verdict.action == "hold_for_review"


def test_inconclusive_zoom_with_glare_is_retried_with_diffuse_light():
    trace, s = run_fake("glare", spots=[FALSE_ALARM], vision=FakeVision(glare_min_zoom=2.0))
    insp = [st for st in trace.steps if st.decision.tool == "inspect_region"]
    assert len(insp) == 2
    assert insp[0].observation["status"] == "inconclusive" and "glare" in insp[0].observation["quality"]["issues"]
    retry = insp[1].decision
    assert retry.args["light"] == "diffuse" and retry.args["exposure_ev"] == pytest.approx(GLARE_EV_COMP)
    assert "inconclusive" in retry.rationale and "glare_frac" in retry.rationale
    assert insp[1].observation["status"] == "refuted" and trace.verdict.label == "PASS"


def test_top_k_inspections_then_deep_then_review():
    spots = [
        Spot((100.0, 80.0, 20.0, 20.0), 0.50, 0.04, p_deep=0.04),
        Spot((300.0, 200.0, 20.0, 20.0), 0.45, 0.05, p_deep=0.04),
        Spot((480.0, 300.0, 20.0, 20.0), 0.40, 0.03, p_deep=0.04),
        Spot((500.0, 40.0, 20.0, 20.0), 0.35, 0.03, p_deep=0.04),
    ]
    trace, s = run_fake("clean", spots=spots)
    tools = _tools(trace)
    assert tools.count("inspect_region") == 3  # top-k <= 3
    inspected = [st.decision.args["region_id"] for st in trace.steps if st.decision.tool == "inspect_region"]
    assert inspected == ["reg_1_1", "reg_1_2", "reg_1_3"]  # ranked by peak*sqrt(area)
    assert tools[-2] == "detect_anomalies" and trace.steps[-2].decision.args["detector"] == "deep"
    assert tools[-1] == "request_human_review"
    assert "unresolved ['reg_1_4']" in trace.steps[-1].decision.rationale


def test_same_location_is_not_zoomed_twice():
    spot = Spot((380.0, 240.0, 24.0, 16.0), p_full=0.5, p_zoom=0.45, p_deep=0.5)  # zoom stays mid-range
    trace, _ = run_fake("clean", spots=[spot])
    assert _tools(trace).count("inspect_region") == 1  # deep region at the same place is not re-zoomed
    assert trace.verdict.label == "NEEDS_REVIEW"


def test_budget_aware_short_run_still_terminates_by_planner():
    settings = Settings(policy=PolicyConfig(max_steps=5))
    trace, _ = run_fake("glare", settings=settings, policy=HumanControlPolicy(settings.policy))
    assert len(trace.steps) <= 5
    assert trace.steps[-1].decision.tool in TERMINAL_TOOLS
    assert not any(st.overridden_by_policy for st in trace.steps)


def test_every_rationale_quotes_evidence():
    for kw in ({"spots": [DEFECT]}, {"spots": [FALSE_ALARM]}, {"register_ok": False}):
        for preset in ("glare", "dark", "defocus"):
            trace, s = run_fake(preset, **kw)
            for st in trace.steps[1:]:
                d = st.decision
                assert d.evidence, (preset, d.tool, d.rationale)
                assert any(e in d.rationale for e in d.evidence), (preset, d.tool, d.rationale)
                assert re.search(r"\d\.\d", d.rationale), d.rationale  # numbers are quoted
                assert set(d.evidence) <= set(ID_RE.findall(" ".join(d.evidence)))


@pytest.mark.parametrize("seed", range(12))
def test_never_loops_on_random_worlds(seed):
    """Random detector worlds + generous budget: the planner itself must end the inspection."""
    rng = np.random.default_rng(seed)
    spots = []
    for _ in range(int(rng.integers(0, 6))):
        x, y = float(rng.uniform(0, 560)), float(rng.uniform(0, 370))
        spots.append(Spot((x, y, 20.0, 20.0), float(rng.uniform(0, 1)), float(rng.uniform(0, 1)), p_deep=float(rng.uniform(0, 1))))
    preset = ("clean", "glare", "dark", "defocus", "harsh")[seed % 5]
    big = PolicyConfig(max_steps=80, max_captures=60, max_deep_calls=10, max_sim_time_ms=1e9)
    settings = Settings(policy=big)
    trace, s = run_fake(
        preset,
        spots=spots,
        settings=settings,
        policy=HumanControlPolicy(big),
        register_ok=bool(rng.random() < 0.8),
        zoom_register_ok=bool(rng.random() < 0.8),
        with_deep=bool(rng.random() < 0.8),
    )
    assert len(trace.steps) <= 25, _tools(trace)
    last = trace.steps[-1]
    assert last.decision.tool in TERMINAL_TOOLS and not last.overridden_by_policy
    # no identical non-terminal action is ever repeated
    keys = [(st.decision.tool, tuple(sorted(st.decision.args.items()))) for st in trace.steps if st.decision.tool not in TERMINAL_TOOLS]
    assert len(keys) == len(set(keys)), keys
    # a defect the fast detector sees strongly at zoom is never passed
    if trace.verdict.label == "PASS":
        assert s.fused_prob()[0] <= 0.15


class _ClipVision(FakeVision):
    """Localized clipping with an in-range mean (e.g. a specular blob the glare detector missed)."""

    def assess_quality(self, capture, reference_sharpness=None, cfg=None) -> QualityReport:
        q = super().assess_quality(capture, reference_sharpness, cfg)
        if capture.settings.light == "ring":
            q.issues = sorted(set(q.issues) | {"overexposed"})
            q.clipped_frac = 0.158
            q.suggested_exposure_delta = -2.0
            q.ok = False
        return q


def test_clipping_only_overexposure_at_nominal_gain_is_treated_as_specular():
    trace, s = run_fake("clean", vision=_ClipVision())
    d = _corrective(trace)[0].decision
    assert d.args["light"] == "diffuse" and d.args["exposure_ev"] == pytest.approx(GLARE_EV_COMP)
    assert "clipped_frac=0.158>0.020" in d.rationale and "specular" in d.rationale
    assert len(_corrective(trace)) == 1 and trace.verdict.label == "PASS"


def test_clipping_after_raising_gain_uses_suggested_delta():
    s, vision = make_session("clean", vision=_ClipVision())
    REG.execute(s, "capture_image", {"exposure_ev": 2.0})
    REG.execute(s, "assess_quality", {"capture_id": "cap_1"})
    d = _next(s)
    assert d.tool == "capture_image" and d.args["light"] == "ring"
    assert d.args["exposure_ev"] == pytest.approx(0.0)  # 2.0 + suggested -2.0
    assert "specular" not in d.rationale and "EV -2.00 (suggested)" in d.rationale
