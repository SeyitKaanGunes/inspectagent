"""Human-control policy: guardrails, budget, and final-verdict gating.

The planner (rule-based or an LLM) only *proposes* actions. ``HumanControlPolicy.vet`` runs before
every tool call and replaces the decision with ``request_human_review`` when

* the tool is unknown,
* the budget is exhausted (steps / captures / deep calls / simulated cycle time) -- the last allowed
  step must be terminal, so every inspection ends inside the budget,
* two or more tool errors happened,
* the wall-clock deadline of the request (``deadline``, a ``time.monotonic()`` value, e.g. derived from
  the Lambda's remaining time) is within ``deadline_margin_s`` -- the inspection ends with a routed
  review instead of being killed mid-loop without a trace,
* ``finish(PASS)`` is proposed with fused P(anomalous) > ``auto_pass_max_prob``, without reliable
  full-view evidence, while *contested evidence* stands (an aligned but unreliable full-view
  detector result that flags the part with ``p >= 0.5`` and unrefuted regions, contradicting a
  reliable result on the same capture - see ``InspectionSession.contested_results``), while an
  *unlocalized zoom alarm* stands (``InspectionSession.unlocalized_zoom_alarms``), or while an
  unresolved candidate region could still be zoomed into / corroborated by deep (:func:`pass_blockers`,
  the rule planner's own PASS precondition, SPEC 3.4 step 7),
* ``finish(FAIL)`` is proposed with fused P(anomalous) < ``auto_fail_min_prob`` or without reliable
  evidence.

Every policy-made review carries a structured trigger (:data:`REVIEW_TRIGGERS`: ``visual`` when the
visual evidence did not support the proposed verdict, else ``budget``, ``errors``, ``planner_error`` or
``invalid``) as the ``review_trigger`` attribute of the decision; the loop records it in the step
observation so evaluation can tell evidence-driven reviews from budget/error stops.

``final_verdict`` maps the session outcome to a line action: PASS -> ``route_pass``; FAIL ->
``route_quarantine`` (scrapping requires a human when ``scrap_requires_human``) or ``route_scrap``;
review -> ``hold_for_review`` with the review ticket id.
"""

from __future__ import annotations

import time

from inspectagent.config import PolicyConfig
from inspectagent.agent.session import InspectionSession
from inspectagent.agent.tools import CAPTURE_TOOLS, DEFAULT_INSPECT_ZOOM, TOOL_NAMES, settings_for_call
from inspectagent.agent.trace import REVIEW_TRIGGERS
from inspectagent.types import Decision, Verdict
from inspectagent.vision.quality import blocking_quality_issues

MAX_TOOL_ERRORS = 2
CONFLICT_CONFIDENCE = 0.5  # proposed-label confidence of a review forced by contested evidence
PASS_MAX_ZOOM_INSPECTIONS = 3  # distinct candidates the agent zooms into before PASS may proceed (rule planner: 3)
DEADLINE_MARGIN_S = 4.0  # force a terminal review when the wall-clock deadline is this close


def _capture_affordable(session: InspectionSession, tool: str, args: dict) -> bool:
    """Capture and simulated-time budgets allow one more capture with these arguments."""
    pol = session.policy_cfg
    if session.counters["captures"] >= pol.max_captures or session.sim_time_ms >= pol.max_sim_time_ms:
        return False
    try:
        s = settings_for_call(session, tool, args)
        est = session.estimate_capture_ms(s) if s is not None else 0.0
    except Exception:
        return False
    return session.sim_time_ms + est <= pol.max_sim_time_ms


def pass_blockers(
    session: InspectionSession, step: int | None = None, max_inspections: int = PASS_MAX_ZOOM_INSPECTIONS
) -> tuple[list[str], str]:
    """Unresolved candidate regions that still block an automatic PASS: ``(region ids, reason)``.

    The rule planner's PASS precondition (SPEC 3.4 steps 7-8), including its exhaustion escape hatch:
    a full-view candidate that is ``uninspected``/``inconclusive`` blocks PASS while it can still be
    resolved - it was not zoomed into (nor its location), fewer than ``max_inspections`` candidates were
    inspected, and the step (this step + a terminal one), capture and simulated-time budgets allow a zoom
    capture; once the zoom candidates are exhausted, unresolved candidates still block PASS while the deep
    detector has not been run on the best full view although it could be. ``step`` is the number of the
    step being vetted (default: the next one).
    """
    unresolved = [r for r in session.full_view_regions() if session.region_status(r.region_id) in ("uninspected", "inconclusive")]
    if not unresolved:
        return [], ""
    pol = session.policy_cfg
    step = session.counters["steps"] + 1 if step is None else int(step)
    steps_ok = step + 1 <= pol.max_steps  # the zoom/deep step plus a terminal step
    n_inspected = len([r for r in session.inspections if session.inspections[r]])
    if steps_ok and n_inspected < max_inspections:
        inspectable = [
            r.region_id
            for r in unresolved
            if not session.inspections.get(r.region_id)
            and not session.location_inspected(r.region_id)
            and _capture_affordable(session, "inspect_region", {"region_id": r.region_id, "zoom": DEFAULT_INSPECT_ZOOM})
        ]
        if inspectable:
            return inspectable, f"unresolved candidate(s) {inspectable} not inspected although a zoom is still affordable"
    best = session.best_full_view()
    q = session.quality.get(best) if best is not None else None
    deep_possible = (
        best is not None
        and session.deep_available
        and steps_ok
        and session.counters["deep_calls"] < pol.max_deep_calls
        and session.counters["errors"] < MAX_TOOL_ERRORS
        and not (q is not None and not q.ok and blocking_quality_issues(q))
        and not session.result_ids(detector="deep", capture_id=best)
    )
    if deep_possible:
        ids = [r.region_id for r in unresolved]
        return ids, f"unresolved candidate(s) {ids} and the deep detector was not run on full view {best}"
    return [], ""


class HumanControlPolicy:
    """Guardrails between planner and tools, and the final routing decision."""

    def __init__(
        self,
        cfg: PolicyConfig | None = None,
        known_tools: tuple[str, ...] | None = None,
        deadline: float | None = None,
        deadline_margin_s: float = DEADLINE_MARGIN_S,
    ) -> None:
        self.cfg = cfg or PolicyConfig()
        self.known_tools = tuple(known_tools or TOOL_NAMES)
        self.deadline = deadline  # time.monotonic() value; None = no wall-clock limit
        self.deadline_margin_s = float(deadline_margin_s)

    # ------------------------------------------------------------------ helpers
    def _proposed_from_prob(self, p: float) -> tuple[str, float]:
        label = "FAIL" if p >= 0.5 else "PASS"
        return label, (p if label == "FAIL" else 1.0 - p)

    def seconds_left(self) -> float | None:
        """Wall-clock seconds until the deadline (None without a deadline)."""
        return None if self.deadline is None else float(self.deadline) - time.monotonic()

    def review_decision(
        self,
        session: InspectionSession,
        step: int,
        reason: str,
        original: Decision | None = None,
        proposed_label: str | None = None,
        confidence: float | None = None,
        trigger: str = "visual",
    ) -> Decision:
        """A policy-made ``request_human_review`` decision quoting the fused evidence.

        ``trigger`` (one of :data:`REVIEW_TRIGGERS`) is attached as ``decision.review_trigger``.
        """
        rep = session.fusion_report()
        p = rep["prob"]
        auto_label, auto_conf = self._proposed_from_prob(p)
        label = proposed_label if proposed_label in ("PASS", "FAIL") else auto_label
        conf = float(confidence) if confidence is not None else (p if label == "FAIL" else 1.0 - p)
        conf = min(max(conf, 0.0), 1.0)
        replaced = ""
        if original is not None:
            replaced = f"; replaced planner decision {original.tool}({_short_args(original.args)})"
        rationale = f"policy: {reason} -> human review [{rep['explain']}]{replaced}"
        evidence = list(dict.fromkeys(rep["evidence"] + (original.evidence if original else [])))
        decision = Decision(
            step=step,
            tool="request_human_review",
            args={"reason": rationale[:2000], "proposed_label": label, "confidence": round(conf, 4)},
            rationale=rationale,
            evidence=evidence,
            planner="policy",
        )
        decision.review_trigger = trigger if trigger in REVIEW_TRIGGERS else "invalid"  # type: ignore[attr-defined]
        return decision

    # ------------------------------------------------------------------ vetting
    def vet(self, decision: Decision, session: InspectionSession) -> tuple[Decision, bool]:
        """Return ``(decision or its replacement, overridden?)``."""
        cfg = self.cfg
        step = max(int(decision.step), session.counters["steps"] + 1)
        tool = decision.tool
        args = decision.args if isinstance(decision.args, dict) else {}

        if tool not in self.known_tools:
            return self.review_decision(session, step, f"unknown tool {tool!r}", decision, trigger="invalid"), True
        if tool == "request_human_review":
            return decision, False

        p, _ = session.fused_prob()
        if tool == "finish":
            label = args.get("label")
            if label == "PASS":
                blocked = self._pass_block_reason(session, p, step)
                if blocked is not None:
                    reason, conf = blocked
                    return self.review_decision(session, step, f"finish(PASS) blocked: {reason}", decision, "PASS", conf), True
                return decision, False
            if label == "FAIL":
                if cfg.require_reliable_evidence and not session.has_reliable_evidence("FAIL"):
                    return self.review_decision(session, step, "finish(FAIL) without reliable evidence", decision, "FAIL"), True
                if p < cfg.auto_fail_min_prob:
                    return (
                        self.review_decision(
                            session, step, f"finish(FAIL) blocked: fused p={p:.3f}<auto_fail_min_prob={cfg.auto_fail_min_prob:.2f}", decision, "FAIL"
                        ),
                        True,
                    )
                return decision, False
            return self.review_decision(session, step, f"finish with invalid label {label!r}", decision, trigger="invalid"), True

        # ---- non-terminal tools: errors and budget
        if session.counters["errors"] >= MAX_TOOL_ERRORS:
            return (
                self.review_decision(
                    session, step, f"{session.counters['errors']} tool errors (limit {MAX_TOOL_ERRORS})", decision, trigger="errors"
                ),
                True,
            )
        left = self.seconds_left()
        if left is not None and left <= self.deadline_margin_s:
            return (
                self.review_decision(
                    session, step, f"wall-clock budget: {max(left, 0.0):.1f}s left (<= {self.deadline_margin_s:.1f}s margin)", decision,
                    trigger="budget",
                ),
                True,
            )
        if step >= cfg.max_steps:
            return (
                self.review_decision(session, step, f"step budget: step {step}/{cfg.max_steps} must be terminal", decision, trigger="budget"),
                True,
            )
        if session.sim_time_ms >= cfg.max_sim_time_ms:
            return (
                self.review_decision(
                    session, step, f"sim-time budget exhausted: {session.sim_time_ms:.0f}>={cfg.max_sim_time_ms:.0f} ms", decision,
                    trigger="budget",
                ),
                True,
            )
        if tool in CAPTURE_TOOLS:
            n = session.counters["captures"]
            if n >= cfg.max_captures:
                return (
                    self.review_decision(session, step, f"capture budget exhausted: {n}/{cfg.max_captures}", decision, trigger="budget"),
                    True,
                )
            try:
                s = settings_for_call(session, tool, args)
                est = session.estimate_capture_ms(s) if s is not None else 0.0
            except Exception:  # invalid args are reported by the registry
                est = 0.0
            if session.sim_time_ms + est > cfg.max_sim_time_ms:
                return (
                    self.review_decision(
                        session,
                        step,
                        f"sim-time budget: {session.sim_time_ms:.0f}+{est:.0f}>{cfg.max_sim_time_ms:.0f} ms",
                        decision,
                        trigger="budget",
                    ),
                    True,
                )
        if tool == "detect_anomalies" and args.get("detector") == "deep":
            n = session.counters["deep_calls"]
            if n >= cfg.max_deep_calls:
                return (
                    self.review_decision(session, step, f"deep-call budget exhausted: {n}/{cfg.max_deep_calls}", decision, trigger="budget"),
                    True,
                )
        return decision, False

    def _pass_block_reason(self, session: InspectionSession, p: float, step: int) -> tuple[str, float | None] | None:
        """Why an automatic PASS is not allowed now: ``(reason, review confidence or None)``; None = allowed."""
        cfg = self.cfg
        if cfg.require_reliable_evidence and not session.has_reliable_evidence("PASS"):
            return "without reliable full-view evidence", None
        if p > cfg.auto_pass_max_prob:
            return f"fused p={p:.3f}>auto_pass_max_prob={cfg.auto_pass_max_prob:.2f}", None
        contested = session.contested_results()
        if contested:
            return f"conflicting evidence - {session.contested_note(contested)}", CONFLICT_CONFIDENCE
        alarms = session.unlocalized_zoom_alarms()
        if alarms:
            ps = ", ".join(f"{a} p={float(session.results[a].prob):.3f}" for a in alarms)
            return f"unlocalized zoom alarm(s) {ps}: a zoom saw an anomaly away from the inspected candidate", CONFLICT_CONFIDENCE
        blockers, why = pass_blockers(session, step=step)
        if blockers:
            return f"{why} -> inspect_region({blockers[0]}) first", CONFLICT_CONFIDENCE
        return None

    # ------------------------------------------------------------------ verdict
    def _gate(self, session: InspectionSession, label: str, p: float) -> str | None:
        cfg = self.cfg
        if label == "PASS":
            blocked = self._pass_block_reason(session, p, session.counters["steps"])
            if blocked is not None:
                return f"PASS blocked: {blocked[0]}"
        elif label == "FAIL":
            if cfg.require_reliable_evidence and not session.has_reliable_evidence("FAIL"):
                return "FAIL without reliable evidence"
            if p < cfg.auto_fail_min_prob:
                return f"FAIL with fused p={p:.3f}<{cfg.auto_fail_min_prob:.2f}"
        else:
            return f"invalid label {label!r}"
        return None

    def final_verdict(self, session: InspectionSession) -> Verdict:
        """Route the part: PASS -> route_pass, FAIL -> route_quarantine/route_scrap, else hold_for_review."""
        cfg = self.cfg
        rep = session.fusion_report()
        p = rep["prob"]
        if session.review is None and session.proposed is not None:
            label = session.proposed.get("label")
            blocked = self._gate(session, label, p)
            if blocked is None:
                if label == "PASS":
                    return Verdict(
                        label="PASS",
                        confidence=1.0 - p,
                        rationale=(
                            f"{session.proposed.get('rationale', '')} | policy: fused p={p:.3f}<={cfg.auto_pass_max_prob:.2f} "
                            f"with reliable evidence -> auto PASS -> route_pass"
                        ),
                        defect_regions=[],
                        action="route_pass",
                        review_id=None,
                        prob_anomalous=p,
                    )
                action = "route_quarantine" if cfg.scrap_requires_human else "route_scrap"
                note = " (scrap requires a human decision)" if cfg.scrap_requires_human else ""
                return Verdict(
                    label="FAIL",
                    confidence=p,
                    rationale=(
                        f"{session.proposed.get('rationale', '')} | policy: fused p={p:.3f}>={cfg.auto_fail_min_prob:.2f} "
                        f"with reliable evidence -> auto FAIL -> {action}{note}"
                    ),
                    defect_regions=session.defect_regions(session.proposed.get("region_ids")),
                    action=action,
                    review_id=None,
                    prob_anomalous=p,
                )
            conf = float(session.proposed.get("confidence", 0.5))
            session.open_review(
                f"policy (final gate): {blocked} [{rep['explain']}]; planner proposed {label}", label if label in ("PASS", "FAIL") else "FAIL", conf, rep["evidence"], source="policy"
            )
        if session.review is None:
            label, conf = self._proposed_from_prob(p)
            session.open_review(
                f"inspection ended without a terminal decision (budget or errors) [{rep['explain']}]", label, conf, rep["evidence"], source="policy"
            )
        r = session.review
        assert r is not None
        regions = session.defect_regions(session.proposed.get("region_ids") if session.proposed else None) if r.get("proposed_label") == "FAIL" else []
        return Verdict(
            label="NEEDS_REVIEW",
            confidence=float(r.get("confidence", 0.5)),
            rationale=str(r.get("reason", "")),
            defect_regions=regions,
            action="hold_for_review",
            review_id=r.get("review_id"),
            prob_anomalous=p,
        )


def _short_args(args: dict) -> str:
    parts = []
    for k, v in (args or {}).items():
        s = f"{v:.3g}" if isinstance(v, float) else str(v)
        if len(s) > 40:
            s = s[:37] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)
