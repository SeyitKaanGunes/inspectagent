"""Agent loop: planner -> policy.vet -> registry.execute -> Step, plus the single-shot baseline.

``run_inspection`` drives one part until a terminal tool (``finish`` / ``request_human_review``)
or the step budget ends, then asks the policy for the routed verdict. Timing model:

* ``compute_ms`` = real compute of tools + planner (vision, detectors, LLM calls);
* ``sim_time_ms`` = simulated physical time of the cell (motion, autofocus, light switch, exposure);
* ``cycle_time_ms`` = compute + sim. The simulator's own rendering time (``render_ms``) is reported
  separately and excluded, because a real camera produces the image in its exposure/readout time.
"""

from __future__ import annotations

import functools
import logging
import platform
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable

import cv2
import numpy as np

from inspectagent.config import Settings, get_settings
from inspectagent.models_io import CategoryBundle, load_bundle
from inspectagent.agent.planner_rule import RulePlanner
from inspectagent.agent.policy import HumanControlPolicy
from inspectagent.agent.session import InspectionSession, VisionBackend
from inspectagent.agent.tools import ToolRegistry, default_registry
from inspectagent.agent.trace import (
    EVIDENCE_KINDS,
    LOGGER_NAME,
    classify_evidence_steps,
    jsonable,
    log_event,
    log_step,
    review_trigger,
)
from inspectagent.types import CaptureSettings, Decision, PartSpec, Step, Trace

_MS_COUNTERS = ("bedrock_latency_ms",)


@functools.lru_cache(maxsize=1)
def _cv_info() -> dict:
    try:
        from inspectagent.vision.cvinfo import cv_info

        return dict(cv_info())
    except Exception:  # cvinfo not available: minimal facts
        return {"opencv_version": cv2.__version__, "arch": platform.machine(), "python": platform.python_version(), "numpy": np.__version__}


def make_planner(name: str = "rule", settings: Settings | None = None, **kwargs: Any) -> Any:
    """Planner factory: ``"rule"`` or ``"bedrock"`` (Amazon Bedrock Converse with rule fallback)."""
    settings = settings or get_settings()
    if name == "rule":
        return RulePlanner(**kwargs)
    if name == "bedrock":
        from inspectagent.agent.planner_bedrock import BedrockPlanner

        kwargs.setdefault("model_id", settings.bedrock_model_id)
        kwargs.setdefault("region", settings.aws_region)
        kwargs.setdefault("max_tokens", settings.bedrock_max_tokens)
        return BedrockPlanner(**kwargs)
    raise ValueError(f"unknown planner {name!r} (expected 'rule' or 'bedrock')")


def _planner_name(planner: Any) -> str:
    name = getattr(planner, "name", None)
    if isinstance(name, str) and name:
        return name
    return type(planner).__name__


def evidence_driven_steps(trace: Trace) -> list[int]:
    """Steps where visual evidence changed the action (shared definition: ``trace.classify_evidence_steps``)."""
    recs = classify_evidence_steps([s.to_dict() for s in trace.steps])
    return [r["step"] for r in recs if r["kind"] in EVIDENCE_KINDS]


def _planner_usage(trace: Trace, session: InspectionSession) -> dict:
    """Per-trace planner accounting: decisions per planner label, fallback decisions, Bedrock usage."""
    labels = Counter(s.decision.planner for s in trace.steps)
    bedrock = {}
    for k, v in sorted(session.counters.items()):
        if k.startswith("bedrock_") and isinstance(v, (int, float)):
            bedrock[k[len("bedrock_") :]] = round(float(v), 3) if k in _MS_COUNTERS else v
    return {
        "planner_counts": dict(sorted(labels.items())),
        "n_fallback_decisions": sum(n for lab, n in labels.items() if "fallback" in str(lab)),
        "bedrock": bedrock,
    }


def compute_totals(trace: Trace, session: InspectionSession) -> dict:
    """``Trace.totals``: counts, timing, tool histogram, final fused probability, evidence actions and
    planner accounting (decisions per planner label, rule-planner fallbacks, Bedrock calls/tokens/latency)."""
    p, ev = session.fused_prob()
    tool_counts = Counter(s.decision.tool for s in trace.steps)
    evidence_steps = evidence_driven_steps(trace)
    c = session.counters
    return jsonable(
        {
            **_planner_usage(trace, session),
            "n_steps": len(trace.steps),
            "n_captures": c["captures"],
            "n_deep": c["deep_calls"],
            "n_fast": c["fast_calls"],
            "n_zoom": c["zoom_inspections"],
            "n_corrective_captures": c["corrective_captures"],
            "n_measurements": c["measurements"],
            "n_overrides": sum(1 for s in trace.steps if s.overridden_by_policy),
            "n_errors": c["errors"],
            "compute_ms": round(session.compute_ms, 3),
            "planner_ms": round(session.planner_ms, 3),
            "tool_ms": round(session.compute_ms - session.planner_ms, 3),
            "render_ms": round(session.render_ms, 3),
            "sim_time_ms": round(session.sim_time_ms, 3),
            "cycle_time_ms": round(session.compute_ms + session.sim_time_ms, 3),
            "tool_counts": dict(sorted(tool_counts.items())),
            "final_prob": p,
            "final_evidence": ev,
            "evidence_driven_steps": evidence_steps,
            "evidence_changed_action": bool(evidence_steps),
        }
    )


def run_inspection(
    part: PartSpec,
    bundle: CategoryBundle | None = None,
    planner: Any = None,
    policy: HumanControlPolicy | None = None,
    settings: Settings | None = None,
    seed: int = 0,
    on_step: Callable[[Step, InspectionSession], None] | None = None,
    scene: np.ndarray | None = None,
    *,
    backend: VisionBackend | None = None,
    registry: ToolRegistry | None = None,
    logger: logging.Logger | None = None,
    session_id: str | None = None,
    deadline: float | None = None,
) -> tuple[Trace, InspectionSession]:
    """Run the agentic inspection of one part; returns ``(trace, session)``.

    ``deadline`` (optional ``time.monotonic()`` value) bounds the wall-clock time: it is handed to the
    policy (which forces a terminal review shortly before it) and to a planner exposing a ``deadline``
    attribute (the Bedrock planner clamps its model calls to it and falls back to the rule planner).
    """
    settings = settings or get_settings()
    if bundle is None:
        bundle = load_bundle(part.category, settings.models_dir)
    planner = planner if planner is not None else RulePlanner()
    policy = policy or HumanControlPolicy(settings.policy)
    if deadline is not None:
        if getattr(policy, "deadline", None) is None:
            policy.deadline = float(deadline)
        if hasattr(planner, "deadline") and getattr(planner, "deadline", None) is None:
            planner.deadline = float(deadline)
    registry = registry or default_registry(settings)
    logger = logger or logging.getLogger(LOGGER_NAME)

    session = InspectionSession(part, bundle, settings, seed, scene, backend=backend, session_id=session_id)
    session.policy_cfg = policy.cfg
    trace = Trace(
        trace_id=f"tr_{session.session_id}",
        part_id=part.part_id,
        category=part.category,
        planner=_planner_name(planner),
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        budget=asdict(policy.cfg),
        ground_truth={"label": part.label} if part.label is not None else None,
        conditions=part.conditions.to_dict(),
        cv_info=_cv_info(),
    )
    log_event(logger, "start", trace.trace_id, part_id=part.part_id, category=part.category, planner=trace.planner, seed=seed)

    for step_no in range(1, policy.cfg.max_steps + 1):
        state = session.planner_state()
        t0 = time.perf_counter()
        planner_failed = False
        try:
            decision = planner.next_action(state, session)
            if not isinstance(decision, Decision):
                raise TypeError(f"planner returned {type(decision).__name__}, expected Decision")
        except Exception as exc:  # planner bug / LLM failure without fallback -> human review
            planner_failed = True
            session.counters["errors"] += 1
            trace.errors.append(f"step {step_no}: planner error {type(exc).__name__}: {exc}")
            logger.warning("planner error at step %d: %s", step_no, exc)
            decision = policy.review_decision(session, step_no, f"planner error {type(exc).__name__}: {exc}", trigger="planner_error")
        dt_plan = (time.perf_counter() - t0) * 1000.0
        session.planner_ms += dt_plan
        session.compute_ms += dt_plan
        decision.step = step_no
        decision, overridden = policy.vet(decision, session)
        overridden = overridden or planner_failed

        sim0, comp0 = session.sim_time_ms, session.compute_ms
        try:
            obs = registry.execute(session, decision.tool, decision.args)
        except Exception as exc:  # defensive: execute() already converts tool errors
            session.counters["errors"] += 1
            obs = {"error": f"{type(exc).__name__}: {exc}", "tool": decision.tool, "artifacts": []}
        if "error" in obs:
            trace.errors.append(f"step {step_no}: {decision.tool}: {obs['error']}")
        if decision.tool == "request_human_review" and isinstance(obs, dict):
            # structured reason of the review: visual evidence vs budget / errors / planner failure
            obs["review_trigger"] = getattr(decision, "review_trigger", None) or (
                "visual" if decision.planner != "policy" else review_trigger({"decision": decision.to_dict(), "observation": {}})
            )
        step = Step(
            step=step_no,
            decision=decision,
            observation=obs,
            artifacts=list(obs.get("artifacts", [])),
            latency_ms=max(session.compute_ms - comp0, 0.0),
            sim_time_ms=session.sim_time_ms - sim0,
            overridden_by_policy=overridden,
        )
        trace.steps.append(step)
        log_step(logger, step, trace.trace_id)
        if on_step is not None:
            try:
                on_step(step, session)
            except Exception as exc:  # observers (UI streaming) must not break the inspection
                trace.errors.append(f"step {step_no}: on_step callback error {type(exc).__name__}: {exc}")
        if registry.is_terminal(decision.tool) and "error" not in obs:
            break

    trace.verdict = policy.final_verdict(session)
    trace.totals = compute_totals(trace, session)
    log_event(
        logger,
        "verdict",
        trace.trace_id,
        label=trace.verdict.label,
        action=trace.verdict.action,
        prob=trace.verdict.prob_anomalous,
        review_id=trace.verdict.review_id,
        n_steps=trace.totals["n_steps"],
        cycle_time_ms=trace.totals["cycle_time_ms"],
    )
    return trace, session


def run_single_shot(
    part: PartSpec,
    bundle: CategoryBundle | None,
    detector: str = "fast",
    oracle: bool = False,
    seed: int = 0,
    settings: Settings | None = None,
    *,
    scene: np.ndarray | None = None,
    backend: VisionBackend | None = None,
) -> dict:
    """Baseline without agent: one default capture (or the oracle clean capture) -> detector -> thresholds.

    ``label`` uses the policy bands (FAIL >= auto_fail_min_prob, PASS <= auto_pass_max_prob, else
    NEEDS_REVIEW); ``label_binary`` thresholds at 0.5.
    """
    settings = settings or get_settings()
    if bundle is None:
        bundle = load_bundle(part.category, settings.models_dir, with_deep=(detector == "deep"))
    if detector not in ("fast", "deep"):
        raise ValueError(f"unknown detector {detector!r}")
    if detector == "deep" and bundle.deep is None:
        raise ValueError(f"deep detector not available for category {part.category!r}")
    session = InspectionSession(part, bundle, settings, seed, scene, backend=backend)
    s = CaptureSettings()
    sim = session.estimate_capture_ms(s)
    t0 = time.perf_counter()
    if oracle:
        cap = session.camera.clean_capture(s.view, "cap_1", 0)
    else:
        cap = session.camera.capture(s, "cap_1", 0)
    render_ms = (time.perf_counter() - t0) * 1000.0
    cap.sim_time_ms = sim
    t1 = time.perf_counter()
    if detector == "deep":
        res = bundle.deep.predict(cap, id_prefix="reg_1_")
    else:
        res = bundle.stat.predict(cap, quality=None, id_prefix="reg_1_")
    latency = (time.perf_counter() - t1) * 1000.0
    pol = settings.policy
    p = float(res.prob)
    if p >= pol.auto_fail_min_prob:
        label = "FAIL"
    elif p <= pol.auto_pass_max_prob:
        label = "PASS"
    else:
        label = "NEEDS_REVIEW"
    return jsonable(
        {
            "part_id": part.part_id,
            "category": part.category,
            "detector": detector,
            "oracle": bool(oracle),
            "seed": int(seed),
            "prob": p,
            "score": float(res.score),
            "reliable": bool(res.reliable),
            "label": label,
            "label_binary": "FAIL" if p >= 0.5 else "PASS",
            "n_regions": len(res.regions),
            "regions": [r.to_dict() for r in res.regions],
            "registration": res.registration,
            "notes": list(res.notes),
            "latency_ms": latency,
            "render_ms": render_ms,
            "sim_time_ms": sim,
            "cycle_time_ms": latency + sim,
            "capture": cap.to_dict(),
        }
    )
