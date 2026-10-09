"""RulePlanner: deterministic active-perception policy (SPEC section 3.4).

Every decision is derived from visual evidence already in the session and its rationale quotes the
evidence ids, the measured numbers and the thresholds that triggered it, e.g.
``"q_1 (cap_1): glare_frac=0.0310>0.0040 -> light ring->diffuse, EV +0.75"``.

Protocol (evaluated top-down at every step):

1. no capture -> default full view (cx=cy=0.5, zoom 1, EV 0, ring, AF off);
2. newest capture not quality-assessed -> ``assess_quality``;
3. full-view quality issues (<= 3 corrective recaptures, all fixes combined in one recapture):
   glare -> diffuse light + EV +0.75; blurred -> autofocus; under/over-exposed -> EV +=
   suggested delta; noisy at EV > 1 -> ring light (if no glare was ever seen) with EV compensation;
   refinement: clipping-only overexposure under ring light at nominal gain (EV <= +0.5, mean luma
   in range, glare not segmented) is treated as a specular reflection -> diffuse light + EV +0.75 first;
   unfixable / repeated settings / out of budget -> accept;
4. best full view not analysed -> ``detect_anomalies(fast)``;
5. no reliable evidence (registration failed / bad quality) -> ``detect_anomalies(deep)`` if
   available and in budget and the capture has no blocking quality issue (a deep result on it would
   be unreliable as well, ``vision.quality.blocking_quality_issues``), else review;
6. fused p >= auto_fail with a region -> zoom once to confirm when p < 0.98, measure, ``finish(FAIL)``;
7. fused p <= auto_pass, no unresolved candidate region and no *contested* alarm (an aligned but
   unreliable full-view result that flags the part against a reliable result on the same capture,
   see ``InspectionSession.contested_results``) -> ``finish(PASS)``;
8. ambiguous or contested -> zoom into up to 3 un-inspected candidates (regions of eligible results
   first, then contested regions; retry an inconclusive one once when its zoom view had a fixable
   quality issue); a confirmation -> measure -> FAIL; a refutation of every contested region clears
   the conflict; when candidates are exhausted -> deep on the full view -> decide by thresholds ->
   otherwise ``request_human_review``.

Termination: each non-terminal action consumes a bounded resource (captures, inspections per
region, distinct inspected regions, deep calls, one measurement per region, settings never
repeated) and the planner switches to a terminal decision when the step budget is about to run out.
"""

from __future__ import annotations

import math

from inspectagent.agent.policy import pass_blockers
from inspectagent.agent.session import ZOOM_CONFIRM_PROB, InspectionSession
from inspectagent.agent.tools import DEFAULT_INSPECT_ZOOM, settings_for_call
from inspectagent.types import CaptureSettings, Decision, QualityReport, Region
from inspectagent.vision.quality import blocking_quality_issues

# Planner knowledge of the cell hardware (light gains match the camera simulator / real ring light).
LIGHT_GAIN = {"ring": 1.0, "diffuse": 0.6, "low_angle": 0.5}
GLARE_EV_COMP = 0.75  # ~log2(1/0.6): compensates the dimmer diffuse light
GLARE_PERSIST_EV = -0.5  # glare that survives a non-ring light: lower exposure to limit clipping
NOISY_EV_LIMIT = 1.0  # noise is attributed to digital gain above this EV
SPECULAR_MAX_EV = 0.5  # clipping-only overexposure is read as specular only at (near) nominal gain
STRONG_FAIL_PROB = 0.98  # no confirmation zoom needed above this fused probability
EXPOSURE_BRACKET_MIN_EV = 0.25  # under/over bracket narrower than this: accept the exposure (no further bisection)
DEFAULT_ARGS = {"cx": 0.5, "cy": 0.5, "zoom": 1.0, "exposure_ev": 0.0, "light": "ring", "autofocus": False}


def _fmt_settings(s: CaptureSettings) -> str:
    return f"EV {s.exposure_ev:+.2f}, {s.light}, AF {'on' if s.autofocus else 'off'}"


def _settings_key(s: CaptureSettings) -> tuple:
    return (round(s.view.cx, 3), round(s.view.cy, 3), round(s.view.zoom, 2), round(s.exposure_ev, 2), s.light, bool(s.autofocus))


class RulePlanner:
    """Deterministic planner implementing the SPEC 3.4 protocol."""

    name = "rule"

    def __init__(
        self,
        max_corrective: int = 3,
        max_inspections: int = 3,
        inspect_zoom: float = DEFAULT_INSPECT_ZOOM,
        max_attempts_per_region: int = 2,
        strong_fail_prob: float = STRONG_FAIL_PROB,
    ) -> None:
        self.max_corrective = int(max_corrective)
        self.max_inspections = int(max_inspections)
        self.inspect_zoom = float(inspect_zoom)
        self.max_attempts_per_region = int(max_attempts_per_region)
        self.strong_fail_prob = float(strong_fail_prob)

    # ------------------------------------------------------------------ public API
    def next_action(self, state: dict, session: InspectionSession) -> Decision:
        """Next tool call for ``session`` (``state`` is its ``planner_state()``; the session is authoritative)."""
        d = self._decide(session)
        d.step = session.counters["steps"] + 1
        d.planner = self.name
        return d

    # ------------------------------------------------------------------ decision helpers
    @staticmethod
    def _d(tool: str, args: dict, rationale: str, evidence: list[str]) -> Decision:
        return Decision(step=0, tool=tool, args=args, rationale=rationale, evidence=list(dict.fromkeys(evidence)), planner="rule")

    def _steps_left(self, session: InspectionSession) -> int:
        return session.policy_cfg.max_steps - session.counters["steps"]

    def _can_capture(self, session: InspectionSession, s: CaptureSettings | None, steps_needed: int) -> bool:
        pol = session.policy_cfg
        if self._steps_left(session) < steps_needed:
            return False
        if session.counters["captures"] >= pol.max_captures:
            return False
        if session.sim_time_ms >= pol.max_sim_time_ms:
            return False
        if s is not None:
            try:
                est = session.estimate_capture_ms(s)
            except Exception:
                est = 0.0
            if session.sim_time_ms + est > pol.max_sim_time_ms:
                return False
        return True

    @staticmethod
    def _blocking_issues(session: InspectionSession, capture_id: str | None) -> list[str]:
        """Quality issues of a capture that make any detector result on it unreliable."""
        q = session.quality.get(capture_id) if capture_id is not None else None
        return blocking_quality_issues(q) if q is not None and not q.ok else []

    def _deep_possible(self, session: InspectionSession, capture_id: str | None) -> bool:
        pol = session.policy_cfg
        if capture_id is None or not session.deep_available:
            return False
        if session.counters["deep_calls"] >= pol.max_deep_calls or self._steps_left(session) < 2:
            return False
        if session.counters["errors"] >= 2:
            return False
        if self._blocking_issues(session, capture_id):
            return False  # the deep result would be unreliable too (glare/clipping/blur shift its features)
        return not session.result_ids(detector="deep", capture_id=capture_id)

    def _deep_note(self, session: InspectionSession) -> str:
        pol = session.policy_cfg
        if not session.deep_available:
            return "deep detector not available"
        if session.counters["deep_calls"] >= pol.max_deep_calls:
            return f"deep budget exhausted ({session.counters['deep_calls']}/{pol.max_deep_calls})"
        if self._steps_left(session) < 2:
            return "no step budget left for deep"
        best = session.best_full_view()
        blocking = self._blocking_issues(session, best)
        if blocking:
            return f"deep skipped: best full view {best} has blocking quality issues {blocking} (a deep result would be unreliable)"
        return "deep already run on the best full view"

    def _review(self, session: InspectionSession, reason: str, evidence: list[str]) -> Decision:
        rep = session.fusion_report()
        p = rep["prob"]
        label = "FAIL" if p >= 0.5 else "PASS"
        conf = p if label == "FAIL" else 1.0 - p
        if label == "PASS" and session.contested_results():
            conf = min(conf, 0.5)  # a contested PASS is a coin flip until a human looks at it
        text = f"{reason}; {rep['explain']}"
        return self._d(
            "request_human_review",
            {"reason": text[:2000], "proposed_label": label, "confidence": round(conf, 4)},
            f"{text} -> request_human_review (proposed {label}, confidence {conf:.2f})",
            evidence + rep["evidence"],
        )

    # ------------------------------------------------------------------ quality fixes
    def _fix(self, session: InspectionSession, q: QualityReport, qid: str, cid: str, s: CaptureSettings) -> tuple[CaptureSettings, list[str], list[str]]:
        """(new settings, applied fixes, accepted-issue notes) derived from a quality report."""
        qc = session.settings.quality
        light, ev, af = s.light, float(s.exposure_ev), bool(s.autofocus)
        fixes: list[str] = []
        accepted: list[str] = []
        delta = 0.0
        switched_off_ring = False
        glare_seen = self._glare_seen(session)
        if "glare" in q.issues:
            if light == "ring":
                light = "diffuse"
                delta += GLARE_EV_COMP
                switched_off_ring = True
                fixes.append(f"glare_frac={q.glare_frac:.4f}>{qc.max_glare_frac:.4f} -> light ring->diffuse, EV {GLARE_EV_COMP:+.2f} (diffuse gain 0.6)")
            else:
                delta += GLARE_PERSIST_EV
                fixes.append(f"glare_frac={q.glare_frac:.4f}>{qc.max_glare_frac:.4f} persists under {light} -> EV {GLARE_PERSIST_EV:+.2f}")
        elif self.specular_suspected(q, s, qc):
            # Localized saturation under ring light with a normal mean level is the signature of a
            # specular reflection the glare detector did not segment: swap the light (same mean
            # brightness) instead of darkening the whole part. If clipping persists under diffuse
            # light, the next recapture applies the suggested exposure delta.
            light = "diffuse"
            delta += GLARE_EV_COMP
            switched_off_ring = True
            fixes.append(
                f"clipping-only overexposure (clipped_frac={q.clipped_frac:.3f}>{qc.max_clipped_frac:.3f} with mean_luma="
                f"{q.mean_luma:.1f}<={qc.max_luma:.0f}) under ring light -> likely specular reflection -> light ring->diffuse, "
                f"EV {GLARE_EV_COMP:+.2f} (same mean brightness)"
            )
        if "blurred" in q.issues:
            sr = f"sharpness_ratio={q.sharpness_ratio:.2f}<{qc.min_sharpness_ratio:.2f}" if q.sharpness_ratio is not None else f"sharpness={q.sharpness:.1f} (absolute heuristic)"
            if not af:
                af = True
                fixes.append(f"{sr} -> autofocus on")
            else:
                accepted.append(f"{sr} despite autofocus (not optically correctable)")
        exp_issue = [i for i in ("underexposed", "overexposed") if i in q.issues]
        if exp_issue:
            stats = f"mean_luma={q.mean_luma:.1f} (range [{qc.min_luma:.0f},{qc.max_luma:.0f}], target {qc.target_luma:.0f}), clipped={q.clipped_frac:.3f}, crushed={q.crushed_frac:.3f}"
            if "overexposed" in exp_issue and switched_off_ring and "underexposed" not in exp_issue:
                if not self.specular_suspected(q, s, qc):
                    fixes.append(f"overexposed ({stats}) attributed to glare clipping -> handled by the light switch")
            else:
                d = float(min(max(q.suggested_exposure_delta, -3.0), 3.0))
                if switched_off_ring and d < 0:
                    d = 0.0  # suggested delta was measured with glare in the frame
                bracket = self._exposure_bracket(session, cid, s, exp_issue) if not switched_off_ring else None
                if bracket is not None:
                    # the opposite exposure issue was measured at another EV of the same view/light: the
                    # good exposure lies in between -> bisect instead of stepping past it again (no oscillation)
                    prev_cid, prev_ev, prev_issue = bracket
                    span = prev_ev - ev
                    if abs(span) < EXPOSURE_BRACKET_MIN_EV:
                        accepted.append(
                            f"{exp_issue[0]} ({stats}) bracketed within {abs(span):.2f} EV by {prev_issue} at EV {prev_ev:+.2f} "
                            f"({prev_cid}) -> best achievable exposure, accepted"
                        )
                        d = 0.0
                    else:
                        d = span / 2.0
                        fixes.append(
                            f"{exp_issue[0]} ({stats}) at EV {ev:+.2f} but {prev_issue} at EV {prev_ev:+.2f} ({prev_cid}) "
                            f"-> bisect EV {d:+.2f}"
                        )
                        delta += d
                        d = 0.0
                if abs(d) >= 0.05:
                    delta += d
                    fixes.append(f"{'/'.join(exp_issue)} ({stats}) -> EV {d:+.2f} (suggested)")
                elif bracket is None:
                    accepted.append(f"{'/'.join(exp_issue)} ({stats}) with negligible suggested delta {q.suggested_exposure_delta:+.2f}")
        if "noisy" in q.issues:
            ns = f"noise_sigma={q.noise_sigma:.2f}>{qc.max_noise_sigma:.2f}"
            if ev > NOISY_EV_LIMIT:
                if light != "ring" and not glare_seen and not switched_off_ring:
                    comp = math.log2(1.0 / LIGHT_GAIN.get(light, 1.0))
                    fixes.append(f"{ns} at EV {ev:+.2f}>{NOISY_EV_LIMIT:+.1f} -> ring light (brighter), EV {-comp:+.2f} to lower the gain")
                    light = "ring"
                    delta -= comp
                elif glare_seen or switched_off_ring:
                    accepted.append(f"{ns} accepted (glare seen -> keep {light} light)")
                else:
                    accepted.append(f"{ns} at EV {ev:+.2f} accepted (already ring light)")
            else:
                accepted.append(f"{ns} at EV {ev:+.2f}<={NOISY_EV_LIMIT:+.1f} accepted (not gain-induced)")
        cc = session.settings.camera
        new_ev = float(min(max(ev + delta, cc.min_ev), cc.max_ev))
        new = CaptureSettings(view=s.view, exposure_ev=round(new_ev, 3), light=light, autofocus=af)
        return new, fixes, accepted

    @staticmethod
    def _exposure_bracket(
        session: InspectionSession, cid: str, s: CaptureSettings, exp_issue: list[str]
    ) -> tuple[str, float, str] | None:
        """``(capture_id, ev, issue)`` of the closest earlier capture of the same view and light whose quality
        report had the *opposite* exposure issue on the correct side of the current EV (None otherwise)."""
        if len(exp_issue) != 1:
            return None
        issue = exp_issue[0]
        opposite = "overexposed" if issue == "underexposed" else "underexposed"
        cur = session.captures.get(cid)
        idx = cur.index if cur is not None else len(session.captures)
        vkey = (round(s.view.cx, 3), round(s.view.cy, 3), round(s.view.zoom, 2))
        ev = float(s.exposure_ev)
        best: tuple[str, float, str] | None = None
        for ocid, cap in session.captures.items():
            if ocid == cid or cap.index >= idx:
                continue
            os_ = cap.settings
            if os_.light != s.light or (round(os_.view.cx, 3), round(os_.view.cy, 3), round(os_.view.zoom, 2)) != vkey:
                continue
            oq = session.quality.get(ocid)
            if oq is None or opposite not in oq.issues or issue in oq.issues:
                continue
            oev = float(os_.exposure_ev)
            if (issue == "underexposed" and oev <= ev) or (issue == "overexposed" and oev >= ev):
                continue
            if best is None or abs(oev - ev) < abs(best[1] - ev):
                best = (ocid, oev, opposite)
        return best

    @staticmethod
    def specular_suspected(q: QualityReport, s: CaptureSettings, qc) -> bool:
        """Overexposure caused only by clipping (mean luma in range) under ring light at nominal gain.

        Clipping that appears after the planner raised the digital gain (EV > SPECULAR_MAX_EV) is
        gain-induced saturation of bright features, not a specular reflection.
        """
        return (
            s.light == "ring"
            and float(s.exposure_ev) <= SPECULAR_MAX_EV
            and "overexposed" in q.issues
            and "glare" not in q.issues
            and q.clipped_frac > qc.max_clipped_frac
            and q.mean_luma <= qc.max_luma
        )

    def _glare_seen(self, session: InspectionSession) -> bool:
        """Glare reported, or specular clipping suspected, on any capture of this session."""
        qc = session.settings.quality
        for cid, r in session.quality.items():
            if "glare" in r.issues:
                return True
            cap = session.captures.get(cid)
            if cap is not None and self.specular_suspected(r, cap.settings, qc):
                return True
        return False

    def _quality_ok_note(self, session: InspectionSession, cid: str) -> str:
        q = session.quality[cid]
        qid = session.quality_ids.get(cid, "q_?")
        qc = session.settings.quality
        sr = f"sharpness_ratio={q.sharpness_ratio:.2f}>={qc.min_sharpness_ratio:.2f}" if q.sharpness_ratio is not None else f"sharpness={q.sharpness:.1f}"
        return (
            f"{qid} ({cid}): quality ok ({sr}, mean_luma={q.mean_luma:.1f} in [{qc.min_luma:.0f},{qc.max_luma:.0f}], "
            f"glare_frac={q.glare_frac:.4f}<={qc.max_glare_frac:.4f}, noise_sigma={q.noise_sigma:.2f}<={qc.max_noise_sigma:.2f})"
        )

    # ------------------------------------------------------------------ protocol
    def _decide(self, session: InspectionSession) -> Decision:
        caps = session.captures
        # 1. nothing seen yet
        if not caps:
            s = CaptureSettings()
            if not self._can_capture(session, s, 1):
                return self._review(session, "no capture possible within budget", [])
            return self._d(
                "capture_image",
                dict(DEFAULT_ARGS),
                "no capture yet -> default full view (cx=0.50, cy=0.50, zoom=1.0, EV +0.00, ring light, AF off)",
                [],
            )
        last = next(reversed(caps))
        # 2. quality first
        if last not in session.quality:
            if self._steps_left(session) <= 1:
                return self._conclude(session, "step budget: last step", [last])
            ls = caps[last].settings
            return self._d(
                "assess_quality",
                {"capture_id": last},
                f"{last} (zoom={ls.view.zoom:.1f}, {_fmt_settings(ls)}) not yet quality-assessed -> assess_quality({last}) before trusting it",
                [last],
            )
        full = [c for c in caps if session.is_full_view(c)]
        if not full:
            s = CaptureSettings()
            if self._can_capture(session, s, 3):
                return self._d("capture_image", dict(DEFAULT_ARGS), "no full-view capture in the session -> default full view", [last])
            return self._review(session, "no full-view capture and no budget to take one", [last])
        if not session.result_ids(view="full"):
            d = self._quality_phase(session, full)
            if d is not None:
                return d
            best = session.best_full_view()
            assert best is not None
            if self._steps_left(session) <= 1:
                return self._conclude(session, "step budget: last step before any detection", [best])
            return self._d(
                "detect_anomalies",
                {"capture_id": best, "detector": "fast"},
                f"{self._quality_note(session, best, full)} -> detect_anomalies(fast) on best full view {best}",
                [session.quality_ids.get(best, best), best],
            )
        return self._evidence_phase(session)

    def _quality_note(self, session: InspectionSession, best: str, full: list[str]) -> str:
        q = session.quality.get(best)
        if q is None or q.ok:
            return self._quality_ok_note(session, best) if q is not None else f"{best}: not assessed"
        qid = session.quality_ids.get(best, "q_?")
        _, _, accepted = self._fix(session, q, qid, best, session.captures[best].settings)
        n_corr = len(full) - 1
        why = "; ".join(accepted) if accepted else f"corrective recaptures {n_corr}/{self.max_corrective} used or budget reached"
        return f"{qid} ({best}): issues {q.issues} accepted ({why})"

    def _quality_phase(self, session: InspectionSession, full: list[str]) -> Decision | None:
        latest = full[-1]
        q = session.quality[latest]
        if not q.issues:
            return None
        n_corr = len(full) - 1
        if n_corr >= self.max_corrective:
            return None
        qid = session.quality_ids.get(latest, "q_?")
        cur = session.captures[latest].settings
        new, fixes, _accepted = self._fix(session, q, qid, latest, cur)
        if not fixes or _settings_key(new) == _settings_key(cur):
            return None
        tried = {_settings_key(session.captures[c].settings) for c in full}
        if _settings_key(new) in tried:
            return None
        # need: this capture + assess + detect + terminal
        if not self._can_capture(session, new, 4):
            return None
        args = {
            "cx": new.view.cx,
            "cy": new.view.cy,
            "zoom": new.view.zoom,
            "exposure_ev": new.exposure_ev,
            "light": new.light,
            "autofocus": new.autofocus,
        }
        rationale = (
            f"{qid} ({latest}) quality issues {q.issues}: " + "; ".join(fixes)
            + f" -> corrective recapture {n_corr + 1}/{self.max_corrective} ({_fmt_settings(cur)} -> {_fmt_settings(new)})"
        )
        return self._d("capture_image", args, rationale, [qid, latest])

    def _unreliable_note(self, session: InspectionSession) -> tuple[str, list[str]]:
        ids = session.result_ids(view="full")
        if not ids:
            return "no full-view detector result", []
        rid = ids[-1]
        res = session.results[rid]
        reg = res.registration or {}
        min_ncc = session.settings.detector.min_ncc
        parts = [f"{rid} ({session.result_info[rid].detector} on {res.capture_id}) unreliable: p={res.prob:.3f}"]
        if reg:
            ncc = reg.get("ncc")
            ncc_s = f"{ncc:.2f}" if isinstance(ncc, (int, float)) else "n/a"
            parts.append(f"registration ok={reg.get('ok')} method={reg.get('method')} ncc={ncc_s} (min {min_ncc:.2f})")
        q = session.quality.get(res.capture_id)
        if q is not None and not q.ok:
            parts.append(f"quality issues {q.issues}")
        return ", ".join(parts), [rid, res.capture_id]

    def _inspect_decision(self, session: InspectionSession, region: Region, why: str, evidence: list[str], overrides: dict | None = None) -> Decision:
        args = {"region_id": region.region_id, "zoom": self.inspect_zoom}
        if overrides:
            args.update(overrides)
        cx, cy = region.center_norm(session.scene_size)
        n_insp = len([r for r in session.inspections if session.inspections[r]])
        pol = session.policy_cfg
        rationale = (
            f"{why}; candidate {region.region_id} (peak={region.peak:.2f}, area={region.area_px} px, centre=({cx:.2f},{cy:.2f})) "
            f"-> inspect_region x{self.inspect_zoom:.0f} to confirm/refute (inspected regions {n_insp}/{self.max_inspections}, "
            f"captures {session.counters['captures']}/{pol.max_captures})"
        )
        return self._d("inspect_region", args, rationale, evidence + [region.region_id])

    def _can_inspect(self, session: InspectionSession, region: Region, overrides: dict | None = None) -> bool:
        args = {"region_id": region.region_id, "zoom": self.inspect_zoom, **(overrides or {})}
        try:
            s = settings_for_call(session, "inspect_region", args)
        except Exception:
            return False
        return self._can_capture(session, s, 2)

    def _measure_or_finish_fail(self, session: InspectionSession, regions: list[Region], why: str, evidence: list[str]) -> Decision:
        rep = session.fusion_report()
        p = rep["prob"]
        top = regions[0] if regions else None
        if top is not None and top.region_id not in session.measurements and self._steps_left(session) >= 2:
            return self._d("measure_defect", {"region_id": top.region_id}, f"{why} -> measure_defect({top.region_id}) before FAIL", evidence + [top.region_id])
        meas = ""
        if top is not None and top.region_id in session.measurements:
            m = session.measurements[top.region_id]
            area, length = m.get("area_mm2"), m.get("length_mm")
            area_s = f"{area:.3f} mm2" if isinstance(area, (int, float)) else "n/a"
            len_s = f"{length:.2f} mm" if isinstance(length, (int, float)) else "n/a"
            meas = f"; {top.region_id} measured: area={area_s}, length={len_s}, type={m.get('type_guess', 'unknown')}"
        ids = [r.region_id for r in regions[:3]]
        pol = session.policy_cfg
        text = f"{why}; fused p={p:.3f}>={pol.auto_fail_min_prob:.2f} [{rep['explain']}]{meas} -> FAIL"
        return self._d(
            "finish",
            {"label": "FAIL", "confidence": round(p, 4), "rationale": text[:4000], "region_ids": ids},
            text,
            evidence + rep["evidence"] + ids,
        )

    def _finish_pass(self, session: InspectionSession, why: str, evidence: list[str]) -> Decision:
        rep = session.fusion_report()
        p = rep["prob"]
        pol = session.policy_cfg
        text = f"fused p={p:.3f}<={pol.auto_pass_max_prob:.2f} [{rep['explain']}]; {why} -> PASS"
        return self._d(
            "finish",
            {"label": "PASS", "confidence": round(1.0 - p, 4), "rationale": text[:4000], "region_ids": []},
            text,
            evidence + rep["evidence"],
        )

    def _conclude(self, session: InspectionSession, why: str, evidence: list[str]) -> Decision:
        """Terminal decision by thresholds (FAIL / PASS) or human review."""
        pol = session.policy_cfg
        p = session.fusion_report()["prob"]
        if p >= pol.auto_fail_min_prob and session.has_reliable_evidence("FAIL"):
            # measures first only when >= 2 steps remain, otherwise finishes immediately
            regs = session.confirmed_regions() or session.full_view_regions()
            return self._measure_or_finish_fail(session, regs, why, evidence)
        contested = session.contested_results()
        alarms = session.unlocalized_zoom_alarms()
        blockers, blockers_why = pass_blockers(session, max_inspections=self.max_inspections)
        if (
            p <= pol.auto_pass_max_prob
            and session.has_reliable_evidence("PASS")
            and not session.confirmed_regions()
            and not contested
            and not alarms
            and not blockers
        ):
            return self._finish_pass(session, why, evidence)
        if p <= pol.auto_pass_max_prob and contested:
            band = f"fused p={p:.3f}<={pol.auto_pass_max_prob:.2f} contradicted by {session.contested_note(contested)}"
        elif p <= pol.auto_pass_max_prob and alarms:
            ps = ", ".join(f"{a} p={float(session.results[a].prob):.3f}" for a in alarms)
            band = f"fused p={p:.3f}<={pol.auto_pass_max_prob:.2f} but unlocalized zoom alarm(s) {ps} stand"
        elif p <= pol.auto_pass_max_prob and blockers:
            band = f"fused p={p:.3f}<={pol.auto_pass_max_prob:.2f} but {blockers_why}"
        else:
            band = f"fused p={p:.3f} not in auto bands (PASS<={pol.auto_pass_max_prob:.2f}, FAIL>={pol.auto_fail_min_prob:.2f})"
        if not session.has_reliable_evidence("PASS"):
            band += ", no reliable full-view evidence"
        return self._review(session, f"{why}; {band}", evidence + contested + alarms)

    def _retry_overrides(self, session: InspectionSession, region: Region) -> tuple[dict, str] | None:
        """Capture overrides for one retry of an inconclusive zoom of ``region`` whose view had a fixable
        quality issue (e.g. underexposed -> EV += suggested delta), with the rationale; None otherwise."""
        attempts = session.inspections.get(region.region_id, [])
        if not attempts or len(attempts) >= self.max_attempts_per_region or attempts[-1].status != "inconclusive":
            return None
        last = attempts[-1]
        q = session.quality.get(last.capture_id)
        if q is None or not q.issues:
            return None
        cur = session.captures[last.capture_id].settings
        new, fixes, _ = self._fix(session, q, last.quality_id, last.capture_id, cur)
        if not fixes or _settings_key(new) == _settings_key(cur):
            return None
        overrides = {"exposure_ev": new.exposure_ev, "light": new.light, "autofocus": new.autofocus}
        if not self._can_inspect(session, region, overrides):
            return None
        return overrides, f"{last.result_id}/{last.quality_id} inconclusive ({last.reason}): " + "; ".join(fixes)

    def _next_candidate(self, session: InspectionSession, unresolved: list[Region]) -> tuple[Region, dict | None, str] | None:
        n_inspected = len([r for r in session.inspections if session.inspections[r]])
        # first: un-inspected candidates (top-k by rank)
        for r in unresolved:
            if session.inspections.get(r.region_id) or session.location_inspected(r.region_id):
                continue  # never zoom twice into the same place as a fresh candidate
            if n_inspected >= self.max_inspections:
                break
            if self._can_inspect(session, r):
                return r, None, "uninspected"
        # then: one retry of an inconclusive candidate whose zoom view had a fixable quality issue
        for r in unresolved:
            retry = self._retry_overrides(session, r)
            if retry is not None:
                return r, retry[0], retry[1]
        return None

    @staticmethod
    def _confirmation_note(session: InspectionSession, region: Region, by: list[str]) -> str:
        insp = [i for lst in session.inspections.values() for i in lst if i.result_id in by and i.status == "confirmed"]
        if not insp:
            return f"{','.join(by)} confirm {region.region_id} (zoom p>={ZOOM_CONFIRM_PROB:.2f}, localized)"
        i = insp[-1]
        loc = f"overlapping {i.overlapping_regions}" if i.overlapping_regions else f"region_peak={i.region_peak:.2f} of map max {i.map_max:.2f}"
        return (
            f"{i.result_id} (zoom x{i.zoom:.0f} on {i.capture_id}, {i.quality_id} ok) confirms {region.region_id}: "
            f"zoom p={i.prob:.3f}>={ZOOM_CONFIRM_PROB:.2f}, {loc}"
        )

    def _evidence_phase(self, session: InspectionSession) -> Decision:
        pol = session.policy_cfg
        best = session.best_full_view()
        rep = session.fusion_report()
        p = rep["prob"]
        fev = rep["evidence"]
        # 5. no reliable evidence -> deep escalation (alignment-free) or review
        if not session.has_reliable_evidence("FAIL"):
            why, ev = self._unreliable_note(session)
            if self._deep_possible(session, best):
                return self._d(
                    "detect_anomalies",
                    {"capture_id": best, "detector": "deep"},
                    f"{why} -> escalate to the alignment-free deep detector on {best} (deep calls "
                    f"{session.counters['deep_calls']}/{pol.max_deep_calls})",
                    ev + [best],
                )
            return self._review(session, f"{why}; {self._deep_note(session)} -> no reliable evidence", ev)
        if self._steps_left(session) <= 1:
            return self._conclude(session, "step budget: last step", fev)

        # confirmed candidates (zoom confirmation) -> measure -> FAIL (or corroborate / review)
        confirmed = session.confirmed_regions()
        if confirmed:
            top = confirmed[0]
            _st, by = session.region_status_detail(top.region_id)
            why = self._confirmation_note(session, top, by)
            if p >= pol.auto_fail_min_prob:
                return self._measure_or_finish_fail(session, confirmed, why, by)
            if self._deep_possible(session, best):
                return self._d(
                    "detect_anomalies",
                    {"capture_id": best, "detector": "deep"},
                    f"{why} but fused p={p:.3f}<{pol.auto_fail_min_prob:.2f} [{rep['explain']}] -> deep detector for corroboration on {best}",
                    by + fev + [best],
                )
            if top.region_id not in session.measurements and self._steps_left(session) >= 2:
                return self._d("measure_defect", {"region_id": top.region_id}, f"{why}; fused p={p:.3f}<{pol.auto_fail_min_prob:.2f} -> measure before human review", by + [top.region_id])
            return self._review(session, f"{why} but fused p={p:.3f}<{pol.auto_fail_min_prob:.2f}", by)

        regions = session.full_view_regions()  # non-refuted, ranked
        # 6. strong evidence
        if p >= pol.auto_fail_min_prob:
            if regions:
                top = regions[0]
                fresh = not session.inspections.get(top.region_id) and not session.location_inspected(top.region_id)
                band = f"fused p={p:.3f}>={pol.auto_fail_min_prob:.2f} but <{self.strong_fail_prob:.2f} [{rep['explain']}]"
                if p < self.strong_fail_prob and fresh and self._can_inspect(session, top):
                    return self._inspect_decision(session, top, band, fev)
                if p < self.strong_fail_prob:
                    # the confirmation zoom was inconclusive because of a fixable quality issue (e.g. it
                    # inherited an exposure that underexposes the zoom view): retry it once before failing
                    retry = self._retry_overrides(session, top)
                    if retry is not None:
                        return self._inspect_decision(session, top, f"{band}; {retry[1]}", fev, retry[0])
                return self._measure_or_finish_fail(session, regions, f"fused p={p:.3f}>={pol.auto_fail_min_prob:.2f} with candidate {top.region_id}", fev)
            if self._deep_possible(session, best):
                return self._d(
                    "detect_anomalies",
                    {"capture_id": best, "detector": "deep"},
                    f"fused p={p:.3f}>={pol.auto_fail_min_prob:.2f} [{rep['explain']}] but no localized region -> deep detector for corroboration on {best}",
                    fev + [best],
                )
            return self._measure_or_finish_fail(session, [], "no localized region", fev)

        unresolved = [r for r in regions if session.region_status(r.region_id) in ("uninspected", "inconclusive")]
        contested_ids = session.contested_results()
        known = {u.region_id for u in unresolved}
        contested = [r for r in session.contested_regions() if r.region_id not in known]
        alarm_ids = session.unlocalized_zoom_alarms()
        alarm_regs = [r for r in session.unlocalized_alarm_regions() if r.region_id not in known]
        # 7. clear PASS (no unresolved candidate, no contested alarm, no unlocalized zoom alarm)
        if p <= pol.auto_pass_max_prob and not unresolved and not contested_ids and not alarm_ids:
            refuted = [r.region_id for r in session.full_view_regions(include_refuted=True) if session.region_status(r.region_id) == "refuted"]
            why = f"candidates {refuted} refuted by zoom" if refuted else "no candidate regions"
            return self._finish_pass(session, why, fev)

        # 8. ambiguous (or contested): zoom into candidates
        if p > pol.auto_pass_max_prob:
            band = f"fused p={p:.3f} [{rep['explain']}] in ambiguous band ({pol.auto_pass_max_prob:.2f}, {pol.auto_fail_min_prob:.2f})"
        elif unresolved:
            band = f"fused p={p:.3f} [{rep['explain']}]<={pol.auto_pass_max_prob:.2f} but {len(unresolved)} unresolved candidate(s)"
        else:
            band = f"fused p={p:.3f} [{rep['explain']}]<={pol.auto_pass_max_prob:.2f}"
        if contested_ids:
            band += f"; conflicting evidence: {session.contested_note(contested_ids)}"
            fev = fev + [rid for rid in contested_ids if rid not in fev]
        if alarm_ids:
            ps = ", ".join(f"{a} p={float(session.results[a].prob):.3f}" for a in alarm_ids)
            band += f"; unlocalized zoom alarm(s) {ps} (anomaly seen away from the inspected candidate) -> their regions are new candidates"
            fev = fev + [rid for rid in alarm_ids if rid not in fev]
        cand = self._next_candidate(session, unresolved + alarm_regs + contested)
        if cand is not None:
            region, overrides, kind = cand
            why = band if kind == "uninspected" else f"{band}; {kind}"
            return self._inspect_decision(session, region, why, fev, overrides)
        if self._deep_possible(session, best):
            return self._d(
                "detect_anomalies",
                {"capture_id": best, "detector": "deep"},
                f"{band}; zoom candidates exhausted -> deep detector on full view {best} (deep calls {session.counters['deep_calls']}/{pol.max_deep_calls})",
                fev + [best],
            )
        unresolved_ids = [r.region_id for r in unresolved]
        why = f"zoom candidates exhausted (unresolved {unresolved_ids}); {self._deep_note(session)}"
        return self._conclude(session, why, fev)


__all__ = ["RulePlanner", "LIGHT_GAIN", "GLARE_EV_COMP", "DEFAULT_ARGS", "STRONG_FAIL_PROB"]
