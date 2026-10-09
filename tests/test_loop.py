"""Agent loop, single-shot baseline, traces/logging, model bundles, and a real-module integration run."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from inspectagent import models_io
from inspectagent.agent.loop import compute_totals, make_planner, run_inspection, run_single_shot
from inspectagent.agent.planner_rule import RulePlanner
from inspectagent.agent.policy import HumanControlPolicy
from inspectagent.agent.tools import TERMINAL_TOOLS
from inspectagent.agent.trace import (
    JsonLineFormatter,
    configure_logging,
    load_trace,
    log_step,
    save_trace,
    step_record,
    trace_from_dict,
    trace_to_json,
)
from inspectagent.config import PolicyConfig, Settings
from inspectagent.types import Decision
from tests.agent_fakes import DEFECT, FALSE_ALARM, FakeVision, make_bundle, make_part, make_scene, run_fake

@pytest.fixture()
def tmp_dir():
    """Temporary directory via ``tempfile`` (pytest's own temp base dir can be locked on Windows sandboxes)."""
    d = Path(tempfile.mkdtemp(prefix="inspectagent_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ loop with fakes


def test_loop_trace_and_totals():
    seen = []
    trace, s = run_fake("glare", spots=[DEFECT], label=1, on_step=lambda st, sess: seen.append((st.step, sess.session_id)))
    t = trace.totals
    assert [st.step for st in trace.steps] == list(range(1, len(trace.steps) + 1))
    assert seen == [(st.step, "test") for st in trace.steps]
    assert t["n_steps"] == len(trace.steps) == sum(t["tool_counts"].values())
    assert t["n_captures"] == s.counters["captures"] == 3  # default, corrected, zoom
    assert t["n_deep"] == 0 and t["n_zoom"] == 1 and t["n_corrective_captures"] == 1
    assert t["cycle_time_ms"] == pytest.approx(t["compute_ms"] + t["sim_time_ms"], abs=1e-2)
    assert t["sim_time_ms"] == pytest.approx(sum(st.sim_time_ms for st in trace.steps), abs=1e-2)
    assert t["final_prob"] == pytest.approx(trace.verdict.prob_anomalous)
    assert t["evidence_changed_action"] is True and t["evidence_driven_steps"]
    assert trace.verdict.label == "FAIL" and trace.ground_truth == {"label": 1}
    assert trace.trace_id == "tr_test" and trace.planner == "rule" and trace.conditions["preset"] == "glare"
    assert trace.budget["max_steps"] == 14 and "opencv_version" in trace.cv_info
    assert all(a in s.artifacts for st in trace.steps for a in st.artifacts)
    json.loads(trace_to_json(trace))


def test_clean_run_has_no_evidence_driven_action():
    trace, _ = run_fake("clean")
    assert trace.totals["evidence_changed_action"] is False and trace.verdict.action == "route_pass"


def test_loop_is_deterministic():
    a, _ = run_fake("harsh", spots=[FALSE_ALARM, DEFECT])
    b, _ = run_fake("harsh", spots=[FALSE_ALARM, DEFECT])
    strip = lambda tr: [(st.decision.tool, st.decision.args, st.decision.rationale, st.observation.get("prob")) for st in tr.steps]
    assert strip(a) == strip(b)
    assert a.verdict.to_dict() == b.verdict.to_dict()


@pytest.mark.parametrize("preset", ["clean", "glare", "defocus", "dark", "harsh"])
def test_loop_terminates_within_budget(preset):
    pol = PolicyConfig()
    trace, s = run_fake(preset, spots=[FALSE_ALARM, DEFECT])
    assert len(trace.steps) <= pol.max_steps
    assert s.counters["captures"] <= pol.max_captures and s.counters["deep_calls"] <= pol.max_deep_calls
    assert trace.steps[-1].decision.tool in TERMINAL_TOOLS
    assert trace.verdict.label != "PASS"  # a real defect is present


class _LoopingPlanner:
    name = "looping"

    def next_action(self, state, session):
        return Decision(step=0, tool="capture_image", args={"cx": 0.5, "cy": 0.5}, rationale="again", evidence=[], planner="looping")


class _RaisingPlanner:
    name = "raising"

    def next_action(self, state, session):
        raise RuntimeError("model unavailable")


class _BadArgsPlanner:
    name = "badargs"

    def next_action(self, state, session):
        return Decision(step=0, tool="assess_quality", args={"capture_id": 42}, rationale="bad", evidence=[], planner="badargs")


class _UnknownToolPlanner:
    name = "unknown"

    def next_action(self, state, session):
        return Decision(step=0, tool="rm_rf", args={}, rationale="?", evidence=[], planner="unknown")


def test_looping_planner_is_stopped_by_capture_budget():
    trace, s = run_fake("clean", planner=_LoopingPlanner())
    assert s.counters["captures"] == 8 and len(trace.steps) == 9
    last = trace.steps[-1]
    assert last.overridden_by_policy and last.decision.tool == "request_human_review" and last.decision.planner == "policy"
    assert "capture budget exhausted: 8/8" in last.decision.rationale
    assert trace.verdict.label == "NEEDS_REVIEW" and trace.verdict.review_id
    assert trace.planner == "looping" and trace.totals["n_overrides"] == 1


def test_looping_planner_is_stopped_by_step_budget():
    pol = PolicyConfig(max_steps=4, max_captures=50, max_sim_time_ms=1e9)
    trace, _ = run_fake("clean", planner=_LoopingPlanner(), policy=HumanControlPolicy(pol), settings=Settings(policy=pol))
    assert len(trace.steps) == 4 and "step 4/4 must be terminal" in trace.steps[-1].decision.rationale


def test_planner_exception_becomes_review():
    trace, s = run_fake("clean", planner=_RaisingPlanner())
    assert len(trace.steps) == 1 and trace.steps[0].overridden_by_policy
    assert "planner error RuntimeError: model unavailable" in trace.steps[0].decision.rationale
    assert trace.errors and trace.verdict.label == "NEEDS_REVIEW"


def test_two_tool_errors_force_review():
    trace, s = run_fake("clean", planner=_BadArgsPlanner())
    assert [st.decision.tool for st in trace.steps] == ["assess_quality", "assess_quality", "request_human_review"]
    assert all("error" in st.observation for st in trace.steps[:2]) and len(trace.errors) == 2
    assert "2 tool errors" in trace.steps[-1].decision.rationale and trace.totals["n_errors"] == 2


def test_unknown_tool_becomes_review():
    trace, _ = run_fake("clean", planner=_UnknownToolPlanner())
    assert len(trace.steps) == 1 and "unknown tool 'rm_rf'" in trace.steps[0].decision.rationale


def test_on_step_errors_do_not_break_the_loop():
    def boom(step, session):
        raise ValueError("ui gone")

    trace, _ = run_fake("clean", on_step=boom)
    assert trace.verdict.label == "PASS" and any("on_step callback error" in e for e in trace.errors)


def test_compute_totals_counts_overrides():
    trace, s = run_fake("clean", planner=_UnknownToolPlanner())
    assert compute_totals(trace, s)["n_overrides"] == 1


def test_make_planner():
    assert isinstance(make_planner("rule"), RulePlanner)
    with pytest.raises(ValueError):
        make_planner("oracle")


# ------------------------------------------------------------------ single shot


def _single(spots=None, **kw):
    vision = FakeVision()
    return run_single_shot(make_part("glare"), make_bundle(spots), scene=make_scene(), backend=vision.backend(), **kw)


def test_single_shot_fast_and_oracle():
    d = _single()
    assert d["detector"] == "fast" and d["oracle"] is False and d["label"] == "PASS" and d["label_binary"] == "PASS"
    assert d["reliable"] is True  # quality is not assessed in the single-shot baseline
    assert d["sim_time_ms"] > 0 and d["cycle_time_ms"] == pytest.approx(d["latency_ms"] + d["sim_time_ms"])
    assert d["capture"]["settings"]["light"] == "ring" and d["capture"]["settings"]["view"]["zoom"] == 1.0
    o = _single(oracle=True)
    assert o["oracle"] is True and o["capture"]["oracle"] is True
    m = _single(spots=[DEFECT])
    assert m["label"] == "NEEDS_REVIEW" and m["label_binary"] == "FAIL" and m["n_regions"] == 1
    json.dumps(m, allow_nan=False)


def test_single_shot_deep():
    d = _single(spots=[FALSE_ALARM], detector="deep")
    assert d["detector"] == "deep" and d["prob"] == pytest.approx(0.05)
    with pytest.raises(ValueError):
        run_single_shot(make_part(), make_bundle(with_deep=False), detector="deep", scene=make_scene(), backend=FakeVision().backend())


# ------------------------------------------------------------------ traces & logging


def test_trace_roundtrip(tmp_dir: Path):
    trace, _ = run_fake("glare", spots=[DEFECT])
    p = save_trace(trace, tmp_dir / "sub dir" / "trace ç.json")
    d = load_trace(p)
    assert d["trace_id"] == trace.trace_id and len(d["steps"]) == len(trace.steps)
    t2 = trace_from_dict(d)
    assert t2.to_dict() == trace.to_dict()
    assert t2.verdict.defect_regions[0].region_id == trace.verdict.defect_regions[0].region_id


def test_log_step_is_one_json_line(caplog):
    trace, _ = run_fake("clean")
    logger = logging.getLogger("inspectagent.test")
    with caplog.at_level(logging.INFO, logger="inspectagent.test"):
        rec = log_step(logger, trace.steps[2], trace.trace_id)
    line = caplog.records[-1].getMessage()
    assert "\n" not in line
    parsed = json.loads(line)
    assert parsed == rec and parsed["evt"] == "step" and parsed["trace_id"] == trace.trace_id
    assert parsed["tool"] == "detect_anomalies" and "prob" in parsed["obs"]
    assert step_record(trace.steps[0], "x")["step"] == 1


def test_loop_logs_start_steps_verdict(caplog):
    with caplog.at_level(logging.INFO, logger="inspectagent.agent"):
        trace, _ = run_fake("clean")
    evts = [json.loads(r.getMessage())["evt"] for r in caplog.records if r.name == "inspectagent.agent" and r.getMessage().startswith("{")]
    assert evts[0] == "start" and evts[-1] == "verdict" and evts.count("step") == len(trace.steps)


def test_json_formatter_wraps_plain_messages():
    fmt = JsonLineFormatter()
    rec = logging.LogRecord("x", logging.WARNING, __file__, 1, "plain %s", ("msg",), None)
    assert json.loads(fmt.format(rec)) == {"evt": "log", "level": "WARNING", "logger": "x", "msg": "plain msg"}
    rec2 = logging.LogRecord("x", logging.INFO, __file__, 1, '{"evt":"step"}', None, None)
    assert fmt.format(rec2) == '{"evt":"step"}'
    import io

    buf = io.StringIO()
    lg = configure_logging("INFO", buf)
    lg.info("hello")
    configure_logging("INFO", buf)  # idempotent: still one handler
    lg.info("again")
    lines = buf.getvalue().strip().splitlines()
    assert [json.loads(x)["msg"] for x in lines] == ["hello", "again"]
    for h in list(lg.handlers):
        if getattr(h, "_inspectagent_json", False):
            lg.removeHandler(h)
    lg.propagate = True


# ------------------------------------------------------------------ model bundles


class _DiskStat:
    def __init__(self, tag: str = "stat"):
        self.tag = tag

    def save(self, path):
        Path(path).write_text(self.tag, encoding="utf-8")


class _DiskDeep(_DiskStat):
    pass


@pytest.fixture()
def fake_loaders(monkeypatch):
    calls = {"stat": 0, "deep": 0}

    def load_stat(path):
        calls["stat"] += 1
        return _DiskStat(Path(path).read_text(encoding="utf-8"))

    def load_deep(path, onnx):
        calls["deep"] += 1
        return _DiskDeep(Path(path).read_text(encoding="utf-8") + "|" + Path(onnx).name)

    monkeypatch.setattr(models_io, "_load_stat", load_stat)
    monkeypatch.setattr(models_io, "_load_deep", load_deep)
    models_io.clear_bundle_cache()
    yield calls
    models_io.clear_bundle_cache()


def test_bundle_save_load_cache(tmp_dir: Path, fake_loaders):
    root = tmp_dir / "models dir"
    assert models_io.bundle_dir("pcb1", root) == root / "pcb1"
    b = models_io.CategoryBundle("pcb1", _DiskStat("s1"), _DiskDeep("d1"), {"n": np.int64(3), "x": np.float32(0.5)}, {"z1": 50.0, "z3": 18.0})
    models_io.save_bundle(b, root)
    meta = json.loads((root / "pcb1" / "meta.json").read_text(encoding="utf-8"))
    assert meta["category"] == "pcb1" and meta["has_deep"] is True and meta["n"] == 3
    assert meta["reference_sharpness"] == {"z1": 50.0, "z3": 18.0}

    lb = models_io.load_bundle("pcb1", root)
    assert lb.stat.tag == "s1" and lb.deep is None and lb.meta["deep_load_status"].startswith("missing backbone")
    (root / models_io.BACKBONE_FILE).write_bytes(b"onnx")
    models_io.clear_bundle_cache()
    lb = models_io.load_bundle("pcb1", root)
    assert lb.deep.tag == "d1|backbone_resnet18.onnx" and lb.has_deep and lb.reference_sharpness["z3"] == 18.0
    assert models_io.load_bundle("pcb1", root) is lb  # cached per process
    assert models_io.load_bundle("pcb1", str(root)) is lb  # same key for str/Path
    assert fake_loaders["stat"] == 2
    nodeep = models_io.load_bundle("pcb1", root, with_deep=False)
    assert nodeep is not lb and nodeep.deep is None and nodeep.meta["deep_load_status"] == "disabled"
    models_io.load_bundle.cache_clear()  # lru_cache-compatible alias
    assert models_io.load_bundle("pcb1", root) is not lb


def test_available_categories_and_missing(tmp_dir: Path, fake_loaders):
    root = tmp_dir / "m"
    assert models_io.available_categories(root) == []
    for cat in ("zzz", "pcb2", "pcb1"):
        models_io.save_bundle(models_io.CategoryBundle(cat, _DiskStat(cat)), root)
    (root / "empty").mkdir()
    assert models_io.available_categories(root) == ["pcb1", "pcb2", "zzz"]
    with pytest.raises(FileNotFoundError):
        models_io.load_bundle("cashew", root)


# ------------------------------------------------------------------ integration with the real vision stack


def _real_modules_available() -> bool:
    try:
        from inspectagent.vision import anomaly_stat, camera, quality, regions  # noqa: F401
    except Exception:
        return False
    return hasattr(anomaly_stat, "StatTemplateModel") and hasattr(camera, "VirtualCamera") and hasattr(camera, "make_conditions")


@pytest.fixture(scope="module")
def real_bundle():
    import cv2

    from inspectagent.config import DetectorConfig
    from inspectagent.types import CaptureSettings, NuisanceConditions, View
    from inspectagent.vision.anomaly_stat import StatTemplateModel
    from inspectagent.vision.camera import VirtualCamera, make_conditions
    from inspectagent.vision.quality import laplacian_sharpness
    from tests.synth import make_board

    stat = StatTemplateModel.fit("pcb1", [make_board(i) for i in range(10)], cfg=DetectorConfig(template_long_side=640), log=None)
    scores: dict[str, list[float]] = {"z1": [], "z3": []}
    for i in range(5):
        cam = VirtualCamera(make_board(100 + i), make_conditions("nominal", np.random.default_rng(i)), seed=i)
        views = [(View(0.5, 0.5, 1.0), "z1"), (View(0.35, 0.4, 3.0), "z3"), (View(0.65, 0.6, 3.0), "z3")]
        for j, (v, b) in enumerate(views):
            scores[b].append(stat.score_capture(cam.capture(CaptureSettings(view=v), f"c{j}", j))[0])
    stat.calibrate(scores)
    ref_cam = VirtualCamera(make_board(999), NuisanceConditions(noise_sigma=0.0, preset="clean"), seed=0)
    ref = {}
    for z in (1.0, 2.0, 3.0, 4.0):
        cap = ref_cam.clean_capture(View(0.5, 0.5, z), "ref", 0)
        ref[f"z{int(z)}"] = float(laplacian_sharpness(cv2.cvtColor(cap.image, cv2.COLOR_BGR2GRAY)))
    return models_io.CategoryBundle("pcb1", stat, None, {"synthetic": True}, ref)


@pytest.mark.skipif(not _real_modules_available(), reason="vision modules not available")
@pytest.mark.parametrize("preset", ["clean", "glare", "defocus", "dark"])
def test_end_to_end_with_real_vision(real_bundle, preset):
    from inspectagent.types import PartSpec
    from inspectagent.vision.camera import make_conditions
    from tests.synth import inject_defect, make_board

    pol = PolicyConfig()
    cond = make_conditions(preset, np.random.default_rng(11))
    defect_scene, _mask = inject_defect(make_board(777), "scratch", seed=3)
    for label, scene in ((0, make_board(555)), (1, defect_scene)):
        part = PartSpec(f"synth_{preset}_{label}", "pcb1", "", label=label, conditions=cond)
        trace, s = run_inspection(part, real_bundle, scene=scene, seed=3)
        assert len(trace.steps) <= pol.max_steps and s.counters["captures"] <= pol.max_captures
        assert s.sim_time_ms <= pol.max_sim_time_ms
        assert trace.steps[-1].decision.tool in TERMINAL_TOOLS
        caps = [st.decision.args for st in trace.steps if st.decision.tool == "capture_image"]
        if preset == "glare":
            assert any(a.get("light") == "diffuse" for a in caps), caps
        if preset == "defocus":
            assert any(a.get("autofocus") is True for a in caps), caps
        if preset == "dark":
            assert any(a.get("exposure_ev", 0.0) > 0.5 for a in caps), caps
        if label == 1:
            assert trace.verdict.label != "PASS", trace.verdict.rationale
        json.loads(trace_to_json(trace))
