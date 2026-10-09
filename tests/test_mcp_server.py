"""MCP server tests: direct tool calls, policy gating, traces and SDK round trips (offline)."""

from __future__ import annotations

import base64
import json
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import cv2
import mcp_types as mtypes
import numpy as np
import pytest
from jsonschema import Draft202012Validator

from inspectagent.config import REPO_ROOT, Settings
from inspectagent.mcp_server import InspectAgentMCP, ToolOutput, build_server, main
from inspectagent.types import PartSpec
from tests.synth import make_board
from tests.test_planner_bedrock import make_real_session

# --------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------


@pytest.fixture()
def workdir() -> Iterator[Path]:
    """Private temporary directory (independent of pytest's shared basetemp)."""
    with tempfile.TemporaryDirectory(prefix="inspectagent_mcp_", ignore_cleanup_errors=True) as d:
        yield Path(d)


@pytest.fixture()
def samples_file(workdir: Path) -> Path:
    """A tiny samples.json in the repository format with two synthetic boards."""
    img = make_board(1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    (workdir / "pcb1").mkdir()
    (workdir / "pcb1" / "board_normal.png").write_bytes(buf.tobytes())
    (workdir / "pcb1" / "board_anomaly.png").write_bytes(buf.tobytes())
    samples = [
        {"id": "board_normal", "category": "pcb1", "label": 0, "label_name": "normal", "image": "pcb1/board_normal.png",
         "mask": None, "suggested_preset": "glare", "note": "Defect-free board."},
        {"id": "board_anomaly", "category": "pcb1", "label": 1, "label_name": "anomaly", "image": "pcb1/board_anomaly.png",
         "mask": None, "suggested_preset": "nominal", "note": "Scratch."},
        {"id": "other_cat", "category": "cashew", "label": 0, "image": "missing.png"},
    ]
    path = workdir / "samples.json"
    path.write_text(json.dumps(samples), encoding="utf-8")
    return path


class SessionFactory:
    """Builds real InspectionSessions (fake quality/detector backends) and records the parts."""

    def __init__(self, prob: float = 0.03, with_region: bool = False) -> None:
        self.prob, self.with_region = prob, with_region
        self.parts: list[PartSpec] = []

    def __call__(self, part: PartSpec, seed: int) -> Any:
        self.parts.append(part)
        session = make_real_session(prob=self.prob, with_region=self.with_region, session_id=f"s_{part.part_id}",
                                    label=part.label if part.label is not None else 0)
        session.part = part  # keep the requested part (label / conditions) for the trace
        return session


def make_app(samples_file: Path, **kw: Any) -> tuple[InspectAgentMCP, SessionFactory]:
    factory = kw.pop("factory", None) or SessionFactory()
    app = InspectAgentMCP(settings=Settings(), samples_path=samples_file, session_factory=factory, **kw)
    return app, factory


def ok(out: ToolOutput) -> dict:
    assert not out.is_error, out.data
    return out.data


def decode(jpeg: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    return img


# --------------------------------------------------------------------------------------------
# tool definitions and static tools
# --------------------------------------------------------------------------------------------


def test_tool_definitions_are_built_from_the_registry(samples_file):
    app, _ = make_app(samples_file)
    tools = {t.name: t for t in app.tool_definitions()}
    for name in ("list_samples", "start_session", "get_trace", "run_inspection", "get_artifact", "close_session"):
        assert name in tools
    from inspectagent.agent.tools import default_registry

    for spec in default_registry().specs():
        t = tools[spec["name"]]
        schema = t.input_schema
        assert schema["required"][0] == "session_id"
        assert set(spec["input_schema"]["properties"]) <= set(schema["properties"])
        assert "step_rationale" in schema["properties"]
        Draft202012Validator.check_schema(schema)
    for name in ("start_session", "run_inspection"):
        assert "glare" in tools[name].input_schema["properties"]["preset"]["enum"]
    assert app.tool_input_schema("capture_image")["required"][0] == "session_id"
    assert app.tool_input_schema("nope") is None


def test_list_samples_hides_labels_unless_asked(samples_file):
    app, _ = make_app(samples_file)
    data = ok(app.call("list_samples", {}))
    assert data["count"] == 3
    first = data["samples"][0]
    assert first == {"id": "board_normal", "category": "pcb1", "suggested_preset": "glare", "image_available": True}
    assert data["samples"][2]["image_available"] is False
    data = ok(app.call("list_samples", {"category": "pcb1", "include_labels": True}))
    assert data["count"] == 2 and data["samples"][1]["label"] == 1 and data["samples"][1]["note"] == "Scratch."
    assert app.call("list_samples", {"bogus": 1}).is_error


def test_start_session_validation(samples_file, workdir):
    app, _ = make_app(samples_file)
    assert "pass sample_id or image_path" in app.call("start_session", {}).data["error"]
    assert "unknown sample_id" in app.call("start_session", {"sample_id": "zzz"}).data["error"]
    assert "unknown preset" in app.call("start_session", {"sample_id": "board_normal", "preset": "foggy"}).data["error"]
    assert "not found" in app.call("start_session", {"sample_id": "other_cat"}).data["error"]
    txt = workdir / "notes.txt"
    txt.write_text("x")
    assert "unsupported image type" in app.call("start_session", {"image_path": str(txt), "category": "pcb1"}).data["error"]
    img_path = samples_file.parent / "pcb1" / "board_normal.png"
    assert "category is required" in app.call("start_session", {"image_path": str(img_path)}).data["error"]
    locked, _ = make_app(samples_file, allow_image_paths=False)
    assert "disabled" in locked.call("start_session", {"image_path": str(img_path), "category": "pcb1"}).data["error"]
    data = ok(app.call("start_session", {"image_path": str(img_path), "category": "pcb1", "preset": "dark", "seed": 3}))
    assert data["part_id"] == "board_normal" and data["preset"] == "dark"


def test_start_session_uses_suggested_preset_and_seeded_conditions(samples_file):
    app, factory = make_app(samples_file)
    data = ok(app.call("start_session", {"sample_id": "board_normal", "seed": 5}))
    assert data["preset"] == "glare" and data["session_id"] == "s_board_normal"
    assert data["policy_enforced"] is True and "capture_image" in data["tools"]
    cond = factory.parts[0].conditions
    assert cond.preset == "glare" and cond.glare_strength > 0
    ok(app.call("start_session", {"sample_id": "board_normal", "seed": 5}))
    assert factory.parts[1].conditions.to_dict() == cond.to_dict()  # deterministic per seed
    assert app.session_ids == ["s_board_normal", "s_board_normal-2"]


# --------------------------------------------------------------------------------------------
# session-scoped tools
# --------------------------------------------------------------------------------------------


def test_manual_session_pass_flow_with_images_and_trace(samples_file):
    app, _ = make_app(samples_file)
    sid = ok(app.call("start_session", {"sample_id": "board_normal", "preset": "clean"}))["session_id"]

    out = app.call("capture_image", {"session_id": sid, "cx": 0.5, "cy": 0.5, "zoom": 1, "exposure_ev": 0, "light": "ring",
                                     "autofocus": False, "step_rationale": "start with the full view"})
    cap = ok(out)
    assert cap["observation"]["capture_id"] == "cap_1" and cap["artifacts"] == ["art_1"]
    assert len(out.images) == 1 and max(decode(out.images[0][1]).shape[:2]) <= 384
    assert cap["budget_used"]["captures"] == "1/8"

    q = ok(app.call("assess_quality", {"session_id": sid, "capture_id": "cap_1"}))
    assert q["observation"]["ok"] is True and q["observation"]["quality_id"] == "q_1"

    trace_before = ok(app.call("get_trace", {"session_id": sid}))
    assert trace_before["ground_truth"] is None and trace_before["conditions"] is None and trace_before["closed"] is False

    det = ok(app.call("detect_anomalies", {"session_id": sid, "capture_id": "cap_1", "detector": "fast"}))
    assert det["observation"]["reliable"] is True and det["fused_prob"] == pytest.approx(0.03)

    fin = ok(app.call("finish", {"session_id": sid, "label": "PASS", "confidence": 0.95, "rationale": "det_1 p=0.03", "region_ids": []}))
    assert "policy_override" not in fin
    assert fin["verdict"]["label"] == "PASS" and fin["verdict"]["action"] == "route_pass" and fin["session_closed"]

    again = app.call("capture_image", {"session_id": sid})
    assert again.is_error and "finished" in again.data["error"]

    trace = ok(app.call("get_trace", {"session_id": sid}))
    assert trace["planner"] == "mcp-client" and trace["closed"] is True
    assert [s["decision"]["tool"] for s in trace["steps"]] == ["capture_image", "assess_quality", "detect_anomalies", "finish"]
    assert trace["steps"][0]["decision"]["rationale"] == "start with the full view"
    assert trace["steps"][0]["decision"]["planner"] == "mcp-client"
    assert "step_rationale" not in trace["steps"][0]["decision"]["args"]
    assert trace["totals"]["n_captures"] == 1 and trace["totals"]["n_steps"] == 4
    assert trace["totals"]["tool_counts"]["capture_image"] == 1
    assert trace["ground_truth"] == {"label": 0} and trace["conditions"]["preset"] == "clean"
    assert trace["verdict"]["label"] == "PASS"


def test_policy_blocks_unsupported_fail_and_routes_to_review(samples_file):
    app, _ = make_app(samples_file)
    sid = ok(app.call("start_session", {"sample_id": "board_anomaly", "preset": "clean"}))["session_id"]
    ok(app.call("capture_image", {"session_id": sid}))
    ok(app.call("assess_quality", {"session_id": sid, "capture_id": "cap_1"}))
    ok(app.call("detect_anomalies", {"session_id": sid, "capture_id": "cap_1"}))
    out = app.call("finish", {"session_id": sid, "label": "FAIL", "confidence": 0.9, "rationale": "I think it is bad", "region_ids": []})
    data = ok(out)
    assert data["policy_override"]["requested"]["tool"] == "finish"
    assert data["tool"] == "request_human_review"
    assert data["verdict"]["label"] == "NEEDS_REVIEW" and data["verdict"]["action"] == "hold_for_review"
    assert data["verdict"]["review_id"]
    trace = ok(app.call("get_trace", {"session_id": sid}))
    assert trace["steps"][-1]["overridden_by_policy"] is True
    assert trace["totals"]["n_policy_overrides"] == 1


def test_confirmed_high_probability_fail_is_accepted(samples_file):
    app, _ = make_app(samples_file, factory=SessionFactory(prob=0.97, with_region=True))
    sid = ok(app.call("start_session", {"sample_id": "board_anomaly", "preset": "clean"}))["session_id"]
    ok(app.call("capture_image", {"session_id": sid}))
    ok(app.call("assess_quality", {"session_id": sid, "capture_id": "cap_1"}))
    det = ok(app.call("detect_anomalies", {"session_id": sid, "capture_id": "cap_1"}))
    rid = det["observation"]["regions"][0]["region_id"]
    meas = ok(app.call("measure_defect", {"session_id": sid, "region_id": rid}))
    assert meas["observation"]["type_guess"] == "spot"
    fin = ok(app.call("finish", {"session_id": sid, "label": "FAIL", "confidence": 0.97, "rationale": f"{rid} p=0.97", "region_ids": [rid]}))
    assert fin["verdict"]["label"] == "FAIL" and fin["verdict"]["action"] == "route_quarantine"
    assert fin["verdict"]["defect_regions"][0]["region_id"] == rid


def test_tool_errors_are_reported_not_raised(samples_file):
    app, _ = make_app(samples_file)
    sid = ok(app.call("start_session", {"sample_id": "board_normal"}))["session_id"]
    out = app.call("assess_quality", {"session_id": sid, "capture_id": "cap_99"})
    assert out.is_error and "cap_99" in out.data["observation"]["error"]
    assert app.call("assess_quality", {"capture_id": "cap_1"}).is_error  # missing session_id
    assert "unknown session_id" in app.call("capture_image", {"session_id": "ghost"}).data["error"]
    assert "unknown tool" in app.call("teleport", {}).data["error"]


def test_get_artifact_and_close_session_and_eviction(samples_file):
    app, _ = make_app(samples_file, max_sessions=2)
    sid = ok(app.call("start_session", {"sample_id": "board_normal"}))["session_id"]
    ok(app.call("capture_image", {"session_id": sid}))
    out = app.call("get_artifact", {"session_id": sid, "artifact_id": "art_1", "max_side": 200})
    assert ok(out)["artifact_id"] == "art_1" and max(decode(out.images[0][1]).shape[:2]) <= 200
    assert "unknown artifact_id" in app.call("get_artifact", {"session_id": sid, "artifact_id": "art_9"}).data["error"]
    assert ok(app.call("close_session", {"session_id": sid}))["closed"] is True
    assert app.session_ids == []
    ids = [ok(app.call("start_session", {"sample_id": "board_normal"}))["session_id"] for _ in range(3)]
    assert app.session_ids == ids[1:]  # least recently used evicted


def test_run_inspection_uses_the_loop_and_returns_final_image(samples_file):
    seen: dict = {}

    def runner(part: PartSpec, planner: Any, seed: int):
        seen.update(part=part, planner=planner, seed=seed)
        session = make_real_session(prob=0.03, session_id="loop_sess")
        from inspectagent.agent.loop import run_inspection

        return run_inspection(part, bundle=session.bundle, planner=planner, seed=seed, scene=session.scene,
                              backend=session.vision, session_id="loop_sess")

    app, _ = make_app(samples_file, inspection_runner=runner)
    out = app.call("run_inspection", {"sample_id": "board_normal", "preset": "clean", "planner": "rule", "seed": 2})
    data = ok(out)
    assert seen["seed"] == 2 and seen["part"].conditions.preset == "clean"
    assert type(seen["planner"]).__name__ == "RulePlanner"
    assert data["session_id"] == "loop_sess" and data["trace_id"] == "tr_loop_sess"
    assert data["verdict"]["label"] in ("PASS", "FAIL", "NEEDS_REVIEW")
    assert data["steps"][0]["tool"] == "capture_image" and data["totals"]["n_steps"] == len(data["steps"])
    assert len(out.images) == 1
    trace = ok(app.call("get_trace", {"session_id": "loop_sess"}))
    assert trace["trace_id"] == "tr_loop_sess" and trace["closed"] is True
    assert app.call("capture_image", {"session_id": "loop_sess"}).is_error  # finished session


def test_call_result_conversion():
    out = ToolOutput({"x": np.float32(1.5), "arr": (1, 2)}, [("artifact art_1", b"\xff\xd8jpeg")])
    res = InspectAgentMCP.to_call_result(out)
    assert res.is_error is False and res.structured_content == {"x": 1.5, "arr": [1, 2]}
    assert json.loads(res.content[0].text) == {"x": 1.5, "arr": [1, 2]}
    assert isinstance(res.content[2], mtypes.ImageContent) and res.content[2].mime_type == "image/jpeg"
    assert base64.b64decode(res.content[2].data) == b"\xff\xd8jpeg"


# --------------------------------------------------------------------------------------------
# MCP SDK round trips
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_in_memory_client_round_trip(samples_file, mode):
    from mcp import Client

    app, _ = make_app(samples_file)
    server = build_server(app)

    async def scenario() -> dict:
        async with Client(server, mode=mode) as client:
            tools = await client.list_tools()
            names = {t.name for t in tools.tools}
            started = await client.call_tool("start_session", {"sample_id": "board_normal", "preset": "clean"})
            sid = started.structured_content["session_id"]
            cap = await client.call_tool("capture_image", {"session_id": sid, "zoom": 1})
            bad = await client.call_tool("capture_image", {"session_id": "nope"})
            return {"names": names, "schema": next(t for t in tools.tools if t.name == "capture_image").input_schema,
                    "sid": sid, "cap": cap, "bad": bad}

    r = anyio.run(scenario)
    assert {"start_session", "capture_image", "finish", "get_trace"} <= r["names"]
    assert r["schema"]["required"][0] == "session_id"
    cap = r["cap"]
    assert cap.is_error is False
    assert json.loads(cap.content[0].text)["observation"]["capture_id"] == "cap_1"
    images = [c for c in cap.content if isinstance(c, mtypes.ImageContent)]
    assert len(images) == 1 and decode(base64.b64decode(images[0].data)).shape[2] == 3
    assert r["bad"].is_error is True


def test_stdio_subprocess_round_trip():
    """`python -m inspectagent.mcp_server` speaks MCP over stdio (real subprocess, no network)."""
    from mcp import Client, StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=["-m", "inspectagent.mcp_server", "--log-level", "WARNING"],
                                   cwd=str(REPO_ROOT), env={"PYTHONUTF8": "1"})

    async def scenario() -> tuple[set[str], dict]:
        with anyio.fail_after(60):
            async with Client(params) as client:
                tools = await client.list_tools()
                res = await client.call_tool("list_samples", {})
                return {t.name for t in tools.tools}, res.structured_content

    names, samples = anyio.run(scenario)
    assert {"list_samples", "start_session", "capture_image", "run_inspection"} <= names
    assert isinstance(samples["samples"], list)


def test_cli_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--http" in capsys.readouterr().out


def test_mcp_json_example_is_valid():
    cfg = json.loads((REPO_ROOT / ".mcp.json").read_text(encoding="utf-8"))
    server = cfg["mcpServers"]["inspectagent"]
    assert server["args"] == ["-m", "inspectagent.mcp_server"]
    assert ".venv" in server["command"] and not Path(server["command"].split(":-")[-1].rstrip("}")).is_absolute()
