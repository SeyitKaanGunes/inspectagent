"""BedrockPlanner tests with a stubbed Converse client (no network, no AWS credentials)."""

from __future__ import annotations

import copy
import json
from typing import Any

import cv2
import numpy as np
import pytest
from botocore.exceptions import ClientError

from inspectagent.agent import planner_bedrock as pb
from inspectagent.agent.planner_bedrock import BedrockPlanner, pick_bedrock_model
from inspectagent.config import Settings
from inspectagent.types import AnomalyResult, Decision, PartSpec, QualityReport, Region
from tests.synth import make_board

MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# --------------------------------------------------------------------------------------------
# fakes shared with tests/test_mcp_server.py
# --------------------------------------------------------------------------------------------


def jpeg_bytes(h: int = 300, w: int = 450, value: int = 90) -> bytes:
    img = np.full((h, w, 3), value, np.uint8)
    cv2.rectangle(img, (20, 20), (w // 2, h // 2), (0, 200, 255), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def throttle(code: str = "ThrottlingException", status: int = 400) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "slow down"}, "ResponseMetadata": {"HTTPStatusCode": status}}, "Converse")


def tool_response(name: str, args: dict, text: str | None = "", tool_use_id: str = "tu_1", extra: list | None = None,
                  in_tok: int = 100, out_tok: int = 20) -> dict:
    content: list[dict] = []
    if text:
        content.append({"text": text})
    content.append({"toolUse": {"toolUseId": tool_use_id, "name": name, "input": args}})
    for e in extra or []:
        content.append(e)
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": "tool_use",
        "usage": {"inputTokens": in_tok, "outputTokens": out_tok, "totalTokens": in_tok + out_tok},
        "metrics": {"latencyMs": 321},
    }


def text_response(text: str) -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 50, "outputTokens": 5},
        "metrics": {"latencyMs": 100},
    }


class StubBedrockClient:
    """Returns canned Converse responses (or raises queued exceptions); records every request."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.requests: list[dict] = []

    def converse(self, **kwargs: Any) -> dict:
        self.requests.append(copy.deepcopy(kwargs))
        if not self.script:
            raise AssertionError("unexpected Converse call")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeRule:
    """Deterministic stand-in for RulePlanner."""

    def __init__(self) -> None:
        self.calls = 0

    def next_action(self, state: dict, session: Any) -> Decision:
        self.calls += 1
        return Decision(step=len(session.history) + 1, tool="capture_image", args={"cx": 0.5, "cy": 0.5, "zoom": 1.0},
                        rationale="rule: no capture yet -> full view", evidence=[], planner="rule")


class FakeSession:
    """Just enough of InspectionSession for the planner."""

    def __init__(self) -> None:
        self.session_id = "sess_fake"
        self.part = PartSpec(part_id="part_7", category="pcb1", image_path="x.jpg", label=1, mask_path="m.png")
        self.counters: dict[str, float] = {"steps": 0}
        self.artifacts: dict[str, bytes] = {}
        self.history: list[dict] = []
        self.last_observation: dict | None = None
        self.planner_memory: dict = {}

    def record(self, tool: str, args: dict, observation: dict, artifact: bool = False) -> None:
        """Simulate ToolRegistry.execute bookkeeping."""
        self.counters["steps"] += 1
        if artifact:
            aid = f"art_{len(self.artifacts) + 1}"
            self.artifacts[aid] = jpeg_bytes(value=40 + 10 * len(self.artifacts))
            observation = {**observation, "artifact_id": aid, "artifacts": [aid]}
        self.last_observation = observation
        self.history.append({"n": self.counters["steps"], "tool": tool, "args": args, "error": observation.get("error")})

    def planner_state(self) -> dict:
        return {
            "session_id": self.session_id,
            "part_id": self.part.part_id,
            "category": self.part.category,
            "budget": {"steps_used": self.counters["steps"], "max_steps": 14},
            "fused_prob": 0.5,
            "last_observation": self.last_observation,
            "history": self.history[-12:],
        }


def real_registry():
    from inspectagent.agent.tools import default_registry

    return default_registry()


def make_planner(script: list[Any], **kw: Any) -> tuple[BedrockPlanner, StubBedrockClient, list[float]]:
    client = StubBedrockClient(script)
    sleeps: list[float] = []
    kw.setdefault("registry", real_registry())
    kw.setdefault("fallback", FakeRule())
    planner = BedrockPlanner(MODEL, client=client, sleep=sleeps.append, **kw)
    return planner, client, sleeps


def assert_protocol(messages: list[dict]) -> None:
    """Converse invariants: starts with user, roles alternate, toolResults answer the previous toolUses."""
    assert messages[0]["role"] == "user"
    for i, m in enumerate(messages):
        assert m["role"] == ("user" if i % 2 == 0 else "assistant")
        assert m["content"], "empty content"
        results = [b["toolResult"]["toolUseId"] for b in m["content"] if "toolResult" in b]
        if m["role"] == "user" and i > 0:
            prev_uses = [b["toolUse"]["toolUseId"] for b in messages[i - 1]["content"] if "toolUse" in b]
            assert sorted(results) == sorted(prev_uses)
        if m["role"] == "user" and i == 0:
            assert not results


# --------------------------------------------------------------------------------------------
# request construction and decision mapping
# --------------------------------------------------------------------------------------------


def test_tool_config_and_system_prompt_from_registry():
    planner, client, _ = make_planner([tool_response("capture_image", {"cx": 0.5, "cy": 0.5, "zoom": 1})])
    session = FakeSession()
    planner.next_action(session.planner_state(), session)
    req = client.requests[0]
    names = [t["toolSpec"]["name"] for t in req["toolConfig"]["tools"]]
    assert set(names) == set(real_registry().names())
    assert req["toolConfig"]["toolChoice"] == {"any": {}}
    for t in req["toolConfig"]["tools"]:
        schema = t["toolSpec"]["inputSchema"]["json"]
        assert schema["type"] == "object" and pb.RATIONALE_FIELD in schema["properties"]
        assert pb.RATIONALE_FIELD not in schema.get("required", [])
        json.dumps(schema)  # strict JSON
    assert req["inferenceConfig"] == {"maxTokens": 1024, "temperature": 0.0}
    assert req["modelId"] == MODEL
    system = req["system"][0]["text"]
    pol = Settings().policy
    for needle in (f"{pol.max_steps} tool calls", f"{pol.max_captures} captures", "Never claim PASS without reliable evidence",
                   "exactly one tool", "Quality first", "request_human_review", "pcb1"):
        assert needle in system


def test_first_decision_maps_tool_use_and_records_usage():
    args = {"cx": 0.5, "cy": 0.5, "zoom": 1.0, "exposure_ev": 0.0, "light": "ring", "autofocus": False,
            pb.RATIONALE_FIELD: "nothing captured yet"}
    planner, client, _ = make_planner([tool_response("capture_image", args, text="No capture yet; take the full view.")])
    session = FakeSession()
    d = planner.next_action(session.planner_state(), session)
    assert d.tool == "capture_image"
    assert pb.RATIONALE_FIELD not in d.args and d.args["light"] == "ring"
    assert d.rationale == "No capture yet; take the full view."
    assert d.planner == f"bedrock:{MODEL}"
    assert d.step == 1
    assert session.counters["bedrock_calls"] == 1
    assert session.counters["bedrock_input_tokens"] == 100
    assert session.counters["bedrock_output_tokens"] == 20
    assert session.counters["bedrock_latency_ms"] >= 0
    log = planner.call_log(session)
    assert log[-1]["ok"] and log[-1]["input_tokens"] == 100 and log[-1]["service_latency_ms"] == 321
    first = client.requests[0]["messages"]
    assert len(first) == 1 and first[0]["role"] == "user"
    text = first[0]["content"][0]["text"]
    assert "pcb1" in text and "part_7" not in text  # category only: part ids can encode the label
    # the conversation lives in the session
    assert isinstance(session.planner_memory["bedrock"], pb.BedrockConversation)


def test_rationale_falls_back_to_tool_argument_and_ids_become_evidence():
    planner, _, _ = make_planner([tool_response("assess_quality", {"capture_id": "cap_1", pb.RATIONALE_FIELD: "check cap_1 first"}, text=None)])
    session = FakeSession()
    d = planner.next_action(session.planner_state(), session)
    assert d.rationale == "check cap_1 first"
    assert d.args == {"capture_id": "cap_1"}
    assert "cap_1" in d.evidence


def test_ground_truth_never_reaches_the_model():
    planner, client, _ = make_planner([tool_response("capture_image", {})])
    session = FakeSession()
    state = session.planner_state()
    state["ground_truth"] = {"label": 1}
    state["conditions"] = {"glare_strength": 0.9}
    state["part"] = {"part_id": "part_7", "label": 1, "mask_path": "m.png"}
    planner.next_action(state, session)
    dumped = json.dumps(client.requests[0]["messages"], default=str)
    assert "glare_strength" not in dumped and "m.png" not in dumped and "ground_truth" not in dumped


# --------------------------------------------------------------------------------------------
# toolUse -> toolResult protocol
# --------------------------------------------------------------------------------------------


def test_tool_result_threading_with_observation_and_thumbnail():
    planner, client, _ = make_planner([
        tool_response("capture_image", {"cx": 0.5}, text="full view", tool_use_id="tu_A"),
        tool_response("assess_quality", {"capture_id": "cap_1"}, text="check q", tool_use_id="tu_B"),
    ])
    session = FakeSession()
    d1 = planner.next_action(session.planner_state(), session)
    session.record(d1.tool, d1.args, {"capture_id": "cap_1", "mean_luma": 101.5}, artifact=True)
    d2 = planner.next_action(session.planner_state(), session)
    assert d2.tool == "assess_quality" and d2.args == {"capture_id": "cap_1"}

    msgs = client.requests[1]["messages"]
    assert len(msgs) == 3
    assert_protocol(msgs)
    assert msgs[1] == {
        "role": "assistant",
        "content": [{"text": "full view"}, {"toolUse": {"toolUseId": "tu_A", "name": "capture_image", "input": {"cx": 0.5}}}],
    }
    tr = msgs[2]["content"][0]["toolResult"]
    assert tr["toolUseId"] == "tu_A" and tr["status"] == "success"
    payload = tr["content"][0]["json"]
    assert payload["observation"]["capture_id"] == "cap_1"
    assert payload["state"]["budget"]["steps_used"] == 1
    assert "last_observation" not in payload["state"]
    images = [b for b in msgs[2]["content"] if "image" in b]
    assert len(images) == 1 and images[0]["image"]["format"] == "jpeg"
    thumb = cv2.imdecode(np.frombuffer(images[0]["image"]["source"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
    assert thumb is not None and max(thumb.shape[:2]) <= 256


def test_error_observation_is_reported_with_error_status():
    planner, client, _ = make_planner([
        tool_response("assess_quality", {"capture_id": "cap_9"}, tool_use_id="tu_1"),
        tool_response("capture_image", {}, tool_use_id="tu_2"),
    ])
    session = FakeSession()
    d1 = planner.next_action(session.planner_state(), session)
    session.record(d1.tool, d1.args, {"error": "unknown capture_id 'cap_9'", "tool": "assess_quality"})
    planner.next_action(session.planner_state(), session)
    tr = client.requests[1]["messages"][2]["content"][0]["toolResult"]
    assert tr["status"] == "error"
    assert "unknown capture_id" in tr["content"][0]["json"]["observation"]["error"]


def test_only_one_tool_per_turn_extra_calls_answered_with_errors():
    extra = [{"toolUse": {"toolUseId": "tu_extra", "name": "assess_quality", "input": {"capture_id": "cap_1"}}}]
    planner, client, _ = make_planner([
        tool_response("capture_image", {"zoom": 1}, tool_use_id="tu_main", extra=extra),
        tool_response("assess_quality", {"capture_id": "cap_1"}, tool_use_id="tu_next"),
    ])
    session = FakeSession()
    d1 = planner.next_action(session.planner_state(), session)
    assert d1.tool == "capture_image"
    session.record(d1.tool, d1.args, {"capture_id": "cap_1"})
    planner.next_action(session.planner_state(), session)
    msgs = client.requests[1]["messages"]
    assert_protocol(msgs)
    results = {b["toolResult"]["toolUseId"]: b["toolResult"] for b in msgs[2]["content"] if "toolResult" in b}
    assert results["tu_main"]["status"] == "success"
    assert results["tu_extra"]["status"] == "error"
    assert "exactly one tool" in results["tu_extra"]["content"][0]["json"]["error"]


def test_policy_override_is_reported_to_the_model():
    planner, client, _ = make_planner([
        tool_response("finish", {"label": "PASS", "confidence": 0.9, "rationale": "looks fine", "region_ids": []}, tool_use_id="tu_f"),
        tool_response("capture_image", {}, tool_use_id="tu_g"),
    ])
    session = FakeSession()
    d1 = planner.next_action(session.planner_state(), session)
    assert d1.args["rationale"] == "looks fine"  # the tool's own rationale argument is kept
    session.record("request_human_review", {"reason": "policy"}, {"review_id": "rev_1", "status": "pending"})
    planner.next_action(session.planner_state(), session)
    payload = client.requests[1]["messages"][2]["content"][0]["toolResult"]["content"][0]["json"]
    assert payload["policy_override"]["requested_tool"] == "finish"
    assert payload["policy_override"]["executed_tool"] == "request_human_review"


def test_explicit_observe_hook_takes_precedence():
    planner, client, _ = make_planner([tool_response("capture_image", {}, tool_use_id="t1"), tool_response("capture_image", {}, tool_use_id="t2")])
    session = FakeSession()
    planner.next_action(session.planner_state(), session)
    planner.observe({"tool": "capture_image", "args": {}, "observation": {"capture_id": "cap_42"}}, session)
    planner.next_action(session.planner_state(), session)
    payload = client.requests[1]["messages"][2]["content"][0]["toolResult"]["content"][0]["json"]
    assert payload["observation"]["capture_id"] == "cap_42"


# --------------------------------------------------------------------------------------------
# retries and fallback
# --------------------------------------------------------------------------------------------


def test_throttling_is_retried_with_backoff():
    planner, client, sleeps = make_planner([throttle(), throttle("ServiceUnavailableException", 503), tool_response("capture_image", {})])
    session = FakeSession()
    d = planner.next_action(session.planner_state(), session)
    assert d.planner.startswith("bedrock:")
    assert sleeps == [pytest.approx(0.6), pytest.approx(1.2)]
    assert session.counters["bedrock_retries"] == 2
    assert session.counters["bedrock_errors"] == 2
    assert len(client.requests) == 3


def test_fallback_after_retries_then_recovery_keeps_protocol():
    rule = FakeRule()
    planner, client, sleeps = make_planner([
        tool_response("capture_image", {"zoom": 1}, tool_use_id="tu_1"),
        throttle(), throttle(), throttle(),  # step 2: 1 try + 2 retries
        tool_response("assess_quality", {"capture_id": "cap_2"}, tool_use_id="tu_3"),
    ], fallback=rule)
    session = FakeSession()
    d1 = planner.next_action(session.planner_state(), session)
    session.record(d1.tool, d1.args, {"capture_id": "cap_1", "mean_luma": 30.0})
    d2 = planner.next_action(session.planner_state(), session)
    assert d2.planner == "rule(fallback)" and d2.rationale.startswith("[Bedrock fallback: ThrottlingException")
    assert rule.calls == 1 and len(sleeps) == 2
    assert session.counters["bedrock_fallbacks"] == 1
    conv = planner.conversation(session)
    assert conv.messages[-1]["role"] == "assistant"  # failed user turn rolled back
    session.record(d2.tool, d2.args, {"capture_id": "cap_2", "mean_luma": 110.0})
    d3 = planner.next_action(session.planner_state(), session)
    assert d3.planner.startswith("bedrock:") and d3.args == {"capture_id": "cap_2"}
    msgs = client.requests[-1]["messages"]
    assert_protocol(msgs)
    assert len(msgs) == 3
    blocks = msgs[2]["content"]
    assert blocks[0]["toolResult"]["toolUseId"] == "tu_1"
    assert blocks[0]["toolResult"]["content"][0]["json"]["observation"]["capture_id"] == "cap_1"
    note = next(b["text"] for b in blocks if "text" in b and "fallback" in b["text"])
    assert "capture_image" in note and "cap_2" in note


def test_non_retryable_error_falls_back_immediately():
    planner, client, sleeps = make_planner([throttle("AccessDeniedException", 403)])
    session = FakeSession()
    d = planner.next_action(session.planner_state(), session)
    assert d.planner == "rule(fallback)" and "AccessDeniedException" in d.rationale
    assert sleeps == [] and len(client.requests) == 1
    assert planner.conversation(session).messages == []


def test_bedrock_disabled_after_consecutive_failures():
    planner, client, _ = make_planner([throttle("ValidationException"), throttle("ValidationException")], max_consecutive_failures=2)
    session = FakeSession()
    for _ in range(3):
        d = planner.next_action(session.planner_state(), session)
        assert d.planner == "rule(fallback)"
        session.record(d.tool, d.args, {"capture_id": f"cap_{session.counters['steps'] + 1}"})
    assert len(client.requests) == 2  # third step did not call Bedrock
    assert "disabled" in planner.conversation(session).disabled_reason
    assert session.counters["bedrock_fallbacks"] == 3


def test_missing_tool_use_gets_one_nudge():
    planner, client, _ = make_planner([text_response("I think we should capture."), tool_response("capture_image", {})])
    session = FakeSession()
    d = planner.next_action(session.planner_state(), session)
    assert d.tool == "capture_image" and d.planner.startswith("bedrock:")
    msgs = client.requests[1]["messages"]
    assert_protocol(msgs)
    assert "exactly one tool call" in msgs[-1]["content"][0]["text"]


def test_missing_tool_use_twice_falls_back_and_rolls_back():
    planner, client, _ = make_planner([text_response("hmm"), text_response("still thinking")])
    session = FakeSession()
    d = planner.next_action(session.planner_state(), session)
    assert d.planner == "rule(fallback)" and "NoToolUse" in d.rationale
    assert planner.conversation(session).messages == []


def test_auto_model_resolution_failure_falls_back_once(monkeypatch):
    calls = {"n": 0}

    def boom(region):
        calls["n"] += 1
        raise RuntimeError("no credentials")

    monkeypatch.setattr(pb, "pick_bedrock_model", boom)
    planner = BedrockPlanner("auto", client=StubBedrockClient([]), registry=real_registry(), fallback=FakeRule())
    session = FakeSession()
    for _ in range(2):
        d = planner.next_action(session.planner_state(), session)
        assert d.planner == "rule(fallback)" and "no credentials" in d.rationale
    assert calls["n"] == 1


def test_auto_model_resolution_success(monkeypatch):
    monkeypatch.setattr(pb, "pick_bedrock_model", lambda region: "us.amazon.nova-pro-v1:0")
    client = StubBedrockClient([tool_response("capture_image", {})])
    planner = BedrockPlanner.from_settings(Settings(), client=client, registry=real_registry(), fallback=FakeRule())
    session = FakeSession()
    d = planner.next_action(session.planner_state(), session)
    assert d.planner == "bedrock:us.amazon.nova-pro-v1:0"
    assert client.requests[0]["modelId"] == "us.amazon.nova-pro-v1:0"


def test_default_fallback_is_the_rule_planner():
    pytest.importorskip("inspectagent.agent.planner_rule")
    from inspectagent.agent.planner_rule import RulePlanner

    planner = BedrockPlanner(MODEL, client=StubBedrockClient([]))
    assert isinstance(planner.fallback, RulePlanner)


# --------------------------------------------------------------------------------------------
# bounded history
# --------------------------------------------------------------------------------------------


def test_history_is_trimmed_but_keeps_first_message_and_protocol():
    n_steps = 7
    script = [tool_response("capture_image", {"zoom": 1}, tool_use_id=f"tu_{i}") for i in range(n_steps)]
    planner, client, _ = make_planner(script, max_turn_pairs=2, max_images_in_history=1)
    session = FakeSession()
    for i in range(n_steps):
        d = planner.next_action(session.planner_state(), session)
        session.record(d.tool, d.args, {"capture_id": f"cap_{i + 1}"}, artifact=True)
    first_user = client.requests[0]["messages"][0]
    for req in client.requests:
        msgs = req["messages"]
        assert len(msgs) <= 1 + 2 * 2
        assert msgs[0] == first_user
        assert_protocol(msgs)
        n_images = sum(1 for m in msgs for b in m["content"] if "image" in b)
        assert n_images <= 1
    last = client.requests[-1]["messages"]
    assert any("trimmed" in b.get("text", "") for b in last[-1]["content"])
    assert any("older image omitted" in b.get("text", "") for m in last for b in m["content"])
    conv = planner.conversation(session)
    assert conv.trimmed_pairs >= 3 and len(conv.messages) <= 1 + 2 * 2 + 1


# --------------------------------------------------------------------------------------------
# model selection
# --------------------------------------------------------------------------------------------


class StubControlClient:
    def __init__(self, profiles: list[dict], models: list[dict], fail_profiles: bool = False) -> None:
        self.profiles, self.models, self.fail_profiles = profiles, models, fail_profiles

    def list_inference_profiles(self, **kw: Any) -> dict:
        if self.fail_profiles:
            raise throttle("AccessDeniedException", 403)
        if kw.get("nextToken") == "p2":
            return {"inferenceProfileSummaries": self.profiles[1:]}
        return {"inferenceProfileSummaries": self.profiles[:1], "nextToken": "p2"}

    def list_foundation_models(self, **kw: Any) -> dict:
        return {"modelSummaries": self.models}


def _profile(pid: str, status: str = "ACTIVE") -> dict:
    return {"inferenceProfileId": pid, "status": status, "type": "SYSTEM_DEFINED"}


def _fm(mid: str, types=("ON_DEMAND",), status="ACTIVE", modalities=("TEXT", "IMAGE")) -> dict:
    return {"modelId": mid, "inferenceTypesSupported": list(types), "modelLifecycle": {"status": status},
            "inputModalities": list(modalities)}


def test_pick_prefers_newest_sonnet_in_region_geo():
    ctl = StubControlClient(
        [
            _profile("us.anthropic.claude-3-5-sonnet-20241022-v2:0"),
            _profile("us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
            _profile("global.anthropic.claude-sonnet-4-5-20250929-v1:0"),
            _profile("eu.anthropic.claude-sonnet-4-5-20250929-v1:0"),
            _profile("us.anthropic.claude-haiku-4-5-20251001-v1:0"),
            _profile("us.anthropic.claude-opus-4-1-20250805-v1:0"),
            _profile("us.amazon.nova-pro-v1:0"),
        ],
        [_fm("anthropic.claude-3-haiku-20240307-v1:0"), _fm("amazon.titan-embed-text-v2:0")],
    )
    assert pick_bedrock_model("us-east-1", bedrock_client=ctl) == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    ranked = pb.list_bedrock_model_candidates("us-east-1", bedrock_client=ctl)
    assert all("opus" not in c["model_id"] and "titan" not in c["model_id"] for c in ranked)
    assert ranked[-1]["family"] == "nova"


def test_pick_falls_back_to_haiku_then_nova():
    ctl = StubControlClient([_profile("us.amazon.nova-lite-v1:0"), _profile("us.anthropic.claude-3-5-haiku-20241022-v1:0")],
                            [_fm("amazon.nova-micro-v1:0", modalities=("TEXT",))])
    assert pick_bedrock_model("us-west-2", bedrock_client=ctl) == "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    ctl = StubControlClient([_profile("eu.amazon.nova-pro-v1:0"), _profile("eu.amazon.nova-lite-v1:0")], [])
    assert pick_bedrock_model("eu-central-1", bedrock_client=ctl) == "eu.amazon.nova-pro-v1:0"
    assert pick_bedrock_model("eu-central-1", bedrock_client=StubControlClient([], [])) is None


def test_pick_uses_foundation_models_when_profiles_unavailable_and_verifies():
    ctl = StubControlClient([], [_fm("anthropic.claude-3-5-sonnet-20240620-v1:0"), _fm("anthropic.claude-3-haiku-20240307-v1:0"),
                                  _fm("anthropic.claude-3-sonnet-20240229-v1:0", status="LEGACY")], fail_profiles=True)

    class Runtime:
        def __init__(self) -> None:
            self.tried: list[str] = []

        def converse(self, **kw: Any) -> dict:
            self.tried.append(kw["modelId"])
            if "sonnet" in kw["modelId"]:
                raise throttle("AccessDeniedException", 403)
            return text_response("OK")

    rt = Runtime()
    picked = pick_bedrock_model("us-east-1", bedrock_client=ctl, runtime_client=rt, verify=True)
    assert picked == "anthropic.claude-3-haiku-20240307-v1:0"
    assert rt.tried[0] == "anthropic.claude-3-5-sonnet-20240620-v1:0"  # ACTIVE sonnet ranked before LEGACY one


def test_classify_model_versions():
    c = pb._classify_model
    assert c("us.anthropic.claude-sonnet-4-5-20250929-v1:0")["version"] == (4, 5)
    assert c("anthropic.claude-sonnet-4-20250514-v1:0")["version"] == (4, 0)
    assert c("anthropic.claude-3-7-sonnet-20250219-v1:0")["version"] == (3, 7)
    assert c("anthropic.claude-haiku-4-5-20251001-v1:0")["family"] == "haiku"
    assert c("us.amazon.nova-2-lite-v1:0") == {"provider": "amazon", "family": "nova", "version": (2, 0), "date": 0, "tier": 2}
    assert c("amazon.nova-micro-v1:0") is None and c("cohere.command-r-v1:0") is None


# --------------------------------------------------------------------------------------------
# end to end with the real session, registry and human-control policy (stub Bedrock only)
# --------------------------------------------------------------------------------------------


class FakeStatDetector:
    """Stand-in for StatTemplateModel.predict with a fixed probability (and optional region)."""

    def __init__(self, prob: float, with_region: bool = False, reliable: bool = True) -> None:
        self.prob, self.with_region, self.reliable = prob, with_region, reliable

    def predict(self, capture: Any, quality: Any = None, id_prefix: str = "") -> AnomalyResult:
        h, w = capture.image.shape[:2]
        amap = np.zeros((h, w), np.float32)
        regions = []
        if self.with_region:
            amap[100:130, 200:260] = 6.0
            regions.append(Region(region_id=f"{id_prefix}1", capture_id=capture.capture_id, source="fast",
                                  bbox_capture=(200, 100, 60, 30), bbox_scene=(230.0, 115.0, 70.0, 35.0), peak=6.0, mean=4.0,
                                  area_px=1800, area_scene_px=2450.0))
        return AnomalyResult(capture_id=capture.capture_id, detector="fast", score=1.0, prob=self.prob, regions=regions,
                             reliable=self.reliable, registration={"method": "fake", "ok": self.reliable, "ncc": 0.9},
                             latency_ms=1.0, anomaly_map=amap, valid_mask=np.full((h, w), 255, np.uint8))


def make_real_session(prob: float = 0.03, with_region: bool = False, session_id: str = "sess_real", label: int = 0):
    """Real InspectionSession over a synthetic board with fake quality/detector backends."""
    from inspectagent.agent.session import InspectionSession, VisionBackend
    from inspectagent.models_io import CategoryBundle
    from inspectagent.vision.camera import VirtualCamera, make_conditions

    scene = make_board(0)

    def assess(cap: Any, ref: Any, qcfg: Any) -> QualityReport:
        return QualityReport(capture_id=cap.capture_id, sharpness=120.0, sharpness_ratio=1.0, mean_luma=float(cap.image.mean()),
                             clipped_frac=0.0, crushed_frac=0.0, noise_sigma=1.5, glare_frac=0.0, glare_boxes=[], issues=[],
                             suggested_exposure_delta=0.0, ok=True)

    def measure(region: Any, cap: Any, amap: Any, px_per_mm: float) -> dict:
        return {"area_mm2": 2.0, "length_mm": 3.0, "width_mm": 1.0, "elongation": 3.0, "orientation_deg": 0.0,
                "contrast": 0.5, "type_guess": "spot"}

    backend = VisionBackend(make_camera=lambda sc, cond, cfg, seed: VirtualCamera(sc, cond, cfg, seed=seed),
                            assess_quality=assess, measure_region=measure, imread=lambda p: scene)
    bundle = CategoryBundle(category="pcb1", stat=FakeStatDetector(prob, with_region), deep=None)
    part = PartSpec(part_id="synth_board", category="pcb1", image_path="synthetic.png", label=label,
                    conditions=make_conditions("clean", np.random.default_rng(0)))
    return InspectionSession(part, bundle, seed=0, scene=scene, backend=backend, session_id=session_id)


def test_end_to_end_with_real_session_registry_and_policy():
    from inspectagent.agent.policy import HumanControlPolicy

    registry = real_registry()
    policy = HumanControlPolicy()
    script = [
        tool_response("capture_image", {"cx": 0.5, "cy": 0.5, "zoom": 1, "exposure_ev": 0, "light": "ring", "autofocus": False},
                      text="Start with the full view.", tool_use_id="t1"),
        tool_response("assess_quality", {"capture_id": "cap_1"}, text="Check quality of cap_1.", tool_use_id="t2"),
        tool_response("detect_anomalies", {"capture_id": "cap_1", "detector": "fast"}, text="q_1 ok -> fast detector.", tool_use_id="t3"),
        tool_response("finish", {"label": "PASS", "confidence": 0.95, "rationale": "det_1 p=0.03, reliable", "region_ids": []},
                      text="det_1: p=0.03 <= 0.15 and reliable -> PASS.", tool_use_id="t4"),
    ]
    planner, client, _ = make_planner(script, registry=registry)
    session = make_real_session(prob=0.03)
    executed = []
    for _ in range(6):
        decision = planner.next_action(session.planner_state(), session)
        decision, overridden = policy.vet(decision, session)
        assert not overridden
        obs = registry.execute(session, decision.tool, decision.args)
        assert "error" not in obs, obs
        executed.append(decision.tool)
        if registry.is_terminal(decision.tool):
            break
    assert executed == ["capture_image", "assess_quality", "detect_anomalies", "finish"]
    verdict = policy.final_verdict(session)
    assert verdict.label == "PASS" and verdict.action == "route_pass"
    # the real observations were threaded back with matching ids and images
    msgs = client.requests[-1]["messages"]
    assert_protocol(msgs)
    tr = msgs[-1]["content"][0]["toolResult"]
    assert tr["toolUseId"] == "t3"
    assert tr["content"][0]["json"]["observation"]["result_id"] == "det_1"
    assert sum(1 for b in msgs[-1]["content"] if "image" in b) == 1
    q_payload = client.requests[2]["messages"][-1]["content"][0]["toolResult"]["content"][0]["json"]
    assert q_payload["observation"]["quality_id"] == "q_1" and q_payload["observation"]["ok"] is True
    assert session.counters["bedrock_calls"] == 4 and session.counters["bedrock_input_tokens"] == 400


def _loop_kwargs(prob: float = 0.03) -> dict:
    session = make_real_session(prob=prob)
    return {"bundle": session.bundle, "scene": session.scene, "backend": session.vision, "seed": 0}


def test_run_inspection_loop_with_bedrock_planner():
    from inspectagent.agent.loop import run_inspection

    script = [
        tool_response("capture_image", {"cx": 0.5, "cy": 0.5, "zoom": 1}, text="Full view first.", tool_use_id="a1"),
        tool_response("assess_quality", {"capture_id": "cap_1"}, text="Quality of cap_1.", tool_use_id="a2"),
        tool_response("detect_anomalies", {"capture_id": "cap_1", "detector": "fast"}, text="q_1 ok -> fast.", tool_use_id="a3"),
        tool_response("finish", {"label": "PASS", "confidence": 0.95, "rationale": "det_1 p=0.03", "region_ids": []},
                      text="det_1 p=0.03 reliable -> PASS.", tool_use_id="a4"),
    ]
    planner, client, _ = make_planner(script)
    part = make_real_session().part
    trace, session = run_inspection(part, planner=planner, session_id="loop_bedrock", **_loop_kwargs())
    assert [s.decision.tool for s in trace.steps] == ["capture_image", "assess_quality", "detect_anomalies", "finish"]
    assert all(s.decision.planner == f"bedrock:{MODEL}" for s in trace.steps)
    assert trace.steps[2].decision.rationale == "q_1 ok -> fast."
    assert trace.verdict.label == "PASS" and trace.verdict.action == "route_pass"
    assert trace.planner == "bedrock"
    assert session.counters["bedrock_calls"] == 4 and session.counters["bedrock_output_tokens"] == 80
    assert_protocol(client.requests[-1]["messages"])


def test_run_inspection_loop_falls_back_to_real_rule_planner_when_bedrock_is_down():
    pytest.importorskip("inspectagent.agent.planner_rule")
    from inspectagent.agent.loop import run_inspection

    client = StubBedrockClient([throttle() for _ in range(6)])
    sleeps: list[float] = []
    planner = BedrockPlanner(MODEL, client=client, registry=real_registry(), sleep=sleeps.append)  # default RulePlanner fallback
    part = make_real_session().part
    trace, session = run_inspection(part, planner=planner, session_id="loop_fallback", **_loop_kwargs())
    assert len(client.requests) == 6  # 2 failed steps x (1 try + 2 retries), then Bedrock is disabled
    assert sleeps == [pytest.approx(0.6), pytest.approx(1.2)] * 2
    assert trace.steps and all(s.decision.planner == "rule(fallback)" for s in trace.steps)
    assert trace.steps[0].decision.tool == "capture_image"
    assert trace.verdict is not None and trace.verdict.label in ("PASS", "NEEDS_REVIEW")
    assert session.counters["bedrock_fallbacks"] == len(trace.steps)


def test_label_bearing_part_id_never_reaches_the_model():
    """Regression: part ids built from dataset paths ('pcb1_anomaly_012') leaked the ground truth into Converse."""
    from inspectagent.agent.loop import run_inspection
    from inspectagent.eval_utils import part_id_from_relpath

    pid = part_id_from_relpath("pcb1/Data/Images/Anomaly/012.JPG")
    assert pid == "pcb1_anomaly_012"
    script = [
        tool_response("capture_image", {"cx": 0.5, "cy": 0.5, "zoom": 1}, text="Full view.", tool_use_id="p1"),
        tool_response("assess_quality", {"capture_id": "cap_1"}, text="Quality.", tool_use_id="p2"),
        tool_response("detect_anomalies", {"capture_id": "cap_1", "detector": "fast"}, text="Detect.", tool_use_id="p3"),
        tool_response("request_human_review", {"reason": "check", "proposed_label": "PASS"}, text="Review.", tool_use_id="p4"),
    ]
    planner, client, _ = make_planner(script)
    base = make_real_session()
    part = PartSpec(part_id=pid, category="pcb1", image_path="pcb1/Data/Images/Anomaly/012.JPG", label=1,
                    mask_path="pcb1/Data/Masks/Anomaly/012.png", conditions=base.part.conditions)
    trace, _ = run_inspection(part, planner=planner, session_id="leak_check", **_loop_kwargs())
    assert len(client.requests) == 4 and trace.part_id == pid
    for req in client.requests:
        dumped = json.dumps(req["messages"], default=str)
        assert pid not in dumped and "_anomaly_" not in dumped and "Anomaly/" not in dumped


def test_deadline_skips_model_calls_and_falls_back(monkeypatch):
    """Regression: no wall-clock bound - a slow Bedrock could outlive the 60 s Lambda without a verdict."""
    now = {"t": 1000.0}
    rule = FakeRule()
    client = StubBedrockClient([tool_response("capture_image", {}, tool_use_id="d1")])
    planner = BedrockPlanner(MODEL, client=client, registry=real_registry(), fallback=rule, sleep=lambda s: None,
                             deadline=1000.0 + 30.0, clock=lambda: now["t"])
    session = FakeSession()
    d1 = planner.next_action(session.planner_state(), session)
    assert d1.planner.startswith("bedrock:")
    session.record(d1.tool, d1.args, {"capture_id": "cap_1"})
    now["t"] = 1000.0 + 25.0  # 5 s before the deadline: less than reserve + min call time
    d2 = planner.next_action(session.planner_state(), session)
    assert d2.planner == "rule(fallback)" and "DeadlineExceeded" in d2.rationale
    assert len(client.requests) == 1 and "deadline" in planner.conversation(session).disabled_reason
    session.record(d2.tool, d2.args, {"capture_id": "cap_2"})
    assert planner.next_action(session.planner_state(), session).planner == "rule(fallback)"  # stays disabled
    assert session.counters["bedrock_deadline_skips"] >= 1


def test_retry_is_skipped_when_it_cannot_finish_before_the_deadline():
    now = {"t": 0.0}
    sleeps: list[float] = []
    client = StubBedrockClient([throttle(), tool_response("capture_image", {})])
    planner = BedrockPlanner(MODEL, client=client, registry=real_registry(), fallback=FakeRule(), sleep=sleeps.append,
                             deadline=9.5, deadline_reserve_s=6.0, min_call_s=3.0, clock=lambda: now["t"])
    session = FakeSession()
    d = planner.next_action(session.planner_state(), session)
    assert d.planner == "rule(fallback)" and "no time left for a retry" in d.rationale
    assert sleeps == [] and len(client.requests) == 1


def test_read_timeout_is_short():
    planner = BedrockPlanner(MODEL, region="us-east-1", registry=real_registry(), fallback=FakeRule())
    client = planner._get_client()
    assert client.meta.config.read_timeout <= 20 and client.meta.config.connect_timeout <= 5


def test_budget_guard_refuses_calls_and_disables_bedrock():
    """Spend control: a shared call budget (e.g. daily DynamoDB counter) gates every Converse call."""
    allowed = {"n": 1}

    def guard() -> bool:
        if allowed["n"] <= 0:
            return False
        allowed["n"] -= 1
        return True

    client = StubBedrockClient([tool_response("capture_image", {}, tool_use_id="b1")])
    planner = BedrockPlanner(MODEL, client=client, registry=real_registry(), fallback=FakeRule(), budget_guard=guard)
    session = FakeSession()
    assert planner.next_action(session.planner_state(), session).planner.startswith("bedrock:")
    session.record("capture_image", {}, {"capture_id": "cap_1"})
    d = planner.next_action(session.planner_state(), session)
    assert d.planner == "rule(fallback)" and "BudgetExhausted" in d.rationale and len(client.requests) == 1
    assert session.counters["bedrock_budget_refusals"] == 1


def test_specular_guard_rewrites_darkening_recapture():
    """A clipping-only overexposure under ring light is a specular reflection: diffuse +0.75 EV, not darker."""
    from types import SimpleNamespace

    from inspectagent.types import CaptureSettings

    planner, client, _ = make_planner([tool_response("capture_image", {"cx": 0.5, "cy": 0.5, "zoom": 1, "exposure_ev": -1.9,
                                                                        "light": "ring"}, text="EV -1.9 (suggested).")])
    session = FakeSession()
    q = QualityReport("cap_1", 120.0, 1.0, 128.0, 0.071, 0.0, 2.0, 0.0, [], ["overexposed"], -1.9, False)
    session.captures = {"cap_1": SimpleNamespace(settings=CaptureSettings())}
    session.quality = {"cap_1": q}
    session.quality_ids = {"cap_1": "q_1"}
    session.settings = Settings()
    d = planner.next_action(session.planner_state(), session)
    assert d.args["light"] == "diffuse" and d.args["exposure_ev"] == pytest.approx(0.75)
    assert d.rationale.startswith("[guard: q_1 clipping-only overexposure") and "cap_1" in d.evidence
    # a global overexposure (mean luma too high) is left to the model's exposure correction
    planner2, _, _ = make_planner([tool_response("capture_image", {"exposure_ev": -1.0}, text="darker")])
    session.quality = {"cap_1": QualityReport("cap_1", 120.0, 1.0, 205.0, 0.071, 0.0, 2.0, 0.0, [], ["overexposed"], -1.0, False)}
    session.planner_memory = {}
    assert planner2.next_action(session.planner_state(), session).args == {"exposure_ev": -1.0}


def test_identical_observations_on_consecutive_steps_are_not_stale():
    err = {"error": "unknown capture_id 'cap_9'", "tool": "assess_quality"}
    planner, client, _ = make_planner([tool_response("assess_quality", {"capture_id": "cap_9"}, tool_use_id=f"t{i}") for i in range(3)])
    session = FakeSession()
    for _ in range(3):
        d = planner.next_action(session.planner_state(), session)
        session.record(d.tool, d.args, dict(err))
    for req in client.requests[1:]:
        payload = req["messages"][-1]["content"][0]["toolResult"]["content"][0]["json"]
        assert payload["observation"]["error"] == err["error"]
    # no new step since the last call -> the observation is not resent as fresh
    rec = pb.find_last_step(session.planner_state(), session)
    assert rec.step == 3 and rec.tool == "assess_quality"
