"""Web API tests (fastapi.testclient) with a fake agent loop, fake model bundles and temp LocalStorage.

Everything runs offline: no models, no VisA data, no AWS. The real camera ``make_conditions`` is used when
``inspectagent.vision.camera`` is importable, otherwise a local fake.
"""

from __future__ import annotations

import base64
import functools
import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import inspectagent.api.app as app_mod
from inspectagent.api.app import (
    ARTIFACT_MAX_SIDE,
    MAX_UPLOAD_BYTES,
    RESPONSE_BUDGET_BYTES,
    create_app,
    encode_artifacts,
    eval_tables,
    sniff_image,
)
from inspectagent.config import Settings
from inspectagent.types import Decision, NuisanceConditions, Region, Step, Trace, Verdict
from tests.synth import make_board as _make_board_uncached

OUTCOME = {"clean": "PASS", "nominal": "PASS", "bright": "PASS", "glare": "PASS", "defocus": "PASS", "harsh": "NEEDS_REVIEW", "dark": "FAIL"}
ACTION = {"PASS": "route_pass", "FAIL": "route_quarantine", "NEEDS_REVIEW": "hold_for_review"}


@functools.lru_cache(maxsize=32)
def _board_cached(seed: int) -> np.ndarray:
    img = _make_board_uncached(seed)
    img.setflags(write=False)
    return img


def make_board(seed: int) -> np.ndarray:
    """Synthetic board (cached: generation costs ~0.15 s; callers get a writable copy)."""
    return _board_cached(seed).copy()


def _jpeg(img: np.ndarray, q: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    assert ok
    return buf.tobytes()


def _png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _decode_data_url(url: str) -> np.ndarray:
    assert url.startswith("data:image/jpeg;base64,")
    img = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    return img


class FakeLoop:
    """Stand-in for inspectagent.agent.loop.run_inspection (SPEC §3.6 signature)."""

    def __init__(self, n_extra_artifacts: int = 0, noisy: bool = False, fail: bool = False) -> None:
        self.calls: list[dict] = []
        self.n_extra = n_extra_artifacts
        self.noisy = noisy
        self.fail = fail

    def __call__(self, part, bundle=None, planner=None, policy=None, settings=None, seed=0, on_step=None, scene=None):
        if self.fail:
            raise RuntimeError("simulated detector crash")
        preset = part.conditions.preset
        self.calls.append({"part": part, "seed": seed, "preset": preset, "planner": planner, "scene_shape": None if scene is None else scene.shape, "bundle": bundle})
        label = OUTCOME.get(preset, "PASS")
        board = make_board(seed)
        artifacts = {
            "art_1": _jpeg(cv2.resize(board, (1000, 667))),  # larger than 512 px -> must be thumbnailed
            "art_2": _jpeg(cv2.resize(board, (300, 200))),  # small JPEG -> passed through
            "art_3": _jpeg(cv2.applyColorMap(cv2.cvtColor(cv2.resize(board, (768, 512)), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)),
        }
        rng = np.random.default_rng(seed)
        extra_ids = []
        for i in range(self.n_extra):
            aid = f"art_x{i}"
            img = rng.integers(0, 256, (683, 1024, 3), dtype=np.uint8) if self.noisy else cv2.resize(board, (1024, 683))
            artifacts[aid] = _jpeg(img, 95)
            extra_ids.append(aid)
        region = Region("reg_3_1", "cap_1", "fast", (100, 120, 30, 20), (117.2, 140.6, 35.2, 23.4), 6.1, 4.2, 600, 820.0)
        prob = {"PASS": 0.04, "FAIL": 0.97, "NEEDS_REVIEW": 0.55}[label]
        steps = [
            Step(1, Decision(1, "capture_image", {"cx": 0.5, "cy": 0.5, "zoom": 1.0, "exposure_ev": 0.0, "light": "ring", "autofocus": False},
                             "No capture yet → full view at EV 0", [], "rule"),
                 {"capture_id": "cap_1", "window": [0, 0, 900, 600], "size": [768, 512], "sim_time_ms": 145.0, "mean_luma": 112.3},
                 ["art_1"], 3.1, 145.0),
            Step(2, Decision(2, "assess_quality", {"capture_id": "cap_1"}, "cap_1 not assessed", ["cap_1"], "rule"),
                 {"capture_id": "cap_1", "ok": True, "issues": [], "sharpness_ratio": 0.93}, ["art_2"], 1.2, 0.0),
            Step(3, Decision(3, "detect_anomalies", {"capture_id": "cap_1", "detector": "fast"}, "q_1 ok → fast detector", ["q_1"], "rule"),
                 {"capture_id": "cap_1", "detector": "fast", "prob": prob, "regions": [region.to_dict()], "reliable": True},
                 ["art_3", *extra_ids], 8.4, 0.0),
        ]
        if label == "NEEDS_REVIEW":
            steps.append(Step(4, Decision(4, "request_human_review", {"reason": "p=0.55 ambiguous", "proposed_label": "FAIL", "confidence": 0.55},
                                          "det_1: p=0.55 between 0.15 and 0.90", ["det_1"], "rule"), {"review_id": "rev_1"}, [], 0.1, 0.0))
        else:
            steps.append(Step(4, Decision(4, "finish", {"label": label, "confidence": 0.9, "rationale": "done", "region_ids": []},
                                          f"det_1: p={prob}", ["det_1"], "rule"), {"label": label}, [], 0.1, 0.0))
        verdict = Verdict(
            label=label, confidence=0.9 if label != "NEEDS_REVIEW" else 0.55, rationale=f"fused p={prob}",
            defect_regions=[region] if label != "PASS" else [], action=ACTION[label],
            review_id="rev_1" if label == "NEEDS_REVIEW" else None, prob_anomalous=prob,
        )
        trace = Trace(
            trace_id=f"tr-{part.part_id}-{seed}", part_id=part.part_id, category=part.category, planner="rule",
            started_at="2026-10-09T12:00:00Z", steps=steps, verdict=verdict,
            totals={"n_steps": len(steps), "n_captures": 1, "n_deep": 0, "compute_ms": 12.8, "sim_time_ms": 145.0, "cycle_time_ms": 157.8},
            ground_truth={"label": part.label} if part.label is not None else None,
            conditions=part.conditions.to_dict(),
        )
        return trace, SimpleNamespace(artifacts=artifacts, session_id=trace.trace_id)


def _fake_conditions(preset: str, rng: np.random.Generator) -> NuisanceConditions:
    if preset not in OUTCOME:
        raise ValueError(f"unknown preset {preset!r}")
    return NuisanceConditions(ambient_ev=float(rng.uniform(-0.2, 0.2)), preset=preset)


@pytest.fixture()
def env(tmp_path: Path, monkeypatch):
    samples = tmp_path / "samples"
    (samples / "pcb1").mkdir(parents=True)
    (samples / "pcb2").mkdir(parents=True)
    _ = (samples / "pcb1" / "normal.jpg").write_bytes(_jpeg(make_board(1)))
    _ = (samples / "pcb1" / "defect.jpg").write_bytes(_jpeg(make_board(2)))
    _ = (samples / "pcb2" / "other.jpg").write_bytes(_jpeg(make_board(3)))
    entries = [
        {"id": "pcb1_normal", "category": "pcb1", "label": 0, "image": "pcb1/normal.jpg", "suggested_preset": "nominal", "note": "normal part"},
        {"id": "pcb1_defect", "category": "pcb1", "label": 1, "image": "pcb1/defect.jpg", "mask": None, "defect_type": "scratch"},
        {"id": "pcb2_other", "category": "pcb2", "label": "normal", "image": "pcb2/other.jpg"},
        {"id": "ghost", "category": "pcb1", "label": 0, "image": "pcb1/missing.jpg"},
    ]
    (samples / "samples.json").write_text(json.dumps(entries), encoding="utf-8")
    results = tmp_path / "results"
    (results / "figures").mkdir(parents=True)
    (results / "figures" / "auroc_by_system.png").write_bytes(_png(np.full((40, 60, 3), 200, np.uint8)))
    (results / "figures" / "notes.txt").write_text("not a figure", encoding="utf-8")
    summary = {
        "generated_by": "scripts/evaluate.py",
        "n_eval": 120,
        "systems": {"single_fast": {"auroc": 0.91, "escape_rate": 0.08}, "agent_rule": {"auroc": 0.95, "escape_rate": 0.03}},
        "per_category": {"pcb1": {"nominal": {"auroc": 0.97}, "harsh": {"auroc": 0.9}}},
        "rows": [{"category": "pcb1", "system": "agent_rule", "f1": 0.9, "tools": ["a", "b"]}],
    }
    (results / "eval_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    settings = Settings(samples_dir=samples, results_dir=results, models_dir=tmp_path / "models",
                        local_storage_dir=tmp_path / "runs", storage="local", bedrock_model_id="")
    loop = FakeLoop()
    real_make_planner = app_mod.make_planner

    def fake_make_planner(name, s):
        if name == "bedrock":
            return real_make_planner(name, s)  # exercises the "not configured" 400 path
        return f"planner:{name}"

    monkeypatch.setattr(app_mod, "run_inspection", loop)
    monkeypatch.setattr(app_mod, "available_categories", lambda models_dir=None: ["pcb1"])
    monkeypatch.setattr(app_mod, "load_bundle", lambda category, models_dir=None, with_deep=True: SimpleNamespace(
        category=category, stat=SimpleNamespace(scene_size=(900, 600)), deep=None, meta={}))
    monkeypatch.setattr(app_mod, "make_planner", fake_make_planner)
    try:
        importlib.import_module("inspectagent.vision.camera")
    except ImportError:
        monkeypatch.setattr(app_mod, "make_conditions", _fake_conditions)
        monkeypatch.setattr(app_mod, "presets", lambda: tuple(OUTCOME))
    app = create_app(settings=settings)
    client = TestClient(app)
    return SimpleNamespace(client=client, app=app, loop=loop, settings=settings, tmp=tmp_path, service=app.state.service)


def _inspect(client, sample_id="pcb1_normal", preset="nominal", seed=0, planner="rule"):
    return client.post("/api/inspect", json={"sample_id": sample_id, "preset": preset, "seed": seed, "planner": planner})


# ------------------------------------------------------------------------------------------- basics


def test_healthz(env) -> None:
    r = env.client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["version"]
    assert body["cv"]["opencv_version"].startswith("5")
    assert "x-response-time-ms" in {k.lower() for k in r.headers}


def test_cors_open(env) -> None:
    r = env.client.options("/api/samples", headers={"Origin": "https://example.org", "Access-Control-Request-Method": "GET"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "*"


def test_static_index_served(env) -> None:
    r = env.client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "InspectAgent" in r.text and "Agentic visual inspection with OpenCV" in r.text
    for asset in ("/static/app.js", "/static/style.css"):
        assert env.client.get(asset).status_code == 200
    assert env.client.get("/api/openapi.json").status_code == 200


# ------------------------------------------------------------------------------------------ samples


def test_samples_listing_and_images(env) -> None:
    r = env.client.get("/api/samples")
    assert r.status_code == 200
    body = r.json()
    ids = [s["id"] for s in body["samples"]]
    assert ids == ["pcb1_normal", "pcb1_defect", "pcb2_other"]  # entry with a missing image is skipped
    s0, s1, s2 = body["samples"]
    assert s0["label"] == "normal" and s0["defective"] is False and s0["suggested_preset"] == "nominal"
    assert s0["description"] == "normal part"
    assert s1["label"] == "anomaly" and s1["defective"] is True and s1["defect_type"] == "scratch"
    assert s0["model_available"] and not s2["model_available"]
    assert s0["thumb_url"] == "/api/samples/pcb1_normal/image"
    assert "image_path" not in s0  # no server paths leak

    thumb = env.client.get(s0["thumb_url"])
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    timg = cv2.imdecode(np.frombuffer(thumb.content, np.uint8), cv2.IMREAD_COLOR)
    assert max(timg.shape[:2]) <= 384
    full = env.client.get(s0["image_url"])
    fimg = cv2.imdecode(np.frombuffer(full.content, np.uint8), cv2.IMREAD_COLOR)
    assert fimg.shape[:2] == (600, 900)
    assert env.client.get("/api/samples/nope/image").status_code == 404


# ------------------------------------------------------------------------------------------ inspect


def test_inspect_json_returns_trace_and_thumbnails(env) -> None:
    r = _inspect(env.client, seed=3)
    assert r.status_code == 200, r.text
    assert len(r.content) < RESPONSE_BUDGET_BYTES
    body = r.json()
    trace = body["trace"]
    assert trace["trace_id"] == "tr-pcb1_normal-3" and trace["verdict"]["label"] == "PASS"
    assert trace["request"] == {"source": "sample", "sample_id": "pcb1_normal", "category": "pcb1", "preset": "nominal", "planner": "rule", "seed": 3}
    assert body["review"] is None and body["persisted"] is True
    assert set(body["artifacts"]) == {"art_1", "art_2", "art_3"}
    for url in body["artifacts"].values():
        assert max(_decode_data_url(url).shape[:2]) <= ARTIFACT_MAX_SIDE
    assert _decode_data_url(body["artifacts"]["art_2"]).shape[:2] == (200, 300)
    call = env.loop.calls[-1]
    assert call["seed"] == 3 and call["preset"] == "nominal" and call["planner"] == "planner:rule"
    assert call["scene_shape"] == (600, 900, 3) and call["part"].label == 0
    assert env.service.counters.snapshot()["inspections"] == 1

    # persisted trace + artifacts are retrievable
    t = env.client.get(f"/api/traces/{trace['trace_id']}")
    assert t.status_code == 200
    assert t.json()["artifact_urls"]["art_1"] == f"/api/traces/{trace['trace_id']}/artifacts/art_1"
    a = env.client.get(body["artifact_urls"]["art_1"])
    assert a.status_code == 200 and a.headers["content-type"] == "image/jpeg"
    assert cv2.imdecode(np.frombuffer(a.content, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (667, 1000)  # original stored

    # same sample/seed again -> deterministic loop id collides -> stored under a unique id
    r2 = _inspect(env.client, seed=3)
    assert r2.json()["trace"]["trace_id"] == "tr-pcb1_normal-3-r2"
    assert r2.json()["trace"]["original_trace_id"] == "tr-pcb1_normal-3"


def test_inspect_validation_errors(env) -> None:
    c = env.client
    assert _inspect(c, sample_id="nope").status_code == 404
    assert _inspect(c, preset="foggy").status_code == 422
    assert _inspect(c, planner="gpt").status_code == 422
    assert c.post("/api/inspect", json={"sample_id": "pcb1_normal", "seed": -1}).status_code == 422
    assert c.post("/api/inspect", json={"preset": "nominal"}).status_code == 422
    assert c.post("/api/inspect", content=b"{not json", headers={"Content-Type": "application/json"}).status_code == 422
    assert c.post("/api/inspect", content=b"a=b", headers={"Content-Type": "application/x-www-form-urlencoded"}).status_code == 415
    r = _inspect(c, planner="bedrock")
    assert r.status_code == 400 and "INSPECTAGENT_BEDROCK_MODEL_ID" in r.json()["detail"]
    r = _inspect(c, sample_id="pcb2_other")
    assert r.status_code == 503 and "pcb2" in r.json()["detail"]
    assert env.loop.calls == []


def test_inspect_error_becomes_500(env, monkeypatch) -> None:
    monkeypatch.setattr(app_mod, "run_inspection", FakeLoop(fail=True))
    r = _inspect(env.client)
    assert r.status_code == 500 and "simulated detector crash" in r.json()["detail"]
    assert env.service.counters.snapshot()["errors"] == 1


def test_response_budget_with_many_large_artifacts(env, monkeypatch) -> None:
    # Shrink the budget so a handful of high-entropy artifacts must trigger the degradation ladder.
    budget = 900_000
    monkeypatch.setattr(app_mod, "RESPONSE_BUDGET_BYTES", budget)
    monkeypatch.setattr(app_mod, "run_inspection", FakeLoop(n_extra_artifacts=12, noisy=True))
    r = _inspect(env.client)
    assert r.status_code == 200
    assert len(r.content) < budget
    body = r.json()
    assert body["artifacts"] and any("response budget" in n or "omitted" in n for n in body["notes"])
    for url in body["artifacts"].values():
        assert max(_decode_data_url(url).shape[:2]) <= ARTIFACT_MAX_SIDE


def test_encode_artifacts_unit() -> None:
    rng = np.random.default_rng(0)
    arts = {f"a{i}": _jpeg(rng.integers(0, 256, (600, 800, 3), dtype=np.uint8), 95) for i in range(4)}
    arts["bad"] = b"not an image"
    urls, notes = encode_artifacts(arts, list(arts), budget_bytes=10**9)
    assert set(urls) == {"a0", "a1", "a2", "a3"} and any("bad" in n for n in notes)
    assert all(max(_decode_data_url(u).shape[:2]) == 512 for u in urls.values())
    small, notes2 = encode_artifacts(arts, list(arts), budget_bytes=60_000)
    assert sum(len(u) for u in small.values()) <= 60_000
    assert small and "a3" in small  # newest evidence is kept when dropping
    none, _ = encode_artifacts(arts, list(arts), budget_bytes=10)
    assert none == {}


# ------------------------------------------------------------------------------------------- upload


def _upload(client, data: bytes, name="part.png", ctype="image/png", category="pcb1", preset="nominal", seed="0"):
    return client.post("/api/inspect", files={"file": (name, data, ctype)}, data={"category": category, "preset": preset, "seed": seed})


def test_upload_validation(env) -> None:
    c = env.client
    board = make_board(5)
    assert _upload(c, b"hello world, not an image", "x.txt", "text/plain").status_code == 415
    assert _upload(c, b"GIF89a" + b"\0" * 100, "x.gif", "application/octet-stream").status_code == 415
    big = b"\xff\xd8\xff" + b"\0" * MAX_UPLOAD_BYTES
    assert _upload(c, big, "big.jpg", "image/jpeg").status_code == 413
    assert _upload(c, b"\xff\xd8\xff\xe0" + b"garbage" * 50, "bad.jpg", "image/jpeg").status_code == 422
    assert _upload(c, _png(board), category="cashew").status_code == 422
    assert c.post("/api/inspect", files={"file": ("p.png", _png(board), "image/png")}, data={"preset": "nominal"}).status_code == 422
    assert c.post("/api/inspect", data={"category": "pcb1"}, files={"other": ("p.png", b"x", "image/png")}).status_code == 422
    assert _upload(c, _png(board), seed="abc").status_code == 422
    assert _upload(c, _png(board), preset="foggy").status_code == 422
    assert _upload(c, _png(board[:40, :40])).status_code == 422  # too small
    assert _upload(c, _png(cv2.resize(board, (600, 600)))).status_code == 422  # aspect 1.0 vs 1.5
    assert env.loop.calls == []


def test_upload_inspection_resizes_and_stores_input(env) -> None:
    small = cv2.resize(make_board(5), (450, 300))
    r = _upload(env.client, _png(small), preset="dark", seed="7")
    assert r.status_code == 200, r.text
    body = r.json()
    assert env.loop.calls[-1]["scene_shape"] == (600, 900, 3)
    assert any("resized from 450x300 to 900x600" in n for n in body["notes"])
    req = body["trace"]["request"]
    assert req["source"] == "upload" and req["filename"] == "part.png" and len(req["sha256"]) == 64
    assert body["trace"]["part_id"].startswith("upload-")
    tid = body["trace"]["trace_id"]
    stored = env.app.state.storage.get_artifact(f"{tid}/input")
    assert cv2.imdecode(np.frombuffer(stored, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (600, 900)
    # FAIL -> quarantined -> scrap-approval review item
    assert body["review"]["kind"] == "scrap_approval" and body["review"]["routing_action"] == "route_quarantine"


def test_sniff_image_headers() -> None:
    img = np.zeros((70, 130, 3), np.uint8)
    assert sniff_image(_png(img)) == ("png", (130, 70))
    assert sniff_image(_jpeg(img)) == ("jpeg", (130, 70))
    assert sniff_image(b"nope") == (None, None)


# ------------------------------------------------------------------------------------------ reviews


def test_review_rescan_flow(env) -> None:
    c = env.client
    r = _inspect(c, sample_id="pcb1_defect", preset="harsh", seed=4)
    assert r.status_code == 200
    review = r.json()["review"]
    assert review["kind"] == "needs_review" and review["status"] == "pending"
    assert review["proposed_label"] == "FAIL" and review["reason"] == "p=0.55 ambiguous" and review["agent_review_id"] == "rev_1"
    assert review["evidence"] and c.get(review["evidence"][-1]["url"]).status_code == 200

    pending = c.get("/api/reviews").json()
    assert pending["count"] == 1 and pending["reviews"][0]["review_id"] == review["review_id"]
    assert pending["reviews"][0]["audit_valid"] is True

    d = c.post(f"/api/reviews/{review['review_id']}/decision", json={"decision": "rescan", "reviewer": "alice", "note": "glare?"})
    assert d.status_code == 200, d.text
    body = d.json()
    assert body["action"] == "route_pass" and body["final_label"] == "PASS"
    last = env.loop.calls[-1]
    assert last["preset"] == "clean" and last["seed"] == 5 and last["part"].part_id == "pcb1_defect"
    rescan_trace = body["rescan"]["trace"]
    assert rescan_trace["links"] == {"rescan_of": r.json()["trace"]["trace_id"], "review_id": review["review_id"]}
    assert rescan_trace["request"]["preset"] == "clean" and body["rescan"]["artifacts"]
    upd = body["review"]
    assert upd["status"] == "rescanned" and upd["rescan_trace_id"] == rescan_trace["trace_id"]
    assert [a["event"] for a in upd["audit"]] == ["created", "decision"] and upd["audit_valid"] is True
    assert upd["audit"][1]["actor"] == "alice" and upd["decision"]["note"] == "glare?"

    original = c.get(f"/api/traces/{r.json()['trace']['trace_id']}").json()
    assert original["links"]["rescanned_by"] == [rescan_trace["trace_id"]]
    assert original["human_decision"]["decision"] == "rescan"

    assert c.get("/api/reviews").json()["count"] == 0
    assert c.get("/api/reviews?status=all").json()["count"] == 1
    again = c.post(f"/api/reviews/{review['review_id']}/decision", json={"decision": "approve_fail"})
    assert again.status_code == 409


def test_review_approve_fail_and_override_pass(env) -> None:
    c = env.client
    fail = _inspect(c, sample_id="pcb1_defect", preset="dark").json()
    assert fail["trace"]["verdict"]["action"] == "route_quarantine"
    rid = fail["review"]["review_id"]
    d = c.post(f"/api/reviews/{rid}/decision", json={"decision": "approve_fail", "reviewer": "bob"})
    assert d.status_code == 200 and d.json()["action"] == "route_scrap" and d.json()["final_label"] == "FAIL"
    assert d.json()["review"]["status"] == "resolved" and d.json()["rescan"] is None

    amb = _inspect(c, sample_id="pcb1_normal", preset="harsh").json()
    d2 = c.post(f"/api/reviews/{amb['review']['review_id']}/decision", json={"decision": "override_pass"})
    assert d2.status_code == 200 and d2.json()["action"] == "route_pass"
    assert d2.json()["review"]["decision"]["reviewer"] == "operator"

    rows = c.get("/api/traces?limit=10").json()["traces"]
    by_id = {t["trace_id"]: t for t in rows}
    assert by_id[fail["trace"]["trace_id"]]["human_action"] == "route_scrap"
    resolved = c.get("/api/reviews?status=resolved").json()
    assert resolved["count"] == 2
    snap = env.service.counters.snapshot()
    assert snap["decisions"] == {"approve_fail": 1, "override_pass": 1} and snap["reviews_created"] == 2


def test_review_errors(env) -> None:
    c = env.client
    assert c.post("/api/reviews/rv-nope/decision", json={"decision": "approve_fail"}).status_code == 404
    rid = _inspect(c, preset="harsh").json()["review"]["review_id"]
    assert c.post(f"/api/reviews/{rid}/decision", json={"decision": "delete"}).status_code == 422
    assert c.post(f"/api/reviews/{rid}/decision", json={"decision": "rescan", "reviewer": "x" * 65}).status_code == 422
    assert c.get("/api/reviews?status=weird").status_code == 422


def test_rescan_of_uploaded_part_uses_stored_input(env) -> None:
    c = env.client
    first = _upload(c, _png(make_board(9)), preset="harsh", seed="11").json()
    rid = first["review"]["review_id"]
    d = c.post(f"/api/reviews/{rid}/decision", json={"decision": "rescan"})
    assert d.status_code == 200, d.text
    last = env.loop.calls[-1]
    assert last["preset"] == "clean" and last["seed"] == 12 and last["scene_shape"] == (600, 900, 3)
    assert last["part"].part_id == first["trace"]["part_id"]
    assert d.json()["rescan"]["trace"]["request"]["source"] == "upload"


# ------------------------------------------------------------------------------------------- traces


def test_traces_listing(env) -> None:
    c = env.client
    assert c.get("/api/traces").json() == {"traces": [], "count": 0}
    for seed in range(3):
        _inspect(c, seed=seed)
    rows = c.get("/api/traces?limit=2").json()
    assert rows["count"] == 2
    assert [t["trace_id"] for t in rows["traces"]] == ["tr-pcb1_normal-2", "tr-pcb1_normal-1"]
    assert rows["traces"][0]["label"] == "PASS" and rows["traces"][0]["preset"] == "nominal"
    assert c.get("/api/traces?limit=0").status_code == 422
    assert c.get("/api/traces/does-not-exist").status_code == 404
    assert c.get("/api/traces/tr-pcb1_normal-0/artifacts/art_99").status_code == 404
    assert c.get("/api/traces/..%2F..%2Fsecret").status_code == 404


# -------------------------------------------------------------------------------- system / metrics


def test_system_info_has_no_secrets(env, monkeypatch) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SUPERSECRETVALUE123")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "SESSIONTOKEN456")
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "inspectagent-fn")
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", "3008")
    r = env.client.get("/api/system")
    assert r.status_code == 200
    assert "SUPERSECRETVALUE123" not in r.text and "SESSIONTOKEN456" not in r.text
    body = r.json()
    assert body["cv"]["opencv_version"].startswith("5")
    assert body["deployment"]["lambda"] is True and body["deployment"]["function_name"] == "inspectagent-fn"
    assert body["deployment"]["memory_mb"] == "3008" and body["deployment"]["arch"]
    assert body["settings"]["storage"] == {"kind": "local"}
    assert body["settings"]["policy"]["auto_fail_min_prob"] == pytest.approx(0.9)
    assert {p["name"]: p["available"] for p in body["planners"]} == {"rule": True, "bedrock": False}
    assert "nominal" in body["presets"] and body["samples"] == 3
    assert body["models"]["available_categories"] == ["pcb1"]


def test_metrics_and_figures(env) -> None:
    c = env.client
    _inspect(c)
    _inspect(c, preset="harsh")
    m = c.get("/api/metrics").json()
    assert m["eval"]["available"] is True and m["eval"]["summary"]["n_eval"] == 120
    titles = [t["title"] for t in m["eval"]["tables"]]
    assert titles[0] == "Overview" and "systems" in titles and "per_category" in titles and "rows" in titles
    systems = next(t for t in m["eval"]["tables"] if t["title"] == "systems")
    assert systems["columns"] == ["key", "auroc", "escape_rate"] and systems["rows"][1] == ["agent_rule", 0.95, 0.03]
    assert m["live"]["inspections"] == 2 and m["live"]["by_label"] == {"PASS": 1, "NEEDS_REVIEW": 1}
    assert m["store"]["pending_reviews"] == 1 and m["store"]["recent_traces"] == 2
    assert [f["name"] for f in m["figures"]] == ["auroc_by_system.png"]
    fig = c.get(m["figures"][0]["url"])
    assert fig.status_code == 200 and fig.headers["content-type"] == "image/png"
    assert c.get("/api/figures/notes.txt").status_code == 404
    assert c.get("/api/figures/..%2Feval_summary.json").status_code == 404


def test_metrics_without_eval_summary(env) -> None:
    (env.tmp / "results" / "eval_summary.json").unlink()
    m = env.client.get("/api/metrics").json()
    assert m["eval"]["available"] is False and m["eval"]["tables"] == []


def test_eval_tables_shapes() -> None:
    tables = eval_tables({"a": 1, "nested": {"b": 2}, "list": [{"x": 1, "y": {"p": 1}}, {"x": 2, "z": [1, 2]}]})
    assert tables[0]["title"] == "Overview" and ["a", 1] in tables[0]["rows"] and ["nested.b", 2] in tables[0]["rows"]
    lst = next(t for t in tables if t["title"] == "list")
    assert lst["columns"] == ["x", "y.p", "z"] and lst["rows"] == [[1, 1, None], [2, None, "1, 2"]]
    two = eval_tables({"cat": {"reg": {"sys": {"auroc": 0.5}}}})
    assert two[0]["columns"] == ["key", "key2", "auroc"] and two[0]["rows"] == [["reg", "sys", 0.5]]


def test_bench_endpoint(env, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "scripts.bench_cv", None)  # import raises ImportError
    r = env.client.get("/api/bench?n=3")
    assert r.status_code == 501
    fake = types.ModuleType("scripts.bench_cv")
    fake.run_bench = lambda n=10: {"registration": {"mean_ms": np.float32(1.5), "n": n}, "arch": "test"}
    monkeypatch.setitem(sys.modules, "scripts.bench_cv", fake)
    r = env.client.get("/api/bench?n=3")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 3 and body["result"]["registration"] == {"mean_ms": 1.5, "n": 3}
    assert env.client.get("/api/bench?n=0").status_code == 422


# ------------------------------------------------------------------------------------------- lambda


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_lambda_handler_function_url_event(monkeypatch) -> None:
    monkeypatch.setenv("INSPECTAGENT_WARMUP", "0")
    if "inspectagent.api.lambda_handler" in sys.modules:
        lh = importlib.reload(sys.modules["inspectagent.api.lambda_handler"])
    else:
        lh = importlib.import_module("inspectagent.api.lambda_handler")
    assert lh.WARMUP_REPORT["skipped"] is True
    event = {
        "version": "2.0",
        "routeKey": "$default",
        "rawPath": "/healthz",
        "rawQueryString": "",
        "headers": {"host": "abc.lambda-url.us-east-1.on.aws", "x-forwarded-proto": "https", "x-forwarded-port": "443"},
        "requestContext": {
            "accountId": "anonymous", "apiId": "abc", "domainName": "abc.lambda-url.us-east-1.on.aws", "domainPrefix": "abc",
            "http": {"method": "GET", "path": "/healthz", "protocol": "HTTP/1.1", "sourceIp": "203.0.113.1", "userAgent": "pytest"},
            "requestId": "req-1", "routeKey": "$default", "stage": "$default", "time": "09/Oct/2026:12:00:00 +0000", "timeEpoch": 1791547200000,
        },
        "isBase64Encoded": False,
    }
    ctx = SimpleNamespace(aws_request_id="req-1", function_name="fn", memory_limit_in_mb=3008, invoked_function_arn="arn", get_remaining_time_in_millis=lambda: 30000)
    resp = lh.handler(event, ctx)
    assert resp["statusCode"] == 200
    assert json.loads(resp["body"])["ok"] is True


# ------------------------------------------------------------------------- real agent loop (fakes)


def test_inspect_with_real_agent_loop(env, monkeypatch) -> None:
    """End-to-end through the real agent loop/policy/tools with the light-weight vision fakes."""
    loop_mod = pytest.importorskip("inspectagent.agent.loop")
    fakes = pytest.importorskip("tests.agent_fakes")
    spot = fakes.Spot(bbox=(400.0, 250.0, 40.0, 30.0), p_full=0.6, p_zoom=0.97, p_deep=0.95)

    def real_loop(part, bundle=None, planner=None, policy=None, settings=None, seed=0, on_step=None, scene=None):
        return loop_mod.run_inspection(part, fakes.make_bundle([spot]), planner=None, policy=policy, settings=settings,
                                       seed=seed, on_step=on_step, scene=scene, backend=fakes.FakeVision().backend())

    monkeypatch.setattr(app_mod, "run_inspection", real_loop)
    r = _inspect(env.client, sample_id="pcb1_defect", preset="nominal", seed=1)
    assert r.status_code == 200, r.text
    body = r.json()
    trace = body["trace"]
    assert trace["steps"] and trace["verdict"]["label"] in ("PASS", "FAIL", "NEEDS_REVIEW")
    assert trace["steps"][0]["decision"]["tool"] == "capture_image"
    assert len(r.content) < RESPONSE_BUDGET_BYTES
    referenced = {a for s in trace["steps"] for a in s["artifacts"]}
    assert referenced and referenced <= set(body["artifacts"])
    for url in body["artifacts"].values():
        assert max(_decode_data_url(url).shape[:2]) <= ARTIFACT_MAX_SIDE
    stored = env.client.get(f"/api/traces/{trace['trace_id']}").json()
    assert stored["verdict"] == trace["verdict"]
    if trace["verdict"]["action"] in ("hold_for_review", "route_quarantine"):
        assert body["review"] and body["review"]["trace_id"] == trace["trace_id"]
    else:
        assert body["review"] is None


def test_make_planner_real_modules() -> None:
    pytest.importorskip("inspectagent.agent.planner_rule")
    pytest.importorskip("inspectagent.agent.planner_bedrock")
    assert type(app_mod.make_planner("rule", Settings())).__name__ == "RulePlanner"
    bed = app_mod.make_planner("bedrock", Settings(bedrock_model_id="test-model", aws_region="us-east-1"))
    assert type(bed).__name__ == "BedrockPlanner"  # constructed lazily: no network call
    with pytest.raises(app_mod.ApiError) as err:
        app_mod.make_planner("bedrock", Settings(bedrock_model_id=""))
    assert err.value.status_code == 400
    with pytest.raises(app_mod.ApiError):
        app_mod.make_planner("oracle", Settings())
