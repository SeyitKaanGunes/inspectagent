"""MCP server exposing the InspectAgent inspection cell to any MCP client (SPEC section 4).

Run over stdio (Claude Code, Kiro, ...)::

    python -m inspectagent.mcp_server
    python -m inspectagent.mcp_server --http --port 8765      # streamable HTTP at http://127.0.0.1:8765/mcp

Tools
-----
* ``list_samples(category?, include_labels?)`` - curated demo parts from ``samples/samples.json``.
* ``start_session(sample_id | image_path, category?, preset, seed)`` -> ``session_id``.
* every tool of the agent tool registry (``capture_image``, ``assess_quality``, ``detect_anomalies``,
  ``inspect_region``, ``measure_defect``, ``request_human_review``, ``finish``) with an extra
  ``session_id`` argument; the wrappers are generated dynamically from ``registry.specs()``.
  Each call is vetted by the human-control policy exactly like the autonomous loop, so an MCP
  client cannot auto-PASS without reliable evidence. New image artifacts are returned as MCP
  ``ImageContent`` thumbnails next to the JSON observation.
* ``get_trace(session_id)``, ``get_artifact(session_id, artifact_id)``, ``close_session(session_id)``.
* ``run_inspection(sample_id, preset, planner="rule"|"bedrock", seed)`` - the full autonomous loop.

Sessions live in memory (bounded, least-recently-used eviction). Implemented on the low-level
``mcp.server.lowlevel.Server`` of the installed ``mcp`` 2.x SDK because tool input schemas come
verbatim from the registry rather than from Python signatures.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import logging
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
import mcp_types as mtypes
import numpy as np

from inspectagent import __version__
from inspectagent.agent.planner_bedrock import (
    RATIONALE_FIELD,
    TERMINAL_TOOLS,
    extract_ids,
    load_default_registry,
    normalize_tool_spec,
    thumbnail_jpeg,
    to_jsonable,
)
from inspectagent.config import Settings, get_settings
from inspectagent.types import Decision, PartSpec, Step, Trace, Verdict

log = logging.getLogger("inspectagent.mcp")

SERVER_NAME = "inspectagent"
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
SAMPLE_IMAGE_KEYS = ("image", "image_path", "path", "file", "full", "src")
SAMPLE_MASK_KEYS = ("mask", "mask_path")

SERVER_INSTRUCTIONS = """\
InspectAgent is an agentic visual-inspection cell (OpenCV 5, simulated active camera over real VisA parts).
Typical manual workflow: list_samples -> start_session(sample_id, preset) -> capture_image(session_id, ...) ->
assess_quality -> (corrective recapture: light/exposure/autofocus) -> detect_anomalies(fast; deep if unreliable)
-> inspect_region on candidates -> measure_defect -> finish(label) or request_human_review. Every action is vetted
by a human-control policy (budget, no PASS without reliable evidence, uncertain FAIL needs a human).
get_trace(session_id) returns the full evidence trace. run_inspection runs the autonomous agent end to end."""


class ToolInputError(ValueError):
    """Invalid arguments for an MCP tool."""


@dataclass
class ToolOutput:
    """Result of one MCP tool call before conversion to MCP content blocks."""

    data: dict
    images: list[tuple[str, bytes]] = field(default_factory=list)  # (caption, JPEG bytes)
    is_error: bool = False


@dataclass
class ManagedSession:
    """An inspection session held by the MCP server."""

    key: str
    session: Any
    sample_id: str | None
    preset: str
    seed: int
    started_at: str
    steps: list[Step] = field(default_factory=list)
    verdict: Verdict | None = None
    trace: Trace | None = None
    closed: bool = False
    errors: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _label_to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return 1 if int(value) else 0
    text = str(value).strip().lower()
    if text in {"anomaly", "anomalous", "defect", "defective", "bad", "fail", "1", "true"}:
        return 1
    if text in {"normal", "good", "ok", "pass", "0", "false"}:
        return 0
    return None


_STATIC_SCHEMAS: dict[str, dict] = {
    "list_samples": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "description": "Only samples of this category."},
            "include_labels": {
                "type": "boolean",
                "default": False,
                "description": "Reveal ground-truth labels (off by default so the agent decides from images).",
            },
        },
        "additionalProperties": False,
    },
    "start_session": {
        "type": "object",
        "properties": {
            "sample_id": {"type": "string", "description": "Id from list_samples."},
            "image_path": {"type": "string", "description": "Local image file (alternative to sample_id; needs category)."},
            "category": {"type": "string", "description": "Model bundle category (required with image_path)."},
            "preset": {
                "type": "string",
                "description": "Hidden capture-condition preset (default: the sample's suggested preset, else nominal).",
            },
            "seed": {"type": "integer", "minimum": 0, "default": 0},
        },
        "additionalProperties": False,
    },
    "get_trace": {
        "type": "object",
        "properties": {"session_id": {"type": "string"}},
        "required": ["session_id"],
        "additionalProperties": False,
    },
    "get_artifact": {
        "type": "object",
        "properties": {
            "session_id": {"type": "string"},
            "artifact_id": {"type": "string", "description": "Artifact id from an observation (e.g. art_3)."},
            "max_side": {"type": "integer", "minimum": 64, "maximum": 1600, "default": 768},
        },
        "required": ["session_id", "artifact_id"],
        "additionalProperties": False,
    },
    "close_session": {
        "type": "object",
        "properties": {"session_id": {"type": "string"}},
        "required": ["session_id"],
        "additionalProperties": False,
    },
    "run_inspection": {
        "type": "object",
        "properties": {
            "sample_id": {"type": "string"},
            "image_path": {"type": "string", "description": "Local image file (alternative to sample_id; needs category)."},
            "category": {"type": "string"},
            "preset": {"type": "string", "description": "Capture-condition preset (default: suggested, else nominal)."},
            "planner": {"type": "string", "enum": ["rule", "bedrock"], "default": "rule"},
            "seed": {"type": "integer", "minimum": 0, "default": 0},
        },
        "additionalProperties": False,
    },
}

_STATIC_DESCRIPTIONS = {
    "list_samples": "List the curated demo parts (id, category, image availability).",
    "start_session": "Present a part to the inspection cell and open an inspection session. Returns session_id, "
    "the budget and the session-scoped tools to call next (start with capture_image).",
    "get_trace": "Full inspection trace of a session: every step (tool, args, rationale, observation, artifacts), "
    "policy overrides, verdict and totals. Ground truth is revealed only after the session is finished.",
    "get_artifact": "Return one image artifact (capture, quality overlay or anomaly heatmap) of a session.",
    "close_session": "Forget a session and free its memory.",
    "run_inspection": "Run the autonomous InspectAgent loop (rule-based or Amazon Bedrock planner) on a part and "
    "return the verdict, the step timeline and the final evidence image.",
}

_READ_ONLY_STATIC = {"list_samples", "get_trace", "get_artifact"}


class InspectAgentMCP:
    """Tool implementations and session store behind the MCP server (callable without MCP)."""

    STATIC_TOOLS = tuple(_STATIC_SCHEMAS)

    def __init__(
        self,
        *,
        registry: Any | None = None,
        settings: Settings | None = None,
        policy: Any | None = None,
        enforce_policy: bool = True,
        session_factory: Callable[[PartSpec, int], Any] | None = None,
        inspection_runner: Callable[[PartSpec, Any, int], tuple[Trace, Any]] | None = None,
        planner_factory: Callable[[str], Any] | None = None,
        samples_path: str | Path | None = None,
        max_sessions: int = 16,
        allow_image_paths: bool = True,
        image_long_side: int = 384,
        max_images_per_result: int = 2,
    ) -> None:
        self.settings = settings or get_settings()
        self._registry = registry
        self._policy = policy
        self.enforce_policy = enforce_policy
        self._session_factory = session_factory
        self._inspection_runner = inspection_runner
        self._planner_factory = planner_factory
        self.samples_path = Path(samples_path) if samples_path else Path(self.settings.samples_dir) / "samples.json"
        self.max_sessions = max(1, int(max_sessions))
        self.allow_image_paths = bool(allow_image_paths)
        self.image_long_side = int(image_long_side)
        self.max_images_per_result = max(0, int(max_images_per_result))
        self._sessions: OrderedDict[str, ManagedSession] = OrderedDict()
        self._store_lock = threading.Lock()
        self._samples_cache: tuple[float, list[dict]] | None = None
        self._cv_info: dict | None = None
        self._policy_warning_done = False

    # ------------------------------------------------------------------ dependencies
    @property
    def registry(self) -> Any:
        """Agent tool registry (loaded lazily from ``inspectagent.agent.tools``)."""
        if self._registry is None:
            self._registry = load_default_registry(self.settings)
        return self._registry

    def _registry_specs(self) -> list[dict]:
        try:
            return [normalize_tool_spec(s) for s in self.registry.specs()]
        except Exception as exc:
            log.warning("agent tool registry unavailable: %s", exc)
            return []

    def _get_policy(self) -> Any | None:
        if not self.enforce_policy:
            return None
        if self._policy is None:
            try:
                from inspectagent.agent.policy import HumanControlPolicy

                self._policy = HumanControlPolicy(self.settings.policy)
            except Exception as exc:
                if not self._policy_warning_done:
                    log.warning("human-control policy unavailable, tools run unvetted: %s", exc)
                    self._policy_warning_done = True
                return None
        return self._policy

    def _load_bundle(self, category: str) -> Any:
        from inspectagent.models_io import load_bundle

        return load_bundle(category, self.settings.models_dir)

    def _new_session(self, part: PartSpec, seed: int) -> Any:
        if self._session_factory is not None:
            return self._session_factory(part, seed)
        from inspectagent.agent.session import InspectionSession

        return InspectionSession(part, self._load_bundle(part.category), settings=self.settings, seed=seed)

    def _make_planner(self, name: str) -> Any:
        if self._planner_factory is not None:
            return self._planner_factory(name)
        from inspectagent.agent.loop import make_planner

        return make_planner(name, self.settings)

    def _run_loop(self, part: PartSpec, planner: Any, seed: int) -> tuple[Trace, Any]:
        if self._inspection_runner is not None:
            return self._inspection_runner(part, planner, seed)
        from inspectagent.agent.loop import run_inspection

        return run_inspection(part, bundle=self._load_bundle(part.category), planner=planner, settings=self.settings, seed=seed)

    def _cv(self) -> dict:
        if self._cv_info is None:
            try:
                from inspectagent.vision.cvinfo import cv_info

                self._cv_info = cv_info()
            except Exception as exc:
                self._cv_info = {"error": f"cv_info unavailable: {exc}"}
        return self._cv_info

    @staticmethod
    def _presets() -> tuple[str, ...] | None:
        try:
            from inspectagent.vision.camera import PRESETS

            return tuple(PRESETS)
        except Exception:
            return None

    # ------------------------------------------------------------------ tool definitions
    def tool_definitions(self) -> list[mtypes.Tool]:
        """MCP tool list: static tools + session-scoped wrappers of every registry tool."""
        tools: list[mtypes.Tool] = []
        presets = self._presets()
        for name, schema in _STATIC_SCHEMAS.items():
            schema = copy.deepcopy(schema)
            if presets and "preset" in schema["properties"]:
                schema["properties"]["preset"]["enum"] = list(presets)
            tools.append(
                mtypes.Tool(
                    name=name,
                    description=_STATIC_DESCRIPTIONS[name],
                    input_schema=schema,
                    annotations=mtypes.ToolAnnotations(
                        read_only_hint=name in _READ_ONLY_STATIC, destructive_hint=False, open_world_hint=False
                    ),
                )
            )
        for spec in self._registry_specs():
            if spec["name"] in _STATIC_SCHEMAS:
                continue
            tools.append(
                mtypes.Tool(
                    name=spec["name"],
                    description=f"{spec['description']} [session-scoped: pass the session_id from start_session]",
                    input_schema=self._session_schema(spec["input_schema"]),
                    annotations=mtypes.ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False),
                )
            )
        return tools

    @staticmethod
    def _session_schema(schema: dict) -> dict:
        schema = copy.deepcopy(schema)
        props = {"session_id": {"type": "string", "description": "Session id returned by start_session."}}
        props.update(schema.get("properties", {}))
        if RATIONALE_FIELD not in props:
            props[RATIONALE_FIELD] = {
                "type": "string",
                "description": "Why you take this action (evidence ids and numbers); recorded in the trace.",
            }
        schema["properties"] = props
        required = [r for r in schema.get("required", []) if r != "session_id"]
        schema["required"] = ["session_id", *required]
        schema.setdefault("type", "object")
        return schema

    def tool_input_schema(self, name: str) -> dict | None:
        """Input schema of tool ``name`` (None when unknown)."""
        for tool in self.tool_definitions():
            if tool.name == name:
                return tool.input_schema
        return None

    # ------------------------------------------------------------------ dispatch
    def call(self, name: str, arguments: dict | None = None) -> ToolOutput:
        """Execute tool ``name`` synchronously; never raises (errors become ``is_error`` outputs)."""
        arguments = dict(arguments or {})
        try:
            if name in _STATIC_SCHEMAS:
                self._validate(name, arguments)
                return getattr(self, f"tool_{name}")(**arguments)
            registry_names = {s["name"] for s in self._registry_specs()}
            if name in registry_names:
                return self.call_session_tool(name, arguments)
            raise ToolInputError(f"unknown tool {name!r}")
        except ToolInputError as exc:
            return ToolOutput({"error": str(exc), "tool": name}, is_error=True)
        except Exception as exc:  # tool crashed: report, keep the server alive
            log.exception("MCP tool %s failed", name)
            return ToolOutput({"error": f"{type(exc).__name__}: {exc}", "tool": name}, is_error=True)

    @staticmethod
    def _validate(name: str, arguments: dict) -> None:
        import jsonschema

        try:
            jsonschema.validate(arguments, _STATIC_SCHEMAS[name])
        except jsonschema.ValidationError as exc:
            raise ToolInputError(f"invalid arguments for {name}: {exc.message}") from exc

    async def acall(self, name: str, arguments: dict | None = None) -> mtypes.CallToolResult:
        """Async MCP entry point: runs the (CPU-bound) tool in a worker thread."""
        out = await anyio.to_thread.run_sync(self.call, name, dict(arguments or {}))
        return self.to_call_result(out)

    @staticmethod
    def to_call_result(out: ToolOutput) -> mtypes.CallToolResult:
        """Convert a :class:`ToolOutput` into an MCP ``CallToolResult`` (JSON text + images)."""
        data = to_jsonable(out.data)
        content: list[Any] = [mtypes.TextContent(type="text", text=json.dumps(data, ensure_ascii=False))]
        for caption, jpeg in out.images:
            content.append(mtypes.TextContent(type="text", text=caption))
            content.append(mtypes.ImageContent(type="image", data=base64.b64encode(jpeg).decode("ascii"), mime_type="image/jpeg"))
        return mtypes.CallToolResult(content=content, structured_content=data, is_error=out.is_error)

    # ------------------------------------------------------------------ session store
    def _register(self, session: Any, *, sample_id: str | None, preset: str, seed: int) -> ManagedSession:
        base = str(getattr(session, "session_id", None) or f"sess_{len(self._sessions) + 1}")
        with self._store_lock:
            key, n = base, 1
            while key in self._sessions:
                n += 1
                key = f"{base}-{n}"
            ms = ManagedSession(key=key, session=session, sample_id=sample_id, preset=preset, seed=seed, started_at=_utc_now())
            self._sessions[key] = ms
            while len(self._sessions) > self.max_sessions:
                evicted, _ = self._sessions.popitem(last=False)
                log.info("evicted MCP session %s", evicted)
        return ms

    def _get(self, session_id: Any) -> ManagedSession:
        if not isinstance(session_id, str) or not session_id:
            raise ToolInputError("session_id is required (call start_session first)")
        with self._store_lock:
            ms = self._sessions.get(session_id)
            if ms is None:
                raise ToolInputError(f"unknown session_id {session_id!r} (sessions are in memory; start a new one)")
            self._sessions.move_to_end(session_id)
        return ms

    @property
    def session_ids(self) -> list[str]:
        """Ids of the sessions currently held in memory."""
        with self._store_lock:
            return list(self._sessions)

    # ------------------------------------------------------------------ samples / parts
    def _load_samples(self) -> list[dict]:
        path = self.samples_path
        if not path.exists():
            return []
        mtime = path.stat().st_mtime
        if self._samples_cache and self._samples_cache[0] == mtime:
            return self._samples_cache[1]
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw.get("samples", []) if isinstance(raw, dict) else raw
        samples = [s for s in items if isinstance(s, dict) and s.get("id")]
        self._samples_cache = (mtime, samples)
        return samples

    def _sample_path(self, sample: dict, keys: tuple[str, ...]) -> Path | None:
        for key in keys:
            value = sample.get(key)
            if isinstance(value, str) and value:
                p = Path(value)
                return p if p.is_absolute() else (self.samples_path.parent / p)
        return None

    def _resolve_part(
        self, sample_id: str | None, image_path: str | None, category: str | None, preset: str | None, seed: int
    ) -> tuple[PartSpec, str]:
        """PartSpec (with hidden nuisance conditions drawn from ``preset``) and the preset used."""
        sample: dict | None = None
        if sample_id:
            sample = next((s for s in self._load_samples() if str(s.get("id")) == sample_id), None)
            if sample is None:
                raise ToolInputError(f"unknown sample_id {sample_id!r} (see list_samples)")
        if not preset:
            preset = str((sample or {}).get("suggested_preset") or "nominal")
        presets = self._presets()
        if presets and preset not in presets:
            raise ToolInputError(f"unknown preset {preset!r}; choose one of {list(presets)}")
        if sample is not None:
            path = self._sample_path(sample, SAMPLE_IMAGE_KEYS)
            if path is None or not path.exists():
                raise ToolInputError(f"image of sample {sample_id!r} not found")
            mask = self._sample_path(sample, SAMPLE_MASK_KEYS)
            part_id, cat = str(sample_id), str(sample.get("category") or category or "")
            label = _label_to_int(sample.get("label"))
        elif image_path:
            if not self.allow_image_paths:
                raise ToolInputError("image_path is disabled on this server (use sample_id)")
            path = Path(image_path).expanduser()
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ToolInputError(f"unsupported image type {path.suffix!r}")
            if not path.is_file():
                raise ToolInputError(f"image_path not found: {image_path}")
            if not category:
                raise ToolInputError("category is required with image_path")
            part_id, cat, label, mask = path.stem, category, None, None
        else:
            raise ToolInputError("pass sample_id or image_path")
        if not cat:
            raise ToolInputError("category unknown for this part")
        from inspectagent.vision.camera import make_conditions

        conditions = make_conditions(preset, np.random.default_rng(seed))
        part = PartSpec(
            part_id=part_id,
            category=cat,
            image_path=str(path),
            label=label,
            mask_path=str(mask) if mask is not None and mask.exists() else None,
            conditions=conditions,
        )
        return part, preset

    # ------------------------------------------------------------------ static tools
    def tool_list_samples(self, category: str | None = None, include_labels: bool = False) -> ToolOutput:
        """Curated demo parts."""
        out = []
        for s in self._load_samples():
            if category and s.get("category") != category:
                continue
            path = self._sample_path(s, SAMPLE_IMAGE_KEYS)
            item = {
                "id": s["id"],
                "category": s.get("category"),
                "suggested_preset": s.get("suggested_preset"),
                "image_available": bool(path and path.exists()),
            }
            if include_labels:  # notes and defect types reveal the ground truth
                for key in ("label", "label_name", "defect_type", "note"):
                    if key in s:
                        item[key] = s[key]
            out.append(item)
        categories: list[str] = []
        try:
            from inspectagent.models_io import available_categories

            categories = list(available_categories(self.settings.models_dir))
        except Exception as exc:
            log.debug("available_categories unavailable: %s", exc)
        return ToolOutput({"samples": out, "count": len(out), "model_categories": categories, "presets": list(self._presets() or [])})

    def tool_start_session(
        self,
        sample_id: str | None = None,
        image_path: str | None = None,
        category: str | None = None,
        preset: str | None = None,
        seed: int = 0,
    ) -> ToolOutput:
        """Open an inspection session for a sample (or a local image)."""
        part, preset = self._resolve_part(sample_id, image_path, category, preset, int(seed))
        session = self._new_session(part, int(seed))
        ms = self._register(session, sample_id=sample_id, preset=preset, seed=int(seed))
        scene_size = getattr(getattr(session, "camera", None), "scene_size", None)
        data = {
            "session_id": ms.key,
            "part_id": part.part_id,
            "category": part.category,
            "preset": preset,
            "seed": int(seed),
            "scene_size": list(scene_size) if scene_size else None,
            "budget": asdict(self.settings.policy),
            "policy_enforced": self._get_policy() is not None,
            "tools": [s["name"] for s in self._registry_specs()],
            "next": "capture_image(session_id, cx=0.5, cy=0.5, zoom=1, exposure_ev=0, light='ring', autofocus=false)",
        }
        return ToolOutput(data)

    def tool_get_trace(self, session_id: str) -> ToolOutput:
        """Trace (steps, verdict, totals) of a session."""
        ms = self._get(session_id)
        with ms.lock:
            trace = ms.trace if ms.trace is not None else self._build_trace(ms)
            data = trace.to_dict()
        data["session_id"] = ms.key
        data["closed"] = ms.closed
        return ToolOutput(data)

    def tool_get_artifact(self, session_id: str, artifact_id: str, max_side: int = 768) -> ToolOutput:
        """One image artifact as an MCP image."""
        ms = self._get(session_id)
        artifacts = getattr(ms.session, "artifacts", {}) or {}
        data = artifacts.get(artifact_id)
        if not isinstance(data, (bytes, bytearray)):
            raise ToolInputError(f"unknown artifact_id {artifact_id!r}; available: {list(artifacts)[-12:]}")
        jpeg = thumbnail_jpeg(bytes(data), int(max_side), quality=88)
        if jpeg is None:
            raise ToolInputError(f"artifact {artifact_id!r} is not a decodable image")
        return ToolOutput({"session_id": ms.key, "artifact_id": artifact_id, "bytes": len(jpeg)}, [(f"artifact {artifact_id}", jpeg)])

    def tool_close_session(self, session_id: str) -> ToolOutput:
        """Drop a session from memory."""
        ms = self._get(session_id)
        with self._store_lock:
            self._sessions.pop(ms.key, None)
        return ToolOutput({"session_id": ms.key, "closed": True, "n_steps": len(ms.steps)})

    def tool_run_inspection(
        self,
        sample_id: str | None = None,
        image_path: str | None = None,
        category: str | None = None,
        preset: str | None = None,
        planner: str = "rule",
        seed: int = 0,
    ) -> ToolOutput:
        """Run the autonomous loop and summarise its trace."""
        part, preset = self._resolve_part(sample_id, image_path, category, preset, int(seed))
        planner_obj = self._make_planner(planner)
        t0 = time.perf_counter()
        trace, session = self._run_loop(part, planner_obj, int(seed))
        wall_ms = (time.perf_counter() - t0) * 1000.0
        ms = self._register(session, sample_id=sample_id, preset=preset, seed=int(seed))
        ms.trace, ms.steps, ms.verdict, ms.closed = trace, list(trace.steps), trace.verdict, True
        steps = [
            {
                "step": s.step,
                "tool": s.decision.tool,
                "args": s.decision.args,
                "rationale": s.decision.rationale,
                "planner": s.decision.planner,
                "overridden_by_policy": s.overridden_by_policy,
                "artifacts": list(s.artifacts),
            }
            for s in trace.steps
        ]
        data = {
            "session_id": ms.key,
            "trace_id": trace.trace_id,
            "part_id": trace.part_id,
            "category": trace.category,
            "planner": trace.planner,
            "verdict": trace.verdict.to_dict() if trace.verdict else None,
            "totals": trace.totals,
            "steps": steps,
            "errors": list(trace.errors),
            "wall_ms": round(wall_ms, 1),
            "hint": "get_trace(session_id) returns full observations; get_artifact(session_id, artifact_id) any image.",
        }
        artifact_ids = [a for s in trace.steps for a in s.artifacts]
        return ToolOutput(data, self._images(session, artifact_ids[-1:]))

    # ------------------------------------------------------------------ session-scoped registry tools
    def call_session_tool(self, name: str, arguments: dict) -> ToolOutput:
        """Execute a registry tool inside a session (policy-vetted, recorded as a trace step)."""
        args = dict(arguments)
        ms = self._get(args.pop("session_id", None))
        session = ms.session
        with ms.lock:
            if ms.closed:
                raise ToolInputError(
                    f"session {ms.key} is finished (verdict {ms.verdict.label if ms.verdict else 'n/a'}); start a new session"
                )
            spec_props = next((s["input_schema"].get("properties", {}) for s in self._registry_specs() if s["name"] == name), {})
            client_rationale = args.pop(RATIONALE_FIELD, None) if RATIONALE_FIELD not in spec_props else args.get(RATIONALE_FIELD)
            rationale = str(client_rationale or args.get("rationale") or args.get("reason") or "MCP client request")
            decision = Decision(
                step=len(ms.steps) + 1,
                tool=name,
                args=args,
                rationale=rationale,
                evidence=extract_ids(rationale, json.dumps(to_jsonable(args))),
                planner="mcp-client",
            )
            requested = {"tool": name, "args": to_jsonable(args)}
            overridden = False
            policy = self._get_policy()
            if policy is not None:
                decision, overridden = policy.vet(decision, session)
            artifacts = getattr(session, "artifacts", None)
            before = set(artifacts) if isinstance(artifacts, dict) else set()
            sim_before = float(getattr(session, "sim_time_ms", 0.0) or 0.0)
            t0 = time.perf_counter()
            try:
                observation = self.registry.execute(session, decision.tool, decision.args)
            except Exception as exc:
                log.warning("tool %s raised: %s", decision.tool, exc)
                observation = {"error": f"{type(exc).__name__}: {exc}"}
                ms.errors.append(f"step {decision.step} {decision.tool}: {type(exc).__name__}: {exc}")
            latency = (time.perf_counter() - t0) * 1000.0
            artifacts = getattr(session, "artifacts", None)
            new_artifacts = [a for a in artifacts if a not in before] if isinstance(artifacts, dict) else []
            step = Step(
                step=decision.step,
                decision=decision,
                observation=to_jsonable(observation) if isinstance(observation, dict) else {"result": to_jsonable(observation)},
                artifacts=new_artifacts,
                latency_ms=latency,
                sim_time_ms=max(0.0, float(getattr(session, "sim_time_ms", 0.0) or 0.0) - sim_before),
                overridden_by_policy=overridden,
            )
            ms.steps.append(step)
            data: dict = {
                "session_id": ms.key,
                "step": step.step,
                "tool": decision.tool,
                "observation": step.observation,
                "artifacts": new_artifacts,
            }
            if overridden:
                data["policy_override"] = {"requested": requested, "executed": {"tool": decision.tool, "args": to_jsonable(decision.args)}, "rationale": decision.rationale}
            fused = self._fused_prob(session)
            if fused is not None:
                data["fused_prob"], data["fused_evidence"] = fused
            if self._is_terminal(decision.tool) and not (isinstance(observation, dict) and observation.get("error")):
                ms.verdict = self._final_verdict(ms)
                ms.closed = True
                data["verdict"] = ms.verdict.to_dict()
                data["session_closed"] = True
            budget = self._budget_status(ms)
            if budget:
                data["budget_used"] = budget
            images = self._images(session, new_artifacts[-self.max_images_per_result :] if self.max_images_per_result else [])
        return ToolOutput(data, images, is_error=bool(isinstance(observation, dict) and observation.get("error")))

    def _is_terminal(self, name: str) -> bool:
        reg = self.registry
        for getter in ("get", "get_spec", "spec"):
            fn = getattr(reg, getter, None)
            if callable(fn):
                try:
                    spec = fn(name)
                except Exception:
                    spec = None
                if spec is not None and hasattr(spec, "terminal"):
                    return bool(spec.terminal)
        tools = getattr(reg, "tools", None)
        if isinstance(tools, dict) and name in tools and hasattr(tools[name], "terminal"):
            return bool(tools[name].terminal)
        return name in TERMINAL_TOOLS

    def _final_verdict(self, ms: ManagedSession) -> Verdict:
        policy = self._get_policy()
        if policy is not None:
            return policy.final_verdict(ms.session)
        # Unvetted mode (policy module unavailable): conservative mapping of the terminal call.
        last = ms.steps[-1].decision
        fused = self._fused_prob(ms.session)
        prob = fused[0] if fused else None
        if last.tool == "finish" and last.args.get("label") in ("PASS", "FAIL"):
            label = str(last.args["label"])
            return Verdict(
                label=label,
                confidence=float(last.args.get("confidence", 0.5) or 0.5),
                rationale=str(last.args.get("rationale", "")),
                action="route_pass" if label == "PASS" else "route_quarantine",
                prob_anomalous=prob,
            )
        review = getattr(ms.session, "review", None)
        review_id = review.get("review_id") if isinstance(review, dict) else None
        return Verdict(label="NEEDS_REVIEW", confidence=0.0, rationale=last.rationale, action="hold_for_review", review_id=review_id, prob_anomalous=prob)

    @staticmethod
    def _fused_prob(session: Any) -> tuple[float, list[str]] | None:
        fn = getattr(session, "fused_prob", None)
        if not callable(fn):
            return None
        try:
            value, evidence = fn()
            return float(value), list(evidence)
        except Exception as exc:
            log.debug("fused_prob failed: %s", exc)
            return None

    def _budget_status(self, ms: ManagedSession) -> dict:
        p = self.settings.policy
        session = ms.session
        captures = getattr(session, "captures", None)
        n_deep = sum(1 for s in ms.steps if s.decision.tool == "detect_anomalies" and s.decision.args.get("detector") == "deep")
        return {
            "steps": f"{len(ms.steps)}/{p.max_steps}",
            "captures": f"{len(captures) if isinstance(captures, dict) else 0}/{p.max_captures}",
            "deep_calls": f"{n_deep}/{p.max_deep_calls}",
            "sim_time_ms": f"{float(getattr(session, 'sim_time_ms', 0.0) or 0.0):.0f}/{p.max_sim_time_ms:.0f}",
        }

    def _images(self, session: Any, artifact_ids: list[str]) -> list[tuple[str, bytes]]:
        artifacts = getattr(session, "artifacts", None)
        if not isinstance(artifacts, dict):
            return []
        images = []
        for aid in artifact_ids:
            data = artifacts.get(aid)
            if isinstance(data, (bytes, bytearray)):
                jpeg = thumbnail_jpeg(bytes(data), self.image_long_side, quality=85)
                if jpeg is not None:
                    images.append((f"artifact {aid}", jpeg))
        return images

    # ------------------------------------------------------------------ traces
    def _build_trace(self, ms: ManagedSession) -> Trace:
        session = ms.session
        part = getattr(session, "part", None)
        steps = ms.steps
        tool_counts: dict[str, int] = {}
        for s in steps:
            tool_counts[s.decision.tool] = tool_counts.get(s.decision.tool, 0) + 1
        compute_ms = float(getattr(session, "compute_ms", 0.0) or 0.0) or sum(s.latency_ms for s in steps)
        sim_ms = float(getattr(session, "sim_time_ms", 0.0) or 0.0)
        captures = getattr(session, "captures", None)
        fused = self._fused_prob(session)
        totals = {
            "n_steps": len(steps),
            "n_captures": len(captures) if isinstance(captures, dict) else tool_counts.get("capture_image", 0),
            "n_deep": sum(1 for s in steps if s.decision.tool == "detect_anomalies" and s.decision.args.get("detector") == "deep"),
            "compute_ms": compute_ms,
            "sim_time_ms": sim_ms,
            "cycle_time_ms": compute_ms + sim_ms,
            "tool_counts": tool_counts,
            "final_prob": fused[0] if fused else None,
            "n_policy_overrides": sum(1 for s in steps if s.overridden_by_policy),
        }
        reveal = ms.closed  # hidden world state and ground truth only after the verdict
        label = getattr(part, "label", None)
        conditions = getattr(part, "conditions", None)
        return Trace(
            trace_id=f"tr_{ms.key}",
            part_id=str(getattr(part, "part_id", ms.sample_id or "unknown")),
            category=str(getattr(part, "category", "unknown")),
            planner="mcp-client",
            started_at=ms.started_at,
            steps=list(steps),
            verdict=ms.verdict,
            budget=asdict(self.settings.policy),
            totals=totals,
            ground_truth={"label": label} if reveal and label is not None else None,
            conditions=conditions.to_dict() if reveal and conditions is not None and hasattr(conditions, "to_dict") else None,
            cv_info=self._cv(),
            errors=list(ms.errors),
        )


# ----------------------------------------------------------------------------------------------
# MCP wiring
# ----------------------------------------------------------------------------------------------


def build_server(app: InspectAgentMCP | None = None) -> Any:
    """Low-level MCP ``Server`` whose tool list is generated from ``app`` at request time."""
    from mcp.server.lowlevel import Server

    app = app or InspectAgentMCP()

    async def on_list_tools(ctx: Any, params: Any) -> mtypes.ListToolsResult:
        return mtypes.ListToolsResult(tools=app.tool_definitions())

    async def on_call_tool(ctx: Any, params: mtypes.CallToolRequestParams) -> mtypes.CallToolResult:
        return await app.acall(params.name, params.arguments or {})

    server = Server(
        SERVER_NAME,
        version=__version__,
        title="InspectAgent",
        description="Agentic visual inspection with OpenCV 5 (active camera, quality-aware detection, human control).",
        instructions=SERVER_INSTRUCTIONS,
        get_tool_input_schema=app.tool_input_schema,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )
    server.inspectagent_app = app  # type: ignore[attr-defined]  # handy for tests/embedding
    return server


async def _run_stdio(server: Any) -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: stdio by default, ``--http`` for streamable HTTP."""
    settings = get_settings()
    ap = argparse.ArgumentParser(prog="python -m inspectagent.mcp_server", description="InspectAgent MCP server")
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--path", default="/mcp", help="HTTP endpoint path")
    ap.add_argument("--allow-image-paths", action="store_true", help="allow start_session(image_path=...) over HTTP")
    ap.add_argument("--max-sessions", type=int, default=16)
    ap.add_argument("--log-level", default=settings.log_level)
    args = ap.parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = InspectAgentMCP(
        settings=settings,
        max_sessions=args.max_sessions,
        allow_image_paths=(not args.http) or args.allow_image_paths,
    )
    server = build_server(app)
    if args.http:
        import uvicorn

        starlette_app = server.streamable_http_app(streamable_http_path=args.path, host=args.host)
        log.info("InspectAgent MCP server on http://%s:%d%s", args.host, args.port, args.path)
        uvicorn.run(starlette_app, host=args.host, port=args.port, log_level=str(args.log_level).lower())
    else:
        anyio.run(_run_stdio, server)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
