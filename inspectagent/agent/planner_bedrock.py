"""Amazon Bedrock planner: Converse tool-use loop with a deterministic fallback (SPEC section 3.5).

``BedrockPlanner.next_action(state, session)`` sends the latest tool observation to a model on
Amazon Bedrock through the ``bedrock-runtime`` **Converse** API (``toolConfig`` built from
``registry.specs()``, ``toolChoice: any``, temperature 0) and turns the returned ``toolUse`` block
into a :class:`~inspectagent.types.Decision`.

Conversation protocol (kept per inspection session, stored on the session object):

* the first user message carries the task and the initial planner state;
* every assistant ``toolUse`` is answered by the next user message with a ``toolResult`` that has
  the same ``toolUseId``, content ``[{"json": {"observation": ..., "state": ...}}]`` and status
  ``"success"`` or ``"error"``; extra tool calls in one turn are answered with an error result
  (exactly one tool is executed per step);
* an optional 256 px JPEG thumbnail of the newest artifact follows as an ``image`` block;
* history is bounded: the first user message is always kept, older (assistant, user) pairs are
  dropped and only the newest ``max_images_in_history`` images survive (older ones become text).

Throttling / 5xx / connection errors are retried twice with exponential backoff; after that (or
on any non-retryable error) the step is decided by :class:`RulePlanner` and the decision is
labelled ``planner="rule(fallback)"``. Token usage and latency of every call are accumulated in
``session.counters`` (``bedrock_calls``, ``bedrock_input_tokens``, ``bedrock_output_tokens``,
``bedrock_latency_ms``, ``bedrock_retries``, ``bedrock_errors``, ``bedrock_fallbacks``) and copied
into ``Trace.totals["bedrock"]`` by the loop.

Wall clock and spend: the client uses a 20 s read timeout; an optional ``deadline`` (``time.monotonic``
value, set by the API from the Lambda's remaining time) skips model calls and retries that would not
finish in time (``DeadlineExceeded``), and an optional ``budget_guard()`` (e.g. an atomic daily
DynamoDB counter) refuses calls once the deployment's Bedrock budget is spent (``BudgetExhausted``);
both disable Bedrock for the rest of the inspection, which continues with the rule planner.
Ground truth never reaches the model: hidden conditions, labels, mask/image paths and the part id
(dataset-derived ids such as ``pcb1_anomaly_012`` reveal the label) are scrubbed from every payload.

``pick_bedrock_model(region)`` lists inference profiles / foundation models and picks an available
Claude (newest Sonnet, else Haiku) or Amazon Nova model; it is used by ``deploy/aws/deploy.py`` and
the CLI (``python -m inspectagent.agent.planner_bedrock --pick-model``).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import re
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from inspectagent.config import Settings, get_settings
from inspectagent.types import Decision

log = logging.getLogger(__name__)

TERMINAL_TOOLS = frozenset({"finish", "request_human_review"})
RATIONALE_FIELD = "step_rationale"

# Error codes worth retrying (throttling, transient server-side failures).
RETRYABLE_ERROR_CODES = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
        "Throttling",
        "ServiceUnavailableException",
        "ServiceUnavailable",
        "InternalServerException",
        "InternalFailure",
        "ModelNotReadyException",
        "ModelTimeoutException",
        "RequestTimeout",
        "RequestTimeoutException",
        "SlowDown",
    }
)
# Transport-level exceptions (botocore class names) that are retried as well.
RETRYABLE_EXCEPTION_NAMES = frozenset(
    {"EndpointConnectionError", "ConnectTimeoutError", "ReadTimeoutError", "ConnectionClosedError", "ConnectionError"}
)

_ID_RE = re.compile(r"\b(?:cap|q|det|reg|art|rev|review|zoom|meas|m)_\d+(?:_\d+)*\b")
_HIDDEN_STATE_KEYS = frozenset({"ground_truth", "conditions", "mask_path", "image_path", "label_gt"})
# never shown to the model: part ids are built from dataset paths / sample ids ("pcb1_anomaly_012") and
# would leak the ground-truth label; the model only needs the category and the visual evidence
_MODEL_HIDDEN_KEYS = frozenset({"part_id", "sample_id"})
DISABLING_ERROR_CODES = frozenset({"DeadlineExceeded", "BudgetExhausted"})  # stop calling Bedrock for this run


class BedrockPlannerError(RuntimeError):
    """A Bedrock call failed (after retries when the error was retryable)."""

    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.retryable = retryable


# --------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------


def to_jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays, tuples and non-finite floats into strict-JSON values."""
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        if obj.size > 64:
            return f"<array shape={list(obj.shape)} dtype={obj.dtype}>"
        return to_jsonable(obj.tolist())
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (bytes, bytearray)):
        return f"<{len(obj)} bytes>"
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return to_jsonable(to_dict())
        except Exception:  # pragma: no cover - defensive
            pass
    return str(obj)


def _compact(obj: Any, max_list: int = 12, max_str: int = 400, depth: int = 0) -> Any:
    """Bound the size of a JSON value: keep the newest ``max_list`` list items, cut long strings."""
    if depth > 8:
        return "<truncated>"
    if isinstance(obj, dict):
        return {k: _compact(v, max_list, max_str, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        items = obj[-max_list:] if len(obj) > max_list else obj
        out = [_compact(v, max_list, max_str, depth + 1) for v in items]
        if len(obj) > max_list:
            out.insert(0, f"<{len(obj) - max_list} older items omitted>")
        return out
    if isinstance(obj, str) and len(obj) > max_str:
        return obj[:max_str] + "..."
    return obj


def _bounded_json(obj: Any, max_chars: int) -> Any:
    """Return a JSON value whose serialization stays under ``max_chars`` (progressive compaction)."""
    value = to_jsonable(obj)
    for max_list, max_str in ((12, 400), (6, 200), (3, 120), (1, 80)):
        value = _compact(value, max_list, max_str)
        if len(json.dumps(value, ensure_ascii=False)) <= max_chars:
            return value
    text = json.dumps(value, ensure_ascii=False)
    return {"truncated_json": text[:max_chars]}


def _scrub_state(state: Any) -> Any:
    """Remove hidden world/ground-truth keys from a planner state before it reaches the model.

    Besides ground truth and hidden conditions this drops the part/sample id (top level and inside
    ``part``): ids like ``pcb1_anomaly_012`` are derived from dataset folders and reveal the label.
    """
    if not isinstance(state, dict):
        return state
    out = {}
    for k, v in state.items():
        if k in _HIDDEN_STATE_KEYS or k in _MODEL_HIDDEN_KEYS:
            continue
        if k == "part" and isinstance(v, dict):
            v = {pk: pv for pk, pv in v.items() if pk not in _HIDDEN_STATE_KEYS and pk not in _MODEL_HIDDEN_KEYS and pk != "label"}
        out[k] = v
    return out


def extract_ids(*texts: str) -> list[str]:
    """Evidence ids (``cap_1``, ``q_2``, ``det_1``, ``reg_1_2``...) mentioned in the given texts."""
    seen: list[str] = []
    for t in texts:
        for m in _ID_RE.findall(t or ""):
            if m not in seen:
                seen.append(m)
    return seen


def _bump(session: Any, key: str, value: float = 1) -> None:
    """Add ``value`` to ``session.counters[key]`` (creating the counters dict when missing)."""
    counters = getattr(session, "counters", None)
    if counters is None:
        counters = {}
        try:
            session.counters = counters
        except Exception:
            return
    counters[key] = counters.get(key, 0) + value


def _error_code(exc: BaseException) -> tuple[str, bool]:
    """Classify an exception from boto3/botocore (or a stub) into (code, retryable)."""
    if isinstance(exc, BedrockPlannerError):
        return exc.code, exc.retryable
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        code = str(response.get("Error", {}).get("Code", "") or type(exc).__name__)
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        retryable = code in RETRYABLE_ERROR_CODES or (isinstance(status, int) and (status == 429 or status >= 500))
        return code, retryable
    name = type(exc).__name__
    return name, name in RETRYABLE_EXCEPTION_NAMES or isinstance(exc, (TimeoutError, ConnectionError))


def thumbnail_jpeg(data: bytes, long_side: int = 256, quality: int = 80) -> bytes | None:
    """Decode a JPEG/PNG byte string, downscale to ``long_side`` and re-encode as JPEG (None on failure)."""
    import cv2  # local import keeps module import cheap for the MCP/API layers

    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    scale = min(1.0, float(long_side) / float(max(h, w)))
    if scale < 1.0:
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None


# --------------------------------------------------------------------------------------------
# registry / tool spec normalization
# --------------------------------------------------------------------------------------------


def load_default_registry(settings: Settings | None = None) -> Any:
    """Return the project's tool registry (``inspectagent.agent.tools``), trying the usual factories."""
    from inspectagent.agent import tools as tools_mod

    for factory in ("default_registry", "get_registry", "build_registry", "make_registry", "create_registry"):
        fn = getattr(tools_mod, factory, None)
        if callable(fn):
            if settings is not None:
                try:
                    return fn(settings)
                except TypeError:
                    pass
            return fn()
    for attr in ("REGISTRY", "registry", "DEFAULT_REGISTRY"):
        reg = getattr(tools_mod, attr, None)
        if reg is not None and hasattr(reg, "specs"):
            return reg
    cls = getattr(tools_mod, "ToolRegistry", None)
    if cls is not None:
        return cls()
    raise ImportError("inspectagent.agent.tools exposes no tool registry")


def normalize_tool_spec(spec: Any) -> dict:
    """Normalize a registry spec (ToolSpec object, plain dict or Bedrock ``toolSpec``) to
    ``{"name", "description", "input_schema"}``."""
    if isinstance(spec, Mapping) and "toolSpec" in spec:
        spec = spec["toolSpec"]
    if isinstance(spec, Mapping):
        name = spec.get("name")
        description = spec.get("description", "")
        schema = None
        for key in ("input_schema", "inputSchema", "parameters", "schema"):
            if key in spec:
                schema = spec[key]
                break
    else:
        name = getattr(spec, "name", None)
        description = getattr(spec, "description", "")
        schema = getattr(spec, "input_schema", None) or getattr(spec, "parameters", None)
    if isinstance(schema, Mapping) and "json" in schema and "type" not in schema:
        schema = schema["json"]
    if not name:
        raise ValueError(f"tool spec without name: {spec!r}")
    schema = copy.deepcopy(dict(schema)) if isinstance(schema, Mapping) else {"type": "object", "properties": {}}
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return {"name": str(name), "description": str(description or name), "input_schema": schema}


def registry_tool_specs(registry: Any) -> list[dict]:
    """Normalized specs of every tool in ``registry`` (``registry.specs()``)."""
    return [normalize_tool_spec(s) for s in registry.specs()]


def bedrock_tool_config(specs: Iterable[dict], tool_choice: str | None = "any", rationale_field: str | None = RATIONALE_FIELD) -> dict:
    """Build a Converse ``toolConfig`` from normalized specs (optionally adding a rationale argument)."""
    tools = []
    for s in specs:
        schema = copy.deepcopy(s["input_schema"])
        if rationale_field and rationale_field not in schema.get("properties", {}):
            schema.setdefault("properties", {})[rationale_field] = {
                "type": "string",
                "description": "One or two sentences citing the evidence ids and numbers that motivate this call "
                "(recorded in the trace, not passed to the tool).",
            }
        tools.append({"toolSpec": {"name": s["name"], "description": s["description"], "inputSchema": {"json": schema}}})
    cfg: dict = {"tools": tools}
    if tool_choice == "any":
        cfg["toolChoice"] = {"any": {}}
    elif tool_choice == "auto":
        cfg["toolChoice"] = {"auto": {}}
    elif tool_choice:
        cfg["toolChoice"] = {"tool": {"name": tool_choice}}
    return cfg


# --------------------------------------------------------------------------------------------
# step / observation discovery
# --------------------------------------------------------------------------------------------


@dataclass
class StepRecord:
    """The most recently executed step as seen by the planner."""

    tool: str | None
    args: dict
    observation: Any
    step: int | None = None
    overridden: bool = False


def _normalize_step(item: Any) -> StepRecord | None:
    if item is None:
        return None
    decision = getattr(item, "decision", None)
    if decision is not None and hasattr(item, "observation"):
        return StepRecord(
            tool=getattr(decision, "tool", None),
            args=dict(getattr(decision, "args", {}) or {}),
            observation=item.observation,
            step=getattr(item, "step", None),
            overridden=bool(getattr(item, "overridden_by_policy", False)),
        )
    if isinstance(item, Mapping):
        dec = item.get("decision") if isinstance(item.get("decision"), Mapping) else {}
        tool = item.get("tool") or dec.get("tool") or item.get("name")
        args = item.get("args") or dec.get("args") or {}
        obs = None
        for key in ("observation", "obs", "result"):
            if key in item:
                obs = item[key]
                break
        if tool is None and obs is None:
            return None
        return StepRecord(
            tool=tool,
            args=dict(args) if isinstance(args, Mapping) else {},
            observation=obs,
            step=next((item[k] for k in ("step", "n") if isinstance(item.get(k), int)), None),
            overridden=bool(item.get("overridden_by_policy", False)),
        )
    return None


def find_last_step(state: Any, session: Any) -> StepRecord | None:
    """Locate the newest executed step (tool + observation) on the session or in the planner state.

    Sources, in order: ``session.steps`` / ``session.trace.steps`` / ``session.history`` /
    ``session.last_step``, then ``state["last_step"|"steps"|"history"]``. When the record found has
    no observation, ``session.last_observation`` (or ``state["last_observation"]``) fills it in.
    """
    state = state if isinstance(state, Mapping) else {}
    rec: StepRecord | None = None
    owners = [(session, "steps"), (getattr(session, "trace", None), "steps"), (session, "history")]
    for owner, attr in owners:
        seq = getattr(owner, attr, None) if owner is not None else None
        if isinstance(seq, (list, tuple)) and seq:
            rec = _normalize_step(seq[-1])
            if rec is not None:
                break
    if rec is None:
        rec = _normalize_step(getattr(session, "last_step", None)) or _normalize_step(state.get("last_step"))
    if rec is None:
        for key in ("steps", "history", "trace"):
            seq = state.get(key)
            if isinstance(seq, list) and seq:
                rec = _normalize_step(seq[-1])
                if rec is not None:
                    break
    last_obs = getattr(session, "last_observation", None)
    if last_obs is None:
        last_obs = state.get("last_observation")
    if rec is None and last_obs is not None:
        return StepRecord(tool=state.get("last_tool"), args={}, observation=last_obs)
    if rec is not None and rec.observation is None and last_obs is not None:
        rec.observation = last_obs
    return rec


# --------------------------------------------------------------------------------------------
# conversation state
# --------------------------------------------------------------------------------------------


@dataclass
class BedrockConversation:
    """Converse history and bookkeeping for one inspection session."""

    model_id: str
    system: list[dict]
    tool_config: dict
    tool_names: set[str]
    own_rationale_tools: set[str]  # tools whose schema already defines the rationale field
    messages: list[dict] = field(default_factory=list)
    pending: list[dict] = field(default_factory=list)  # toolUse blocks of the last assistant turn
    carry: list[dict] = field(default_factory=list)  # toolResult blocks of a failed call (resent next time)
    fallback_steps: list[dict] = field(default_factory=list)  # steps decided by the fallback since last answer
    observed: list[StepRecord] = field(default_factory=list)  # pushed via BedrockPlanner.observe()
    sent_artifacts: set[str] = field(default_factory=set)
    last_obs_hash: str | None = None
    calls: list[dict] = field(default_factory=list)
    decisions: int = 0
    trimmed_pairs: int = 0
    consecutive_failures: int = 0
    disabled_reason: str | None = None


SYSTEM_PROMPT_TEMPLATE = """\
You are the planner of InspectAgent, an automated visual-inspection cell for '{category}' parts.
You drive an actively controllable camera (view centre cx/cy, zoom, exposure, light mode, autofocus) and \
OpenCV 5 analysis tools. Each turn you receive the latest tool observation as JSON (plus a compact state of \
all evidence so far) and sometimes a small thumbnail of the newest image.

INSPECTION PROTOCOL
1. Quality first. Capture the full view (cx=0.5, cy=0.5, zoom=1) and call assess_quality on every new \
capture before you analyse it.
2. Fix capture problems with evidence-driven recaptures of the full view (at most 3): glare -> light="diffuse" \
and exposure_ev +0.75; blurred -> autofocus=true; under/over-exposed -> exposure_ev += suggested_exposure_delta \
(stay within [-3, 3]); combine all fixes in one recapture. Specular rule: "overexposed" caused only by clipping \
(clipped_frac > max_clipped_frac while mean_luma <= max_luma, no "glare" issue) under light="ring" at \
exposure_ev <= +0.5 is a specular reflection the glare detector missed -> switch to light="diffuse" with \
exposure_ev +0.75 instead of lowering the exposure. If an exposure correction flips under- into overexposed \
(or back), use the EV halfway between the two captures.
3. Detection. Run detect_anomalies(detector="fast") on the best full view. If that result is not reliable \
(registration failed or too little valid area) escalate to detector="deep" on the full view (it is \
alignment-free; deep results on zoomed views are never reliable). Results of both detectors on a capture with \
blocking quality issues (glare, clipping, crushed shadows, blur, heavy noise) are unreliable, so fix the capture \
first; moderate noise is tolerated.
4. Zoom on candidates. Whenever un-inspected or inconclusive candidate regions exist (listed in \
unresolved_candidates) call inspect_region on the strongest ones (zoom 3, at most 3) - also when P(anomalous) \
<= {auto_pass}: a small defect can be diluted to a low image-level p while its region is still emitted. A \
confirmed region -> measure_defect -> finish(label="FAIL"). If a zoom reports an anomaly away from the \
candidate (state unlocalized_zoom_alarms), inspect_region on that zoom result's own region.
5. Deep on ambiguity. If zoom inspections refute every candidate but doubt remains, run \
detect_anomalies(detector="deep") on the full view.
6. Ask a human when unsure. If the evidence is still inconclusive, or the budget is nearly used up, call \
request_human_review with the evidence and your proposed label.

BUDGET (hard limits enforced by the cell)
At most {max_steps} tool calls, {max_captures} captures, {max_deep_calls} deep-detector calls and \
{max_sim_time_ms:.0f} ms of simulated camera time per part. Prefer the cheapest action that resolves the current \
uncertainty; do not repeat an action whose result you already have.

SAFETY RULES
- Never claim PASS without reliable evidence: finish(label="PASS") only after a reliable detector result on a \
good-quality full view with fused P(anomalous) <= {auto_pass}, and only when no un-inspected or inconclusive \
candidate region remains (or the zoom candidates are exhausted and the deep detector has corroborated) and no \
unlocalized zoom alarm stands. PASS is refused otherwise.
- If the state shows contested_evidence (an aligned detector alarm contradicting a reliable result on the same \
capture), resolve it with inspect_region on its regions or ask a human; PASS is refused while it stands.
- finish(label="FAIL") on your own only when P(anomalous) >= {auto_fail} and a defect region is localized. Every \
uncertain FAIL needs human approval: call request_human_review(proposed_label="FAIL", ...).
- A human-control policy vets each action and may replace it (for example with a human review); accept its outcome.
- You do not know the ground truth. Use only ids that appear in observations (cap_*, q_*, det_*, reg_*, ...).

OUTPUT CONTRACT
- Call exactly one tool per turn; additional tool calls are ignored.
- Before the tool call write one or two short sentences of rationale that cite the evidence ids and numbers \
behind the action (e.g. "q_1: glare_frac=0.031 > 0.004 -> diffuse light, EV +0.75"), and put the same \
sentence in the tool's "{rationale_field}" argument.
- Arguments must satisfy the tool's JSON schema; numeric ranges are clamped by the cell.
"""


def build_system_prompt(settings: Settings | None = None, category: str = "unknown", rationale_field: str = RATIONALE_FIELD) -> str:
    """System prompt: inspection protocol, budget, safety rules and output contract."""
    settings = settings or get_settings()
    p = settings.policy
    return SYSTEM_PROMPT_TEMPLATE.format(
        category=category,
        auto_pass=p.auto_pass_max_prob,
        auto_fail=p.auto_fail_min_prob,
        max_steps=p.max_steps,
        max_captures=p.max_captures,
        max_deep_calls=p.max_deep_calls,
        max_sim_time_ms=p.max_sim_time_ms,
        rationale_field=rationale_field,
    )


# --------------------------------------------------------------------------------------------
# planner
# --------------------------------------------------------------------------------------------


class BedrockPlanner:
    """Amazon Bedrock Converse tool-use planner with RulePlanner fallback."""

    name = "bedrock"

    def __init__(
        self,
        model_id: str | None = None,
        client: Any | None = None,
        region: str | None = None,
        max_tokens: int = 1024,
        fallback: Any | None = None,
        *,
        registry: Any | None = None,
        settings: Settings | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
        backoff_base_s: float = 0.6,
        backoff_max_s: float = 8.0,
        send_images: bool = True,
        thumbnail_side: int = 256,
        max_turn_pairs: int = 8,
        max_images_in_history: int = 2,
        max_observation_chars: int = 6000,
        max_state_chars: int = 6000,
        tool_choice: str | None = "any",
        rationale_field: str | None = RATIONALE_FIELD,
        tool_result_status: bool = True,
        prompt_cache: bool = False,
        max_consecutive_failures: int = 2,
        max_nudges: int = 1,
        sleep: Callable[[float], None] = time.sleep,
        deadline: float | None = None,
        read_timeout_s: float = 20.0,
        min_call_s: float = 3.0,
        deadline_reserve_s: float = 6.0,
        budget_guard: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        specular_guard: bool = True,
    ) -> None:
        self.settings = settings
        self.model_id = (model_id or "").strip()
        self.client = client
        self.region = region
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self._fallback = fallback
        self._registry = registry
        self.max_retries = max(0, int(max_retries))
        self.backoff_base_s = float(backoff_base_s)
        self.backoff_max_s = float(backoff_max_s)
        self.send_images = bool(send_images)
        self.thumbnail_side = int(thumbnail_side)
        self.max_turn_pairs = max(1, int(max_turn_pairs))
        self.max_images_in_history = max(0, int(max_images_in_history))
        self.max_observation_chars = int(max_observation_chars)
        self.max_state_chars = int(max_state_chars)
        self.tool_choice = tool_choice
        self.rationale_field = rationale_field
        self.tool_result_status = bool(tool_result_status)
        self.prompt_cache = bool(prompt_cache)
        self.max_consecutive_failures = max(1, int(max_consecutive_failures))
        self.max_nudges = max(0, int(max_nudges))
        self._sleep = sleep
        # wall-clock control: ``deadline`` is a ``clock()`` (time.monotonic) value - e.g. derived from the Lambda's
        # remaining time - before which the whole inspection must end; model calls are skipped once less than
        # ``min_call_s`` (+ ``deadline_reserve_s`` for the remaining tool steps and persistence) is left
        self.deadline = deadline
        self.read_timeout_s = max(1.0, float(read_timeout_s))
        self.min_call_s = max(0.0, float(min_call_s))
        self.deadline_reserve_s = max(0.0, float(deadline_reserve_s))
        self._clock = clock
        # spend control: ``budget_guard()`` is consulted before every Converse call (one call = one unit) and
        # returns False when the shared (e.g. daily, per deployment) Bedrock call budget is exhausted
        self.budget_guard = budget_guard
        self.specular_guard = bool(specular_guard)
        self._model_resolution_error: str | None = None
        self._conversations: dict[str, BedrockConversation] = {}  # used only if the session refuses attributes
        self.last_call: dict | None = None

    # ----------------------------------------------------------------- construction helpers
    @classmethod
    def from_settings(cls, settings: Settings | None = None, client: Any | None = None, **kwargs: Any) -> "BedrockPlanner":
        """Planner configured from :class:`Settings` (model id ``""``/``"auto"`` is picked lazily)."""
        settings = settings or get_settings()
        return cls(
            model_id=settings.bedrock_model_id or "auto",
            client=client,
            region=settings.aws_region,
            max_tokens=settings.bedrock_max_tokens,
            settings=settings,
            **kwargs,
        )

    @property
    def planner_label(self) -> str:
        """Value written into ``Decision.planner`` for model-made decisions."""
        return f"bedrock:{self.model_id or 'auto'}"

    @property
    def fallback(self) -> Any:
        """Fallback planner (``RulePlanner`` by default, created lazily)."""
        if self._fallback is None:
            from inspectagent.agent.planner_rule import RulePlanner

            self._fallback = RulePlanner()
        return self._fallback

    def _settings_for(self, session: Any) -> Settings:
        s = getattr(session, "settings", None)
        if isinstance(s, Settings):
            return s
        return self.settings or get_settings()

    def _registry_for(self, session: Any) -> Any:
        if self._registry is not None:
            return self._registry
        reg = getattr(session, "registry", None)
        if reg is not None and hasattr(reg, "specs"):
            return reg
        self._registry = load_default_registry(self._settings_for(session))
        return self._registry

    def _get_client(self) -> Any:
        if self.client is None:
            import boto3
            from botocore.config import Config

            region = self.region or (self.settings or get_settings()).aws_region
            # a short read timeout: one hung call must not consume the whole (Lambda) request budget
            self.client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                config=Config(retries={"max_attempts": 1, "mode": "standard"}, connect_timeout=5, read_timeout=self.read_timeout_s),
            )
        return self.client

    def time_left(self) -> float | None:
        """Seconds available for model calls before the deadline (after the reserve); None without a deadline."""
        if self.deadline is None:
            return None
        return float(self.deadline) - self._clock() - self.deadline_reserve_s

    def _resolve_model(self) -> str:
        if self.model_id and self.model_id.lower() != "auto":
            return self.model_id
        if self._model_resolution_error:
            raise BedrockPlannerError("NoModel", self._model_resolution_error)
        region = self.region or (self.settings or get_settings()).aws_region
        try:
            picked = pick_bedrock_model(region)
        except Exception as exc:  # no credentials, no permission, no network ...
            self._model_resolution_error = f"model auto-selection failed: {type(exc).__name__}: {exc}"
            raise BedrockPlannerError("NoModel", self._model_resolution_error) from exc
        if not picked:
            self._model_resolution_error = f"no Claude/Nova model available in {region}"
            raise BedrockPlannerError("NoModel", self._model_resolution_error)
        self.model_id = picked
        return picked

    # ----------------------------------------------------------------- conversation storage
    @staticmethod
    def _session_key(session: Any) -> str:
        return str(getattr(session, "session_id", None) or id(session))

    MEMORY_KEY = "bedrock"

    def conversation(self, session: Any) -> BedrockConversation | None:
        """The conversation stored for ``session`` (None before the first call)."""
        memory = getattr(session, "planner_memory", None)
        if isinstance(memory, dict) and isinstance(memory.get(self.MEMORY_KEY), BedrockConversation):
            return memory[self.MEMORY_KEY]
        conv = getattr(session, "bedrock_conversation", None)
        if isinstance(conv, BedrockConversation):
            return conv
        return self._conversations.get(self._session_key(session))

    def _store_conversation(self, session: Any, conv: BedrockConversation | None) -> None:
        memory = getattr(session, "planner_memory", None)
        if isinstance(memory, dict):  # SPEC: history is kept in the session
            if conv is None:
                memory.pop(self.MEMORY_KEY, None)
            else:
                memory[self.MEMORY_KEY] = conv
            return
        try:
            session.bedrock_conversation = conv
        except Exception:
            if conv is None:
                self._conversations.pop(self._session_key(session), None)
            else:
                self._conversations[self._session_key(session)] = conv

    def _new_conversation(self, session: Any, model_id: str) -> BedrockConversation:
        settings = self._settings_for(session)
        part = getattr(session, "part", None)
        category = getattr(part, "category", None) or "unknown"
        specs = registry_tool_specs(self._registry_for(session))
        own = {s["name"] for s in specs if self.rationale_field and self.rationale_field in s["input_schema"].get("properties", {})}
        system: list[dict] = [{"text": build_system_prompt(settings, category, self.rationale_field or "rationale")}]
        if self.prompt_cache:
            system.append({"cachePoint": {"type": "default"}})
        conv = BedrockConversation(
            model_id=model_id,
            system=system,
            tool_config=bedrock_tool_config(specs, self.tool_choice, self.rationale_field),
            tool_names={s["name"] for s in specs},
            own_rationale_tools=own,
        )
        self._store_conversation(session, conv)
        return conv

    def _try_new_conversation(self, session: Any, model_id: str) -> BedrockConversation | None:
        try:
            return self._new_conversation(session, model_id)
        except Exception as exc:  # registry unavailable -> cannot build a toolConfig
            log.warning("bedrock planner: cannot build tool config: %s", exc)
            return None

    def reset(self, session: Any) -> None:
        """Forget the conversation of ``session``."""
        self._store_conversation(session, None)
        self._conversations.pop(self._session_key(session), None)

    def observe(self, step: Any, session: Any) -> None:
        """Optional hook: hand the executed step (``Step`` or dict) to the planner explicitly.

        ``next_action`` otherwise discovers the newest step on the session / in the planner state.
        """
        conv = self.conversation(session)
        rec = _normalize_step(step)
        if conv is not None and rec is not None:
            conv.observed.append(rec)
            del conv.observed[:-4]

    def call_log(self, session: Any) -> list[dict]:
        """Per-call usage records (tokens, latency, stop reason, tool) for ``session``."""
        conv = self.conversation(session)
        return list(conv.calls) if conv else []

    # ----------------------------------------------------------------- main entry point
    def next_action(self, state: dict, session: Any) -> Decision:
        """Ask the model for the next tool call; fall back to the rule planner on failure."""
        state = state if isinstance(state, dict) else {}
        conv = self.conversation(session)
        if conv is not None and conv.disabled_reason:
            return self._fallback_decision(state, session, conv, conv.disabled_reason, count_error=False)
        try:
            model_id = self._resolve_model()
        except BedrockPlannerError as exc:
            if conv is None:
                conv = self._try_new_conversation(session, self.model_id or "auto")
            if conv is not None:
                conv.disabled_reason = str(exc)
            return self._fallback_decision(state, session, conv, str(exc))
        if conv is None:
            conv = self._try_new_conversation(session, model_id)
            if conv is None:
                return self._fallback_decision(state, session, None, "tool registry unavailable (no toolConfig)")

        record = self._latest_step(conv, state, session)
        sent_before = set(conv.sent_artifacts)
        user_content = self._build_user_content(conv, state, session, record)
        start_len = len(conv.messages)
        carry_before, fallback_before = list(conv.carry), list(conv.fallback_steps)
        self._append_user(conv, user_content)
        conv.carry, conv.fallback_steps = [], []
        self._trim_history(conv)
        start_len = min(start_len, len(conv.messages) - 1)

        step_no = self._step_number(state, session, conv)
        try:
            response = self._converse(conv, session, step_no)
            tool_uses, texts, message = self._parse_response(response)
            nudges = 0
            while not tool_uses and nudges < self.max_nudges:
                nudges += 1
                conv.messages.append(message)
                conv.messages.append(
                    {"role": "user", "content": [{"text": "Protocol reminder: respond with exactly one tool call now."}]}
                )
                response = self._converse(conv, session, step_no)
                tool_uses, texts, message = self._parse_response(response)
            if not tool_uses:
                raise BedrockPlannerError("NoToolUse", "the model did not call a tool")
        except BedrockPlannerError as exc:
            # roll back this step so the Converse history keeps alternating roles
            del conv.messages[start_len:]
            conv.carry = [b for b in user_content if "toolResult" in b] or carry_before
            conv.fallback_steps = fallback_before
            conv.sent_artifacts = sent_before
            conv.consecutive_failures += 1
            if exc.code in DISABLING_ERROR_CODES:
                conv.disabled_reason = f"Bedrock disabled for this inspection: {exc}"
            elif conv.consecutive_failures >= self.max_consecutive_failures:
                conv.disabled_reason = f"Bedrock disabled after {conv.consecutive_failures} failed steps (last: {exc})"
            return self._fallback_decision(state, session, conv, str(exc))

        conv.messages.append(message)
        conv.pending = tool_uses
        conv.consecutive_failures = 0
        conv.decisions += 1
        if record is not None:
            conv.last_obs_hash = self._record_hash(record)
        decision = self._decision_from_tool_use(conv, tool_uses[0], texts, step_no)
        return self._apply_specular_guard(decision, session) if self.specular_guard else decision

    def _apply_specular_guard(self, decision: Decision, session: Any) -> Decision:
        """Rewrite an exposure-only darkening recapture into the specular remedy (diffuse light, EV +0.75).

        Mirrors ``RulePlanner.specular_suspected``: clipping-only overexposure (mean luma in range, no glare
        segmented) of the newest assessed full view under ring light at nominal gain is a specular reflection;
        darkening the whole part would leave the (now unsaturated) highlight in the image unmasked.
        """
        if decision.tool != "capture_image":
            return decision
        captures, quality = getattr(session, "captures", None), getattr(session, "quality", None)
        if not isinstance(captures, Mapping) or not isinstance(quality, Mapping):
            return decision
        full = [cid for cid, cap in captures.items() if float(cap.settings.view.zoom) <= 1.05 and cid in quality]
        if not full:
            return decision
        cid = full[-1]
        cur, q = captures[cid].settings, quality[cid]
        settings = getattr(session, "settings", None)
        qc = settings.quality if isinstance(settings, Settings) else (self.settings or get_settings()).quality
        from inspectagent.agent.planner_rule import GLARE_EV_COMP, RulePlanner

        if not RulePlanner.specular_suspected(q, cur, qc):
            return decision
        args = dict(decision.args)
        try:
            zoom = float(args.get("zoom", 1.0))
            new_ev = float(args.get("exposure_ev", 0.0))
        except (TypeError, ValueError):
            return decision
        if zoom > 1.05 or args.get("light", "ring") != "ring" or new_ev >= float(cur.exposure_ev):
            return decision
        args.update(light="diffuse", exposure_ev=round(min(float(cur.exposure_ev) + GLARE_EV_COMP, 3.0), 3))
        note = (
            f"[guard: {self.quality_id(session, cid)} clipping-only overexposure (clipped_frac={q.clipped_frac:.3f}>"
            f"{qc.max_clipped_frac:.3f}, mean_luma={q.mean_luma:.1f}<={qc.max_luma:.0f}) under ring light -> specular "
            f"reflection: light ring->diffuse, EV {GLARE_EV_COMP:+.2f} instead of darkening to EV {new_ev:+.2f}] "
        )
        return replace(decision, args=args, rationale=note + decision.rationale, evidence=list(dict.fromkeys(decision.evidence + [cid])))

    @staticmethod
    def quality_id(session: Any, capture_id: str) -> str:
        ids = getattr(session, "quality_ids", None)
        return str(ids.get(capture_id, capture_id)) if isinstance(ids, Mapping) else capture_id

    # ----------------------------------------------------------------- message building
    def _latest_step(self, conv: BedrockConversation, state: dict, session: Any) -> StepRecord | None:
        if conv.observed:
            rec = conv.observed[-1]
            conv.observed.clear()
            return rec
        rec = find_last_step(state, session)
        if rec is not None and conv.last_obs_hash is not None and self._record_hash(rec) == conv.last_obs_hash:
            return None  # stale: no new step since the last model call
        return rec

    @staticmethod
    def _record_hash(rec: StepRecord) -> str:
        """Identity of an executed step (step number + tool + observation) for staleness checks."""
        try:
            text = json.dumps(to_jsonable(rec.observation), sort_keys=True, ensure_ascii=False)
        except Exception:  # pragma: no cover - defensive
            text = repr(rec.observation)
        return hashlib.sha1(f"{rec.step}|{rec.tool}|{text}".encode("utf-8")).hexdigest()

    def _state_payload(self, state: dict) -> Any:
        # the observation travels separately, so the state copy of it is dropped to save tokens
        scrubbed = {k: v for k, v in _scrub_state(state).items() if k != "last_observation"}
        return _bounded_json(scrubbed, self.max_state_chars)

    def _initial_text(self, state: dict, session: Any) -> str:
        # no part id: dataset-derived ids ("pcb1_anomaly_012") would reveal the ground truth
        part = getattr(session, "part", None)
        category = getattr(part, "category", None) or "unknown"
        state_json = json.dumps(self._state_payload(state), ensure_ascii=False)
        return (
            f"Inspect the part presented to the cell (category '{category}'). Decide from the visual evidence only. "
            f"Follow the inspection protocol, call exactly one tool per turn and cite evidence ids in your rationale.\n"
            f"Current state JSON:\n{state_json}"
        )

    def _tool_result_blocks(self, conv: BedrockConversation, record: StepRecord | None, state: dict) -> list[dict]:
        blocks: list[dict] = []
        for i, tu in enumerate(conv.pending):
            tool_use_id = tu.get("toolUseId", "")
            if i == 0:
                obs = record.observation if record is not None else None
                payload: dict = {
                    "observation": _bounded_json(obs, self.max_observation_chars) if obs is not None else None,
                    "state": self._state_payload(state),
                }
                if obs is None:
                    payload["note"] = "observation not available; use the state summary"
                if record is not None and record.tool and record.tool != tu.get("name"):
                    payload["policy_override"] = {
                        "requested_tool": tu.get("name"),
                        "executed_tool": record.tool,
                        "executed_args": to_jsonable(record.args),
                        "note": "The human-control policy replaced the requested action.",
                    }
                is_error = isinstance(obs, Mapping) and bool(obs.get("error"))
            else:
                payload = {"error": "Ignored: exactly one tool call per turn is executed."}
                is_error = True
            result: dict = {"toolUseId": tool_use_id, "content": [{"json": payload}]}
            if self.tool_result_status:
                result["status"] = "error" if is_error else "success"
            blocks.append({"toolResult": result})
        return blocks

    def _fallback_note(self, conv: BedrockConversation, record: StepRecord | None, state: dict, include_state: bool) -> str:
        lines = ["Bedrock was unavailable for some steps; the rule-based fallback planner executed:"]
        for fs in conv.fallback_steps:
            lines.append(f"- step {fs.get('step')}: {fs.get('tool')}({json.dumps(fs.get('args', {}), ensure_ascii=False)})")
        obs = record.observation if record is not None else None
        latest = {"observation": _bounded_json(obs, self.max_observation_chars) if obs is not None else None}
        lines.append("Latest observation JSON: " + json.dumps(latest, ensure_ascii=False))
        if include_state:
            lines.append("Current state JSON: " + json.dumps(self._state_payload(state), ensure_ascii=False))
        return "\n".join(lines)

    def _image_blocks(self, conv: BedrockConversation, record: StepRecord | None, session: Any) -> list[dict]:
        if not self.send_images:
            return []
        artifacts = getattr(session, "artifacts", None)
        if not isinstance(artifacts, Mapping) or not artifacts:
            return []
        chosen: str | None = None
        if record is not None:
            refs: list[str] = []

            def walk(o: Any) -> None:
                if isinstance(o, str):
                    if o in artifacts and o not in refs:
                        refs.append(o)
                elif isinstance(o, Mapping):
                    for v in o.values():
                        walk(v)
                elif isinstance(o, (list, tuple)):
                    for v in o:
                        walk(v)

            walk(record.observation)
            fresh = [r for r in refs if r not in conv.sent_artifacts]
            chosen = fresh[-1] if fresh else None
        if chosen is None:
            unsent = [k for k in artifacts if k not in conv.sent_artifacts]
            chosen = unsent[-1] if unsent else None
        if chosen is None:
            return []
        conv.sent_artifacts.update(artifacts.keys())  # older unsent artifacts are superseded by the newest
        data = artifacts.get(chosen)
        if not isinstance(data, (bytes, bytearray)):
            return []
        jpeg = thumbnail_jpeg(bytes(data), self.thumbnail_side)
        if jpeg is None:
            return []
        return [
            {"text": f"Thumbnail of {chosen} (newest image, long side <= {self.thumbnail_side}px):"},
            {"image": {"format": "jpeg", "source": {"bytes": jpeg}}},
        ]

    def _build_user_content(self, conv: BedrockConversation, state: dict, session: Any, record: StepRecord | None) -> list[dict]:
        blocks: list[dict] = []
        state_included = True
        if not conv.messages:
            blocks.append({"text": self._initial_text(state, session)})
        elif conv.carry:
            blocks.extend(conv.carry)
            state_included = False  # the carried results hold an older state
        elif conv.pending:
            blocks.extend(self._tool_result_blocks(conv, record, state))
        else:
            state_included = False
        if conv.fallback_steps:
            blocks.append({"text": self._fallback_note(conv, record, state, include_state=not state_included)})
            state_included = True
        if conv.trimmed_pairs and conv.messages:
            blocks.append(
                {"text": f"({conv.trimmed_pairs} older turns were trimmed from the history; the state JSON summarises all evidence.)"}
            )
        if not blocks:
            blocks.append({"text": "Choose the next action. Current state JSON: " + json.dumps(self._state_payload(state), ensure_ascii=False)})
        blocks.extend(self._image_blocks(conv, record, session))
        return blocks

    @staticmethod
    def _append_user(conv: BedrockConversation, content: list[dict]) -> None:
        if conv.messages and conv.messages[-1].get("role") == "user":
            conv.messages[-1] = {"role": "user", "content": list(conv.messages[-1]["content"]) + content}
        else:
            conv.messages.append({"role": "user", "content": content})

    def _trim_history(self, conv: BedrockConversation) -> None:
        """Keep the first user message plus the newest (assistant, user) pairs; drop old images."""
        msgs = conv.messages
        max_tail = 2 * self.max_turn_pairs
        if len(msgs) - 1 > max_tail:
            tail = msgs[1:]
            drop = len(tail) - max_tail
            drop += drop % 2  # always drop whole (assistant, user) pairs
            conv.trimmed_pairs += drop // 2
            conv.messages = [msgs[0]] + tail[drop:]
            msgs = conv.messages
        # bound the number of image blocks (newest first)
        kept = 0
        for idx in range(len(msgs) - 1, -1, -1):
            m = msgs[idx]
            if m.get("role") != "user" or not any("image" in b for b in m["content"]):
                continue
            new_content = []
            for b in reversed(m["content"]):
                if "image" in b:
                    if kept < self.max_images_in_history:
                        kept += 1
                        new_content.append(b)
                    else:
                        new_content.append({"text": "[older image omitted from history]"})
                else:
                    new_content.append(b)
            msgs[idx] = {"role": "user", "content": list(reversed(new_content))}

    # ----------------------------------------------------------------- model call
    def _converse(self, conv: BedrockConversation, session: Any, step_no: int) -> dict:
        client = self._get_client()
        kwargs: dict = {
            "modelId": conv.model_id,
            "messages": list(conv.messages),
            "system": conv.system,
            "inferenceConfig": {"maxTokens": self.max_tokens, "temperature": self.temperature},
            "toolConfig": conv.tool_config,
        }
        attempts = 0
        while True:
            attempts += 1
            left = self.time_left()
            if left is not None and left < self.min_call_s:
                _bump(session, "bedrock_deadline_skips")
                raise BedrockPlannerError(
                    "DeadlineExceeded", f"{max(left, 0.0):.1f}s left for model calls before the inspection deadline (< {self.min_call_s:.1f}s)"
                )
            if self.budget_guard is not None and not self.budget_guard():
                _bump(session, "bedrock_budget_refusals")
                raise BedrockPlannerError("BudgetExhausted", "the Bedrock call budget of this deployment is exhausted")
            t0 = time.perf_counter()
            try:
                response = client.converse(**kwargs)
            except Exception as exc:
                latency = (time.perf_counter() - t0) * 1000.0
                code, retryable = _error_code(exc)
                _bump(session, "bedrock_errors")
                _bump(session, "bedrock_latency_ms", latency)
                rec = {"step": step_no, "attempt": attempts, "ok": False, "error": code, "latency_ms": round(latency, 1)}
                conv.calls.append(rec)
                self.last_call = rec
                if retryable and attempts <= self.max_retries:
                    delay = min(self.backoff_max_s, self.backoff_base_s * (2 ** (attempts - 1)))
                    left = self.time_left()
                    if left is not None and left < delay + self.min_call_s:
                        _bump(session, "bedrock_deadline_skips")
                        raise BedrockPlannerError(
                            "DeadlineExceeded", f"{code}: no time left for a retry ({max(left, 0.0):.1f}s before the deadline)"
                        ) from exc
                    _bump(session, "bedrock_retries")
                    log.warning("bedrock converse %s (attempt %d), retrying in %.1fs", code, attempts, delay)
                    self._sleep(delay)
                    continue
                raise BedrockPlannerError(code, str(exc), retryable) from exc
            latency = (time.perf_counter() - t0) * 1000.0
            usage = response.get("usage") or {}
            metrics = response.get("metrics") or {}
            in_tok = int(usage.get("inputTokens", 0) or 0)
            out_tok = int(usage.get("outputTokens", 0) or 0)
            _bump(session, "bedrock_calls")
            _bump(session, "bedrock_input_tokens", in_tok)
            _bump(session, "bedrock_output_tokens", out_tok)
            _bump(session, "bedrock_latency_ms", latency)
            cache_read = int(usage.get("cacheReadInputTokens", 0) or 0)
            if cache_read:
                _bump(session, "bedrock_cache_read_tokens", cache_read)
            rec = {
                "step": step_no,
                "attempt": attempts,
                "ok": True,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency_ms": round(latency, 1),
                "service_latency_ms": metrics.get("latencyMs"),
                "stop_reason": response.get("stopReason"),
                "model_id": conv.model_id,
            }
            conv.calls.append(rec)
            self.last_call = rec
            return response

    @staticmethod
    def _parse_response(response: Mapping) -> tuple[list[dict], list[str], dict]:
        try:
            message = response["output"]["message"]
            content = list(message.get("content") or [])
        except (KeyError, TypeError, AttributeError) as exc:
            raise BedrockPlannerError("MalformedResponse", f"no output message: {exc}") from exc
        tool_uses = [b["toolUse"] for b in content if isinstance(b, Mapping) and "toolUse" in b]
        texts = [b["text"] for b in content if isinstance(b, Mapping) and isinstance(b.get("text"), str)]
        if not content:
            content = [{"text": "(empty response)"}]
        return tool_uses, texts, {"role": "assistant", "content": copy.deepcopy(content)}

    def _decision_from_tool_use(self, conv: BedrockConversation, tool_use: Mapping, texts: list[str], step_no: int) -> Decision:
        name = str(tool_use.get("name", ""))
        raw = tool_use.get("input")
        args = dict(raw) if isinstance(raw, Mapping) else {}
        extra = None
        if self.rationale_field and name not in conv.own_rationale_tools:
            extra = args.pop(self.rationale_field, None)
        text = " ".join(t.strip() for t in texts if t and t.strip())
        rationale = text or (str(extra).strip() if extra else "") or f"Model selected {name}."
        if len(rationale) > 800:
            rationale = rationale[:800] + "..."
        evidence = extract_ids(rationale, json.dumps(to_jsonable(args), ensure_ascii=False))
        return Decision(step=step_no, tool=name, args=args, rationale=rationale, evidence=evidence, planner=self.planner_label)

    def _fallback_decision(
        self, state: dict, session: Any, conv: BedrockConversation | None, reason: str, count_error: bool = True
    ) -> Decision:
        _bump(session, "bedrock_fallbacks")
        decision = self.fallback.next_action(state, session)
        short = reason if len(reason) <= 160 else reason[:160] + "..."
        decision = replace(decision, planner="rule(fallback)", rationale=f"[Bedrock fallback: {short}] {decision.rationale}")
        if conv is not None:
            conv.fallback_steps.append({"step": decision.step, "tool": decision.tool, "args": to_jsonable(decision.args)})
        if count_error:
            log.warning("bedrock planner fell back to rule planner: %s", short)
        return decision

    @staticmethod
    def _step_number(state: Mapping, session: Any, conv: BedrockConversation) -> int:
        for key in ("next_step", "step"):
            v = state.get(key)
            if isinstance(v, int) and not isinstance(v, bool):
                return v
        steps = getattr(session, "steps", None)
        if isinstance(steps, list):
            return len(steps) + 1
        budget = state.get("budget")
        candidates = [state.get("n_steps"), budget.get("steps_used") if isinstance(budget, Mapping) else None]
        for v in candidates:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v) + 1
        return conv.decisions + len(conv.fallback_steps) + 1


# --------------------------------------------------------------------------------------------
# model selection
# --------------------------------------------------------------------------------------------

_FAMILY_ORDER = ("sonnet", "haiku", "nova", "opus")
_NOVA_TIERS = {"premier": 4, "pro": 3, "lite": 2}
_EXCLUDED_MARKERS = ("embed", "micro", "sonic", "canvas", "reel", "instant", "claude-v2", "rerank", "image")


def _geo_prefixes(region: str) -> tuple[str, ...]:
    region = (region or "").lower()
    if region.startswith("us-gov"):
        return ("us-gov",)
    if region.startswith("us-") or region.startswith("ca-"):
        return ("us",)
    if region.startswith("eu-"):
        return ("eu",)
    if region == "ap-northeast-1":
        return ("jp", "apac")
    if region in ("ap-southeast-2", "ap-southeast-4"):
        return ("au", "apac")
    if region.startswith("ap-"):
        return ("apac",)
    return ()


def _classify_model(model_id: str) -> dict | None:
    """Family / version / date of a Claude or Nova model id (None for anything else)."""
    mid = model_id.lower()
    base = mid.split("/")[-1]
    if any(m in base for m in _EXCLUDED_MARKERS):
        return None
    m = re.search(r"anthropic\.claude-([a-z0-9\-]+?)(?:-v\d+(?::\d+)?)?$", base)
    if m:
        body = m.group(1)
        family = next((f for f in ("sonnet", "haiku", "opus") if f in body), None)
        if family is None:
            return None
        date_m = re.search(r"(20\d{6})", body)
        date = int(date_m.group(1)) if date_m else 0
        numeric = body.replace(family, " ")
        if date_m:
            numeric = numeric.replace(date_m.group(1), " ")
        version = tuple(int(x) for x in re.findall(r"\d+", numeric))[:2] or (0,)
        version = version + (0,) * (2 - len(version))
        return {"provider": "anthropic", "family": family, "version": version, "date": date, "tier": 0}
    m = re.search(r"amazon\.nova-(?:(\d+)-)?([a-z]+)", base)
    if m:
        tier = _NOVA_TIERS.get(m.group(2))
        if tier is None:
            return None
        gen = int(m.group(1)) if m.group(1) else 1
        return {"provider": "amazon", "family": "nova", "version": (gen, 0), "date": 0, "tier": tier}
    return None


def _paginate(call: Callable[..., Mapping], key: str, **kwargs: Any) -> list[dict]:
    items: list[dict] = []
    token = None
    for _ in range(50):
        params = dict(kwargs)
        if token:
            params["nextToken"] = token
        resp = call(**params)
        items.extend(resp.get(key) or [])
        token = resp.get("nextToken")
        if not token:
            break
    return items


def list_bedrock_model_candidates(
    region: str | None = None,
    *,
    bedrock_client: Any | None = None,
    families: Iterable[str] = ("sonnet", "haiku", "nova"),
) -> list[dict]:
    """Rank invocable Claude/Nova models (inference profiles + on-demand foundation models), best first."""
    region = region or get_settings().aws_region
    if bedrock_client is None:
        import boto3

        bedrock_client = boto3.client("bedrock", region_name=region)
    wanted = [f for f in _FAMILY_ORDER if f in set(families)] or list(_FAMILY_ORDER)
    geo = _geo_prefixes(region)
    candidates: dict[str, dict] = {}

    def consider(model_id: str, kind: str, access_rank: int, active: bool) -> None:
        info = _classify_model(model_id)
        if info is None or info["family"] not in wanted:
            return
        cand = {
            "model_id": model_id,
            "kind": kind,
            "provider": info["provider"],
            "family": info["family"],
            "version": list(info["version"]),
            "date": info["date"],
            "tier": info["tier"],
            "access_rank": access_rank,
            "active": active,
        }
        cand["_key"] = (
            1 if active else 0,
            len(wanted) - wanted.index(info["family"]),
            info["version"],
            info["date"],
            info["tier"],
            access_rank,
        )
        prev = candidates.get(model_id)
        if prev is None or cand["_key"] > prev["_key"]:
            candidates[model_id] = cand

    errors: list[str] = []
    try:
        profiles = _paginate(bedrock_client.list_inference_profiles, "inferenceProfileSummaries", typeEquals="SYSTEM_DEFINED")
    except Exception as exc:
        errors.append(f"list_inference_profiles: {type(exc).__name__}: {exc}")
        profiles = []
    for p in profiles:
        pid = str(p.get("inferenceProfileId") or "")
        if not pid:
            continue
        prefix = pid.split(".", 1)[0].lower()
        access = 3 if prefix in geo else (1 if prefix == "global" else 2)
        consider(pid, "inference_profile", access, str(p.get("status", "ACTIVE")).upper() == "ACTIVE")
    try:
        resp = bedrock_client.list_foundation_models(byOutputModality="TEXT")
        models = resp.get("modelSummaries") or []
    except Exception as exc:
        errors.append(f"list_foundation_models: {type(exc).__name__}: {exc}")
        models = []
    for m in models:
        mid = str(m.get("modelId") or "")
        if not mid or "ON_DEMAND" not in (m.get("inferenceTypesSupported") or []):
            continue
        if "IMAGE" not in (m.get("inputModalities") or ["IMAGE"]):
            continue
        lifecycle = str((m.get("modelLifecycle") or {}).get("status", "ACTIVE")).upper()
        consider(mid, "foundation_model", 0, lifecycle == "ACTIVE")
    if not candidates and len(errors) == 2:
        raise RuntimeError("; ".join(errors))
    ranked = sorted(candidates.values(), key=lambda c: c["_key"], reverse=True)
    for c in ranked:
        c.pop("_key", None)
    return ranked


def _probe_model(runtime_client: Any, model_id: str) -> bool:
    try:
        runtime_client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "Reply with OK."}]}],
            inferenceConfig={"maxTokens": 8, "temperature": 0.0},
        )
        return True
    except Exception as exc:
        log.info("bedrock model probe failed for %s: %s", model_id, exc)
        return False


def pick_bedrock_model(
    region: str | None = None,
    *,
    bedrock_client: Any | None = None,
    runtime_client: Any | None = None,
    families: Iterable[str] = ("sonnet", "haiku", "nova"),
    verify: bool = False,
    max_probes: int = 6,
) -> str | None:
    """Pick an available Bedrock model id: newest Claude Sonnet, else Haiku, else Amazon Nova.

    With ``verify=True`` each candidate (best first, at most ``max_probes``) is probed with a tiny
    Converse request and the first one that answers is returned (catches models without access).
    Returns ``None`` when nothing suitable is listed.
    """
    region = region or get_settings().aws_region
    ranked = list_bedrock_model_candidates(region, bedrock_client=bedrock_client, families=families)
    if not ranked:
        return None
    if not verify:
        return ranked[0]["model_id"]
    if runtime_client is None:
        import boto3

        runtime_client = boto3.client("bedrock-runtime", region_name=region)
    for cand in ranked[: max(1, max_probes)]:
        if _probe_model(runtime_client, cand["model_id"]):
            return cand["model_id"]
    return None


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m inspectagent.agent.planner_bedrock --pick-model [--region R] [--verify]``."""
    ap = argparse.ArgumentParser(description="InspectAgent Bedrock planner utilities")
    ap.add_argument("--pick-model", action="store_true", help="list candidate models and print the pick")
    ap.add_argument("--region", default=None)
    ap.add_argument("--verify", action="store_true", help="probe candidates with a tiny Converse call")
    ap.add_argument("--families", default="sonnet,haiku,nova")
    args = ap.parse_args(argv)
    if not args.pick_model:
        ap.print_help()
        return 0
    region = args.region or get_settings().aws_region
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    try:
        ranked = list_bedrock_model_candidates(region, families=families)
        picked = pick_bedrock_model(region, families=families, verify=args.verify) if ranked else None
    except Exception as exc:
        print(json.dumps({"region": region, "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    print(json.dumps({"region": region, "picked": picked, "candidates": ranked[:10]}, indent=2))
    return 0 if picked else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
