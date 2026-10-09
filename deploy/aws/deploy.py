#!/usr/bin/env python3
"""InspectAgent AWS deployer: CloudFormation + CodeBuild (ARM / x86) + ECR + Lambda Function URL.

Idempotent, prints every step. Typical use (from the repository root)::

    python deploy/aws/deploy.py --region us-east-1                 # arm64 (Graviton) function
    python deploy/aws/deploy.py --region us-east-1 --arch both     # + x86_64 comparison function (<prefix>-x86)
    python deploy/aws/deploy.py --skip-build                       # redeploy the latest image already in ECR
    python deploy/aws/deploy.py --destroy --yes                    # tear everything down

Flow: (1) create/update the stack without a new image (bucket, table, ECR repo, CodeBuild projects, roles);
(2) zip the repository with the rules of ``deploy/.dockerignore`` (Docker semantics) and upload it to
``s3://<bucket>/source/src-<sha256>.zip``; (3) start CodeBuild per architecture and stream its status
(skipped when an image for exactly this source already exists in ECR); (4) update the stack with the
digest-pinned ImageUri(s); (5) smoke-test ``GET /healthz`` on each Function URL and print the URL(s).

Only ``boto3`` is required (imported lazily, so ``--help`` and the zip helpers work without it).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

DEPLOY_AWS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEPLOY_AWS_DIR.parent.parent
TEMPLATE_PATH = DEPLOY_AWS_DIR / "template.yaml"
DOCKERIGNORE_PATH = REPO_ROOT / "deploy" / ".dockerignore"
BUILD_DIR = DEPLOY_AWS_DIR / ".build"

DEFAULT_STACK_NAME = "inspectagent"
DEFAULT_RESERVED_CONCURRENCY = 5
TEMPLATE_BODY_LIMIT = 51_200  # bytes accepted by CloudFormation's TemplateBody
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)  # fixed timestamps -> byte-identical zips for identical sources
USER_AGENT = "inspectagent-deploy/1.0"

STACK_FAILED_FOR_UPDATE = {"UPDATE_ROLLBACK_FAILED", "DELETE_FAILED"}
STACK_NEEDS_RECREATE = {"ROLLBACK_COMPLETE", "ROLLBACK_FAILED", "CREATE_FAILED"}


class DeployError(RuntimeError):
    """A deployment step failed; the message says what to do next."""


@dataclass(frozen=True)
class ArchSpec:
    """How one Lambda architecture maps onto template parameters/outputs and image tags."""

    name: str  # Lambda architecture name
    image_param: str  # template parameter carrying the image URI
    project_output: str  # stack output with the CodeBuild project name
    url_output: str  # stack output with the Function URL
    function_output: str  # stack output with the function name
    tag_suffix: str  # appended to image tags by the buildspec (TAG_SUFFIX)
    function_suffix: str  # appended to the stack prefix to form the function name


ARCH_SPECS: dict[str, ArchSpec] = {
    "arm64": ArchSpec("arm64", "ImageUri", "ProjectName", "FunctionUrl", "FunctionName", "", ""),
    "x86_64": ArchSpec("x86_64", "ImageUriX86", "ProjectNameX86", "FunctionUrlX86", "FunctionNameX86", "-x86_64", "-x86"),
}
ARCH_CHOICES = {"arm64": ("arm64",), "x86_64": ("x86_64",), "both": ("arm64", "x86_64")}


# ------------------------------------------------------------------------------------------------
# .dockerignore semantics (mirrors moby/patternmatcher + the classic context walker)
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _IgnorePattern:
    text: str  # cleaned pattern without the leading "!"
    exclusion: bool  # True for "!pattern" (re-include)
    regex: re.Pattern[str]


def _clean_pattern(p: str) -> str:
    """Normalise like Go's filepath.Clean + ToSlash and strip a leading '/'."""
    p = p.replace("\\", "/")
    parts: list[str] = []
    for seg in p.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts and parts[-1] != "..":
                parts.pop()
                continue
        parts.append(seg)
    cleaned = "/".join(parts)
    return cleaned or "."


def _pattern_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a Docker ignore pattern into an anchored regex (``**`` crosses directories)."""
    out = ["^"]
    i, n = 0, len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                i += 2
                if i < n and pattern[i] == "/":
                    i += 1
                out.append(".*" if i >= n else "(.*/)?")
                continue
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        elif ch == "\\" and i + 1 < n:
            i += 1
            out.append(re.escape(pattern[i]))
        elif ch == "[":
            j = i + 1
            if j < n and pattern[j] == "^":
                j += 1
            if j < n and pattern[j] == "]":
                j += 1
            while j < n and pattern[j] != "]":
                j += 2 if pattern[j] == "\\" else 1
            if j >= n:  # unterminated class: literal "["
                out.append(re.escape(ch))
            else:  # character class, copied verbatim (Go and Python agree on simple classes)
                out.append(pattern[i : j + 1])
                i = j
        else:
            out.append(re.escape(ch))
        i += 1
    out.append("$")
    return re.compile("".join(out))


class DockerIgnore:
    """Docker ``.dockerignore`` matcher: last matching rule wins, rules also match parent dirs."""

    def __init__(self, lines: Iterable[str]):
        self.patterns: list[_IgnorePattern] = []
        for raw in lines:
            line = raw.lstrip("﻿")
            if line.startswith("#"):
                continue
            line = line.strip()
            if not line:
                continue
            exclusion = line.startswith("!")
            if exclusion:
                line = line[1:].strip()
                if not line:
                    raise ValueError("illegal exclusion pattern: '!'")
            cleaned = _clean_pattern(line)
            if cleaned == ".":
                continue
            self.patterns.append(_IgnorePattern(cleaned, exclusion, _pattern_to_regex(cleaned)))

    @classmethod
    def from_file(cls, path: Path) -> "DockerIgnore":
        return cls(path.read_text(encoding="utf-8").splitlines())

    @property
    def has_exclusions(self) -> bool:
        return any(p.exclusion for p in self.patterns)

    def is_ignored(self, rel_path: str) -> bool:
        """True when ``rel_path`` (posix, relative to the context root) is excluded from the context."""
        rel = _clean_pattern(rel_path)
        parts = rel.split("/")
        parents = ["/".join(parts[: i + 1]) for i in range(len(parts) - 1)]
        matched = False
        for pat in self.patterns:
            if pat.exclusion != matched:
                continue  # an inclusion can only flip an excluded path, and vice versa
            hit = bool(pat.regex.match(rel)) or any(pat.regex.match(p) for p in parents)
            if hit:
                matched = not pat.exclusion
        return matched

    def can_skip_dir(self, rel_dir: str) -> bool:
        """An excluded directory is pruned unless some '!pattern' starts with its path (moby rule)."""
        if not self.has_exclusions:
            return True
        prefix = _clean_pattern(rel_dir) + "/"
        return not any(p.exclusion and (p.text + "/").startswith(prefix) for p in self.patterns)


def iter_context_files(root: Path, ignore: DockerIgnore, skip: Iterable[Path] = ()) -> Iterator[str]:
    """Yield posix relative paths of the files Docker would send as build context, sorted."""
    root = root.resolve()
    skip_set = {p.resolve() for p in skip}
    for dirpath, dirnames, filenames in os.walk(root):
        cur = Path(dirpath)
        rel_dir = cur.relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir
        keep_dirs = []
        for d in sorted(dirnames):
            rel = f"{rel_dir}/{d}" if rel_dir else d
            if (cur / d).is_symlink():
                continue
            if ignore.is_ignored(rel) and ignore.can_skip_dir(rel):
                continue
            keep_dirs.append(d)
        dirnames[:] = keep_dirs
        for f in sorted(filenames):
            full = cur / f
            if full.resolve() in skip_set:
                continue
            rel = f"{rel_dir}/{f}" if rel_dir else f
            if not ignore.is_ignored(rel):
                yield rel


@dataclass
class SourceZip:
    path: Path
    sha256: str
    files: list[str]
    size: int

    @property
    def tag(self) -> str:
        """Content-addressed image tag base, e.g. ``src-3f2a9c0d1e4b``."""
        return f"src-{self.sha256[:12]}"


def build_source_zip(root: Path, out_path: Path, ignore_file: Path | None = None) -> SourceZip:
    """Zip the build context of ``root`` (rules from ``ignore_file``) deterministically.

    Entries are sorted, timestamps fixed and permissions normalised, so identical sources give a
    byte-identical archive and therefore the same sha256 / image tag (which makes rebuilds skippable).
    """
    root = Path(root).resolve()
    ignore_path = Path(ignore_file) if ignore_file else root / "deploy" / ".dockerignore"
    if not ignore_path.is_file():
        raise DeployError(f"ignore file not found: {ignore_path}")
    ignore = DockerIgnore.from_file(ignore_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".partial")
    files = sorted(iter_context_files(root, ignore, skip=(out_path, tmp)))
    if not files:
        raise DeployError(f"nothing to zip under {root} (check {ignore_path})")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel in files:
            src = root / rel
            info = zipfile.ZipInfo(rel, date_time=ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3  # unix: identical archives on Windows and Linux
            info.external_attr = (0o100644 & 0xFFFF) << 16
            info.file_size = src.stat().st_size
            with open(src, "rb") as fin, zf.open(info, "w") as fout:
                while chunk := fin.read(1 << 20):
                    fout.write(chunk)
    os.replace(tmp, out_path)
    digest = hashlib.sha256()
    with open(out_path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return SourceZip(out_path, digest.hexdigest(), files, out_path.stat().st_size)


def git_short_sha(root: Path = REPO_ROOT) -> str:
    """Short commit id (``-dirty`` when the tree has changes) or ``unknown``."""
    try:
        res = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, timeout=10
        )
        if res.returncode != 0 or not res.stdout.strip():
            return "unknown"
        sha = res.stdout.strip()
        st = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, timeout=20)
        return f"{sha}-dirty" if st.returncode == 0 and st.stdout.strip() else sha
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# ------------------------------------------------------------------------------------------------
# Pure helpers (unit-tested offline)
# ------------------------------------------------------------------------------------------------
def prefix_from_stack_name(stack_name: str) -> str:
    """Physical-name prefix accepted by the template (``^[a-z][a-z0-9-]{1,20}$``)."""
    p = re.sub(r"[^a-z0-9-]+", "-", stack_name.lower()).strip("-")
    p = re.sub(r"^[^a-z]+", "", p)
    p = p[:21].rstrip("-")
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,20}", p):
        raise DeployError(f"cannot derive a resource prefix from stack name {stack_name!r}; use letters/digits/hyphens")
    return p


def choose_reserved_concurrency(account_limit: int, available_unreserved: int, desired: int, n_functions: int) -> int:
    """Reserved concurrency per function that the account can afford (0 = do not reserve).

    Lambda keeps at least ``min(100, account_limit)`` executions unreserved; new accounts with a quota of
    10 therefore cannot reserve anything. ``available_unreserved`` must already include the
    reservations currently held by this stack's own functions.
    """
    if desired <= 0 or n_functions <= 0:
        return 0
    min_unreserved = min(100, account_limit)
    if available_unreserved - desired * n_functions >= min_unreserved:
        return desired
    return 0


_GEO_PREFIXES = ("us-gov.", "us.", "eu.", "apac.", "jp.", "au.", "ca.", "global.")
_EXCLUDED_MODEL_HINTS = ("opus", "instant", "claude-v2", "claude-2", "micro", "canvas", "reel", "sonic", "embed")


def _region_geo(region: str) -> str:
    if region.startswith("us-gov-"):
        return "us-gov."
    head = region.split("-", 1)[0]
    return {"us": "us.", "eu": "eu.", "ap": "apac.", "ca": "ca."}.get(head, "")


def _strip_geo(model_id: str) -> tuple[str, str]:
    for g in _GEO_PREFIXES:
        if model_id.startswith(g):
            return g, model_id[len(g):]
    return "", model_id


def _model_rank(model_id: str, region: str) -> tuple | None:
    """Sort key (higher is better) for planner models, or None if unsuitable (no tool use / too costly)."""
    geo, base = _strip_geo(model_id.lower())
    if any(h in base for h in _EXCLUDED_MODEL_HINTS):
        return None
    region_geo = _region_geo(region)
    if geo and geo == region_geo:
        geo_rank = 3
    elif geo == "global.":
        geo_rank = 2
    elif geo:
        geo_rank = 1 if region_geo in ("", geo) else 0
    else:
        geo_rank = 1
    if "anthropic.claude" in base:
        fam = "sonnet" if "sonnet" in base else "haiku" if "haiku" in base else ""
        if not fam:
            return None
        m = re.search(rf"claude-(?:(\d+)-(?:(\d)-)?{fam}|{fam}-(\d+)(?:-(\d{{1,2}})(?!\d))?)", base)
        if not m:
            return None
        major = int(m.group(1) or m.group(3) or 0)
        minor = int(m.group(2) or m.group(4) or 0)
        date_m = re.search(r"(?<!\d)(\d{8})(?!\d)", base)
        date = int(date_m.group(1)) if date_m else 0
        return (2, (major, minor), {"sonnet": 2, "haiku": 1}[fam], date, geo_rank)
    if "amazon.nova" in base:
        fam = "pro" if "-pro" in base else "lite" if "-lite" in base else "premier" if "premier" in base else ""
        if not fam:
            return None
        ver_m = re.search(r"nova-(\d+)-", base)
        major = int(ver_m.group(1)) if ver_m else 1
        return (1, (major, 0), {"pro": 3, "lite": 2, "premier": 1}[fam], 0, geo_rank)
    return None


def rank_bedrock_candidates(profiles: Sequence[dict], models: Sequence[dict], region: str) -> list[str]:
    """Rank Bedrock model ids for the planner: Anthropic Claude (newest Sonnet/Haiku) > Amazon Nova.

    ``profiles`` are ``bedrock.list_inference_profiles`` summaries, ``models`` are
    ``bedrock.list_foundation_models`` summaries (only ACTIVE, ON_DEMAND, TEXT-output models qualify).
    Inference profiles matching the region's geography are preferred over global ones and over
    plain in-region model ids.
    """
    scored: dict[str, tuple] = {}
    for p in profiles:
        pid = p.get("inferenceProfileId") or ""
        if not pid or p.get("status", "ACTIVE") != "ACTIVE":
            continue
        key = _model_rank(pid, region)
        if key is not None:
            scored[pid] = key
    for m in models:
        mid = m.get("modelId") or ""
        if not mid or mid.count(":") > 1:  # skip context-length variants like ...-v1:0:200k
            continue
        if (m.get("modelLifecycle") or {}).get("status", "ACTIVE") != "ACTIVE":
            continue
        if "ON_DEMAND" not in (m.get("inferenceTypesSupported") or []):
            continue
        if "TEXT" not in (m.get("outputModalities") or ["TEXT"]):
            continue
        key = _model_rank(mid, region)
        if key is not None and mid not in scored:
            scored[mid] = key
    return [mid for mid, _ in sorted(scored.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]


def _call_external_picker(
    fn: Callable[..., Any], client: Callable[[str], Any], session: Any, region: str, verify: bool
) -> str | None:
    """Call ``planner_bedrock.pick_bedrock_model`` mapping its parameters by name.

    Known names get the region, the boto3 session, Bedrock / Bedrock-runtime clients (from ``client``)
    or the ``verify`` flag; other parameters keep their defaults.
    """
    kwargs: dict[str, Any] = {}
    for name, param in inspect.signature(fn).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if name in ("region", "region_name", "aws_region"):
            kwargs[name] = region
        elif name in ("session", "boto3_session", "boto_session"):
            kwargs[name] = session
        elif name in ("client", "bedrock", "bedrock_client"):
            kwargs[name] = client("bedrock")
        elif name in ("runtime", "runtime_client", "bedrock_runtime"):
            kwargs[name] = client("bedrock-runtime")
        elif name == "verify":
            kwargs[name] = bool(verify)
        elif param.default is inspect.Parameter.empty:
            raise TypeError(f"unsupported required parameter {name!r}")
    result = fn(**kwargs)
    if isinstance(result, str):
        return result or None
    if isinstance(result, dict):
        return result.get("model_id") or result.get("modelId") or None
    return None


def _default_http_get(url: str, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - https Function URL
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _client_error_code(exc: BaseException) -> str:
    resp = getattr(exc, "response", None) or {}
    return str((resp.get("Error") or {}).get("Code", ""))


def _client_error_message(exc: BaseException) -> str:
    resp = getattr(exc, "response", None) or {}
    return str((resp.get("Error") or {}).get("Message", exc))


# ------------------------------------------------------------------------------------------------
# Deployer
# ------------------------------------------------------------------------------------------------
@dataclass
class DeployOptions:
    archs: tuple[str, ...] = ("arm64",)
    bedrock_model: str = "auto"  # auto | none | <model or inference-profile id>
    verify_model: bool = True
    skip_build: bool = False
    reserved_concurrency: int | None = None  # None = 5 if the account quota allows it, else 0
    allow_marketplace: bool = False
    build_compute: str = "BUILD_GENERAL1_SMALL"
    smoke_test: bool = True
    bench_n: int = 0
    root: Path = REPO_ROOT
    use_planner_picker: bool = True


@dataclass
class DeployResult:
    stack_name: str
    region: str
    outputs: dict[str, str]
    images: dict[str, str] = field(default_factory=dict)
    urls: dict[str, str] = field(default_factory=dict)
    health: dict[str, dict] = field(default_factory=dict)
    bedrock_model_id: str = ""


class Deployer:
    """Drives one stack. All AWS access goes through ``session.client(...)`` (boto3 Session or a fake)."""

    def __init__(
        self,
        session: Any,
        stack_name: str = DEFAULT_STACK_NAME,
        region: str = "us-east-1",
        *,
        log: Callable[[str], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        http_get: Callable[[str, float], tuple[int, bytes]] = _default_http_get,
        poll_seconds: float = 10.0,
        stack_timeout_s: float = 3600.0,
        build_timeout_s: float = 3600.0,
    ):
        self.session = session
        self.stack_name = stack_name
        self.prefix = prefix_from_stack_name(stack_name)
        self.region = region
        self._log = log or self._print
        self.sleep = sleep
        self.http_get = http_get
        self.poll_seconds = poll_seconds
        self.stack_timeout_s = stack_timeout_s
        self.build_timeout_s = build_timeout_s
        self.build_dir = BUILD_DIR  # source zip + last_deploy.json (git-ignored)
        self.results_dir = REPO_ROOT / "results"  # remote bench output
        self._clients: dict[str, Any] = {}

    # -- plumbing -------------------------------------------------------------------------------
    @staticmethod
    def _print(msg: str) -> None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    def log(self, msg: str) -> None:
        self._log(msg)

    def client(self, name: str) -> Any:
        if name not in self._clients:
            kwargs: dict[str, Any] = {"region_name": self.region}
            try:
                from botocore.config import Config

                kwargs["config"] = Config(retries={"max_attempts": 10, "mode": "adaptive"}, user_agent_extra=USER_AGENT)
            except ImportError:  # pragma: no cover - botocore always ships with boto3
                pass
            self._clients[name] = self.session.client(name, **kwargs)
        return self._clients[name]

    @staticmethod
    def _client_error_type() -> type[BaseException]:
        try:
            from botocore.exceptions import ClientError

            return ClientError
        except ImportError:  # pragma: no cover
            return RuntimeError

    # -- identity / inputs ------------------------------------------------------------------------
    def caller_identity(self) -> dict:
        ident = self.client("sts").get_caller_identity()
        self.log(f"AWS account {ident.get('Account')} as {ident.get('Arn')} in {self.region}")
        return ident

    def resolve_bedrock_model(self, choice: str, verify: bool = True, use_planner_picker: bool = True) -> str:
        """``none`` -> '' (Bedrock disabled), ``<id>`` -> as is, ``auto`` -> best available model."""
        c = (choice or "").strip()
        if c.lower() in ("none", "off", ""):
            self.log("Bedrock planner disabled (--bedrock-model none)")
            return ""
        if c.lower() != "auto":
            self.log(f"Bedrock model: {c} (as given)")
            return c
        picked: str | None = None
        if use_planner_picker:
            if str(REPO_ROOT) not in sys.path:
                sys.path.append(str(REPO_ROOT))
            try:
                from inspectagent.agent.planner_bedrock import pick_bedrock_model  # type: ignore[attr-defined]

                picked = _call_external_picker(pick_bedrock_model, self.client, self.session, self.region, verify)
                if picked:
                    how = "access verified" if verify else "not verified"
                    self.log(f"Bedrock model (planner_bedrock.pick_bedrock_model, {how}): {picked}")
                    return picked
            except ImportError:
                pass
            except Exception as exc:  # noqa: BLE001 - external helper; fall back to the built-in picker
                self.log(f"planner_bedrock.pick_bedrock_model unavailable ({type(exc).__name__}: {exc}); using built-in picker")
        picked = self.pick_bedrock_model_local(verify=verify)
        if not picked:
            self.log("No suitable Bedrock model found in this region; the Bedrock planner stays disabled (rule planner only)")
            return ""
        return picked

    def pick_bedrock_model_local(self, verify: bool = True) -> str | None:
        """List inference profiles + foundation models, rank them and (optionally) verify access."""
        ClientError = self._client_error_type()
        bedrock = self.client("bedrock")
        profiles: list[dict] = []
        models: list[dict] = []
        try:
            token: str | None = None
            while True:
                kw: dict[str, Any] = {"typeEquals": "SYSTEM_DEFINED", "maxResults": 1000}
                if token:
                    kw["nextToken"] = token
                resp = bedrock.list_inference_profiles(**kw)
                profiles.extend(resp.get("inferenceProfileSummaries", []))
                token = resp.get("nextToken")
                if not token:
                    break
        except ClientError as exc:
            self.log(f"  list_inference_profiles failed: {_client_error_code(exc)} {_client_error_message(exc)}")
        try:
            models = bedrock.list_foundation_models(byOutputModality="TEXT").get("modelSummaries", [])
        except ClientError as exc:
            self.log(f"  list_foundation_models failed: {_client_error_code(exc)} {_client_error_message(exc)}")
        ranked = rank_bedrock_candidates(profiles, models, self.region)
        if not ranked:
            return None
        self.log(f"  Bedrock candidates: {', '.join(ranked[:5])}")
        if not verify:
            self.log(f"Bedrock model (auto, not verified): {ranked[0]}")
            return ranked[0]
        runtime = self.client("bedrock-runtime")
        last_err = ""
        for mid in ranked[:5]:
            try:
                runtime.converse(
                    modelId=mid,
                    messages=[{"role": "user", "content": [{"text": "Reply with OK."}]}],
                    inferenceConfig={"maxTokens": 5, "temperature": 0.0},
                )
                self.log(f"Bedrock model (auto, access verified with a 5-token Converse call): {mid}")
                return mid
            except ClientError as exc:
                last_err = f"{_client_error_code(exc)}: {_client_error_message(exc)}"
                self.log(f"  {mid}: not usable ({last_err})")
        self.log(
            f"Bedrock model (auto, access NOT verified - {last_err}): {ranked[0]}. Enable model access in the Bedrock "
            "console; until then the Bedrock planner falls back to the rule planner."
        )
        return ranked[0]

    def function_name(self, arch: str) -> str:
        return self.prefix + ARCH_SPECS[arch].function_suffix

    def resolve_reserved_concurrency(self, requested: int | None, n_functions: int) -> int:
        """Explicit value -> as is; None -> 5 when the account concurrency quota allows it, else 0."""
        if requested is not None:
            return max(0, int(requested))
        ClientError = self._client_error_type()
        lam = self.client("lambda")
        try:
            limits = lam.get_account_settings()["AccountLimit"]
        except ClientError as exc:
            self.log(f"  get_account_settings failed ({_client_error_code(exc)}); not reserving concurrency")
            return 0
        current = 0
        for arch in ARCH_SPECS:
            try:
                current += int(lam.get_function_concurrency(FunctionName=self.function_name(arch)).get("ReservedConcurrentExecutions") or 0)
            except ClientError as exc:
                if _client_error_code(exc) != "ResourceNotFoundException":
                    raise
        limit = int(limits.get("ConcurrentExecutions", 0))
        unreserved = int(limits.get("UnreservedConcurrentExecutions", 0))
        chosen = choose_reserved_concurrency(limit, unreserved + current, DEFAULT_RESERVED_CONCURRENCY, n_functions)
        if chosen == 0:
            self.log(
                f"  Account concurrency quota is {limit} (unreserved {unreserved}); reserving {DEFAULT_RESERVED_CONCURRENCY} "
                "per function is not possible -> ReservedConcurrency=0 (request a quota increase to enable the cap)"
            )
        else:
            self.log(f"  Reserved concurrency: {chosen} per function (account quota {limit})")
        return chosen

    # -- CloudFormation -------------------------------------------------------------------------
    def describe_stack(self) -> dict | None:
        ClientError = self._client_error_type()
        try:
            stacks = self.client("cloudformation").describe_stacks(StackName=self.stack_name)["Stacks"]
        except ClientError as exc:
            if _client_error_code(exc) == "ValidationError" and "does not exist" in _client_error_message(exc):
                return None
            raise
        return stacks[0] if stacks else None

    @staticmethod
    def stack_outputs(stack: dict | None) -> dict[str, str]:
        return {o["OutputKey"]: o["OutputValue"] for o in (stack or {}).get("Outputs", []) or []}

    @staticmethod
    def stack_parameters(stack: dict | None) -> dict[str, str]:
        return {p["ParameterKey"]: p.get("ParameterValue", "") for p in (stack or {}).get("Parameters", []) or []}

    def _event_ids(self) -> set[str]:
        ClientError = self._client_error_type()
        try:
            events = self.client("cloudformation").describe_stack_events(StackName=self.stack_name)["StackEvents"]
        except ClientError:
            return set()
        return {e["EventId"] for e in events}

    def _print_new_events(self, seen: set[str]) -> list[dict]:
        ClientError = self._client_error_type()
        try:
            events = self.client("cloudformation").describe_stack_events(StackName=self.stack_name)["StackEvents"]
        except ClientError:
            return []
        fresh = [e for e in events if e["EventId"] not in seen]
        for e in reversed(fresh):  # API returns newest first
            seen.add(e["EventId"])
            reason = e.get("ResourceStatusReason") or ""
            ts = e.get("Timestamp")
            ts_s = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else str(ts or "")[11:19]
            self.log(
                f"    {ts_s}  {e.get('ResourceStatus', ''):<28} {e.get('ResourceType', ''):<28} "
                f"{e.get('LogicalResourceId', '')}" + (f"  ({reason})" if reason else "")
            )
        return fresh

    def wait_stack(self, seen: set[str], success: set[str], allow_missing: bool = False) -> str:
        """Poll until the stack leaves *_IN_PROGRESS, streaming events; raise unless it ends in ``success``."""
        deadline = time.monotonic() + self.stack_timeout_s
        failures: list[str] = []
        while True:
            stack = self.describe_stack()
            if stack is None:
                if allow_missing:
                    self._print_new_events(seen)
                    return "DELETE_COMPLETE"
                raise DeployError(f"stack {self.stack_name} disappeared while waiting")
            for e in self._print_new_events(seen):
                if str(e.get("ResourceStatus", "")).endswith("_FAILED") and e.get("ResourceStatusReason"):
                    failures.append(f"{e.get('LogicalResourceId')}: {e.get('ResourceStatusReason')}")
            status = stack["StackStatus"]
            if not status.endswith("_IN_PROGRESS"):
                if status in success:
                    return status
                detail = "; ".join(failures[-5:]) or stack.get("StackStatusReason", "")
                raise DeployError(f"stack {self.stack_name} ended in {status}: {detail}")
            if time.monotonic() > deadline:
                raise DeployError(f"timed out waiting for stack {self.stack_name} (status {status})")
            self.sleep(self.poll_seconds)

    def _template_body(self) -> str:
        body = TEMPLATE_PATH.read_text(encoding="utf-8")
        if len(body.encode("utf-8")) > TEMPLATE_BODY_LIMIT:
            raise DeployError(f"{TEMPLATE_PATH} exceeds the {TEMPLATE_BODY_LIMIT}-byte TemplateBody limit")
        return body

    def prepare_stack(self) -> dict | None:
        """Wait for in-flight operations and delete stacks stuck after a failed creation."""
        stack = self.describe_stack()
        if stack is None:
            return None
        status = stack["StackStatus"]
        if status.endswith("_IN_PROGRESS"):
            self.log(f"Stack {self.stack_name} is {status}; waiting for it to settle")
            self.wait_stack(self._event_ids(), set(_ALL_SETTLED), allow_missing=True)
            stack = self.describe_stack()
            if stack is None:
                return None
            status = stack["StackStatus"]
        if status in STACK_NEEDS_RECREATE:
            self.log(f"Stack {self.stack_name} is {status} (failed creation); deleting it before re-creating")
            seen = self._event_ids()
            self.client("cloudformation").delete_stack(StackName=self.stack_name)
            self.wait_stack(seen, {"DELETE_COMPLETE"}, allow_missing=True)
            return None
        if status in STACK_FAILED_FOR_UPDATE:
            raise DeployError(
                f"stack {self.stack_name} is {status}; fix it in the CloudFormation console "
                "(continue rollback / retry delete) or run with --destroy, then deploy again"
            )
        return stack

    def deploy_stack(self, params: dict[str, str | None], label: str) -> dict[str, str]:
        """Create or update the stack. ``None`` values mean UsePreviousValue. Returns the outputs."""
        cfn = self.client("cloudformation")
        stack = self.prepare_stack()
        body = self._template_body()
        if stack is None:
            plist = [{"ParameterKey": k, "ParameterValue": v or ""} for k, v in params.items()]
            self.log(f"Creating stack {self.stack_name} ({label})")
            cfn.create_stack(
                StackName=self.stack_name,
                TemplateBody=body,
                Parameters=plist,
                Capabilities=["CAPABILITY_IAM"],
                Tags=[{"Key": "project", "Value": "inspectagent"}],
            )
            self.wait_stack(set(), {"CREATE_COMPLETE"})
        else:
            existing = self.stack_parameters(stack)
            plist = []
            for k, v in params.items():
                if v is None and k in existing:
                    plist.append({"ParameterKey": k, "UsePreviousValue": True})
                else:
                    plist.append({"ParameterKey": k, "ParameterValue": v or ""})
            seen = self._event_ids()
            self.log(f"Updating stack {self.stack_name} ({label})")
            try:
                cfn.update_stack(
                    StackName=self.stack_name,
                    TemplateBody=body,
                    Parameters=plist,
                    Capabilities=["CAPABILITY_IAM"],
                    Tags=[{"Key": "project", "Value": "inspectagent"}],
                )
            except self._client_error_type() as exc:
                if "No updates are to be performed" in _client_error_message(exc):
                    self.log("  stack already up to date")
                    return self.stack_outputs(self.describe_stack())
                raise
            self.wait_stack(seen, {"UPDATE_COMPLETE"})
        outputs = self.stack_outputs(self.describe_stack())
        self.log(f"  stack ready: {', '.join(sorted(outputs))}")
        return outputs

    # -- source upload + CodeBuild ------------------------------------------------------------------
    def upload_source(self, src: SourceZip, bucket: str) -> str:
        """Upload the zip to ``source/<tag>.zip`` unless an object of the same size is already there."""
        ClientError = self._client_error_type()
        s3 = self.client("s3")
        key = f"source/{src.tag}.zip"
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
            if int(head.get("ContentLength", -1)) == src.size:
                self.log(f"  s3://{bucket}/{key} already uploaded")
                return key
        except ClientError as exc:
            if _client_error_code(exc) not in ("404", "NoSuchKey", "NotFound"):
                raise
        self.log(f"  uploading {src.size / 1e6:.1f} MB -> s3://{bucket}/{key}")
        try:
            from boto3.s3.transfer import TransferConfig

            s3.upload_file(str(src.path), bucket, key, Config=TransferConfig(multipart_threshold=64 * 1024 * 1024))
        except ImportError:  # pragma: no cover - boto3 not installed (fake sessions in tests)
            with open(src.path, "rb") as fh:
                s3.put_object(Bucket=bucket, Key=key, Body=fh)
        return key

    def find_image(self, repo_name: str, repo_uri: str, tag: str) -> str | None:
        """Digest-pinned URI of ``repo:tag`` or None when the tag does not exist."""
        ClientError = self._client_error_type()
        try:
            details = self.client("ecr").describe_images(repositoryName=repo_name, imageIds=[{"imageTag": tag}])["imageDetails"]
        except ClientError as exc:
            if _client_error_code(exc) in ("ImageNotFoundException", "RepositoryNotFoundException"):
                return None
            raise
        if not details:
            return None
        return f"{repo_uri}@{details[0]['imageDigest']}"

    def start_build(self, project: str, bucket: str, key: str, image_tag: str, git_sha: str) -> str:
        resp = self.client("codebuild").start_build(
            projectName=project,
            sourceTypeOverride="S3",
            sourceLocationOverride=f"{bucket}/{key}",
            environmentVariablesOverride=[
                {"name": "IMAGE_TAG", "value": image_tag, "type": "PLAINTEXT"},
                {"name": "GIT_SHA", "value": git_sha, "type": "PLAINTEXT"},
            ],
        )
        build = resp["build"]
        self.log(f"  started CodeBuild {build['id']}")
        return build["id"]

    def wait_builds(self, builds: dict[str, str]) -> None:
        """Poll CodeBuild until every build finishes; print phase changes; tail logs on failure."""
        cb = self.client("codebuild")
        deadline = time.monotonic() + self.build_timeout_s
        last_phase: dict[str, str] = {}
        pending = dict(builds)  # arch -> build id
        failed: list[str] = []
        while pending:
            resp = cb.batch_get_builds(ids=list(pending.values()))
            by_id = {b["id"]: b for b in resp.get("builds", [])}
            for arch, bid in list(pending.items()):
                b = by_id.get(bid)
                if b is None:
                    continue
                phase = b.get("currentPhase", "")
                if phase and phase != last_phase.get(arch):
                    last_phase[arch] = phase
                    self.log(f"  [{arch}] phase {phase}")
                status = b.get("buildStatus", "IN_PROGRESS")
                if status == "IN_PROGRESS":
                    continue
                pending.pop(arch)
                secs = ""
                if b.get("startTime") is not None and b.get("endTime") is not None and hasattr(b["endTime"], "__sub__"):
                    secs = f" in {(b['endTime'] - b['startTime']).total_seconds():.0f}s"
                self.log(f"  [{arch}] build {status}{secs}")
                if status != "SUCCEEDED":
                    failed.append(arch)
                    self._report_build_failure(arch, b)
            if pending:
                if time.monotonic() > deadline:
                    raise DeployError(f"timed out waiting for CodeBuild: {', '.join(pending.values())}")
                self.sleep(self.poll_seconds)
        if failed:
            raise DeployError(f"image build failed for {', '.join(failed)} (see the log tail above)")

    def _report_build_failure(self, arch: str, build: dict) -> None:
        for ph in build.get("phases", []) or []:
            if ph.get("phaseStatus") not in (None, "SUCCEEDED"):
                msgs = "; ".join(c.get("message", "") for c in ph.get("contexts", []) or [] if c.get("message"))
                self.log(f"  [{arch}] {ph.get('phaseType')}: {ph.get('phaseStatus')} {msgs}")
        logs = build.get("logs") or {}
        if logs.get("deepLink"):
            self.log(f"  [{arch}] full log: {logs['deepLink']}")
        group, stream = logs.get("groupName"), logs.get("streamName")
        if group and stream:
            try:
                events = self.client("logs").get_log_events(
                    logGroupName=group, logStreamName=stream, limit=40, startFromHead=False
                )["events"]
                for ev in events:
                    self.log(f"  [{arch}] | {ev.get('message', '').rstrip()}")
            except self._client_error_type() as exc:
                self.log(f"  [{arch}] could not read build log: {_client_error_code(exc)}")

    def build_images(self, outputs: dict[str, str], archs: Sequence[str], root: Path) -> dict[str, str]:
        """Zip + upload the source and build the missing images. Returns {arch: digest-pinned URI}."""
        bucket, repo_uri = outputs["BucketName"], outputs["RepoUri"]
        repo_name = outputs.get("RepoName") or repo_uri.split("/", 1)[1]
        ignore_file = Path(root) / "deploy" / ".dockerignore"
        self.log(f"  zipping {root} with the rules of {ignore_file}")
        src = build_source_zip(root, self.build_dir / "source.zip", ignore_file)
        self.log(f"  {len(src.files)} files, {src.size / 1e6:.1f} MB, sha256 {src.sha256[:12]} -> tag {src.tag}")
        images: dict[str, str] = {}
        to_build: dict[str, str] = {}
        for arch in archs:
            tag = src.tag + ARCH_SPECS[arch].tag_suffix
            uri = self.find_image(repo_name, repo_uri, tag)
            if uri:
                self.log(f"  [{arch}] image for this exact source already in ECR ({tag}); skipping build")
                images[arch] = uri
            else:
                to_build[arch] = tag
        if to_build:
            key = self.upload_source(src, bucket)
            git_sha = git_short_sha(root)
            builds = {arch: self.start_build(outputs[ARCH_SPECS[arch].project_output], bucket, key, src.tag, git_sha) for arch in to_build}
            self.wait_builds(builds)
            for arch, tag in to_build.items():
                uri = self.find_image(repo_name, repo_uri, tag)
                if not uri:
                    raise DeployError(f"build for {arch} succeeded but {repo_uri}:{tag} is not in ECR")
                images[arch] = uri
        for arch, uri in images.items():
            self.log(f"  [{arch}] image {uri}")
        return images

    def latest_images(self, outputs: dict[str, str], archs: Sequence[str]) -> dict[str, str]:
        """``--skip-build``: digest of ``latest<suffix>`` for each architecture."""
        repo_uri = outputs["RepoUri"]
        repo_name = outputs.get("RepoName") or repo_uri.split("/", 1)[1]
        images = {}
        for arch in archs:
            tag = "latest" + ARCH_SPECS[arch].tag_suffix
            uri = self.find_image(repo_name, repo_uri, tag)
            if not uri:
                raise DeployError(f"--skip-build: no {repo_uri}:{tag} image in ECR; run once without --skip-build")
            self.log(f"  [{arch}] reusing {uri}")
            images[arch] = uri
        return images

    # -- verification ---------------------------------------------------------------------------
    def smoke_test(self, url: str, attempts: int = 8) -> dict:
        """GET <url>/healthz until it returns {"ok": true}; returns the payload (+ latency_ms)."""
        target = url.rstrip("/") + "/healthz"
        delay = 3.0
        last = ""
        for i in range(1, attempts + 1):
            t0 = time.perf_counter()
            try:
                status, body = self.http_get(target, 60.0)
                ms = (time.perf_counter() - t0) * 1000.0
                if status == 200:
                    data = json.loads(body.decode("utf-8"))
                    if isinstance(data, dict) and data.get("ok") is True:
                        cv = data.get("cv") or {}
                        self.log(
                            f"  healthz OK in {ms:.0f} ms: version {data.get('version')}, OpenCV {cv.get('opencv_version')}, "
                            f"arch {cv.get('arch')}, runtime {cv.get('runtime')}"
                        )
                        return {**data, "latency_ms": round(ms, 1)}
                    last = f"unexpected body {body[:200]!r}"
                else:
                    last = f"HTTP {status}: {body[:200]!r}"
            except Exception as exc:  # noqa: BLE001 - network errors are retried
                last = f"{type(exc).__name__}: {exc}"
            if i < attempts:
                self.log(f"  healthz attempt {i}/{attempts} failed ({last}); retrying in {delay:.0f}s")
                self.sleep(delay)
                delay = min(delay * 2, 20.0)
        raise DeployError(f"smoke test failed for {target}: {last}")

    def remote_bench(self, arch: str, url: str, n: int) -> dict | None:
        """Run ``GET /api/bench?n=N`` on a deployed function and save it under results/."""
        target = url.rstrip("/") + f"/api/bench?n={int(n)}"
        try:
            status, body = self.http_get(target, 70.0)
        except Exception as exc:  # noqa: BLE001
            self.log(f"  [{arch}] bench request failed: {type(exc).__name__}: {exc}")
            return None
        if status != 200:
            self.log(f"  [{arch}] bench returned HTTP {status}")
            return None
        data = json.loads(body.decode("utf-8"))
        out = self.results_dir / f"bench_lambda_{arch}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        record = {"function_url": url, "lambda_arch": arch, "measured_at": datetime.now(timezone.utc).isoformat(), "bench": data}
        out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        self.log(f"  [{arch}] bench saved to {out}")
        return data

    # -- top-level flows --------------------------------------------------------------------------
    def base_parameters(self, model_id: str, reserved: int, opts: DeployOptions) -> dict[str, str | None]:
        return {
            "StackPrefix": self.prefix,
            "BedrockModelId": model_id,
            "ReservedConcurrency": str(int(reserved)),
            "AllowMarketplaceSubscribe": "true" if opts.allow_marketplace else "false",
            "BuildComputeType": opts.build_compute,
        }

    def deploy(self, opts: DeployOptions) -> DeployResult:
        archs = tuple(dict.fromkeys(opts.archs))
        for a in archs:
            if a not in ARCH_SPECS:
                raise DeployError(f"unknown architecture {a!r}")
        total = 6
        self.log(f"== [1/{total}] Inputs (stack {self.stack_name}, prefix {self.prefix}, region {self.region}, arch {'+'.join(archs)})")
        self.caller_identity()
        model_id = self.resolve_bedrock_model(opts.bedrock_model, opts.verify_model, opts.use_planner_picker)
        existing = self.describe_stack()
        prev = self.stack_parameters(existing)
        deployed_archs = {a for a, s in ARCH_SPECS.items() if prev.get(s.image_param)}
        reserved = self.resolve_reserved_concurrency(opts.reserved_concurrency, len(set(archs) | deployed_archs))

        self.log(f"== [2/{total}] CloudFormation stack, phase 1 (infrastructure; existing images kept)")
        params = self.base_parameters(model_id, reserved, opts)
        for spec in ARCH_SPECS.values():
            params[spec.image_param] = None if prev.get(spec.image_param) is not None else ""
        outputs = self.deploy_stack(params, "phase 1")

        if opts.skip_build:
            self.log(f"== [3/{total}] Images: --skip-build, reusing the latest images in ECR")
            images = self.latest_images(outputs, archs)
        else:
            self.log(f"== [3/{total}] Images: zip source, upload, CodeBuild ({', '.join(archs)})")
            images = self.build_images(outputs, archs, opts.root)

        self.log(f"== [4/{total}] CloudFormation stack, phase 2 (Lambda function(s) + Function URL(s))")
        params = self.base_parameters(model_id, reserved, opts)
        for arch, spec in ARCH_SPECS.items():
            params[spec.image_param] = images[arch] if arch in images else None
        outputs = self.deploy_stack(params, "phase 2: " + ", ".join(f"{a}={u.rsplit('@', 1)[-1][:19]}" for a, u in images.items()))

        result = DeployResult(self.stack_name, self.region, outputs, images=images, bedrock_model_id=model_id)
        for arch in archs:
            url = outputs.get(ARCH_SPECS[arch].url_output)
            if not url:
                raise DeployError(f"stack has no {ARCH_SPECS[arch].url_output} output after phase 2")
            result.urls[arch] = url

        self.log(f"== [5/{total}] Smoke test")
        if opts.smoke_test:
            for arch, url in result.urls.items():
                self.log(f"  [{arch}] GET {url.rstrip('/')}/healthz")
                result.health[arch] = self.smoke_test(url)
            if opts.bench_n > 0:
                for arch, url in result.urls.items():
                    self.remote_bench(arch, url, opts.bench_n)
        else:
            self.log("  skipped (--no-smoke-test)")

        self.log(f"== [6/{total}] Done")
        self._write_summary(result)
        self._print_summary(result)
        return result

    def _write_summary(self, result: DeployResult) -> None:
        try:
            self.build_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "stack_name": result.stack_name,
                "region": result.region,
                "deployed_at": datetime.now(timezone.utc).isoformat(),
                "outputs": result.outputs,
                "images": result.images,
                "urls": result.urls,
                "bedrock_model_id": result.bedrock_model_id,
                "health": {a: {k: h.get(k) for k in ("version", "latency_ms")} | {"cv": h.get("cv")} for a, h in result.health.items()},
            }
            (self.build_dir / "last_deploy.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except OSError as exc:
            self.log(f"  (could not write deploy summary: {exc})")

    def _print_summary(self, result: DeployResult) -> None:
        line = "=" * 78
        print(line, flush=True)
        print(f"InspectAgent deployed: stack {result.stack_name} in {result.region}", flush=True)
        for arch, url in result.urls.items():
            h = result.health.get(arch) or {}
            cv = h.get("cv") or {}
            extra = f"  (healthz ok, OpenCV {cv.get('opencv_version')} on {cv.get('arch')})" if h else ""
            print(f"  {arch:<7} {url}{extra}", flush=True)
        o = result.outputs
        print(f"  bucket  {o.get('BucketName')}\n  table   {o.get('TableName')}\n  repo    {o.get('RepoUri')}", flush=True)
        print(f"  bedrock {result.bedrock_model_id or '(disabled - rule planner only)'}", flush=True)
        print(line, flush=True)

    def empty_bucket(self, bucket: str) -> int:
        """Delete every object version and delete marker (CloudFormation cannot delete a non-empty bucket)."""
        ClientError = self._client_error_type()
        s3 = self.client("s3")
        deleted = 0
        kw: dict[str, Any] = {"Bucket": bucket}
        try:
            while True:
                resp = s3.list_object_versions(**kw)
                objs = [{"Key": v["Key"], "VersionId": v["VersionId"]} for v in resp.get("Versions", []) + resp.get("DeleteMarkers", [])]
                for i in range(0, len(objs), 1000):
                    res = s3.delete_objects(Bucket=bucket, Delete={"Objects": objs[i : i + 1000], "Quiet": True})
                    if res.get("Errors"):
                        first = res["Errors"][0]
                        raise DeployError(f"could not delete s3://{bucket}/{first.get('Key')}: {first.get('Code')} {first.get('Message')}")
                    deleted += len(objs[i : i + 1000])
                if not resp.get("IsTruncated"):
                    break
                kw = {"Bucket": bucket, "KeyMarker": resp.get("NextKeyMarker"), "VersionIdMarker": resp.get("NextVersionIdMarker")}
                kw = {k: v for k, v in kw.items() if v}
        except ClientError as exc:
            if _client_error_code(exc) == "NoSuchBucket":
                return deleted
            raise
        return deleted

    def destroy(self) -> None:
        self.log(f"== [1/3] Locating stack {self.stack_name} in {self.region}")
        self.caller_identity()
        stack = self.prepare_stack_for_delete()
        if stack is None:
            self.log("Stack not found - nothing to destroy")
            return
        outputs = self.stack_outputs(stack)
        bucket = outputs.get("BucketName")
        self.log("== [2/3] Emptying the artifacts bucket (ECR images are removed by EmptyOnDelete)")
        if bucket:
            n = self.empty_bucket(bucket)
            self.log(f"  deleted {n} object version(s) from s3://{bucket}")
        self.log(f"== [3/3] Deleting stack {self.stack_name}")
        for attempt in (1, 2):
            seen = self._event_ids()
            self.client("cloudformation").delete_stack(StackName=self.stack_name)
            try:
                self.wait_stack(seen, {"DELETE_COMPLETE"}, allow_missing=True)
                break
            except DeployError:
                # Objects written by the still-running function after the first emptying make the bucket
                # deletion fail once; empty it again and retry.
                if attempt == 2 or not bucket:
                    raise
                self.log("  deletion failed; emptying the bucket again and retrying")
                self.empty_bucket(bucket)
        self.log("Destroyed. (Bedrock model access and account-level settings are untouched.)")

    def prepare_stack_for_delete(self) -> dict | None:
        stack = self.describe_stack()
        if stack is not None and stack["StackStatus"].endswith("_IN_PROGRESS"):
            self.log(f"  stack is {stack['StackStatus']}; waiting")
            self.wait_stack(self._event_ids(), {s for s in _ALL_SETTLED}, allow_missing=True)
            stack = self.describe_stack()
        return stack


_ALL_SETTLED = {
    "CREATE_COMPLETE", "CREATE_FAILED", "ROLLBACK_COMPLETE", "ROLLBACK_FAILED", "UPDATE_COMPLETE",
    "UPDATE_ROLLBACK_COMPLETE", "UPDATE_ROLLBACK_FAILED", "DELETE_FAILED", "DELETE_COMPLETE", "IMPORT_COMPLETE",
    "IMPORT_ROLLBACK_COMPLETE", "UPDATE_FAILED",
}


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deploy.py",
        description="Deploy InspectAgent to AWS: CloudFormation stack, CodeBuild image build (Graviton arm64 and/or "
        "x86_64), Lambda container + public Function URL, smoke test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
        "  python deploy/aws/deploy.py --region us-east-1\n"
        "  python deploy/aws/deploy.py --region us-east-1 --arch both --bench 5\n"
        "  python deploy/aws/deploy.py --skip-build --bedrock-model none\n"
        "  python deploy/aws/deploy.py --destroy --yes\n",
    )
    p.add_argument("--region", default=None, help="AWS region (default: AWS_REGION / profile region / us-east-1)")
    p.add_argument("--profile", default=None, help="named AWS CLI profile")
    p.add_argument("--stack-name", default=DEFAULT_STACK_NAME, help="CloudFormation stack name (default: %(default)s)")
    p.add_argument(
        "--bedrock-model",
        default="auto",
        metavar="auto|none|<id>",
        help="Bedrock model or inference-profile id for the Bedrock planner; 'auto' picks the newest available "
        "Claude Sonnet/Haiku (else Amazon Nova) and verifies access with a 5-token Converse call; "
        "'none' disables Bedrock (default: %(default)s)",
    )
    p.add_argument("--no-verify-model", action="store_true", help="with --bedrock-model auto: skip the access check call")
    p.add_argument("--arch", choices=sorted(ARCH_CHOICES), default="arm64",
                   help="arm64 = Graviton function; x86_64 = only the <prefix>-x86 function; both = both (default: %(default)s)")
    p.add_argument("--skip-build", action="store_true", help="do not zip/build; redeploy the latest image(s) already in ECR")
    p.add_argument("--destroy", action="store_true", help="empty the bucket and delete the whole stack")
    p.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation (required for --destroy without a TTY)")
    p.add_argument("--reserved-concurrency", type=int, default=None,
                   help="reserved concurrency per function (default: 5 if the account quota allows, else 0)")
    p.add_argument("--allow-marketplace-subscribe", action="store_true",
                   help="grant aws-marketplace Subscribe/ViewSubscriptions (via Bedrock only) for first-use model enablement")
    p.add_argument("--build-compute", choices=("small", "large"), default="small", help="CodeBuild compute size (default: %(default)s)")
    p.add_argument("--no-smoke-test", action="store_true", help="skip GET /healthz after deployment")
    p.add_argument("--bench", type=int, default=0, metavar="N",
                   help="after deploying, run GET /api/bench?n=N on each function and save results/bench_lambda_<arch>.json")
    p.add_argument("--poll-seconds", type=float, default=10.0, help="status polling interval (default: %(default)s)")
    return p


def _resolve_region(arg_region: str | None, session: Any) -> str:
    return arg_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or getattr(session, "region_name", None) or "us-east-1"


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    args = build_arg_parser().parse_args(argv)
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError, NoRegionError
    except ImportError:
        print("ERROR: boto3 is required: pip install boto3", file=sys.stderr)
        return 2
    try:
        session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
        region = _resolve_region(args.region, session)
        deployer = Deployer(session, args.stack_name, region, poll_seconds=max(0.0, args.poll_seconds))
        if args.destroy:
            if not args.yes:
                if not sys.stdin.isatty():
                    raise DeployError("refusing to destroy without --yes in a non-interactive shell")
                answer = input(f"Delete stack '{args.stack_name}' in {region} and ALL its data? Type the stack name to confirm: ")
                if answer.strip() != args.stack_name:
                    print("Aborted.")
                    return 1
            deployer.destroy()
            return 0
        opts = DeployOptions(
            archs=ARCH_CHOICES[args.arch],
            bedrock_model=args.bedrock_model,
            verify_model=not args.no_verify_model,
            skip_build=args.skip_build,
            reserved_concurrency=args.reserved_concurrency,
            allow_marketplace=args.allow_marketplace_subscribe,
            build_compute={"small": "BUILD_GENERAL1_SMALL", "large": "BUILD_GENERAL1_LARGE"}[args.build_compute],
            smoke_test=not args.no_smoke_test,
            bench_n=max(0, args.bench),
        )
        deployer.deploy(opts)
        return 0
    except NoCredentialsError:
        print("ERROR: no AWS credentials found. Run `aws configure` (or `aws sso login`) or pass --profile.", file=sys.stderr)
        return 2
    except NoRegionError:
        print("ERROR: no AWS region configured; pass --region.", file=sys.stderr)
        return 2
    except DeployError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ClientError as exc:
        print(f"ERROR: AWS API call failed: {_client_error_code(exc)}: {_client_error_message(exc)}", file=sys.stderr)
        return 1
    except BotoCoreError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume (every step is idempotent).", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
