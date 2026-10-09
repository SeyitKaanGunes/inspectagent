"""OpenCV build / architecture / runtime introspection and thread configuration.

``cv_info()`` is embedded in every trace and served by ``/healthz`` and ``/api/system`` so
that results can be attributed to a concrete OpenCV build (stock wheel vs. a
Cloud-Optimized "COOL" build), CPU architecture (x86_64 vs. aarch64/Graviton) and runtime
(AWS Lambda, container, local).
"""

from __future__ import annotations

import os
import platform
import re
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

__all__ = ["cv_info", "configure_threads", "detect_backend", "detect_runtime", "parse_build_info"]

_BACKEND_ENV = "INSPECTAGENT_CV_BACKEND"
_COOL_RE = re.compile(r"\bCOOL\b|cloud[\s_-]*optimi[sz]ed", re.IGNORECASE)


@lru_cache(maxsize=1)
def _build_information() -> str:
    try:
        return cv2.getBuildInformation()
    except Exception:  # pragma: no cover - extremely unusual builds
        return ""


def _field(text: str, key: str) -> str:
    """Value of the first ``key: value`` line in OpenCV build information ('' if absent)."""
    m = re.search(rf"^\s*{re.escape(key)}\s*:\s*(.*?)\s*$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def parse_build_info(text: str) -> dict:
    """Extract the fields InspectAgent reports from ``cv2.getBuildInformation()`` text."""
    return {
        "cpu_baseline": _field(text, "Baseline"),
        "cpu_dispatch": _field(text, "Dispatched code generation"),
        "parallel_framework": _field(text, "Parallel framework"),
        "build_timestamp": _field(text, "Timestamp"),
        "build_host": _field(text, "Host"),
        "compiler": _field(text, "C++ Compiler"),
        "onnxruntime": _field(text, "ONNX Runtime"),
        "opencl": _field(text, "OpenCL"),
    }


def detect_backend(text: str | None = None, env: dict | None = None) -> tuple[str, str]:
    """Return ``(build_backend, evidence)``: ``"cool"`` or ``"stock"`` plus the raw evidence string.

    ``cool`` if env ``INSPECTAGENT_CV_BACKEND=cool`` or the build information mentions
    ``COOL`` / ``Cloud-Optimized`` (case-insensitive).
    """
    environ = os.environ if env is None else env
    text = _build_information() if text is None else text
    val = str(environ.get(_BACKEND_ENV, "")).strip()
    if val.lower() == "cool":
        return "cool", f"env:{_BACKEND_ENV}={val}"
    for line in text.splitlines():
        if _COOL_RE.search(line):
            return "cool", f"build_info:{line.strip()}"
    if val:
        return "stock", f"env:{_BACKEND_ENV}={val}"
    return "stock", "no COOL/Cloud-Optimized marker in cv2.getBuildInformation()"


def detect_runtime(env: dict | None = None) -> str:
    """``"lambda"`` on AWS Lambda, ``"container"`` inside Docker/ECS/Kubernetes, else ``"local"``."""
    environ = os.environ if env is None else env
    if environ.get("AWS_LAMBDA_FUNCTION_NAME") or str(environ.get("AWS_EXECUTION_ENV", "")).startswith("AWS_Lambda"):
        return "lambda"
    if environ.get("ECS_CONTAINER_METADATA_URI") or environ.get("ECS_CONTAINER_METADATA_URI_V4"):
        return "container"
    if environ.get("KUBERNETES_SERVICE_HOST") or environ.get("container"):
        return "container"
    if env is None:
        try:
            if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
                return "container"
            cg = Path("/proc/1/cgroup")
            if cg.is_file() and re.search(r"docker|containerd|kubepods|libpod", cg.read_text(errors="ignore")):
                return "container"
        except OSError:
            pass
    return "local"


def _dnn_engines() -> list[str]:
    dnn = getattr(cv2, "dnn", None)
    if dnn is None:
        return []
    return sorted(name for name in dir(dnn) if name.startswith("ENGINE_"))


def cv_info() -> dict:
    """OpenCV build / arch / runtime summary (JSON-serializable)."""
    text = _build_information()
    backend, evidence = detect_backend(text)
    parsed = parse_build_info(text)
    return {
        "opencv_version": cv2.__version__,
        "build_backend": backend,
        "backend_evidence": evidence,
        "arch": platform.machine(),
        "cpu_baseline": parsed["cpu_baseline"],
        "cpu_dispatch": parsed["cpu_dispatch"],
        "parallel_framework": parsed["parallel_framework"],
        "threads": int(cv2.getNumThreads()),
        "cpu_count": int(os.cpu_count() or 1),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "dnn_engines": _dnn_engines(),
        "runtime": detect_runtime(),
        "platform": sys.platform,
        "build_timestamp": parsed["build_timestamp"],
        "opencl": parsed["opencl"],
        "onnxruntime": parsed["onnxruntime"],
    }


def configure_threads(n: int | None = None) -> int:
    """Set the OpenCV worker thread count and return the effective value.

    ``n=None`` uses ``Settings.cv_threads`` (env ``INSPECTAGENT_CV_THREADS``); a value
    ``<= 0`` restores OpenCV's default (all cores). ``n=1`` runs single-threaded.
    """
    if n is None:
        try:
            from inspectagent.config import get_settings

            n = int(get_settings().cv_threads)
        except Exception:
            n = 0
    if n > 0:
        cv2.setNumThreads(int(n))
    else:
        cv2.setNumThreads(-1)  # negative = reset to the library default
    return int(cv2.getNumThreads())
