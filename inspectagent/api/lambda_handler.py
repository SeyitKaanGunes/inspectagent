"""AWS Lambda entry point: ``inspectagent.api.lambda_handler.handler`` (Function URL / API Gateway v2).

The FastAPI app is wrapped by Mangum (``lifespan="off"``). Model warm-up runs once per container at
import time (Lambda init phase) and is fully guarded: a failure is logged and the function still serves
requests (bundles are then loaded lazily on first use). Every invocation emits one structured JSON log
line (request id, cold start, path, status, duration) for CloudWatch.

Environment knobs:
    INSPECTAGENT_WARMUP=0                    skip warm-up (e.g. in tests)
    INSPECTAGENT_WARMUP_CATEGORIES=a,b       restrict warm-up to these categories
    INSPECTAGENT_WARMUP_BUDGET_S=8           stop loading further bundles after this many seconds
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from mangum import Mangum

from inspectagent.api.app import app, available_categories, cv_info, load_bundle, log_event
from inspectagent.config import get_settings

__all__ = ["handler", "asgi_handler", "warmup", "app"]

_COLD_START = True


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off", "")


def warmup() -> dict:
    """Configure OpenCV threads and preload model bundles (never raises)."""
    report: dict[str, Any] = {"loaded": [], "failed": {}, "skipped": False}
    if not _env_flag("INSPECTAGENT_WARMUP", True):
        report["skipped"] = True
        log_event("warmup", **report)
        return report
    t0 = time.perf_counter()
    settings = get_settings()
    try:
        try:
            from inspectagent.vision.cvinfo import configure_threads

            report["threads"] = configure_threads(settings.cv_threads or None)
        except ImportError:
            import cv2

            if settings.cv_threads:
                cv2.setNumThreads(int(settings.cv_threads))
            report["threads"] = cv2.getNumThreads()
        info = cv_info()
        report["opencv"] = info.get("opencv_version")
        report["arch"] = info.get("arch")
        report["build_backend"] = info.get("build_backend")

        wanted = [c.strip() for c in os.environ.get("INSPECTAGENT_WARMUP_CATEGORIES", "").split(",") if c.strip()]
        budget_s = float(os.environ.get("INSPECTAGENT_WARMUP_BUDGET_S", "8"))
        categories = available_categories(settings.models_dir)
        if wanted:
            categories = [c for c in categories if c in wanted]
        for cat in categories:
            if time.perf_counter() - t0 > budget_s:
                report["budget_exhausted"] = True
                break
            try:
                load_bundle(cat, models_dir=settings.models_dir, with_deep=True)
                report["loaded"].append(cat)
            except Exception as exc:  # deep model missing etc. -> try the fast detector only
                try:
                    load_bundle(cat, models_dir=settings.models_dir, with_deep=False)
                    report["loaded"].append(f"{cat}(fast-only)")
                except Exception as exc2:
                    report["failed"][cat] = f"{type(exc2).__name__}: {exc2} (deep: {type(exc).__name__})"
    except Exception as exc:  # warm-up must never break the container init
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    log_event("warmup", logging.WARNING if report.get("error") or report["failed"] else logging.INFO, **report)
    return report


WARMUP_REPORT = warmup()

asgi_handler = Mangum(app, lifespan="off")


def handler(event: dict, context: Any) -> dict:
    """Lambda handler: delegate to Mangum and log one structured line per invocation."""
    global _COLD_START
    cold, _COLD_START = _COLD_START, False
    t0 = time.perf_counter()
    status: Any = None
    try:
        response = asgi_handler(event, context)
        status = response.get("statusCode") if isinstance(response, dict) else None
        return response
    except Exception as exc:
        status = 500
        log_event("invocation_error", logging.ERROR, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        ctx_http = ((event or {}).get("requestContext") or {}).get("http") or {}
        log_event(
            "invocation",
            request_id=getattr(context, "aws_request_id", None),
            cold_start=cold,
            method=ctx_http.get("method") or (event or {}).get("httpMethod"),
            path=(event or {}).get("rawPath") or (event or {}).get("path"),
            status=status,
            ms=round((time.perf_counter() - t0) * 1000.0, 1),
            memory_mb=os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE"),
        )
