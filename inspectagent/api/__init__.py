"""Web API (FastAPI + static UI), storage backends and the AWS Lambda entry point.

``create_app`` is resolved lazily so that ``inspectagent.api.storage`` can be used without importing
FastAPI or building the default application. The ASGI application object itself lives at
``inspectagent.api.app:app`` (e.g. ``uvicorn inspectagent.api.app:app``); note that the package
attribute ``inspectagent.api.app`` is the *module* of that name.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    if name == "create_app":
        return importlib.import_module("inspectagent.api.app").create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
