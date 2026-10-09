"""Vision layer: camera simulator, quality assessment, registration, detectors, overlays.

Only the lightweight, dependency-free I/O helpers are imported eagerly; everything else is
resolved lazily on first attribute access (``from inspectagent.vision import assess_quality``
works) so importing the package stays cheap and independent of optional modules.
"""

from __future__ import annotations

import importlib
from typing import Any

from inspectagent.vision.imageio import decode_image, encode_jpeg, imread, imwrite

_LAZY: dict[str, str] = {
    # camera
    "VirtualCamera": "camera",
    "make_conditions": "camera",
    "PRESETS": "camera",
    "LIGHT_GAIN": "camera",
    # quality
    "assess_quality": "quality",
    "laplacian_sharpness": "quality",
    "estimate_noise_sigma": "quality",
    "detect_glare": "quality",
    # overlays
    "heatmap_overlay": "overlay",
    "draw_regions": "overlay",
    "draw_quality": "overlay",
    "thumbnail": "overlay",
    "jpeg_data_url": "overlay",
    # build/runtime info
    "cv_info": "cvinfo",
    "configure_threads": "cvinfo",
}

__all__ = ["imread", "imwrite", "encode_jpeg", "decode_image", *_LAZY]


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{mod}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
