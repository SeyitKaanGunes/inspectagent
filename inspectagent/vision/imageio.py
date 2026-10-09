"""Unicode-safe image I/O helpers.

``cv2.imread`` / ``cv2.imwrite`` take narrow (ANSI) paths on Windows and silently fail
for paths containing spaces/non-ASCII characters under some locales (the dev box home
directory is ``C:\\Users\\Seyit Kaan`` with a Turkish locale). Everything in InspectAgent
therefore reads files through ``np.fromfile`` + ``cv2.imdecode`` and writes through
``cv2.imencode`` + ``ndarray.tofile``.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import cv2
import numpy as np

__all__ = ["imread", "imwrite", "encode_jpeg", "encode_png", "decode_image", "to_uint8"]

_DEFAULT_EXT = ".png"


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Return ``img`` as uint8 (clipping/rounding other dtypes; floats in [0, 1] are scaled by 255).

    Float images whose maximum is <= 1.0 are interpreted as normalized intensities.
    """
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * 255
    if np.issubdtype(arr.dtype, np.floating):
        a = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
        if a.size and float(a.max()) <= 1.0:
            a = a * 255.0
        return np.clip(a + 0.5, 0, 255).astype(np.uint8)
    if arr.dtype == np.uint16:
        return (arr >> 8).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image from ``path`` (unicode-safe).

    Raises ``FileNotFoundError`` if the file does not exist and ``ValueError`` if it is
    empty or cannot be decoded.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"image not found: {p}")
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"empty image file: {p}")
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"cannot decode image: {p}")
    return img


def imwrite(path: str | Path, img: np.ndarray, params: list[int] | None = None) -> None:
    """Write ``img`` to ``path`` (format from the extension, PNG if none); creates parent dirs.

    The file is written atomically (temporary file + ``os.replace``) so concurrent readers
    never observe a partially written image. Raises ``ValueError`` if encoding fails.
    """
    p = Path(path)
    ext = p.suffix.lower() or _DEFAULT_EXT
    arr = np.asarray(img)
    if arr.dtype not in (np.uint8, np.uint16) and ext not in (".exr", ".hdr", ".pfm", ".tif", ".tiff"):
        arr = to_uint8(arr)
    try:
        ok, buf = cv2.imencode(ext, arr, list(params) if params else [])
    except cv2.error as exc:  # e.g. no encoder for this extension
        raise ValueError(f"cannot encode image as {ext!r}: {p} ({exc})") from exc
    if not ok:
        raise ValueError(f"cannot encode image as {ext!r}: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        buf.tofile(str(tmp))
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    """Encode an image (BGR or gray; non-uint8 is converted) as JPEG bytes."""
    q = int(min(100, max(1, round(quality))))
    ok, buf = cv2.imencode(".jpg", to_uint8(img), [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def encode_png(img: np.ndarray, compression: int = 3) -> bytes:
    """Encode an image as PNG bytes (lossless; uint16 preserved)."""
    arr = np.asarray(img)
    if arr.dtype not in (np.uint8, np.uint16):
        arr = to_uint8(arr)
    ok, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, int(min(9, max(0, compression)))])
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


def decode_image(data: bytes | bytearray | memoryview, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Decode encoded image bytes (JPEG/PNG/...). Raises ``ValueError`` on empty/corrupt data."""
    buf = np.frombuffer(bytes(data) if isinstance(data, memoryview) else data, dtype=np.uint8)
    if buf.size == 0:
        raise ValueError("empty image buffer")
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise ValueError("cannot decode image buffer")
    return img
