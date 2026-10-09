"""Vision infrastructure: unicode-safe image I/O (vision.imageio), OpenCV build info (vision.cvinfo)
and the lazy ``inspectagent.vision`` package namespace."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from inspectagent.vision import cvinfo
from inspectagent.vision.imageio import decode_image, encode_jpeg, encode_png, imread, imwrite, to_uint8


@pytest.fixture
def unicode_dir(tmp_path):
    # spaces + Turkish characters, like C:\Users\Seyit Kaan on a Turkish-locale box
    return tmp_path / "Seyit Kaan ğüşöçİı" / "alt klasör"


def test_png_roundtrip_unicode_path_creates_parents(board, unicode_dir):
    path = unicode_dir / "görüntü 1.png"
    imwrite(path, board)
    assert path.is_file()
    back = imread(path)
    assert back.dtype == np.uint8 and back.shape == board.shape
    assert np.array_equal(back, board)
    # no temporary files left behind
    assert sorted(p.name for p in unicode_dir.iterdir()) == ["görüntü 1.png"]


def test_jpeg_roundtrip_and_flags(board, unicode_dir):
    path = unicode_dir / "kart.jpg"
    imwrite(str(path), board, [cv2.IMWRITE_JPEG_QUALITY, 95])
    back = imread(path)
    assert back.shape == board.shape
    assert float(np.abs(back.astype(np.int16) - board.astype(np.int16)).mean()) < 3.0
    gray = imread(path, cv2.IMREAD_GRAYSCALE)
    assert gray.ndim == 2 and gray.shape == board.shape[:2]


def test_imwrite_without_extension_defaults_to_png(board, tmp_path):
    path = tmp_path / "no_ext"
    imwrite(path, board)
    assert np.array_equal(imread(path), board)


def test_imwrite_converts_float_images(tmp_path):
    img = np.linspace(0.0, 1.0, 64 * 32, dtype=np.float32).reshape(32, 64)
    path = tmp_path / "f.png"
    imwrite(path, img)
    back = imread(path, cv2.IMREAD_GRAYSCALE)
    assert back.dtype == np.uint8 and back[0, 0] == 0 and back[-1, -1] == 255


def test_imread_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        imread(tmp_path / "missing.png")
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(ValueError):
        imread(empty)
    junk = tmp_path / "junk.jpg"
    junk.write_bytes(b"this is not an image" * 10)
    with pytest.raises(ValueError):
        imread(junk)


def test_imwrite_unknown_extension_raises(board, tmp_path):
    with pytest.raises(ValueError):
        imwrite(tmp_path / "x.notanimageformat", board)
    assert not any(tmp_path.iterdir())


def test_encode_decode_jpeg(board):
    hi = encode_jpeg(board, 95)
    lo = encode_jpeg(board, 30)
    assert isinstance(hi, bytes) and hi[:2] == b"\xff\xd8"
    assert len(lo) < len(hi)
    dec = decode_image(hi)
    assert dec.shape == board.shape and dec.dtype == np.uint8
    assert decode_image(bytearray(hi)).shape == board.shape
    assert decode_image(memoryview(hi)).shape == board.shape
    assert decode_image(hi, cv2.IMREAD_GRAYSCALE).ndim == 2
    # out-of-range quality is clamped, gray and float inputs are accepted
    assert decode_image(encode_jpeg(board[..., 0], 500), cv2.IMREAD_UNCHANGED).ndim == 2
    assert decode_image(encode_jpeg(board.astype(np.float32), 0)).shape == board.shape


def test_encode_png_lossless(board):
    assert np.array_equal(decode_image(encode_png(board)), board)
    deep = (np.arange(40 * 30, dtype=np.uint16).reshape(30, 40) * 50).astype(np.uint16)
    assert np.array_equal(decode_image(encode_png(deep), cv2.IMREAD_UNCHANGED), deep)


def test_decode_image_errors():
    with pytest.raises(ValueError):
        decode_image(b"")
    with pytest.raises(ValueError):
        decode_image(b"\x00\x01garbage")


def test_to_uint8():
    assert to_uint8(np.array([0.0, 0.5, 1.0], np.float32)).tolist() == [0, 128, 255]
    assert to_uint8(np.array([-5.0, 100.4, 300.0])).tolist() == [0, 100, 255]
    assert to_uint8(np.array([True, False])).tolist() == [255, 0]
    assert to_uint8(np.array([0, 65535], np.uint16)).tolist() == [0, 255]
    assert to_uint8(np.array([np.nan, 0.25], np.float32)).tolist() == [0, 64]
    u = np.array([1, 2], np.uint8)
    assert to_uint8(u) is u


# ---------------------------------------------------------------------------------- cvinfo
def test_cv_info_contract(monkeypatch):
    monkeypatch.delenv("INSPECTAGENT_CV_BACKEND", raising=False)
    info = cvinfo.cv_info()
    for key in ("opencv_version", "build_backend", "backend_evidence", "arch", "cpu_baseline", "cpu_dispatch",
                "parallel_framework", "threads", "python", "numpy", "dnn_engines", "runtime"):
        assert key in info, key
    assert info["opencv_version"] == cv2.__version__
    assert info["build_backend"] in ("stock", "cool") and info["backend_evidence"]
    assert info["runtime"] in ("lambda", "local", "container")
    assert isinstance(info["threads"], int) and info["threads"] >= 1
    assert all(e.startswith("ENGINE_") for e in info["dnn_engines"])
    json.dumps(info)


def test_cool_backend_detection(monkeypatch):
    assert cvinfo.detect_backend("General configuration\n  Parallel framework: pthreads\n", env={})[0] == "stock"
    kind, ev = cvinfo.detect_backend("  Build flavour: Cloud-Optimized OpenCV (aarch64)\n", env={})
    assert kind == "cool" and "Cloud-Optimized" in ev
    assert cvinfo.detect_backend("  Vendor: cool build\n", env={})[0] == "cool"
    assert cvinfo.detect_backend("  Algorithm: Cooley-Tukey FFT\n", env={})[0] == "stock"  # not a word match
    kind, ev = cvinfo.detect_backend("", env={"INSPECTAGENT_CV_BACKEND": "COOL"})
    assert kind == "cool" and ev.startswith("env:")
    monkeypatch.setenv("INSPECTAGENT_CV_BACKEND", "cool")
    assert cvinfo.cv_info()["build_backend"] == "cool"


def test_runtime_detection():
    assert cvinfo.detect_runtime({"AWS_LAMBDA_FUNCTION_NAME": "inspectagent"}) == "lambda"
    assert cvinfo.detect_runtime({"AWS_EXECUTION_ENV": "AWS_Lambda_python3.13"}) == "lambda"
    assert cvinfo.detect_runtime({"ECS_CONTAINER_METADATA_URI_V4": "http://x"}) == "container"
    assert cvinfo.detect_runtime({}) == "local"


def test_parse_build_info():
    text = (
        "  CPU/HW features:\n"
        "    Baseline:   NEON FP16\n"
        "    Dispatched code generation:   NEON_DOTPROD\n"
        "  Parallel framework:   pthreads\n"
    )
    parsed = cvinfo.parse_build_info(text)
    assert parsed["cpu_baseline"] == "NEON FP16" and parsed["cpu_dispatch"] == "NEON_DOTPROD"
    assert parsed["parallel_framework"] == "pthreads"


def test_configure_threads():
    before = cv2.getNumThreads()
    try:
        assert cvinfo.configure_threads(2) == 2
        assert cvinfo.configure_threads(0) >= 1  # <= 0 restores the library default
    finally:
        cv2.setNumThreads(before)


def test_vision_package_lazy_exports():
    import inspectagent.vision as vision

    assert vision.imread is imread
    from inspectagent.vision.camera import VirtualCamera
    from inspectagent.vision.quality import assess_quality

    assert vision.VirtualCamera is VirtualCamera and vision.assess_quality is assess_quality
    assert "heatmap_overlay" in dir(vision)
    with pytest.raises(AttributeError):
        vision.does_not_exist  # noqa: B018
