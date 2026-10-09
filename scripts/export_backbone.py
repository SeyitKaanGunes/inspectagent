"""Export the ResNet18 feature backbone (layer2 + layer3) to ONNX for ``cv2.dnn`` (SPEC §2.7).

Dev-only: needs torch + torchvision (``requirements-export.txt``); runtime inference uses OpenCV only.

* Model: torchvision ``resnet18`` with ``IMAGENET1K_V1`` weights, truncated after ``layer3``; outputs
  ``layer2`` (stride 8, 128 ch) and ``layer3`` (stride 16, 256 ch); input ``input`` (1, 3, H, W),
  ImageNet-normalised RGB (see ``inspectagent.vision.anomaly_deep.make_blob``).
* Exporter: the legacy TorchScript exporter (``dynamo=False``), opset 17. It emits a plain
  Conv/Relu/MaxPool/Add graph that every cv2.dnn engine loads. The dynamo exporter additionally needs
  ``onnxscript`` and is not used.
* Shapes: dynamic H/W by default (verified below by running cv2.dnn at two different input sizes);
  ``--fixed-size W H`` exports a static graph instead (then the detector always runs at that size).
* Verification: loads the file with ``cv2.dnn.readNetFromONNX`` for each engine, compares the outputs
  with torch on identical blobs (max abs diff), checks ``make_blob`` against torchvision-style
  preprocessing, and prints per-engine latency. Exit code 1 if any check fails.

Usage::

    python scripts/export_backbone.py                       # -> models/backbone_resnet18.onnx
    python scripts/export_backbone.py --fixed-size 448 304  # static graph
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from inspectagent.vision.anomaly_deep import (  # noqa: E402
    IMAGENET_MEAN_RGB,
    IMAGENET_STD_RGB,
    INPUT_NAME,
    OUTPUT_NAMES,
    default_onnx_path,
    make_blob,
    patch_features_from_layers,
)

ENGINES = {"new": "ENGINE_NEW", "auto": "ENGINE_AUTO", "classic": "ENGINE_CLASSIC", "ort": "ENGINE_ORT"}


def _import_torch():
    try:
        import torch
        import torchvision
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "torch/torchvision are required for the export (dev-only):\n"
            "  uv pip install --python .venv/Scripts/python.exe -r requirements-export.txt"
        ) from e
    return torch, torchvision


def build_backbone(weights: str = "IMAGENET1K_V1"):
    """ResNet18 truncated after layer3, returning ``(layer2, layer3)``; eval mode."""
    torch, torchvision = _import_torch()
    nn = torch.nn
    w = getattr(torchvision.models.ResNet18_Weights, weights)
    base = torchvision.models.resnet18(weights=w)

    class Backbone(nn.Module):
        """Stem + layer1..3 of ResNet18; forward returns (layer2, layer3)."""

        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
            self.layer1, self.layer2, self.layer3 = m.layer1, m.layer2, m.layer3

        def forward(self, x):
            x = self.layer1(self.stem(x))
            f2 = self.layer2(x)
            return f2, self.layer3(f2)

    return Backbone(base).eval()


def export_onnx(model, out: Path, opset: int = 17, fixed_size: tuple[int, int] | None = None) -> Path:
    """Export with the legacy TorchScript exporter; dynamic H/W unless ``fixed_size=(W, H)`` is given."""
    torch, _ = _import_torch()
    w, h = fixed_size or (448, 320)
    dummy = torch.zeros(1, 3, h, w)
    dynamic_axes = None
    if fixed_size is None:
        dynamic_axes = {
            INPUT_NAME: {2: "height", 3: "width"},
            OUTPUT_NAMES[0]: {2: "height_s8", 3: "width_s8"},
            OUTPUT_NAMES[1]: {2: "height_s16", 3: "width_s16"},
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX export.*")
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy,),
                str(tmp),
                dynamo=False,
                opset_version=opset,
                input_names=[INPUT_NAME],
                output_names=list(OUTPUT_NAMES),
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
            )
    _annotate(tmp, fixed_size, opset)
    tmp.replace(out)
    if Path(str(out) + ".data").exists() or Path(str(tmp) + ".data").exists():
        raise RuntimeError("exporter wrote external data; cv2.dnn needs a single self-contained file")
    return out


def _annotate(path: Path, fixed_size: tuple[int, int] | None, opset: int) -> None:
    """Add metadata props and run the ONNX checker (skipped if the ``onnx`` package is absent)."""
    try:
        import onnx
    except ImportError:
        print("  (onnx package not installed: skipping checker/metadata)")
        return
    torch, torchvision = _import_torch()
    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    props = {
        "inspectagent.model": "resnet18-IMAGENET1K_V1 truncated after layer3",
        "inspectagent.outputs": ",".join(OUTPUT_NAMES),
        "inspectagent.input": "1x3xHxW RGB, (x/255 - mean) / std",
        "inspectagent.mean_rgb": ",".join(str(v) for v in IMAGENET_MEAN_RGB),
        "inspectagent.std_rgb": ",".join(str(v) for v in IMAGENET_STD_RGB),
        "inspectagent.shape": "dynamic" if fixed_size is None else f"fixed {fixed_size[0]}x{fixed_size[1]}",
        "inspectagent.exporter": f"torch {torch.__version__} legacy TorchScript, opset {opset}",
        "inspectagent.torchvision": torchvision.__version__,
    }
    for k, v in props.items():
        e = m.metadata_props.add()
        e.key, e.value = k, v
    onnx.save(m, str(path))


def _test_image(w: int, h: int, seed: int) -> np.ndarray:
    """Deterministic natural-ish BGR test image (smoothed noise + shapes), uint8."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.normal(128, 60, (h, w, 3)).astype(np.float32), (0, 0), 2.0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(12):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        cv2.circle(img, (x, y), int(rng.integers(4, max(5, w // 8))), color, -1, cv2.LINE_AA)
    return img


def verify(path: Path, model, sizes: list[tuple[int, int]], engines: list[str], tol: float) -> bool:
    """Run cv2.dnn on ``sizes`` with each engine, compare to torch, print diffs/latency. True if all pass."""
    torch, _ = _import_torch()
    ok = True
    # Load by path: the in-memory overload ignores the engine argument in OpenCV 5.0.0.
    source: str | np.ndarray = str(path.resolve())
    if not source.isascii():
        print("NOTE: non-ASCII path -> in-memory load; cv2 then always uses the new engine")
        source = np.frombuffer(path.read_bytes(), np.uint8)

    # preprocessing parity: make_blob vs torchvision-style normalisation of the same resized image
    w0, h0 = sizes[0]
    img = _test_image(w0, h0, 0)
    rgb = img[..., ::-1].astype(np.float32) / 255.0
    ref = ((rgb - np.array(IMAGENET_MEAN_RGB, np.float32)) / np.array(IMAGENET_STD_RGB, np.float32)).transpose(2, 0, 1)[None]
    pre_diff = float(np.abs(make_blob(img, (w0, h0)) - ref).max())
    print(f"preprocessing: make_blob vs torchvision-style max|diff| = {pre_diff:.3e}")
    ok &= pre_diff < 1e-4

    refs = {}
    for i, (w, h) in enumerate(sizes):
        blob = make_blob(_test_image(w, h, i + 1), (w, h))
        with torch.no_grad():
            r2, r3 = model(torch.from_numpy(blob))
        refs[(w, h)] = (blob, r2.numpy(), r3.numpy())

    print(f"{'engine':8s} {'input WxH':>10s} {'layer2':>16s} {'layer3':>16s} {'max|diff| l2':>13s} {'l3':>10s} {'feat':>10s} {'ms':>7s}")
    working = []
    for name in engines:
        attr = ENGINES[name]
        if not hasattr(cv2.dnn, attr):
            print(f"{name:8s} not available in this OpenCV build")
            continue
        try:
            net = cv2.dnn.readNetFromONNX(source, getattr(cv2.dnn, attr))
        except cv2.error as e:
            print(f"{name:8s} LOAD FAILED: {str(e).strip().splitlines()[-1][:160]}")
            continue
        eng_ok = True
        for w, h in sizes:
            blob, r2, r3 = refs[(w, h)]
            try:
                net.setInput(blob)
                net.forward(list(OUTPUT_NAMES))  # warm-up (allocations / shape inference)
                times = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    net.setInput(blob)
                    o2, o3 = net.forward(list(OUTPUT_NAMES))
                    times.append((time.perf_counter() - t0) * 1e3)
                ms = float(np.median(times))
            except cv2.error as e:
                print(f"{name:8s} {w:>4d}x{h:<5d} FORWARD FAILED: {str(e).strip().splitlines()[-1][:160]}")
                eng_ok = False
                continue
            if o2.shape != r2.shape or o3.shape != r3.shape:
                print(f"{name:8s} {w:>4d}x{h:<5d} SHAPE MISMATCH {o2.shape} {o3.shape} vs {r2.shape} {r3.shape}")
                eng_ok = False
                continue
            d2 = float(np.abs(o2 - r2).max())
            d3 = float(np.abs(o3 - r3).max())
            fd = float(np.abs(patch_features_from_layers(o2, o3) - patch_features_from_layers(r2, r3)).max())
            good = max(d2, d3, fd) <= tol
            eng_ok &= good
            print(
                f"{name:8s} {w:>4d}x{h:<5d} {str(o2.shape[1:]):>16s} {str(o3.shape[1:]):>16s} "
                f"{d2:13.3e} {d3:10.3e} {fd:10.3e} {ms:7.1f}{'' if good else '  > tol!'}"
            )
        if eng_ok:
            working.append(name)
    print(f"engines passing all checks (tol {tol:g}): {working or 'NONE'}")
    return ok and bool(working)


def _parse_sizes(text: str) -> list[tuple[int, int]]:
    sizes = []
    for part in text.split(","):
        w, h = part.lower().split("x")
        sizes.append((int(w), int(h)))
    return sizes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=default_onnx_path(), help="output .onnx (default: models/backbone_resnet18.onnx)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fixed-size", type=int, nargs=2, metavar=("W", "H"), help="export a static graph of this input size")
    ap.add_argument("--verify-sizes", default="448x320,320x224", help="comma-separated WxH (multiples of 16)")
    ap.add_argument("--engines", default="new,auto,classic", help=f"cv2.dnn engines to verify: {','.join(ENGINES)}")
    ap.add_argument("--tol", type=float, default=1e-3, help="max abs diff allowed between cv2.dnn and torch")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)

    torch, torchvision = _import_torch()
    torch.manual_seed(0)
    print(f"torch {torch.__version__}, torchvision {torchvision.__version__}, OpenCV {cv2.__version__}")
    model = build_backbone()
    fixed = tuple(args.fixed_size) if args.fixed_size else None
    t0 = time.perf_counter()
    out = export_onnx(model, args.out, args.opset, fixed)
    data = out.read_bytes()
    print(
        f"exported {out} ({len(data) / 1e6:.2f} MB, opset {args.opset}, "
        f"{'dynamic H/W' if fixed is None else f'fixed {fixed[0]}x{fixed[1]}'}) in {time.perf_counter() - t0:.1f}s"
    )
    print(f"sha256 {hashlib.sha256(data).hexdigest()}")
    if args.no_verify:
        return 0
    sizes = [fixed] if fixed else _parse_sizes(args.verify_sizes)
    for w, h in sizes:
        if w % 16 or h % 16:
            raise SystemExit(f"verify size {w}x{h} is not a multiple of 16")
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    unknown = [e for e in engines if e not in ENGINES]
    if unknown:
        raise SystemExit(f"unknown engines {unknown}; choose from {list(ENGINES)}")
    passed = verify(out, model, sizes, engines, args.tol)
    if not passed and fixed is None:
        print("dynamic-shape verification failed: re-run with --fixed-size W H (e.g. 448 304)")
    print("VERIFY:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
