"""Synthetic PCB-like scenes with injectable defects (offline test fixtures).

``make_board(seed)`` draws one *normal* instance of a fixed board design (layout is the
same for every seed, only small manufacturing/photometric variation changes), so a
set of normal boards can train the statistical template exactly like VisA normals.
``inject_defect(img, kind, seed)`` returns a defective copy plus its ground-truth mask.
"""

from __future__ import annotations

import cv2
import numpy as np

BOARD_SIZE = (900, 600)  # (W, H), 3:2 like VisA PCBs
DEFECT_KINDS = ("scratch", "missing", "spot", "solder")


def _layout(rng: np.random.Generator):
    w, h = BOARD_SIZE
    traces = []
    for _ in range(26):
        x1, y1 = int(rng.integers(40, w - 40)), int(rng.integers(40, h - 40))
        if rng.random() < 0.5:
            x2, y2 = int(rng.integers(40, w - 40)), y1
        else:
            x2, y2 = x1, int(rng.integers(40, h - 40))
        traces.append((x1, y1, x2, y2, int(rng.integers(3, 7))))
    pads = [(int(rng.integers(50, w - 50)), int(rng.integers(50, h - 50)), int(rng.integers(6, 12))) for _ in range(40)]
    chips = []
    for _ in range(6):
        cw, ch = int(rng.integers(60, 130)), int(rng.integers(40, 90))
        chips.append((int(rng.integers(40, w - cw - 40)), int(rng.integers(40, h - ch - 40)), cw, ch))
    holes = [(int(rng.integers(30, w - 30)), int(rng.integers(30, h - 30))) for _ in range(12)]
    return traces, pads, chips, holes


def make_board(seed: int = 0, design_seed: int = 7) -> np.ndarray:
    """One normal board instance (BGR uint8, BOARD_SIZE)."""
    design = np.random.default_rng(design_seed)
    traces, pads, chips, holes = _layout(design)
    rng = np.random.default_rng(10_000 + seed)
    w, h = BOARD_SIZE
    base = np.array([38, 92, 30], np.float32) + rng.normal(0, 3, 3).astype(np.float32)  # solder-mask green (BGR)
    canvas = np.empty((h, w, 3), np.float32)
    canvas[:] = base
    tex = cv2.GaussianBlur(rng.normal(0, 6, (h, w)).astype(np.float32), (0, 0), 3)
    canvas += tex[..., None]
    # OpenCV 5 text rendering requires 8-bit images, so draw on uint8.
    img = np.clip(canvas, 0, 255).astype(np.uint8)
    copper = (60, 150, 200)
    for x1, y1, x2, y2, t in traces:
        j = rng.integers(-1, 2, 2)
        cv2.line(img, (x1 + int(j[0]), y1 + int(j[1])), (x2 + int(j[0]), y2 + int(j[1])), copper, t, cv2.LINE_AA)
    for x, y, r in pads:
        cv2.circle(img, (x, y), r, (170, 190, 205), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), max(2, r // 3), (40, 40, 40), -1, cv2.LINE_AA)
    for x, y, cw, ch in chips:
        cv2.rectangle(img, (x, y), (x + cw, y + ch), (28, 28, 30), -1)
        for px in range(x + 6, x + cw - 4, 10):
            cv2.rectangle(img, (px, y - 6), (px + 4, y), (190, 195, 200), -1)
            cv2.rectangle(img, (px, y + ch), (px + 4, y + ch + 6), (190, 195, 200), -1)
        cv2.putText(img, "IC", (x + 8, y + ch // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    for x, y in holes:
        cv2.circle(img, (x, y), 7, (15, 15, 15), -1, cv2.LINE_AA)
    cv2.putText(img, "INSPECT-1", (30, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)
    gain = 1.0 + rng.normal(0, 0.02)
    out = img.astype(np.float32) * gain + rng.normal(0, 1.5, img.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def inject_defect(img: np.ndarray, kind: str = "scratch", seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (defective image, mask uint8 0/255)."""
    rng = np.random.default_rng(20_000 + seed)
    out = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cx, cy = int(rng.integers(int(0.2 * w), int(0.8 * w))), int(rng.integers(int(0.2 * h), int(0.8 * h)))
    if kind == "scratch":
        ang = rng.uniform(0, np.pi)
        L = int(rng.integers(50, 110))
        p1 = (int(cx - L / 2 * np.cos(ang)), int(cy - L / 2 * np.sin(ang)))
        p2 = (int(cx + L / 2 * np.cos(ang)), int(cy + L / 2 * np.sin(ang)))
        cv2.line(out, p1, p2, (215, 225, 230), 3, cv2.LINE_AA)
        cv2.line(mask, p1, p2, 255, 5)
    elif kind == "missing":
        cw, ch = 70, 50
        x0, y0 = max(0, cx - cw // 2), max(0, cy - ch // 2)
        patch = out[y0 : y0 + ch, x0 : x0 + cw]
        patch[:] = (38, 92, 30)
        mask[y0 : y0 + ch, x0 : x0 + cw] = 255
    elif kind == "spot":
        r = int(rng.integers(9, 16))
        cv2.circle(out, (cx, cy), r, (20, 25, 35), -1, cv2.LINE_AA)
        cv2.circle(mask, (cx, cy), r + 1, 255, -1)
    elif kind == "solder":
        axes = (int(rng.integers(10, 18)), int(rng.integers(6, 11)))
        cv2.ellipse(out, (cx, cy), axes, float(rng.uniform(0, 180)), 0, 360, (200, 205, 210), -1, cv2.LINE_AA)
        cv2.ellipse(mask, (cx, cy), (axes[0] + 1, axes[1] + 1), 0, 0, 360, 255, -1)
    else:
        raise ValueError(f"unknown defect kind {kind!r}")
    return out, mask


def make_dataset(n_normal: int = 24, n_defect: int = 8, seed: int = 0):
    """(normals, [(defective, mask, kind)]) for quick model fitting in tests."""
    normals = [make_board(seed * 1000 + i) for i in range(n_normal)]
    defects = []
    for j in range(n_defect):
        kind = DEFECT_KINDS[j % len(DEFECT_KINDS)]
        img, mask = inject_defect(make_board(seed * 1000 + 500 + j), kind, seed=j)
        defects.append((img, mask, kind))
    return normals, defects
