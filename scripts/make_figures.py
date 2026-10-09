"""Publication figures from ``results/eval_summary.json`` and ``results/eval_items.csv``.

Usage::

    python scripts/make_figures.py [--results results] [--out results/figures]

Writes 1800 x 1200 px PNGs (9 x 6 in at 200 dpi) to ``results/figures/``:

* ``escape_false_reject.png`` - escape / false-reject / review rate per system and regime (95 % CI)
* ``auroc_heatmap.png``       - image AUROC per category x (system, regime) (+ pixel AUROC panel)
* ``risk_coverage.png``       - automation (coverage) vs accuracy curves + each system's operating point
* ``steps_cycle_time.png``    - agent step-count distribution and cycle-time distribution per system
* ``tool_usage.png``          - mean tool calls per inspection (agent systems)
* ``evidence_action.png``     - how often visual evidence changed the agent's next action

Colours: a validated colour-blind-safe categorical palette (adjacent-pair CVD dE >= 9),
one fixed colour per system across every figure; lighter step of the same hue = nominal
regime, full step = harsh regime. Every bar carries a visible value label.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIGSIZE = (9.0, 6.0)
DPI = 200  # -> 1800 x 1200 px

# validated categorical slots (light surface), assigned to systems in fixed order
SYSTEM_ORDER = ("agent_rule", "single_fast", "single_deep", "oracle_fast", "agent_bedrock")
SYSTEM_COLOR = {
    "agent_rule": "#2a78d6",
    "single_fast": "#eb6834",
    "single_deep": "#1baf7a",
    "oracle_fast": "#eda100",
    "agent_bedrock": "#e87ba4",
}
SYSTEM_LABEL = {
    "agent_rule": "Agent (rule planner)",
    "single_fast": "Single shot - fast",
    "single_deep": "Single shot - deep",
    "oracle_fast": "Oracle capture - fast",
    "agent_bedrock": "Agent (Bedrock planner)",
}
REGIME_LABEL = {"nominal": "Nominal conditions", "harsh": "Harsh conditions"}
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "figure.titleweight": "bold",
        }
    )


def _tint(hex_color: str, amount: float = 0.45) -> str:
    """Lighter step of a colour (mix with white) - used for the nominal regime."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(round(c + (255 - c) * amount)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _num(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def _systems(summary: dict) -> list[str]:
    present = set(summary.get("systems") or [])
    return [s for s in SYSTEM_ORDER if s in present] + sorted(present - set(SYSTEM_ORDER))


def _color(system: str) -> str:
    return SYSTEM_COLOR.get(system, MUTED)


def _new_fig(title: str, subtitle: str | None = None):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, layout="constrained")
    # reserve a header band so the title/subtitle never collide with panel titles
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.905 if subtitle else 0.94))
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.012, 0.938, subtitle, ha="left", va="top", color=INK2, fontsize=9)
    return fig


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)  # no bbox_inches: keep exactly 1800 x 1200 px
    plt.close(fig)
    return path


def _no_data(ax, msg: str = "no data") -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=MUTED, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


# --------------------------------------------------------------------------- figures


def fig_escape_false_reject(summary: dict, out: Path) -> Path:
    overall = summary.get("overall") or {}
    regimes = [r for r in summary.get("regimes", []) if r in overall]
    systems = _systems(summary)
    fig = _new_fig(
        "Escapes, false rejects and human reviews",
        "All eval parts pooled over categories; error bars = bootstrap 95 % CI; lower is better",
    )
    gs = fig.add_gridspec(1, 3)
    metrics = (("escape_rate", "Escape rate\n(defective -> PASS)"), ("false_reject_rate", "False-reject rate\n(normal -> FAIL)"), ("review_rate", "Review rate\n(sent to a person)"))
    handles = {}
    for j, (key, title) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, j])
        ax.set_title(title, fontsize=10)
        if not regimes:
            _no_data(ax)
            continue
        n_s = max(1, len(systems))
        width = 0.8 / n_s
        x = np.arange(len(regimes))
        ymax = 0.0
        for i, s in enumerate(systems):
            vals, lo, hi = [], [], []
            for r in regimes:
                m = overall.get(r, {}).get(s) or {}
                v = _num(m.get(key))
                ci = (m.get("ci95") or {}).get(key) or [None, None]
                vals.append(v)
                lo.append(v - _num(ci[0]) if math.isfinite(_num(ci[0])) and math.isfinite(v) else 0.0)
                hi.append(_num(ci[1]) - v if math.isfinite(_num(ci[1])) and math.isfinite(v) else 0.0)
            pos = x - 0.4 + width * (i + 0.5)
            vv = np.array([v if math.isfinite(v) else 0.0 for v in vals])
            bars = ax.bar(pos, vv, width=width * 0.92, color=_color(s), edgecolor="white", linewidth=1.0, label=SYSTEM_LABEL.get(s, s))
            ax.errorbar(pos, vv, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
            handles[s] = bars
            for p, v, h in zip(pos, vals, hi):
                if math.isfinite(v):
                    ax.text(p, v + max(h, 0) + 0.006, f"{100 * v:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
                    ymax = max(ymax, v + max(h, 0))
        ax.set_xticks(x, [r.capitalize() for r in regimes])
        ax.set_ylim(0, max(0.05, ymax * 1.18))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="x", visible=False)
    if handles:
        fig.legend([handles[s] for s in systems if s in handles], [SYSTEM_LABEL.get(s, s) for s in systems if s in handles],
                   loc="outside lower center", ncol=min(5, len(handles)))
    return _save(fig, out / "escape_false_reject.png")


def fig_auroc_heatmap(summary: dict, out: Path) -> Path:
    results = summary.get("results") or {}
    systems = _systems(summary)
    regimes = summary.get("regimes") or []
    cats = [c for c in (summary.get("categories") or []) if c in results] + (["ALL"] if "ALL" in results else [])
    cols = [(s, r) for s in systems for r in regimes]
    pixel = {k: v for k, v in (summary.get("pixel_auroc") or {}).items() if not str(k).startswith("_")}
    has_pixel = any(isinstance(v, dict) and v for v in pixel.values())
    fig = _new_fig("Image-level AUROC by category", "Score = calibrated P(anomalous) (agents: final fused probability); eval split")
    gs = fig.add_gridspec(1, 2 if has_pixel else 1, width_ratios=[4.2, 1.0] if has_pixel else [1])
    ax = fig.add_subplot(gs[0, 0])
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    if not cats or not cols:
        _no_data(ax)
    else:
        mat = np.full((len(cats), len(cols)), np.nan)
        for i, c in enumerate(cats):
            for j, (s, r) in enumerate(cols):
                mat[i, j] = _num(((results.get(c) or {}).get(r) or {}).get(s, {}).get("auroc"))
        _heat(ax, mat, cats, [f"{SYSTEM_LABEL.get(s, s).replace(' - ', chr(10))}\n{r}" for s, r in cols], cmap, 0.5, 1.0)
        ax.set_title("Image AUROC (0.5 = chance, 1.0 = perfect)", fontsize=10)
    if has_pixel:
        ax2 = fig.add_subplot(gs[0, 1])
        pc = [c for c in [*(summary.get("categories") or []), "ALL"] if c in pixel]
        dets = ["fast", "deep"]
        pm = np.array([[_num((pixel.get(c) or {}).get(d, {}).get("pooled")) for d in dets] for c in pc]) if pc else np.zeros((0, 2))
        _heat(ax2, pm, pc, ["fast", "deep"], cmap, 0.5, 1.0)
        ax2.set_title("Pixel AUROC\n(oracle capture)", fontsize=10)
    return _save(fig, out / "auroc_heatmap.png")


def _heat(ax, mat: np.ndarray, rows: list[str], cols: list[str], cmap, vmin: float, vmax: float) -> None:
    ax.grid(False)
    masked = np.ma.masked_invalid(mat)
    ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if math.isfinite(v):
                frac = (min(max(v, vmin), vmax) - vmin) / (vmax - vmin)
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="white" if frac > 0.55 else INK)
            else:
                ax.text(j, i, "-", ha="center", va="center", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(cols)), cols, fontsize=7)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)


def fig_risk_coverage(summary: dict, out: Path) -> Path:
    curves = summary.get("risk_coverage") or {}
    overall = summary.get("overall") or {}
    regimes = [r for r in summary.get("regimes", []) if r in curves]
    systems = _systems(summary)
    fig = _new_fig(
        "Automation vs accuracy (risk-coverage)",
        "Lines: auto-decide the most confident parts first (decision p >= 0.5); dots: each system's actual operating point",
    )
    gs = fig.add_gridspec(1, max(1, len(regimes)))
    handles = {}
    for k, regime in enumerate(regimes or ["-"]):
        ax = fig.add_subplot(gs[0, k])
        if regime not in curves:
            _no_data(ax)
            continue
        ax.set_title(REGIME_LABEL.get(regime, regime), fontsize=10)
        for s in systems:
            c = curves[regime].get(s)
            if not c:
                continue
            xs = np.array([_num(v) for v in c["coverage"]])
            ys = np.array([_num(v) for v in c["accuracy"]])
            ok = np.isfinite(xs) & np.isfinite(ys)
            (ln,) = ax.plot(xs[ok], ys[ok], color=_color(s), linewidth=2.0, label=SYSTEM_LABEL.get(s, s))
            handles[s] = ln
            m = (overall.get(regime) or {}).get(s) or {}
            ox, oy = _num(m.get("automation_rate")), _num(m.get("accuracy_auto"))
            if math.isfinite(ox) and math.isfinite(oy):
                ax.scatter([ox], [oy], s=46, color=_color(s), edgecolor="white", linewidth=1.6, zorder=5)
        ax.set_xlim(0, 1.02)
        ymin = min([y for ln in ax.get_lines() for y in ln.get_ydata() if math.isfinite(y)] or [0.5])
        ax.set_ylim(max(0.0, math.floor((min(ymin, 0.9) - 0.03) * 20) / 20), 1.005)
        ax.set_xlabel("Automation rate (fraction of parts decided without a person)")
        if k == 0:
            ax.set_ylabel("Accuracy on automatically decided parts")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    if handles:
        fig.legend([handles[s] for s in systems if s in handles], [SYSTEM_LABEL.get(s, s) for s in systems if s in handles],
                   loc="outside lower center", ncol=min(5, len(handles)))
    return _save(fig, out / "risk_coverage.png")


def fig_steps_cycle(rows: list[dict], summary: dict, out: Path) -> Path:
    systems = _systems(summary)
    regimes = summary.get("regimes") or sorted({r["regime"] for r in rows})
    ok = [r for r in rows if not r.get("error")]
    fig = _new_fig("Inspection effort", "Left: tool calls (steps) per agent inspection; right: simulated cycle time = compute + camera motion/AF/exposure")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25])
    ax = fig.add_subplot(gs[0, 0])
    agents = [s for s in systems if s.startswith("agent")]
    plotted = False
    if agents:
        series = []
        for s in agents:
            for r in regimes:
                vals = [int(_num(x["n_steps"])) for x in ok if x["system"] == s and x["regime"] == r and math.isfinite(_num(x.get("n_steps")))]
                if vals:
                    series.append((s, r, vals))
        if series:
            lo = min(min(v) for *_, v in series)
            hi = max(max(v) for *_, v in series)
            bins = np.arange(lo, hi + 1)
            width = 0.8 / len(series)
            for i, (s, r, vals) in enumerate(series):
                counts = np.array([np.mean(np.array(vals) == b) for b in bins])
                col = _tint(_color(s)) if r == "nominal" else _color(s)
                ax.bar(bins - 0.4 + width * (i + 0.5), counts, width=width * 0.92, color=col, edgecolor="white", linewidth=0.8,
                       label=f"{SYSTEM_LABEL.get(s, s)} - {r} (median {np.median(vals):.0f})")
            ax.set_xticks(bins)
            top = max(float(np.mean(np.array(v) == b)) for *_, v in series for b in bins)
            ax.set_ylim(0, top * (1.25 + 0.07 * len(series)))  # headroom for the legend
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
            ax.set_xlabel("Steps per inspection")
            ax.set_ylabel("Share of inspections")
            ax.legend(loc="upper right", fontsize=7.5)
            ax.grid(axis="x", visible=False)
            plotted = True
    if not plotted:
        _no_data(ax, "no agent runs")
    ax.set_title("Agent steps per part", fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    data, pos, cols, labels = [], [], [], []
    p = 0.0
    for s in systems:
        for r in regimes:
            vals = [_num(x.get("cycle_time_ms")) / 1000.0 for x in ok if x["system"] == s and x["regime"] == r]
            vals = [v for v in vals if math.isfinite(v)]
            if vals:
                data.append(vals)
                pos.append(p)
                cols.append(_tint(_color(s)) if r == "nominal" else _color(s))
                labels.append(r)
            p += 1.0
        p += 0.6
    if data:
        bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True, showfliers=True, medianprops={"color": INK, "linewidth": 1.2},
                        whiskerprops={"color": INK2, "linewidth": 0.8}, capprops={"color": INK2, "linewidth": 0.8},
                        flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": MUTED, "markeredgecolor": "none", "alpha": 0.6})
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_edgecolor("white")
        ax.set_xticks(pos, labels, fontsize=7.5)
        centers, names = [], []
        p = 0.0
        for s in systems:
            centers.append(p + (len(regimes) - 1) / 2)
            names.append(SYSTEM_LABEL.get(s, s).replace(" - ", "\n").replace(" (", "\n("))
            p += len(regimes) + 0.6
        sec = ax.secondary_xaxis("bottom")
        sec.set_xticks(centers, names, fontsize=7.5)
        sec.tick_params(pad=16, length=0)
        sec.spines["bottom"].set_visible(False)
        ax.set_ylabel("Cycle time (s)")
        ax.grid(axis="x", visible=False)
    else:
        _no_data(ax)
    ax.set_title("Cycle time per part (lighter = nominal, darker = harsh)", fontsize=10)
    return _save(fig, out / "steps_cycle_time.png")


def fig_tool_usage(summary: dict, out: Path) -> Path:
    overall = summary.get("overall") or {}
    regimes = summary.get("regimes") or []
    agents = [s for s in _systems(summary) if s.startswith("agent")]
    fig = _new_fig("Tool usage per inspection", "Mean number of calls of each tool per part (lighter = nominal, darker = harsh)")
    gs = fig.add_gridspec(1, max(1, len(agents)))
    if not agents:
        _no_data(fig.add_subplot(gs[0, 0]), "no agent runs")
        return _save(fig, out / "tool_usage.png")
    for k, s in enumerate(agents):
        ax = fig.add_subplot(gs[0, k])
        tools: list[str] = []
        per: dict[str, dict] = {}
        for r in regimes:
            tu = ((overall.get(r) or {}).get(s) or {}).get("tool_usage") or {}
            per[r] = tu.get("mean_per_run") or {}
            tools += [t for t in per[r] if t not in tools]
        order = ["capture_image", "assess_quality", "detect_anomalies", "inspect_region", "measure_defect", "request_human_review", "finish"]
        tools = [t for t in order if t in tools] + sorted(set(tools) - set(order))
        if not tools:
            _no_data(ax)
            continue
        y = np.arange(len(tools))
        h = 0.8 / max(1, len(regimes))
        xmax = 0.0
        for i, r in enumerate(regimes):
            vals = np.array([_num(per[r].get(t, 0.0)) for t in tools])
            vals = np.where(np.isfinite(vals), vals, 0.0)
            col = _tint(_color(s)) if r == "nominal" else _color(s)
            yy = y - 0.4 + h * (i + 0.5)
            ax.barh(yy, vals, height=h * 0.9, color=col, edgecolor="white", linewidth=0.8, label=r)
            for v, yv in zip(vals, yy):
                ax.text(v + 0.03, yv, f"{v:.2f}", va="center", fontsize=7, color=INK2)
            xmax = max(xmax, float(vals.max()) if vals.size else 0.0)
        ax.set_yticks(y, tools)
        ax.invert_yaxis()
        ax.set_xlim(0, max(1.0, xmax * 1.2))
        ax.set_xlabel("Calls per inspection")
        ax.set_title(SYSTEM_LABEL.get(s, s), fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right")
    return _save(fig, out / "tool_usage.png")


def fig_evidence_action(summary: dict, out: Path) -> Path:
    overall = summary.get("overall") or {}
    results = summary.get("results") or {}
    regimes = summary.get("regimes") or []
    agents = [s for s in _systems(summary) if s.startswith("agent")]
    keys = (
        ("quality_recapture", "Quality issue ->\ncorrective recapture"),
        ("zoom", "Candidate ->\nzoom re-inspection"),
        ("zoom_confirmed", "Zoom\nconfirmed"),
        ("zoom_refuted", "Zoom\nrefuted"),
        ("zoom_inconclusive", "Zoom\ninconclusive"),
        ("deep", "Escalation to\ndeep detector"),
        ("review_visual", "Evidence-driven\nhuman review"),
        ("any", "Any evidence-\ndriven change"),
    )
    fig = _new_fig("Visual evidence changed the next action", "Share of agent inspections in which a quality or detection result triggered each action")
    n_panels = max(1, len(agents))
    gs = fig.add_gridspec(2 if agents else 1, n_panels, height_ratios=[1.0, 0.85] if agents else [1])
    if not agents:
        _no_data(fig.add_subplot(gs[0, 0]), "no agent runs")
        return _save(fig, out / "evidence_action.png")
    for k, s in enumerate(agents):
        ax = fig.add_subplot(gs[0, k])
        x = np.arange(len(keys))
        w = 0.8 / max(1, len(regimes))
        for i, r in enumerate(regimes):
            rates = ((overall.get(r) or {}).get(s) or {}).get("evidence_rates") or {}
            vals = np.array([_num(rates.get(key)) for key, _ in keys])
            vals = np.where(np.isfinite(vals), vals, 0.0)
            col = _tint(_color(s)) if r == "nominal" else _color(s)
            xx = x - 0.4 + w * (i + 0.5)
            ax.bar(xx, vals, width=w * 0.92, color=col, edgecolor="white", linewidth=0.8, label=r)
            for v, xv in zip(vals, xx):
                ax.text(xv, v + 0.01, f"{100 * v:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.set_xticks(x, [lab for _, lab in keys], fontsize=7)
        ax.set_ylim(0, 1.3)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(SYSTEM_LABEL.get(s, s), fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", ncol=len(regimes))
        # per-category "any" rate
        ax2 = fig.add_subplot(gs[1, k])
        cats = [c for c in (summary.get("categories") or []) if c in results]
        xc = np.arange(len(cats))
        for i, r in enumerate(regimes):
            vals = np.array([_num((((results.get(c) or {}).get(r) or {}).get(s) or {}).get("evidence_changed_action_rate")) for c in cats])
            vals = np.where(np.isfinite(vals), vals, 0.0)
            col = _tint(_color(s)) if r == "nominal" else _color(s)
            xx = xc - 0.4 + w * (i + 0.5)
            ax2.bar(xx, vals, width=w * 0.92, color=col, edgecolor="white", linewidth=0.8)
            for v, xv in zip(vals, xx):
                ax2.text(xv, v + 0.01, f"{100 * v:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax2.set_xticks(xc, cats, fontsize=8)
        ax2.set_ylim(0, 1.15)
        ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax2.set_title("Any evidence-driven change, by category", fontsize=9)
        ax2.grid(axis="x", visible=False)
    return _save(fig, out / "evidence_action.png")


# --------------------------------------------------------------------------- driver


def make_all(items_csv: Path, summary_json: Path, out_dir: Path) -> list[Path]:
    """Render every figure; returns the written paths."""
    from scripts.evaluate import read_rows

    _style()
    summary = json.loads(Path(summary_json).read_text(encoding="utf-8"))
    rows = read_rows(Path(items_csv)) if Path(items_csv).is_file() else []
    out_dir = Path(out_dir)
    return [
        fig_escape_false_reject(summary, out_dir),
        fig_auroc_heatmap(summary, out_dir),
        fig_risk_coverage(summary, out_dir),
        fig_steps_cycle(rows, summary, out_dir),
        fig_tool_usage(summary, out_dir),
        fig_evidence_action(summary, out_dir),
    ]


def main(argv: list[str] | None = None) -> int:
    from inspectagent.config import get_settings

    res = get_settings().results_dir
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=res)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    summary = args.results / "eval_summary.json"
    if not summary.is_file():
        raise SystemExit(f"{summary} not found - run scripts/evaluate.py first")
    written = make_all(args.results / "eval_items.csv", summary, args.out or args.results / "figures")
    for p in written:
        print(f"[figures] {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
