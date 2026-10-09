"""Render the architecture and agent-workflow diagrams (SVG + PNG).

    python scripts/make_diagrams.py            # writes docs/architecture.{svg,png}, docs/agent_workflow.{svg,png}

SVGs are hand-laid-out (no graphviz dependency). PNGs are rasterised with Playwright using the
locally installed Chrome (``channel="chrome"``) or Playwright's Chromium if available.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

INK = "#1d2433"
MUTED = "#5b6475"
LINE = "#8a93a6"
AWS = "#ff9900"
AWS_BG = "#fff8ee"
CV = "#2f6fdb"
CV_BG = "#eef4ff"
AGENT = "#7a3fd1"
AGENT_BG = "#f5efff"
HUMAN = "#0f8a5f"
HUMAN_BG = "#ebf8f2"
WARN = "#c46a00"
FONT = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', Consolas, monospace"


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def right(self, dy: float = 0) -> tuple[float, float]:
        return self.x + self.w, self.cy + dy

    def left(self, dy: float = 0) -> tuple[float, float]:
        return self.x, self.cy + dy

    def top(self, dx: float = 0) -> tuple[float, float]:
        return self.cx + dx, self.y

    def bottom(self, dx: float = 0) -> tuple[float, float]:
        return self.cx + dx, self.y + self.h


class Svg:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.parts: list[str] = []

    def rect(self, b: Box, fill="#fff", stroke=LINE, sw=1.5, rx=10, dash: str | None = None) -> Box:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<rect x="{b.x}" y="{b.y}" width="{b.w}" height="{b.h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
        )
        return b

    def text(self, x, y, s, size=14, weight=400, color=INK, anchor="start", family=FONT, italic=False) -> None:
        st = ' font-style="italic"' if italic else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{color}" '
            f'text-anchor="{anchor}"{st}>{escape(s)}</text>'
        )

    def lines(self, x, y, rows: list[str], size=12.5, color=MUTED, gap=17, family=FONT, anchor="start") -> None:
        for i, r in enumerate(rows):
            self.text(x, y + i * gap, r, size=size, color=color, family=family, anchor=anchor)

    def card(self, b: Box, title: str, rows: list[str], accent: str, bg: str, mono_rows=False, dash=None, title_size=15) -> Box:
        self.rect(b, fill=bg, stroke=accent, sw=1.6, dash=dash)
        self.parts.append(f'<rect x="{b.x}" y="{b.y}" width="6" height="{b.h}" rx="3" fill="{accent}"/>')
        self.text(b.x + 18, b.y + 26, title, size=title_size, weight=650)
        self.lines(b.x + 18, b.y + 48, rows, family=MONO if mono_rows else FONT, size=12 if mono_rows else 12.5)
        return b

    def arrow(self, p1, p2, color=LINE, label: str | None = None, dash=None, sw=1.8, label_dy=-7, via: list | None = None) -> None:
        pts = [p1] + (via or []) + [p2]
        d = "M " + " L ".join(f"{x} {y}" for x, y in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        mid = color.replace("#", "")
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{da} marker-end="url(#ah_{mid})"/>')
        if label:
            (ax, ay), (bx, by) = pts[len(pts) // 2 - 1], pts[len(pts) // 2]
            mx, my = (ax + bx) / 2, (ay + by) / 2
            tw = 6.6 * len(label) + 10
            self.parts.append(
                f'<rect x="{mx - tw / 2}" y="{my + label_dy - 13}" width="{tw}" height="18" rx="4" fill="#ffffff" fill-opacity="0.92"/>'
            )
            self.text(mx, my + label_dy, label, size=11.5, color=color if color != LINE else MUTED, anchor="middle")

    def render(self) -> str:
        colors = {LINE, AWS, CV, AGENT, HUMAN, WARN, INK}
        markers = "".join(
            f'<marker id="ah_{c.replace("#", "")}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{c}"/></marker>'
            for c in colors
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
            f"<defs>{markers}</defs>"
            f'<rect width="100%" height="100%" fill="#ffffff"/>' + "".join(self.parts) + "</svg>"
        )


def architecture() -> str:
    s = Svg(1800, 1200)
    s.text(60, 70, "InspectAgent — architecture", size=34, weight=750)
    s.text(60, 104, "Agentic visual inspection with OpenCV 5 on AWS Graviton (arm64). Solid = live path, dashed = build / optional paths.", size=16, color=MUTED)

    # Clients
    browser = s.card(Box(60, 190, 250, 110), "Operator / judge", ["Web UI (browser)", "inspect · review · results"], HUMAN, HUMAN_BG)
    reviewer = s.card(Box(60, 340, 250, 110), "Human reviewer", ["approve scrap · override", "· request rescan"], HUMAN, HUMAN_BG)
    mcp = s.card(Box(60, 560, 250, 130), "MCP client", ["Claude Code · Kiro · any", "MCP-compatible agent", "(stdio / streamable HTTP)"], AGENT, AGENT_BG)

    # AWS cloud
    s.rect(Box(370, 150, 1380, 990), fill=AWS_BG, stroke=AWS, sw=2, rx=18)
    s.text(395, 185, "AWS Cloud  ·  us-east-1", size=17, weight=700, color=WARN)

    furl = s.card(Box(400, 215, 230, 95), "Lambda Function URL", ["HTTPS endpoint (public demo)", "CORS · reserved concurrency"], AWS, "#ffffff")

    # Lambda container
    lam = s.rect(Box(670, 205, 720, 690), fill="#ffffff", stroke=AWS, sw=2, rx=14)
    s.text(690, 238, "AWS Lambda  ·  arm64 (Graviton)  ·  container image  ·  Python 3.13", size=16, weight=700)
    s.text(690, 260, "FastAPI + Mangum  ·  models & samples baked into the image  ·  /tmp scratch only", size=12.5, color=MUTED)

    api = s.card(Box(690, 280, 330, 120), "Web API", ["/api/inspect   /api/reviews", "/api/traces    /api/metrics", "/api/system    /api/bench"], CV, CV_BG, mono_rows=True)
    pol = s.card(Box(1040, 280, 330, 120), "Human-control policy", ["budget: steps · captures · time", "auto-PASS ≤ 0.15 · auto-FAIL ≥ 0.90", "FAIL → quarantine; scrap needs human"], HUMAN, HUMAN_BG)
    planner = s.card(Box(690, 420, 330, 115), "Planner", ["RulePlanner (deterministic)", "BedrockPlanner (Converse tool use)", "→ fallback to rules on error"], AGENT, AGENT_BG)
    reg = s.card(Box(1040, 420, 330, 115), "Tool registry (JSON Schema)", ["capture_image · assess_quality", "detect_anomalies · inspect_region", "measure_defect · finish · review"], AGENT, AGENT_BG, mono_rows=False)

    cvbox = s.rect(Box(690, 555, 680, 320), fill=CV_BG, stroke=CV, sw=1.8, rx=12)
    s.text(708, 585, "OpenCV 5 perception tools", size=16, weight=700, color=CV)
    tools = [
        ("Active camera", ["view · zoom · exposure", "light · autofocus", "(simulator over VisA)"]),
        ("Quality", ["Laplacian sharpness", "Immerkær noise", "glare blobs (HSV)"]),
        ("Registration", ["ORB + RANSAC affine", "ECC refine", "phaseCorrelate"]),
        ("Fast detector", ["Lab statistical template", "noise-aware z-map", "connected components"]),
        ("Deep detector", ["cv2.dnn ResNet18 ONNX", "PatchCore coreset kNN", "alignment-free"]),
        ("Measure", ["minAreaRect geometry", "mm via px/mm", "defect type guess"]),
    ]
    for i, (t, rows) in enumerate(tools):
        col, row = i % 3, i // 3
        b = Box(708 + col * 220, 602 + row * 132, 205, 118)
        s.rect(b, fill="#ffffff", stroke=CV, sw=1.2, rx=9)
        s.text(b.x + 14, b.y + 24, t, size=14, weight=650)
        s.lines(b.x + 14, b.y + 46, rows, size=12)

    # AWS services column
    bed = s.card(Box(1440, 215, 280, 120), "Amazon Bedrock", ["Claude / Nova via Converse", "tool use · image blocks", "(optional LLM planner)"], AWS, "#ffffff")
    s3 = s.card(Box(1440, 365, 280, 105), "Amazon S3", ["traces/*.json", "artifacts/*.jpg (30-day lifecycle)"], AWS, "#ffffff")
    ddb = s.card(Box(1440, 500, 280, 105), "Amazon DynamoDB", ["review queue · audit log", "single table (pk/sk), on-demand"], AWS, "#ffffff")
    cw = s.card(Box(1440, 635, 280, 105), "Amazon CloudWatch Logs", ["one JSON line per agent step", "(observability / replay)"], AWS, "#ffffff")

    # Build / deploy row
    cfn = s.card(Box(400, 930, 250, 110), "AWS CloudFormation", ["one stack: S3 · DDB · ECR", "CodeBuild · IAM · Lambda"], AWS, "#ffffff", dash="6 5")
    cb = s.card(Box(690, 930, 290, 110), "AWS CodeBuild (ARM)", ["native arm64 docker build", "source zip from S3"], AWS, "#ffffff", dash="6 5")
    ecr = s.card(Box(1020, 930, 250, 110), "Amazon ECR", ["arm64 image", "scan on push"], AWS, "#ffffff", dash="6 5")
    cool = s.card(Box(1310, 920, 410, 130), "EC2 Graviton4 + COOL (optional)", ["Cloud-Optimized OpenCV Library AMI", "benchmark & batch evaluation path:", "COOL vs stock OpenCV 5 vs x86"], CV, "#ffffff", dash="6 5")

    mcpsrv = s.card(Box(60, 760, 250, 140), "MCP server", ["python -m inspectagent", ".mcp_server — same tools,", "session-scoped, local or", "container"], AGENT, AGENT_BG)

    # arrows
    s.arrow(browser.right(), furl.left(-10), color=HUMAN, label="HTTPS")
    s.arrow(furl.right(), api.left(-20), color=AWS)
    s.arrow(reviewer.right(), furl.left(25), color=HUMAN, via=[(345, 395), (345, 287)], label="")
    s.arrow(mcp.bottom(), mcpsrv.top(), color=AGENT, label="stdio")
    s.arrow(mcpsrv.right(), (690, 830), color=AGENT, label="same tool registry", dash="6 5", via=[(500, 830)])
    s.arrow(api.bottom(), planner.top(), color=AGENT)
    s.arrow(planner.right(), reg.left(), color=AGENT, label="tool call", label_dy=-50)
    s.arrow(pol.bottom(), reg.top(), color=HUMAN, label="vet / gate")
    s.arrow(reg.bottom(), (1205, 555), color=CV, label="invoke")
    s.arrow((1390, 275), bed.left(), color=AWS, label="Converse")
    s.arrow((1390, 417), s3.left(), color=AWS)
    s.arrow((1390, 552), ddb.left(), color=AWS)
    s.arrow((1390, 687), cw.left(), color=AWS)
    s.arrow(cfn.right(), cb.left(), color=AWS, dash="6 5")
    s.arrow(cb.right(), ecr.left(), color=AWS, dash="6 5", label="push")
    s.arrow(ecr.top(), (1145, 895), color=AWS, dash="6 5", label="image")

    s.text(60, 1180, "VisA dataset © Amazon, CC BY 4.0. Camera = deterministic active-camera simulator over real VisA images (replaceable by a real camera driver).", size=13, color=MUTED)
    return s.render()


def agent_workflow() -> str:
    s = Svg(1800, 1200)
    s.text(60, 70, "InspectAgent — perception → decision → action loop", size=34, weight=750)
    s.text(60, 104, "Every arrow out of 'Decide' is triggered by an OpenCV 5 measurement; each step is logged with the evidence ids and thresholds that caused it.", size=16, color=MUTED)

    cap = s.card(Box(70, 200, 330, 150), "1 · Act: capture_image", ["view (cx, cy, zoom)", "exposure EV · light mode", "autofocus on/off", "→ capture cap_n"], CV, CV_BG)
    per = s.card(Box(520, 170, 400, 230), "2 · Perceive (OpenCV 5)", ["assess_quality: sharpness ratio,", "  luma / clipping, noise σ, glare %", "detect_anomalies(fast): register →", "  photometric norm → z-map → regions", "detect_anomalies(deep): cv2.dnn", "  ResNet18 features → kNN distance", "inspect_region: zoomed re-capture"], CV, CV_BG)
    dec = s.card(Box(1040, 170, 400, 230), "3 · Decide (planner + policy)", ["fused P(anomalous) with refutation", "quality issues → corrective capture", "ambiguous → zoom top candidates", "registration failed → deep model", "budget / uncertainty → human", "policy vets every decision"], AGENT, AGENT_BG)

    s.arrow(cap.right(), per.left(-10), color=CV, label="image")
    s.arrow(per.right(), dec.left(-10), color=CV, label="measurements + ids")

    actions = [
        ("Recapture", ["glare > 0.4 % → diffuse light, EV +0.75", "blur ratio < 0.55 → autofocus", "dark / bright → EV += suggested Δ"], CV, CV_BG),
        ("Zoom & confirm", ["inspect_region(top candidates, zoom 3)", "confirmed → measure_defect", "refuted → lower fused probability"], CV, CV_BG),
        ("Escalate compute", ["detect_anomalies(deep) when the fast", "detector is unreliable or ambiguous"], AGENT, AGENT_BG),
        ("Finish", ["PASS → route_pass", "FAIL → route_quarantine", "(only inside auto thresholds)"], HUMAN, HUMAN_BG),
        ("Ask a human", ["request_human_review(evidence)", "→ hold_for_review ticket in", "DynamoDB review queue"], HUMAN, HUMAN_BG),
    ]
    boxes = []
    bus_y = 470
    s.parts.append(f'<path d="M {dec.cx} {dec.y + dec.h} L {dec.cx} {bus_y}" stroke="{AGENT}" stroke-width="2.2" fill="none"/>')
    s.parts.append(f'<path d="M {70 + 160} {bus_y} L {70 + 4 * 340 + 160} {bus_y}" stroke="{AGENT}" stroke-width="2.2" fill="none"/>')
    s.text(dec.cx + 12, bus_y - 30, "one tool call per step", size=12.5, color=AGENT, weight=600)
    for i, (t, rows, ac, bg) in enumerate(actions):
        b = s.card(Box(70 + i * 340, 560, 320, 140), t, rows, ac, bg)
        boxes.append(b)
        s.arrow((b.cx, bus_y), b.top(), color=AGENT)
    # loops back to capture / perception
    s.arrow(boxes[0].left(), cap.left(20), color=CV, via=[(40, 630), (40, 295)], label="")
    s.text(48, 520, "loop", size=13, color=CV, weight=600)
    s.arrow(boxes[1].bottom(-100), cap.bottom(-80), color=CV, dash="6 5",
            via=[(boxes[1].cx - 100, 740), (25, 740), (25, 380), (cap.cx - 80, 380)])
    s.text(250, 760, "zoomed re-capture of a candidate region", size=12, color=CV)
    s.arrow(boxes[2].bottom(80), per.bottom(150), color=AGENT, dash="6 5",
            via=[(boxes[2].cx + 80, 760), (1080, 760), (1080, 430), (per.cx + 150, 430)])
    s.text(860, 785, "deep (cv2.dnn) result", size=12, color=AGENT)

    hum = s.card(Box(1040, 820, 560, 150), "Human-in-the-loop (web UI / API)", ["approve_fail → route_scrap", "override_pass → route_pass (audit-logged)", "rescan → new inspection with clean preset, traces linked"], HUMAN, HUMAN_BG)
    s.arrow(boxes[4].bottom(), hum.top(-100), color=HUMAN)
    s.arrow(boxes[3].bottom(), (1420 - 340, 820), color=HUMAN, dash="6 5")

    trace = s.card(Box(70, 820, 860, 150), "Observability: trace of every step", ["{step, tool, args, rationale, evidence ids, observation, latency_ms, sim_time_ms, policy_override}", "→ S3 traces/*.json · CloudWatch JSON lines · shown live in the UI timeline", "proves which OpenCV output changed which later decision"], INK, "#f4f6fa", mono_rows=False)

    s.text(60, 1050, "Guardrails: max 14 steps · 8 captures · 2 deep calls · 5 s simulated cycle time; unknown tools or invalid arguments are rejected by JSON-Schema validation;", size=14, color=MUTED)
    s.text(60, 1074, "a PASS needs reliable evidence and P(anomalous) ≤ 0.15, a FAIL needs ≥ 0.90 — everything else is routed to a person.", size=14, color=MUTED)
    return s.render()


def rasterize(svg_path: Path, png_path: Path, width: int, height: int) -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed; skipping PNG")
        return False
    html = f"<html><body style='margin:0'>{svg_path.read_text(encoding='utf-8')}</body></html>"
    with sync_playwright() as p:
        browser = None
        for kw in ({"channel": "chrome"}, {}):
            try:
                browser = p.chromium.launch(**kw)
                break
            except Exception as exc:  # noqa: BLE001 - try the next launcher
                print(f"launch {kw or 'chromium'} failed: {exc.__class__.__name__}")
        if browser is None:
            return False
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        page.set_content(html)
        page.screenshot(path=str(png_path), clip={"x": 0, "y": 0, "width": width, "height": height})
        browser.close()
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DOCS)
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("architecture", architecture), ("agent_workflow", agent_workflow)):
        svg = args.out / f"{name}.svg"
        svg.write_text(fn(), encoding="utf-8")
        print("wrote", svg)
        if not args.no_png and rasterize(svg, args.out / f"{name}.png", 1800, 1200):
            print("wrote", args.out / f"{name}.png")


if __name__ == "__main__":
    main()
