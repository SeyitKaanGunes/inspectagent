# InspectAgent — Engineering Specification (internal contract)

**One-liner.** An agentic visual-inspection cell: an agent drives an (actively controllable)
camera, uses OpenCV 5 tools to judge image quality and find anomalies, and *changes what it does
next* based on that visual evidence — re-exposing, refocusing, switching light, zooming into
suspicious regions, escalating to a deeper detector, or asking a human for approval. Runs as a
serverless container on AWS Graviton (Lambda arm64) with S3 + DynamoDB + Amazon Bedrock, and is
exposed to any MCP client (Claude Code, Kiro, …) as an MCP server.

Target competition: OpenCV AI Competition 2026 powered by AWS — primary focus path **Agentic Vision
with OpenCV 5** (image results must change a later plan/tool call/action/human-approval request).

This file is the contract between modules. Shared types live in `inspectagent/types.py`, all
tunables in `inspectagent/config.py` (do not duplicate constants elsewhere — import them).

## 0. Environment facts (verified)

* Python **3.13** venv at `.venv` (`.venv\Scripts\python.exe` on Windows). Use `uv pip install --python .venv\Scripts\python.exe ...` to add packages (pip is not installed in the venv).
* `opencv-contrib-python-headless==5.0.0.93` → `cv2.__version__ == "5.0.0"`. Available: `cv2.ORB_create`, `cv2.SIFT_create`,
  `cv2.findHomography`, `cv2.estimateAffinePartial2D`, `cv2.findTransformECC`, `cv2.phaseCorrelate`, `cv2.connectedComponentsWithStats`,
  `cv2.createCLAHE`, `cv2.dnn.readNetFromONNX`, `cv2.dnn.blobFromImageWithParams`, `cv2.dnn.Image2BlobParams`, `cv2.quality`, `cv2.saliency`.
  **Not top-level in 5.0:** `AKAZE_create`, `KAZE_create`, `BRISK_create` (AKAZE/KAZE live under `cv2.xfeatures2d_*`). Use ORB (+SIFT fallback).
  `cv2.dnn` exposes engines `ENGINE_AUTO/ENGINE_CLASSIC/ENGINE_NEW/ENGINE_ORT`.
* numpy 2.5.3, fastapi 0.143.0, uvicorn 0.54.0, mangum 0.22.0, boto3 1.43.110, **mcp 2.3.0** (API may differ from 1.x — introspect the installed package before using it), pydantic 2.14.0, pytest 9.1.1, httpx 0.28.1, matplotlib 3.11.2, playwright 1.63.0, imageio-ffmpeg 0.6.0, onnx 1.23.2.
* Windows 11 dev box (i7-13700H, 32 GB, RTX 4050). Deployment target is Linux arm64 (AWS Lambda, `public.ecr.aws/lambda/python:3.13`, arm64).
* Dataset: VisA (Amazon, CC BY 4.0) `data/raw/VisA_20220922.tar`. Categories used: `config.CATEGORIES`.
  Layout inside tar: `<category>/Data/Images/{Normal,Anomaly}/*.JPG`, `<category>/Data/Masks/Anomaly/*.png`,
  `split_csv/1cls.csv` (columns `object,split,label,image,mask`; label `normal|anomaly`; paths relative to VisA root).

## 1. Repository layout

```
inspectagent/
  types.py config.py __init__.py (version)
  vision/
    cvinfo.py          OpenCV build/arch/COOL introspection, thread setup
    camera.py          VirtualCamera (active camera simulator) + condition presets
    quality.py         assess_quality()
    registration.py    register() — ORB/affine → ECC → phase-correlation fallback chain
    anomaly_stat.py    StatTemplateModel — fast, registration-based statistical template detector
    anomaly_deep.py    PatchCoreDNN — deep detector, ResNet18 ONNX features via cv2.dnn + coreset kNN
    regions.py         extract_regions(), measure_region()
    overlay.py         heatmap/box overlays, JPEG encoding, thumbnails
    imageio.py         read/write helpers (imdecode for unicode paths on Windows!)
  agent/
    session.py         InspectionSession (state, artifacts, ids)
    tools.py           ToolRegistry: tool specs (JSON Schema) + implementations
    policy.py          HumanControlPolicy: guardrails, budget, final verdict gating
    planner_rule.py    RulePlanner — deterministic active-perception policy
    planner_bedrock.py BedrockPlanner — Amazon Bedrock Converse tool-use planner (+fallback)
    loop.py            run_inspection()
    trace.py           trace (de)serialization, structured logging
  models_io.py         CategoryBundle load/save (models/<category>/...)
  mcp_server.py        MCP server exposing the tools + run_inspection
  api/
    app.py             FastAPI app (+ static UI)
    storage.py         LocalStorage / AwsStorage (S3 + DynamoDB)
    lambda_handler.py  Mangum handler
    static/            index.html, app.js, style.css
scripts/   prepare_visa.py build_models.py export_backbone.py evaluate.py make_figures.py bench_cv.py make_diagrams.py
deploy/    Dockerfile.lambda Dockerfile aws/template.yaml aws/buildspec.yml aws/deploy.py
tests/     pytest suite (synthetic data; must not require VisA, AWS, or network)
docs/      technical_report.md, diagrams, evaluation
samples/   curated demo parts (VisA images, CC BY 4.0) + samples.json
models/    per-category bundles (generated; not committed — published as a release asset)
```

**Windows path rule:** the user's home contains non-ASCII characters (`Seyit Kaan`, Turkish locale). Never
use `cv2.imread`/`cv2.imwrite` directly on paths; use `vision.imageio.imread/imwrite` which go through
`np.fromfile` + `cv2.imdecode` / `cv2.imencode` + `tofile`.

## 2. Vision layer

### 2.1 `vision/imageio.py`
```python
def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray   # raises FileNotFoundError / ValueError
def imwrite(path: str | Path, img: np.ndarray, params: list[int] | None = None) -> None  # creates parent dirs
def encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes
def decode_image(data: bytes, flags=cv2.IMREAD_COLOR) -> np.ndarray
```

### 2.2 `vision/cvinfo.py`
```python
def cv_info() -> dict   # {"opencv_version", "build_backend": "stock"|"cool", "arch": platform.machine(),
                        #  "cpu_baseline", "cpu_dispatch", "parallel_framework", "threads", "python", "numpy",
                        #  "dnn_engines": [...], "runtime": "lambda"|"local"|"container"}
def configure_threads(n: int | None = None) -> int   # cv2.setNumThreads; returns effective
```
COOL detection: `build_backend="cool"` if env `INSPECTAGENT_CV_BACKEND=cool` **or** build info contains `COOL` /
`Cloud-Optimized` (case-insensitive). Report the raw evidence string too (`"backend_evidence"`).

### 2.3 `vision/camera.py` — active camera simulator (the "world")
The VisA image is the physical part. The virtual camera has fixed sensor resolution and a controllable view,
exposure compensation, light mode and autofocus. Hidden `NuisanceConditions` (per part) degrade captures.

```python
class VirtualCamera:
    def __init__(self, scene_bgr: np.ndarray, conditions: NuisanceConditions, cfg: CameraConfig | None = None, seed: int = 0)
    scene_size: tuple[int, int]                      # (W_s, H_s)
    sensor_size: tuple[int, int]                     # long side = cfg.sensor_long_side, aspect = scene aspect
    def view_window(self, view: View) -> tuple[float, float, float, float]   # x0,y0,w,h (scene px), clamped (shifted) inside scene
    def capture(self, settings: CaptureSettings, capture_id: str, index: int) -> Capture
    def clean_capture(self, view: View, capture_id: str, index: int) -> Capture   # oracle: NO photometric/blur/noise/glare degradations, but SAME pose jitter; oracle=True
    def estimate_time_ms(self, prev: CaptureSettings | None, settings: CaptureSettings) -> float
def make_conditions(preset: str, rng: np.random.Generator) -> NuisanceConditions
PRESETS = ("clean", "nominal", "dark", "bright", "defocus", "glare", "harsh")   # "harsh" = random mix of strong degradations
```
Physics (implement in this order, float32, linear light):
1. Pose jitter (fixed per camera instance): `warpAffine` of the scene by rotation `rot_deg` about the centre and shift
   `(shift_x*W, shift_y*H)`, `BORDER_REPLICATE`. Precompute once in `__init__`.
2. Crop view window from the jittered scene, resize to sensor size (`INTER_AREA` when downscaling, `INTER_CUBIC` when upscaling).
3. sRGB→linear (`(x/255)**2.2`). Multiply by `light_gain` (ring 1.0, diffuse 0.6, low_angle 0.5) × `2**ambient_ev`.
   * `low_angle`: multiply by `(1 + 0.6 * relief)` where relief = normalized Sobel-x of gray (directional shading), clipped to [-0.6, 0.6].
   * `ring` with `glare_strength>0`: add specular glare: 1–3 soft elliptical blobs placed in **scene coordinates** from
     `np.random.default_rng(glare_seed)` (radius 3–8 % of scene diagonal, stay within scene), mapped to the capture via the window;
     intensity `glare_strength * 2.5` (linear units, saturating). Glare is absent for diffuse/low_angle.
4. Digital gain `2**exposure_ev`.
5. Defocus: Gaussian blur sigma = `defocus_sigma` if not autofocus else `0.35`.
6. Noise: `std = gain * (noise_sigma/255 + 0.012*sqrt(signal_before_gain))`; rng seeded by `(seed, index)` → deterministic.
7. Clip [0,1], linear→sRGB, uint8, JPEG round-trip at `jpeg_quality`.
`clean_capture` skips 3–7 (except it still renders at the sensor resolution of the view).
`make_conditions`: `clean` = all zeros except noise 0, `nominal` = small random (|ambient_ev|≤0.3, defocus≤0.6, noise 1–2, glare 0 w.p. 0.8 else ≤0.2,
pose rot ≤1.5°, shift ≤1.5 %), `dark` ambient −1.6…−2.4, `bright` +1.2…+1.8, `defocus` sigma 2.5–4.5, `glare` strength 0.5–0.9,
`harsh` = combine 1–2 strong degradations at random. Pose jitter for all non-clean presets: rot ≤2.5°, shift ≤2.5 %.

### 2.4 `vision/quality.py`
```python
def assess_quality(capture: Capture, reference_sharpness: float | None = None, cfg: QualityConfig | None = None) -> QualityReport
def laplacian_sharpness(gray: np.ndarray) -> float
def estimate_noise_sigma(gray: np.ndarray) -> float        # Immerkær fast noise estimation
def detect_glare(bgr: np.ndarray) -> tuple[float, list[list[int]]]  # saturated low-saturation blobs (HSV V>=245, S<=40), area-filtered
```
Issues: `blurred` (sharpness_ratio < min_sharpness_ratio; if reference unknown use absolute heuristic on Laplacian variance),
`underexposed` (mean < min_luma or crushed > max), `overexposed` (mean > max_luma or clipped > max), `noisy` (noise > max),
`glare` (glare_frac > max_glare_frac). `suggested_exposure_delta = log2(target_lin/mean_lin)` computed in linear light, clamped ±3.
`ok = not issues`.

### 2.5 `vision/registration.py`
```python
@dataclass
class RegistrationResult:
    ok: bool; method: str; M: np.ndarray  # 2x3 affine mapping MOVING (capture) px -> FIXED (reference) px
    inliers: int; inlier_ratio: float; ncc: float; dx: float; dy: float; rot_deg: float; scale: float
    def to_dict(self) -> dict   # without M (or M as list)
def register(moving_gray: np.ndarray, fixed_gray: np.ndarray, valid_fixed: np.ndarray | None = None, cfg: DetectorConfig | None = None) -> RegistrationResult
```
Chain: CLAHE-normalize both → ORB(nfeatures=4000) + BFMatcher Hamming kNN ratio 0.8 → `estimateAffinePartial2D` RANSAC (thr 3px) →
accept if inliers ≥ min_inliers, ratio ≥ min_inlier_ratio, |rot| ≤ 8°, scale ∈ [0.85, 1.15]; optionally refine with ECC (MOTION_EUCLIDEAN or
AFFINE) on a 0.5× pyramid level. If ORB fails → ECC alone (identity init) → if fails → `phaseCorrelate` translation. Final NCC between warped
moving and fixed (in valid area) decides `ok` (ncc ≥ min_ncc).

### 2.6 `vision/anomaly_stat.py` — fast detector
```python
class StatTemplateModel:
    category: str; scene_size: tuple[int,int]; scale: float   # template px = scene px * scale
    mu_lab: np.ndarray (Ht,Wt,3) float32; sigma_lab: same; ref_gray: uint8 (Ht,Wt); calib: dict
    @classmethod
    def fit(cls, category: str, normal_images: list[np.ndarray] | Iterable[np.ndarray], cfg: DetectorConfig | None = None, log=print) -> "StatTemplateModel"
    def predict(self, capture: Capture, quality: QualityReport | None = None, id_prefix: str = "") -> AnomalyResult
    def score_capture(self, capture: Capture) -> tuple[float, np.ndarray, np.ndarray, RegistrationResult]  # score, zmap, valid, reg
    def calibrate(self, normal_scores_by_zoom: dict[str, list[float]], labeled: list[tuple[float, int, str]] | None = None) -> None
    def prob(self, score: float, zoom: float) -> float
    def save(self, path: Path) -> None ; @classmethod load(cls, path: Path) -> "StatTemplateModel"   # npz, float16 arrays
```
Fit: pick the medoid normal (downscaled gray, L2 to the median image), register all normals to it (registration.register), drop failures,
accumulate Lab mean/std (Welford, float64), `sigma = sqrt(std² + sigma_floor²)` then Gaussian-smooth sigma (σ=1.5). Template resolution:
long side `template_long_side`.
Predict for a capture with commanded window: (1) render expected template in capture grid: `M = scene_to_capture ∘ (1/scale)`, warp
`mu_lab`, `sigma_lab`, `ref_gray`, valid mask; (2) register capture gray → expected ref gray; (3) warp capture into expected frame;
(4) photometric normalisation: per Lab channel robust linear fit capture≈a·mu+b on valid px (subsample, 2-pass trimmed LSQ), a∈[0.3,3];
(5) noise-aware sigma: `sqrt(sigma² + (1.2·noise_L)²)` where noise_L is the estimated noise converted to L units;
(6) `z = max_c |x−mu|/sigma`, Gaussian blur (zmap_blur_sigma), zero outside eroded valid mask; (7) score = mean of top-k (k=max(topk_min,
topk_frac·N)); (8) regions via `regions.extract_regions(zmap, thr=calib["region_thr"][zoom_bucket])`; map region boxes to scene px through
inverse(residual) then `capture_to_scene`. `reliable = reg.ok and valid_frac ≥ min_valid_frac and (quality is None or not
quality.blocking_quality_issues(quality))` — every quality issue blocks except `noisy` with `noise_sigma ≤ NOISE_TOLERANCE_SIGMA` (14.0),
which step 5's noise-aware sigma absorbs (integration change: the rule planner accepts moderate noise it cannot correct, e.g. after the
+0.75 EV glare compensation, so the detector must not discard those captures).
Zoom buckets: `"z1"` (zoom<1.5), `"z2"` (<2.5), `"z3"` (<3.5), `"z4"`. Calibration stores per bucket `{"mu","sd","region_thr"}` from normal
captures and a global logistic `{"a","b"}` on standardized scores fitted on labeled calibration data (fallback a=1.6,b=−4.0 if no labels):
`prob = sigmoid(a·(score−mu)/sd + b)`.

### 2.7 `vision/anomaly_deep.py` — deep detector on `cv2.dnn`
```python
class PatchCoreDNN:
    def __init__(self, onnx_path: Path, input_size: tuple[int,int])   # (W,H) multiple of 16
    @classmethod
    def fit(cls, category, normal_images, onnx_path, cfg=None, log=print) -> "PatchCoreDNN"   # memory bank via greedy coreset
    def features(self, bgr: np.ndarray) -> np.ndarray          # (h, w, D) patch features, D=384
    def predict(self, capture: Capture, id_prefix: str = "") -> AnomalyResult   # map upsampled to capture size, Gaussian σ=4
    def calibrate(self, normal_scores: list[float], labeled: list[tuple[float,int]] | None = None) -> None
    def save(self, path: Path) / load(cls, path: Path, onnx_path: Path)
```
Backbone: torchvision ResNet18 ImageNet weights, outputs `layer2` (stride 8, 128 ch) and `layer3` (stride 16, 256 ch), exported to ONNX
by `scripts/export_backbone.py` (dev-only torch dependency; dynamic H/W if cv2.dnn handles it, otherwise fixed per-category size).
Inference strictly through `cv2.dnn` (`readNetFromONNX`, `blobFromImageWithParams` with ImageNet mean/std, `net.forward([out names])`).
Patch feature = concat(layer2, upsample(layer3)) → 3×3 avg-pool. Memory bank: ≤`deep_train_images` normals → random 20 % subsample →
greedy k-center coreset (random projection to 128-d for selection) of `deep_coreset_size` → float16. NN distance via chunked matmul.
Image score = max of the smoothed map. Model file `models/backbone_resnet18.onnx` shared by categories. Deep result `reliable=True`
whenever inference succeeded (it is alignment-free — that is why the agent escalates to it when registration fails), unless the optional
`predict(capture, id_prefix="", quality=None)` quality report has a blocking issue (same rule as §2.6; glare scores as an anomaly in the
deep map). The calibration logistic is L2-regularised (slope ridge 0.5, like §2.6) so separable calibration sets stay graded.

### 2.8 `vision/regions.py`
```python
def extract_regions(amap: np.ndarray, thr: float, capture: Capture, residual_M: np.ndarray | None, source: str,
                    min_area: int, max_regions: int, id_prefix: str) -> list[Region]
def measure_region(region: Region, capture: Capture, amap: np.ndarray | None, px_per_mm: float) -> dict
# -> {"area_mm2","length_mm","width_mm","elongation","orientation_deg","contrast","type_guess": "scratch|spot|missing_or_extra|contamination|unknown"}
```
Threshold → morphology (open 3×3, close 5×5) → `connectedComponentsWithStats` → rank by `peak*sqrt(area)` → top-N. measure_region uses
`minAreaRect` on the thresholded component inside the bbox.

### 2.9 `vision/overlay.py`
```python
def heatmap_overlay(bgr, amap, vmax=None, alpha=0.45) -> np.ndarray   # COLORMAP_INFERNO (TURBO ok), masked where low
def draw_regions(bgr, regions, color=(0,0,255), label=True) -> np.ndarray
def draw_quality(bgr, q: QualityReport) -> np.ndarray   # glare boxes + issue badges
def thumbnail(bgr, long_side=384) -> np.ndarray
```

### 2.10 `models_io.py`
```python
@dataclass
class CategoryBundle:
    category: str; stat: StatTemplateModel; deep: PatchCoreDNN | None; meta: dict; reference_sharpness: dict[str, float]  # per zoom bucket
def bundle_dir(category, models_dir=None) -> Path        # models/<category>/
def save_bundle(b: CategoryBundle, models_dir=None) -> None   # stat.npz, deep.npz, meta.json
def load_bundle(category, models_dir=None, with_deep=True) -> CategoryBundle   # cached (lru) per process
def available_categories(models_dir=None) -> list[str]
```

## 3. Agent layer

### 3.1 Tools (`agent/tools.py`)
`ToolRegistry` holds `ToolSpec(name, description, input_schema: dict, fn: Callable[[InspectionSession, dict], dict], terminal: bool)`.
`registry.specs()` → list of JSON-schema specs (used verbatim by Bedrock `toolConfig` and MCP). `registry.execute(session, name, args) -> dict`
validates args against the schema (jsonschema), clamps numeric ranges, records artifacts in the session, returns a compact JSON observation
(never raw arrays). Tools:

| tool | args | returns (observation) |
|---|---|---|
| `capture_image` | `cx,cy∈[0,1]`, `zoom∈[1,4]`, `exposure_ev∈[-3,3]`, `light∈{ring,diffuse,low_angle}`, `autofocus:bool` | `capture_id`, window, size, `sim_time_ms`, `mean_luma`, artifact id |
| `assess_quality` | `capture_id` | QualityReport dict + artifact (quality overlay) |
| `detect_anomalies` | `capture_id`, `detector∈{fast,deep}` | AnomalyResult dict (regions with ids), artifact (heatmap overlay) |
| `inspect_region` | `region_id`, `zoom∈[2,4]` (default 3), optional `exposure_ev`, `light`, `autofocus` (default: reuse last good settings) | zoom capture id, quality summary, fast result on zoom view, `confirmed: bool`, `region_peak`, `prob` |
| `measure_defect` | `region_id` | measure_region dict |
| `request_human_review` | `reason`, `proposed_label∈{PASS,FAIL}`, `confidence` | `review_id` (terminal) |
| `finish` | `label∈{PASS,FAIL}`, `confidence`, `rationale`, `region_ids` | proposed verdict (terminal) |

Evidence fusion helper in tools/session: `session.fused_prob()` = max over reliable fast results on full views, confirmed zoom results, and deep
results, with refutation: a zoom inspection that does NOT confirm a candidate (zoom prob < low threshold, good quality) lowers that
candidate's contribution. Document the exact rule in code.
*Contested evidence* (`session.contested_results()`): a full-view result excluded from the fusion whose registration succeeded, with
p ≥ 0.5 and unrefuted regions, while a reliable result on the *same capture* keeps the fused p low (two detectors disagree on the same
pixels). It blocks auto-PASS (policy) until a zoom refutes its regions; the rule planner zooms into them, else asks a human.

### 3.2 `agent/session.py`
```python
class InspectionSession:
    def __init__(self, part: PartSpec, bundle: CategoryBundle, settings: Settings | None = None, seed: int = 0, scene: np.ndarray | None = None)
    session_id; part; bundle; camera: VirtualCamera
    captures: dict[str, Capture]; quality: dict[str, QualityReport]; results: dict[str, AnomalyResult]; regions: dict[str, Region]
    measurements: dict[str, dict]; artifacts: dict[str, bytes] (JPEG); proposed: dict | None; review: dict | None
    sim_time_ms: float; compute_ms: float; counters: dict
    def new_id(self, kind: str) -> str              # "cap_1", "q_1", "det_1", "reg_1_2", "art_3"... deterministic
    def add_artifact(self, img: np.ndarray, kind: str) -> str
    def last_good_settings(self) -> CaptureSettings
    def planner_state(self) -> dict                 # compact JSON history for planners (no arrays)
    def fused_prob(self) -> tuple[float, list[str]]  # value + evidence ids
```

### 3.3 `agent/policy.py` — human control
```python
class HumanControlPolicy:
    def __init__(self, cfg: PolicyConfig | None = None)
    def vet(self, decision: Decision, session) -> tuple[Decision, bool]   # (possibly replaced decision, overridden?)
       # budget exceeded (steps/captures/deep calls/sim time) -> request_human_review; unknown tool -> request_human_review;
       # finish(PASS) with fused_prob > auto_pass_max_prob, no reliable evidence or contested evidence -> request_human_review;
       # finish(FAIL) with fused_prob < auto_fail_min_prob -> request_human_review
    def final_verdict(self, session) -> Verdict
       # PASS -> route_pass; FAIL -> route_quarantine (scrap requires human); review -> hold_for_review (review ticket id)
```

### 3.4 `agent/planner_rule.py` — deterministic active-perception policy
`RulePlanner.next_action(state: dict, session) -> Decision`. Protocol:
1. No capture yet → `capture_image(0.5,0.5,1,ev=0,ring,af=False)`.
2. Last capture not quality-assessed → `assess_quality`.
3. Quality issues (≤3 corrective recaptures of the full view): `glare` → `light=diffuse`, `ev += 0.75` (compensate); `blurred` → `autofocus=True`;
   under/over-exposed → `ev += suggested_exposure_delta` (clamped); `noisy` with ev>1 → switch to ring if no glare seen, else accept.
   Combine fixes in one recapture. Rationale must quote the measured values and thresholds.
4. Full view OK (or retries exhausted) and not yet analysed → `detect_anomalies(fast)` on the best full view.
5. Fast unreliable (registration failed) → `detect_anomalies(deep)` (if budget and the capture has no blocking quality issue — a deep
   result on it would be unreliable too) else review.
6. p ≥ auto_fail and a region exists → `inspect_region(top)` once to confirm when p < 0.98, then `measure_defect` → `finish(FAIL)`.
7. p ≤ auto_pass and no region above threshold and no contested evidence → `finish(PASS)` (contested regions are zoom candidates).
8. Ambiguous → `inspect_region` on top-k (≤3) un-inspected candidates (zoom 3); a confirmation → `measure_defect` → `finish(FAIL)`; all
   refuted → `detect_anomalies(deep)` on full view → decide by thresholds → otherwise `request_human_review` with the evidence.
Every `Decision.rationale` names the evidence ids and numbers (e.g. "q_1: glare_frac=0.031>0.004 → diffuse light, EV +0.75").

### 3.5 `agent/planner_bedrock.py`
`BedrockPlanner(model_id, client=None, region=None, max_tokens=1024, fallback=RulePlanner())`. Uses `bedrock-runtime` **Converse** API with
`toolConfig` from `registry.specs()`, a system prompt describing the inspection protocol, safety rules and budget, and per step the latest
observation JSON (plus an optional 256-px JPEG thumbnail of the newest capture/overlay as an image content block). Keeps the conversation
history in the session. Parses `toolUse` blocks → `Decision` (rationale = model text). Guardrails: unknown tool/invalid args → validated by
the registry, at most one tool per step, retries 2 on throttling, then falls back to `RulePlanner` (decision.planner = "rule(fallback)").
Must be fully testable with a stubbed client (no network in tests).

### 3.6 `agent/loop.py`
```python
def run_inspection(part: PartSpec, bundle: CategoryBundle | None = None, planner=None, policy=None, settings=None, seed=0,
                   on_step: Callable[[Step, InspectionSession], None] | None = None, scene: np.ndarray | None = None) -> tuple[Trace, InspectionSession]
def run_single_shot(part, bundle, detector="fast", oracle=False, seed=0, settings=None) -> dict   # baseline for evaluation (no agent)
```
Loop: decision = planner.next_action(...) → policy.vet → registry.execute → Step → on_step → stop on terminal tool or budget → verdict =
policy.final_verdict(session). Exceptions inside a tool become an observation `{"error": ...}` and a policy-forced review after 2 errors.
Trace.totals: `n_steps, n_captures, n_deep, compute_ms, sim_time_ms, cycle_time_ms (=compute+sim), tool_counts, final_prob`.

### 3.7 `agent/trace.py`
`trace_to_json(trace) -> str`, `save_trace(trace, path)`, `load_trace(path) -> dict`, `log_step(logger, step, trace_id)` emitting one JSON line per
step (CloudWatch-friendly: `{"evt":"step","trace_id",...}`).

## 4. MCP server (`mcp_server.py`)
Exposes over stdio (and streamable HTTP when `--http`): `list_samples`, `start_session(sample_id|image_path, category, preset, seed)` →
`session_id`, all tools of §3.1 with an extra `session_id` arg, `get_trace(session_id)`, and `run_inspection(sample_id, preset, planner="rule")`.
Sessions are kept in memory. Include `.mcp.json` example for Claude Code. Use the installed `mcp` 2.3.0 API (introspect: likely
`mcp.server.fastmcp.FastMCP` or `mcp.server.mcpserver`); tests use the SDK's in-memory client if available, else call handlers directly.

## 5. Web API (`api/app.py`) + UI
FastAPI, JSON only, CORS open (demo). Endpoints:
* `GET /healthz` → `{"ok": true, "version", "cv": cv_info()}`
* `GET /api/samples` → curated samples from `samples/samples.json` (`id, category, label, thumb_url`)
* `GET /api/samples/{id}/image` → JPEG (thumbnail by default, `?full=1` full)
* `POST /api/inspect` JSON `{sample_id, preset: str="nominal", planner: "rule"|"bedrock"="rule", seed: int=0}` or multipart upload
  (`file`, `category`, `preset`) → runs the loop synchronously → `{trace: Trace.to_dict(), artifacts: {id: "data:image/jpeg;base64,..."}}`
  (thumbnails ≤ 512 px, keep total response < 5 MB for Lambda). Persists trace + artifacts via storage. Creates a review item when needed.
* `GET /api/traces/{trace_id}`; `GET /api/traces?limit=20`
* `GET /api/reviews?status=pending|all`; `POST /api/reviews/{review_id}/decision` `{decision: "approve_fail"|"override_pass"|"rescan", reviewer, note}`
  → updates item, appends audit record, returns the final routing action.
* `GET /api/metrics` → summary of `results/eval_summary.json` (if present) + live counters.
* `GET /api/system` → cv_info, settings (non-secret), deployment info (Lambda env vars `AWS_LAMBDA_FUNCTION_NAME`, region, arch).
* `GET /api/bench?n=10` → runs `scripts/bench_cv.run_bench` in-process (latency of core ops), for x86-vs-Graviton evidence.
* `/` → static single-page UI.
Storage (`api/storage.py`): `Storage` protocol with `put_trace, get_trace, list_traces, put_artifact, put_review, list_reviews, update_review`.
`LocalStorage(dir)` (JSON files) and `AwsStorage(bucket, table, region)` (S3 `traces/{id}.json`, `artifacts/{id}.jpg`; DynamoDB single table
pk/sk: `TRACE#id`, `REVIEW#id`). Choose by `settings.storage`.
UI (vanilla JS, no build step; responsive): sample gallery + preset/planner selectors → "Inspect" → animated step timeline: each step shows
the tool, rationale, key numbers and the image artifact (capture / quality overlay / heatmap); verdict card with routing action and
confidence; tabs: **Inspect**, **Review queue** (approve / override / rescan), **Results** (eval tables + figures), **System** (OpenCV build,
arch=aarch64 on Graviton, Lambda info, bench button). Clear, professional styling (dark/light).

## 6. AWS deployment (`deploy/`)
* `Dockerfile.lambda`: `FROM public.ecr.aws/lambda/python:3.13` (built natively for arm64 by CodeBuild ARM), installs `requirements.txt`,
  copies `inspectagent/`, `samples/`, `models/`, sets `CMD ["inspectagent.api.lambda_handler.handler"]`.
* `Dockerfile`: generic `python:3.13-slim` + uvicorn (multi-arch) for local/ECS use.
* `aws/template.yaml` (CloudFormation): S3 bucket (artifacts, private, SSE, lifecycle 30 d), DynamoDB on-demand table, ECR repo, CodeBuild project
  (ARM_CONTAINER `aws/codebuild/amazonlinux-aarch64-standard:3.0`, privileged, source = S3 zip), IAM roles (least privilege), Lambda
  (`PackageType: Image`, `Architectures: [arm64]`, 3008 MB, 60 s, reserved concurrency 5, env vars), Function URL (AuthType NONE, CORS),
  log group (14 d). Image-dependent resources behind a condition `HasImage` (parameter `ImageUri`).
* `aws/buildspec.yml`: docker build + push `:git-sha` and `:latest`.
* `aws/deploy.py` (boto3, idempotent, prints progress): `--region`, `--stack-name inspectagent`, `--bedrock-model auto|<id>`, `--arch arm64|x86_64|both`;
  steps: deploy stack (no image) → zip repo (incl. models/, samples/) → upload to S3 → start CodeBuild → wait → update stack with ImageUri →
  print Function URL; `--destroy` tears everything down. `auto` Bedrock model: pick an available Anthropic Claude (prefer newest Sonnet/Haiku
  inference profile) or Amazon Nova via `bedrock.list_inference_profiles` / `list_foundation_models`.
* `scripts/bench_cv.py`: `run_bench(n=10) -> dict` timing registration, quality, stat predict, deep predict, full agent loop on a bundled sample;
  CLI writes `results/bench_<arch>_<backend>.json`.

## 7. Evaluation (`scripts/evaluate.py`)
Data split (deterministic, seed 0): VisA `1cls` train normals → 90 % fit / 10 % `val_normal` (normal-score calibration); VisA test →
stratified 40 % `calib` (logistic + thresholds) / 60 % `eval` (all reported numbers). Conditions per eval part: regimes `nominal` and `harsh`
(seeded per part). Systems: `oracle_fast` (clean capture, single shot), `single_fast`, `single_deep` (one default capture), `agent_rule`
(full loop), `agent_bedrock` (subset, only if AWS creds; capped cost). Metrics per category × regime × system: image AUROC (score = final
fused prob for agents), accuracy, precision, recall, F1 at operating point, **escape rate** (defective → PASS), **false-reject rate** (normal →
FAIL), **review rate**, automation rate, accuracy on auto-decided parts, mean steps/captures/deep calls, mean compute ms, mean simulated cycle
time, tool-usage histogram, and a "visual evidence changed the action" rate (fraction of runs where a quality or detection result triggered a
corrective capture, zoom, deep escalation or review). Also pixel-AUROC of the fast/deep maps on oracle captures (localization). Outputs:
`results/eval_items.csv`, `results/eval_summary.json`, `results/figures/*.png`, `results/traces/` (curated example traces showing evidence →
action changes), `results/failures/` (overlays of escapes and false rejects). Multiprocessing; resumable (skip items already in CSV).

## 8. Testing
`pytest -q` must pass offline on Windows and Linux in < 2 min: synthetic "PCB-like" scenes (`tests/synth.py`: random rectangles, traces,
pads, text) with injected defects; camera physics invariants; quality detects each degradation; registration recovers known affine;
stat model detects injected defect and passes clean; deep model test skipped if ONNX file missing; policy guardrails; rule planner reacts to
glare/blur/dark; loop terminates within budget; API endpoints via `fastapi.testclient`; MCP tools callable; Bedrock planner with stub client.

## 9. Non-goals / honesty rules
* Never fabricate numbers. Every number in docs comes from a script output file in `results/`.
* The camera is a simulator over real VisA images — say so clearly; the camera interface is designed so a real PTZ/industrial camera driver
  can replace it.
* Report failures and limitations (non-aligned categories, simulator gap, threshold calibration on a held-out split, etc.).
