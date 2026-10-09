"""scripts/bench_cv.py: timing helpers, synthetic fallback scene, run_bench(n=2) and the CLI (offline)."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from inspectagent.config import Settings, reset_settings
from scripts import bench_cv

CORE_OPS = ("camera_capture", "assess_quality", "registration", "stat_predict", "run_inspection")


@pytest.fixture
def work_dir():
    """Fresh temporary directory (own mkdtemp instead of pytest's shared basetemp, which may be unreadable
    when several sandboxed processes share one Windows account)."""
    d = Path(tempfile.mkdtemp(prefix="inspectagent-test-"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _vision_missing() -> str | None:
    """Reason to skip when the concurrently written vision/agent modules are not importable yet."""
    try:
        import inspectagent.agent.loop  # noqa: F401
        import inspectagent.agent.planner_rule  # noqa: F401
        import inspectagent.models_io  # noqa: F401
        import inspectagent.vision.anomaly_stat  # noqa: F401
        import inspectagent.vision.camera  # noqa: F401
        import inspectagent.vision.cvinfo  # noqa: F401
        import inspectagent.vision.imageio  # noqa: F401
        import inspectagent.vision.quality  # noqa: F401
        import inspectagent.vision.registration  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"vision/agent modules not importable yet: {type(exc).__name__}: {exc}"
    return None


requires_vision = pytest.mark.skipif(_vision_missing() is not None, reason=str(_vision_missing()))


def _empty_settings(work_dir: Path) -> Settings:
    """Settings whose models/samples dirs are empty -> forces the synthetic fallback."""
    s = Settings()
    s.models_dir = work_dir / "models"
    s.samples_dir = work_dir / "samples"
    s.models_dir.mkdir()
    s.samples_dir.mkdir()
    return s


def test_summarize_ms():
    s = bench_cv.summarize_ms([10.0, 20.0, 30.0, 40.0])
    assert s == {"n": 4, "median_ms": 25.0, "p90_ms": 37.0, "mean_ms": 25.0, "min_ms": 10.0, "max_ms": 40.0}
    with pytest.raises(ValueError):
        bench_cv.summarize_ms([])


def test_synthetic_board_is_deterministic():
    a, b = bench_cv.synthetic_board(3), bench_cv.synthetic_board(3)
    assert a.shape == (bench_cv.SYNTH_SIZE[1], bench_cv.SYNTH_SIZE[0], 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)
    c = bench_cv.synthetic_board(4)  # same layout, small instance variation
    diff = np.abs(a.astype(np.int16) - c.astype(np.int16))
    assert 0 < diff.mean() < 12


def test_default_out_path_naming():
    p = bench_cv.default_out_path({"cv": {"arch": "AMD64", "build_backend": "stock"}})
    assert p.name == "bench_x86_64_stock.json" and p.parent.name == "results"
    assert bench_cv.default_out_path({"cv": {"arch": "arm64", "build_backend": "cool"}}).name == "bench_aarch64_cool.json"


def test_run_bench_rejects_bad_n():
    with pytest.raises(ValueError):
        bench_cv.run_bench(n=0)


@requires_vision
def test_run_bench_synthetic_fallback(work_dir):
    report = bench_cv.run_bench(n=2, settings=_empty_settings(work_dir))
    json.dumps(report)  # JSON-serializable (served by GET /api/bench)
    assert report["schema"] == bench_cv.SCHEMA and report["n"] == 2
    assert report["source"]["kind"] == "synthetic" and report["source"]["has_deep"] is False
    assert report["source"]["scene_size"] == list(bench_cv.SYNTH_SIZE)
    for op in CORE_OPS:
        t = report["timings_ms"][op]
        assert t["n"] == 2 and t["median_ms"] > 0 and t["p90_ms"] >= t["median_ms"] >= t["min_ms"]
    assert "deep_predict" not in report["timings_ms"]
    assert report["details"]["deep_predict"] == {"skipped": "bundle has no deep detector"}
    loop = report["details"]["run_inspection"]
    assert len(loop["runs"]) == 2 and all(r["n_steps"] >= 1 for r in loop["runs"])
    assert all(r["verdict"] in ("PASS", "FAIL", "NEEDS_REVIEW") for r in loop["runs"])
    assert report["cv"]["opencv_version"].startswith("5.")
    assert "fit_stat_template" in report["setup_ms"]
    assert any("synthetic fallback" in n for n in report["notes"])


@requires_vision
def test_run_bench_unknown_category_falls_back_with_note(work_dir):
    report = bench_cv.run_bench(n=1, category="pcb4", settings=_empty_settings(work_dir), warmup=0)
    assert report["source"]["kind"] == "synthetic"
    assert any("pcb4" in n for n in report["notes"])


@requires_vision
def test_run_bench_prefers_bundled_normal_sample(work_dir):
    from inspectagent.models_io import CategoryBundle, clear_bundle_cache, save_bundle
    from inspectagent.vision.imageio import imwrite

    synth, scene, _fit_ms = bench_cv._synthetic_bundle()
    settings = _empty_settings(work_dir)
    save_bundle(CategoryBundle("pcb1", synth.stat, None, {}, dict(synth.reference_sharpness)), settings.models_dir)
    imwrite(settings.samples_dir / "pcb1" / "normal.jpg", scene)
    samples = [
        {"id": "pcb1_anomaly_x", "category": "pcb1", "label": 1, "image": "pcb1/missing.jpg"},
        {"id": "pcb1_normal_x", "category": "pcb1", "label": 0, "image": "pcb1/normal.jpg"},
        {"id": "cashew_normal_x", "category": "cashew", "label": 0, "image": "cashew/none.jpg"},  # no bundle
    ]
    (settings.samples_dir / "samples.json").write_text(json.dumps(samples), encoding="utf-8")
    try:
        report = bench_cv.run_bench(n=1, warmup=0, settings=settings)
    finally:
        clear_bundle_cache()
    assert report["source"]["kind"] == "bundle"
    assert report["source"]["sample_id"] == "pcb1_normal_x" and report["source"]["category"] == "pcb1"
    assert "load_bundle" in report["setup_ms"] and report["notes"] == []
    assert set(CORE_OPS) <= set(report["timings_ms"])


@requires_vision
def test_cli_writes_report(work_dir, monkeypatch, capsys):
    monkeypatch.setenv("INSPECTAGENT_MODELS_DIR", str(work_dir / "no_models"))
    monkeypatch.setenv("INSPECTAGENT_SAMPLES_DIR", str(work_dir / "no_samples"))
    reset_settings()
    try:
        out = work_dir / "bench.json"
        assert bench_cv.main(["--n", "1", "--warmup", "0", "--out", str(out)]) == 0
    finally:
        monkeypatch.undo()
        reset_settings()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(CORE_OPS) <= set(data["timings_ms"])
    printed = capsys.readouterr().out
    assert "median ms" in printed and str(out) in printed
