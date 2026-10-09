"""Unit tests for inspectagent.eval_utils (small hand-checked examples) and the
pure parts of the evaluation / data scripts (offline, no VisA, no AWS)."""

from __future__ import annotations

import csv
import json
import math
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pytest

from inspectagent import eval_utils as eu


@pytest.fixture
def tmp_path():
    """Private temp dir (pytest's shared basetemp may be unreadable in sandboxed runs)."""
    d = Path(tempfile.mkdtemp(prefix="ia_eval_test_"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- ids & seeds


def test_stable_seed_is_deterministic_and_bounded():
    a = eu.stable_seed("pcb1_anomaly_000", "harsh")
    assert a == eu.stable_seed("pcb1_anomaly_000", "harsh")
    assert a != eu.stable_seed("pcb1_anomaly_000", "nominal")
    assert 0 <= a < 2**31
    # value pinned: blake2b is platform independent, so seeds never drift between machines
    assert eu.stable_seed(0, "pcb1", "fit") == eu.stable_seed("0", "pcb1", "fit")


def test_part_id_from_relpath():
    assert eu.part_id_from_relpath("pcb1/Data/Images/Anomaly/012.JPG") == "pcb1_anomaly_012"
    assert eu.part_id_from_relpath("pcb1\\Data\\Images\\Normal\\0012.JPG") == "pcb1_normal_0012"
    assert eu.part_id_from_relpath("cashew/x.png") == "cashew_part_x"


@pytest.mark.parametrize("zoom,bucket", [(1.0, "z1"), (1.49, "z1"), (1.5, "z2"), (2.4, "z2"), (3.0, "z3"), (3.5, "z4"), (4.0, "z4")])
def test_zoom_bucket(zoom, bucket):
    assert eu.zoom_bucket(zoom) == bucket


# --------------------------------------------------------------------------- AUROC


def test_rankdata_ties_average():
    np.testing.assert_allclose(eu.rankdata([10, 20, 20, 5]), [2.0, 3.5, 3.5, 1.0])
    np.testing.assert_allclose(eu.rankdata([1, 1, 1]), [2.0, 2.0, 2.0])
    assert eu.rankdata([]).size == 0


def test_auroc_hand_checked():
    # positives ranks 2 and 4 -> U = 6 - 3 = 3 -> 3 / (2*2) = 0.75 (matches sklearn)
    assert eu.auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert eu.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert eu.auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == pytest.approx(0.0)
    assert eu.auroc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)  # tie counts 1/2
    # 3 negatives, 2 positives, one tie between classes: pairs (p>n)=5, ties=1 -> (5 + 0.5)/6
    assert eu.auroc([0.1, 0.3, 0.6, 0.6, 0.9], [0, 0, 0, 1, 1]) == pytest.approx(5.5 / 6)


def test_auroc_degenerate_and_nonfinite():
    assert math.isnan(eu.auroc([0.1, 0.2], [1, 1]))
    assert math.isnan(eu.auroc([], []))
    # NaN score dropped with its label
    assert eu.auroc([0.1, float("nan"), 0.9], [0, 1, 1]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        eu.auroc([0.1, 0.2], [0])


def test_auroc_matches_pairwise_definition_random():
    rng = np.random.default_rng(3)
    s = np.round(rng.normal(size=60), 1)  # rounding creates ties
    y = rng.integers(0, 2, 60)
    pos, neg = s[y == 1], s[y == 0]
    pair = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (pos.size * neg.size)
    assert eu.auroc(s, y) == pytest.approx(pair)


def test_pixel_auroc_perfect_and_sampling():
    amap = np.zeros((20, 20), np.float32)
    mask = np.zeros((20, 20), np.uint8)
    mask[5:10, 5:10] = 255
    amap[5:10, 5:10] = 1.0
    res = eu.pixel_auroc([amap, amap], [mask, mask], max_pos=10, max_neg=30, seed=0)
    assert res["pooled"] == pytest.approx(1.0)
    assert res["mean_per_image"] == pytest.approx(1.0)
    assert res["n_images"] == 2
    s, y = eu.sample_pixels(amap, mask, 10, 30, np.random.default_rng(0))
    assert int(y.sum()) == 10 and int((y == 0).sum()) == 30 and s.shape == y.shape
    empty = eu.pixel_auroc([], [])
    assert empty["n_images"] == 0 and math.isnan(empty["pooled"])


# --------------------------------------------------------------------------- decisions


def test_normalize_decision():
    assert eu.normalize_decision("pass") == "PASS"
    assert eu.normalize_decision("FAIL") == "FAIL"
    assert eu.normalize_decision(None) == "NEEDS_REVIEW"
    assert eu.normalize_decision("review") == "NEEDS_REVIEW"
    assert eu.normalize_decision(1) == "FAIL" and eu.normalize_decision(0) == "PASS"
    assert eu.normalize_decision(True) == "FAIL"
    with pytest.raises(ValueError):
        eu.normalize_decision("maybe")


def test_decide_three_way():
    assert eu.decide_three_way(0.1, True, 0.15, 0.9) == "PASS"
    assert eu.decide_three_way(0.15, True, 0.15, 0.9) == "PASS"
    assert eu.decide_three_way(0.95, True, 0.15, 0.9) == "FAIL"
    assert eu.decide_three_way(0.5, True, 0.15, 0.9) == "NEEDS_REVIEW"
    assert eu.decide_three_way(0.01, False, 0.15, 0.9) == "NEEDS_REVIEW"  # unreliable never auto-decides
    assert eu.decide_three_way(None, True, 0.15, 0.9) == "NEEDS_REVIEW"


def test_binary_metrics_hand_checked():
    m = eu.binary_metrics([1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0])
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 3)
    assert m["accuracy"] == pytest.approx(5 / 7)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(3 / 4)
    z = eu.binary_metrics([0, 0], [0, 0])
    assert math.isnan(z["precision"]) and math.isnan(z["recall"]) and math.isnan(z["f1"])


def test_decision_metrics_hand_checked():
    labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    decisions = ["FAIL", "FAIL", "PASS", "NEEDS_REVIEW", "PASS", "PASS", "PASS", "FAIL", "NEEDS_REVIEW", "PASS"]
    m = eu.decision_metrics(labels, decisions)
    assert (m["tp"], m["fn"], m["rp"], m["tn"], m["fp"], m["rn"]) == (2, 1, 1, 4, 1, 1)
    assert m["n"] == 10 and m["n_pos"] == 4 and m["n_neg"] == 6
    assert m["escape_rate"] == pytest.approx(1 / 4)
    assert m["false_reject_rate"] == pytest.approx(1 / 6)
    assert m["review_rate"] == pytest.approx(2 / 10)
    assert m["automation_rate"] == pytest.approx(8 / 10)
    assert m["accuracy_auto"] == pytest.approx(6 / 8)
    assert m["accuracy_hitl"] == pytest.approx(8 / 10)
    # line view: flagged = FAIL or REVIEW -> TP=3, FP=2, FN=1, TN=4
    assert m["accuracy"] == pytest.approx(7 / 10)
    assert m["precision"] == pytest.approx(3 / 5)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(1 - m["escape_rate"])
    assert m["f1"] == pytest.approx(2 * 0.6 * 0.75 / 1.35)
    assert m["precision_auto"] == pytest.approx(2 / 3)
    assert m["recall_auto"] == pytest.approx(2 / 4)


def test_decision_metrics_empty_classes():
    m = eu.decision_metrics([0, 0], ["PASS", "NEEDS_REVIEW"])
    assert math.isnan(m["escape_rate"])
    assert m["false_reject_rate"] == 0.0
    assert m["review_rate"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        eu.decision_metrics([2], ["PASS"])


# --------------------------------------------------------------------------- risk-coverage


def test_risk_coverage_hand_checked():
    probs = [0.95, 0.05, 0.6, 0.45]
    labels = [1, 0, 0, 1]
    rc = eu.risk_coverage(probs, labels)
    # confidence order: 0.95(0.45) 0.05(0.45) [tie group] -> 0.6(0.1) -> 0.45(0.05)
    assert rc["coverage"] == pytest.approx([0.5, 0.75, 1.0])
    assert rc["risk"] == pytest.approx([0.0, 1 / 3, 2 / 4])
    assert rc["accuracy"] == pytest.approx([1.0, 2 / 3, 0.5])
    assert rc["aurc"] == pytest.approx(0.0 * 0.5 + (1 / 3) * 0.25 + 0.5 * 0.25)
    assert rc["n"] == 4
    assert eu.risk_coverage([], [])["n"] == 0


def test_resample_curve():
    out = eu.resample_curve([0.5, 1.0], [1.0, 0.5], [0.25, 0.5, 0.75, 1.0])
    assert math.isnan(out[0])
    assert out[1:] == pytest.approx([1.0, 0.75, 0.5])


# --------------------------------------------------------------------------- splits


def test_split_fraction_counts_and_determinism():
    items = list(range(20))
    a, b = eu.split_fraction(items, 0.9, seed=0)
    assert len(a) == 18 and len(b) == 2
    assert sorted(a + b) == items
    assert a == sorted(a) and b == sorted(b)  # input order kept
    assert eu.split_fraction(items, 0.9, seed=0) == (a, b)
    assert eu.split_fraction(items, 0.9, seed=1) != (a, b)
    with pytest.raises(ValueError):
        eu.split_fraction(items, 1.5, 0)


def test_stratified_split_preserves_label_ratio():
    labels = [0] * 10 + [1] * 5
    items = [f"x{i}" for i in range(15)]
    a, b = eu.stratified_split(items, labels, 0.4, seed=0)
    lab = dict(zip(items, labels))
    assert sum(1 for x in a if lab[x] == 0) == 4  # round(0.4*10)
    assert sum(1 for x in a if lab[x] == 1) == 2  # round(0.4*5)
    assert len(b) == 9 and set(a).isdisjoint(b) and set(a) | set(b) == set(items)
    assert eu.stratified_split(items, labels, 0.4, seed=0) == (a, b)


def test_stratified_subsample():
    items = list(range(30))
    labels = [0] * 20 + [1] * 10
    sub = eu.stratified_subsample(items, labels, 9, seed=0)
    assert len(sub) <= 9 and len(sub) >= 8
    assert sum(1 for i in sub if labels[i] == 1) >= 2
    assert eu.stratified_subsample(items, labels, 100, 0) == items
    assert eu.stratified_subsample(items, labels, 0, 0) == []


# --------------------------------------------------------------------------- bootstrap


def test_bootstrap_ci_contains_estimate_and_is_deterministic():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 1.0, 200)
    lo, hi = eu.bootstrap_ci(lambda a: float(np.mean(a)), [x], n_boot=500, seed=1)
    assert lo < x.mean() < hi
    assert hi - lo < 0.5
    assert (lo, hi) == eu.bootstrap_ci(lambda a: float(np.mean(a)), [x], n_boot=500, seed=1)


def test_bootstrap_ci_auroc_and_degenerate():
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.4, 0.6])
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    lo, hi = eu.bootstrap_ci(eu.auroc, [s, y], n_boot=400, seed=0)
    assert 0.0 <= lo <= eu.auroc(s, y) <= hi <= 1.0
    lo2, hi2 = eu.bootstrap_ci(eu.auroc, [np.array([0.1, 0.2]), np.array([1, 1])], n_boot=50)
    assert math.isnan(lo2) and math.isnan(hi2)
    assert all(math.isnan(v) for v in eu.bootstrap_ci(np.mean, [np.array([])]))


def test_rate_and_wilson_ci():
    r, lo, hi = eu.rate_ci([True, False, False, False], n_boot=300, seed=0)
    assert r == pytest.approx(0.25) and lo <= r <= hi
    lo, hi = eu.wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert all(math.isnan(v) for v in eu.wilson_ci(0, 0))


# --------------------------------------------------------------------------- aggregation


def test_describe_and_tool_histogram():
    d = eu.describe([1.0, 2.0, 3.0, None, float("nan")])
    assert d["n"] == 3 and d["mean"] == pytest.approx(2.0) and d["median"] == pytest.approx(2.0)
    assert eu.describe([])["n"] == 0
    h = eu.tool_histogram([{"capture_image": 2, "finish": 1}, {"capture_image": 1, "inspect_region": 2}, None])
    assert h["n_runs"] == 3
    assert h["total"] == {"capture_image": 3, "finish": 1, "inspect_region": 2}
    assert h["mean_per_run"]["capture_image"] == pytest.approx(1.0)
    assert h["runs_using"]["inspect_region"] == pytest.approx(1 / 3)


def _step(tool, args=None, obs=None, evidence=None, overridden=False):
    return {
        "decision": {"tool": tool, "args": args or {}, "evidence": evidence or []},
        "observation": obs or {},
        "overridden_by_policy": overridden,
    }


def test_evidence_action_flags():
    steps = [
        _step("capture_image"),
        _step("assess_quality", obs={"ok": False, "issues": ["glare"]}),
        _step("capture_image", args={"light": "diffuse"}, evidence=["q_1"]),
        _step("assess_quality", obs={"ok": True, "issues": []}),
        _step("detect_anomalies", args={"detector": "fast"}),
        _step("inspect_region", obs={"confirmed": False, "status": "refuted"}),
        _step("detect_anomalies", args={"detector": "deep"}),
        _step("request_human_review", overridden=True),
    ]
    f = eu.evidence_action_flags(steps)
    assert f["quality_recapture"] and f["zoom"] and f["zoom_refuted"] and not f["zoom_confirmed"]
    assert f["deep"] and f["review"] and f["review_visual"] and f["any"] and f["policy_overrides"] == 1
    plain = eu.evidence_action_flags(
        [_step("capture_image"), _step("assess_quality", obs={"ok": True, "issues": []}), _step("detect_anomalies"), _step("finish")]
    )
    assert not plain["any"] and plain["policy_overrides"] == 0


def test_inconclusive_zoom_is_not_counted_as_refuted():
    """Regression: obs['confirmed'] is False for inconclusive zooms too; only status == 'refuted' refutes."""
    f = eu.evidence_action_flags([_step("inspect_region", obs={"confirmed": False, "status": "inconclusive"})])
    assert f["zoom"] and f["zoom_inconclusive"] and not f["zoom_refuted"] and not f["zoom_confirmed"]
    old = eu.evidence_action_flags([_step("inspect_region", obs={"confirmed": False})])  # no status: never inferred
    assert old["zoom"] and not old["zoom_refuted"]


def test_non_visual_policy_reviews_are_not_evidence_driven():
    def policy_review(reason: str, trigger: str | None = None) -> dict:
        s = _step("request_human_review", obs={"review_trigger": trigger} if trigger else {}, overridden=True)
        s["decision"].update(planner="policy", rationale=f"policy: {reason} -> human review [fused p=0.1]")
        return s

    base = [_step("capture_image"), _step("assess_quality", obs={"ok": True, "issues": []}), _step("detect_anomalies")]
    for reason in ("step budget: step 14/14 must be terminal", "2 tool errors (limit 2)", "planner error RuntimeError: LLM down",
                   "unknown tool 'rm_rf'", "wall-clock budget: 2.0s left (<= 4.0s margin)"):
        f = eu.evidence_action_flags(base + [policy_review(reason)])  # legacy trace: parsed from the rationale
        assert f["review"] and f["review_nonvisual"] and not f["review_visual"] and not f["any"], reason
    f = eu.evidence_action_flags(base + [policy_review("anything", trigger="budget")])  # structured trigger wins
    assert not f["any"]
    f = eu.evidence_action_flags(base + [policy_review("finish(PASS) blocked: fused p=0.400>auto_pass_max_prob=0.15")])
    assert f["review_visual"] and f["any"]


def test_nan_to_none():
    out = eu.nan_to_none({"a": float("nan"), "b": [np.float32(1.5), np.int64(2)], "c": np.array([1.0, np.inf])})
    assert out == {"a": None, "b": [1.5, 2], "c": [1.0, None]}
    json.dumps(out, allow_nan=False)


# --------------------------------------------------------------------------- prepare_visa (pure parts)


def _fake_visa_tar(tmp_path: Path) -> Path:
    """Tiny tar with VisA layout: 2 categories, split csv and an unsafe member."""
    src = tmp_path / "src"
    rows = [["object", "split", "label", "image", "mask"]]
    for cat in ("pcb1", "candle"):
        for i in range(20):
            rel = f"{cat}/Data/Images/Normal/{i:04d}.JPG"
            (src / rel).parent.mkdir(parents=True, exist_ok=True)
            (src / rel).write_bytes(b"jpg")
            rows.append([cat, "train" if i < 15 else "test", "normal", rel, ""])
        for i in range(5):
            rel = f"{cat}/Data/Images/Anomaly/{i:03d}.JPG"
            mrel = f"{cat}/Data/Masks/Anomaly/{i:03d}.png"
            (src / rel).parent.mkdir(parents=True, exist_ok=True)
            (src / mrel).parent.mkdir(parents=True, exist_ok=True)
            (src / rel).write_bytes(b"jpg")
            (src / mrel).write_bytes(b"png")
            rows.append([cat, "test", "anomaly", rel, mrel])
    (src / "split_csv").mkdir(parents=True)
    with (src / "split_csv" / "1cls.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    (src / "split_csv" / "2cls_highshot.csv").write_text("x", encoding="utf-8")
    tar_path = tmp_path / "visa.tar"
    with tarfile.open(tar_path, "w") as t:
        for p in sorted(src.rglob("*")):
            t.add(p, arcname=p.relative_to(src).as_posix(), recursive=False)
    return tar_path


def test_prepare_visa_extract_and_split(tmp_path):
    from scripts import prepare_visa as pv

    tar_path = _fake_visa_tar(tmp_path)
    dest = tmp_path / "visa"
    info = pv.extract(tar_path, dest, ("pcb1",))
    assert (dest / "split_csv" / "1cls.csv").is_file()
    assert not (dest / "candle").exists()  # category filter
    assert not (dest / "split_csv" / "2cls_highshot.csv").exists()
    assert info["files"] == 20 + 5 * 2 + 1
    # second call is a no-op thanks to the marker
    assert pv.extract(tar_path, dest, ("pcb1",))["files"] == info["files"]

    rows = pv.read_split_csv(dest / "split_csv" / "1cls.csv", ("pcb1",))
    s1 = pv.build_splits(rows, ("pcb1",), seed=0)["pcb1"]
    s2 = pv.build_splits(rows, ("pcb1",), seed=0)["pcb1"]
    assert s1 == s2  # deterministic
    assert len(s1["fit"]) == 14 and len(s1["val_normal"]) == 1  # 90/10 of 15 train normals (rounded)
    assert set(s1["fit"]).isdisjoint(s1["val_normal"])
    calib_lab = [d["label"] for d in s1["calib"]]
    eval_lab = [d["label"] for d in s1["eval"]]
    assert calib_lab.count(0) == 2 and calib_lab.count(1) == 2  # round(0.4*5) each
    assert eval_lab.count(0) == 3 and eval_lab.count(1) == 3
    assert all(d["mask"] for d in s1["calib"] + s1["eval"] if d["label"] == 1)
    assert all(d["mask"] is None for d in s1["calib"] + s1["eval"] if d["label"] == 0)
    assert pv.verify_files({"pcb1": s1}, dest) == {"pcb1": 0}


def test_prepare_visa_wanted_filter():
    from scripts import prepare_visa as pv

    assert pv._wanted("./pcb1/Data/Images/Normal/0001.JPG", ("pcb1",)) == "pcb1/Data/Images/Normal/0001.JPG"
    assert pv._wanted("split_csv/1cls.csv", ("pcb1",)) == "split_csv/1cls.csv"
    assert pv._wanted("candle/Data/x.JPG", ("pcb1",)) is None
    assert pv._wanted("pcb1", ("pcb1",)) is None  # bare directory entry


# --------------------------------------------------------------------------- evaluate / figures (pure parts)


def _synthetic_rows(n_parts: int = 24) -> list[dict]:
    rng = np.random.default_rng(0)
    rows = []
    for cat in ("pcb1", "cashew"):
        for i in range(n_parts):
            label = int(i % 3 == 0)
            for regime in ("nominal", "harsh"):
                for system in ("oracle_fast", "single_fast", "agent_rule"):
                    p = float(np.clip(rng.normal(0.75 if label else 0.2, 0.2), 0, 1))
                    dec = eu.decide_three_way(p, True, 0.15, 0.9)
                    agent = system.startswith("agent")
                    rows.append(
                        {
                            "part_id": f"{cat}_{i}",
                            "category": cat,
                            "label": label,
                            "regime": regime,
                            "system": system,
                            "decision": dec,
                            "prob": p,
                            "score": p,
                            "reliable": True,
                            "n_steps": int(rng.integers(5, 12)) if agent else 1,
                            "n_captures": int(rng.integers(1, 4)) if agent else 1,
                            "n_deep": int(rng.integers(0, 2)) if agent else 0,
                            "compute_ms": float(rng.uniform(100, 900)),
                            "sim_time_ms": float(rng.uniform(100, 2000)),
                            "cycle_time_ms": float(rng.uniform(200, 3000)),
                            "tool_counts": json.dumps({"capture_image": 2, "assess_quality": 2, "detect_anomalies": 1, "finish": 1}) if agent else "",
                            "ev_quality_recapture": bool(rng.random() < 0.3) if agent else "",
                            "ev_zoom": bool(rng.random() < 0.3) if agent else "",
                            "ev_deep": bool(rng.random() < 0.2) if agent else "",
                            "ev_review": dec == "NEEDS_REVIEW" if agent else "",
                            "ev_any": bool(rng.random() < 0.6) if agent else "",
                            "error": "",
                        }
                    )
    return rows


def test_evaluate_summarize_on_synthetic_rows(tmp_path):
    from scripts import evaluate as ev

    rows = _synthetic_rows()
    path = tmp_path / "eval_items.csv"
    ev.append_rows(path, rows[:10])
    ev.append_rows(path, rows[10:])
    loaded = ev.read_rows(path)
    assert len(loaded) == len(rows)
    assert ev.done_keys(loaded) == {(r["part_id"], r["regime"], r["system"]) for r in rows}
    summary = ev.summarize(loaded, pixel=None, n_boot=50)
    res = summary["results"]
    assert set(res) == {"pcb1", "cashew", "ALL"}
    m = res["ALL"]["nominal"]["agent_rule"]
    assert m["n"] == 48 and m["n_pos"] == 16
    assert 0.0 <= m["auroc"] <= 1.0
    assert m["escape_rate"] + m["recall"] == pytest.approx(1.0)
    assert "ci95" in m and len(m["ci95"]["auroc"]) == 2
    assert m["tool_usage"]["n_runs"] == 48
    assert 0.0 <= m["evidence_changed_action_rate"] <= 1.0
    json.dumps(summary, allow_nan=False)  # strict JSON


def test_make_figures_smoke(tmp_path):
    from scripts import evaluate as ev
    from scripts import make_figures as mf

    rows = _synthetic_rows(12)
    items = tmp_path / "eval_items.csv"
    ev.append_rows(items, rows)
    summary = ev.summarize(ev.read_rows(items), pixel={"pcb1": {"fast": {"pooled": 0.9, "mean_per_image": 0.88, "n_images": 3}}}, n_boot=20)
    spath = tmp_path / "eval_summary.json"
    spath.write_text(json.dumps(summary), encoding="utf-8")
    out = tmp_path / "figs"
    written = mf.make_all(items, spath, out)
    assert len(written) >= 6
    for p in written:
        assert p.is_file() and p.stat().st_size > 5000
    import cv2

    img = cv2.imdecode(np.fromfile(str(written[0]), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 1800 and img.shape[0] == 1200
