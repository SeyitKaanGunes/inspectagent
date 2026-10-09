"""Tests for inspectagent.api.storage (LocalStorage + AwsStorage with stubbed boto3 clients)."""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from inspectagent.api.storage import (
    AwsStorage,
    ConflictError,
    LocalStorage,
    NotFoundError,
    StorageError,
    append_audit,
    make_storage,
    trace_summary,
    verify_audit,
)
from inspectagent.config import Settings


def _trace(tid: str, started: str, label: str = "PASS", **extra) -> dict:
    t = {
        "trace_id": tid,
        "part_id": f"part-{tid}",
        "category": "pcb1",
        "planner": "rule",
        "started_at": started,
        "steps": [{"step": 1, "decision": {"tool": "capture_image"}, "artifacts": ["art_1"]}],
        "verdict": {"label": label, "action": "route_pass" if label == "PASS" else "hold_for_review", "confidence": 0.9, "prob_anomalous": 0.05},
        "totals": {"n_steps": 1, "cycle_time_ms": 210.5},
        "request": {"preset": "nominal", "source": "sample"},
    }
    t.update(extra)
    return t


def _review(rid: str, created: str, status: str = "pending") -> dict:
    r = {"review_id": rid, "trace_id": f"t-{rid}", "status": status, "created_at": created, "kind": "needs_review", "version": 1, "audit": []}
    append_audit(r, {"event": "created", "actor": "agent:rule", "ts": created})
    return r


# --------------------------------------------------------------------------------------------- audit


def test_audit_chain_detects_tampering() -> None:
    r = _review("rv-1", "2026-01-01T00:00:00.000Z")
    append_audit(r, {"event": "decision", "actor": "alice", "decision": "approve_fail"})
    assert [a["seq"] for a in r["audit"]] == [0, 1]
    assert r["audit"][1]["prev_hash"] == r["audit"][0]["hash"]
    assert verify_audit(r)
    r["audit"][0]["actor"] = "mallory"
    assert not verify_audit(r)


def test_trace_summary_fields() -> None:
    s = trace_summary(_trace("t1", "2026-01-01T00:00:00Z", human_decision={"action": "route_scrap"}, review_ref="rv-t1"))
    assert s["trace_id"] == "t1" and s["label"] == "PASS" and s["n_steps"] == 1
    assert s["preset"] == "nominal" and s["human_action"] == "route_scrap" and s["review_id"] == "rv-t1"


# --------------------------------------------------------------------------------------------- local


def test_local_traces_roundtrip_and_listing(tmp_path: Path) -> None:
    st = LocalStorage(tmp_path / "runs")
    assert st.list_traces() == []
    assert not (tmp_path / "runs").exists()  # lazy: nothing created until the first write
    st.put_trace(_trace("t1", "2026-01-01T00:00:01Z"))
    st.put_trace(_trace("t2", "2026-01-01T00:00:03Z", "NEEDS_REVIEW"))
    st.put_trace(_trace("t3", "2026-01-01T00:00:02Z"))
    st.put_trace(_trace("t1", "2026-01-01T00:00:01Z", "FAIL"))  # overwrite -> listed once, latest summary
    assert st.trace_exists("t1") and not st.trace_exists("nope")
    assert st.get_trace("t1")["verdict"]["label"] == "FAIL"
    assert st.get_trace("missing") is None
    rows = st.list_traces(limit=10)
    assert [r["trace_id"] for r in rows] == ["t2", "t3", "t1"]
    assert rows[2]["label"] == "FAIL"
    assert [r["trace_id"] for r in st.list_traces(limit=2)] == ["t2", "t3"]


def test_local_artifacts_and_id_safety(tmp_path: Path) -> None:
    st = LocalStorage(tmp_path)
    st.put_artifact("t1/art_1", b"\xff\xd8\xffjpeg")
    assert st.get_artifact("t1/art_1") == b"\xff\xd8\xffjpeg"
    assert st.get_artifact("t1/art_9") is None
    # ids with characters that are illegal in Windows file names are hashed, not rejected
    st.put_trace(_trace("weird:id*1", "2026-01-01T00:00:00Z"))
    assert st.get_trace("weird:id*1")["trace_id"] == "weird:id*1"
    for bad in ("../escape", "a/b", "..", "", "x" * 200, "a\x00b"):
        with pytest.raises(StorageError):
            st.get_trace(bad)
    for bad in ("t1", "t1/../x", "../t1/art", "t1/a/b"):
        with pytest.raises(StorageError):
            st.put_artifact(bad, b"x")
    assert not any(p.name == "escape.json" for p in tmp_path.rglob("*"))


def test_local_reviews_update_and_conflicts(tmp_path: Path) -> None:
    st = LocalStorage(tmp_path)
    st.put_review(_review("rv-a", "2026-01-01T00:00:01Z"))
    st.put_review(_review("rv-b", "2026-01-01T00:00:02Z"))
    st.put_review(_review("rv-c", "2026-01-01T00:00:03Z", status="resolved"))
    assert [r["review_id"] for r in st.list_reviews("pending")] == ["rv-b", "rv-a"]
    assert [r["review_id"] for r in st.list_reviews("all")] == ["rv-c", "rv-b", "rv-a"]
    assert len(st.list_reviews(None, limit=1)) == 1

    upd = st.update_review("rv-a", {"status": "resolved", "final_action": "route_scrap", "audit": "ignored", "review_id": "x"},
                           audit_event={"event": "decision", "actor": "bob"}, expect_status="pending")
    assert upd["status"] == "resolved" and upd["review_id"] == "rv-a" and upd["version"] == 2
    assert len(upd["audit"]) == 2 and verify_audit(upd)
    assert st.get_review("rv-a")["final_action"] == "route_scrap"
    with pytest.raises(ConflictError):
        st.update_review("rv-a", {"status": "resolved"}, expect_status="pending")
    with pytest.raises(NotFoundError):
        st.update_review("rv-zzz", {})


def test_local_concurrent_updates_are_serialized(tmp_path: Path) -> None:
    st = LocalStorage(tmp_path)
    st.put_review(_review("rv-x", "2026-01-01T00:00:00Z"))
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            for j in range(5):
                st.update_review("rv-x", {"note": f"{i}-{j}"}, audit_event={"event": "note", "actor": f"w{i}"})
        except Exception as exc:  # pragma: no cover - surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    final = st.get_review("rv-x")
    assert final["version"] == 1 + 30 and len(final["audit"]) == 31 and verify_audit(final)


def test_local_index_tolerates_corrupt_lines(tmp_path: Path) -> None:
    st = LocalStorage(tmp_path)
    st.put_trace(_trace("t1", "2026-01-01T00:00:00Z"))
    with open(tmp_path / "traces" / "_index.jsonl", "a", encoding="utf-8") as fh:
        fh.write("{not json\n\n")
    st.put_trace(_trace("t2", "2026-01-01T00:00:05Z"))
    assert [r["trace_id"] for r in st.list_traces()] == ["t2", "t1"]


# ----------------------------------------------------------------------------------------------- aws


def _client_error(code: str, op: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, op)


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, str]] = {}
        self.fail_with: str | None = None

    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str) -> dict:
        if self.fail_with:
            raise _client_error(self.fail_with, "PutObject")
        self.objects[(Bucket, Key)] = (bytes(Body), ContentType)
        return {"ETag": "x"}

    def get_object(self, Bucket: str, Key: str) -> dict:
        if self.fail_with:
            raise _client_error(self.fail_with, "GetObject")
        if (Bucket, Key) not in self.objects:
            raise _client_error("NoSuchKey", "GetObject")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)][0])}


class FakeDynamo:
    """Minimal low-level DynamoDB client: put/get/query with the condition forms AwsStorage uses."""

    def __init__(self, page_size: int = 2) -> None:
        self.items: dict[tuple[str, str], dict] = {}
        self.page_size = page_size
        self.calls: list[tuple[str, dict]] = []
        self.before_put = None

    @staticmethod
    def _key(item: dict) -> tuple[str, str]:
        return item["pk"]["S"], item["sk"]["S"]

    def put_item(self, TableName: str, Item: dict, ConditionExpression: str | None = None,
                 ExpressionAttributeNames: dict | None = None, ExpressionAttributeValues: dict | None = None) -> dict:
        self.calls.append(("put_item", {"TableName": TableName, "Item": Item, "ConditionExpression": ConditionExpression}))
        if self.before_put:
            self.before_put(Item)
        key = self._key(Item)
        existing = self.items.get(key)
        if ConditionExpression == "attribute_not_exists(pk)":
            if existing is not None:
                raise _client_error("ConditionalCheckFailedException", "PutItem")
        elif ConditionExpression == "#v = :v":
            attr = ExpressionAttributeNames["#v"]
            if existing is None or existing.get(attr) != ExpressionAttributeValues[":v"]:
                raise _client_error("ConditionalCheckFailedException", "PutItem")
        elif ConditionExpression is not None:
            raise AssertionError(f"unexpected condition {ConditionExpression}")
        self.items[key] = json.loads(json.dumps(Item))
        return {}

    def get_item(self, TableName: str, Key: dict, ConsistentRead: bool = False, ProjectionExpression: str | None = None) -> dict:
        self.calls.append(("get_item", {"Key": Key}))
        item = self.items.get(self._key(Key))
        return {"Item": json.loads(json.dumps(item))} if item else {}

    def query(self, TableName: str, KeyConditionExpression: str, ExpressionAttributeValues: dict,
              ScanIndexForward: bool = True, Limit: int | None = None, ExclusiveStartKey: dict | None = None) -> dict:
        assert KeyConditionExpression == "pk = :p"
        pk = ExpressionAttributeValues[":p"]["S"]
        rows = sorted((v for (p, _), v in self.items.items() if p == pk), key=lambda it: it["sk"]["S"], reverse=not ScanIndexForward)
        if ExclusiveStartKey:
            idx = [self._key(r) for r in rows].index(self._key(ExclusiveStartKey))
            rows = rows[idx + 1 :]
        page = min(self.page_size, Limit or self.page_size)
        out = rows[:page]
        resp: dict = {"Items": json.loads(json.dumps(out)), "Count": len(out)}
        if len(rows) > page:
            resp["LastEvaluatedKey"] = {"pk": out[-1]["pk"], "sk": out[-1]["sk"]}
        return resp


@pytest.fixture()
def aws() -> tuple[AwsStorage, FakeS3, FakeDynamo]:
    s3, ddb = FakeS3(), FakeDynamo()
    return AwsStorage("bucket-1", "table-1", region="eu-west-1", s3_client=s3, ddb_client=ddb), s3, ddb


def test_aws_requires_names() -> None:
    with pytest.raises(StorageError):
        AwsStorage("", "t")


def test_aws_traces(aws) -> None:
    st, s3, ddb = aws
    for i, ts in enumerate(["2026-01-01T00:00:01Z", "2026-01-01T00:00:05Z", "2026-01-01T00:00:03Z"]):
        st.put_trace(_trace(f"t{i}", ts))
    body, ctype = s3.objects[("bucket-1", "traces/t0.json")]
    assert ctype == "application/json" and json.loads(body)["trace_id"] == "t0"
    meta = ddb.items[("TRACE#t0", "META")]
    assert meta["s3_key"]["S"] == "traces/t0.json" and json.loads(meta["doc"]["S"])["label"] == "PASS"
    assert ("INDEX#TRACE", "2026-01-01T00:00:01Z#t0") in ddb.items
    assert st.trace_exists("t1") and not st.trace_exists("nope")
    assert st.get_trace("t2")["started_at"] == "2026-01-01T00:00:03Z"
    assert st.get_trace("nope") is None
    # pagination (page size 2) + newest first
    assert [r["trace_id"] for r in st.list_traces(limit=10)] == ["t1", "t2", "t0"]
    assert [r["trace_id"] for r in st.list_traces(limit=2)] == ["t1", "t2"]
    assert st.list_traces(limit=0) == []


def test_aws_artifacts_and_errors(aws) -> None:
    st, s3, _ = aws
    uri = st.put_artifact("t1/art_1", b"jpeg-bytes")
    assert uri == "s3://bucket-1/artifacts/t1/art_1.jpg"
    assert s3.objects[("bucket-1", "artifacts/t1/art_1.jpg")] == (b"jpeg-bytes", "image/jpeg")
    assert st.get_artifact("t1/art_1") == b"jpeg-bytes"
    assert st.get_artifact("t1/none") is None
    s3.fail_with = "AccessDenied"
    with pytest.raises(StorageError):
        st.get_artifact("t1/art_1")
    with pytest.raises(StorageError):
        st.put_artifact("t1/art_2", b"x")
    with pytest.raises(StorageError):
        st.put_artifact("../evil", b"x")


def test_aws_reviews_flow(aws) -> None:
    st, _, ddb = aws
    st.put_review(_review("rv-1", "2026-01-01T00:00:01Z"))
    st.put_review(_review("rv-2", "2026-01-01T00:00:02Z"))
    st.put_review(_review("rv-3", "2026-01-01T00:00:03Z", status="resolved"))
    assert ("REVIEW#rv-1", "AUDIT#00000") in ddb.items
    with pytest.raises(ConflictError):
        st.put_review(_review("rv-1", "2026-01-01T00:00:01Z"))
    assert st.get_review("rv-2")["status"] == "pending"
    assert st.get_review("nope") is None
    assert [r["review_id"] for r in st.list_reviews("pending")] == ["rv-2", "rv-1"]
    assert [r["review_id"] for r in st.list_reviews("all", limit=2)] == ["rv-3", "rv-2"]

    upd = st.update_review("rv-1", {"status": "resolved", "final_action": "route_pass"},
                           audit_event={"event": "decision", "actor": "carol"}, expect_status="pending")
    assert upd["version"] == 2 and verify_audit(upd)
    canonical = ddb.items[("REVIEW#rv-1", "META")]
    assert canonical["version"] == {"N": "2"} and canonical["status"]["S"] == "resolved"
    assert json.loads(ddb.items[("INDEX#REVIEW", "2026-01-01T00:00:01Z#rv-1")]["doc"]["S"])["status"] == "resolved"
    audit_item = ddb.items[("REVIEW#rv-1", "AUDIT#00001")]
    assert json.loads(audit_item["doc"]["S"])["actor"] == "carol"
    assert [r["review_id"] for r in st.list_reviews("pending")] == ["rv-2"]
    with pytest.raises(ConflictError):
        st.update_review("rv-1", {}, expect_status="pending")
    with pytest.raises(NotFoundError):
        st.update_review("rv-404", {})


def test_aws_optimistic_locking(aws) -> None:
    st, _, ddb = aws
    st.put_review(_review("rv-1", "2026-01-01T00:00:01Z"))

    def concurrent_writer(item: dict) -> None:
        # Another writer bumps the version between our read and our conditional write.
        if item["pk"]["S"] == "REVIEW#rv-1" and item["sk"]["S"] == "META" and item["version"]["N"] == "2":
            ddb.items[("REVIEW#rv-1", "META")]["version"] = {"N": "7"}

    ddb.before_put = concurrent_writer
    with pytest.raises(ConflictError):
        st.update_review("rv-1", {"status": "resolved"}, audit_event={"event": "decision", "actor": "x"})


def test_aws_calls_match_botocore_models(monkeypatch) -> None:
    """Validate request shapes against the real botocore service models (Stubber, no network)."""
    import boto3
    from botocore.stub import ANY, Stubber

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    s3 = boto3.client("s3", region_name="us-east-1")
    ddb = boto3.client("dynamodb", region_name="us-east-1")
    st = AwsStorage("bkt", "tbl", s3_client=s3, ddb_client=ddb)
    with Stubber(s3) as s3s, Stubber(ddb) as ds:
        s3s.add_response("put_object", {}, {"Bucket": "bkt", "Key": "traces/t1.json", "Body": ANY, "ContentType": "application/json"})
        ds.add_response("put_item", {}, {"TableName": "tbl", "Item": ANY})
        ds.add_response("put_item", {}, {"TableName": "tbl", "Item": ANY})
        st.put_trace(_trace("t1", "2026-01-01T00:00:00Z"))
        ds.add_response("query", {"Items": [], "Count": 0}, {
            "TableName": "tbl", "KeyConditionExpression": "pk = :p",
            "ExpressionAttributeValues": {":p": {"S": "INDEX#TRACE"}}, "ScanIndexForward": False, "Limit": 5})
        assert st.list_traces(limit=5) == []
        review = _review("rv-1", "2026-01-01T00:00:00Z")
        ds.add_response("put_item", {}, {"TableName": "tbl", "Item": ANY, "ConditionExpression": "attribute_not_exists(pk)"})
        ds.add_response("put_item", {}, {"TableName": "tbl", "Item": ANY})
        ds.add_response("put_item", {}, {"TableName": "tbl", "Item": ANY, "ConditionExpression": "attribute_not_exists(pk)"})
        st.put_review(review)
        stored = dict(review, version=1)
        ds.add_response("get_item", {"Item": {"pk": {"S": "REVIEW#rv-1"}, "sk": {"S": "META"}, "doc": {"S": json.dumps(stored)}}},
                        {"TableName": "tbl", "Key": {"pk": {"S": "REVIEW#rv-1"}, "sk": {"S": "META"}}, "ConsistentRead": True})
        ds.add_response("put_item", {}, {"TableName": "tbl", "Item": ANY, "ConditionExpression": "#v = :v",
                                         "ExpressionAttributeNames": {"#v": "version"}, "ExpressionAttributeValues": {":v": {"N": "1"}}})
        ds.add_response("put_item", {}, {"TableName": "tbl", "Item": ANY})
        ds.add_response("put_item", {}, {"TableName": "tbl", "Item": ANY, "ConditionExpression": "attribute_not_exists(pk)"})
        st.update_review("rv-1", {"status": "resolved"}, audit_event={"event": "decision", "actor": "dan"}, expect_status="pending")
        s3s.assert_no_pending_responses()
        ds.assert_no_pending_responses()


# ------------------------------------------------------------------------------------------- factory


def test_make_storage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    monkeypatch.delenv("INSPECTAGENT_LOCAL_STORAGE_DIR", raising=False)
    local = make_storage(Settings(storage="local", local_storage_dir=tmp_path / "r"))
    assert isinstance(local, LocalStorage) and local.root == tmp_path / "r"
    aws = make_storage(Settings(storage="aws", s3_bucket="b", ddb_table="t", aws_region="eu-west-1"))
    assert isinstance(aws, AwsStorage) and aws.describe()["bucket"] == "b"
    fallback = make_storage(Settings(storage="aws", s3_bucket="", ddb_table="", local_storage_dir=tmp_path / "f"))
    assert isinstance(fallback, LocalStorage)
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "fn")
    in_lambda = make_storage(Settings(storage="local", local_storage_dir=tmp_path / "ro"))
    assert isinstance(in_lambda, LocalStorage) and in_lambda.root.name == "runs" and "inspectagent" in str(in_lambda.root)
