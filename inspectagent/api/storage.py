"""Persistence for inspection traces, artifacts and human-review items.

Two interchangeable backends implement the :class:`Storage` protocol:

* :class:`LocalStorage` — JSON / JPEG files under ``settings.local_storage_dir``
  (thread-safe, atomic writes, no external services). Layout::

      traces/<trace_id>.json        full trace (Trace.to_dict() + API metadata)
      traces/_index.jsonl           append-only trace summaries (listing)
      artifacts/<trace_id>/<art_id>.jpg
      reviews/<review_id>.json      review item incl. hash-chained audit log

* :class:`AwsStorage` — S3 (``traces/{id}.json``, ``artifacts/{trace_id}/{art_id}.jpg``) plus a
  DynamoDB **single table** with string keys ``pk`` / ``sk``:

  ==========================  ==========================  ===============================================
  pk                          sk                          content
  ==========================  ==========================  ===============================================
  ``TRACE#<trace_id>``        ``META``                    trace summary + S3 key
  ``REVIEW#<review_id>``      ``META``                    canonical review document (optimistic ``version``)
  ``REVIEW#<review_id>``      ``AUDIT#<seq:05d>``         immutable audit record (write-once)
  ``INDEX#TRACE``             ``<started_at>#<trace_id>`` listing item (newest first via ScanIndexForward=False)
  ``INDEX#REVIEW``            ``<created_at>#<review_id>`` listing item with a copy of the review document
  ==========================  ==========================  ===============================================

  Documents are stored as a JSON string attribute ``doc`` (no float/Decimal conversion issues). All AWS
  calls go through injected ``s3_client`` / ``ddb_client`` objects (low-level boto3 clients by default), so
  tests can stub them without network access.

Audit records of a review form a SHA-256 hash chain (``prev_hash`` -> ``hash``) so tampering with an
earlier decision is detectable with :func:`verify_audit`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger("inspectagent.api.storage")

__all__ = [
    "Storage",
    "StorageError",
    "NotFoundError",
    "ConflictError",
    "LocalStorage",
    "AwsStorage",
    "make_storage",
    "trace_summary",
    "append_audit",
    "verify_audit",
    "validate_id",
    "utc_now",
    "REVIEW_STATUSES",
]

REVIEW_STATUSES = ("pending", "resolved", "rescanned")
_MAX_ID_LEN = 128
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")
_FORBIDDEN_ID_CHARS = re.compile(r"[/\\\x00-\x1f\x7f]")
_GENESIS = "0" * 64


class StorageError(RuntimeError):
    """Backend failure (I/O, AWS error, invalid id)."""


class NotFoundError(StorageError, KeyError):
    """Requested object does not exist."""


class ConflictError(StorageError):
    """Concurrent modification or unexpected state (e.g. review no longer pending)."""


def utc_now() -> str:
    """ISO-8601 UTC timestamp with millisecond precision (lexicographically sortable)."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def validate_id(value: str, what: str = "id") -> str:
    """Return ``value`` if it is a usable object id (1-128 chars, no slashes/control chars, not '.'/'..')."""
    if not isinstance(value, str) or not value or len(value) > _MAX_ID_LEN:
        raise StorageError(f"invalid {what}: must be a non-empty string of at most {_MAX_ID_LEN} chars")
    if _FORBIDDEN_ID_CHARS.search(value) or value in (".", ".."):
        raise StorageError(f"invalid {what}: {value!r}")
    return value


def _fname(value: str) -> str:
    """Map an id to a filesystem-safe file stem (ids with unusual characters are hashed)."""
    if _SAFE_NAME.fullmatch(value) and value not in (".", ".."):
        return value
    return "x" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:40]


def _split_artifact_id(artifact_id: str) -> tuple[str, str]:
    """``"<trace_id>/<art_id>"`` -> (trace_id, art_id), both validated."""
    if not isinstance(artifact_id, str) or artifact_id.count("/") != 1:
        raise StorageError(f"invalid artifact id {artifact_id!r}: expected '<trace_id>/<artifact>'")
    trace_id, art = artifact_id.split("/")
    return validate_id(trace_id, "trace id"), validate_id(art, "artifact id")


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def append_audit(review: dict, event: dict) -> dict:
    """Append ``event`` to ``review["audit"]`` (in place) with a SHA-256 hash chain; returns the record."""
    audit = review.setdefault("audit", [])
    prev = audit[-1]["hash"] if audit else _GENESIS
    record = {k: v for k, v in event.items() if k not in ("hash", "prev_hash", "seq")}
    record.setdefault("ts", utc_now())
    record["seq"] = len(audit)
    record["prev_hash"] = prev
    record["hash"] = hashlib.sha256((prev + _canonical(record)).encode("utf-8")).hexdigest()
    audit.append(record)
    return record


def verify_audit(review: dict) -> bool:
    """True when the audit records of ``review`` form an unbroken, untampered hash chain."""
    prev = _GENESIS
    for i, rec in enumerate(review.get("audit") or []):
        body = {k: v for k, v in rec.items() if k != "hash"}
        if body.get("prev_hash") != prev or body.get("seq") != i:
            return False
        digest = hashlib.sha256((prev + _canonical(body)).encode("utf-8")).hexdigest()
        if digest != rec.get("hash"):
            return False
        prev = digest
    return True


def trace_summary(trace: dict) -> dict:
    """Compact listing row for a stored trace (works for Trace.to_dict() plus API metadata)."""
    verdict = trace.get("verdict") or {}
    totals = trace.get("totals") or {}
    request = trace.get("request") or {}
    gt = trace.get("ground_truth") or {}
    conditions = trace.get("conditions") or {}
    return {
        "trace_id": trace.get("trace_id"),
        "part_id": trace.get("part_id"),
        "category": trace.get("category"),
        "planner": trace.get("planner"),
        "started_at": trace.get("started_at"),
        "label": verdict.get("label"),
        "action": verdict.get("action"),
        "confidence": verdict.get("confidence"),
        "prob_anomalous": verdict.get("prob_anomalous"),
        "n_steps": totals.get("n_steps", len(trace.get("steps") or [])),
        "cycle_time_ms": totals.get("cycle_time_ms"),
        "preset": request.get("preset") or conditions.get("preset"),
        "source": request.get("source"),
        "ground_truth": gt.get("label") if isinstance(gt, dict) else None,
        "review_id": trace.get("review_ref"),
        "human_action": (trace.get("human_decision") or {}).get("action"),
        "links": trace.get("links") or {},
    }


def _merge_review(review: dict, patch: dict) -> dict:
    updated = copy.deepcopy(review)
    for k, v in patch.items():
        if k in ("review_id", "audit", "version"):
            continue
        updated[k] = v
    return updated


@runtime_checkable
class Storage(Protocol):
    """Persistence contract used by the web API (SPEC §5)."""

    kind: str

    def put_trace(self, trace: dict) -> None: ...

    def get_trace(self, trace_id: str) -> dict | None: ...

    def trace_exists(self, trace_id: str) -> bool: ...

    def list_traces(self, limit: int = 20) -> list[dict]: ...

    def put_artifact(self, artifact_id: str, data: bytes, content_type: str = "image/jpeg") -> str: ...

    def get_artifact(self, artifact_id: str) -> bytes | None: ...

    def put_review(self, review: dict) -> None: ...

    def get_review(self, review_id: str) -> dict | None: ...

    def list_reviews(self, status: str | None = "pending", limit: int = 100) -> list[dict]: ...

    def update_review(
        self, review_id: str, patch: dict, audit_event: dict | None = None, expect_status: str | None = None
    ) -> dict: ...

    def describe(self) -> dict: ...


# --------------------------------------------------------------------------------------------- local


class LocalStorage:
    """JSON-file storage (thread-safe; directories are created lazily on first write)."""

    kind = "local"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._lock = threading.RLock()

    # paths ------------------------------------------------------------------------------------
    @property
    def _traces(self) -> Path:
        return self.root / "traces"

    @property
    def _reviews(self) -> Path:
        return self.root / "reviews"

    @property
    def _artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def _index(self) -> Path:
        return self._traces / "_index.jsonl"

    def _trace_path(self, trace_id: str) -> Path:
        return self._traces / f"{_fname(validate_id(trace_id, 'trace id'))}.json"

    def _review_path(self, review_id: str) -> Path:
        return self._reviews / f"{_fname(validate_id(review_id, 'review id'))}.json"

    def _artifact_path(self, artifact_id: str) -> Path:
        trace_id, art = _split_artifact_id(artifact_id)
        return self._artifacts / _fname(trace_id) / f"{_fname(art)}.jpg"

    # io helpers -------------------------------------------------------------------------------
    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise StorageError(f"cannot read {path.name}: {exc}") from exc

    @staticmethod
    def _dump(obj: dict) -> bytes:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")

    # traces -----------------------------------------------------------------------------------
    def put_trace(self, trace: dict) -> None:
        trace_id = validate_id(str(trace.get("trace_id") or ""), "trace id")
        with self._lock:
            self._atomic_write(self._trace_path(trace_id), self._dump(trace))
            self._index.parent.mkdir(parents=True, exist_ok=True)
            with open(self._index, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(trace_summary(trace), ensure_ascii=False, default=str) + "\n")

    def get_trace(self, trace_id: str) -> dict | None:
        with self._lock:
            return self._read_json(self._trace_path(trace_id))

    def trace_exists(self, trace_id: str) -> bool:
        with self._lock:
            return self._trace_path(trace_id).exists()

    def list_traces(self, limit: int = 20) -> list[dict]:
        limit = max(0, int(limit))
        with self._lock:
            try:
                lines = self._index.read_text(encoding="utf-8").splitlines()
            except FileNotFoundError:
                return []
        first_seen: dict[str, int] = {}
        latest: dict[str, dict] = {}
        for seq, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log.warning(json.dumps({"evt": "storage_index_corrupt_line", "seq": seq}))
                continue
            tid = row.get("trace_id")
            if not tid:
                continue
            first_seen.setdefault(tid, seq)
            latest[tid] = row
        order = sorted(latest, key=lambda t: (str(latest[t].get("started_at") or ""), first_seen[t]), reverse=True)
        return [latest[t] for t in order[:limit]]

    # artifacts --------------------------------------------------------------------------------
    def put_artifact(self, artifact_id: str, data: bytes, content_type: str = "image/jpeg") -> str:
        path = self._artifact_path(artifact_id)
        with self._lock:
            self._atomic_write(path, bytes(data))
        return str(path)

    def get_artifact(self, artifact_id: str) -> bytes | None:
        path = self._artifact_path(artifact_id)
        with self._lock:
            try:
                return path.read_bytes()
            except FileNotFoundError:
                return None

    # reviews ----------------------------------------------------------------------------------
    def put_review(self, review: dict) -> None:
        review_id = validate_id(str(review.get("review_id") or ""), "review id")
        with self._lock:
            self._atomic_write(self._review_path(review_id), self._dump(review))

    def get_review(self, review_id: str) -> dict | None:
        with self._lock:
            return self._read_json(self._review_path(review_id))

    def list_reviews(self, status: str | None = "pending", limit: int = 100) -> list[dict]:
        limit = max(0, int(limit))
        items: list[dict] = []
        with self._lock:
            if not self._reviews.exists():
                return []
            for path in self._reviews.glob("*.json"):
                try:
                    item = self._read_json(path)
                except StorageError as exc:
                    log.warning(json.dumps({"evt": "storage_review_unreadable", "file": path.name, "error": str(exc)}))
                    continue
                if item is None:
                    continue
                if status not in (None, "all") and item.get("status") != status:
                    continue
                items.append(item)
        items.sort(key=lambda r: (str(r.get("created_at") or ""), str(r.get("review_id"))), reverse=True)
        return items[:limit]

    def update_review(
        self, review_id: str, patch: dict, audit_event: dict | None = None, expect_status: str | None = None
    ) -> dict:
        with self._lock:
            current = self.get_review(review_id)
            if current is None:
                raise NotFoundError(f"review {review_id!r} not found")
            if expect_status is not None and current.get("status") != expect_status:
                raise ConflictError(
                    f"review {review_id!r} is {current.get('status')!r}, expected {expect_status!r}"
                )
            updated = _merge_review(current, patch)
            updated["updated_at"] = utc_now()
            updated["version"] = int(current.get("version", 0)) + 1
            if audit_event:
                append_audit(updated, audit_event)
            self._atomic_write(self._review_path(review_id), self._dump(updated))
            return updated

    def describe(self) -> dict:
        return {"kind": self.kind, "root": str(self.root)}


# ----------------------------------------------------------------------------------------------- aws


def _client_error_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return str((response.get("Error") or {}).get("Code") or "")
    return ""


class AwsStorage:
    """S3 + DynamoDB (single table, ``pk``/``sk`` string keys) storage. See module docstring for the layout."""

    kind = "aws"

    def __init__(
        self,
        bucket: str,
        table: str,
        region: str | None = None,
        s3_client: Any | None = None,
        ddb_client: Any | None = None,
        prefix: str = "",
    ) -> None:
        if not bucket or not table:
            raise StorageError("AwsStorage requires both an S3 bucket and a DynamoDB table name")
        self.bucket = bucket
        self.table = table
        self.region = region
        self.prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
        self._s3 = s3_client
        self._ddb = ddb_client
        self._client_lock = threading.Lock()

    # clients ----------------------------------------------------------------------------------
    @property
    def s3(self) -> Any:
        if self._s3 is None:
            with self._client_lock:
                if self._s3 is None:
                    import boto3

                    self._s3 = boto3.client("s3", region_name=self.region)
        return self._s3

    @property
    def ddb(self) -> Any:
        if self._ddb is None:
            with self._client_lock:
                if self._ddb is None:
                    import boto3

                    self._ddb = boto3.client("dynamodb", region_name=self.region)
        return self._ddb

    # keys -------------------------------------------------------------------------------------
    def _trace_key(self, trace_id: str) -> str:
        return f"{self.prefix}traces/{validate_id(trace_id, 'trace id')}.json"

    def _artifact_key(self, artifact_id: str) -> str:
        trace_id, art = _split_artifact_id(artifact_id)
        return f"{self.prefix}artifacts/{trace_id}/{art}.jpg"

    @staticmethod
    def _s(value: Any) -> dict:
        return {"S": str(value)}

    # s3 helpers -------------------------------------------------------------------------------
    def _s3_get(self, key: str) -> bytes | None:
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # botocore.exceptions.ClientError (kept generic for stubs)
            if _client_error_code(exc) in ("NoSuchKey", "404", "NotFound"):
                return None
            raise StorageError(f"S3 get_object {key} failed: {exc}") from exc
        body = resp["Body"]
        return body.read() if hasattr(body, "read") else bytes(body)

    def _s3_put(self, key: str, data: bytes, content_type: str) -> None:
        try:
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        except Exception as exc:
            raise StorageError(f"S3 put_object {key} failed: {exc}") from exc

    # ddb helpers ------------------------------------------------------------------------------
    def _ddb_call(self, op: str, **kwargs: Any) -> dict:
        try:
            return getattr(self.ddb, op)(TableName=self.table, **kwargs)
        except Exception as exc:
            if _client_error_code(exc) == "ConditionalCheckFailedException":
                raise ConflictError(f"DynamoDB {op}: concurrent modification") from exc
            raise StorageError(f"DynamoDB {op} failed: {exc}") from exc

    def _query_all(self, pk: str, limit: int, newest_first: bool = True, page_cap: int = 20) -> list[dict]:
        items: list[dict] = []
        kwargs: dict[str, Any] = {
            "KeyConditionExpression": "pk = :p",
            "ExpressionAttributeValues": {":p": self._s(pk)},
            "ScanIndexForward": not newest_first,
        }
        for _ in range(page_cap):
            if limit:
                kwargs["Limit"] = max(1, limit - len(items))
            resp = self._ddb_call("query", **kwargs)
            items.extend(resp.get("Items") or [])
            lek = resp.get("LastEvaluatedKey")
            if not lek or (limit and len(items) >= limit):
                break
            kwargs["ExclusiveStartKey"] = lek
        return items[:limit] if limit else items

    @staticmethod
    def _doc(item: dict | None) -> dict | None:
        if not item or "doc" not in item:
            return None
        return json.loads(item["doc"]["S"])

    # traces -----------------------------------------------------------------------------------
    def put_trace(self, trace: dict) -> None:
        trace_id = validate_id(str(trace.get("trace_id") or ""), "trace id")
        key = self._trace_key(trace_id)
        self._s3_put(key, json.dumps(trace, ensure_ascii=False, default=str).encode("utf-8"), "application/json")
        summary = trace_summary(trace)
        doc = json.dumps(summary, ensure_ascii=False, default=str)
        started = str(summary.get("started_at") or utc_now())
        self._ddb_call(
            "put_item",
            Item={
                "pk": self._s(f"TRACE#{trace_id}"),
                "sk": self._s("META"),
                "trace_id": self._s(trace_id),
                "started_at": self._s(started),
                "s3_key": self._s(key),
                "doc": self._s(doc),
            },
        )
        self._ddb_call(
            "put_item",
            Item={
                "pk": self._s("INDEX#TRACE"),
                "sk": self._s(f"{started}#{trace_id}"),
                "trace_id": self._s(trace_id),
                "doc": self._s(doc),
            },
        )

    def get_trace(self, trace_id: str) -> dict | None:
        raw = self._s3_get(self._trace_key(trace_id))
        return None if raw is None else json.loads(raw.decode("utf-8"))

    def trace_exists(self, trace_id: str) -> bool:
        resp = self._ddb_call(
            "get_item",
            Key={"pk": self._s(f"TRACE#{validate_id(trace_id, 'trace id')}"), "sk": self._s("META")},
            ProjectionExpression="pk",
        )
        return bool(resp.get("Item"))

    def list_traces(self, limit: int = 20) -> list[dict]:
        limit = max(0, int(limit))
        if limit == 0:
            return []
        items = self._query_all("INDEX#TRACE", limit)
        out: list[dict] = []
        seen: set[str] = set()
        for it in items:
            doc = self._doc(it)
            if doc and doc.get("trace_id") not in seen:
                seen.add(doc.get("trace_id"))
                out.append(doc)
        return out

    # artifacts --------------------------------------------------------------------------------
    def put_artifact(self, artifact_id: str, data: bytes, content_type: str = "image/jpeg") -> str:
        key = self._artifact_key(artifact_id)
        self._s3_put(key, bytes(data), content_type)
        return f"s3://{self.bucket}/{key}"

    def get_artifact(self, artifact_id: str) -> bytes | None:
        return self._s3_get(self._artifact_key(artifact_id))

    # reviews ----------------------------------------------------------------------------------
    def _review_items(self, review: dict, version: int) -> tuple[dict, dict]:
        rid = review["review_id"]
        doc = json.dumps(review, ensure_ascii=False, default=str)
        created = str(review.get("created_at") or utc_now())
        status = str(review.get("status") or "pending")
        canonical = {
            "pk": self._s(f"REVIEW#{rid}"),
            "sk": self._s("META"),
            "review_id": self._s(rid),
            "status": self._s(status),
            "created_at": self._s(created),
            "version": {"N": str(version)},
            "doc": self._s(doc),
        }
        index = {
            "pk": self._s("INDEX#REVIEW"),
            "sk": self._s(f"{created}#{rid}"),
            "review_id": self._s(rid),
            "status": self._s(status),
            "doc": self._s(doc),
        }
        return canonical, index

    def _put_audit_records(self, review_id: str, records: list[dict]) -> None:
        for rec in records:
            self._ddb_call(
                "put_item",
                Item={
                    "pk": self._s(f"REVIEW#{review_id}"),
                    "sk": self._s(f"AUDIT#{int(rec.get('seq', 0)):05d}"),
                    "ts": self._s(rec.get("ts") or utc_now()),
                    "hash": self._s(rec.get("hash") or ""),
                    "doc": self._s(json.dumps(rec, ensure_ascii=False, default=str)),
                },
                ConditionExpression="attribute_not_exists(pk)",
            )

    def put_review(self, review: dict) -> None:
        rid = validate_id(str(review.get("review_id") or ""), "review id")
        review = dict(review)
        review["version"] = int(review.get("version") or 1)
        canonical, index = self._review_items(review, review["version"])
        self._ddb_call("put_item", Item=canonical, ConditionExpression="attribute_not_exists(pk)")
        self._ddb_call("put_item", Item=index)
        self._put_audit_records(rid, list(review.get("audit") or []))

    def get_review(self, review_id: str) -> dict | None:
        resp = self._ddb_call(
            "get_item",
            Key={"pk": self._s(f"REVIEW#{validate_id(review_id, 'review id')}"), "sk": self._s("META")},
            ConsistentRead=True,
        )
        return self._doc(resp.get("Item"))

    def list_reviews(self, status: str | None = "pending", limit: int = 100) -> list[dict]:
        limit = max(0, int(limit))
        if limit == 0:
            return []
        # Filtering happens client-side so a status filter never truncates pagination (demo-scale queue).
        items = self._query_all("INDEX#REVIEW", 0, page_cap=10)
        out: list[dict] = []
        for it in items:
            doc = self._doc(it)
            if doc is None:
                continue
            if status not in (None, "all") and doc.get("status") != status:
                continue
            out.append(doc)
            if len(out) >= limit:
                break
        return out

    def update_review(
        self, review_id: str, patch: dict, audit_event: dict | None = None, expect_status: str | None = None
    ) -> dict:
        current = self.get_review(review_id)
        if current is None:
            raise NotFoundError(f"review {review_id!r} not found")
        if expect_status is not None and current.get("status") != expect_status:
            raise ConflictError(f"review {review_id!r} is {current.get('status')!r}, expected {expect_status!r}")
        old_version = int(current.get("version") or 1)
        updated = _merge_review(current, patch)
        updated["updated_at"] = utc_now()
        updated["version"] = old_version + 1
        new_records: list[dict] = []
        if audit_event:
            new_records.append(append_audit(updated, audit_event))
        canonical, index = self._review_items(updated, updated["version"])
        self._ddb_call(
            "put_item",
            Item=canonical,
            ConditionExpression="#v = :v",
            ExpressionAttributeNames={"#v": "version"},
            ExpressionAttributeValues={":v": {"N": str(old_version)}},
        )
        self._ddb_call("put_item", Item=index)
        self._put_audit_records(review_id, new_records)
        return updated

    def describe(self) -> dict:
        return {"kind": self.kind, "bucket": self.bucket, "table": self.table, "region": self.region}


# ------------------------------------------------------------------------------------------- factory


def make_storage(settings: Any) -> "LocalStorage | AwsStorage":
    """Build the storage backend selected by ``settings.storage`` (``local`` | ``aws``).

    ``aws`` without bucket/table falls back to local storage (logged). Inside AWS Lambda the default local
    directory (read-only image filesystem) is replaced by ``/tmp/inspectagent/runs`` unless the directory
    was configured explicitly via ``INSPECTAGENT_LOCAL_STORAGE_DIR``.
    """
    kind = str(getattr(settings, "storage", "local") or "local").lower()
    if kind == "aws":
        bucket = getattr(settings, "s3_bucket", "")
        table = getattr(settings, "ddb_table", "")
        if bucket and table:
            return AwsStorage(bucket, table, region=getattr(settings, "aws_region", None))
        log.warning(json.dumps({"evt": "storage_fallback", "reason": "aws storage selected but bucket/table unset"}))
    root = Path(getattr(settings, "local_storage_dir"))
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME") and not os.environ.get("INSPECTAGENT_LOCAL_STORAGE_DIR"):
        root = Path(tempfile.gettempdir()) / "inspectagent" / "runs"
    return LocalStorage(root)
