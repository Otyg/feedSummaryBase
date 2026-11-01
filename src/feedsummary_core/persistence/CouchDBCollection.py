# LICENSE HEADER MANAGED BY add-license-header
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Martin Vesterlund
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""A small MongoDB-style collection API on top of CouchDB's HTTP interface.

``CouchDBStore`` reuses the MongoDB store's business logic.  This module supplies
the subset of the pymongo collection API that logic relies on:

* one CouchDB database per collection (``<prefix>_<collection>``),
* single-document atomicity through ``_rev`` compare-and-swap retries,
* unique indexes through claim documents (``~uq~...``),
* Mongo filter semantics evaluated client-side, with CouchDB Mango used only to
  narrow the candidate set.

Top-level fields that start with ``_`` (other than ``_id``) are stored with a
``mongo`` prefix because CouchDB reserves them; ``_mongo_sort_ts`` therefore lives
in ``mongo_mongo_sort_ts``.  This matches ``mongo2couch`` migrated data.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import http.client
import json
import logging
import ssl
import threading
import time
import uuid
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import quote, urlencode, urlsplit

from feedsummary_core.persistence.MongoDBStore import DuplicateKeyError

logger = logging.getLogger(__name__)

_CLAIM_PREFIX = "~uq~"
_CLAIM_TTL_SECONDS = 30
_PAGE_SIZE = 1000
_KEYS_CHUNK = 200
_MAX_CAS_ATTEMPTS = 25
_RESERVED_PREFIXES = ("_design/", "_local/", _CLAIM_PREFIX)
_NUMBER_TYPES = (int, float)


class CouchDBError(Exception):
    """Raised when CouchDB returns an unexpected response."""


class _Conflict(Exception):
    """Internal signal: a document revision changed underneath us."""


# HTTP transport


class CouchDBClient:
    """Minimal CouchDB HTTP client (stdlib only, one pooled connection)."""

    def __init__(
        self,
        url: str = "http://localhost:5984",
        *,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 30.0,
        verify_tls: bool = True,
    ):
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"Invalid CouchDB url: {url!r}")
        self.url = url
        self._scheme = parts.scheme
        self._host = parts.hostname
        self._port = parts.port or (443 if parts.scheme == "https" else 80)
        self._base = parts.path.rstrip("/")
        self._timeout = float(timeout)
        self._verify_tls = bool(verify_tls)
        username = username if username is not None else parts.username
        password = password if password is not None else parts.password
        self._auth: Optional[str] = None
        if username:
            token = base64.b64encode(f"{username}:{password or ''}".encode()).decode()
            self._auth = f"Basic {token}"
        self._lock = threading.Lock()
        self._conn: Optional[http.client.HTTPConnection] = None

    def _connect(self) -> http.client.HTTPConnection:
        if self._scheme == "https":
            context = ssl.create_default_context()
            if not self._verify_tls:
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
            return http.client.HTTPSConnection(
                self._host, self._port, timeout=self._timeout, context=context
            )
        return http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Any]:
        target = self._base + path
        if params:
            target += "?" + urlencode(
                {
                    key: value if isinstance(value, str) else json.dumps(value)
                    for key, value in params.items()
                }
            )
        headers = {"Accept": "application/json"}
        if self._auth:
            headers["Authorization"] = self._auth
        payload = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            payload = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()
        with self._lock:
            for attempt in (0, 1):
                try:
                    if self._conn is None:
                        self._conn = self._connect()
                    self._conn.request(method, target, body=payload, headers=headers)
                    response = self._conn.getresponse()
                    data = response.read()
                    break
                except (OSError, http.client.HTTPException) as exc:
                    if self._conn is not None:
                        self._conn.close()
                        self._conn = None
                    if attempt:
                        raise CouchDBError(f"CouchDB request failed: {method} {path}: {exc}") from exc
        parsed: Any = None
        if data:
            try:
                parsed = json.loads(data)
            except ValueError:
                parsed = {"raw": data.decode("utf-8", "replace")}
        return response.status, parsed


# Mongo-style filter evaluation


def _get_path(doc: Any, path: str) -> Tuple[bool, Any]:
    current = doc
    for segment in path.split("."):
        if isinstance(current, dict):
            if segment not in current:
                return False, None
            current = current[segment]
        elif isinstance(current, list) and segment.isdigit():
            index = int(segment)
            if index >= len(current):
                return False, None
            current = current[index]
        else:
            return False, None
    return True, current


def _is_number(value: Any) -> bool:
    return isinstance(value, _NUMBER_TYPES) and not isinstance(value, bool)


def _comparable(left: Any, right: Any) -> bool:
    return (
        (_is_number(left) and _is_number(right))
        or (isinstance(left, str) and isinstance(right, str))
        or (isinstance(left, bool) and isinstance(right, bool))
    )


def _equals(found: bool, value: Any, expected: Any) -> bool:
    if expected is None:
        return (not found) or value is None
    if not found:
        return False
    if isinstance(value, list) and not isinstance(expected, list):
        return any(_equals(True, item, expected) for item in value)
    if isinstance(expected, bool) != isinstance(value, bool):
        return False
    return value == expected


def _candidates(found: bool, value: Any) -> Iterable[Any]:
    if isinstance(value, list):
        return value
    return [value] if found else []


def _operator_matches(found: bool, value: Any, operator: str, operand: Any) -> bool:
    if operator == "$eq":
        return _equals(found, value, operand)
    if operator == "$ne":
        return not _equals(found, value, operand)
    if operator == "$in":
        return any(_equals(found, value, option) for option in operand)
    if operator == "$nin":
        return not any(_equals(found, value, option) for option in operand)
    if operator == "$exists":
        return found == bool(operand)
    if operator == "$type":
        if operand == "number":
            return found and any(_is_number(item) for item in _candidates(found, value))
        raise NotImplementedError(f"$type {operand!r} is not supported")
    if operator in ("$gt", "$gte", "$lt", "$lte"):
        for item in _candidates(found, value):
            if not _comparable(item, operand):
                continue
            if (
                (operator == "$gt" and item > operand)
                or (operator == "$gte" and item >= operand)
                or (operator == "$lt" and item < operand)
                or (operator == "$lte" and item <= operand)
            ):
                return True
        return False
    raise NotImplementedError(f"Query operator {operator} is not supported")


def _matches(doc: Dict[str, Any], query: Optional[Dict[str, Any]]) -> bool:
    for key, condition in (query or {}).items():
        if key == "$and":
            if not all(_matches(doc, sub) for sub in condition):
                return False
        elif key == "$or":
            if not any(_matches(doc, sub) for sub in condition):
                return False
        elif key.startswith("$"):
            raise NotImplementedError(f"Query operator {key} is not supported")
        else:
            found, value = _get_path(doc, key)
            if isinstance(condition, dict) and condition and all(
                str(name).startswith("$") for name in condition
            ):
                if not all(
                    _operator_matches(found, value, name, operand)
                    for name, operand in condition.items()
                ):
                    return False
            elif not _equals(found, value, condition):
                return False
    return True


def _sort_key(value: Any) -> Tuple[int, Any]:
    if value is None:
        return (0, 0)
    if isinstance(value, bool):
        return (1, int(value))
    if _is_number(value):
        return (2, value)
    if isinstance(value, str):
        return (3, value)
    return (4, json.dumps(value, sort_keys=True, default=str))


def _normalize_sort(key_or_list: Any, direction: Optional[int]) -> List[Tuple[str, int]]:
    if isinstance(key_or_list, str):
        return [(key_or_list, 1 if direction is None else int(direction))]
    return [(str(name), int(order)) for name, order in key_or_list]


def _project(doc: Dict[str, Any], projection: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not projection:
        return doc
    include = [key for key, flag in projection.items() if flag and key != "_id"]
    exclude = [key for key, flag in projection.items() if not flag and key != "_id"]
    if include:
        result: Dict[str, Any] = {}
        if projection.get("_id", 1) and "_id" in doc:
            result["_id"] = doc["_id"]
        for path in include:
            found, value = _get_path(doc, path)
            if not found:
                continue
            target = result
            segments = path.split(".")
            for segment in segments[:-1]:
                target = target.setdefault(segment, {})
            target[segments[-1]] = value
        return result
    result = {key: value for key, value in doc.items() if key not in exclude}
    if not projection.get("_id", 1):
        result.pop("_id", None)
    return result


def _apply_update(
    doc: Dict[str, Any], update: Dict[str, Any], *, inserting: bool
) -> Dict[str, Any]:
    result = copy.deepcopy(doc)

    def container(path: str, create: bool) -> Tuple[Optional[Dict[str, Any]], str]:
        segments = path.split(".")
        target: Any = result
        for segment in segments[:-1]:
            if not isinstance(target, dict):
                return None, segments[-1]
            if segment not in target:
                if not create:
                    return None, segments[-1]
                target[segment] = {}
            target = target[segment]
        return (target if isinstance(target, dict) else None), segments[-1]

    for operator, fields in update.items():
        if operator == "$setOnInsert" and not inserting:
            continue
        if operator in ("$set", "$setOnInsert"):
            for path, value in fields.items():
                target, leaf = container(path, True)
                if target is not None:
                    target[leaf] = value
        elif operator == "$unset":
            for path in fields:
                target, leaf = container(path, False)
                if target is not None:
                    target.pop(leaf, None)
        elif operator == "$inc":
            for path, amount in fields.items():
                target, leaf = container(path, True)
                if target is not None:
                    target[leaf] = (target.get(leaf) or 0) + amount
        elif operator == "$max":
            for path, candidate in fields.items():
                target, leaf = container(path, True)
                if target is not None:
                    current = target.get(leaf)
                    if current is None or _comparable(current, candidate) and candidate > current:
                        target[leaf] = candidate
        else:
            raise NotImplementedError(f"Update operator {operator} is not supported")
    return result


# Query planning


def _couch_field(path: str) -> str:
    head, dot, rest = path.partition(".")
    if head.startswith("_") and head != "_id":
        head = "mongo" + head
    return head + dot + rest


def _plan(query: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Optional[List[str]], bool]:
    """Split a Mongo filter into a Mango pre-filter, an id list and an ``exact`` flag.

    ``exact`` means the pre-filter alone is equivalent to the full filter, which
    lets sorting and limiting be delegated to CouchDB.
    """

    selector: Dict[str, Any] = {}
    ids: Optional[List[str]] = None
    exact = True

    def scalar(value: Any) -> bool:
        return isinstance(value, (str, bool)) or _is_number(value)

    def intersect(current: Optional[List[str]], new: List[str]) -> List[str]:
        return new if current is None else [item for item in current if item in set(new)]

    def visit(node: Dict[str, Any]) -> None:
        nonlocal ids, exact
        for key, condition in node.items():
            if key == "$and":
                for sub in condition:
                    visit(sub)
                continue
            if key.startswith("$"):
                exact = False
                continue
            if key == "_id":
                if scalar(condition):
                    ids = intersect(ids, [str(condition)])
                    continue
                if (
                    isinstance(condition, dict)
                    and set(condition) == {"$in"}
                    and all(scalar(item) for item in condition["$in"])
                ):
                    ids = intersect(ids, [str(item) for item in condition["$in"]])
                    continue
                exact = False
                continue
            field = _couch_field(key)
            if scalar(condition):
                selector.setdefault(field, {}).update({"$eq": condition})
            elif isinstance(condition, dict) and condition:
                handled = True
                parts: Dict[str, Any] = {}
                for name, operand in condition.items():
                    if name == "$in" and all(scalar(item) for item in operand):
                        parts["$in"] = list(operand)
                    elif name in ("$gt", "$gte", "$lt", "$lte") and _is_number(operand):
                        parts[name] = operand
                        parts["$type"] = "number"
                    else:
                        handled = False
                if handled:
                    selector.setdefault(field, {}).update(parts)
                else:
                    exact = False
            else:
                exact = False

    visit(query or {})
    return selector, ids, exact


# Claim-based unique indexes


class UniqueSpec:
    """A unique index emulated through claim documents."""

    def __init__(
        self,
        name: str,
        fields: Sequence[str],
        partial: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):
        self.name = name
        self.fields = tuple(fields)
        self.partial = partial

    def key(self, doc: Dict[str, Any]) -> Optional[str]:
        if self.partial is not None and not self.partial(doc):
            return None
        values = [_get_path(doc, field)[1] for field in self.fields]
        digest = hashlib.sha1(
            json.dumps(values, sort_keys=True, default=str).encode()
        ).hexdigest()
        return f"{_CLAIM_PREFIX}{self.name}~{digest}"


# Results and cursors


class InsertOneResult:
    def __init__(self, inserted_id: Any):
        self.inserted_id = inserted_id
        self.acknowledged = True


class InsertManyResult:
    def __init__(self, inserted_ids: List[Any]):
        self.inserted_ids = inserted_ids
        self.acknowledged = True


class UpdateResult:
    def __init__(self, matched: int, modified: int, upserted_id: Any = None):
        self.matched_count = matched
        self.modified_count = modified
        self.upserted_id = upserted_id
        self.acknowledged = True


class DeleteResult:
    def __init__(self, deleted: int):
        self.deleted_count = deleted
        self.acknowledged = True


class _Cursor:
    def __init__(self, collection: "CouchDBCollection", query, projection):
        self._collection = collection
        self._query = query or {}
        self._projection = projection
        self._sort: List[Tuple[str, int]] = []
        self._skip = 0
        self._limit: Optional[int] = None

    def sort(self, key_or_list: Any, direction: Optional[int] = None) -> "_Cursor":
        self._sort = _normalize_sort(key_or_list, direction)
        return self

    def skip(self, count: int) -> "_Cursor":
        self._skip = max(0, int(count))
        return self

    def limit(self, count: int) -> "_Cursor":
        self._limit = int(count) if int(count) > 0 else None
        return self

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        docs = self._collection._execute(
            self._query, self._projection, self._sort, self._skip, self._limit
        )
        return iter(docs)


# Collection


class CouchDBCollection:
    def __init__(
        self,
        client: CouchDBClient,
        db_name: str,
        name: str,
        *,
        int_ids: bool = False,
        uniques: Sequence[UniqueSpec] = (),
        create_missing: bool = True,
    ):
        self.client = client
        self.db_name = db_name
        self.name = name
        self.int_ids = int_ids
        self.uniques = tuple(uniques)
        self.create_missing = create_missing
        self._path = "/" + quote(db_name, safe="")
        self._ensured = False

    # Encoding

    def _enc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in doc.items():
            if key == "_id":
                out["_id"] = str(value)
            elif key == "_rev":
                continue
            elif key.startswith("_"):
                out["mongo" + key] = value
            else:
                out[key] = value
        return out

    def _dec(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        doc: Dict[str, Any] = {}
        for key, value in raw.items():
            if key == "_rev":
                continue
            if key == "_id":
                doc["_id"] = int(value) if self.int_ids and str(value).isdigit() else value
            elif key.startswith("mongo_"):
                doc[key[5:]] = value
            else:
                doc[key] = value
        return doc

    def _doc_path(self, doc_id: Any) -> str:
        return f"{self._path}/{quote(str(doc_id), safe='')}"

    # Database lifecycle

    def ensure(self) -> None:
        if self._ensured:
            return
        status, body = self.client.request("PUT", self._path)
        if status not in (201, 202, 412):
            raise CouchDBError(f"Cannot create database {self.db_name}: {status} {body}")
        self._ensured = True

    def create_index(self, fields: Sequence[str], *, name: Optional[str] = None) -> None:
        couch_fields = [_couch_field(field) for field in fields]
        status, body = self.client.request(
            "POST",
            f"{self._path}/_index",
            {
                "index": {"fields": couch_fields},
                "type": "json",
                "name": name or "idx_" + "_".join(couch_fields),
            },
        )
        if status not in (200, 201):
            raise CouchDBError(f"Cannot create index on {self.db_name}: {status} {body}")

    def _request(self, method: str, path: str, body: Any = None, params=None) -> Tuple[int, Any]:
        status, payload = self.client.request(method, path, body, params)
        if (
            status == 404
            and self.create_missing
            and not self._ensured
            and isinstance(payload, dict)
            and payload.get("reason") == "Database does not exist."
        ):
            self.ensure()
            status, payload = self.client.request(method, path, body, params)
        return status, payload

    # Raw reads

    def _get_raw(self, doc_id: Any) -> Optional[Dict[str, Any]]:
        status, body = self._request("GET", self._doc_path(doc_id))
        if status == 200:
            return body
        if status == 404:
            return None
        raise CouchDBError(f"GET {self.db_name}/{doc_id}: {status} {body}")

    def _get_many_raw(self, ids: Sequence[str]) -> Iterator[Dict[str, Any]]:
        for start in range(0, len(ids), _KEYS_CHUNK):
            chunk = list(ids[start : start + _KEYS_CHUNK])
            status, body = self._request(
                "POST",
                f"{self._path}/_all_docs",
                {"keys": chunk},
                {"include_docs": "true"},
            )
            if status == 404:
                return
            if status != 200:
                raise CouchDBError(f"_all_docs {self.db_name}: {status} {body}")
            for row in body.get("rows", []):
                raw = row.get("doc")
                if raw:
                    yield raw

    def _find_raw(
        self,
        selector: Dict[str, Any],
        *,
        fields: Optional[List[str]] = None,
        sort: Optional[List[Dict[str, str]]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        bookmark: Optional[str] = None
        remaining = limit
        while True:
            page = _PAGE_SIZE if remaining is None else min(_PAGE_SIZE, remaining)
            request: Dict[str, Any] = {"selector": selector, "limit": page}
            if fields:
                request["fields"] = sorted(set(fields) | {"_id", "_rev"})
            if sort:
                request["sort"] = sort
            if bookmark:
                request["bookmark"] = bookmark
            status, body = self._request("POST", f"{self._path}/_find", request)
            if status == 404:
                return
            if status != 200:
                raise CouchDBError(f"_find {self.db_name}: {status} {body}")
            docs = body.get("docs", [])
            for raw in docs:
                yield raw
            if remaining is not None:
                remaining -= len(docs)
                if remaining <= 0:
                    return
            bookmark = body.get("bookmark")
            if len(docs) < page or not bookmark:
                return

    @staticmethod
    def _reserved(raw: Dict[str, Any]) -> bool:
        return str(raw.get("_id", "")).startswith(_RESERVED_PREFIXES)

    def _scan(
        self,
        query: Dict[str, Any],
        fields: Optional[List[str]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield raw candidate docs; callers still apply the exact filter."""

        selector, ids, _ = _plan(query)
        if ids is not None:
            source: Iterable[Dict[str, Any]] = self._get_many_raw(ids)
        else:
            selector = dict(selector) or {"_id": {"$gt": None}}
            source = self._find_raw(selector, fields=fields)
        for raw in source:
            if not self._reserved(raw):
                yield raw

    def _matching(self, query: Dict[str, Any]) -> Iterator[Tuple[Dict[str, Any], str]]:
        for raw in self._scan(query):
            doc = self._dec(raw)
            if _matches(doc, query):
                yield doc, raw["_rev"]

    def _first(self, query: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], str]]:
        selector, ids, _ = _plan(query)
        if ids is not None and len(ids) == 1 and not selector:
            raw = self._get_raw(ids[0])
            if raw is not None and not self._reserved(raw):
                doc = self._dec(raw)
                if _matches(doc, query):
                    return doc, raw["_rev"]
            return None
        for found in self._matching(query):
            return found
        return None

    def _execute(
        self,
        query: Dict[str, Any],
        projection: Optional[Dict[str, Any]],
        sort: List[Tuple[str, int]],
        skip: int,
        limit: Optional[int],
    ) -> List[Dict[str, Any]]:
        selector, ids, exact = _plan(query)
        needed: Optional[List[str]] = None
        included = [key for key, flag in (projection or {}).items() if key != "_id"]
        if projection and all(projection[key] for key in included) and (
            included or projection.get("_id")
        ):
            names = included + [name for name, _ in sort] + _query_fields(query)
            needed = ["_id"] + [_couch_field(name.split(".")[0]) for name in names]

        pushed = self._push_sort(selector, ids, exact, sort, skip, limit, needed)
        if pushed is not None:
            docs = pushed
        else:
            docs = [
                doc
                for doc in (self._dec(raw) for raw in self._scan(query, needed))
                if _matches(doc, query)
            ]
            for name, direction in reversed(sort):
                docs.sort(
                    key=lambda doc, name=name: _sort_key(_get_path(doc, name)[1]),
                    reverse=direction < 0,
                )
            docs = docs[skip:] if limit is None else docs[skip : skip + limit]
        return [_project(doc, projection) for doc in docs]

    def _push_sort(self, selector, ids, exact, sort, skip, limit, needed):
        """Let CouchDB sort and limit when its index and semantics are sufficient."""

        if not sort or not exact or ids is not None or limit is None:
            return None
        directions = {direction for _, direction in sort}
        if len(directions) != 1 or any(name == "_id" or "." in name for name, _ in sort):
            return None
        order = "desc" if directions == {-1} else "asc"
        pushed_selector = {key: dict(value) for key, value in selector.items()}
        for name, _ in sort:
            field = _couch_field(name)
            pushed_selector.setdefault(field, {}).setdefault("$gt", None)
        try:
            raws = list(
                self._find_raw(
                    pushed_selector,
                    fields=needed,
                    sort=[{_couch_field(name): order} for name, _ in sort],
                    limit=skip + limit,
                )
            )
        except CouchDBError:
            return None
        docs = [self._dec(raw) for raw in raws if not self._reserved(raw)]
        return docs[skip : skip + limit]

    # Writes

    def _put(
        self,
        doc: Dict[str, Any],
        *,
        rev: Optional[str],
        old: Optional[Dict[str, Any]],
    ) -> None:
        """Write one document, honouring unique claims; raises _Conflict/DuplicateKeyError."""

        owner = str(doc["_id"])
        acquired: List[str] = []
        released: List[str] = []
        try:
            for spec in self.uniques:
                new_key = spec.key(doc)
                old_key = spec.key(old) if old is not None else None
                if new_key and new_key != old_key:
                    if self._acquire(spec, new_key, owner):
                        acquired.append(new_key)
                if old_key and old_key != new_key:
                    released.append(old_key)
            payload = self._enc(doc)
            if rev:
                payload["_rev"] = rev
            status, body = self._request("PUT", self._doc_path(owner), payload)
            if status == 409:
                raise _Conflict()
            if status not in (201, 202):
                raise CouchDBError(f"PUT {self.db_name}/{owner}: {status} {body}")
        except BaseException:
            for key in acquired:
                self._release(key, owner)
            raise
        for key in released:
            self._release(key, owner)

    def _acquire(self, spec: UniqueSpec, claim_id: str, owner: str) -> bool:
        body = {"owner": owner, "ts": time.time(), "index": spec.name}
        path = self._doc_path(claim_id)
        status, payload = self._request("PUT", path, body)
        if status in (201, 202):
            return True
        if status != 409:
            raise CouchDBError(f"claim {self.db_name}/{claim_id}: {status} {payload}")
        claim = self._get_raw(claim_id)
        if claim is None:
            return self._acquire(spec, claim_id, owner)
        if claim.get("owner") == owner:
            return False
        holder = self._get_raw(claim["owner"])
        in_flight = holder is None and time.time() - float(claim.get("ts") or 0) < _CLAIM_TTL_SECONDS
        holds = holder is not None and spec.key(self._dec(holder)) == claim_id
        if holds or in_flight:
            raise DuplicateKeyError(
                f"duplicate key in {self.name}: unique index {spec.name}"
            )
        body["_rev"] = claim["_rev"]
        status, payload = self._request("PUT", path, body)
        if status in (201, 202):
            return True
        raise DuplicateKeyError(f"duplicate key in {self.name}: unique index {spec.name}")

    def _release(self, claim_id: str, owner: str) -> None:
        try:
            claim = self._get_raw(claim_id)
            if claim is not None and claim.get("owner") == owner:
                self._request(
                    "DELETE", self._doc_path(claim_id), params={"rev": claim["_rev"]}
                )
        except CouchDBError:
            logger.warning("Could not release claim %s", claim_id, exc_info=True)

    def _bulk_delete(self, rows: List[Tuple[Dict[str, Any], str]]) -> int:
        deleted = 0
        for start in range(0, len(rows), _PAGE_SIZE):
            chunk = rows[start : start + _PAGE_SIZE]
            payload = [
                {"_id": str(doc["_id"]), "_rev": rev, "_deleted": True} for doc, rev in chunk
            ]
            status, body = self._request("POST", f"{self._path}/_bulk_docs", {"docs": payload})
            if status not in (201, 202):
                raise CouchDBError(f"_bulk_docs {self.db_name}: {status} {body}")
            for (doc, _), result in zip(chunk, body):
                if result.get("ok"):
                    deleted += 1
                    for spec in self.uniques:
                        key = spec.key(doc)
                        if key:
                            self._release(key, str(doc["_id"]))
        return deleted

    def insert_one(self, document: Dict[str, Any]) -> InsertOneResult:
        doc = dict(document)
        if doc.get("_id") is None:
            doc["_id"] = uuid.uuid4().hex
        try:
            self._put(doc, rev=None, old=None)
        except _Conflict:
            raise DuplicateKeyError(f"duplicate _id in {self.name}: {doc['_id']}") from None
        return InsertOneResult(doc["_id"])

    def insert_many(self, documents: Iterable[Dict[str, Any]]) -> InsertManyResult:
        docs = [dict(document) for document in documents]
        for doc in docs:
            if doc.get("_id") is None:
                doc["_id"] = uuid.uuid4().hex
        if self.uniques:
            for doc in docs:
                self.insert_one(doc)
            return InsertManyResult([doc["_id"] for doc in docs])
        for start in range(0, len(docs), _PAGE_SIZE):
            chunk = docs[start : start + _PAGE_SIZE]
            status, body = self._request(
                "POST",
                f"{self._path}/_bulk_docs",
                {"docs": [self._enc(doc) for doc in chunk]},
            )
            if status not in (201, 202):
                raise CouchDBError(f"_bulk_docs {self.db_name}: {status} {body}")
            failed = [row for row in body if row.get("error")]
            if failed:
                if all(row.get("error") == "conflict" for row in failed):
                    raise DuplicateKeyError(f"duplicate _id in {self.name}: {failed[0].get('id')}")
                raise CouchDBError(f"_bulk_docs {self.db_name}: {failed[0]}")
        return InsertManyResult([doc["_id"] for doc in docs])

    def find(self, query: Optional[Dict[str, Any]] = None, projection=None) -> _Cursor:
        return _Cursor(self, query, projection)

    def find_one(self, query=None, projection=None, sort=None) -> Optional[Dict[str, Any]]:
        cursor = self.find(query, projection).limit(1)
        if sort:
            cursor.sort(sort)
        return next(iter(cursor), None)

    def count_documents(self, query: Optional[Dict[str, Any]] = None) -> int:
        return len(self._execute(query or {}, {"_id": 1}, [], 0, None))

    def distinct(self, key: str, query: Optional[Dict[str, Any]] = None) -> List[Any]:
        seen: List[Any] = []
        for doc in self._execute(query or {}, None, [], 0, None):
            found, value = _get_path(doc, key)
            if not found:
                continue
            for item in value if isinstance(value, list) else [value]:
                if item not in seen:
                    seen.append(item)
        return seen

    def max_numeric_id(self) -> int:
        highest = 0
        start = None
        while True:
            params: Dict[str, Any] = {"limit": _PAGE_SIZE}
            if start is not None:
                params.update({"startkey": json.dumps(start), "skip": 1})
            status, body = self._request("GET", f"{self._path}/_all_docs", params=params)
            rows = body.get("rows", []) if status == 200 else []
            for row in rows:
                if str(row["id"]).isdigit():
                    highest = max(highest, int(row["id"]))
            if len(rows) < _PAGE_SIZE:
                return highest
            start = rows[-1]["id"]

    def _modify(
        self,
        query: Dict[str, Any],
        build: Callable[[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]],
        *,
        upsert_doc: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], bool]:
        """CAS loop returning (before, after, inserted)."""

        for attempt in range(_MAX_CAS_ATTEMPTS):
            found = self._first(query)
            if found is None:
                if upsert_doc is None:
                    return None, None, False
                new_doc = upsert_doc()
                if new_doc.get("_id") is None:
                    new_doc["_id"] = uuid.uuid4().hex
                try:
                    self._put(new_doc, rev=None, old=None)
                except _Conflict:
                    raise DuplicateKeyError(
                        f"duplicate _id in {self.name}: {new_doc['_id']}"
                    ) from None
                return None, new_doc, True
            old, rev = found
            new_doc = build(old)
            if new_doc is None:
                return old, None, False
            new_doc["_id"] = old["_id"]
            try:
                self._put(new_doc, rev=rev, old=old)
            except _Conflict:
                time.sleep(min(0.01 * (attempt + 1), 0.1))
                continue
            return old, new_doc, False
        raise CouchDBError(f"Too many write conflicts on {self.db_name}")

    @staticmethod
    def _seed_from_filter(query: Dict[str, Any]) -> Dict[str, Any]:
        seed: Dict[str, Any] = {}
        for key, value in query.items():
            if key.startswith("$"):
                continue
            if isinstance(value, dict) and any(str(name).startswith("$") for name in value):
                continue
            seed[key] = value
        return seed

    def update_one(self, query, update, upsert: bool = False) -> UpdateResult:
        before, after, inserted = self._modify(
            query,
            lambda old: _apply_update(old, update, inserting=False),
            upsert_doc=(
                (lambda: _apply_update(self._seed_from_filter(query), update, inserting=True))
                if upsert
                else None
            ),
        )
        if inserted:
            return UpdateResult(0, 0, after["_id"])
        return UpdateResult(1 if before is not None else 0, 1 if after is not None else 0)

    def update_many(self, query, update, upsert: bool = False) -> UpdateResult:
        matched = modified = 0
        for doc, _ in list(self._matching(query)):
            key = {"_id": doc["_id"]}
            before, after, _ = self._modify(
                {"$and": [key, query]}, lambda old: _apply_update(old, update, inserting=False)
            )
            if before is not None:
                matched += 1
                modified += 1 if after is not None else 0
        return UpdateResult(matched, modified)

    def find_one_and_update(
        self,
        query,
        update,
        projection=None,
        upsert: bool = False,
        sort=None,
        return_document: bool = False,
    ) -> Optional[Dict[str, Any]]:
        before, after, inserted = self._modify(
            query,
            lambda old: _apply_update(old, update, inserting=False),
            upsert_doc=(
                (lambda: _apply_update(self._seed_from_filter(query), update, inserting=True))
                if upsert
                else None
            ),
        )
        result = after if return_document else before
        return _project(result, projection) if result is not None else None

    def replace_one(self, query, replacement: Dict[str, Any], upsert: bool = False) -> UpdateResult:
        def build(old: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            doc = dict(replacement)
            doc["_id"] = old["_id"]
            return doc

        def seed() -> Dict[str, Any]:
            doc = dict(replacement)
            if doc.get("_id") is None:
                doc["_id"] = self._seed_from_filter(query).get("_id")
            return doc

        before, after, inserted = self._modify(
            query, build, upsert_doc=seed if upsert else None
        )
        if inserted:
            return UpdateResult(0, 0, after["_id"])
        return UpdateResult(1 if before is not None else 0, 1 if before is not None else 0)

    def delete_one(self, query) -> DeleteResult:
        for _ in range(_MAX_CAS_ATTEMPTS):
            found = self._first(query)
            if found is None:
                return DeleteResult(0)
            doc, rev = found
            if self._bulk_delete([(doc, rev)]):
                return DeleteResult(1)
        raise CouchDBError(f"Too many write conflicts on {self.db_name}")

    def delete_many(self, query) -> DeleteResult:
        return DeleteResult(self._bulk_delete(list(self._matching(query or {}))))


def _query_fields(query: Optional[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for key, condition in (query or {}).items():
        if key in ("$and", "$or"):
            for sub in condition:
                names.extend(_query_fields(sub))
        elif not key.startswith("$") and key != "_id":
            names.append(key)
    return names


class CouchDBDatabase:
    """Attribute/item access to collections, like ``pymongo.database.Database``."""

    def __init__(
        self,
        client: CouchDBClient,
        prefix: str,
        *,
        int_id_collections: Sequence[str] = (),
        uniques: Optional[Dict[str, Sequence[UniqueSpec]]] = None,
        create_missing: bool = True,
    ):
        self.client = client
        self.prefix = prefix
        self._int_ids = set(int_id_collections)
        self._uniques = dict(uniques or {})
        self._create_missing = create_missing
        self._collections: Dict[str, CouchDBCollection] = {}

    def __getitem__(self, name: str) -> CouchDBCollection:
        if name not in self._collections:
            self._collections[name] = CouchDBCollection(
                self.client,
                f"{self.prefix}_{name}".lower(),
                name,
                int_ids=name in self._int_ids,
                uniques=self._uniques.get(name, ()),
                create_missing=self._create_missing,
            )
        return self._collections[name]

    def __getattr__(self, name: str) -> CouchDBCollection:
        if name.startswith("__"):
            raise AttributeError(name)
        return self[name]
