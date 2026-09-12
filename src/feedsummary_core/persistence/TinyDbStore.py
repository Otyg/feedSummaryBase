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
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any, Dict, List, Optional, Set, Tuple

from tinydb import Query, TinyDB
from tinydb.operations import delete as delete_field
from feedsummary_core.persistence import CleanupPolicy
from feedsummary_core.long_term.reconciliation import (
    validate_cluster_merge_operation,
)
from feedsummary_core.long_term.membership_edit import validate_cluster_membership_edit
from feedsummary_core.long_term.review import validate_cluster_review_resolution
from feedsummary_core.persistence.tag_relations import (
    PARENT_CHILD_RELATION,
    proposed_parent_child_edges,
)
from feedsummary_core.tagging_rules import VULNERABILITY_TAG_CATEGORY, is_cve_tag

logger = logging.getLogger(__name__)
_LONG_TERM_LOCKS: Dict[str, threading.Lock] = {}
_LONG_TERM_LOCKS_GUARD = threading.Lock()
_ASSIGNMENT_OPERATION_FIELD = "last_assignment_operation_id"
_SNAPSHOT_OPERATION_FIELD = "last_snapshot_operation_id"
_SNAPSHOT_PENDING_FIELD = "pending_operation_id"

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]


@contextmanager
def _db_file_lock(database_path: str, suffix: str):
    resolved = os.path.abspath(database_path)
    with _LONG_TERM_LOCKS_GUARD:
        thread_lock = _LONG_TERM_LOCKS.setdefault(f"{resolved}:{suffix}", threading.Lock())
    with thread_lock:
        lock_path = f"{resolved}.{suffix}"
        with open(lock_path, "a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _article_file_lock(database_path: str):
    """Serialize TinyDB article writes across threads and, on Unix, processes."""

    with _db_file_lock(database_path, "articles.lock"):
        yield


@contextmanager
def _long_term_file_lock(database_path: str):
    """Serialize TinyDB long-term writes across threads and, on Unix, processes."""

    with _db_file_lock(database_path, "long-term.lock"):
        yield


def _normalize_summary_id(value: Any) -> Optional[str]:
    summary_id = str(value or "").strip()
    if summary_id.lower() in {"", "none", "null"}:
        return None
    return summary_id


class TinyDBStore:
    """
    TinyDB-backed store (JSON file).
    Uses TinyDB doc_id as the integer ID for jobs/temp summaries etc.
    """

    def __init__(self, path: str = "news_docs.json"):
        self.path = path

    def _db(self) -> TinyDB:
        return TinyDB(self.path)

    def get_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        db = self._db()
        A = Query()
        res = db.table("articles").search(A.id == article_id)
        db.close()
        return res[0] if res else None

    def upsert_article(self, article_doc: Dict[str, Any]) -> None:
        with _article_file_lock(self.path):
            db = self._db()
            try:
                A = Query()
                db.table("articles").upsert(article_doc, A.id == article_doc["id"])
            finally:
                db.close()

    def update_article_embedding(
        self,
        article_id: str,
        embedding_vector: List[float],
        *,
        model: Optional[str] = None,
        source_hash: Optional[str] = None,
        purpose: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> bool:
        """Persist a purpose-specific embedding for an existing article."""
        if not article_id or not embedding_vector or not all(
            isinstance(value, (int, float)) for value in embedding_vector
        ):
            return False
        normalized = [float(value) for value in embedding_vector]
        purpose_name = None if purpose is None else str(purpose).strip().lower()
        if purpose_name is None:
            fields = {
                "embedding_vector": normalized,
                "embedding_model": str(model or ""),
                "embedding_source_hash": str(source_hash or ""),
                "embedding_updated_at": int(time.time()),
            }
        else:
            if purpose_name not in {"similarity", "tagging"}:
                raise ValueError(f"Unsupported article embedding purpose: {purpose}")
            prefix = f"{purpose_name}_embedding"
            fields = {
                f"{prefix}_vector": normalized,
                f"{prefix}_model": str(model or ""),
                f"{prefix}_source_hash": str(source_hash or ""),
                f"{prefix}_instruction": str(instruction or "").strip(),
                f"{prefix}_updated_at": int(time.time()),
            }
        with _article_file_lock(self.path):
            db = self._db()
            try:
                A = Query()
                updated = db.table("articles").update(fields, A.id == str(article_id))
                return bool(updated)
            finally:
                db.close()

    def list_articles(self, limit: int = 2000) -> List[Dict[str, Any]]:
        """
        Returnera artiklar utan att använda 'summarized'-flagga.
        OBS: här returnerar vi själva dokumenten (dvs id = artikelns id).
        """
        db = self._db()
        docs = list(db.table("articles"))
        db.close()
        out = [dict(d) for d in docs]
        # sort oldest-first på published_ts för stabil batching
        out.sort(key=lambda r: int(r.get("published_ts") or r.get("fetched_at") or 0))
        return out[:limit]

    def iter_articles(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Yield every article oldest-first without the list API's default cap."""
        db = self._db()
        try:
            articles = [dict(row) for row in db.table("articles")]
        finally:
            db.close()
        articles.sort(
            key=lambda row: int(row.get("published_ts") or row.get("fetched_at") or 0)
        )
        maximum = int(limit) if limit is not None and int(limit) > 0 else None
        yield from articles[:maximum]

    def list_articles_by_filter(
        self,
        *,
        sources: List[str],
        since_ts: int,
        until_ts: Optional[int] = None,
        limit: int = 2000,
    ) -> List[Dict[str, Any]]:
        """
        Filtrera artiklar baserat på:
          - source ∈ sources
          - published_ts >= since_ts
          - och om until_ts: published_ts <= until_ts
        """
        srcset: Set[str] = {str(s) for s in (sources or []) if str(s).strip()}
        db = self._db()
        at = db.table("articles")

        def match(row: Dict[str, Any]) -> bool:
            if srcset and row.get("source") not in srcset:
                return False
            ts = row.get("published_ts")
            if not isinstance(ts, int) or ts <= 0:
                # om published_ts saknas: fall back fetched_at
                ts = row.get("fetched_at")
                if not isinstance(ts, int) or ts <= 0:
                    return False
            if ts < since_ts:
                return False
            if until_ts is not None and ts > until_ts:
                return False
            return True

        rows = at.search(match)
        db.close()

        rows_sorted = sorted(
            rows, key=lambda r: int(r.get("published_ts") or r.get("fetched_at") or 0)
        )
        return [dict(r) for r in rows_sorted[:limit]]

    def list_unsummarized_articles(self, limit: int = 200) -> List[Dict[str, Any]]:
        db = self._db()
        A = Query()
        res = db.table("articles").search((A.summarized != True))  # noqa: E712
        db.close()
        return res[:limit]  # pyright: ignore[reportReturnType]

    def mark_articles_summarized(self, article_ids: List[str]) -> None:
        """
        Legacy: Behålls för bakåtkomp, men pipeline använder den inte längre.
        """
        with _article_file_lock(self.path):
            db = self._db()
            try:
                A = Query()
                ts = int(time.time())
                for aid in article_ids:
                    db.table("articles").update(
                        {"summarized": True, "summarized_at": ts},
                        A.id == aid,
                    )
            finally:
                db.close()

    def save_summary_doc(self, summary_doc: Dict[str, Any]) -> Any:
        db = self._db()
        t = db.table("summary_docs")
        Q = Query()

        doc = dict(summary_doc or {})
        if "created" not in doc:
            doc["created"] = int(time.time())
        if "kind" not in doc:
            doc["kind"] = "summary"

        if doc.get("id"):
            sid = str(doc["id"])
            t.upsert(doc, Q.id == sid)
            db.close()
            return sid

        doc_id = t.insert(doc)
        try:
            t.update({"id": f"summary_doc_{doc_id}"}, doc_ids=[doc_id])
        except Exception:
            pass
        db.close()
        return doc_id

    def get_summary_doc(self, summary_doc_id: str) -> Optional[Dict[str, Any]]:
        db = self._db()
        t = db.table("summary_docs")
        Q = Query()
        rows = t.search(Q.id == str(summary_doc_id))
        db.close()
        return rows[0] if rows else None

    def list_summary_docs(self) -> List[Dict[str, Any]]:
        db = self._db()
        docs = list(db.table("summary_docs"))
        db.close()
        out = [dict(d) for d in docs]
        out.sort(key=lambda r: r.get("created", 0), reverse=True)
        return out

    def get_latest_summary_doc(self) -> Optional[Dict[str, Any]]:
        docs = self.list_summary_docs()
        return docs[0] if docs else None

    def create_job(self) -> int:
        db = self._db()
        jid = db.table("jobs").insert(
            {
                "created_at": int(time.time()),
                "started_at": None,
                "finished_at": None,
                "status": "queued",
                "message": "",
                "summary_id": None,
            }
        )
        db.close()
        logger.info("Job %s created", jid)
        return int(jid)

    def update_job(self, job_id: int, **fields) -> None:
        if "summary_id" in fields:
            fields["summary_id"] = _normalize_summary_id(fields.get("summary_id"))
        db = self._db()
        db.table("jobs").update(fields, doc_ids=[int(job_id)])
        logger.info("Job %s updated: %s", job_id, fields)
        db.close()

    def get_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        db = self._db()
        doc = db.table("jobs").get(doc_id=int(job_id))
        db.close()
        if not doc:
            return None
        return {"id": int(job_id), **dict(doc)}  # type: ignore

    def list_jobs(self, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Returnerar jobs som dictar med 'id' (TinyDB doc_id) inkluderad.
        Robust mot TinyDB-versioner: försök läsa doc_id från Document om möjligt,
        annars fall back till intern 'doc_id' om den finns.
        """
        lim = int(limit) if limit and int(limit) > 0 else 200

        db = self._db()
        t = db.table("jobs")

        out: List[Dict[str, Any]] = []

        # TinyDB >=4: Table.all() returnerar Document med .doc_id
        try:
            rows = t.all()
            for r in rows:
                try:
                    jid = int(getattr(r, "doc_id"))  # Document
                except Exception:
                    # fallback: om någon råkat skriva in "id" i payloaden
                    jid = int((r.get("id") or 0))
                if jid <= 0:
                    continue
                out.append({"id": jid, **dict(r)})
        except Exception:
            # Ultimat fallback: iterera t (brukar också ge Document)
            try:
                for r in t:
                    try:
                        jid = int(getattr(r, "doc_id"))
                    except Exception:
                        jid = int((r.get("id") or 0))
                    if jid <= 0:
                        continue
                    out.append({"id": jid, **dict(r)})
            except Exception as e:
                logger.warning("list_jobs failed: %s", e)
                out = []

        db.close()

        out.sort(key=lambda r: int(r.get("created_at") or 0), reverse=True)
        return out[:lim]

    def get_articles_by_ids(self, article_ids: List[str]) -> List[Dict[str, Any]]:
        db = self._db()
        at = db.table("articles")
        out: List[Dict[str, Any]] = []
        for aid in article_ids:
            rows = at.search(lambda r: r.get("id") == aid)
            if rows:
                out.append(rows[0])
        db.close()
        return out

    # Long-term threat-landscape analysis

    def list_articles_for_long_term(
        self,
        *,
        after_fetched_at: int = 0,
        after_article_id: str = "",
        until_fetched_at: Optional[int] = None,
        sources: Optional[List[str]] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        after = (int(after_fetched_at), str(after_article_id or ""))
        until = int(until_fetched_at) if until_fetched_at is not None else None
        source_set = {str(source).strip() for source in sources or [] if str(source).strip()}
        db = self._db()
        try:
            rows = []
            for raw in db.table("articles"):
                row = dict(raw)
                cursor = (int(row.get("fetched_at") or 0), str(row.get("id") or ""))
                if cursor <= after or (until is not None and cursor[0] > until):
                    continue
                if source_set and str(row.get("source") or "") not in source_set:
                    continue
                rows.append(row)
            rows.sort(key=lambda row: (int(row.get("fetched_at") or 0), str(row.get("id") or "")))
            return rows[: max(1, int(limit))]
        finally:
            db.close()

    def get_long_term_cursor(self, profile_id: str) -> Dict[str, Any]:
        profile_id = str(profile_id or "").strip()
        if not profile_id:
            raise ValueError("profile_id must not be empty")
        db = self._db()
        try:
            row = db.table("long_term_state").get(Query().profile_id == profile_id)
            if row:
                return dict(row)
            return {
                "profile_id": profile_id,
                "cursor_fetched_at": 0,
                "cursor_article_id": "",
                "lease_owner": None,
                "lease_until": 0,
                "updated_at": 0,
            }
        finally:
            db.close()

    def claim_long_term_lease(
        self,
        profile_id: str,
        owner_id: str,
        *,
        now_ts: int,
        lease_seconds: int,
    ) -> bool:
        profile_id = str(profile_id or "").strip()
        owner_id = str(owner_id or "").strip()
        if not profile_id or not owner_id or lease_seconds < 1:
            raise ValueError("profile, owner and positive lease_seconds are required")
        with _long_term_file_lock(self.path):
            db = self._db()
            try:
                table = db.table("long_term_state")
                query = Query()
                existing = table.get(query.profile_id == profile_id)
                state = dict(existing) if existing else {
                    "profile_id": profile_id,
                    "cursor_fetched_at": 0,
                    "cursor_article_id": "",
                }
                current_owner = str(state.get("lease_owner") or "")
                current_until = int(state.get("lease_until") or 0)
                if current_owner and current_owner != owner_id and current_until > int(now_ts):
                    return False
                state.update(
                    {
                        "lease_owner": owner_id,
                        "lease_until": int(now_ts) + int(lease_seconds),
                        "updated_at": int(now_ts),
                    }
                )
                table.upsert(state, query.profile_id == profile_id)
                return True
            finally:
                db.close()

    def release_long_term_lease(self, profile_id: str, owner_id: str) -> bool:
        with _long_term_file_lock(self.path):
            db = self._db()
            try:
                table = db.table("long_term_state")
                query = Query()
                row = table.get(query.profile_id == str(profile_id))
                if not row or row.get("lease_owner") != str(owner_id):
                    return False
                state = dict(row)
                state.update({"lease_owner": None, "lease_until": 0, "updated_at": int(time.time())})
                table.update(state, query.profile_id == str(profile_id))
                return True
            finally:
                db.close()

    def renew_long_term_lease(
        self,
        profile_id: str,
        owner_id: str,
        *,
        now_ts: int,
        lease_seconds: int,
    ) -> bool:
        profile_id = str(profile_id or "").strip()
        owner_id = str(owner_id or "").strip()
        if not profile_id or not owner_id or lease_seconds < 1:
            raise ValueError("profile, owner and positive lease_seconds are required")
        now_ts = int(now_ts)
        with _long_term_file_lock(self.path):
            db = self._db()
            try:
                table = db.table("long_term_state")
                query = Query()
                row = table.get(query.profile_id == profile_id)
                if (
                    not row
                    or row.get("lease_owner") != owner_id
                    or int(row.get("lease_until") or 0) <= now_ts
                ):
                    return False
                state = dict(row)
                state.update(
                    {
                        "lease_until": now_ts + int(lease_seconds),
                        "updated_at": now_ts,
                    }
                )
                table.update(state, query.profile_id == profile_id)
                return True
            finally:
                db.close()

    def advance_long_term_cursor(
        self,
        profile_id: str,
        owner_id: str,
        *,
        expected_fetched_at: int,
        expected_article_id: str,
        fetched_at: int,
        article_id: str,
        now_ts: int,
    ) -> bool:
        expected = (int(expected_fetched_at), str(expected_article_id or ""))
        new_cursor = (int(fetched_at), str(article_id or ""))
        if new_cursor < expected or new_cursor[0] < 1 or not new_cursor[1]:
            raise ValueError("new cursor must be complete and cannot move backwards")
        with _long_term_file_lock(self.path):
            db = self._db()
            try:
                table = db.table("long_term_state")
                query = Query()
                row = table.get(query.profile_id == str(profile_id))
                if not row:
                    return False
                state = dict(row)
                current = (
                    int(state.get("cursor_fetched_at") or 0),
                    str(state.get("cursor_article_id") or ""),
                )
                if (
                    current != expected
                    or state.get("lease_owner") != str(owner_id)
                    or int(state.get("lease_until") or 0) <= int(now_ts)
                ):
                    return False
                state.update(
                    {
                        "cursor_fetched_at": new_cursor[0],
                        "cursor_article_id": new_cursor[1],
                        "updated_at": int(now_ts),
                    }
                )
                table.update(state, query.profile_id == str(profile_id))
                return True
            finally:
                db.close()

    def get_threat_cluster(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        return self._get_long_term_doc("threat_clusters", str(cluster_id))

    def list_threat_clusters(
        self,
        profile_id: str,
        *,
        statuses: Optional[List[str]] = None,
        min_last_seen_ts: Optional[int] = None,
        embedding_model: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        embedding_instruction: Optional[str] = None,
        min_member_count: Optional[int] = None,
        limit: int = 10000,
        offset: int = 0,
        include_vectors: bool = True,
    ) -> List[Dict[str, Any]]:
        status_set = {str(status) for status in statuses or [] if str(status)}
        db = self._db()
        try:
            rows = []
            for raw in db.table("threat_clusters"):
                row = dict(raw)
                if str(row.get("profile_id")) != str(profile_id):
                    continue
                if status_set and str(row.get("status")) not in status_set:
                    continue
                if min_last_seen_ts is not None and int(row.get("last_seen_ts") or 0) < int(
                    min_last_seen_ts
                ):
                    continue
                if embedding_model is not None and row.get("embedding_model") != embedding_model:
                    continue
                if embedding_dimension is not None and int(
                    row.get("embedding_dimension") or 0
                ) != int(embedding_dimension):
                    continue
                if (
                    embedding_instruction is not None
                    and row.get("embedding_instruction") != embedding_instruction
                ):
                    continue
                if min_member_count is not None and int(row.get("member_count") or 0) < int(
                    min_member_count
                ):
                    continue
                rows.append(row)
            rows.sort(key=lambda row: (-int(row.get("last_seen_ts") or 0), str(row.get("id"))))
            start = max(0, int(offset))
            selected = rows[start : start + max(1, int(limit))]
            if not include_vectors:
                return [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"centroid", "vector_sum"}
                    }
                    for row in selected
                ]
            return selected
        finally:
            db.close()

    def save_threat_cluster(
        self,
        cluster_doc: Dict[str, Any],
        *,
        expected_membership_revision: Optional[int] = None,
    ) -> bool:
        doc = dict(cluster_doc or {})
        required = (
            "id",
            "profile_id",
            "status",
            "last_seen_ts",
            "embedding_model",
            "embedding_dimension",
            "embedding_instruction",
            "membership_revision",
        )
        if any(doc.get(field) is None for field in required):
            raise ValueError("cluster document is incomplete")
        doc.setdefault("updated_at", int(time.time()))
        with _long_term_file_lock(self.path):
            db = self._db()
            try:
                table = db.table("threat_clusters")
                query = Query()
                existing = table.get(query.id == str(doc["id"]))
                if existing is None and expected_membership_revision is not None:
                    return False
                if existing is not None and (
                    expected_membership_revision is None
                    or int(existing.get("membership_revision") or 0)
                    != int(expected_membership_revision)
                ):
                    return False
                if existing is None:
                    table.insert(doc)
                else:
                    def replace_document(row):
                        row.clear()
                        row.update(doc)

                    table.update(replace_document, query.id == str(doc["id"]))
                return True
            finally:
                db.close()

    def resolve_threat_cluster_review(
        self,
        cluster_doc: Dict[str, Any],
        *,
        expected_membership_revision: int,
    ) -> bool:
        doc = validate_cluster_review_resolution(cluster_doc)
        doc.setdefault("updated_at", int(time.time()))
        with _long_term_file_lock(self.path):
            db = self._db()
            try:
                table = db.table("threat_clusters")
                query = Query()
                existing = table.get(query.id == str(doc["id"]))
                if (
                    existing is None
                    or int(existing.get("membership_revision") or 0)
                    != int(expected_membership_revision)
                    or str(existing.get("status") or "") != "needs_review"
                    or existing.get("review_decision")
                ):
                    return False

                def replace_document(row):
                    row.clear()
                    row.update(doc)

                table.update(replace_document, query.id == str(doc["id"]))
                return True
            finally:
                db.close()

    def get_cluster_membership(
        self, profile_id: str, article_id: str
    ) -> Optional[Dict[str, Any]]:
        db = self._db()
        try:
            row = db.table("threat_cluster_memberships").get(
                (Query().profile_id == str(profile_id)) & (Query().article_id == str(article_id))
            )
            return dict(row) if row else None
        finally:
            db.close()

    def save_cluster_membership(self, membership_doc: Dict[str, Any]) -> bool:
        doc = dict(membership_doc or {})
        required = ("profile_id", "article_id", "cluster_id", "assigned_at")
        if any(doc.get(field) is None for field in required):
            raise ValueError("membership document is incomplete")
        with _long_term_file_lock(self.path):
            db = self._db()
            try:
                table = db.table("threat_cluster_memberships")
                query = Query()
                match = (query.profile_id == str(doc["profile_id"])) & (
                    query.article_id == str(doc["article_id"])
                )
                if table.contains(match):
                    return False
                table.insert(doc)
                return True
            finally:
                db.close()

    @staticmethod
    def _apply_cluster_assignment(db: TinyDB, operation: Dict[str, Any]) -> bool:
        cluster = dict(operation["cluster"])
        membership = dict(operation["membership"])
        operation_id = str(operation["id"])
        expected = operation.get("expected_membership_revision")
        cluster_table = db.table("threat_clusters")
        membership_table = db.table("threat_cluster_memberships")
        query = Query()
        membership_match = (query.profile_id == str(membership["profile_id"])) & (
            query.article_id == str(membership["article_id"])
        )
        existing_membership = membership_table.get(membership_match)
        if existing_membership is not None and str(
            existing_membership.get("cluster_id")
        ) != str(cluster["id"]):
            return False

        cluster_match = query.id == str(cluster["id"])
        existing_cluster = cluster_table.get(cluster_match)
        target_revision = int(cluster["membership_revision"])
        if existing_cluster is None:
            if expected is not None:
                return False
            cluster[_ASSIGNMENT_OPERATION_FIELD] = operation_id
            cluster_table.insert(cluster)
        else:
            current_revision = int(existing_cluster.get("membership_revision") or 0)
            if current_revision == target_revision:
                if existing_membership is None and str(
                    existing_cluster.get(_ASSIGNMENT_OPERATION_FIELD) or ""
                ) != operation_id:
                    return False
            elif expected is None or current_revision != int(expected):
                return False
            else:
                cluster[_ASSIGNMENT_OPERATION_FIELD] = operation_id

                def replace_document(row):
                    row.clear()
                    row.update(cluster)

                cluster_table.update(replace_document, cluster_match)

        if existing_membership is None:
            membership_table.insert(membership)
        return True

    def save_cluster_assignment(
        self,
        cluster_doc: Dict[str, Any],
        membership_doc: Dict[str, Any],
        *,
        expected_membership_revision: Optional[int] = None,
    ) -> bool:
        """Persist an assignment under a process lock and recoverable journal."""

        cluster = dict(cluster_doc or {})
        membership = dict(membership_doc or {})
        cluster_required = (
            "id",
            "profile_id",
            "status",
            "last_seen_ts",
            "embedding_model",
            "embedding_dimension",
            "embedding_instruction",
            "membership_revision",
        )
        membership_required = ("profile_id", "article_id", "cluster_id", "assigned_at")
        if any(cluster.get(field) is None for field in cluster_required):
            raise ValueError("cluster document is incomplete")
        if any(membership.get(field) is None for field in membership_required):
            raise ValueError("membership document is incomplete")
        if (
            str(cluster["id"]) != str(membership["cluster_id"])
            or str(cluster["profile_id"]) != str(membership["profile_id"])
        ):
            raise ValueError("cluster and membership identities do not match")
        cluster.setdefault("updated_at", int(time.time()))
        operation_id = f"{membership['profile_id']}:{membership['article_id']}"
        operation = {
            "id": operation_id,
            "cluster": cluster,
            "membership": membership,
            "expected_membership_revision": expected_membership_revision,
            "created_at": int(time.time()),
        }

        with _long_term_file_lock(self.path):
            db = self._db()
            try:
                journal = db.table("long_term_assignment_journal")
                query = Query()
                for pending in list(journal):
                    if self._apply_cluster_assignment(db, dict(pending)):
                        journal.remove(query.id == str(pending.get("id")))

                membership_match = (
                    query.profile_id == str(membership["profile_id"])
                ) & (query.article_id == str(membership["article_id"]))
                if db.table("threat_cluster_memberships").contains(membership_match):
                    return False
                if journal.contains(query.id == operation_id):
                    return False
                journal.insert(operation)
                if not self._apply_cluster_assignment(db, operation):
                    return False
                journal.remove(query.id == operation_id)
                return True
            finally:
                db.close()

    def list_cluster_memberships(
        self, cluster_id: str, *, limit: int = 10000
    ) -> List[Dict[str, Any]]:
        return self._list_long_term_docs(
            "threat_cluster_memberships",
            lambda row: row.get("cluster_id") == str(cluster_id),
            lambda row: (int(row.get("assigned_at") or 0), str(row.get("article_id") or "")),
            limit,
        )

    @staticmethod
    def _apply_cluster_reconciliation(
        db: TinyDB, operation: Dict[str, Any]
    ) -> bool:
        reconciliation_table = db.table("long_term_cluster_reconciliations")
        cluster_table = db.table("threat_clusters")
        membership_table = db.table("threat_cluster_memberships")
        query = Query()
        operation_id = str(operation["id"])
        primary_id = str(operation["primary_cluster_id"])
        profile_id = str(operation["profile_id"])
        source_ids = {str(value) for value in operation["source_cluster_ids"]}
        expected = operation["expected_membership_revisions"]
        existing_operation = reconciliation_table.get(query.id == operation_id)
        if existing_operation is not None:
            return (
                str(existing_operation.get("primary_cluster_id")) == primary_id
                and set(existing_operation.get("source_cluster_ids") or [])
                == source_ids
            )

        targets = {
            str(operation["merged_cluster"]["id"]): dict(
                operation["merged_cluster"]
            ),
            **{
                str(row["id"]): dict(row)
                for row in operation["superseded_clusters"]
            },
        }
        for cluster_id in source_ids:
            current = cluster_table.get(query.id == cluster_id)
            if current is None:
                return False
            current_revision = int(current.get("membership_revision") or 0)
            target_revision = int(targets[cluster_id]["membership_revision"])
            already_target = (
                current_revision == target_revision
                and str(current.get("reconciliation_id") or "") == operation_id
            )
            if not already_target and current_revision != int(expected[cluster_id]):
                return False

        memberships = [
            dict(row)
            for row in membership_table
            if str(row.get("profile_id") or "") == profile_id
            and (
                str(row.get("cluster_id") or "") in source_ids
                or str(row.get("reconciliation_id") or "") == operation_id
            )
        ]
        if len(memberships) != int(operation["merged_cluster"]["member_count"]):
            return False

        for cluster_id, target in targets.items():
            def replace_cluster(row, replacement=target):
                row.clear()
                row.update(replacement)

            cluster_table.update(replace_cluster, query.id == cluster_id)

        for membership in memberships:
            previous_id = str(membership.get("cluster_id") or "")
            lineage = [
                str(value) for value in membership.get("cluster_lineage") or []
            ]
            if previous_id != primary_id and previous_id not in lineage:
                lineage.append(previous_id)
                membership["previous_cluster_id"] = previous_id
            membership.update(
                {
                    "cluster_id": primary_id,
                    "cluster_lineage": lineage,
                    "cluster_membership_revision": int(
                        operation["membership_revision_by_article_id"][
                            str(membership["article_id"])
                        ]
                    ),
                    "reconciliation_id": operation_id,
                    "reconciled_at": int(operation["reconciled_at"]),
                }
            )
            match = (query.profile_id == profile_id) & (
                query.article_id == str(membership["article_id"])
            )

            def replace_membership(row, replacement=membership):
                row.clear()
                row.update(replacement)

            membership_table.update(replace_membership, match)

        stored_operation = {
            **operation,
            "status": "applied",
            "membership_count": len(memberships),
            "applied_at": int(operation["reconciled_at"]),
        }
        reconciliation_table.insert(stored_operation)
        return True

    def apply_cluster_reconciliation(
        self, reconciliation_doc: Dict[str, Any]
    ) -> bool:
        """Apply or replay a recoverable multi-cluster merge operation."""

        operation = validate_cluster_merge_operation(reconciliation_doc)
        operation_id = str(operation["id"])
        with _long_term_file_lock(self.path):
            db = self._db()
            try:
                journal = db.table("long_term_reconciliation_journal")
                query = Query()
                for pending in list(journal):
                    if self._apply_cluster_reconciliation(db, dict(pending)):
                        journal.remove(query.id == str(pending.get("id")))
                existing = db.table("long_term_cluster_reconciliations").get(
                    query.id == operation_id
                )
                if existing is not None:
                    if (
                        str(existing.get("primary_cluster_id"))
                        != operation["primary_cluster_id"]
                        or list(existing.get("source_cluster_ids") or [])
                        != operation["source_cluster_ids"]
                    ):
                        raise ValueError(
                            "reconciliation id already has different content"
                        )
                    return True
                if journal.contains(query.id == operation_id):
                    return False
                journal.insert(operation)
                if not self._apply_cluster_reconciliation(db, operation):
                    return False
                journal.remove(query.id == operation_id)
                return True
            finally:
                db.close()

    def get_cluster_reconciliation(
        self, reconciliation_id: str
    ) -> Optional[Dict[str, Any]]:
        return self._get_long_term_doc(
            "long_term_cluster_reconciliations", str(reconciliation_id)
        )

    @staticmethod
    def _apply_cluster_membership_edit(db: TinyDB, operation: Dict[str, Any]) -> bool:
        query = Query()
        edits = db.table("long_term_cluster_membership_edits")
        existing = edits.get(query.id == operation["id"])
        if existing is not None:
            return (
                existing.get("article_id") == operation["article_id"]
                and existing.get("target_cluster_id") == operation["target_cluster_id"]
            )
        clusters = db.table("threat_clusters")
        memberships = db.table("threat_cluster_memberships")
        source_id = operation["source_cluster_id"]
        target_id = operation["target_cluster_id"]
        expected_revisions = {
            source_id: operation["expected_source_membership_revision"],
            target_id: operation.get("expected_target_membership_revision"),
        }
        finals = {
            source_id: operation["source_cluster"],
            target_id: operation["target_cluster"],
        }
        for cluster_id, expected in expected_revisions.items():
            current = clusters.get(query.id == cluster_id)
            already_final = (
                current is not None
                and current.get("membership_edit_id") == operation["id"]
                and int(current.get("membership_revision") or 0)
                == int(finals[cluster_id]["membership_revision"])
            )
            if not already_final and (
                expected is None and current is not None
                or expected is not None
                and (
                    current is None
                    or int(current.get("membership_revision") or 0) != int(expected)
                )
            ):
                return False
        final_memberships = {
            row["article_id"]: row for row in operation["memberships"]
        }
        for article_id, expected_cluster in operation["expected_memberships"].items():
            match = (query.profile_id == operation["profile_id"]) & (
                query.article_id == article_id
            )
            current = memberships.get(match)
            already_final = (
                current is not None
                and current.get("membership_edit_id") == operation["id"]
                and current.get("cluster_id") == final_memberships[article_id]["cluster_id"]
            )
            if not already_final and (
                current is None or current.get("cluster_id") != expected_cluster
            ):
                return False

        def replace(document):
            def callback(row):
                row.clear()
                row.update(document)
            return callback

        for cluster_id, document in finals.items():
            if clusters.contains(query.id == cluster_id):
                clusters.update(replace(document), query.id == cluster_id)
            else:
                clusters.insert(document)
        for article_id, document in final_memberships.items():
            memberships.update(
                replace(document),
                (query.profile_id == operation["profile_id"])
                & (query.article_id == article_id),
            )
        edits.insert(
            {**operation, "status": "applied", "applied_at": operation["edited_at"]}
        )
        return True

    def apply_cluster_membership_edit(self, edit_doc: Dict[str, Any]) -> bool:
        operation = validate_cluster_membership_edit(edit_doc)
        with _long_term_file_lock(self.path):
            db = self._db()
            try:
                journal = db.table("long_term_membership_edit_journal")
                query = Query()
                for pending in list(journal):
                    if self._apply_cluster_membership_edit(db, dict(pending)):
                        journal.remove(query.id == pending.get("id"))
                existing = db.table("long_term_cluster_membership_edits").get(
                    query.id == operation["id"]
                )
                if existing is not None:
                    if (
                        existing.get("article_id") != operation["article_id"]
                        or existing.get("target_cluster_id") != operation["target_cluster_id"]
                    ):
                        raise ValueError("membership edit id already has different content")
                    return True
                journal.insert(operation)
                if not self._apply_cluster_membership_edit(db, operation):
                    return False
                journal.remove(query.id == operation["id"])
                return True
            finally:
                db.close()

    def get_cluster_membership_edit(self, edit_id: str) -> Optional[Dict[str, Any]]:
        return self._get_long_term_doc("long_term_cluster_membership_edits", str(edit_id))

    def get_long_term_quarantine(
        self, profile_id: str, article_id: str
    ) -> Optional[Dict[str, Any]]:
        db = self._db()
        try:
            query = Query()
            row = db.table("long_term_article_quarantine").get(
                (query.profile_id == str(profile_id))
                & (query.article_id == str(article_id))
            )
            return dict(row) if row else None
        finally:
            db.close()

    def save_long_term_quarantine(self, quarantine_doc: Dict[str, Any]) -> bool:
        doc = dict(quarantine_doc or {})
        required = ("profile_id", "article_id", "reason", "observed_at")
        if any(doc.get(field) is None for field in required):
            raise ValueError("quarantine document is incomplete")
        with _long_term_file_lock(self.path):
            db = self._db()
            try:
                table = db.table("long_term_article_quarantine")
                query = Query()
                match = (query.profile_id == str(doc["profile_id"])) & (
                    query.article_id == str(doc["article_id"])
                )
                existing = table.get(match)
                observed_at = int(doc["observed_at"])
                merged = {
                    **(dict(existing) if existing else {}),
                    **doc,
                    "status": "open",
                    "first_seen_at": int(existing.get("first_seen_at") or observed_at)
                    if existing
                    else observed_at,
                    "last_seen_at": observed_at,
                    "attempt_count": int(existing.get("attempt_count") or 0) + 1
                    if existing
                    else 1,
                    "resolved_at": None,
                }
                if existing:
                    table.update(merged, match)
                else:
                    table.insert(merged)
                return True
            finally:
                db.close()

    def resolve_long_term_quarantine(
        self, profile_id: str, article_id: str, *, resolved_at: int
    ) -> bool:
        with _long_term_file_lock(self.path):
            db = self._db()
            try:
                table = db.table("long_term_article_quarantine")
                query = Query()
                match = (
                    (query.profile_id == str(profile_id))
                    & (query.article_id == str(article_id))
                    & (query.status == "open")
                )
                if not table.contains(match):
                    return False
                table.update(
                    {"status": "resolved", "resolved_at": int(resolved_at)},
                    match,
                )
                return True
            finally:
                db.close()

    def list_long_term_quarantine(
        self,
        profile_id: str,
        *,
        status: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        return self._list_long_term_docs(
            "long_term_article_quarantine",
            lambda row: row.get("profile_id") == str(profile_id)
            and (status is None or row.get("status") == str(status)),
            lambda row: (-int(row.get("last_seen_at") or 0), str(row.get("article_id") or "")),
            limit,
        )

    def save_cluster_snapshot(self, snapshot_doc: Dict[str, Any]) -> bool:
        doc = dict(snapshot_doc or {})
        required = (
            "id",
            "profile_id",
            "cluster_id",
            "membership_revision",
            "prompt_version",
            "created_at",
        )
        uniqueness = lambda row: (
            row.get("id") == doc.get("id")
            or (
                row.get("cluster_id") == doc.get("cluster_id")
                and row.get("membership_revision") == doc.get("membership_revision")
                and row.get("prompt_version") == doc.get("prompt_version")
            )
        )
        return self._insert_long_term_doc(
            "threat_cluster_snapshots", doc, required, uniqueness
        )

    @staticmethod
    def _apply_snapshot_revision(db: TinyDB, operation: Dict[str, Any]) -> bool:
        cluster = dict(operation["cluster"])
        snapshot = dict(operation["snapshot"])
        operation_id = str(operation["id"])
        expected_membership = int(operation["expected_membership_revision"])
        expected_summarized = int(operation["expected_summarized_revision"])
        query = Query()
        cluster_table = db.table("threat_clusters")
        snapshot_table = db.table("threat_cluster_snapshots")
        cluster_match = query.id == str(cluster["id"])
        current = cluster_table.get(cluster_match)
        if current is None or int(current.get("membership_revision") or 0) != expected_membership:
            return False
        current_summarized = int(current.get("summarized_revision") or 0)
        target_summarized = int(cluster["summarized_revision"])
        if current_summarized not in {expected_summarized, target_summarized}:
            return False
        snapshot_match = (query.id == str(snapshot["id"])) | (
            (query.cluster_id == str(snapshot["cluster_id"]))
            & (query.membership_revision == int(snapshot["membership_revision"]))
            & (query.prompt_version == str(snapshot["prompt_version"]))
        )
        existing_snapshot = snapshot_table.get(snapshot_match)
        if existing_snapshot is None:
            pending_snapshot = dict(snapshot)
            pending_snapshot[_SNAPSHOT_PENDING_FIELD] = operation_id
            snapshot_table.insert(pending_snapshot)
            existing_snapshot = pending_snapshot
        elif str(existing_snapshot.get("id")) != str(snapshot["id"]):
            return False
        snapshot_pending = str(existing_snapshot.get(_SNAPSHOT_PENDING_FIELD) or "")
        if current_summarized != target_summarized:
            cluster[_SNAPSHOT_OPERATION_FIELD] = operation_id

            def replace_document(row):
                row.clear()
                row.update(cluster)

            cluster_table.update(replace_document, cluster_match)
        else:
            if str(current.get("latest_snapshot_id") or "") != str(snapshot["id"]):
                return False
            if snapshot_pending and snapshot_pending != operation_id:
                return False
            if snapshot_pending and str(current.get(_SNAPSHOT_OPERATION_FIELD) or "") != operation_id:
                return False
        if snapshot_pending:
            snapshot_table.update(
                delete_field(_SNAPSHOT_PENDING_FIELD),
                (query.id == str(snapshot["id"]))
                & (query[_SNAPSHOT_PENDING_FIELD] == snapshot_pending),
            )
        return True

    def save_cluster_snapshot_revision(
        self,
        cluster_doc: Dict[str, Any],
        snapshot_doc: Dict[str, Any],
        *,
        expected_membership_revision: int,
        expected_summarized_revision: int,
    ) -> bool:
        """Journal a snapshot and its cluster summary revision as one operation."""

        cluster = dict(cluster_doc or {})
        snapshot = dict(snapshot_doc or {})
        if (
            not cluster.get("id")
            or not snapshot.get("id")
            or str(cluster.get("id")) != str(snapshot.get("cluster_id"))
            or str(cluster.get("profile_id")) != str(snapshot.get("profile_id"))
            or int(cluster.get("membership_revision") or -1)
            != int(expected_membership_revision)
            or int(cluster.get("summarized_revision") or -1)
            != int(snapshot.get("membership_revision") or -1)
            or not int(expected_summarized_revision)
            < int(cluster.get("summarized_revision") or -1)
            <= int(expected_membership_revision)
        ):
            raise ValueError("snapshot and cluster revisions or identities do not match")
        operation_id = f"{cluster['id']}:{expected_membership_revision}:{snapshot['prompt_version']}"
        operation = {
            "id": operation_id,
            "cluster": cluster,
            "snapshot": snapshot,
            "expected_membership_revision": int(expected_membership_revision),
            "expected_summarized_revision": int(expected_summarized_revision),
            "created_at": int(time.time()),
        }
        with _long_term_file_lock(self.path):
            db = self._db()
            try:
                journal = db.table("long_term_snapshot_journal")
                query = Query()
                for pending in list(journal):
                    if self._apply_snapshot_revision(db, dict(pending)):
                        journal.remove(query.id == str(pending.get("id")))
                current = db.table("threat_clusters").get(
                    query.id == str(cluster["id"])
                )
                if current is None or (
                    int(current.get("membership_revision") or 0)
                    != int(expected_membership_revision)
                    or int(current.get("summarized_revision") or 0)
                    != int(expected_summarized_revision)
                ):
                    return False
                if journal.contains(query.id == operation_id):
                    return False
                journal.insert(operation)
                if not self._apply_snapshot_revision(db, operation):
                    return False
                journal.remove(query.id == operation_id)
                return True
            finally:
                db.close()

    def list_cluster_snapshots(
        self,
        profile_id: str,
        *,
        cluster_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        return self._list_long_term_docs(
            "threat_cluster_snapshots",
            lambda row: row.get("profile_id") == str(profile_id)
            and not str(row.get(_SNAPSHOT_PENDING_FIELD) or "").strip()
            and (cluster_id is None or row.get("cluster_id") == str(cluster_id)),
            lambda row: (-int(row.get("created_at") or 0), str(row.get("id") or "")),
            limit,
        )

    def get_cluster_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        row = self._get_long_term_doc("threat_cluster_snapshots", str(snapshot_id))
        if row and str(row.get(_SNAPSHOT_PENDING_FIELD) or "").strip():
            return None
        return row

    def save_threat_landscape_report(self, report_doc: Dict[str, Any]) -> bool:
        doc = dict(report_doc or {})
        return self._insert_long_term_doc(
            "threat_landscape_reports",
            doc,
            ("id", "profile_id", "period_end_ts", "created_at"),
            lambda row: row.get("id") == doc.get("id"),
        )

    def get_threat_landscape_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        return self._get_long_term_doc("threat_landscape_reports", str(report_id))

    def list_threat_landscape_reports(
        self, profile_id: str, *, limit: int = 100
    ) -> List[Dict[str, Any]]:
        return self._list_long_term_docs(
            "threat_landscape_reports",
            lambda row: row.get("profile_id") == str(profile_id),
            lambda row: (-int(row.get("period_end_ts") or 0), str(row.get("id") or "")),
            limit,
        )

    def create_long_term_run(self, run_doc: Dict[str, Any]) -> bool:
        doc = dict(run_doc or {})
        return self._insert_long_term_doc(
            "long_term_runs",
            doc,
            ("id", "profile_id", "run_type", "started_at", "status"),
            lambda row: row.get("id") == doc.get("id"),
        )

    def get_long_term_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self._get_long_term_doc("long_term_runs", str(run_id))

    def list_long_term_runs(
        self, profile_id: str, *, limit: int = 100
    ) -> List[Dict[str, Any]]:
        return self._list_long_term_docs(
            "long_term_runs",
            lambda row: row.get("profile_id") == str(profile_id),
            lambda row: (-int(row.get("started_at") or 0), str(row.get("id") or "")),
            limit,
        )

    def update_long_term_run(
        self,
        run_id: str,
        *,
        expected_status: str,
        fields: Dict[str, Any],
    ) -> bool:
        updates = dict(fields or {})
        updates.pop("id", None)
        with _long_term_file_lock(self.path):
            db = self._db()
            try:
                table = db.table("long_term_runs")
                query = Query()
                match = (query.id == str(run_id)) & (query.status == str(expected_status))
                row = table.get(match)
                if not row:
                    return False
                doc = dict(row)
                doc.update(updates)
                table.update(doc, match)
                return True
            finally:
                db.close()

    def _get_long_term_doc(self, table_name: str, document_id: str) -> Optional[Dict[str, Any]]:
        db = self._db()
        try:
            row = db.table(table_name).get(Query().id == str(document_id))
            return dict(row) if row else None
        finally:
            db.close()

    def _list_long_term_docs(self, table_name, predicate, sort_key, limit):
        db = self._db()
        try:
            rows = [dict(row) for row in db.table(table_name) if predicate(row)]
            rows.sort(key=sort_key)
            return rows[: max(1, int(limit))]
        finally:
            db.close()

    def _insert_long_term_doc(self, table_name, doc, required, uniqueness) -> bool:
        if any(doc.get(field) is None for field in required):
            raise ValueError("document is incomplete")
        with _long_term_file_lock(self.path):
            db = self._db()
            try:
                table = db.table(table_name)
                if table.contains(uniqueness):
                    return False
                table.insert(doc)
                return True
            finally:
                db.close()

    def put_temp_summary(self, job_id: int, payload: Dict[str, Any]) -> None:
        db = self._db()
        t = db.table("temp_summaries")
        T = Query()
        doc = dict(payload or {})
        doc["job_id"] = int(job_id)
        if "created_at" not in doc:
            doc["created_at"] = int(time.time())
        t.upsert(doc, T.job_id == int(job_id))
        db.close()

    def save_temp_summary(self, job_id: int, summary_text: str, meta: Dict[str, Any]) -> None:
        self.put_temp_summary(job_id, {"summary": summary_text, "meta": meta or {}})

    def get_temp_summary(self, job_id: int) -> Optional[Dict[str, Any]]:
        db = self._db()
        t = db.table("temp_summaries")
        T = Query()
        rows = t.search(T.job_id == int(job_id))
        db.close()
        return rows[0] if rows else None

    def _run_long_term_cleanup(
        self, *, cutoff: int, db: Optional[TinyDB] = None
    ) -> Dict[str, int]:
        """Remove expired long-term history while preserving live provenance."""

        removed = {
            "long_term_reports": 0,
            "long_term_snapshots": 0,
            "long_term_clusters": 0,
            "long_term_memberships": 0,
            "long_term_runs": 0,
            "long_term_quarantine": 0,
        }
        managed_db = db is None
        if db is None:
            db = self._db()
        try:
            reports = db.table("threat_landscape_reports")
            retained_snapshot_ids = {
                str(snapshot_id)
                for report in reports
                if int(report.get("period_end_ts") or 0) >= cutoff
                for snapshot_id in report.get("input_snapshot_ids") or []
            }
            before = len(reports)
            reports.remove(lambda row: int(row.get("period_end_ts") or 0) < cutoff)
            removed["long_term_reports"] = max(0, before - len(reports))

            snapshots = db.table("threat_cluster_snapshots")
            before = len(snapshots)
            snapshots.remove(
                lambda row: int(row.get("created_at") or 0) < cutoff
                and str(row.get("id") or "") not in retained_snapshot_ids
            )
            removed["long_term_snapshots"] = max(0, before - len(snapshots))
            protected_cluster_ids = {
                str(row.get("cluster_id") or "") for row in snapshots
            }
            protected_cluster_ids.update(
                str(row.get("cluster", {}).get("id") or "")
                for row in db.table("long_term_snapshot_journal")
                if isinstance(row.get("cluster"), dict)
            )
            protected_cluster_ids.update(
                str(row.get("cluster", {}).get("id") or "")
                for row in db.table("long_term_assignment_journal")
                if isinstance(row.get("cluster"), dict)
            )
            for row in db.table("long_term_cluster_reconciliations"):
                protected_cluster_ids.update(
                    str(value) for value in row.get("source_cluster_ids") or []
                )
            for row in db.table("long_term_reconciliation_journal"):
                protected_cluster_ids.update(
                    str(value) for value in row.get("source_cluster_ids") or []
                )
            for table_name in (
                "long_term_cluster_membership_edits",
                "long_term_membership_edit_journal",
            ):
                for row in db.table(table_name):
                    protected_cluster_ids.update(
                        str(row.get(field) or "")
                        for field in ("source_cluster_id", "target_cluster_id")
                    )
            protected_cluster_ids.discard("")

            clusters = db.table("threat_clusters")
            expired_cluster_ids = {
                str(row.get("id") or "")
                for row in clusters
                if str(row.get("status") or "") == "closed"
                and int(row.get("last_seen_ts") or 0) < cutoff
                and str(row.get("id") or "") not in protected_cluster_ids
            }
            expired_cluster_ids.discard("")
            before = len(clusters)
            clusters.remove(lambda row: str(row.get("id") or "") in expired_cluster_ids)
            removed["long_term_clusters"] = max(0, before - len(clusters))

            memberships = db.table("threat_cluster_memberships")
            before = len(memberships)
            memberships.remove(
                lambda row: str(row.get("cluster_id") or "") in expired_cluster_ids
            )
            removed["long_term_memberships"] = max(0, before - len(memberships))

            runs = db.table("long_term_runs")
            before = len(runs)
            runs.remove(
                lambda row: str(row.get("status") or "") in {"done", "failed", "blocked"}
                and int(row.get("finished_at") or row.get("started_at") or 0) < cutoff
            )
            removed["long_term_runs"] = max(0, before - len(runs))

            quarantine = db.table("long_term_article_quarantine")
            before = len(quarantine)
            quarantine.remove(
                lambda row: str(row.get("status") or "") == "resolved"
                and int(row.get("resolved_at") or 0) < cutoff
            )
            removed["long_term_quarantine"] = max(0, before - len(quarantine))
            return removed
        finally:
            if managed_db:
                db.close()

    def run_cleanup(self, pol: CleanupPolicy) -> Dict[str, int]:
        """
        Cleanup for TinyDB schema:
        tables: articles, summary_docs, temp_summaries, jobs
        """
        now = int(time.time())
        cut_articles = now - pol.articles_days * 86400
        cut_daily = now - pol.daily_summaries_days * 86400
        cut_weekly = now - pol.weekly_summaries_days * 86400
        cut_temp = now - pol.temp_summaries_days * 86400
        cut_jobs = now - pol.jobs_days * 86400
        cut_long_term = (
            now - pol.long_term_days * 86400 if pol.long_term_days > 0 else -1
        )

        removed = {"articles": 0, "summary_docs": 0, "temp_summaries": 0, "jobs": 0}

        # TinyDB import here to avoid dependency if user doesn't use it
        from tinydb import TinyDB

        with _article_file_lock(self.path), _long_term_file_lock(self.path):
            db = TinyDB(self.path)
            try:
                # Articles
                at = db.table("articles")
                # remove uses a predicate for each row
                before = len(at)
                at.remove(
                    lambda r: int(r.get("published_ts") or r.get("fetched_at") or 0)
                    < cut_articles
                )
                removed["articles"] = max(0, before - len(at))

                # Temp summaries
                tt = db.table("temp_summaries")
                before = len(tt)
                tt.remove(lambda r: int(r.get("created_at") or 0) < cut_temp)
                removed["temp_summaries"] = max(0, before - len(tt))

                # Jobs (only done/failed)
                jt = db.table("jobs")
                before = len(jt)

                def job_old_finished(r: Dict[str, Any]) -> bool:
                    ts = int(r.get("finished_at") or r.get("created_at") or 0)
                    st = str(r.get("status") or "")
                    return ts < cut_jobs and st in ("done", "failed")

                jt.remove(job_old_finished)
                removed["jobs"] = max(0, before - len(jt))

                # Summary docs
                sd = db.table("summary_docs")
                before = len(sd)

                def sum_should_remove(r: Dict[str, Any]) -> bool:
                    created = int(r.get("created") or 0)
                    # we need prompt_package; in tinydb it is stored inside the doc itself
                    pkg = ""
                    sel = r.get("selection")
                    if isinstance(sel, dict):
                        pkg = str(sel.get("prompt_package") or "").lower().strip()
                    kind = "other"
                    if "weekly" in pkg:
                        kind = "weekly"
                    elif "daily" in pkg:
                        kind = "daily"

                    if kind == "daily":
                        return created < cut_daily
                    if kind == "weekly":
                        return created < cut_weekly
                    return created < cut_weekly

                sd.remove(sum_should_remove)
                removed["summary_docs"] = max(0, before - len(sd))
                removed.update(self._run_long_term_cleanup(cutoff=cut_long_term, db=db))
            finally:
                db.close()
        return removed

    # ============================================================================
    # Tag management methods
    # ============================================================================

    def add_tag(
        self,
        name: str,
        category: str = "GENERAL",
        description: Optional[str] = None,
    ) -> Optional[int]:
        """Add a new tag to the database."""
        if not name or not isinstance(name, str):
            return None

        name = name.strip().lower()
        if not name:
            return None

        db = self._db()
        try:
            t = db.table("tags")
            Q = Query()

            # Check if tag already exists
            existing = t.search(Q.name == name)
            if existing:
                return int(existing[0].doc_id)

            # Insert new tag
            tag_id = t.insert({
                "name": name,
                "category": category,
                "description": description,
                "created_at": int(time.time()),
            })
            return int(tag_id)
        finally:
            db.close()

    def get_tag_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a tag by name (case-insensitive)."""
        if not name or not isinstance(name, str):
            return None

        name = name.strip().lower()
        if not name:
            return None

        db = self._db()
        try:
            Q = Query()
            rows = db.table("tags").search(Q.name == name)
            if rows:
                row = rows[0]
                try:
                    tag_id = int(getattr(row, "doc_id"))
                except Exception:
                    tag_id = int(row.get("id", 0))
                tag_dict = {
                    "id": tag_id,
                    "name": row.get("name"),
                    "category": row.get("category", "GENERAL"),
                    "description": row.get("description"),
                    "created_at": row.get("created_at"),
                }
                # Include embedding_vector if present
                if "embedding_vector" in row:
                    tag_dict["embedding_vector"] = row.get("embedding_vector")
                    tag_dict["embedding_model"] = row.get("embedding_model", "")
                    tag_dict["embedding_source_hash"] = row.get("embedding_source_hash", "")
                    tag_dict["embedding_updated_at"] = row.get("embedding_updated_at")
                return tag_dict
            return None
        finally:
            db.close()

    def get_all_tags(self) -> List[Dict[str, Any]]:
        """Get all tags."""
        db = self._db()
        try:
            rows = db.table("tags").all()
            out: List[Dict[str, Any]] = []
            for row in rows:
                try:
                    tag_id = int(getattr(row, "doc_id"))
                except Exception:
                    tag_id = int(row.get("id", 0))
                tag_dict = {
                    "id": tag_id,
                    "name": row.get("name", ""),
                    "category": row.get("category", "GENERAL"),
                    "description": row.get("description"),
                    "synonyms": row.get("synonyms", []),
                    "created_at": row.get("created_at"),
                }
                # Include embedding_vector if present
                if "embedding_vector" in row:
                    tag_dict["embedding_vector"] = row.get("embedding_vector")
                    tag_dict["embedding_model"] = row.get("embedding_model", "")
                    tag_dict["embedding_source_hash"] = row.get("embedding_source_hash", "")
                    tag_dict["embedding_updated_at"] = row.get("embedding_updated_at")
                out.append(tag_dict)
            # Sort by name
            out.sort(key=lambda x: x.get("name", ""))
            return out
        finally:
            db.close()

    def get_tag_relations(self, tag_id: int) -> Dict[str, List[Dict[str, Any]]]:
        tag_id = int(tag_id)
        db = self._db()
        try:
            tags_table = db.table("tags")
            relations = db.table("tag_relations")
            tags_by_id = {}
            for row in tags_table.all():
                related_id = int(getattr(row, "doc_id", row.get("id", 0)))
                tags_by_id[related_id] = {
                    "id": related_id,
                    "name": row.get("name", ""),
                    "category": row.get("category", "GENERAL"),
                    "description": row.get("description"),
                }
            rows = [
                row
                for row in relations.all()
                if row.get("relation_type", PARENT_CHILD_RELATION)
                == PARENT_CHILD_RELATION
                and (
                    int(row.get("parent_tag_id", 0)) == tag_id
                    or int(row.get("child_tag_id", 0)) == tag_id
                )
            ]
            parents = [
                tags_by_id[int(row["parent_tag_id"])]
                for row in rows
                if int(row["child_tag_id"]) == tag_id
                and int(row["parent_tag_id"]) in tags_by_id
            ]
            children = [
                tags_by_id[int(row["child_tag_id"])]
                for row in rows
                if int(row["parent_tag_id"]) == tag_id
                and int(row["child_tag_id"]) in tags_by_id
            ]
            parents.sort(key=lambda tag: str(tag.get("name") or "").casefold())
            children.sort(key=lambda tag: str(tag.get("name") or "").casefold())
            return {"parents": parents, "children": children}
        finally:
            db.close()

    def set_tag_relations(
        self,
        tag_id: int,
        *,
        parent_ids: Optional[List[int]] = None,
        child_ids: Optional[List[int]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        tag_id = int(tag_id)
        db = self._db()
        try:
            tags_by_id = {}
            for row in db.table("tags").all():
                related_id = int(getattr(row, "doc_id", row.get("id", 0)))
                tags_by_id[related_id] = row
            relations = db.table("tag_relations")
            existing_edges = {
                (int(row["parent_tag_id"]), int(row["child_tag_id"]))
                for row in relations.all()
                if row.get("relation_type", PARENT_CHILD_RELATION)
                == PARENT_CHILD_RELATION
            }
            proposed_edges = proposed_parent_child_edges(
                tag_id,
                parent_ids=parent_ids,
                child_ids=child_ids,
                tags_by_id=tags_by_id,
                existing_edges=existing_edges,
            )
            relations.remove(
                lambda row: row.get("relation_type", PARENT_CHILD_RELATION)
                == PARENT_CHILD_RELATION
                and (
                    int(row.get("parent_tag_id", 0)) == tag_id
                    or int(row.get("child_tag_id", 0)) == tag_id
                )
            )
            now = int(time.time())
            for parent_id, child_id in proposed_edges:
                if parent_id != tag_id and child_id != tag_id:
                    continue
                relations.insert(
                    {
                        "relation_type": PARENT_CHILD_RELATION,
                        "parent_tag_id": parent_id,
                        "child_tag_id": child_id,
                        "created_at": now,
                    }
                )
        finally:
            db.close()
        return self.get_tag_relations(tag_id)

    def add_article_tags(
        self,
        article_id: str,
        tag_ids: List,  # Can be List[int] or List[Dict] with 'tag_id' and optional 'reasoning'
    ) -> None:
        """Add tags to an article (replaces existing tags).
        
        Args:
            article_id: Article ID
            tag_ids: List of tag IDs (int) or list of dicts with 'tag_id' and optional 'reasoning'
        """
        if not article_id or not tag_ids:
            return

        article_id = str(article_id).strip()
        if not article_id:
            return

        db = self._db()
        try:
            at = db.table("article_tags")
            Q = Query()

            # Remove existing tags for this article
            at.remove(Q.article_id == article_id)

            # Add new tags
            now_ts = int(time.time())
            for tag_entry in tag_ids:
                # Handle both int (backward compatibility) and dict formats
                if isinstance(tag_entry, dict):
                    tag_id = tag_entry.get("tag_id") or tag_entry.get("id")
                    reasoning = tag_entry.get("reasoning", "")
                else:
                    tag_id = int(tag_entry)
                    reasoning = ""
                
                if not isinstance(tag_id, int) or tag_id <= 0:
                    continue
                
                record = {
                    "article_id": article_id,
                    "tag_id": tag_id,
                    "created_at": now_ts,
                }
                
                # Add reasoning if provided
                if reasoning:
                    record["motivering"] = str(reasoning).strip()
                
                at.insert(record)
        finally:
            db.close()

    def get_article_tags(self, article_id: str) -> List[Dict[str, Any]]:
        """Get all tags for an article."""
        if not article_id:
            return []

        article_id = str(article_id).strip()
        db = self._db()
        try:
            at = db.table("article_tags")
            tags_table = db.table("tags")
            Q = Query()

            # Get tag IDs for this article
            article_tag_rows = at.search(Q.article_id == article_id)
            if not article_tag_rows:
                return []

            tag_ids = [row.get("tag_id") for row in article_tag_rows]

            # Get tag details
            out: List[Dict[str, Any]] = []
            for tag_id in tag_ids:
                tag_rows = tags_table.search(Q.id == tag_id)
                if not tag_rows:
                    # Try doc_id based search
                    try:
                        tag = tags_table.get(doc_id=int(tag_id))
                        if tag:
                            try:
                                tid = int(getattr(tag, "doc_id"))
                            except Exception:
                                tid = int(tag.get("id", tag_id))
                            out.append({
                                "id": tid,
                                "name": tag.get("name", ""),
                                "category": tag.get("category", "GENERAL"),
                                "description": tag.get("description"),
                                "created_at": tag.get("created_at"),
                            })
                    except Exception:
                        pass
                else:
                    row = tag_rows[0]
                    try:
                        tid = int(getattr(row, "doc_id"))
                    except Exception:
                        tid = int(row.get("id", tag_id))
                    out.append({
                        "id": tid,
                        "name": row.get("name", ""),
                        "category": row.get("category", "GENERAL"),
                        "description": row.get("description"),
                        "created_at": row.get("created_at"),
                    })

            # Sort by name
            out.sort(key=lambda x: x.get("name", ""))
            return out
        finally:
            db.close()

    def remove_article_tag(self, article_id: str, tag_id: int) -> bool:
        """Remove a specific tag from an article.
        
        Args:
            article_id: Article ID
            tag_id: Tag ID to remove
            
        Returns:
            True if tag was removed, False otherwise
        """
        if not article_id or not tag_id:
            return False

        article_id = str(article_id).strip()
        db = self._db()
        try:
            at = db.table("article_tags")
            Q = Query()
            removed = at.remove((Q.article_id == article_id) & (Q.tag_id == int(tag_id)))
            return len(removed) > 0
        finally:
            db.close()

    def add_tag_to_article(self, article_id: str, tag_id: int) -> bool:
        """Add a tag to an article without removing existing tags.
        
        Args:
            article_id: Article ID
            tag_id: Tag ID to add
            
        Returns:
            True if tag was added, False if already associated
        """
        if not article_id or not tag_id:
            return False

        article_id = str(article_id).strip()
        tag_id = int(tag_id)
        db = self._db()
        try:
            at = db.table("article_tags")
            Q = Query()
            
            # Check if already exists
            existing = at.search((Q.article_id == article_id) & (Q.tag_id == tag_id))
            if existing:
                return False

            now_ts = int(time.time())
            at.insert({
                "article_id": article_id,
                "tag_id": tag_id,
                "created_at": now_ts,
            })
            return True
        finally:
            db.close()

    def create_tag(
        self, name: str, category: str = "GENERAL", description: str = "", synonyms: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Create a new tag.
        
        Args:
            name: Tag name
            category: Tag category (GENERAL, DOMAIN_ENTITY, etc.)
            description: Optional description
            synonyms: Optional list of synonym strings
            
        Returns:
            Created tag dict, or None if tag already exists
        """
        if not name:
            return None

        name = name.strip()
        category = category.strip() or "GENERAL"
        description = description.strip() if description else ""
        synonyms = [s.strip().lower() for s in (synonyms or [])] if synonyms else []

        db = self._db()
        try:
            tags_table = db.table("tags")
            Q = Query()
            
            # Check if tag already exists
            existing = tags_table.search(Q.name == name)
            if existing:
                return None

            now_ts = int(time.time())
            doc_id = tags_table.insert({
                "name": name,
                "category": category,
                "description": description,
                "synonyms": synonyms,
                "created_at": now_ts,
            })

            return {
                "id": int(doc_id),
                "name": name,
                "category": category,
                "description": description,
                "synonyms": synonyms,
                "created_at": now_ts,
            }
        finally:
            db.close()

    def update_tag(
        self, tag_id: int, name: str = None, category: str = None, description: str = None, synonyms: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Update an existing tag.
        
        Args:
            tag_id: Tag ID to update
            name: New name (optional)
            category: New category (optional)
            description: New description (optional)
            synonyms: New synonyms list (optional)
            
        Returns:
            Updated tag dict, or None if tag not found
        """
        if not tag_id:
            return None

        db = self._db()
        try:
            tags_table = db.table("tags")
            
            # Get current tag
            try:
                tag_row = tags_table.get(doc_id=int(tag_id))
            except Exception:
                return None

            # Prepare updates
            updates = {}
            if name is not None:
                updates["name"] = name.strip()
                updates["embedding_vector"] = None
                updates["embedding_model"] = None
                updates["embedding_source_hash"] = None
                updates["embedding_updated_at"] = None
            if category is not None:
                updates["category"] = category.strip() or "GENERAL"
            if description is not None:
                updates["description"] = description.strip() if description else ""
            if synonyms is not None:
                updates["synonyms"] = [s.strip().lower() for s in synonyms] if synonyms else []

            if not updates:
                # No changes, return current tag
                return {
                    "id": int(tag_id),
                    "name": tag_row.get("name", ""),
                    "category": tag_row.get("category", "GENERAL"),
                    "description": tag_row.get("description", ""),
                    "synonyms": tag_row.get("synonyms", []),
                    "created_at": tag_row.get("created_at", 0),
                }

            # Update the tag
            tags_table.update(updates, doc_ids=[int(tag_id)])

            new_category = updates.get("category")
            if new_category is not None and new_category != str(
                tag_row.get("category") or "GENERAL"
            ):
                relations = db.table("tag_relations")
                relations.remove(
                    lambda row: int(row.get("parent_tag_id", 0)) == int(tag_id)
                    or int(row.get("child_tag_id", 0)) == int(tag_id)
                )

            # Get updated tag
            updated_row = tags_table.get(doc_id=int(tag_id))
            return {
                "id": int(tag_id),
                "name": updated_row.get("name", ""),
                "category": updated_row.get("category", "GENERAL"),
                "description": updated_row.get("description", ""),
                "synonyms": updated_row.get("synonyms", []),
                "created_at": updated_row.get("created_at", 0),
            }
        except Exception as e:
            logger.error(f"Error updating tag: {e}")
            return None
        finally:
            db.close()

    def delete_tag(self, tag_id: int) -> bool:
        """Delete a tag and remove it from all articles.
        
        Args:
            tag_id: Tag ID to delete
            
        Returns:
            True if tag was deleted, False if not found
        """
        if not tag_id:
            return False

        db = self._db()
        try:
            # Delete from article_tags first
            at = db.table("article_tags")
            Q = Query()
            at.remove(Q.tag_id == int(tag_id))
            db.table("tag_relations").remove(
                lambda row: int(row.get("parent_tag_id", 0)) == int(tag_id)
                or int(row.get("child_tag_id", 0)) == int(tag_id)
            )
            
            # Delete the tag
            tags_table = db.table("tags")
            removed = tags_table.remove(doc_ids=[int(tag_id)])
            
            return len(removed) > 0
        except Exception as e:
            logger.error(f"Error deleting tag: {e}")
            return False
        finally:
            db.close()

    def migrate_synonym_to_main_tag(self, main_tag_id: int, synonym_tag_ids: List[int]) -> Tuple[int, int]:
        """
        Migrate articles from synonym tags to main tag and delete synonyms.
        
        When a tag becomes a synonym of another tag:
        1. All articles using the synonym tag get the main tag instead
        2. The synonym tag is deleted from the database
        
        Args:
            main_tag_id: ID of the main tag that synonyms map to
            synonym_tag_ids: List of tag IDs that are now synonyms
            
        Returns:
            Tuple of (articles_migrated, synonyms_deleted)
        """
        if not main_tag_id or not synonym_tag_ids:
            return 0, 0
        
        articles_migrated = 0
        synonyms_deleted = 0
        
        db = self._db()
        try:
            at = db.table("article_tags")
            Q = Query()
            
            # For each synonym tag, find articles and update them
            for synonym_tag_id in synonym_tag_ids:
                if synonym_tag_id == main_tag_id:
                    # Don't process main tag
                    continue
                
                # Find all article_tags entries using this synonym tag
                article_tag_entries = at.search(Q.tag_id == int(synonym_tag_id))
                
                for entry in article_tag_entries:
                    article_id = entry.get("article_id")
                    
                    # Check if article already has main tag
                    existing_main = at.search(
                        (Q.article_id == article_id) & 
                        (Q.tag_id == int(main_tag_id))
                    )
                    
                    if not existing_main:
                        # Add main tag to article
                        at.insert({
                            "article_id": article_id,
                            "tag_id": int(main_tag_id),
                            "timestamp": int(time.time())
                        })
                    
                    # Remove synonym tag from article
                    at.remove(
                        (Q.article_id == article_id) & 
                        (Q.tag_id == int(synonym_tag_id))
                    )
                    articles_migrated += 1
                
                # Delete the synonym tag itself
                tags_table = db.table("tags")
                db.table("tag_relations").remove(
                    lambda row: int(row.get("parent_tag_id", 0))
                    == int(synonym_tag_id)
                    or int(row.get("child_tag_id", 0)) == int(synonym_tag_id)
                )
                removed = tags_table.remove(doc_ids=[int(synonym_tag_id)])
                if removed:
                    synonyms_deleted += 1
                    logger.info(f"[TagMigration] Deleted synonym tag {synonym_tag_id}, migrated {len(article_tag_entries)} articles")
            
            return articles_migrated, synonyms_deleted
        except Exception as e:
            logger.error(f"Error migrating synonym tags: {e}")
            return 0, 0
        finally:
            db.close()

    def update_tag_embedding(
        self,
        tag_id: int,
        embedding_vector: List[float],
        *,
        model: Optional[str] = None,
        source_hash: Optional[str] = None,
    ) -> bool:
        """
        Update the embedding vector for a tag.
        
        Args:
            tag_id: Tag ID (doc_id in TinyDB)
            embedding_vector: List of floats representing the embedding
            
        Returns:
            True if successful, False otherwise
        """
        if not isinstance(tag_id, int) or tag_id <= 0:
            return False
        
        if not embedding_vector or not all(isinstance(x, (int, float)) for x in embedding_vector):
            return False
        
        db = self._db()
        try:
            t = db.table("tags")
            updated = t.update(
                {
                    "embedding_vector": [float(value) for value in embedding_vector],
                    "embedding_model": str(model or ""),
                    "embedding_source_hash": str(source_hash or ""),
                    "embedding_updated_at": int(time.time()),
                },
                doc_ids=[tag_id],
            )
            return bool(updated)
        except Exception as e:
            logger.error(f"Error updating tag embedding: {e}")
            return False
        finally:
            db.close()

    def get_tags_by_embedding_similarity(
        self,
        embedding_vector: List[float],
        similarity_threshold: float = 0.75,
        limit: int = 10,
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find tags with embeddings similar to the given embedding.
        Uses cosine similarity.
        
        Args:
            embedding_vector: Target embedding vector
            similarity_threshold: Minimum similarity score (0.0-1.0)
            limit: Maximum number of results
            
        Returns:
            List of tags sorted by similarity (highest first)
        """
        if not embedding_vector or limit <= 0:
            return []
        
        db = self._db()
        try:
            rows = db.table("tags").all()
            results: List[Tuple[Dict[str, Any], float]] = []
            
            for row in rows:
                if "embedding_vector" not in row or not row.get("embedding_vector"):
                    continue
                if model is not None and str(row.get("embedding_model") or "") != str(model):
                    continue
                
                try:
                    tag_embedding = row.get("embedding_vector")
                    if not isinstance(tag_embedding, list):
                        continue
                    
                    # Compute cosine similarity
                    similarity = self._cosine_similarity(embedding_vector, tag_embedding)
                    
                    if similarity >= similarity_threshold:
                        try:
                            tag_id = int(getattr(row, "doc_id"))
                        except Exception:
                            tag_id = int(row.get("id", 0))
                        
                        tag_dict = {
                            "id": tag_id,
                            "name": row.get("name", ""),
                            "category": row.get("category", "GENERAL"),
                            "description": row.get("description"),
                            "created_at": row.get("created_at"),
                            "embedding_vector": tag_embedding,
                            "_similarity_score": similarity,  # Include similarity for debugging
                        }
                        results.append((tag_dict, similarity))
                except Exception:
                    continue
            
            # Sort by similarity descending
            results.sort(key=lambda x: -x[1])
            
            return [tag for tag, _ in results[:limit]]
        finally:
            db.close()

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        try:
            import math
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(b * b for b in vec2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        except Exception:
            return 0.0

    def cleanup_unused_tags(self, days: int = 30) -> int:
        """Remove tags that haven't been used in X days."""
        cutoff = int(time.time()) - (days * 86400)
        db = self._db()
        try:
            at = db.table("article_tags")
            tags_table = db.table("tags")
            Q = Query()

            # Find tag IDs that have been used
            all_article_tags = at.all()
            used_tag_ids = set()
            for row in all_article_tags:
                created_at = int(row.get("created_at", 0))
                if created_at > cutoff:
                    used_tag_ids.add(int(row.get("tag_id", 0)))

            # Remove tags that are not used and are old
            before = len(tags_table)
            removed_ids = {
                int(getattr(row, "doc_id", row.get("id", 0)))
                for row in tags_table.all()
                if int(row.get("created_at", 0)) < cutoff
                and int(getattr(row, "doc_id", row.get("id", 0))) not in used_tag_ids
            }
            if removed_ids:
                tags_table.remove(doc_ids=sorted(removed_ids))
                db.table("tag_relations").remove(
                    lambda row: int(row.get("parent_tag_id", 0)) in removed_ids
                    or int(row.get("child_tag_id", 0)) in removed_ids
                )
            after = len(tags_table)
            return max(0, before - after)
        finally:
            db.close()

    def get_articles_by_tags(
        self,
        tag_names: List[str],
        match_mode: str = "any",
    ) -> List[Dict[str, Any]]:
        """
        Get articles tagged with one or more tags.

        Args:
            tag_names: List of tag names to search for
            match_mode: "any" (OR) or "all" (AND)

        Returns:
            List of article dicts
        """
        if not tag_names:
            return []

        tag_names_lower = [str(t).strip().lower() for t in tag_names if t]
        if not tag_names_lower:
            return []

        db = self._db()
        try:
            tags_table = db.table("tags")
            at = db.table("article_tags")
            Q = Query()

            # Find tag IDs matching the names
            tag_rows = tags_table.search(
                lambda r: r.get("name", "").lower() in tag_names_lower
            )
            tag_ids = [int(getattr(row, "doc_id", row.get("id", 0))) for row in tag_rows]

            if not tag_ids:
                return []

            # Find articles
            if match_mode == "all":
                # Articles with ALL tags
                article_tag_rows = at.all()
                article_tag_counts: Dict[str, int] = {}
                for row in article_tag_rows:
                    if int(row.get("tag_id", 0)) in tag_ids:
                        article_id = row.get("article_id")
                        article_tag_counts[article_id] = article_tag_counts.get(article_id, 0) + 1

                article_ids = [
                    aid for aid, count in article_tag_counts.items()
                    if count == len(tag_ids)
                ]
            else:
                # Articles with ANY tag
                article_tag_rows = at.search(
                    lambda r: int(r.get("tag_id", 0)) in tag_ids
                )
                article_ids = list(set(row.get("article_id") for row in article_tag_rows))

            if not article_ids:
                return []

            # Fetch article documents
            return self.get_articles_by_ids(article_ids)

        finally:
            db.close()

    def get_all_categories(self) -> List[Dict[str, Any]]:
        """Get all tag categories."""
        db = self._db()
        try:
            categories_table = db.table("tag_categories")
            docs = list(categories_table.all())
            out = []
            for doc in docs:
                try:
                    doc_id = int(getattr(doc, "doc_id"))
                except Exception:
                    doc_id = int(doc.get("id", 0))
                
                out.append({
                    "id": doc_id,
                    "name": doc.get("name", ""),
                    "label": doc.get("label", ""),
                    "bg_color": doc.get("bg_color", "bg-secondary"),
                    "text_color": doc.get("text_color", "text-dark"),
                    "description": doc.get("description", ""),
                    "created_at": doc.get("created_at", 0),
                })
            return out
        finally:
            db.close()

    def get_category(self, category_id: int) -> Optional[Dict[str, Any]]:
        """Get a category by ID."""
        db = self._db()
        try:
            categories_table = db.table("tag_categories")
            doc = categories_table.get(doc_id=category_id)
            if not doc:
                return None
            
            return {
                "id": category_id,
                "name": doc.get("name", ""),
                "label": doc.get("label", ""),
                "bg_color": doc.get("bg_color", "bg-secondary"),
                "text_color": doc.get("text_color", "text-dark"),
                "description": doc.get("description", ""),
                "created_at": doc.get("created_at", 0),
            }
        finally:
            db.close()

    def create_category(
        self,
        name: str,
        label: str,
        bg_color: str = "bg-secondary",
        text_color: str = "text-dark",
        description: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Create a new tag category."""
        if not name or not label:
            return None

        db = self._db()
        try:
            categories_table = db.table("tag_categories")
            Q = Query()
            
            # Check if category already exists
            existing = categories_table.search(Q.name == name)
            if existing:
                return None

            now_ts = int(time.time())
            doc_id = categories_table.insert({
                "name": name,
                "label": label,
                "bg_color": bg_color,
                "text_color": text_color,
                "description": description,
                "created_at": now_ts,
            })

            return {
                "id": int(doc_id),
                "name": name,
                "label": label,
                "bg_color": bg_color,
                "text_color": text_color,
                "description": description,
                "created_at": now_ts,
            }
        finally:
            db.close()

    def update_category(
        self,
        category_id: int,
        label: str = None,
        bg_color: str = None,
        text_color: str = None,
        description: str = None,
    ) -> bool:
        """Update an existing category."""
        db = self._db()
        try:
            categories_table = db.table("tag_categories")
            
            # Build update dict with only non-None values
            update_data = {}
            if label is not None:
                update_data["label"] = label
            if bg_color is not None:
                update_data["bg_color"] = bg_color
            if text_color is not None:
                update_data["text_color"] = text_color
            if description is not None:
                update_data["description"] = description
            
            if not update_data:
                return False
            
            categories_table.update(update_data, doc_ids=[category_id])
            return True
        finally:
            db.close()

    def delete_category(self, category_id: int) -> bool:
        """Delete a category."""
        db = self._db()
        try:
            categories_table = db.table("tag_categories")
            categories_table.remove(doc_ids=[category_id])
            return True
        finally:
            db.close()

    def initialize_default_categories(self) -> None:
        """Initialize default tag categories if they don't exist."""
        db = self._db()
        try:
            categories_table = db.table("tag_categories")
            
            # Define default categories
            defaults = [
                {
                    "name": "GENERAL",
                    "label": "Allmän",
                    "bg_color": "bg-secondary",
                    "text_color": "text-dark",
                },
                {
                    "name": "DOMAIN_ENTITY",
                    "label": "Domän-enhet",
                    "bg_color": "bg-info",
                    "text_color": "text-dark",
                },
                {
                    "name": "VULNERABILITY",
                    "label": "Sårbarhet",
                    "bg_color": "bg-danger",
                    "text_color": "text-white",
                },
                {
                    "name": "THREAT",
                    "label": "Hot",
                    "bg_color": "bg-danger",
                    "text_color": "text-white",
                },
                {
                    "name": "LOCATION",
                    "label": "Plats",
                    "bg_color": "bg-success",
                    "text_color": "text-dark",
                },
                {
                    "name": "PERSON",
                    "label": "Person",
                    "bg_color": "bg-warning",
                    "text_color": "text-dark",
                },
                {
                    "name": "ORGANIZATION",
                    "label": "Organisation",
                    "bg_color": "bg-warning",
                    "text_color": "text-dark",
                },
                {
                    "name": "PRODUCT",
                    "label": "Produkt",
                    "bg_color": "bg-warning",
                    "text_color": "text-dark",
                },
            ]
            
            Q = Query()
            now_ts = int(time.time())
            
            for default in defaults:
                # Check if category exists
                existing = categories_table.search(Q.name == default["name"])
                if not existing:
                    categories_table.insert({
                        "name": default["name"],
                        "label": default["label"],
                        "bg_color": default["bg_color"],
                        "text_color": default["text_color"],
                        "description": "",
                        "created_at": now_ts,
                    })

            # Migrate tags created before the dedicated CVE category existed.
            tags_table = db.table("tags")
            for tag in tags_table.all():
                if (
                    is_cve_tag(tag.get("name"))
                    and tag.get("category") != VULNERABILITY_TAG_CATEGORY
                ):
                    db.table("tag_relations").remove(
                        lambda row: int(row.get("parent_tag_id", 0)) == int(tag.doc_id)
                        or int(row.get("child_tag_id", 0)) == int(tag.doc_id)
                    )
                    tags_table.update(
                        {"category": VULNERABILITY_TAG_CATEGORY},
                        doc_ids=[tag.doc_id],
                    )
        finally:
            db.close()
