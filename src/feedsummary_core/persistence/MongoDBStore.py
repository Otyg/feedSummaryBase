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

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterator
from typing import Any, Dict, List, Optional, Tuple

from feedsummary_core.persistence.CleanUpPolicy import CleanupPolicy
from feedsummary_core.persistence.tag_relations import (
    PARENT_CHILD_RELATION,
    proposed_parent_child_edges,
)
from feedsummary_core.tagging_rules import VULNERABILITY_TAG_CATEGORY, is_cve_tag

try:
    from pymongo import ASCENDING, DESCENDING, MongoClient
    from pymongo.errors import DuplicateKeyError
except ImportError:  # pragma: no cover - exercised when the optional backend is unused
    ASCENDING = 1
    DESCENDING = -1
    MongoClient = None  # type: ignore[assignment]

    class DuplicateKeyError(Exception):
        pass


logger = logging.getLogger(__name__)

_DEFAULT_CATEGORIES = [
    ("GENERAL", "Allmän", "bg-secondary", "text-dark"),
    ("DOMAIN_ENTITY", "Domän-enhet", "bg-info", "text-dark"),
    ("VULNERABILITY", "Sårbarhet", "bg-danger", "text-white"),
    ("THREAT", "Hot", "bg-danger", "text-white"),
    ("LOCATION", "Plats", "bg-success", "text-dark"),
    ("PERSON", "Person", "bg-warning", "text-dark"),
    ("ORGANIZATION", "Organisation", "bg-warning", "text-dark"),
    ("PRODUCT", "Produkt", "bg-warning", "text-dark"),
]


def _now_ts() -> int:
    return int(time.time())


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _normalize_summary_id(value: Any) -> Optional[str]:
    summary_id = str(value or "").strip()
    if summary_id.lower() in {"", "none", "null"}:
        return None
    return summary_id


def _sort_ts(doc: Dict[str, Any]) -> int:
    published_ts = _safe_int(doc.get("published_ts"))
    return published_ts if published_ts > 0 else _safe_int(doc.get("fetched_at"))


def _public_doc(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if doc is None:
        return None
    result = {key: value for key, value in doc.items() if not key.startswith("_mongo_")}
    result.pop("_id", None)
    return result


class MongoDBStore:
    """MongoDB-backed persistence store with TinyDB/SQLite feature parity."""

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        database: str = "feedsummary",
        *,
        client: Any = None,
        connect_timeout_ms: int = 5000,
        initialize_schema: bool = True,
    ):
        if not database or not str(database).strip():
            raise ValueError("database must be a non-empty string")

        self.uri = str(uri)
        self.database_name = str(database).strip()
        self._owns_client = client is None
        if client is None:
            if MongoClient is None:
                raise ImportError(
                    "MongoDB persistence requires pymongo; install feedsummary-core[mongodb]"
                )
            client = MongoClient(self.uri, serverSelectionTimeoutMS=int(connect_timeout_ms))

        self.client = client
        self.db = client[self.database_name]
        if initialize_schema:
            self._init_db()

    def close(self) -> None:
        if self._owns_client and self.client is not None:
            self.client.close()

    def __enter__(self) -> "MongoDBStore":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def _init_db(self) -> None:
        indexes = (
            (self.db.articles, [("source", ASCENDING), ("_mongo_sort_ts", ASCENDING)], {}),
            (self.db.articles, [("_mongo_sort_ts", ASCENDING)], {}),
            (self.db.articles, [("fetched_at", ASCENDING), ("_id", ASCENDING)], {}),
            (
                self.db.articles,
                [("url", ASCENDING)],
                {"unique": True, "partialFilterExpression": {"url": {"$gt": ""}}},
            ),
            (self.db.summary_docs, [("created", DESCENDING)], {}),
            (self.db.jobs, [("created_at", DESCENDING)], {}),
            (self.db.temp_summaries, [("created_at", ASCENDING)], {}),
            (self.db.tags, [("normalized_name", ASCENDING)], {"unique": True}),
            (
                self.db.article_tags,
                [("article_id", ASCENDING), ("tag_id", ASCENDING)],
                {"unique": True},
            ),
            (self.db.article_tags, [("tag_id", ASCENDING)], {}),
            (
                self.db.tag_relations,
                [
                    ("relation_type", ASCENDING),
                    ("parent_tag_id", ASCENDING),
                    ("child_tag_id", ASCENDING),
                ],
                {"unique": True},
            ),
            (self.db.tag_relations, [("child_tag_id", ASCENDING)], {}),
            (self.db.tag_categories, [("name", ASCENDING)], {"unique": True}),
            (
                self.db.threat_clusters,
                [
                    ("profile_id", ASCENDING),
                    ("status", ASCENDING),
                    ("embedding_model", ASCENDING),
                    ("embedding_dimension", ASCENDING),
                    ("last_seen_ts", DESCENDING),
                ],
                {},
            ),
            (
                self.db.threat_cluster_memberships,
                [("profile_id", ASCENDING), ("article_id", ASCENDING)],
                {"unique": True},
            ),
            (
                self.db.threat_cluster_memberships,
                [("cluster_id", ASCENDING), ("assigned_at", ASCENDING)],
                {},
            ),
            (
                self.db.long_term_article_quarantine,
                [("profile_id", ASCENDING), ("article_id", ASCENDING)],
                {"unique": True},
            ),
            (
                self.db.long_term_article_quarantine,
                [("profile_id", ASCENDING), ("status", ASCENDING), ("last_seen_at", DESCENDING)],
                {},
            ),
            (
                self.db.threat_cluster_snapshots,
                [
                    ("cluster_id", ASCENDING),
                    ("membership_revision", ASCENDING),
                    ("prompt_version", ASCENDING),
                ],
                {"unique": True},
            ),
            (
                self.db.threat_cluster_snapshots,
                [("profile_id", ASCENDING), ("created_at", DESCENDING)],
                {},
            ),
            (
                self.db.threat_landscape_reports,
                [("profile_id", ASCENDING), ("period_end_ts", DESCENDING)],
                {},
            ),
            (
                self.db.long_term_runs,
                [
                    ("profile_id", ASCENDING),
                    ("run_type", ASCENDING),
                    ("started_at", DESCENDING),
                ],
                {},
            ),
            (self.db.long_term_runs, [("status", ASCENDING)], {}),
        )
        for collection, keys, options in indexes:
            collection.create_index(keys, **options)

        self._seed_counter("jobs", self.db.jobs)
        self._seed_counter("tags", self.db.tags)
        self._seed_counter("tag_categories", self.db.tag_categories)

    def _seed_counter(self, name: str, collection: Any) -> None:
        latest = collection.find_one({"_id": {"$type": "number"}}, sort=[("_id", DESCENDING)])
        highest = _safe_int(latest.get("_id")) if latest else 0
        self.db.counters.update_one(
            {"_id": name},
            {"$max": {"seq": highest}},
            upsert=True,
        )

    def _next_id(self, name: str) -> int:
        counter = self.db.counters.find_one_and_update(
            {"_id": name},
            {"$inc": {"seq": 1}},
            upsert=True,
            return_document=True,
        )
        return int(counter["seq"])

    # Articles

    def get_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        return _public_doc(self.db.articles.find_one({"_id": str(article_id)}))

    def upsert_article(self, article_doc: Dict[str, Any]) -> None:
        if not isinstance(article_doc, dict):
            raise ValueError("article_doc must be a dict")
        if not article_doc.get("id"):
            raise ValueError("article_doc must contain 'id'")

        doc = dict(article_doc)
        doc["id"] = str(doc["id"])
        doc["_id"] = doc["id"]
        existing = self.db.articles.find_one(
            {"_id": doc["_id"]},
            {
                "similarity_embedding_vector": 1,
                "similarity_embedding_model": 1,
                "similarity_embedding_source_hash": 1,
                "similarity_embedding_instruction": 1,
                "similarity_embedding_updated_at": 1,
                "tagging_embedding_vector": 1,
                "tagging_embedding_model": 1,
                "tagging_embedding_source_hash": 1,
                "tagging_embedding_instruction": 1,
                "tagging_embedding_updated_at": 1,
            },
        )
        if existing:
            for field in (
                "similarity_embedding_vector",
                "similarity_embedding_model",
                "similarity_embedding_source_hash",
                "similarity_embedding_instruction",
                "similarity_embedding_updated_at",
                "tagging_embedding_vector",
                "tagging_embedding_model",
                "tagging_embedding_source_hash",
                "tagging_embedding_instruction",
                "tagging_embedding_updated_at",
            ):
                if field not in doc and field in existing:
                    doc[field] = existing[field]
        doc["_mongo_sort_ts"] = _sort_ts(doc)
        self.db.articles.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    def update_article_embedding(
        self,
        article_id: str,
        embedding_vector: List[float],
        *,
        model: Optional[str] = None,
        source_hash: Optional[str] = None,
        purpose: str = "similarity",
        instruction: Optional[str] = None,
    ) -> bool:
        """Persist a purpose-specific embedding for an existing article."""
        if (
            not article_id
            or not embedding_vector
            or not all(isinstance(value, (int, float)) for value in embedding_vector)
        ):
            return False
        purpose = str(purpose).strip().lower()
        if purpose not in {"similarity", "tagging"}:
            raise ValueError(f"Unsupported article embedding purpose: {purpose}")
        prefix = f"{purpose}_embedding"
        result = self.db.articles.update_one(
            {"_id": str(article_id)},
            {
                "$set": {
                    f"{prefix}_vector": [float(value) for value in embedding_vector],
                    f"{prefix}_model": str(model or ""),
                    f"{prefix}_source_hash": str(source_hash or ""),
                    f"{prefix}_instruction": str(instruction or "").strip(),
                    f"{prefix}_updated_at": _now_ts(),
                },
                "$unset": {
                    "embedding_vector": "",
                    "embedding_model": "",
                    "embedding_source_hash": "",
                    "embedding_updated_at": "",
                },
            },
        )
        return result.matched_count > 0

    def list_articles(self, limit: int = 2000) -> List[Dict[str, Any]]:
        cursor = self.db.articles.find().sort("_mongo_sort_ts", ASCENDING).limit(max(0, int(limit)))
        return [_public_doc(doc) for doc in cursor]  # type: ignore[misc]

    def iter_articles(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Yield every article oldest-first without the list API's default cap."""
        cursor = self.db.articles.find().sort("_mongo_sort_ts", ASCENDING)
        if limit is not None and int(limit) > 0:
            cursor = cursor.limit(int(limit))
        for doc in cursor:
            article = _public_doc(doc)
            if isinstance(article, dict):
                yield article

    def list_articles_by_filter(
        self,
        *,
        sources: List[str],
        since_ts: int,
        until_ts: Optional[int] = None,
        limit: int = 2000,
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {"_mongo_sort_ts": {"$gte": _safe_int(since_ts)}}
        normalized_sources = [
            str(source).strip() for source in sources or [] if str(source).strip()
        ]
        if normalized_sources:
            query["source"] = {"$in": normalized_sources}
        if until_ts is not None:
            query["_mongo_sort_ts"]["$lte"] = _safe_int(until_ts)
        cursor = (
            self.db.articles.find(query).sort("_mongo_sort_ts", ASCENDING).limit(max(0, int(limit)))
        )
        return [_public_doc(doc) for doc in cursor]  # type: ignore[misc]

    def list_unsummarized_articles(self, limit: int = 200) -> List[Dict[str, Any]]:
        cursor = (
            self.db.articles.find({"summarized": {"$ne": True}})
            .sort("_mongo_sort_ts", ASCENDING)
            .limit(max(0, int(limit)))
        )
        return [_public_doc(doc) for doc in cursor]  # type: ignore[misc]

    def mark_articles_summarized(self, article_ids: List[str]) -> None:
        ids = [str(article_id) for article_id in article_ids or [] if str(article_id).strip()]
        if ids:
            self.db.articles.update_many(
                {"_id": {"$in": ids}},
                {"$set": {"summarized": True, "summarized_at": _now_ts()}},
            )

    def get_articles_by_ids(self, article_ids: List[str]) -> List[Dict[str, Any]]:
        ids = [str(article_id) for article_id in article_ids or [] if str(article_id).strip()]
        if not ids:
            return []
        found = {
            str(doc["_id"]): _public_doc(doc)
            for doc in self.db.articles.find({"_id": {"$in": ids}})
        }
        return [found[article_id] for article_id in ids if article_id in found]  # type: ignore[misc]

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
        after_ts = max(0, _safe_int(after_fetched_at))
        query: Dict[str, Any] = {
            "$or": [
                {"fetched_at": {"$gt": after_ts}},
                {"fetched_at": after_ts, "_id": {"$gt": str(after_article_id or "")}},
            ]
        }
        if until_fetched_at is not None:
            query = {
                "$and": [query, {"fetched_at": {"$lte": _safe_int(until_fetched_at)}}]
            }
        normalized_sources = [
            str(source).strip() for source in sources or [] if str(source).strip()
        ]
        if normalized_sources:
            source_query = {"source": {"$in": normalized_sources}}
            query = {"$and": [query, source_query]}
        cursor = (
            self.db.articles.find(query)
            .sort([("fetched_at", ASCENDING), ("_id", ASCENDING)])
            .limit(max(1, int(limit)))
        )
        return [_public_doc(doc) for doc in cursor]  # type: ignore[misc]

    def get_long_term_cursor(self, profile_id: str) -> Dict[str, Any]:
        profile_id = str(profile_id or "").strip()
        if not profile_id:
            raise ValueError("profile_id must not be empty")
        doc = _public_doc(self.db.long_term_state.find_one({"_id": profile_id}))
        if doc:
            return doc
        return {
            "profile_id": profile_id,
            "cursor_fetched_at": 0,
            "cursor_article_id": "",
            "lease_owner": None,
            "lease_until": 0,
            "updated_at": 0,
        }

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
        now_ts = _safe_int(now_ts)
        try:
            result = self.db.long_term_state.find_one_and_update(
                {
                    "_id": profile_id,
                    "$or": [
                        {"lease_owner": owner_id},
                        {"lease_owner": None},
                        {"lease_owner": {"$exists": False}},
                        {"lease_until": {"$lte": now_ts}},
                    ],
                },
                {
                    "$setOnInsert": {
                        "profile_id": profile_id,
                        "cursor_fetched_at": 0,
                        "cursor_article_id": "",
                    },
                    "$set": {
                        "lease_owner": owner_id,
                        "lease_until": now_ts + int(lease_seconds),
                        "updated_at": now_ts,
                    },
                },
                upsert=True,
                return_document=True,
            )
            return result is not None
        except DuplicateKeyError:
            return False

    def release_long_term_lease(self, profile_id: str, owner_id: str) -> bool:
        result = self.db.long_term_state.update_one(
            {"_id": str(profile_id), "lease_owner": str(owner_id)},
            {
                "$set": {
                    "lease_owner": None,
                    "lease_until": 0,
                    "updated_at": _now_ts(),
                }
            },
        )
        return result.matched_count > 0

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
        expected = (_safe_int(expected_fetched_at), str(expected_article_id or ""))
        new_cursor = (_safe_int(fetched_at), str(article_id or ""))
        if new_cursor < expected or new_cursor[0] < 1 or not new_cursor[1]:
            raise ValueError("new cursor must be complete and cannot move backwards")
        result = self.db.long_term_state.update_one(
            {
                "_id": str(profile_id),
                "lease_owner": str(owner_id),
                "lease_until": {"$gt": _safe_int(now_ts)},
                "cursor_fetched_at": expected[0],
                "cursor_article_id": expected[1],
            },
            {
                "$set": {
                    "cursor_fetched_at": new_cursor[0],
                    "cursor_article_id": new_cursor[1],
                    "updated_at": _safe_int(now_ts),
                }
            },
        )
        return result.matched_count > 0

    def get_threat_cluster(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        return _public_doc(self.db.threat_clusters.find_one({"_id": str(cluster_id)}))

    def list_threat_clusters(
        self,
        profile_id: str,
        *,
        statuses: Optional[List[str]] = None,
        min_last_seen_ts: Optional[int] = None,
        embedding_model: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        embedding_instruction: Optional[str] = None,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {"profile_id": str(profile_id)}
        normalized_statuses = [str(status) for status in statuses or [] if str(status)]
        if normalized_statuses:
            query["status"] = {"$in": normalized_statuses}
        if min_last_seen_ts is not None:
            query["last_seen_ts"] = {"$gte": _safe_int(min_last_seen_ts)}
        if embedding_model is not None:
            query["embedding_model"] = str(embedding_model)
        if embedding_dimension is not None:
            query["embedding_dimension"] = _safe_int(embedding_dimension)
        if embedding_instruction is not None:
            query["embedding_instruction"] = str(embedding_instruction)
        cursor = (
            self.db.threat_clusters.find(query)
            .sort([("last_seen_ts", DESCENDING), ("_id", ASCENDING)])
            .limit(max(1, int(limit)))
        )
        return [_public_doc(doc) for doc in cursor]  # type: ignore[misc]

    def save_threat_cluster(
        self,
        cluster_doc: Dict[str, Any],
        *,
        expected_membership_revision: Optional[int] = None,
    ) -> bool:
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
        doc = dict(cluster_doc or {})
        if any(doc.get(field) is None for field in required):
            raise ValueError("cluster document is incomplete")
        doc["id"] = str(doc["id"])
        doc["_id"] = doc["id"]
        doc.setdefault("updated_at", _now_ts())
        if expected_membership_revision is None:
            try:
                self.db.threat_clusters.insert_one(doc)
                return True
            except DuplicateKeyError:
                return False
        result = self.db.threat_clusters.replace_one(
            {
                "_id": doc["_id"],
                "membership_revision": int(expected_membership_revision),
            },
            doc,
        )
        return result.matched_count > 0

    def get_cluster_membership(
        self, profile_id: str, article_id: str
    ) -> Optional[Dict[str, Any]]:
        return _public_doc(
            self.db.threat_cluster_memberships.find_one(
                {"profile_id": str(profile_id), "article_id": str(article_id)}
            )
        )

    def save_cluster_membership(self, membership_doc: Dict[str, Any]) -> bool:
        doc = dict(membership_doc or {})
        required = ("profile_id", "article_id", "cluster_id", "assigned_at")
        if any(doc.get(field) is None for field in required):
            raise ValueError("membership document is incomplete")
        doc["_id"] = f"{doc['profile_id']}:{doc['article_id']}"
        try:
            self.db.threat_cluster_memberships.insert_one(doc)
            return True
        except DuplicateKeyError:
            return False

    def _apply_cluster_assignment(self, operation: Dict[str, Any]) -> bool:
        cluster = dict(operation["cluster"])
        membership = dict(operation["membership"])
        expected = operation.get("expected_membership_revision")
        cluster["id"] = str(cluster["id"])
        cluster["_id"] = cluster["id"]
        membership["_id"] = f"{membership['profile_id']}:{membership['article_id']}"

        existing_membership = self.db.threat_cluster_memberships.find_one(
            {"_id": membership["_id"]}
        )
        if existing_membership is not None and str(
            existing_membership.get("cluster_id")
        ) != cluster["id"]:
            return False

        existing_cluster = self.db.threat_clusters.find_one(
            {"_id": cluster["_id"]}, {"membership_revision": 1}
        )
        target_revision = int(cluster["membership_revision"])
        if existing_cluster is None:
            if expected is not None:
                return False
            try:
                self.db.threat_clusters.insert_one(cluster)
            except DuplicateKeyError:
                return False
        else:
            current_revision = int(existing_cluster.get("membership_revision") or 0)
            if current_revision != target_revision:
                if expected is None or current_revision != int(expected):
                    return False
                result = self.db.threat_clusters.replace_one(
                    {"_id": cluster["_id"], "membership_revision": int(expected)},
                    cluster,
                )
                if result.matched_count < 1:
                    return False

        if existing_membership is None:
            try:
                self.db.threat_cluster_memberships.insert_one(membership)
            except DuplicateKeyError:
                return False
        return True

    def save_cluster_assignment(
        self,
        cluster_doc: Dict[str, Any],
        membership_doc: Dict[str, Any],
        *,
        expected_membership_revision: Optional[int] = None,
    ) -> bool:
        """Persist a recoverable cluster/membership pair using a write journal."""

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
        cluster.setdefault("updated_at", _now_ts())
        operation_id = f"{membership['profile_id']}:{membership['article_id']}"
        operation = {
            "_id": operation_id,
            "id": operation_id,
            "cluster": cluster,
            "membership": membership,
            "expected_membership_revision": expected_membership_revision,
            "created_at": _now_ts(),
        }

        for pending in self.db.long_term_assignment_journal.find().sort(
            [("created_at", ASCENDING), ("_id", ASCENDING)]
        ):
            if self._apply_cluster_assignment(pending):
                self.db.long_term_assignment_journal.delete_one({"_id": pending["_id"]})

        if self.db.threat_cluster_memberships.find_one(
            {"_id": operation_id}, {"_id": 1}
        ):
            return False
        try:
            self.db.long_term_assignment_journal.insert_one(operation)
        except DuplicateKeyError:
            return False
        if not self._apply_cluster_assignment(operation):
            return False
        self.db.long_term_assignment_journal.delete_one({"_id": operation_id})
        return True

    def list_cluster_memberships(
        self, cluster_id: str, *, limit: int = 10000
    ) -> List[Dict[str, Any]]:
        cursor = (
            self.db.threat_cluster_memberships.find({"cluster_id": str(cluster_id)})
            .sort([("assigned_at", ASCENDING), ("article_id", ASCENDING)])
            .limit(max(1, int(limit)))
        )
        return [_public_doc(doc) for doc in cursor]  # type: ignore[misc]

    def get_long_term_quarantine(
        self, profile_id: str, article_id: str
    ) -> Optional[Dict[str, Any]]:
        return _public_doc(
            self.db.long_term_article_quarantine.find_one(
                {"profile_id": str(profile_id), "article_id": str(article_id)}
            )
        )

    def save_long_term_quarantine(self, quarantine_doc: Dict[str, Any]) -> bool:
        doc = dict(quarantine_doc or {})
        required = ("profile_id", "article_id", "reason", "observed_at")
        if any(doc.get(field) is None for field in required):
            raise ValueError("quarantine document is incomplete")
        observed_at = _safe_int(doc.pop("observed_at"))
        profile_id = str(doc["profile_id"])
        article_id = str(doc["article_id"])
        self.db.long_term_article_quarantine.update_one(
            {"profile_id": profile_id, "article_id": article_id},
            {
                "$setOnInsert": {
                    "_id": f"{profile_id}:{article_id}",
                    "first_seen_at": observed_at,
                    "attempt_count": 0,
                },
                "$set": {
                    **doc,
                    "profile_id": profile_id,
                    "article_id": article_id,
                    "status": "open",
                    "last_seen_at": observed_at,
                    "resolved_at": None,
                },
                "$inc": {"attempt_count": 1},
            },
            upsert=True,
        )
        return True

    def resolve_long_term_quarantine(
        self, profile_id: str, article_id: str, *, resolved_at: int
    ) -> bool:
        result = self.db.long_term_article_quarantine.update_one(
            {
                "profile_id": str(profile_id),
                "article_id": str(article_id),
                "status": "open",
            },
            {"$set": {"status": "resolved", "resolved_at": _safe_int(resolved_at)}},
        )
        return result.matched_count > 0

    def list_long_term_quarantine(
        self,
        profile_id: str,
        *,
        status: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {"profile_id": str(profile_id)}
        if status is not None:
            query["status"] = str(status)
        cursor = (
            self.db.long_term_article_quarantine.find(query)
            .sort([("last_seen_at", DESCENDING), ("article_id", ASCENDING)])
            .limit(max(1, int(limit)))
        )
        return [_public_doc(doc) for doc in cursor]  # type: ignore[misc]

    def save_cluster_snapshot(self, snapshot_doc: Dict[str, Any]) -> bool:
        return self._insert_long_term_document(
            self.db.threat_cluster_snapshots,
            snapshot_doc,
            (
                "id",
                "profile_id",
                "cluster_id",
                "membership_revision",
                "prompt_version",
                "created_at",
            ),
        )

    def _apply_snapshot_revision(self, operation: Dict[str, Any]) -> bool:
        cluster = dict(operation["cluster"])
        snapshot = dict(operation["snapshot"])
        expected_membership = int(operation["expected_membership_revision"])
        expected_summarized = int(operation["expected_summarized_revision"])
        cluster["_id"] = str(cluster["id"])
        snapshot["_id"] = str(snapshot["id"])
        current = self.db.threat_clusters.find_one(
            {"_id": cluster["_id"]},
            {"membership_revision": 1, "summarized_revision": 1},
        )
        if current is None or int(current.get("membership_revision") or 0) != expected_membership:
            return False
        current_summarized = int(current.get("summarized_revision") or 0)
        target_summarized = int(cluster["summarized_revision"])
        if current_summarized not in {expected_summarized, target_summarized}:
            return False
        existing_snapshot = self.db.threat_cluster_snapshots.find_one(
            {
                "$or": [
                    {"_id": snapshot["_id"]},
                    {
                        "cluster_id": str(snapshot["cluster_id"]),
                        "membership_revision": int(snapshot["membership_revision"]),
                        "prompt_version": str(snapshot["prompt_version"]),
                    },
                ]
            }
        )
        if existing_snapshot is None:
            try:
                self.db.threat_cluster_snapshots.insert_one(snapshot)
            except DuplicateKeyError:
                return False
        elif str(existing_snapshot.get("_id")) != snapshot["_id"]:
            return False
        if current_summarized != target_summarized:
            cluster_filter: Dict[str, Any] = {
                "_id": cluster["_id"],
                "membership_revision": expected_membership,
            }
            if expected_summarized:
                cluster_filter["summarized_revision"] = expected_summarized
            else:
                cluster_filter["$or"] = [
                    {"summarized_revision": 0},
                    {"summarized_revision": {"$exists": False}},
                ]
            result = self.db.threat_clusters.replace_one(
                cluster_filter,
                cluster,
            )
            if result.matched_count < 1:
                return False
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
            "_id": operation_id,
            "id": operation_id,
            "cluster": cluster,
            "snapshot": snapshot,
            "expected_membership_revision": int(expected_membership_revision),
            "expected_summarized_revision": int(expected_summarized_revision),
            "created_at": _now_ts(),
        }
        for pending in self.db.long_term_snapshot_journal.find().sort(
            [("created_at", ASCENDING), ("_id", ASCENDING)]
        ):
            if self._apply_snapshot_revision(pending):
                self.db.long_term_snapshot_journal.delete_one({"_id": pending["_id"]})
        current = self.db.threat_clusters.find_one(
            {"_id": str(cluster["id"])},
            {"membership_revision": 1, "summarized_revision": 1},
        )
        if current is None or (
            int(current.get("membership_revision") or 0)
            != int(expected_membership_revision)
            or int(current.get("summarized_revision") or 0)
            != int(expected_summarized_revision)
        ):
            return False
        try:
            self.db.long_term_snapshot_journal.insert_one(operation)
        except DuplicateKeyError:
            return False
        if not self._apply_snapshot_revision(operation):
            return False
        self.db.long_term_snapshot_journal.delete_one({"_id": operation_id})
        return True

    def list_cluster_snapshots(
        self,
        profile_id: str,
        *,
        cluster_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {"profile_id": str(profile_id)}
        if cluster_id is not None:
            query["cluster_id"] = str(cluster_id)
        cursor = (
            self.db.threat_cluster_snapshots.find(query)
            .sort([("created_at", DESCENDING), ("_id", ASCENDING)])
            .limit(max(1, int(limit)))
        )
        return [_public_doc(doc) for doc in cursor]  # type: ignore[misc]

    def save_threat_landscape_report(self, report_doc: Dict[str, Any]) -> bool:
        return self._insert_long_term_document(
            self.db.threat_landscape_reports,
            report_doc,
            ("id", "profile_id", "period_end_ts", "created_at"),
        )

    def get_threat_landscape_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        return _public_doc(
            self.db.threat_landscape_reports.find_one({"_id": str(report_id)})
        )

    def list_threat_landscape_reports(
        self, profile_id: str, *, limit: int = 100
    ) -> List[Dict[str, Any]]:
        cursor = (
            self.db.threat_landscape_reports.find({"profile_id": str(profile_id)})
            .sort([("period_end_ts", DESCENDING), ("_id", ASCENDING)])
            .limit(max(1, int(limit)))
        )
        return [_public_doc(doc) for doc in cursor]  # type: ignore[misc]

    def create_long_term_run(self, run_doc: Dict[str, Any]) -> bool:
        return self._insert_long_term_document(
            self.db.long_term_runs,
            run_doc,
            ("id", "profile_id", "run_type", "started_at", "status"),
        )

    def get_long_term_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        return _public_doc(self.db.long_term_runs.find_one({"_id": str(run_id)}))

    def update_long_term_run(
        self,
        run_id: str,
        *,
        expected_status: str,
        fields: Dict[str, Any],
    ) -> bool:
        updates = dict(fields or {})
        updates.pop("id", None)
        updates.pop("_id", None)
        if not updates:
            return False
        result = self.db.long_term_runs.update_one(
            {"_id": str(run_id), "status": str(expected_status)},
            {"$set": updates},
        )
        return result.matched_count > 0

    @staticmethod
    def _insert_long_term_document(collection, document, required) -> bool:
        doc = dict(document or {})
        if any(doc.get(field) is None for field in required):
            raise ValueError("document is incomplete")
        doc["id"] = str(doc["id"])
        doc["_id"] = doc["id"]
        try:
            collection.insert_one(doc)
            return True
        except DuplicateKeyError:
            return False

    # Summary documents

    def save_summary_doc(self, summary_doc: Dict[str, Any]) -> Any:
        if not isinstance(summary_doc, dict):
            raise ValueError("summary_doc must be a dict")
        doc = dict(summary_doc)
        created = _safe_int(doc.get("created")) or _now_ts()
        summary_id = str(doc.get("id") or f"summary_{created}")
        doc.update({"_id": summary_id, "id": summary_id, "created": created})
        doc.setdefault("kind", "summary")
        self.db.summary_docs.replace_one({"_id": summary_id}, doc, upsert=True)
        return summary_id

    def get_summary_doc(self, summary_doc_id: str) -> Optional[Dict[str, Any]]:
        return _public_doc(self.db.summary_docs.find_one({"_id": str(summary_doc_id)}))

    def list_summary_docs(self) -> List[Dict[str, Any]]:
        return [_public_doc(doc) for doc in self.db.summary_docs.find().sort("created", DESCENDING)]  # type: ignore[misc]

    def get_latest_summary_doc(self) -> Optional[Dict[str, Any]]:
        return _public_doc(self.db.summary_docs.find_one(sort=[("created", DESCENDING)]))

    # Jobs and resumable temporary summaries

    def create_job(self) -> int:
        job_id = self._next_id("jobs")
        self.db.jobs.insert_one(
            {
                "_id": job_id,
                "id": job_id,
                "created_at": _now_ts(),
                "started_at": None,
                "finished_at": None,
                "status": "queued",
                "message": "",
                "summary_id": None,
            }
        )
        logger.info("Job %s created", job_id)
        return job_id

    def update_job(self, job_id: int, **fields: Any) -> None:
        job_id = _safe_int(job_id)
        if job_id <= 0:
            raise ValueError("job_id must be a positive int")
        updates = dict(fields)
        updates.pop("_id", None)
        updates.pop("id", None)
        if "summary_id" in updates:
            updates["summary_id"] = _normalize_summary_id(updates.get("summary_id"))
        if updates:
            self.db.jobs.update_one({"_id": job_id}, {"$set": updates})
        logger.info("Job %s updated: %s", job_id, updates)

    def get_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        return _public_doc(self.db.jobs.find_one({"_id": _safe_int(job_id)}))

    def list_jobs(self, limit: int = 200) -> List[Dict[str, Any]]:
        cursor = self.db.jobs.find().sort("created_at", DESCENDING).limit(max(0, int(limit)))
        return [_public_doc(doc) for doc in cursor]  # type: ignore[misc]

    def put_temp_summary(self, job_id: int, payload: Dict[str, Any]) -> None:
        job_id = _safe_int(job_id)
        doc = dict(payload or {})
        doc.update({"_id": job_id, "job_id": job_id})
        doc.setdefault("created_at", _now_ts())
        self.db.temp_summaries.replace_one({"_id": job_id}, doc, upsert=True)

    def save_temp_summary(self, job_id: int, summary_text: str, meta: Dict[str, Any]) -> None:
        self.put_temp_summary(job_id, {"summary": summary_text or "", "meta": meta or {}})

    def get_temp_summary(self, job_id: int) -> Optional[Dict[str, Any]]:
        return _public_doc(self.db.temp_summaries.find_one({"_id": _safe_int(job_id)}))

    def run_cleanup(self, pol: CleanupPolicy) -> Dict[str, int]:
        now = _now_ts()
        cut_articles = now - pol.articles_days * 86400
        cut_daily = now - pol.daily_summaries_days * 86400
        cut_weekly = now - pol.weekly_summaries_days * 86400
        cut_temp = now - pol.temp_summaries_days * 86400
        cut_jobs = now - pol.jobs_days * 86400

        removed = {
            "articles": self.db.articles.delete_many(
                {"_mongo_sort_ts": {"$lt": cut_articles}}
            ).deleted_count,
            "temp_summaries": self.db.temp_summaries.delete_many(
                {"created_at": {"$lt": cut_temp}}
            ).deleted_count,
            "jobs": self.db.jobs.delete_many(
                {
                    "$and": [
                        {
                            "$or": [
                                {"finished_at": {"$lt": cut_jobs}},
                                {
                                    "finished_at": None,
                                    "created_at": {"$lt": cut_jobs},
                                },
                            ]
                        },
                        {"status": {"$in": ["done", "failed"]}},
                    ]
                }
            ).deleted_count,
            "summary_docs": 0,
        }

        summary_ids = []
        for doc in self.db.summary_docs.find({"created": {"$lt": max(cut_daily, cut_weekly)}}):
            created = _safe_int(doc.get("created"))
            selection = doc.get("selection") if isinstance(doc.get("selection"), dict) else {}
            package = str(selection.get("prompt_package") or "").lower()
            cutoff = cut_daily if "daily" in package and "weekly" not in package else cut_weekly
            if created < cutoff:
                summary_ids.append(doc["_id"])
        if summary_ids:
            removed["summary_docs"] = self.db.summary_docs.delete_many(
                {"_id": {"$in": summary_ids}}
            ).deleted_count
        return {name: int(count) for name, count in removed.items()}

    # Tags

    @staticmethod
    def _tag_doc(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        public = _public_doc(doc)
        if public is not None:
            public.pop("normalized_name", None)
        return public

    def add_tag(
        self,
        name: str,
        category: str = "GENERAL",
        description: Optional[str] = None,
    ) -> Optional[int]:
        if not isinstance(name, str) or not name.strip():
            return None
        normalized = name.strip().lower()
        existing = self.db.tags.find_one({"normalized_name": normalized}, {"_id": 1})
        if existing:
            return int(existing["_id"])

        tag_id = self._next_id("tags")
        now = _now_ts()
        try:
            self.db.tags.insert_one(
                {
                    "_id": tag_id,
                    "id": tag_id,
                    "name": normalized,
                    "normalized_name": normalized,
                    "category": category,
                    "description": description,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            return tag_id
        except DuplicateKeyError:
            existing = self.db.tags.find_one({"normalized_name": normalized}, {"_id": 1})
            return int(existing["_id"]) if existing else None

    def get_tag_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        if not isinstance(name, str) or not name.strip():
            return None
        return self._tag_doc(self.db.tags.find_one({"normalized_name": name.strip().lower()}))

    def get_all_tags(self) -> List[Dict[str, Any]]:
        return [self._tag_doc(doc) for doc in self.db.tags.find().sort("name", ASCENDING)]  # type: ignore[misc]

    def get_tag_relations(self, tag_id: int) -> Dict[str, List[Dict[str, Any]]]:
        tag_id = int(tag_id)
        rows = list(
            self.db.tag_relations.find(
                {
                    "relation_type": PARENT_CHILD_RELATION,
                    "$or": [{"parent_tag_id": tag_id}, {"child_tag_id": tag_id}],
                }
            )
        )
        parent_ids = [row["parent_tag_id"] for row in rows if row["child_tag_id"] == tag_id]
        child_ids = [row["child_tag_id"] for row in rows if row["parent_tag_id"] == tag_id]
        parents = [
            self._tag_doc(doc)
            for doc in self.db.tags.find({"_id": {"$in": parent_ids}}).sort("name", ASCENDING)
        ]
        children = [
            self._tag_doc(doc)
            for doc in self.db.tags.find({"_id": {"$in": child_ids}}).sort("name", ASCENDING)
        ]
        return {"parents": parents, "children": children}  # type: ignore[dict-item]

    def set_tag_relations(
        self,
        tag_id: int,
        *,
        parent_ids: Optional[List[int]] = None,
        child_ids: Optional[List[int]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        tag_id = int(tag_id)
        tags_by_id = {
            int(doc["_id"]): doc
            for doc in self.db.tags.find({}, {"_id": 1, "category": 1})
        }
        existing_edges = {
            (int(doc["parent_tag_id"]), int(doc["child_tag_id"]))
            for doc in self.db.tag_relations.find(
                {"relation_type": PARENT_CHILD_RELATION},
                {"parent_tag_id": 1, "child_tag_id": 1},
            )
        }
        proposed_edges = proposed_parent_child_edges(
            tag_id,
            parent_ids=parent_ids,
            child_ids=child_ids,
            tags_by_id=tags_by_id,
            existing_edges=existing_edges,
        )
        self.db.tag_relations.delete_many(
            {
                "relation_type": PARENT_CHILD_RELATION,
                "$or": [{"parent_tag_id": tag_id}, {"child_tag_id": tag_id}],
            }
        )
        now = _now_ts()
        rows = [
            {
                "_id": f"{PARENT_CHILD_RELATION}:{parent_id}:{child_id}",
                "relation_type": PARENT_CHILD_RELATION,
                "parent_tag_id": parent_id,
                "child_tag_id": child_id,
                "created_at": now,
            }
            for parent_id, child_id in proposed_edges
            if parent_id == tag_id or child_id == tag_id
        ]
        if rows:
            self.db.tag_relations.insert_many(rows)
        return self.get_tag_relations(tag_id)

    def iter_articles_with_tags(
        self,
        *,
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield canonically tagged articles for read-only exports.

        The export starts from ``article_tags`` so an article is included only
        when a tagging pass produced at least one persisted association.  The
        optional category filter applies to the returned labels, not to article
        eligibility; an article tagged only with excluded entity categories is
        therefore returned with an empty ``tags`` list and can act as a negative
        training example.
        """
        pipeline: List[Dict[str, Any]] = [
            {
                "$group": {
                    "_id": "$article_id",
                    "tag_ids": {"$addToSet": "$tag_id"},
                }
            },
            {
                "$lookup": {
                    "from": "articles",
                    "localField": "_id",
                    "foreignField": "_id",
                    "as": "article_docs",
                }
            },
            {"$unwind": "$article_docs"},
            {
                "$lookup": {
                    "from": "tags",
                    "localField": "tag_ids",
                    "foreignField": "_id",
                    "as": "tag_docs",
                }
            },
            {"$sort": {"article_docs._mongo_sort_ts": ASCENDING, "_id": ASCENDING}},
        ]
        if limit is not None and int(limit) > 0:
            pipeline.append({"$limit": int(limit)})

        allowed = {
            str(category).strip().casefold()
            for category in categories or []
            if str(category).strip()
        }
        for row in self.db.article_tags.aggregate(pipeline):
            article = _public_doc(row.get("article_docs"))
            if not isinstance(article, dict):
                continue
            tags = []
            for raw_tag in row.get("tag_docs") or []:
                tag = self._tag_doc(raw_tag)
                if not isinstance(tag, dict):
                    continue
                category = str(tag.get("category") or "GENERAL").casefold()
                if allowed and category not in allowed:
                    continue
                tags.append(tag)
            tags.sort(key=lambda tag: str(tag.get("name") or "").casefold())
            yield {"article": article, "tags": tags}

    def add_article_tags(self, article_id: str, tag_ids: List) -> None:
        if not article_id or not tag_ids:
            return
        article_id = str(article_id).strip()
        if not article_id:
            return
        self.db.article_tags.delete_many({"article_id": article_id})
        now = _now_ts()
        records: Dict[int, Dict[str, Any]] = {}
        for entry in tag_ids:
            if isinstance(entry, dict):
                tag_id = entry.get("tag_id") or entry.get("id")
                reasoning = entry.get("reasoning") or ""
            else:
                try:
                    tag_id = int(entry)
                except (TypeError, ValueError):
                    continue
                reasoning = ""
            if not isinstance(tag_id, int) or tag_id <= 0:
                continue
            record = {
                "_id": f"{article_id}:{tag_id}",
                "article_id": article_id,
                "tag_id": tag_id,
                "created_at": now,
            }
            if reasoning:
                record["motivering"] = str(reasoning).strip()
            records[tag_id] = record
        if records:
            self.db.article_tags.insert_many(list(records.values()))

    def get_article_tags(self, article_id: str) -> List[Dict[str, Any]]:
        article_id = str(article_id or "").strip()
        if not article_id:
            return []
        tag_ids = [row["tag_id"] for row in self.db.article_tags.find({"article_id": article_id})]
        if not tag_ids:
            return []
        return [
            self._tag_doc(doc)
            for doc in self.db.tags.find({"_id": {"$in": tag_ids}}).sort("name", ASCENDING)
        ]  # type: ignore[misc]

    def remove_article_tag(self, article_id: str, tag_id: int) -> bool:
        if not article_id or not tag_id:
            return False
        result = self.db.article_tags.delete_one(
            {"article_id": str(article_id).strip(), "tag_id": int(tag_id)}
        )
        return result.deleted_count > 0

    def add_tag_to_article(self, article_id: str, tag_id: int) -> bool:
        if not article_id or not tag_id:
            return False
        article_id = str(article_id).strip()
        tag_id = int(tag_id)
        try:
            self.db.article_tags.insert_one(
                {
                    "_id": f"{article_id}:{tag_id}",
                    "article_id": article_id,
                    "tag_id": tag_id,
                    "created_at": _now_ts(),
                }
            )
            return True
        except DuplicateKeyError:
            return False

    def create_tag(
        self,
        name: str,
        category: str = "GENERAL",
        description: str = "",
        synonyms: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not name or not str(name).strip():
            return None
        clean_name = str(name).strip()
        normalized = clean_name.lower()
        if self.db.tags.find_one({"normalized_name": normalized}, {"_id": 1}):
            return None
        tag_id = self._next_id("tags")
        doc = {
            "_id": tag_id,
            "id": tag_id,
            "name": clean_name,
            "normalized_name": normalized,
            "category": str(category).strip() or "GENERAL",
            "description": str(description or "").strip(),
            "synonyms": [str(item).strip().lower() for item in synonyms or []],
            "created_at": _now_ts(),
        }
        try:
            self.db.tags.insert_one(doc)
        except DuplicateKeyError:
            return None
        return self._tag_doc(doc)

    def update_tag(
        self,
        tag_id: int,
        name: Optional[str] = None,
        category: Optional[str] = None,
        description: Optional[str] = None,
        synonyms: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not tag_id:
            return None
        updates: Dict[str, Any] = {}
        if name is not None:
            updates.update(
                {
                    "name": name.strip(),
                    "normalized_name": name.strip().lower(),
                    "embedding_vector": None,
                    "embedding_model": None,
                    "embedding_source_hash": None,
                    "embedding_updated_at": None,
                }
            )
        if category is not None:
            updates["category"] = category.strip() or "GENERAL"
        if description is not None:
            updates["description"] = str(description or "").strip()
        if synonyms is not None:
            updates["synonyms"] = [str(item).strip().lower() for item in synonyms]
        if updates:
            existing = self.db.tags.find_one({"_id": int(tag_id)}, {"category": 1})
            if existing is None:
                return None
            new_category = updates.get("category")
            category_changed = new_category is not None and new_category != str(
                existing.get("category") or "GENERAL"
            )
            updates["updated_at"] = _now_ts()
            try:
                result = self.db.tags.update_one({"_id": int(tag_id)}, {"$set": updates})
            except DuplicateKeyError:
                return None
            if result.matched_count == 0:
                return None
            if category_changed:
                self.db.tag_relations.delete_many(
                    {
                        "$or": [
                            {"parent_tag_id": int(tag_id)},
                            {"child_tag_id": int(tag_id)},
                        ]
                    }
                )
        return self._tag_doc(self.db.tags.find_one({"_id": int(tag_id)}))

    def delete_tag(self, tag_id: int) -> bool:
        if not tag_id:
            return False
        tag_id = int(tag_id)
        self.db.article_tags.delete_many({"tag_id": tag_id})
        self.db.tag_relations.delete_many(
            {"$or": [{"parent_tag_id": tag_id}, {"child_tag_id": tag_id}]}
        )
        return self.db.tags.delete_one({"_id": tag_id}).deleted_count > 0

    def migrate_synonym_to_main_tag(
        self, main_tag_id: int, synonym_tag_ids: List[int]
    ) -> Tuple[int, int]:
        if not main_tag_id or not synonym_tag_ids:
            return 0, 0
        main_tag_id = int(main_tag_id)
        migrated = 0
        deleted = 0
        for synonym_id in {int(value) for value in synonym_tag_ids if int(value) != main_tag_id}:
            associations = list(self.db.article_tags.find({"tag_id": synonym_id}))
            for association in associations:
                article_id = association["article_id"]
                self.db.article_tags.update_one(
                    {"article_id": article_id, "tag_id": main_tag_id},
                    {
                        "$setOnInsert": {
                            "_id": f"{article_id}:{main_tag_id}",
                            "article_id": article_id,
                            "tag_id": main_tag_id,
                            "created_at": _now_ts(),
                        }
                    },
                    upsert=True,
                )
                migrated += 1
            self.db.article_tags.delete_many({"tag_id": synonym_id})
            self.db.tag_relations.delete_many(
                {
                    "$or": [
                        {"parent_tag_id": synonym_id},
                        {"child_tag_id": synonym_id},
                    ]
                }
            )
            deleted += self.db.tags.delete_one({"_id": synonym_id}).deleted_count
        return migrated, deleted

    def update_tag_embedding(
        self,
        tag_id: int,
        embedding_vector: List[float],
        *,
        model: Optional[str] = None,
        source_hash: Optional[str] = None,
    ) -> bool:
        if not isinstance(tag_id, int) or tag_id <= 0:
            return False
        if not embedding_vector or not all(
            isinstance(value, (int, float)) for value in embedding_vector
        ):
            return False
        result = self.db.tags.update_one(
            {"_id": tag_id},
            {
                "$set": {
                    "embedding_vector": [float(value) for value in embedding_vector],
                    "embedding_model": str(model or ""),
                    "embedding_source_hash": str(source_hash or ""),
                    "embedding_updated_at": _now_ts(),
                    "updated_at": _now_ts(),
                }
            },
        )
        return result.matched_count > 0

    def get_tags_by_embedding_similarity(
        self,
        embedding_vector: List[float],
        similarity_threshold: float = 0.75,
        limit: int = 10,
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not embedding_vector or limit <= 0:
            return []
        results: List[Tuple[Dict[str, Any], float]] = []
        query: Dict[str, Any] = {"embedding_vector": {"$exists": True}}
        if model is not None:
            query["embedding_model"] = str(model)
        for doc in self.db.tags.find(query):
            tag_embedding = doc.get("embedding_vector")
            if not isinstance(tag_embedding, list):
                continue
            similarity = self._cosine_similarity(embedding_vector, tag_embedding)
            if similarity >= similarity_threshold:
                tag = self._tag_doc(doc)
                if tag is not None:
                    tag["_similarity_score"] = similarity
                    results.append((tag, similarity))
        results.sort(key=lambda item: -item[1])
        return [tag for tag, _ in results[:limit]]

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(value * value for value in vec1))
            magnitude2 = math.sqrt(sum(value * value for value in vec2))
            return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0.0
        except (TypeError, ValueError):
            return 0.0

    def cleanup_unused_tags(self, days: int = 30) -> int:
        cutoff = _now_ts() - int(days) * 86400
        used_ids = self.db.article_tags.distinct("tag_id", {"created_at": {"$gt": cutoff}})
        unused_ids = self.db.tags.distinct(
            "_id", {"created_at": {"$lt": cutoff}, "_id": {"$nin": used_ids}}
        )
        if not unused_ids:
            return 0
        result = self.db.tags.delete_many({"_id": {"$in": unused_ids}})
        self.db.article_tags.delete_many({"tag_id": {"$in": unused_ids}})
        self.db.tag_relations.delete_many(
            {
                "$or": [
                    {"parent_tag_id": {"$in": unused_ids}},
                    {"child_tag_id": {"$in": unused_ids}},
                ]
            }
        )
        return int(result.deleted_count)

    def get_articles_by_tags(
        self, tag_names: List[str], match_mode: str = "any"
    ) -> List[Dict[str, Any]]:
        names = [str(name).strip().lower() for name in tag_names or [] if str(name).strip()]
        if not names:
            return []
        tag_ids = self.db.tags.distinct("_id", {"normalized_name": {"$in": names}})
        if not tag_ids:
            return []
        if match_mode == "all":
            pipeline = [
                {"$match": {"tag_id": {"$in": tag_ids}}},
                {"$group": {"_id": "$article_id", "tag_ids": {"$addToSet": "$tag_id"}}},
                {"$match": {f"tag_ids.{len(tag_ids) - 1}": {"$exists": True}}},
            ]
            article_ids = [row["_id"] for row in self.db.article_tags.aggregate(pipeline)]
        else:
            article_ids = self.db.article_tags.distinct("article_id", {"tag_id": {"$in": tag_ids}})
        return self.get_articles_by_ids(article_ids)

    # Tag categories

    @staticmethod
    def _category_doc(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return _public_doc(doc)

    def get_all_categories(self) -> List[Dict[str, Any]]:
        return [
            self._category_doc(doc) for doc in self.db.tag_categories.find().sort("name", ASCENDING)
        ]  # type: ignore[misc]

    def get_category(self, category_id: int) -> Optional[Dict[str, Any]]:
        return self._category_doc(self.db.tag_categories.find_one({"_id": int(category_id)}))

    def create_category(
        self,
        name: str,
        label: str,
        bg_color: str = "bg-secondary",
        text_color: str = "text-dark",
        description: str = "",
    ) -> Optional[Dict[str, Any]]:
        if not name or not label:
            return None
        category_id = self._next_id("tag_categories")
        doc = {
            "_id": category_id,
            "id": category_id,
            "name": str(name),
            "label": str(label),
            "bg_color": str(bg_color),
            "text_color": str(text_color),
            "description": str(description),
            "created_at": _now_ts(),
        }
        try:
            self.db.tag_categories.insert_one(doc)
        except DuplicateKeyError:
            return None
        return self._category_doc(doc)

    def update_category(
        self,
        category_id: int,
        label: Optional[str] = None,
        bg_color: Optional[str] = None,
        text_color: Optional[str] = None,
        description: Optional[str] = None,
    ) -> bool:
        updates = {
            key: value
            for key, value in {
                "label": label,
                "bg_color": bg_color,
                "text_color": text_color,
                "description": description,
            }.items()
            if value is not None
        }
        if not updates:
            return False
        return (
            self.db.tag_categories.update_one(
                {"_id": int(category_id)}, {"$set": updates}
            ).matched_count
            > 0
        )

    def delete_category(self, category_id: int) -> bool:
        return self.db.tag_categories.delete_one({"_id": int(category_id)}).deleted_count > 0

    def initialize_default_categories(self) -> None:
        now = _now_ts()
        for name, label, bg_color, text_color in _DEFAULT_CATEGORIES:
            existing = self.db.tag_categories.find_one({"name": name}, {"_id": 1})
            if existing:
                continue
            category_id = self._next_id("tag_categories")
            try:
                self.db.tag_categories.insert_one(
                    {
                        "_id": category_id,
                        "id": category_id,
                        "name": name,
                        "label": label,
                        "bg_color": bg_color,
                        "text_color": text_color,
                        "description": "",
                        "created_at": now,
                    }
                )
            except DuplicateKeyError:
                pass

        for tag in self.db.tags.find({}, {"_id": 1, "name": 1, "category": 1}):
            if is_cve_tag(tag.get("name")) and tag.get("category") != VULNERABILITY_TAG_CATEGORY:
                self.db.tag_relations.delete_many(
                    {
                        "$or": [
                            {"parent_tag_id": tag["_id"]},
                            {"child_tag_id": tag["_id"]},
                        ]
                    }
                )
                self.db.tags.update_one(
                    {"_id": tag["_id"]},
                    {"$set": {"category": VULNERABILITY_TAG_CATEGORY, "updated_at": now}},
                )
