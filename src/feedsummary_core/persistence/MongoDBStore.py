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
from typing import Any, Dict, List, Optional, Tuple

from feedsummary_core.persistence.CleanUpPolicy import CleanupPolicy
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
            (self.db.tag_categories, [("name", ASCENDING)], {"unique": True}),
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
                "embedding_vector": 1,
                "embedding_model": 1,
                "embedding_source_hash": 1,
                "embedding_updated_at": 1,
            },
        )
        if existing:
            for field in (
                "embedding_vector",
                "embedding_model",
                "embedding_source_hash",
                "embedding_updated_at",
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
    ) -> bool:
        """Persist a reusable embedding for an existing article."""
        if (
            not article_id
            or not embedding_vector
            or not all(isinstance(value, (int, float)) for value in embedding_vector)
        ):
            return False
        result = self.db.articles.update_one(
            {"_id": str(article_id)},
            {
                "$set": {
                    "embedding_vector": [float(value) for value in embedding_vector],
                    "embedding_model": str(model or ""),
                    "embedding_source_hash": str(source_hash or ""),
                    "embedding_updated_at": _now_ts(),
                }
            },
        )
        return result.matched_count > 0

    def list_articles(self, limit: int = 2000) -> List[Dict[str, Any]]:
        cursor = self.db.articles.find().sort("_mongo_sort_ts", ASCENDING).limit(max(0, int(limit)))
        return [_public_doc(doc) for doc in cursor]  # type: ignore[misc]

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
            updates["updated_at"] = _now_ts()
            try:
                result = self.db.tags.update_one({"_id": int(tag_id)}, {"$set": updates})
            except DuplicateKeyError:
                return None
            if result.matched_count == 0:
                return None
        return self._tag_doc(self.db.tags.find_one({"_id": int(tag_id)}))

    def delete_tag(self, tag_id: int) -> bool:
        if not tag_id:
            return False
        tag_id = int(tag_id)
        self.db.article_tags.delete_many({"tag_id": tag_id})
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
                self.db.tags.update_one(
                    {"_id": tag["_id"]},
                    {"$set": {"category": VULNERABILITY_TAG_CATEGORY, "updated_at": now}},
                )
