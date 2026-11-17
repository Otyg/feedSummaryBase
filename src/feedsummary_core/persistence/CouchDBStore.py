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

from collections.abc import Iterator
from typing import Any, Dict, List, Optional

from feedsummary_core.persistence.CouchDBCollection import (
    CouchDBClient,
    CouchDBDatabase,
    UniqueSpec,
)
from feedsummary_core.persistence.MongoDBStore import MongoDBStore, _public_doc

# (collection, index fields); mirrors the MongoDB store's indexes.
_INDEXES = (
    ("articles", ("source", "_mongo_sort_ts")),
    ("articles", ("_mongo_sort_ts",)),
    ("articles", ("fetched_at",)),
    ("articles", ("url",)),
    ("summary_docs", ("created",)),
    ("jobs", ("created_at",)),
    ("temp_summaries", ("created_at",)),
    ("tags", ("normalized_name",)),
    ("tags", ("name",)),
    ("article_tags", ("article_id",)),
    ("article_tags", ("tag_id",)),
    ("tag_relations", ("parent_tag_id",)),
    ("tag_relations", ("child_tag_id",)),
    ("tag_categories", ("name",)),
)

_COLLECTIONS = (
    "articles",
    "article_tags",
    "counters",
    "jobs",
    "summary_docs",
    "tag_categories",
    "tag_relations",
    "tags",
    "temp_summaries",
)

_UNIQUES = {
    "articles": (
        UniqueSpec("url", ("url",), partial=lambda doc: isinstance(doc.get("url"), str) and doc["url"] > ""),
    ),
    "tags": (UniqueSpec("normalized_name", ("normalized_name",)),),
    "tag_categories": (UniqueSpec("name", ("name",)),),
}


class CouchDBStore(MongoDBStore):
    """CouchDB-backed persistence store with MongoDB-store feature parity.

    Each logical collection lives in its own CouchDB database named
    ``<database>_<collection>`` (for example ``feedsummary_articles``), which is the
    layout produced by the Mongo-to-CouchDB migration.  The business logic is
    inherited from :class:`MongoDBStore`; see ``CouchDBCollection`` for how the
    required Mongo operations map onto CouchDB.

    Atomic single-document updates use ``_rev`` compare-and-swap retries.
    """

    def __init__(
        self,
        url: str = "http://localhost:5984",
        database: str = "feedsummary",
        *,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client: Any = None,
        timeout: float = 30.0,
        verify_tls: bool = True,
        initialize_schema: bool = True,
    ):
        if not database or not str(database).strip():
            raise ValueError("database must be a non-empty string")

        self.uri = str(url)
        self.database_name = str(database).strip()
        self._owns_client = client is None
        if client is None:
            client = CouchDBClient(
                self.uri,
                username=username,
                password=password,
                timeout=timeout,
                verify_tls=verify_tls,
            )
        self.client = client
        self.db = CouchDBDatabase(
            client,
            self.database_name,
            int_id_collections=("jobs", "tags", "tag_categories", "temp_summaries"),
            uniques=_UNIQUES,
            create_missing=initialize_schema,
        )
        if initialize_schema:
            self._init_db()

    def _init_db(self) -> None:
        for name in _COLLECTIONS:
            self.db[name].ensure()
        for name, fields in _INDEXES:
            self.db[name].create_index(fields)
        for name in ("jobs", "tags", "tag_categories"):
            self._seed_counter(name, self.db[name])

    def _seed_counter(self, name: str, collection: Any) -> None:
        highest = collection.max_numeric_id()
        current = self.db.counters.find_one({"_id": name})
        if current is None or int(current.get("seq") or 0) < highest:
            self.db.counters.update_one({"_id": name}, {"$max": {"seq": highest}}, upsert=True)

    def iter_articles_with_tags(
        self,
        *,
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield canonically tagged articles for read-only exports (oldest first)."""

        tag_ids_by_article: Dict[str, set] = {}
        for row in self.db.article_tags.find({}, {"article_id": 1, "tag_id": 1}):
            tag_ids_by_article.setdefault(str(row["article_id"]), set()).add(row["tag_id"])

        sort_keys = {
            str(row["_id"]): int(row.get("_mongo_sort_ts") or 0)
            for row in self.db.articles.find({}, {"_mongo_sort_ts": 1})
            if str(row["_id"]) in tag_ids_by_article
        }
        ordered = sorted(sort_keys, key=lambda article_id: (sort_keys[article_id], article_id))
        if limit is not None and int(limit) > 0:
            ordered = ordered[: int(limit)]

        allowed = {
            str(category).strip().casefold()
            for category in categories or []
            if str(category).strip()
        }
        chunk_size = 100
        for start in range(0, len(ordered), chunk_size):
            chunk = ordered[start : start + chunk_size]
            articles = {
                str(doc["_id"]): doc for doc in self.db.articles.find({"_id": {"$in": chunk}})
            }
            wanted_tags = sorted({tag for article_id in chunk for tag in tag_ids_by_article[article_id]})
            tag_docs = {
                doc["_id"]: doc for doc in self.db.tags.find({"_id": {"$in": wanted_tags}})
            }
            for article_id in chunk:
                article = _public_doc(articles.get(article_id))
                if not isinstance(article, dict):
                    continue
                tags = []
                for tag_id in tag_ids_by_article[article_id]:
                    tag = self._tag_doc(tag_docs.get(tag_id))
                    if not isinstance(tag, dict):
                        continue
                    category = str(tag.get("category") or "GENERAL").casefold()
                    if allowed and category not in allowed:
                        continue
                    tags.append(tag)
                tags.sort(key=lambda tag: str(tag.get("name") or "").casefold())
                yield {"article": article, "tags": tags}

    def get_articles_by_tags(
        self, tag_names: List[str], match_mode: str = "any"
    ) -> List[Dict[str, Any]]:
        names = [str(name).strip().lower() for name in tag_names or [] if str(name).strip()]
        if not names:
            return []
        tag_ids = self.db.tags.distinct("_id", {"normalized_name": {"$in": names}})
        if not tag_ids:
            return []
        rows = self.db.article_tags.find(
            {"tag_id": {"$in": tag_ids}}, {"article_id": 1, "tag_id": 1}
        )
        matched: Dict[str, set] = {}
        for row in rows:
            matched.setdefault(str(row["article_id"]), set()).add(row["tag_id"])
        if match_mode == "all":
            article_ids = [aid for aid, found in matched.items() if len(found) >= len(tag_ids)]
        else:
            article_ids = list(matched)
        return self.get_articles_by_ids(article_ids)
