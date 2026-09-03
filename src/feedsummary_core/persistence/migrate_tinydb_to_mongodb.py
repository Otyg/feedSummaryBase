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

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tinydb import TinyDB

from feedsummary_core.persistence.MongoDBStore import MongoDBStore

try:
    from pymongo import ReplaceOne
except ImportError:  # pragma: no cover - dry-run works without the MongoDB extra
    ReplaceOne = None  # type: ignore[assignment]


MIGRATED_TABLES = (
    "articles",
    "summary_docs",
    "jobs",
    "temp_summaries",
    "tags",
    "article_tags",
    "tag_relations",
    "tag_categories",
)


class MigrationConflict(RuntimeError):
    """Raised when two records cannot be represented by MongoDB's unique indexes."""


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _sort_ts(doc: Dict[str, Any]) -> int:
    published = _safe_int(doc.get("published_ts"))
    return published if published > 0 else _safe_int(doc.get("fetched_at"))


def _chunks(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(items), max(1, size)):
        yield items[start : start + max(1, size)]


def _source_rows(db: TinyDB, table_name: str) -> List[tuple[int, Dict[str, Any]]]:
    rows = []
    for document in db.table(table_name).all():
        doc_id = _safe_int(getattr(document, "doc_id", document.get("id")))
        rows.append((doc_id, dict(document)))
    return rows


def _new_report(source: Path, database: str, dry_run: bool) -> Dict[str, Any]:
    return {
        "source": str(source),
        "database": database,
        "dry_run": dry_run,
        "collections": {
            name: {"source": 0, "prepared": 0, "written": 0, "skipped": 0, "verified": 0}
            for name in MIGRATED_TABLES
        },
        "id_remaps": {"articles": 0, "tags": 0, "tag_categories": 0},
        "warnings": [],
    }


def _register_unique(
    seen: Dict[str, Any],
    key: str,
    value: Any,
    *,
    kind: str,
    conflict_policy: str,
) -> Optional[Any]:
    existing = seen.get(key)
    if existing is None:
        seen[key] = value
        return None
    if conflict_policy == "error":
        raise MigrationConflict(f"Duplicate {kind} in TinyDB: {key!r}")
    return existing


def _prepare_source(
    source: Path,
    database: str,
    *,
    dry_run: bool,
    conflict_policy: str,
) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any], Dict[int, int], Dict[str, str]]:
    report = _new_report(source, database, dry_run)
    prepared: Dict[str, List[Dict[str, Any]]] = {name: [] for name in MIGRATED_TABLES}
    tag_id_map: Dict[int, int] = {}
    article_id_map: Dict[str, str] = {}

    db = TinyDB(str(source))
    try:
        known_tables = set(MIGRATED_TABLES) | {"_default"}
        ignored = sorted(db.tables() - known_tables)
        if ignored:
            report["warnings"].append(f"Ignored unknown TinyDB tables: {', '.join(ignored)}")

        seen_article_ids: Dict[str, str] = {}
        seen_urls: Dict[str, str] = {}
        article_rows = _source_rows(db, "articles")
        report["collections"]["articles"]["source"] = len(article_rows)
        for _, source_doc in article_rows:
            doc = dict(source_doc)
            article_id = str(doc.get("id") or "").strip()
            if not article_id:
                report["collections"]["articles"]["skipped"] += 1
                report["warnings"].append("Skipped article without id")
                continue
            duplicate_id = _register_unique(
                seen_article_ids,
                article_id,
                article_id,
                kind="article id",
                conflict_policy=conflict_policy,
            )
            if duplicate_id is not None:
                article_id_map[article_id] = duplicate_id
                report["collections"]["articles"]["skipped"] += 1
                continue
            url = str(doc.get("url") or "").strip()
            if url:
                duplicate_url_id = _register_unique(
                    seen_urls,
                    url,
                    article_id,
                    kind="article URL",
                    conflict_policy=conflict_policy,
                )
                if duplicate_url_id is not None:
                    article_id_map[article_id] = duplicate_url_id
                    report["id_remaps"]["articles"] += 1
                    report["collections"]["articles"]["skipped"] += 1
                    continue
            article_id_map[article_id] = article_id
            doc.pop("_id", None)
            doc.update({"_id": article_id, "id": article_id, "_mongo_sort_ts": _sort_ts(doc)})
            prepared["articles"].append(doc)

        summary_rows = _source_rows(db, "summary_docs")
        report["collections"]["summary_docs"]["source"] = len(summary_rows)
        seen_summary_ids: Dict[str, str] = {}
        for doc_id, source_doc in summary_rows:
            doc = dict(source_doc)
            summary_id = str(doc.get("id") or f"summary_doc_{doc_id}")
            duplicate = _register_unique(
                seen_summary_ids,
                summary_id,
                summary_id,
                kind="summary id",
                conflict_policy=conflict_policy,
            )
            if duplicate is not None:
                report["collections"]["summary_docs"]["skipped"] += 1
                continue
            doc.pop("_id", None)
            doc.update({"_id": summary_id, "id": summary_id})
            prepared["summary_docs"].append(doc)

        job_rows = _source_rows(db, "jobs")
        report["collections"]["jobs"]["source"] = len(job_rows)
        for doc_id, source_doc in job_rows:
            if doc_id <= 0:
                report["collections"]["jobs"]["skipped"] += 1
                report["warnings"].append("Skipped job without a positive TinyDB doc_id")
                continue
            doc = dict(source_doc)
            doc.pop("_id", None)
            doc.update({"_id": doc_id, "id": doc_id})
            prepared["jobs"].append(doc)

        temp_rows = _source_rows(db, "temp_summaries")
        report["collections"]["temp_summaries"]["source"] = len(temp_rows)
        seen_temp_ids: set[int] = set()
        for _, source_doc in temp_rows:
            doc = dict(source_doc)
            job_id = _safe_int(doc.get("job_id"))
            if job_id <= 0 or job_id in seen_temp_ids:
                if job_id in seen_temp_ids and conflict_policy == "error":
                    raise MigrationConflict(f"Duplicate temp summary job_id in TinyDB: {job_id}")
                report["collections"]["temp_summaries"]["skipped"] += 1
                continue
            seen_temp_ids.add(job_id)
            doc.pop("_id", None)
            doc.update({"_id": job_id, "job_id": job_id})
            prepared["temp_summaries"].append(doc)

        tag_rows = _source_rows(db, "tags")
        report["collections"]["tags"]["source"] = len(tag_rows)
        seen_tag_names: Dict[str, int] = {}
        tags_by_id: Dict[int, Dict[str, Any]] = {}

        def map_tag_alias(alias_id: int, canonical_id: int) -> None:
            if alias_id <= 0:
                return
            existing_id = tag_id_map.get(alias_id)
            if existing_id is not None and existing_id != canonical_id:
                if conflict_policy == "error":
                    raise MigrationConflict(
                        f"TinyDB tag id {alias_id} maps to both {existing_id} and {canonical_id}"
                    )
                report["warnings"].append(
                    f"Kept first mapping for ambiguous TinyDB tag id {alias_id}"
                )
                return
            tag_id_map[alias_id] = canonical_id

        for doc_id, source_doc in tag_rows:
            name = str(source_doc.get("name") or "").strip()
            if doc_id <= 0 or not name:
                report["collections"]["tags"]["skipped"] += 1
                report["warnings"].append("Skipped tag without name or positive TinyDB doc_id")
                continue
            normalized = name.lower()
            canonical_id = _register_unique(
                seen_tag_names,
                normalized,
                doc_id,
                kind="normalized tag name",
                conflict_policy=conflict_policy,
            )
            if canonical_id is not None:
                map_tag_alias(doc_id, int(canonical_id))
                map_tag_alias(_safe_int(source_doc.get("id")), int(canonical_id))
                report["id_remaps"]["tags"] += 1
                report["collections"]["tags"]["skipped"] += 1
                canonical = tags_by_id[int(canonical_id)]
                synonyms = {
                    str(item).strip().lower()
                    for item in canonical.get("synonyms", [])
                    if str(item).strip()
                }
                synonyms.update(
                    str(item).strip().lower()
                    for item in source_doc.get("synonyms", [])
                    if str(item).strip()
                )
                if normalized != str(canonical.get("name") or "").lower():
                    synonyms.add(normalized)
                canonical["synonyms"] = sorted(synonyms)
                continue
            map_tag_alias(doc_id, doc_id)
            map_tag_alias(_safe_int(source_doc.get("id")), doc_id)
            doc = dict(source_doc)
            doc.pop("_id", None)
            doc.update({"_id": doc_id, "id": doc_id, "name": name, "normalized_name": normalized})
            tags_by_id[doc_id] = doc
            prepared["tags"].append(doc)

        category_rows = _source_rows(db, "tag_categories")
        report["collections"]["tag_categories"]["source"] = len(category_rows)
        seen_category_names: Dict[str, int] = {}
        for doc_id, source_doc in category_rows:
            name = str(source_doc.get("name") or "").strip()
            if doc_id <= 0 or not name:
                report["collections"]["tag_categories"]["skipped"] += 1
                continue
            canonical_id = _register_unique(
                seen_category_names,
                name,
                doc_id,
                kind="category name",
                conflict_policy=conflict_policy,
            )
            if canonical_id is not None:
                report["id_remaps"]["tag_categories"] += 1
                report["collections"]["tag_categories"]["skipped"] += 1
                continue
            doc = dict(source_doc)
            doc.pop("_id", None)
            doc.update({"_id": doc_id, "id": doc_id, "name": name})
            prepared["tag_categories"].append(doc)

        relation_rows = _source_rows(db, "article_tags")
        report["collections"]["article_tags"]["source"] = len(relation_rows)
        relations: Dict[str, Dict[str, Any]] = {}
        for _, source_doc in relation_rows:
            source_article_id = str(source_doc.get("article_id") or "").strip()
            source_tag_id = _safe_int(source_doc.get("tag_id"))
            article_id = article_id_map.get(source_article_id, source_article_id)
            tag_id = tag_id_map.get(source_tag_id)
            if not article_id or tag_id is None:
                report["collections"]["article_tags"]["skipped"] += 1
                report["warnings"].append(
                    f"Skipped dangling article-tag relation: {source_article_id!r}/{source_tag_id}"
                )
                continue
            relation_id = f"{article_id}:{tag_id}"
            if relation_id in relations:
                report["collections"]["article_tags"]["skipped"] += 1
                continue
            doc = dict(source_doc)
            doc.pop("_id", None)
            doc.update({"_id": relation_id, "article_id": article_id, "tag_id": int(tag_id)})
            relations[relation_id] = doc
        prepared["article_tags"] = list(relations.values())

        tag_relation_rows = _source_rows(db, "tag_relations")
        report["collections"]["tag_relations"]["source"] = len(tag_relation_rows)
        tag_relations: Dict[str, Dict[str, Any]] = {}
        for _, source_doc in tag_relation_rows:
            parent_id = tag_id_map.get(_safe_int(source_doc.get("parent_tag_id")))
            child_id = tag_id_map.get(_safe_int(source_doc.get("child_tag_id")))
            relation_type = str(source_doc.get("relation_type") or "parent_child")
            if parent_id is None or child_id is None or parent_id == child_id:
                report["collections"]["tag_relations"]["skipped"] += 1
                report["warnings"].append("Skipped dangling or self-referential tag relation")
                continue
            parent_category = str(tags_by_id[int(parent_id)].get("category") or "GENERAL")
            child_category = str(tags_by_id[int(child_id)].get("category") or "GENERAL")
            if parent_category != child_category:
                report["collections"]["tag_relations"]["skipped"] += 1
                report["warnings"].append("Skipped cross-category tag relation")
                continue
            relation_id = f"{relation_type}:{parent_id}:{child_id}"
            if relation_id in tag_relations:
                report["collections"]["tag_relations"]["skipped"] += 1
                continue
            doc = dict(source_doc)
            doc.pop("_id", None)
            doc.update(
                {
                    "_id": relation_id,
                    "relation_type": relation_type,
                    "parent_tag_id": int(parent_id),
                    "child_tag_id": int(child_id),
                }
            )
            tag_relations[relation_id] = doc
        prepared["tag_relations"] = list(tag_relations.values())
    finally:
        db.close()

    for name, documents in prepared.items():
        report["collections"][name]["prepared"] = len(documents)
    return prepared, report, tag_id_map, article_id_map


def _resolve_target_conflicts(
    prepared: Dict[str, List[Dict[str, Any]]],
    store: MongoDBStore,
    report: Dict[str, Any],
    *,
    conflict_policy: str,
) -> None:
    tag_remaps: Dict[int, int] = {}
    retained_tags = []
    for doc in prepared["tags"]:
        existing = store.db.tags.find_one({"normalized_name": doc["normalized_name"]}, {"_id": 1})
        if existing and existing["_id"] != doc["_id"]:
            if conflict_policy == "error":
                raise MigrationConflict(
                    f"MongoDB already has tag {doc['normalized_name']!r} with id {existing['_id']}"
                )
            try:
                tag_remaps[int(doc["_id"])] = int(existing["_id"])
            except (TypeError, ValueError) as exc:
                raise MigrationConflict("Existing MongoDB tag id is not an integer") from exc
            report["id_remaps"]["tags"] += 1
            report["collections"]["tags"]["skipped"] += 1
            continue
        retained_tags.append(doc)
    prepared["tags"] = retained_tags

    article_remaps: Dict[str, str] = {}
    retained_articles = []
    for doc in prepared["articles"]:
        url = str(doc.get("url") or "").strip()
        existing = store.db.articles.find_one({"url": url}, {"_id": 1}) if url else None
        if existing and str(existing["_id"]) != doc["_id"]:
            if conflict_policy == "error":
                raise MigrationConflict(
                    f"MongoDB already has article URL {url!r} with id {existing['_id']}"
                )
            article_remaps[str(doc["_id"])] = str(existing["_id"])
            report["id_remaps"]["articles"] += 1
            report["collections"]["articles"]["skipped"] += 1
            continue
        retained_articles.append(doc)
    prepared["articles"] = retained_articles

    retained_categories = []
    for doc in prepared["tag_categories"]:
        existing = store.db.tag_categories.find_one({"name": doc["name"]}, {"_id": 1})
        if existing and existing["_id"] != doc["_id"]:
            if conflict_policy == "error":
                raise MigrationConflict(
                    f"MongoDB already has category {doc['name']!r} with id {existing['_id']}"
                )
            report["id_remaps"]["tag_categories"] += 1
            report["collections"]["tag_categories"]["skipped"] += 1
            continue
        retained_categories.append(doc)
    prepared["tag_categories"] = retained_categories

    if tag_remaps or article_remaps:
        remapped_relations: Dict[str, Dict[str, Any]] = {}
        for relation in prepared["article_tags"]:
            article_id = article_remaps.get(
                str(relation["article_id"]), str(relation["article_id"])
            )
            tag_id = tag_remaps.get(int(relation["tag_id"]), int(relation["tag_id"]))
            relation_id = f"{article_id}:{tag_id}"
            doc = dict(relation)
            doc.update({"_id": relation_id, "article_id": article_id, "tag_id": tag_id})
            remapped_relations[relation_id] = doc
        prepared["article_tags"] = list(remapped_relations.values())

    if tag_remaps:
        remapped_tag_relations: Dict[str, Dict[str, Any]] = {}
        for relation in prepared["tag_relations"]:
            parent_id = tag_remaps.get(
                int(relation["parent_tag_id"]), int(relation["parent_tag_id"])
            )
            child_id = tag_remaps.get(
                int(relation["child_tag_id"]), int(relation["child_tag_id"])
            )
            if parent_id == child_id:
                report["collections"]["tag_relations"]["skipped"] += 1
                continue
            relation_type = str(relation.get("relation_type") or "parent_child")
            relation_id = f"{relation_type}:{parent_id}:{child_id}"
            doc = dict(relation)
            doc.update(
                {
                    "_id": relation_id,
                    "parent_tag_id": parent_id,
                    "child_tag_id": child_id,
                }
            )
            remapped_tag_relations[relation_id] = doc
        prepared["tag_relations"] = list(remapped_tag_relations.values())

    target_categories = {
        int(doc["_id"]): str(doc.get("category") or "GENERAL")
        for doc in store.db.tags.find({}, {"_id": 1, "category": 1})
        if isinstance(doc.get("_id"), int)
    }
    target_categories.update(
        {
            int(doc["_id"]): str(doc.get("category") or "GENERAL")
            for doc in prepared["tags"]
        }
    )
    valid_tag_relations = []
    for relation in prepared["tag_relations"]:
        if target_categories.get(int(relation["parent_tag_id"])) != target_categories.get(
            int(relation["child_tag_id"])
        ):
            report["collections"]["tag_relations"]["skipped"] += 1
            continue
        valid_tag_relations.append(relation)
    prepared["tag_relations"] = valid_tag_relations

    for name, documents in prepared.items():
        report["collections"][name]["prepared"] = len(documents)


def _write_documents(collection: Any, documents: List[Dict[str, Any]], batch_size: int) -> int:
    written = 0
    for batch in _chunks(documents, batch_size):
        if ReplaceOne is not None:
            operations = [ReplaceOne({"_id": doc["_id"]}, doc, upsert=True) for doc in batch]
            try:
                collection.bulk_write(operations, ordered=True)
                written += len(batch)
                continue
            except TypeError:
                # mongomock and some compatible servers lag PyMongo's bulk API.
                pass
        for doc in batch:
            collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)
            written += 1
    return written


def migrate_tinydb_to_mongodb(
    tinydb_path: str,
    *,
    uri: str = "mongodb://localhost:27017",
    database: str = "feedsummary",
    batch_size: int = 500,
    conflict_policy: str = "error",
    dry_run: bool = False,
    verify: bool = True,
    client: Any = None,
    connect_timeout_ms: int = 5000,
) -> Dict[str, Any]:
    """Migrate every known FeedSummary TinyDB table into MongoDB."""
    source = Path(os.path.expandvars(os.path.expanduser(tinydb_path))).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"TinyDB file does not exist: {source}")
    if conflict_policy not in {"error", "keep-existing"}:
        raise ValueError("conflict_policy must be 'error' or 'keep-existing'")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    prepared, report, _, _ = _prepare_source(
        source,
        database,
        dry_run=dry_run,
        conflict_policy=conflict_policy,
    )
    if dry_run:
        return report

    store = MongoDBStore(
        uri=uri,
        database=database,
        client=client,
        connect_timeout_ms=connect_timeout_ms,
    )
    try:
        _resolve_target_conflicts(
            prepared,
            store,
            report,
            conflict_policy=conflict_policy,
        )
        for name in MIGRATED_TABLES:
            documents = prepared[name]
            report["collections"][name]["written"] = _write_documents(
                store.db[name], documents, batch_size
            )

        store._seed_counter("jobs", store.db.jobs)
        store._seed_counter("tags", store.db.tags)
        store._seed_counter("tag_categories", store.db.tag_categories)

        if verify:
            for name, documents in prepared.items():
                ids = [doc["_id"] for doc in documents]
                verified = 0
                for id_batch in _chunks([{"_id": value} for value in ids], batch_size):
                    batch_ids = [item["_id"] for item in id_batch]
                    verified += store.db[name].count_documents({"_id": {"$in": batch_ids}})
                report["collections"][name]["verified"] = int(verified)
                if verified != len(ids):
                    raise RuntimeError(
                        f"Verification failed for {name}: expected {len(ids)}, found {verified}"
                    )
        return report
    finally:
        store.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate a FeedSummary TinyDB database to MongoDB."
    )
    parser.add_argument("tinydb_path", help="Path to the TinyDB JSON file")
    parser.add_argument(
        "--uri",
        default=os.environ.get("FEEDSUMMARY_MONGODB_URI", "mongodb://localhost:27017"),
        help="MongoDB URI (default: FEEDSUMMARY_MONGODB_URI or localhost)",
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("FEEDSUMMARY_MONGODB_DATABASE", "feedsummary"),
        help="MongoDB database name (default: FEEDSUMMARY_MONGODB_DATABASE or feedsummary)",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--conflict-policy",
        choices=("error", "keep-existing"),
        default="error",
        help="How to handle unique URL/name collisions",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing")
    parser.add_argument("--no-verify", action="store_true", help="Skip post-write verification")
    parser.add_argument("--connect-timeout-ms", type=int, default=5000)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = migrate_tinydb_to_mongodb(
            args.tinydb_path,
            uri=args.uri,
            database=args.database,
            batch_size=args.batch_size,
            conflict_policy=args.conflict_policy,
            dry_run=args.dry_run,
            verify=not args.no_verify,
            connect_timeout_ms=args.connect_timeout_ms,
        )
    except Exception as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
