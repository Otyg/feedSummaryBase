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
#

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from feedsummary_core.persistence import CleanupPolicy
from feedsummary_core.persistence.helpers import classify_summary_doc

logger = logging.getLogger(__name__)


def _now_ts() -> int:
    return int(time.time())


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, bool):
            return int(v)
        return int(v)
    except Exception:
        return default


def _published_ts(doc: Dict[str, Any]) -> int:
    ts = doc.get("published_ts")
    if isinstance(ts, int) and ts > 0:
        return ts
    fa = doc.get("fetched_at")
    if isinstance(fa, int) and fa > 0:
        return fa
    return 0


def _normalize_summary_id(value: Any) -> Optional[str]:
    summary_id = str(value or "").strip()
    if summary_id.lower() in {"", "none", "null"}:
        return None
    return summary_id


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _json_loads(s: Optional[str]) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


class SqliteStore:
    """
    SQLite-backed store.

    Design choice:
    - Store the full document as JSON (doc_json) for flexibility.
    - Also store a few indexed columns for filtering/sorting (source, published_ts, fetched_at, etc.).
    """

    def __init__(
        self,
        path: str = "news_docs.sqlite",
        *,
        pragmas: Optional[Dict[str, str]] = None,
    ):
        self.path = str(Path(path))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._pragmas = pragmas or {
            "journal_mode": "WAL",
            "synchronous": "NORMAL",
            "temp_store": "MEMORY",
            "foreign_keys": "ON",
        }
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        for k, v in self._pragmas.items():
            try:
                con.execute(f"PRAGMA {k}={v}")
            except Exception:
                pass
        return con

    def _init_db(self) -> None:
        con = self._connect()
        try:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id            TEXT PRIMARY KEY,
                    url           TEXT,
                    source        TEXT,
                    title         TEXT,
                    published     TEXT,
                    published_ts  INTEGER,
                    fetched_at    INTEGER,
                    content_hash  TEXT,
                    summarized    INTEGER DEFAULT 0,
                    summarized_at INTEGER,
                    doc_json      TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_articles_source_ts
                  ON articles(source, published_ts);

                CREATE INDEX IF NOT EXISTS idx_articles_published_ts
                  ON articles(published_ts);

                CREATE INDEX IF NOT EXISTS idx_articles_fetched_at
                  ON articles(fetched_at);

                CREATE TABLE IF NOT EXISTS summary_docs (
                    id       TEXT PRIMARY KEY,
                    created  INTEGER,
                    kind     TEXT,
                    doc_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_summary_docs_created
                  ON summary_docs(created);

                CREATE TABLE IF NOT EXISTS jobs (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at  INTEGER,
                    started_at  INTEGER,
                    finished_at INTEGER,
                    status      TEXT,
                    message     TEXT,
                    summary_id  TEXT,
                    fields_json TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_created_at
                  ON jobs(created_at);

                CREATE TABLE IF NOT EXISTS temp_summaries (
                    job_id     INTEGER PRIMARY KEY,
                    created_at INTEGER,
                    summary    TEXT,
                    meta_json  TEXT
                );
                
                CREATE TABLE IF NOT EXISTS tags (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    name            TEXT UNIQUE NOT NULL,
                    category        TEXT DEFAULT 'GENERAL',
                    description     TEXT,
                    embedding_vector TEXT,
                    created_at      INTEGER,
                    updated_at      INTEGER
                );
                
                CREATE INDEX IF NOT EXISTS idx_tags_name
                    ON tags(name);
                
                CREATE INDEX IF NOT EXISTS idx_tags_category
                    ON tags(category);
                
                CREATE TABLE IF NOT EXISTS article_tags (
                    article_id TEXT NOT NULL,
                    tag_id     INTEGER NOT NULL,
                    created_at INTEGER,
                    motivering TEXT,
                    PRIMARY KEY (article_id, tag_id),
                    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_article_tags_article
                    ON article_tags(article_id);
                
                CREATE INDEX IF NOT EXISTS idx_article_tags_tag
                    ON article_tags(tag_id);
                
                CREATE UNIQUE INDEX IF NOT EXISTS ux_articles_url
                    ON articles(url)
                    WHERE url IS NOT NULL AND url != '';
                """
            )
            con.commit()
        finally:
            con.close()

    def get_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        con = self._connect()
        try:
            row = con.execute(
                "SELECT doc_json FROM articles WHERE id = ?",
                (str(article_id),),
            ).fetchone()
            if not row:
                return None
            doc = _json_loads(row["doc_json"]) or {}
            if "id" not in doc:
                doc["id"] = str(article_id)
            return doc
        finally:
            con.close()

    def upsert_article(self, article_doc: Dict[str, Any]) -> None:
        if not isinstance(article_doc, dict):
            raise ValueError("article_doc must be a dict")
        if not article_doc.get("id"):
            raise ValueError("article_doc must contain 'id'")

        doc = dict(article_doc)
        aid = str(doc["id"])

        url = (doc.get("url") or "").strip() or None
        source = (doc.get("source") or "").strip() or None
        title = (doc.get("title") or "").strip() or None
        published = (doc.get("published") or "").strip() or None
        published_ts = _safe_int(doc.get("published_ts"), 0) or None
        fetched_at = _safe_int(doc.get("fetched_at"), 0) or None
        content_hash = (doc.get("content_hash") or "").strip() or None

        summarized = 1 if bool(doc.get("summarized")) else 0
        summarized_at = _safe_int(doc.get("summarized_at"), 0) or None

        doc_json = _json_dumps(doc)

        con = self._connect()
        try:
            con.execute(
                """
                INSERT INTO articles (
                    id, url, source, title, published, published_ts, fetched_at,
                    content_hash, summarized, summarized_at, doc_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    url=excluded.url,
                    source=excluded.source,
                    title=excluded.title,
                    published=excluded.published,
                    published_ts=excluded.published_ts,
                    fetched_at=excluded.fetched_at,
                    content_hash=excluded.content_hash,
                    summarized=excluded.summarized,
                    summarized_at=excluded.summarized_at,
                    doc_json=excluded.doc_json
                """,
                (
                    aid,
                    url,
                    source,
                    title,
                    published,
                    published_ts,
                    fetched_at,
                    content_hash,
                    summarized,
                    summarized_at,
                    doc_json,
                ),
            )
            con.commit()
        finally:
            con.close()

    def list_articles(self, limit: int = 2000) -> List[Dict[str, Any]]:
        con = self._connect()
        try:
            rows = con.execute(
                """
                SELECT doc_json
                FROM articles
                ORDER BY COALESCE(published_ts, fetched_at, 0) ASC
                LIMIT ?
                """,
                (_safe_int(limit, 2000),),
            ).fetchall()

            out: List[Dict[str, Any]] = []
            for r in rows:
                doc = _json_loads(r["doc_json"]) or {}
                if isinstance(doc, dict):
                    out.append(doc)
            return out
        finally:
            con.close()

    def list_articles_by_filter(
        self,
        *,
        sources: List[str],
        since_ts: int,
        until_ts: Optional[int] = None,
        limit: int = 2000,
    ) -> List[Dict[str, Any]]:
        srcs = [str(s).strip() for s in (sources or []) if str(s).strip()]
        if not srcs:
            return []

        since_i = _safe_int(since_ts, 0)
        until_i = _safe_int(until_ts, 0) if until_ts is not None else None
        lim = _safe_int(limit, 2000)

        placeholders = ",".join(["?"] * len(srcs))
        params: List[Any] = []
        params.extend(srcs)
        params.append(since_i)

        where = f"source IN ({placeholders}) AND COALESCE(published_ts, fetched_at, 0) >= ?"
        if until_i is not None:
            where += " AND COALESCE(published_ts, fetched_at, 0) <= ?"
            params.append(until_i)

        params.append(lim)

        con = self._connect()
        try:
            rows = con.execute(
                f"""
                SELECT doc_json
                FROM articles
                WHERE {where}
                ORDER BY COALESCE(published_ts, fetched_at, 0) ASC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()

            out: List[Dict[str, Any]] = []
            for r in rows:
                doc = _json_loads(r["doc_json"]) or {}
                if isinstance(doc, dict):
                    out.append(doc)
            return out
        finally:
            con.close()

    def list_unsummarized_articles(self, limit: int = 200) -> List[Dict[str, Any]]:
        con = self._connect()
        try:
            rows = con.execute(
                """
                SELECT doc_json
                FROM articles
                WHERE summarized != 1
                ORDER BY COALESCE(published_ts, fetched_at, 0) ASC
                LIMIT ?
                """,
                (_safe_int(limit, 200),),
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                doc = _json_loads(r["doc_json"]) or {}
                if isinstance(doc, dict):
                    out.append(doc)
            return out
        finally:
            con.close()

    def mark_articles_summarized(self, article_ids: List[str]) -> None:
        ids = [str(x) for x in (article_ids or []) if str(x).strip()]
        if not ids:
            return

        ts = _now_ts()
        con = self._connect()
        try:
            con.execute("BEGIN")
            for aid in ids:
                row = con.execute(
                    "SELECT doc_json FROM articles WHERE id = ?",
                    (aid,),
                ).fetchone()
                if not row:
                    continue
                doc = _json_loads(row["doc_json"]) or {}
                if isinstance(doc, dict):
                    doc["summarized"] = True
                    doc["summarized_at"] = ts
                    con.execute(
                        """
                        UPDATE articles
                        SET summarized = 1, summarized_at = ?, doc_json = ?
                        WHERE id = ?
                        """,
                        (ts, _json_dumps(doc), aid),
                    )
                else:
                    con.execute(
                        "UPDATE articles SET summarized = 1, summarized_at = ? WHERE id = ?",
                        (ts, aid),
                    )
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def save_summary_doc(self, summary_doc: Dict[str, Any]) -> Any:
        if not isinstance(summary_doc, dict):
            raise ValueError("summary_doc must be a dict")

        doc = dict(summary_doc)
        created = _safe_int(doc.get("created"), 0) or _now_ts()
        kind = str(doc.get("kind") or "summary")
        sid = str(doc.get("id") or "").strip()

        if not sid:
            sid = f"summary_{created}"

        doc["id"] = sid
        doc["created"] = created
        doc["kind"] = kind

        con = self._connect()
        try:
            con.execute(
                """
                INSERT INTO summary_docs (id, created, kind, doc_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    created=excluded.created,
                    kind=excluded.kind,
                    doc_json=excluded.doc_json
                """,
                (sid, created, kind, _json_dumps(doc)),
            )
            con.commit()
            return sid
        finally:
            con.close()

    def get_summary_doc(self, summary_doc_id: str) -> Optional[Dict[str, Any]]:
        sid = str(summary_doc_id)
        con = self._connect()
        try:
            row = con.execute(
                "SELECT doc_json FROM summary_docs WHERE id = ?",
                (sid,),
            ).fetchone()
            if not row:
                return None
            doc = _json_loads(row["doc_json"]) or {}
            if isinstance(doc, dict) and "id" not in doc:
                doc["id"] = sid
            return doc if isinstance(doc, dict) else None
        finally:
            con.close()

    def list_summary_docs(self) -> List[Dict[str, Any]]:
        con = self._connect()
        try:
            rows = con.execute(
                """
                SELECT doc_json
                FROM summary_docs
                ORDER BY COALESCE(created, 0) DESC
                """
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                doc = _json_loads(r["doc_json"]) or {}
                if isinstance(doc, dict):
                    out.append(doc)
            return out
        finally:
            con.close()

    def get_latest_summary_doc(self) -> Optional[Dict[str, Any]]:
        con = self._connect()
        try:
            row = con.execute(
                """
                SELECT doc_json
                FROM summary_docs
                ORDER BY COALESCE(created, 0) DESC
                LIMIT 1
                """
            ).fetchone()
            if not row:
                return None
            doc = _json_loads(row["doc_json"]) or {}
            return doc if isinstance(doc, dict) else None
        finally:
            con.close()

    def create_job(self) -> int:
        con = self._connect()
        try:
            cur = con.execute(
                """
                INSERT INTO jobs (created_at, started_at, finished_at, status, message, summary_id, fields_json)
                VALUES (?, NULL, NULL, ?, ?, NULL, ?)
                """,
                (_now_ts(), "queued", "", _json_dumps({})),
            )
            con.commit()
            jid = int(cur.lastrowid)  # type: ignore
            logger.info("Job %s created", jid)
            return jid
        finally:
            con.close()

    def update_job(self, job_id: int, **fields) -> None:
        jid = _safe_int(job_id, 0)
        if jid <= 0:
            raise ValueError("job_id must be a positive int")

        known_cols = {
            "created_at",
            "started_at",
            "finished_at",
            "status",
            "message",
            "summary_id",
        }

        con = self._connect()
        try:
            row = con.execute("SELECT fields_json FROM jobs WHERE id = ?", (jid,)).fetchone()
            extra = _json_loads(row["fields_json"]) if row else {}
            if not isinstance(extra, dict):
                extra = {}

            if "summary_id" in fields:
                fields["summary_id"] = _normalize_summary_id(fields.get("summary_id"))

            set_parts: List[str] = []
            params: List[Any] = []

            for k in list(fields.keys()):
                if k in known_cols:
                    set_parts.append(f"{k} = ?")
                    params.append(fields[k])

            for k, v in fields.items():
                if k not in known_cols:
                    extra[k] = v

            set_parts.append("fields_json = ?")
            params.append(_json_dumps(extra))

            params.append(jid)

            con.execute(
                f"UPDATE jobs SET {', '.join(set_parts)} WHERE id = ?",
                tuple(params),
            )
            con.commit()
            logger.info("Job %s updated: %s", jid, fields)
        finally:
            con.close()

    def get_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        jid = _safe_int(job_id, 0)
        con = self._connect()
        try:
            row = con.execute(
                """
                SELECT id, created_at, started_at, finished_at, status, message, summary_id, fields_json
                FROM jobs
                WHERE id = ?
                """,
                (jid,),
            ).fetchone()
            if not row:
                return None

            out: Dict[str, Any] = {
                "id": int(row["id"]),
                "created_at": row["created_at"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
                "status": row["status"],
                "message": row["message"],
                "summary_id": row["summary_id"],
            }
            extra = _json_loads(row["fields_json"])
            if isinstance(extra, dict):
                out.update(extra)
            return out
        finally:
            con.close()

    def list_jobs(self, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Listar jobs i senaste-först ordning.
        Används av Qt för att visa avbrutna/återupptagbara jobb.
        """
        lim = _safe_int(limit, 200)
        con = self._connect()
        try:
            rows = con.execute(
                """
                SELECT id, created_at, started_at, finished_at, status, message, summary_id, fields_json
                FROM jobs
                ORDER BY COALESCE(created_at, 0) DESC
                LIMIT ?
                """,
                (lim,),
            ).fetchall()

            out: List[Dict[str, Any]] = []
            for row in rows:
                doc: Dict[str, Any] = {
                    "id": int(row["id"]),
                    "created_at": row["created_at"],
                    "started_at": row["started_at"],
                    "finished_at": row["finished_at"],
                    "status": row["status"],
                    "message": row["message"],
                    "summary_id": row["summary_id"],
                }
                extra = _json_loads(row["fields_json"])
                if isinstance(extra, dict):
                    doc.update(extra)
                out.append(doc)
            return out
        finally:
            con.close()

    def get_articles_by_ids(self, article_ids: List[str]) -> List[Dict[str, Any]]:
        ids = [str(x) for x in (article_ids or []) if str(x).strip()]
        if not ids:
            return []

        placeholders = ",".join(["?"] * len(ids))
        con = self._connect()
        try:
            rows = con.execute(
                f"SELECT doc_json FROM articles WHERE id IN ({placeholders})",
                tuple(ids),
            ).fetchall()

            by_id: Dict[str, Dict[str, Any]] = {}
            for r in rows:
                doc = _json_loads(r["doc_json"]) or {}
                if isinstance(doc, dict) and doc.get("id"):
                    by_id[str(doc["id"])] = doc

            ordered: List[Dict[str, Any]] = []
            for aid in ids:
                if aid in by_id:
                    ordered.append(by_id[aid])
            return ordered
        finally:
            con.close()

    def save_temp_summary(self, job_id: int, summary_text: str, meta: Dict[str, Any]) -> None:
        jid = _safe_int(job_id, 0)
        con = self._connect()
        try:
            con.execute(
                """
                INSERT INTO temp_summaries (job_id, created_at, summary, meta_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    created_at=excluded.created_at,
                    summary=excluded.summary,
                    meta_json=excluded.meta_json
                """,
                (jid, _now_ts(), summary_text or "", _json_dumps(meta or {})),
            )
            con.commit()
        finally:
            con.close()

    def get_temp_summary(self, job_id: int) -> Optional[Dict[str, Any]]:
        jid = _safe_int(job_id, 0)
        con = self._connect()
        try:
            row = con.execute(
                "SELECT job_id, created_at, summary, meta_json FROM temp_summaries WHERE job_id = ?",
                (jid,),
            ).fetchone()
            if not row:
                return None
            return {
                "job_id": int(row["job_id"]),
                "created_at": row["created_at"],
                "summary": row["summary"],
                "meta": _json_loads(row["meta_json"]) or {},
            }
        finally:
            con.close()

    def run_cleanup(self, pol: CleanupPolicy) -> Dict[str, int]:
        """
        Cleanup for SqliteStore schema (articles, summary_docs, temp_summaries, jobs).
        summary_docs: delete based on created + prompt_package classification.
        """
        now = int(time.time())
        cut_articles = now - pol.articles_days * 86400
        cut_daily = now - pol.daily_summaries_days * 86400
        cut_weekly = now - pol.weekly_summaries_days * 86400
        cut_temp = now - pol.temp_summaries_days * 86400
        cut_jobs = now - pol.jobs_days * 86400

        removed = {"articles": 0, "summary_docs": 0, "temp_summaries": 0, "jobs": 0}

        con = self._connect()
        try:
            con.row_factory = sqlite3.Row

            # Articles
            cur = con.execute(
                "DELETE FROM articles WHERE COALESCE(published_ts, fetched_at, 0) < ?",
                (cut_articles,),
            )
            removed["articles"] = cur.rowcount if cur.rowcount is not None else 0

            # Temp summaries
            cur = con.execute(
                "DELETE FROM temp_summaries WHERE COALESCE(created_at, 0) < ?",
                (cut_temp,),
            )
            removed["temp_summaries"] = cur.rowcount if cur.rowcount is not None else 0

            # Jobs: only delete old finished ones (never touch running/queued)
            cur = con.execute(
                """
                DELETE FROM jobs
                WHERE COALESCE(finished_at, created_at, 0) < ?
                AND COALESCE(status, '') IN ('done','failed')
                """,
                (cut_jobs,),
            )
            removed["jobs"] = cur.rowcount if cur.rowcount is not None else 0

            # Summary docs: fetch candidates up to max cutoff, classify in Python, delete by id
            max_cut = max(cut_daily, cut_weekly)
            rows = con.execute(
                "SELECT id, created, doc_json FROM summary_docs WHERE COALESCE(created,0) < ?",
                (max_cut,),
            ).fetchall()

            to_delete: List[str] = []
            for r in rows:
                sid = str(r["id"])
                created = int(r["created"] or 0)
                kind = classify_summary_doc(str(r["doc_json"] or ""))

                if kind == "daily" and created < cut_daily:
                    to_delete.append(sid)
                elif kind == "weekly" and created < cut_weekly:
                    to_delete.append(sid)
                elif kind == "other" and created < cut_weekly:
                    # unknown => keep like weekly by default
                    to_delete.append(sid)

            if to_delete:
                # chunk deletes
                chunk = 200
                for i in range(0, len(to_delete), chunk):
                    part = to_delete[i : i + chunk]
                    placeholders = ",".join(["?"] * len(part))
                    cur = con.execute(
                        f"DELETE FROM summary_docs WHERE id IN ({placeholders})",
                        tuple(part),
                    )
                    removed["summary_docs"] += cur.rowcount if cur.rowcount is not None else 0

            con.commit()
            return removed
        finally:
            con.close()

    # ============================================================================
    # Tag management methods
    # ============================================================================

    def add_tag(
        self,
        name: str,
        category: str = "GENERAL",
        description: Optional[str] = None,
    ) -> Optional[int]:
        """Add a new tag (returns tag ID or None if it already exists)."""
        if not name or not isinstance(name, str):
            return None

        name = name.strip().lower()
        if not name:
            return None

        con = self._connect()
        try:
            # Try to insert
            try:
                cur = con.execute(
                    """
                    INSERT INTO tags (name, category, description, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (name, category, description, _now_ts(), _now_ts()),
                )
                con.commit()
                return int(cur.lastrowid)  # type: ignore
            except sqlite3.IntegrityError:
                # Tag already exists, fetch and return its ID
                row = con.execute(
                    "SELECT id FROM tags WHERE name = ?",
                    (name,),
                ).fetchone()
                if row:
                    return int(row["id"])
                return None
        finally:
            con.close()

    def get_tag_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a tag by name (case-insensitive)."""
        if not name or not isinstance(name, str):
            return None

        name = name.strip().lower()
        if not name:
            return None

        con = self._connect()
        try:
            row = con.execute(
                "SELECT id, name, category, description, embedding_vector, created_at FROM tags WHERE name = ?",
                (name,),
            ).fetchone()
            if row:
                tag_dict = {
                    "id": int(row["id"]),
                    "name": row["name"],
                    "category": row["category"],
                    "description": row["description"],
                    "created_at": row["created_at"],
                }
                # Parse embedding_vector if present
                embedding_str = row.get("embedding_vector")
                if embedding_str:
                    try:
                        import json
                        tag_dict["embedding_vector"] = json.loads(embedding_str)
                    except Exception:
                        pass
                return tag_dict
            return None
        finally:
            con.close()

    def get_all_tags(self) -> List[Dict[str, Any]]:
        """Get all tags."""
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT id, name, category, description, embedding_vector, created_at FROM tags ORDER BY name"
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for row in rows:
                tag_dict = {
                    "id": int(row["id"]),
                    "name": row["name"],
                    "category": row["category"],
                    "description": row["description"],
                    "created_at": row["created_at"],
                }
                # Parse embedding_vector if present
                embedding_str = row.get("embedding_vector")
                if embedding_str:
                    try:
                        import json
                        tag_dict["embedding_vector"] = json.loads(embedding_str)
                    except Exception:
                        pass
                out.append(tag_dict)
            return out
        finally:
            con.close()

    def add_article_tags(
        self,
        article_id: str,
        tag_ids: List,  # Can be List[int] or List[Dict] with 'tag_id' and optional 'reasoning'
    ) -> None:
        """Add tags to an article (removes existing tags first).
        
        Args:
            article_id: Article ID
            tag_ids: List of tag IDs (int) or list of dicts with 'tag_id' and optional 'reasoning'
        """
        if not article_id or not tag_ids:
            return

        article_id = str(article_id).strip()
        if not article_id:
            return

        con = self._connect()
        try:
            # Remove existing tags for this article
            con.execute("DELETE FROM article_tags WHERE article_id = ?", (article_id,))

            # Add new tags
            now_ts = _now_ts()
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
                
                con.execute(
                    """
                    INSERT INTO article_tags (article_id, tag_id, created_at, motivering)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(article_id, tag_id) DO UPDATE SET motivering = excluded.motivering
                    """,
                    (article_id, tag_id, now_ts, reasoning if reasoning else None),
                )

            con.commit()
        finally:
            con.close()

    def get_article_tags(self, article_id: str) -> List[Dict[str, Any]]:
        """Get all tags for an article."""
        if not article_id:
            return []

        article_id = str(article_id).strip()
        con = self._connect()
        try:
            rows = con.execute(
                """
                SELECT t.id, t.name, t.category, t.description, t.created_at
                FROM tags t
                JOIN article_tags at ON t.id = at.tag_id
                WHERE at.article_id = ?
                ORDER BY t.name
                """,
                (article_id,),
            ).fetchall()

            out: List[Dict[str, Any]] = []
            for row in rows:
                out.append({
                    "id": int(row["id"]),
                    "name": row["name"],
                    "category": row["category"],
                    "description": row["description"],
                    "created_at": row["created_at"],
                })
            return out
        finally:
            con.close()

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
        con = self._connect()
        try:
            cursor = con.execute(
                "DELETE FROM article_tags WHERE article_id = ? AND tag_id = ?",
                (article_id, int(tag_id)),
            )
            con.commit()
            return cursor.rowcount > 0
        finally:
            con.close()

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
        con = self._connect()
        try:
            now_ts = _now_ts()
            con.execute(
                """
                INSERT INTO article_tags (article_id, tag_id, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(article_id, tag_id) DO NOTHING
                """,
                (article_id, tag_id, now_ts),
            )
            con.commit()
            return con.total_changes > 0
        finally:
            con.close()

    def create_tag(
        self, name: str, category: str = "GENERAL", description: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Create a new tag.
        
        Args:
            name: Tag name
            category: Tag category (GENERAL, DOMAIN_ENTITY, etc.)
            description: Optional description
            
        Returns:
            Created tag dict, or None if tag already exists
        """
        if not name:
            return None

        name = name.strip()
        category = category.strip() or "GENERAL"
        description = description.strip() if description else ""

        con = self._connect()
        try:
            # Check if tag already exists
            existing = con.execute(
                "SELECT id FROM tags WHERE name = ?", (name,)
            ).fetchone()
            if existing:
                return None

            now_ts = _now_ts()
            cursor = con.execute(
                """
                INSERT INTO tags (name, category, description, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (name, category, description, now_ts),
            )
            con.commit()

            tag_id = cursor.lastrowid
            return {
                "id": int(tag_id),
                "name": name,
                "category": category,
                "description": description,
                "created_at": now_ts,
            }
        finally:
            con.close()

    def update_tag(
        self, tag_id: int, name: str = None, category: str = None, description: str = None
    ) -> Optional[Dict[str, Any]]:
        """Update an existing tag.
        
        Args:
            tag_id: Tag ID to update
            name: New name (optional)
            category: New category (optional)
            description: New description (optional)
            
        Returns:
            Updated tag dict, or None if tag not found
        """
        if not tag_id:
            return None

        con = self._connect()
        try:
            # Get current tag
            row = con.execute("SELECT * FROM tags WHERE id = ?", (int(tag_id),)).fetchone()
            if not row:
                return None

            # Prepare updates
            updates = {}
            if name is not None:
                updates["name"] = name.strip()
            if category is not None:
                updates["category"] = category.strip() or "GENERAL"
            if description is not None:
                updates["description"] = description.strip() if description else ""

            if not updates:
                # No changes
                return {
                    "id": int(row["id"]),
                    "name": row["name"],
                    "category": row["category"],
                    "description": row["description"],
                    "created_at": row["created_at"],
                }

            # Build update SQL
            set_clauses = [f"{k} = ?" for k in updates.keys()]
            values = list(updates.values())
            values.append(int(tag_id))

            con.execute(
                f"UPDATE tags SET {', '.join(set_clauses)} WHERE id = ?",
                values,
            )
            con.commit()

            # Get updated tag
            updated_row = con.execute("SELECT * FROM tags WHERE id = ?", (int(tag_id),)).fetchone()
            return {
                "id": int(updated_row["id"]),
                "name": updated_row["name"],
                "category": updated_row["category"],
                "description": updated_row["description"],
                "created_at": updated_row["created_at"],
            }
        finally:
            con.close()

    def delete_tag(self, tag_id: int) -> bool:
        """Delete a tag and remove it from all articles.
        
        Args:
            tag_id: Tag ID to delete
            
        Returns:
            True if tag was deleted, False if not found
        """
        if not tag_id:
            return False

        con = self._connect()
        try:
            # Delete from article_tags first
            con.execute("DELETE FROM article_tags WHERE tag_id = ?", (int(tag_id),))
            
            # Delete the tag
            cursor = con.execute("DELETE FROM tags WHERE id = ?", (int(tag_id),))
            con.commit()
            
            return cursor.rowcount > 0
        finally:
            con.close()

    def cleanup_unused_tags(self, days: int = 30) -> int:
        """Remove tags that haven't been used in X days."""
        cutoff = _now_ts() - (days * 86400)
        con = self._connect()
        try:
            # Find tags not used in the last X days
            cur = con.execute(
                """
                DELETE FROM tags
                WHERE id NOT IN (
                    SELECT DISTINCT tag_id FROM article_tags
                    WHERE created_at > ?
                )
                AND created_at < ?
                """,
                (cutoff, cutoff),
            )
            con.commit()
            return cur.rowcount if cur.rowcount is not None else 0
        finally:
            con.close()

    def update_tag_embedding(self, tag_id: int, embedding_vector: List[float]) -> bool:
        """
        Update the embedding vector for a tag.
        
        Args:
            tag_id: Tag ID
            embedding_vector: List of floats representing the embedding
            
        Returns:
            True if successful, False otherwise
        """
        if not isinstance(tag_id, int) or tag_id <= 0:
            return False
        
        if not embedding_vector or not all(isinstance(x, (int, float)) for x in embedding_vector):
            return False
        
        con = self._connect()
        try:
            import json
            embedding_json = json.dumps(embedding_vector, separators=(",", ":"))
            con.execute(
                "UPDATE tags SET embedding_vector = ?, updated_at = ? WHERE id = ?",
                (embedding_json, _now_ts(), tag_id),
            )
            con.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating tag embedding: {e}")
            return False
        finally:
            con.close()

    def get_tags_by_embedding_similarity(
        self,
        embedding_vector: List[float],
        similarity_threshold: float = 0.75,
        limit: int = 10,
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
        
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT id, name, category, description, embedding_vector, created_at FROM tags WHERE embedding_vector IS NOT NULL ORDER BY name"
            ).fetchall()
            
            results: List[Tuple[Dict[str, Any], float]] = []
            
            for row in rows:
                try:
                    import json
                    embedding_str = row.get("embedding_vector")
                    if not embedding_str:
                        continue
                    
                    tag_embedding = json.loads(embedding_str)
                    if not isinstance(tag_embedding, list):
                        continue
                    
                    # Compute cosine similarity
                    similarity = self._cosine_similarity(embedding_vector, tag_embedding)
                    
                    if similarity >= similarity_threshold:
                        tag_dict = {
                            "id": int(row["id"]),
                            "name": row["name"],
                            "category": row["category"],
                            "description": row["description"],
                            "created_at": row["created_at"],
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
            con.close()

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

        tag_names = [str(t).strip().lower() for t in tag_names if t]
        if not tag_names:
            return []

        con = self._connect()
        try:
            # Get tag IDs
            placeholders = ",".join(["?"] * len(tag_names))
            rows = con.execute(
                f"SELECT id FROM tags WHERE name IN ({placeholders})",
                tuple(tag_names),
            ).fetchall()

            tag_ids = [int(row["id"]) for row in rows]
            if not tag_ids:
                return []

            if match_mode == "all":
                # Articles that have ALL tags
                tag_id_placeholders = ",".join(["?"] * len(tag_ids))
                article_rows = con.execute(
                    f"""
                    SELECT article_id, COUNT(*) as tag_count
                    FROM article_tags
                    WHERE tag_id IN ({tag_id_placeholders})
                    GROUP BY article_id
                    HAVING tag_count = ?
                    """,
                    tuple(tag_ids) + (len(tag_ids),),
                ).fetchall()
            else:
                # Articles that have ANY tag (default)
                tag_id_placeholders = ",".join(["?"] * len(tag_ids))
                article_rows = con.execute(
                    f"""
                    SELECT DISTINCT article_id
                    FROM article_tags
                    WHERE tag_id IN ({tag_id_placeholders})
                    """,
                    tuple(tag_ids),
                ).fetchall()

            article_ids = [str(row["article_id"]) for row in article_rows]
            if not article_ids:
                return []

            # Fetch full article documents
            return self.get_articles_by_ids(article_ids)

        finally:
            con.close()
