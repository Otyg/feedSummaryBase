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
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from tinydb import Query, TinyDB

from feedsummary_core.persistence import CleanupPolicy
from feedsummary_core.tagging_rules import VULNERABILITY_TAG_CATEGORY, is_cve_tag

logger = logging.getLogger(__name__)


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
        db = self._db()
        A = Query()
        db.table("articles").upsert(article_doc, A.id == article_doc["id"])
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
        db = self._db()
        A = Query()
        ts = int(time.time())
        for aid in article_ids:
            db.table("articles").update({"summarized": True, "summarized_at": ts}, A.id == aid)
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

        removed = {"articles": 0, "summary_docs": 0, "temp_summaries": 0, "jobs": 0}

        # TinyDB import here to avoid dependency if user doesn't use it
        from tinydb import TinyDB

        db = TinyDB(self.path)
        try:
            # Articles
            at = db.table("articles")
            # remove uses a predicate for each row
            before = len(at)
            at.remove(
                lambda r: int(r.get("published_ts") or r.get("fetched_at") or 0) < cut_articles
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

            return removed
        finally:
            db.close()

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
                out.append(tag_dict)
            # Sort by name
            out.sort(key=lambda x: x.get("name", ""))
            return out
        finally:
            db.close()

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

    def update_tag_embedding(self, tag_id: int, embedding_vector: List[float]) -> bool:
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
            Q = Query()
            # Update the tag with the embedding_vector
            t.update({"embedding_vector": embedding_vector}, doc_ids=[tag_id])
            return True
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
            tags_table.remove(
                lambda r: (
                    int(r.get("created_at", 0)) < cutoff
                    and int(r.get("id", 0)) not in used_tag_ids
                )
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
                    tags_table.update(
                        {"category": VULNERABILITY_TAG_CATEGORY},
                        doc_ids=[tag.doc_id],
                    )
        finally:
            db.close()
