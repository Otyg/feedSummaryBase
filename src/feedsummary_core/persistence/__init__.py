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

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
from feedsummary_core.persistence.CleanUpPolicy import CleanupPolicy
from feedsummary_core.persistence.TinyDbStore import TinyDBStore
from feedsummary_core.persistence.SqliteStore import SqliteStore
from feedsummary_core.persistence.MongoDBStore import MongoDBStore
from feedsummary_core.persistence.tag_relations import TagRelationError


class StoreError(Exception):
    """Base exception for persistence-layer failures."""

    pass


class NewsStore(Protocol):
    """Protocol that all article and summary stores must implement."""

    def get_article(self, article_id: str) -> Optional[Dict[str, Any]]: ...

    def upsert_article(self, article_doc: Dict[str, Any]) -> None: ...

    def update_article_embedding(
        self,
        article_id: str,
        embedding_vector: List[float],
        *,
        model: Optional[str] = None,
        source_hash: Optional[str] = None,
        purpose: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> bool: ...

    def list_unsummarized_articles(self, limit: int = 200) -> List[Dict[str, Any]]: ...

    def list_articles(self, limit: int = 2000) -> List[Dict[str, Any]]: ...

    def iter_articles(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]: ...

    def list_articles_by_filter(
        self,
        *,
        sources: List[str],
        since_ts: int,
        until_ts: Optional[int] = None,
        limit: int = 2000,
    ) -> List[Dict[str, Any]]: ...

    def mark_articles_summarized(self, article_ids: List[str]) -> None: ...

    def save_summary_doc(self, summary_doc: Dict[str, Any]) -> Any: ...

    def get_summary_doc(self, summary_doc_id: str) -> Optional[Dict[str, Any]]: ...

    def list_summary_docs(self) -> List[Dict[str, Any]]: ...

    def get_latest_summary_doc(self) -> Optional[Dict[str, Any]]: ...

    # Jobs / resume support
    def create_job(self) -> int: ...

    def update_job(self, job_id: int, **fields) -> None: ...

    def get_job(self, job_id: int) -> Optional[Dict[str, Any]]: ...

    def list_jobs(self, limit: int = 200) -> List[Dict[str, Any]]: ...

    def get_articles_by_ids(self, article_ids: List[str]) -> List[Dict[str, Any]]: ...

    def save_temp_summary(self, job_id: int, summary_text: str, meta: Dict[str, Any]) -> None: ...

    def get_temp_summary(self, job_id: int) -> Optional[Dict[str, Any]]: ...

    def run_cleanup(self, pol: CleanupPolicy) -> Dict[str, int]: ...

    # Tag management
    def add_tag(
        self,
        name: str,
        category: str = "GENERAL",
        description: Optional[str] = None,
    ) -> Optional[int]: ...

    def get_tag_by_name(self, name: str) -> Optional[Dict[str, Any]]: ...

    def get_all_tags(self) -> List[Dict[str, Any]]: ...

    def get_tag_relations(self, tag_id: int) -> Dict[str, List[Dict[str, Any]]]: ...

    def set_tag_relations(
        self,
        tag_id: int,
        *,
        parent_ids: Optional[List[int]] = None,
        child_ids: Optional[List[int]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]: ...

    def iter_articles_with_tags(
        self,
        *,
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ): ...

    def add_article_tags(self, article_id: str, tag_ids: List) -> None: ...

    def add_tag_to_article(self, article_id: str, tag_id: int) -> bool: ...

    def get_article_tags(self, article_id: str) -> List[Dict[str, Any]]: ...

    def cleanup_unused_tags(self, days: int = 30) -> int: ...

    def remove_article_tag(self, article_id: str, tag_id: int) -> bool: ...

    def create_tag(
        self,
        name: str,
        category: str = "GENERAL",
        description: str = "",
    ) -> Optional[Dict[str, Any]]: ...

    def update_tag(
        self,
        tag_id: int,
        name: Optional[str] = None,
        category: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]: ...

    def delete_tag(self, tag_id: int) -> bool: ...

    def update_tag_embedding(
        self,
        tag_id: int,
        embedding_vector: List[float],
        *,
        model: Optional[str] = None,
        source_hash: Optional[str] = None,
    ) -> bool: ...

    def get_tags_by_embedding_similarity(
        self,
        embedding_vector: List[float],
        similarity_threshold: float = 0.75,
        limit: int = 10,
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]: ...

    def get_articles_by_tags(
        self,
        tag_names: List[str],
        match_mode: str = "any",
    ) -> List[Dict[str, Any]]: ...

    def get_all_categories(self) -> List[Dict[str, Any]]: ...

    def get_category(self, category_id: int) -> Optional[Dict[str, Any]]: ...

    def create_category(
        self,
        name: str,
        label: str,
        bg_color: str = "bg-secondary",
        text_color: str = "text-dark",
        description: str = "",
    ) -> Optional[Dict[str, Any]]: ...

    def update_category(
        self,
        category_id: int,
        label: Optional[str] = None,
        bg_color: Optional[str] = None,
        text_color: Optional[str] = None,
        description: Optional[str] = None,
    ) -> bool: ...

    def delete_category(self, category_id: int) -> bool: ...

    def initialize_default_categories(self) -> None: ...

    # Long-term threat-landscape analysis
    def list_articles_for_long_term(
        self,
        *,
        after_fetched_at: int = 0,
        after_article_id: str = "",
        until_fetched_at: Optional[int] = None,
        sources: Optional[List[str]] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]: ...

    def get_long_term_cursor(self, profile_id: str) -> Dict[str, Any]: ...

    def claim_long_term_lease(
        self,
        profile_id: str,
        owner_id: str,
        *,
        now_ts: int,
        lease_seconds: int,
    ) -> bool: ...

    def renew_long_term_lease(
        self,
        profile_id: str,
        owner_id: str,
        *,
        now_ts: int,
        lease_seconds: int,
    ) -> bool: ...

    def release_long_term_lease(self, profile_id: str, owner_id: str) -> bool: ...

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
    ) -> bool: ...

    def get_threat_cluster(self, cluster_id: str) -> Optional[Dict[str, Any]]: ...

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
    ) -> List[Dict[str, Any]]: ...

    def save_threat_cluster(
        self,
        cluster_doc: Dict[str, Any],
        *,
        expected_membership_revision: Optional[int] = None,
    ) -> bool: ...

    def resolve_threat_cluster_review(
        self,
        cluster_doc: Dict[str, Any],
        *,
        expected_membership_revision: int,
    ) -> bool: ...

    def get_cluster_membership(
        self, profile_id: str, article_id: str
    ) -> Optional[Dict[str, Any]]: ...

    def save_cluster_membership(self, membership_doc: Dict[str, Any]) -> bool: ...

    def save_cluster_assignment(
        self,
        cluster_doc: Dict[str, Any],
        membership_doc: Dict[str, Any],
        *,
        expected_membership_revision: Optional[int] = None,
    ) -> bool: ...

    def get_long_term_quarantine(
        self, profile_id: str, article_id: str
    ) -> Optional[Dict[str, Any]]: ...

    def save_long_term_quarantine(self, quarantine_doc: Dict[str, Any]) -> bool: ...

    def resolve_long_term_quarantine(
        self, profile_id: str, article_id: str, *, resolved_at: int
    ) -> bool: ...

    def list_long_term_quarantine(
        self,
        profile_id: str,
        *,
        status: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]: ...

    def list_cluster_memberships(
        self, cluster_id: str, *, limit: int = 10000
    ) -> List[Dict[str, Any]]: ...

    def apply_cluster_reconciliation(
        self, reconciliation_doc: Dict[str, Any]
    ) -> bool: ...

    def get_cluster_reconciliation(
        self, reconciliation_id: str
    ) -> Optional[Dict[str, Any]]: ...

    def apply_cluster_membership_edit(self, edit_doc: Dict[str, Any]) -> bool: ...

    def get_cluster_membership_edit(
        self, edit_id: str
    ) -> Optional[Dict[str, Any]]: ...

    def save_cluster_snapshot(self, snapshot_doc: Dict[str, Any]) -> bool: ...

    def save_cluster_snapshot_revision(
        self,
        cluster_doc: Dict[str, Any],
        snapshot_doc: Dict[str, Any],
        *,
        expected_membership_revision: int,
        expected_summarized_revision: int,
    ) -> bool: ...

    def get_cluster_snapshot(
        self, snapshot_id: str
    ) -> Optional[Dict[str, Any]]: ...

    def list_cluster_snapshots(
        self,
        profile_id: str,
        *,
        cluster_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]: ...

    def save_threat_landscape_report(self, report_doc: Dict[str, Any]) -> bool: ...

    def get_threat_landscape_report(self, report_id: str) -> Optional[Dict[str, Any]]: ...

    def list_threat_landscape_reports(
        self, profile_id: str, *, limit: int = 100
    ) -> List[Dict[str, Any]]: ...

    def create_long_term_run(self, run_doc: Dict[str, Any]) -> bool: ...

    def get_long_term_run(self, run_id: str) -> Optional[Dict[str, Any]]: ...

    def list_long_term_runs(
        self, profile_id: str, *, limit: int = 100
    ) -> List[Dict[str, Any]]: ...

    def update_long_term_run(
        self,
        run_id: str,
        *,
        expected_status: str,
        fields: Dict[str, Any],
    ) -> bool: ...


def _expand_path(p: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(p))
    return str(Path(expanded).resolve())


def create_store(cfg: Dict[str, Any]) -> NewsStore:
    """Instantiate the configured storage backend and ensure its parent path exists."""

    provider = (cfg.get("provider") or cfg.get("type") or "tinydb").lower()
    initialize_schema = bool(cfg.get("initialize_schema", True))

    if provider == "tinydb":
        raw_path = cfg.get("path", "news_docs.json")
        path = _expand_path(raw_path)
        if initialize_schema:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        elif not Path(path).is_file():
            raise FileNotFoundError(f"TinyDB database does not exist: {path}")
        return TinyDBStore(path=path)  # type: ignore

    if provider in ("sqlite", "sqlite3"):
        raw_path = cfg.get("path", "news_docs.sqlite")
        path = _expand_path(raw_path)
        return SqliteStore(path=path, initialize_schema=initialize_schema)  # type: ignore

    if provider in ("mongo", "mongodb"):
        return MongoDBStore(
            uri=cfg.get("uri", "mongodb://localhost:27017"),
            database=cfg.get("database") or cfg.get("database_name") or "feedsummary",
            client=cfg.get("client"),
            connect_timeout_ms=int(cfg.get("connect_timeout_ms", 5000)),
            initialize_schema=initialize_schema,
        )  # type: ignore

    raise ValueError(f"Unsupported store provider: {provider}")
