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
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
from feedsummary_core.persistence.CleanUpPolicy import CleanupPolicy
from feedsummary_core.persistence.TinyDbStore import TinyDBStore
from feedsummary_core.persistence.SqliteStore import SqliteStore
from feedsummary_core.persistence.MongoDBStore import MongoDBStore


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
    ) -> bool: ...

    def list_unsummarized_articles(self, limit: int = 200) -> List[Dict[str, Any]]: ...

    def list_articles(self, limit: int = 2000) -> List[Dict[str, Any]]: ...

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


def _expand_path(p: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(p))
    return str(Path(expanded).resolve())


def create_store(cfg: Dict[str, Any]) -> NewsStore:
    """Instantiate the configured storage backend and ensure its parent path exists."""

    provider = (cfg.get("provider") or cfg.get("type") or "tinydb").lower()

    if provider == "tinydb":
        raw_path = cfg.get("path", "news_docs.json")
        path = _expand_path(raw_path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return TinyDBStore(path=path)  # type: ignore

    if provider in ("sqlite", "sqlite3"):
        raw_path = cfg.get("path", "news_docs.sqlite")
        path = _expand_path(raw_path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return SqliteStore(path=path)  # type: ignore

    if provider in ("mongo", "mongodb"):
        return MongoDBStore(
            uri=cfg.get("uri", "mongodb://localhost:27017"),
            database=cfg.get("database") or cfg.get("database_name") or "feedsummary",
            client=cfg.get("client"),
            connect_timeout_ms=int(cfg.get("connect_timeout_ms", 5000)),
            initialize_schema=bool(cfg.get("initialize_schema", True)),
        )  # type: ignore

    raise ValueError(f"Unsupported store provider: {provider}")
