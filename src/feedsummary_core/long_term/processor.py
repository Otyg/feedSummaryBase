# LICENSE HEADER MANAGED BY add-license-header
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Martin Vesterlund
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""Incremental, resumable orchestration for deterministic threat clustering."""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from feedsummary_core.long_term.clustering import (
    AssignmentAction,
    ClusteringSettings,
    VectorValidationError,
    add_cluster_member,
    assign_article,
    cluster_status_at,
    cosine_similarity,
    create_cluster,
)
from feedsummary_core.long_term.models import (
    ACTIVE_CLUSTER_STATUSES,
    ClusterStatus,
    EmbeddingSignature,
    ThreatCluster,
)

_CVE_PATTERN = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
_STRONG_TAG_CATEGORIES = frozenset(
    {"THREAT", "VULNERABILITY", "ORGANIZATION", "PRODUCT"}
)


class LongTermStore(Protocol):
    """Narrow persistence contract needed by the incremental processor."""

    def list_articles_for_long_term(self, **kwargs: Any) -> list[dict[str, Any]]: ...

    def get_long_term_cursor(self, profile_id: str) -> dict[str, Any]: ...

    def get_articles_by_ids(self, article_ids: list[str]) -> list[dict[str, Any]]: ...

    def get_article_tags(self, article_id: str) -> list[dict[str, Any]]: ...

    def claim_long_term_lease(
        self,
        profile_id: str,
        owner_id: str,
        *,
        now_ts: int,
        lease_seconds: int,
    ) -> bool: ...

    def release_long_term_lease(self, profile_id: str, owner_id: str) -> bool: ...

    def advance_long_term_cursor(self, profile_id: str, owner_id: str, **kwargs: Any) -> bool: ...

    def list_threat_clusters(self, profile_id: str, **kwargs: Any) -> list[dict[str, Any]]: ...

    def save_threat_cluster(
        self,
        cluster_doc: dict[str, Any],
        *,
        expected_membership_revision: int | None = None,
    ) -> bool: ...

    def get_cluster_membership(
        self, profile_id: str, article_id: str
    ) -> dict[str, Any] | None: ...

    def save_cluster_assignment(
        self,
        cluster_doc: dict[str, Any],
        membership_doc: dict[str, Any],
        *,
        expected_membership_revision: int | None = None,
    ) -> bool: ...

    def save_long_term_quarantine(self, quarantine_doc: dict[str, Any]) -> bool: ...

    def resolve_long_term_quarantine(
        self, profile_id: str, article_id: str, *, resolved_at: int
    ) -> bool: ...

    def list_long_term_quarantine(
        self, profile_id: str, *, status: str | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]: ...

    def create_long_term_run(self, run_doc: dict[str, Any]) -> bool: ...

    def update_long_term_run(
        self,
        run_id: str,
        *,
        expected_status: str,
        fields: dict[str, Any],
    ) -> bool: ...


class ArticleQualityError(ValueError):
    """An article cannot safely move through the long-term cursor yet."""


class LeaseUnavailableError(RuntimeError):
    """Another worker currently owns the profile lease."""


class ConcurrentAssignmentError(RuntimeError):
    """The cluster revision changed while an assignment was being saved."""


@dataclass(frozen=True)
class IncrementalSettings:
    batch_size: int = 200
    lease_seconds: int = 300
    max_future_skew_seconds: int = 3600
    quarantine_retry_limit: int = 50
    dormant_after_days: int = 30
    close_after_days: int = 180
    clustering: ClusteringSettings = field(default_factory=ClusteringSettings)

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        if self.max_future_skew_seconds < 0:
            raise ValueError("max_future_skew_seconds cannot be negative")
        if self.quarantine_retry_limit < 0:
            raise ValueError("quarantine_retry_limit cannot be negative")
        if self.dormant_after_days < 1:
            raise ValueError("dormant_after_days must be positive")
        if self.close_after_days < self.dormant_after_days:
            raise ValueError("close_after_days cannot be lower than dormant_after_days")


@dataclass(frozen=True)
class ArticleAssignment:
    article_id: str
    action: str
    cluster_id: str | None
    reason: str
    similarity: float | None = None


@dataclass(frozen=True)
class IncrementalBatchResult:
    run_id: str | None
    profile_id: str
    dry_run: bool
    fetched: int
    assigned: int
    existing: int
    created: int
    matched: int
    needs_review: int
    quarantined: int
    quarantine_retried: int
    filtered: int
    lifecycle_updated: int
    cursor_fetched_at: int
    cursor_article_id: str
    blocked_article_id: str | None
    blocked_reason: str | None
    assignments: tuple[ArticleAssignment, ...]


def _article_time(article: dict[str, Any]) -> int:
    return int(article.get("published_ts") or article.get("fetched_at") or 0)


def _article_embedding(
    article: dict[str, Any], now_ts: int, max_future_skew: int
) -> tuple[str, int, Sequence[float], EmbeddingSignature]:
    article_id = str(article.get("id") or "").strip()
    if not article_id:
        raise ArticleQualityError("article_id_missing")
    article_ts = _article_time(article)
    if article_ts < 1:
        raise ArticleQualityError("article_timestamp_missing")
    if article_ts > now_ts + max_future_skew:
        raise ArticleQualityError("article_timestamp_in_future")
    vector = article.get("similarity_embedding_vector")
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)) or not vector:
        raise ArticleQualityError("similarity_embedding_missing")
    model = str(article.get("similarity_embedding_model") or "").strip()
    if not model:
        raise ArticleQualityError("similarity_embedding_model_missing")
    instruction = str(article.get("similarity_embedding_instruction") or "").strip()
    if not instruction:
        raise ArticleQualityError("similarity_embedding_instruction_missing")
    signature = EmbeddingSignature(model, len(vector), instruction)
    cosine_similarity(vector, vector)
    return article_id, article_ts, vector, signature


def _membership_document(
    *,
    profile_id: str,
    article_id: str,
    article_ts: int,
    cluster: ThreatCluster,
    action: AssignmentAction,
    reason: str,
    similarity: float | None,
    second_similarity: float | None,
    assigned_at: int,
    strong_indicators: Sequence[str],
    article: dict[str, Any],
) -> dict[str, Any]:
    return {
        "profile_id": profile_id,
        "article_id": article_id,
        "cluster_id": cluster.id,
        "article_ts": article_ts,
        "assigned_at": assigned_at,
        "assignment_action": action.value,
        "assignment_reason": reason,
        "similarity": similarity,
        "second_similarity": second_similarity,
        "cluster_membership_revision": cluster.membership_revision,
        "algorithm_version": cluster.algorithm_version,
        "embedding_model": cluster.embedding_signature.model,
        "embedding_dimension": cluster.embedding_signature.dimensions,
        "embedding_instruction": cluster.embedding_signature.instruction,
        "strong_indicators": list(strong_indicators),
        "evidence": {
            "title": str(article.get("title") or "").strip(),
            "url": str(article.get("url") or "").strip(),
            "source": str(article.get("source") or "").strip(),
            "published_ts": article_ts,
        },
    }


def _matches_required_tags(
    article_tags: Sequence[dict[str, Any]],
    required_tags: frozenset[str],
    match_mode: str,
) -> bool:
    if not required_tags:
        return True
    names = {
        str(tag.get("name") or "").strip().casefold()
        for tag in article_tags
        if str(tag.get("name") or "").strip()
    }
    if match_mode == "any":
        return bool(required_tags.intersection(names))
    return required_tags.issubset(names)


def _strong_indicators(
    article: dict[str, Any], article_tags: Sequence[dict[str, Any]]
) -> tuple[str, ...]:
    text = "\n".join(
        str(article.get(field) or "")
        for field in ("title", "text", "content", "summary")
    )
    indicators = {f"cve:{match.group(0).casefold()}" for match in _CVE_PATTERN.finditer(text)}
    for tag in article_tags:
        category = str(tag.get("category") or "").strip().upper()
        name = str(tag.get("name") or "").strip().casefold()
        if not name or category not in _STRONG_TAG_CATEGORIES:
            continue
        cves = _CVE_PATTERN.findall(name)
        if cves:
            indicators.update(f"cve:{value.casefold()}" for value in cves)
        else:
            indicators.add(f"{category.casefold()}:{name}")
    return tuple(sorted(indicators))


def run_incremental_clustering(
    store: LongTermStore,
    *,
    profile_id: str,
    owner_id: str,
    sources: list[str] | None = None,
    required_tags: list[str] | None = None,
    required_tags_match: str = "all",
    since_fetched_at: int | None = None,
    until_fetched_at: int | None = None,
    now_ts: int | None = None,
    settings: IncrementalSettings | None = None,
    dry_run: bool = False,
) -> IncrementalBatchResult:
    """Process one bounded batch and advance its cursor only after durable writes."""

    profile_id = str(profile_id or "").strip()
    owner_id = str(owner_id or "").strip()
    if not profile_id or not owner_id:
        raise ValueError("profile_id and owner_id are required")
    required_tags_match = str(required_tags_match or "all").strip().lower()
    if required_tags_match not in {"all", "any"}:
        raise ValueError("required_tags_match must be 'all' or 'any'")
    normalized_required_tags = frozenset(
        str(tag).strip().casefold() for tag in required_tags or [] if str(tag).strip()
    )
    settings = settings or IncrementalSettings()
    now_ts = int(now_ts or time.time())
    until_fetched_at = int(until_fetched_at or now_ts)
    cursor = store.get_long_term_cursor(profile_id)
    stored_cursor_pair = (
        int(cursor.get("cursor_fetched_at") or 0),
        str(cursor.get("cursor_article_id") or ""),
    )
    scan_cursor_pair = stored_cursor_pair
    if since_fetched_at is not None:
        scan_cursor_pair = max(scan_cursor_pair, (int(since_fetched_at), ""))
    lease_claimed = False
    run_id = None if dry_run else f"long_term_run_{uuid.uuid4().hex}"
    if not dry_run:
        lease_claimed = store.claim_long_term_lease(
            profile_id,
            owner_id,
            now_ts=now_ts,
            lease_seconds=settings.lease_seconds,
        )
        if not lease_claimed:
            raise LeaseUnavailableError(f"profile lease is already held: {profile_id}")
        if not store.create_long_term_run(
            {
                "id": run_id,
                "profile_id": profile_id,
                "run_type": "incremental_clustering",
                "status": "running",
                "started_at": now_ts,
                "until_fetched_at": until_fetched_at,
                "cursor_before": list(stored_cursor_pair),
                "scan_after": list(scan_cursor_pair),
            }
        ):
            store.release_long_term_lease(profile_id, owner_id)
            raise ConcurrentAssignmentError("could not create long-term run record")

    assignments: list[ArticleAssignment] = []
    counts = {
        "assigned": 0,
        "existing": 0,
        "created": 0,
        "matched": 0,
        "needs_review": 0,
        "quarantined": 0,
        "quarantine_retried": 0,
        "filtered": 0,
        "lifecycle_updated": 0,
    }
    blocked_article_id: str | None = None
    blocked_reason: str | None = None
    rows: list[dict[str, Any]] = []
    final_cursor = stored_cursor_pair
    cluster_cache: dict[str, list[ThreatCluster]] = {}
    try:
        new_rows = store.list_articles_for_long_term(
            after_fetched_at=scan_cursor_pair[0],
            after_article_id=scan_cursor_pair[1],
            until_fetched_at=until_fetched_at,
            sources=sources,
            limit=settings.batch_size,
        )
        quarantine_rows: list[dict[str, Any]] = []
        if settings.quarantine_retry_limit:
            open_quarantine = store.list_long_term_quarantine(
                profile_id,
                status="open",
                limit=settings.quarantine_retry_limit,
            )
            retry_ids = [
                str(row.get("article_id") or "")
                for row in open_quarantine
                if str(row.get("article_id") or "")
            ]
            quarantine_rows = store.get_articles_by_ids(retry_ids) if retry_ids else []
            counts["quarantine_retried"] = len(quarantine_rows)
        rows_by_id = {
            str(row.get("id") or ""): row
            for row in (*quarantine_rows, *new_rows)
            if str(row.get("id") or "")
        }
        rows = list(rows_by_id.values())
        fetched_cursor = max(
            (
                (int(row.get("fetched_at") or 0), str(row.get("id") or ""))
                for row in new_rows
            ),
            default=scan_cursor_pair,
        )
        ordered = sorted(rows, key=lambda row: (_article_time(row), str(row.get("id") or "")))
        for article in ordered:
            article_id = str(article.get("id") or "").strip()
            article_tags = store.get_article_tags(article_id) if article_id else []
            if normalized_required_tags and not _matches_required_tags(
                article_tags,
                normalized_required_tags,
                required_tags_match,
            ):
                counts["filtered"] += 1
                if not dry_run:
                    store.resolve_long_term_quarantine(
                        profile_id, article_id, resolved_at=now_ts
                    )
                assignments.append(
                    ArticleAssignment(
                        article_id,
                        "filtered",
                        None,
                        "required_tags_not_matched",
                    )
                )
                continue
            indicators = _strong_indicators(article, article_tags)
            try:
                article_id, article_ts, vector, signature = _article_embedding(
                    article, now_ts, settings.max_future_skew_seconds
                )
            except (ArticleQualityError, VectorValidationError, ValueError) as exc:
                reason = str(exc)
                if not article_id:
                    blocked_reason = reason
                    break
                if not dry_run:
                    store.save_long_term_quarantine(
                        {
                            "profile_id": profile_id,
                            "article_id": article_id,
                            "reason": reason,
                            "observed_at": now_ts,
                            "fetched_at": int(article.get("fetched_at") or 0),
                            "article_ts": _article_time(article),
                            "embedding_model": str(
                                article.get("similarity_embedding_model") or ""
                            ),
                            "embedding_instruction": str(
                                article.get("similarity_embedding_instruction") or ""
                            ),
                        }
                    )
                counts["quarantined"] += 1
                assignments.append(
                    ArticleAssignment(article_id, "quarantined", None, reason)
                )
                continue

            existing = store.get_cluster_membership(profile_id, article_id)
            if existing is not None:
                if not dry_run:
                    store.resolve_long_term_quarantine(
                        profile_id, article_id, resolved_at=now_ts
                    )
                counts["existing"] += 1
                assignments.append(
                    ArticleAssignment(
                        article_id,
                        "existing",
                        str(existing.get("cluster_id") or "") or None,
                        "membership_already_exists",
                    )
                )
                continue

            candidates = cluster_cache.get(signature.key)
            if candidates is None:
                candidate_docs = store.list_threat_clusters(
                    profile_id,
                    statuses=[status.value for status in ACTIVE_CLUSTER_STATUSES],
                    embedding_model=signature.model,
                    embedding_dimension=signature.dimensions,
                    embedding_instruction=signature.instruction,
                )
                candidates = [ThreatCluster.from_document(doc) for doc in candidate_docs]
                cluster_cache[signature.key] = candidates

            decision = assign_article(
                profile_id=profile_id,
                article_ts=article_ts,
                embedding=vector,
                signature=signature,
                candidates=candidates,
                strong_indicators=indicators,
                settings=settings.clustering,
            )
            previous_revision: int | None = None
            if decision.action is AssignmentAction.MATCH:
                index = next(
                    i for i, candidate in enumerate(candidates) if candidate.id == decision.cluster_id
                )
                previous = candidates[index]
                previous_revision = previous.membership_revision
                cluster = add_cluster_member(
                    previous,
                    article_ts=article_ts,
                    embedding=vector,
                    strong_indicators=indicators,
                )
                candidates[index] = cluster
                counts["matched"] += 1
            else:
                cluster = create_cluster(
                    profile_id=profile_id,
                    article_id=article_id,
                    article_ts=article_ts,
                    embedding=vector,
                    signature=signature,
                    strong_indicators=indicators,
                )
                if decision.action is AssignmentAction.NEEDS_REVIEW:
                    cluster = replace(cluster, status=ClusterStatus.NEEDS_REVIEW)
                    counts["needs_review"] += 1
                else:
                    counts["created"] += 1
                candidates.append(cluster)

            membership = _membership_document(
                profile_id=profile_id,
                article_id=article_id,
                article_ts=article_ts,
                cluster=cluster,
                action=decision.action,
                reason=decision.reason,
                similarity=decision.similarity,
                second_similarity=decision.second_similarity,
                assigned_at=now_ts,
                strong_indicators=indicators,
                article=article,
            )
            if not dry_run and not store.save_cluster_assignment(
                cluster.to_document(),
                membership,
                expected_membership_revision=previous_revision,
            ):
                if store.get_cluster_membership(profile_id, article_id) is None:
                    raise ConcurrentAssignmentError(
                        f"cluster revision changed while assigning article {article_id}"
                    )
                counts["existing"] += 1
                assignments.append(
                    ArticleAssignment(
                        article_id,
                        "existing",
                        cluster.id,
                        "membership_won_by_concurrent_retry",
                    )
                )
                continue
            counts["assigned"] += 1
            if not dry_run:
                store.resolve_long_term_quarantine(
                    profile_id, article_id, resolved_at=now_ts
                )
            assignments.append(
                ArticleAssignment(
                    article_id,
                    decision.action.value,
                    cluster.id,
                    decision.reason,
                    decision.similarity,
                )
            )

        lifecycle_reference_ts = max(
            (_article_time(row) for row in rows),
            default=now_ts,
        )
        lifecycle_docs = store.list_threat_clusters(
            profile_id,
            statuses=[ClusterStatus.ACTIVE.value, ClusterStatus.DORMANT.value],
        )
        for document in lifecycle_docs:
            current = ThreatCluster.from_document(document)
            status = cluster_status_at(
                current,
                now_ts=lifecycle_reference_ts,
                dormant_after_days=settings.dormant_after_days,
                close_after_days=settings.close_after_days,
            )
            if status == current.status:
                continue
            updated = replace(current, status=status)
            if not dry_run and not store.save_threat_cluster(
                updated.to_document(),
                expected_membership_revision=current.membership_revision,
            ):
                raise ConcurrentAssignmentError(
                    f"cluster revision changed during lifecycle update: {current.id}"
                )
            counts["lifecycle_updated"] += 1

        if (
            new_rows
            and blocked_reason is None
            and not dry_run
            and fetched_cursor > stored_cursor_pair
        ):
            if not store.advance_long_term_cursor(
                profile_id,
                owner_id,
                expected_fetched_at=stored_cursor_pair[0],
                expected_article_id=stored_cursor_pair[1],
                fetched_at=fetched_cursor[0],
                article_id=fetched_cursor[1],
                now_ts=now_ts,
            ):
                raise ConcurrentAssignmentError("long-term cursor changed during batch")
            final_cursor = fetched_cursor

        if not dry_run:
            store.update_long_term_run(
                run_id,
                expected_status="running",
                fields={
                    "status": "blocked" if blocked_reason else "done",
                    "finished_at": now_ts,
                    "fetched": len(rows),
                    **counts,
                    "blocked_article_id": blocked_article_id,
                    "blocked_reason": blocked_reason,
                    "cursor_after": list(final_cursor),
                },
            )
    except Exception:
        if not dry_run and run_id is not None:
            store.update_long_term_run(
                run_id,
                expected_status="running",
                fields={"status": "failed", "finished_at": now_ts},
            )
        raise
    finally:
        if lease_claimed:
            store.release_long_term_lease(profile_id, owner_id)

    return IncrementalBatchResult(
        run_id=run_id,
        profile_id=profile_id,
        dry_run=dry_run,
        fetched=len(rows),
        cursor_fetched_at=final_cursor[0],
        cursor_article_id=final_cursor[1],
        blocked_article_id=blocked_article_id,
        blocked_reason=blocked_reason,
        assignments=tuple(assignments),
        **counts,
    )
