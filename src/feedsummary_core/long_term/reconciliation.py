# LICENSE HEADER MANAGED BY add-license-header
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Martin Vesterlund

"""Read-only proposal generation for reconciling existing event clusters."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from math import sqrt
from typing import Any

from feedsummary_core.long_term.identity import membership_is_strict_cve_record
from feedsummary_core.long_term.models import ClusterStatus, ThreatCluster

_TITLE_TOKEN = re.compile(r"[a-z0-9][a-z0-9.+_-]*", re.IGNORECASE)
_TITLE_STOP_WORDS = frozenset(
    {
        "about", "affecting", "all", "and", "attack", "attacks", "critical",
        "cyber", "exploited", "exploitation", "flaw", "from", "latest", "new",
        "security", "the", "this", "under", "vulnerability", "vulnerabilities",
        "warns", "with", "zero-day", "zeroday",
    }
)
_IDENTITY_PREFIXES = ("organization:", "product:")
_SUMMARY_TITLE_PATTERN = re.compile(
    r"\b(?:bulletin|catalog|multiple|roundup|newsletter|security updates|weekly)\b"
    r"|\b(?:two|three|four|five|six|seven|eight|nine|\d+)\b"
    r"(?:\W+\w+){0,8}\W+\b(?:flaws|vulnerabilities)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ReconciliationSettings:
    """Thresholds for conservative, read-only merge proposals."""

    auto_merge_similarity: float = 0.88
    review_similarity: float = 0.82
    candidate_window_days: int = 60
    max_review_pairs: int = 500

    def __post_init__(self) -> None:
        if not 0 <= self.review_similarity <= self.auto_merge_similarity <= 1:
            raise ValueError("reconciliation similarities must satisfy 0 <= review <= auto <= 1")
        if self.candidate_window_days < 1:
            raise ValueError("candidate_window_days must be positive")
        if self.max_review_pairs < 0:
            raise ValueError("max_review_pairs cannot be negative")


@dataclass(frozen=True)
class ReconciliationEdge:
    left_cluster_id: str
    right_cluster_id: str
    disposition: str
    reason: str
    similarity: float
    shared_cves: tuple[str, ...]
    shared_identities: tuple[str, ...]
    shared_title_terms: tuple[str, ...]
    left_strict_cve: bool
    right_strict_cve: bool
    left_summary_like: bool
    right_summary_like: bool
    interval_gap_days: float
    left_title: str
    right_title: str


@dataclass(frozen=True)
class ReconciliationGroup:
    primary_cluster_id: str
    cluster_ids: tuple[str, ...]
    titles: tuple[str, ...]
    edge_count: int


@dataclass(frozen=True)
class ReconciliationResult:
    cluster_count: int
    compared_pair_count: int
    auto_merge_edges: tuple[ReconciliationEdge, ...]
    review_edges: tuple[ReconciliationEdge, ...]
    blocked_strict_cve_pair_count: int
    groups: tuple[ReconciliationGroup, ...]

    def to_document(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ClusterView:
    cluster: ThreatCluster
    normalized_centroid: tuple[float, ...]
    strict_cve: bool
    summary_like: bool
    cves: frozenset[str]
    identities: frozenset[str]
    title_terms: frozenset[str]
    title: str


def _title_terms(title: str) -> frozenset[str]:
    return frozenset(
        token
        for token in (match.group(0).casefold() for match in _TITLE_TOKEN.finditer(title))
        if len(token) >= 4
        and token not in _TITLE_STOP_WORDS
        and not token.startswith("cve-")
        and not token.isdigit()
    )


def _cluster_view(
    document: dict[str, Any], memberships: Iterable[dict[str, Any]]
) -> _ClusterView:
    cluster = ThreatCluster.from_document(document)
    rows = tuple(memberships)
    indicators = {
        str(value).strip().casefold()
        for row in rows
        for value in row.get("strong_indicators") or ()
        if str(value).strip()
    } | set(cluster.strong_indicators)
    titles = [
        str((row.get("evidence") or {}).get("title") or "").strip()
        for row in rows
    ]
    titles = [value for value in titles if value]
    title = titles[0] if titles else cluster.id
    norm = sqrt(sum(value * value for value in cluster.centroid))
    if norm == 0:
        raise ValueError(f"cluster {cluster.id} has a zero centroid")
    return _ClusterView(
        cluster=cluster,
        normalized_centroid=tuple(value / norm for value in cluster.centroid),
        strict_cve=bool(rows) and all(
            membership_is_strict_cve_record(row) for row in rows
        ),
        # Use the representative evidence title. A multi-member incident can
        # legitimately contain one headline about several chained flaws, while a
        # cluster represented by a weekly roundup must remain review-only.
        summary_like=bool(_SUMMARY_TITLE_PATTERN.search(title)),
        cves=frozenset(value for value in indicators if value.startswith("cve:")),
        identities=frozenset(
            value for value in indicators if value.startswith(_IDENTITY_PREFIXES)
        ),
        title_terms=frozenset().union(*(_title_terms(value) for value in titles)),
        title=title,
    )


def _interval_gap_seconds(left: ThreatCluster, right: ThreatCluster) -> int:
    if left.last_seen_ts < right.first_seen_ts:
        return right.first_seen_ts - left.last_seen_ts
    if right.last_seen_ts < left.first_seen_ts:
        return left.first_seen_ts - right.last_seen_ts
    return 0


def _edge(left: _ClusterView, right: _ClusterView, similarity: float) -> ReconciliationEdge:
    shared_cves = tuple(sorted(left.cves & right.cves))
    shared_identities = tuple(sorted(left.identities & right.identities))
    shared_title_terms = tuple(sorted(left.title_terms & right.title_terms))
    disjoint_strict_cves = (
        left.strict_cve
        and right.strict_cve
        and left.cves
        and right.cves
        and left.cves.isdisjoint(right.cves)
    )
    if disjoint_strict_cves:
        disposition = "blocked"
        reason = "distinct_strict_cve_records"
    elif similarity < 0:  # pragma: no cover - caller always supplies a real score
        disposition = "blocked"
        reason = "invalid_similarity"
    elif left.strict_cve != right.strict_cve:
        disposition = "review"
        reason = "narrative_to_strict_cve_requires_review"
    elif left.summary_like or right.summary_like:
        disposition = "review"
        reason = "summary_or_recurring_bulletin_requires_review"
    elif not (shared_cves or (shared_identities and shared_title_terms)):
        disposition = "review"
        reason = "semantic_similarity_without_strong_identity_support"
    else:
        disposition = "auto_merge"
        reason = "semantic_similarity_with_event_identity_support"
    return ReconciliationEdge(
        left_cluster_id=left.cluster.id,
        right_cluster_id=right.cluster.id,
        disposition=disposition,
        reason=reason,
        similarity=round(similarity, 6),
        shared_cves=shared_cves,
        shared_identities=shared_identities,
        shared_title_terms=shared_title_terms,
        left_strict_cve=left.strict_cve,
        right_strict_cve=right.strict_cve,
        left_summary_like=left.summary_like,
        right_summary_like=right.summary_like,
        interval_gap_days=round(
            _interval_gap_seconds(left.cluster, right.cluster) / 86400, 3
        ),
        left_title=left.title,
        right_title=right.title,
    )


def propose_cluster_reconciliation(
    clusters: Iterable[dict[str, Any]],
    memberships_by_cluster: dict[str, Iterable[dict[str, Any]]],
    *,
    settings: ReconciliationSettings | None = None,
) -> ReconciliationResult:
    """Find conservative merge groups without changing persistence state."""

    settings = settings or ReconciliationSettings()
    views = [
        _cluster_view(row, memberships_by_cluster.get(str(row.get("id") or ""), ()))
        for row in clusters
        if not str(row.get("superseded_by_cluster_id") or "").strip()
    ]
    compatible: dict[str, list[_ClusterView]] = defaultdict(list)
    for view in views:
        compatible[view.cluster.embedding_signature.key].append(view)

    auto_edges: list[ReconciliationEdge] = []
    review_edges: list[ReconciliationEdge] = []
    blocked_count = 0
    compared = 0
    max_gap = settings.candidate_window_days * 86400
    for group in compatible.values():
        group.sort(key=lambda view: (view.cluster.first_seen_ts, view.cluster.id))
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                if right.cluster.first_seen_ts - left.cluster.last_seen_ts > max_gap:
                    break
                if _interval_gap_seconds(left.cluster, right.cluster) > max_gap:
                    continue
                # Title overlap is sufficient to produce a review proposal, but
                # _edge requires a stronger persisted identity for auto-merge.
                if not (
                    left.cves & right.cves
                    or left.identities & right.identities
                    or left.title_terms & right.title_terms
                ):
                    continue
                compared += 1
                similarity = sum(
                    a * b
                    for a, b in zip(left.normalized_centroid, right.normalized_centroid)
                )
                if similarity < settings.review_similarity:
                    continue
                edge = _edge(left, right, similarity)
                if edge.disposition == "blocked":
                    blocked_count += 1
                elif (
                    edge.disposition == "auto_merge"
                    and similarity >= settings.auto_merge_similarity
                ):
                    auto_edges.append(edge)
                else:
                    review_edges.append(
                        ReconciliationEdge(
                            **{**asdict(edge), "disposition": "review"}
                        )
                    )

    auto_edges.sort(
        key=lambda edge: (-edge.similarity, edge.left_cluster_id, edge.right_cluster_id)
    )
    review_edges.sort(
        key=lambda edge: (-edge.similarity, edge.left_cluster_id, edge.right_cluster_id)
    )
    review_edges = review_edges[: settings.max_review_pairs]

    parent = {view.cluster.id: view.cluster.id for view in views}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for edge in auto_edges:
        union(edge.left_cluster_id, edge.right_cluster_id)
    components: dict[str, set[str]] = defaultdict(set)
    for edge in auto_edges:
        components[find(edge.left_cluster_id)].update(
            (edge.left_cluster_id, edge.right_cluster_id)
        )
    view_by_id = {view.cluster.id: view for view in views}
    groups = []
    for ids in components.values():
        ordered = sorted(ids)
        primary = min(
            ordered,
            key=lambda cluster_id: (
                -view_by_id[cluster_id].cluster.member_count,
                view_by_id[cluster_id].cluster.first_seen_ts,
                cluster_id,
            ),
        )
        edge_count = sum(
            edge.left_cluster_id in ids and edge.right_cluster_id in ids
            for edge in auto_edges
        )
        groups.append(
            ReconciliationGroup(
                primary_cluster_id=primary,
                cluster_ids=tuple(ordered),
                titles=tuple(view_by_id[value].title for value in ordered),
                edge_count=edge_count,
            )
        )
    groups.sort(key=lambda group: (-len(group.cluster_ids), group.primary_cluster_id))
    return ReconciliationResult(
        cluster_count=len(views),
        compared_pair_count=compared,
        auto_merge_edges=tuple(auto_edges),
        review_edges=tuple(review_edges),
        blocked_strict_cve_pair_count=blocked_count,
        groups=tuple(groups),
    )


def build_cluster_merge_operation(
    clusters: Iterable[dict[str, Any]],
    memberships_by_cluster: dict[str, Iterable[dict[str, Any]]],
    *,
    primary_cluster_id: str,
    reconciliation_id: str,
    reconciled_at: int,
) -> dict[str, Any]:
    """Build the complete deterministic state transition for an approved merge."""

    if not reconciliation_id.strip() or reconciled_at < 1:
        raise ValueError("reconciliation id and timestamp are required")
    documents = {str(row.get("id") or ""): dict(row) for row in clusters}
    if len(documents) < 2 or primary_cluster_id not in documents:
        raise ValueError("a merge needs a primary and at least one secondary cluster")
    membership_rows = {
        cluster_id: tuple(memberships_by_cluster.get(cluster_id, ()))
        for cluster_id in documents
    }
    views = {
        cluster_id: _cluster_view(document, membership_rows[cluster_id])
        for cluster_id, document in documents.items()
    }
    signatures = {view.cluster.embedding_signature for view in views.values()}
    profiles = {view.cluster.profile_id for view in views.values()}
    if len(signatures) != 1 or len(profiles) != 1:
        raise ValueError("merged clusters must have one profile and embedding signature")
    if any(view.cluster.superseded_by_cluster_id for view in views.values()):
        raise ValueError("a superseded cluster cannot be merged again")
    for cluster_id, view in views.items():
        membership_count = len(membership_rows[cluster_id])
        if membership_count != view.cluster.member_count:
            raise ValueError(
                f"cluster {cluster_id} member count does not match persisted memberships"
            )

    ordered_ids = tuple(sorted(documents))
    source_clusters = [views[cluster_id].cluster for cluster_id in ordered_ids]
    ordered_memberships = sorted(
        (
            dict(row)
            for cluster_id in ordered_ids
            for row in membership_rows[cluster_id]
        ),
        key=lambda row: (
            int(
                row.get("article_ts")
                or (row.get("evidence") or {}).get("published_ts")
                or row.get("assigned_at")
                or 0
            ),
            int(row.get("assigned_at") or 0),
            str(row.get("article_id") or ""),
        ),
    )
    article_ids = [str(row.get("article_id") or "") for row in ordered_memberships]
    if not all(article_ids) or len(set(article_ids)) != len(article_ids):
        raise ValueError("merged memberships must have unique article ids")
    dimensions = source_clusters[0].embedding_signature.dimensions
    vector_sum = tuple(
        sum(cluster.vector_sum[index] for cluster in source_clusters)
        for index in range(dimensions)
    )
    member_count = sum(cluster.member_count for cluster in source_clusters)
    base_revision = max(cluster.membership_revision for cluster in source_clusters)
    membership_revision_by_article_id = {
        article_id: base_revision + index
        for index, article_id in enumerate(article_ids, start=1)
    }
    membership_revision = base_revision + member_count
    primary = views[primary_cluster_id].cluster
    merged = replace(
        primary,
        vector_sum=vector_sum,
        centroid=tuple(value / member_count for value in vector_sum),
        member_count=member_count,
        membership_revision=membership_revision,
        first_seen_ts=min(cluster.first_seen_ts for cluster in source_clusters),
        last_seen_ts=max(cluster.last_seen_ts for cluster in source_clusters),
        status=ClusterStatus.ACTIVE,
        strong_indicators=tuple(
            sorted(
                {
                    indicator
                    for cluster in source_clusters
                    for indicator in cluster.strong_indicators
                }
            )
        ),
        strict_cve_identity=all(view.strict_cve for view in views.values()),
        summarized_revision=base_revision,
        latest_snapshot_id=None,
        last_summarized_at=None,
        superseded_by_cluster_id=None,
        reconciliation_id=reconciliation_id,
        reconciled_at=reconciled_at,
    ).to_document()
    merged.update(
        {
            "updated_at": reconciled_at,
            "reconciliation_count": int(
                documents[primary_cluster_id].get("reconciliation_count") or 0
            )
            + 1,
            "reconciled_cluster_ids": list(ordered_ids),
        }
    )

    superseded = []
    for cluster_id in ordered_ids:
        if cluster_id == primary_cluster_id:
            continue
        cluster = views[cluster_id].cluster
        tombstone = replace(
            cluster,
            status=ClusterStatus.CLOSED,
            centroid=cluster.centroid,
            vector_sum=tuple(0.0 for _ in cluster.vector_sum),
            member_count=0,
            membership_revision=cluster.membership_revision + 1,
            superseded_by_cluster_id=primary_cluster_id,
            reconciliation_id=reconciliation_id,
            reconciled_at=reconciled_at,
        ).to_document()
        tombstone["updated_at"] = reconciled_at
        superseded.append(tombstone)

    return {
        "id": reconciliation_id,
        "profile_id": next(iter(profiles)),
        "primary_cluster_id": primary_cluster_id,
        "source_cluster_ids": list(ordered_ids),
        "expected_membership_revisions": {
            cluster_id: views[cluster_id].cluster.membership_revision
            for cluster_id in ordered_ids
        },
        "membership_revision_by_article_id": membership_revision_by_article_id,
        "merged_cluster": merged,
        "superseded_clusters": superseded,
        "reconciled_at": reconciled_at,
        "status": "approved",
        "snapshot_policy": "preserve_history_rebuild_primary",
    }


def validate_cluster_merge_operation(operation: dict[str, Any]) -> dict[str, Any]:
    """Validate an apply document before a backend begins a state transition."""

    document = dict(operation or {})
    required = (
        "id",
        "profile_id",
        "primary_cluster_id",
        "source_cluster_ids",
        "expected_membership_revisions",
        "membership_revision_by_article_id",
        "merged_cluster",
        "superseded_clusters",
        "reconciled_at",
        "snapshot_policy",
    )
    if any(document.get(field) is None for field in required):
        raise ValueError("cluster reconciliation operation is incomplete")
    source_ids = tuple(str(value) for value in document["source_cluster_ids"])
    if len(source_ids) < 2 or len(set(source_ids)) != len(source_ids):
        raise ValueError("cluster reconciliation needs distinct source clusters")
    primary_id = str(document["primary_cluster_id"])
    profile_id = str(document["profile_id"])
    if primary_id not in source_ids or int(document["reconciled_at"]) < 1:
        raise ValueError("cluster reconciliation primary or timestamp is invalid")
    expected = {
        str(key): int(value)
        for key, value in dict(document["expected_membership_revisions"]).items()
    }
    if set(expected) != set(source_ids):
        raise ValueError("expected revisions must cover every source cluster")
    membership_revisions = {
        str(key): int(value)
        for key, value in dict(
            document["membership_revision_by_article_id"]
        ).items()
    }
    merged = dict(document["merged_cluster"])
    if (
        str(merged.get("id") or "") != primary_id
        or str(merged.get("profile_id") or "") != profile_id
        or int(merged.get("member_count") or 0) < 2
        or int(merged.get("membership_revision") or 0) <= expected[primary_id]
        or merged.get("latest_snapshot_id") is not None
    ):
        raise ValueError("merged cluster does not satisfy reconciliation invariants")
    base_revision = int(merged.get("summarized_revision") or 0)
    expected_new_revisions = set(
        range(base_revision + 1, int(merged["membership_revision"]) + 1)
    )
    if (
        len(membership_revisions) != int(merged["member_count"])
        or set(membership_revisions.values()) != expected_new_revisions
    ):
        raise ValueError("membership reconciliation revisions are not contiguous")
    tombstones = {
        str(row.get("id") or ""): dict(row)
        for row in document["superseded_clusters"]
    }
    if set(tombstones) != set(source_ids) - {primary_id}:
        raise ValueError("superseded cluster set does not match source clusters")
    for cluster_id, row in tombstones.items():
        if (
            str(row.get("profile_id") or "") != profile_id
            or str(row.get("status") or "") != ClusterStatus.CLOSED.value
            or int(row.get("member_count", -1)) != 0
            or str(row.get("superseded_by_cluster_id") or "") != primary_id
            or str(row.get("reconciliation_id") or "") != str(document["id"])
            or int(row.get("membership_revision") or 0) <= expected[cluster_id]
        ):
            raise ValueError("superseded cluster does not satisfy lineage invariants")
    if document["snapshot_policy"] != "preserve_history_rebuild_primary":
        raise ValueError("unsupported reconciliation snapshot policy")
    document.update(
        {
            "id": str(document["id"]),
            "profile_id": profile_id,
            "primary_cluster_id": primary_id,
            "source_cluster_ids": list(source_ids),
            "expected_membership_revisions": expected,
            "membership_revision_by_article_id": membership_revisions,
            "merged_cluster": merged,
            "superseded_clusters": list(tombstones.values()),
            "reconciled_at": int(document["reconciled_at"]),
        }
    )
    return document
