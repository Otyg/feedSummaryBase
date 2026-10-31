# LICENSE HEADER MANAGED BY add-license-header
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Martin Vesterlund

"""Audited state transitions for manually correcting cluster memberships."""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from feedsummary_core.long_term.clustering import stable_cluster_id
from feedsummary_core.long_term.identity import membership_is_strict_cve_record
from feedsummary_core.long_term.models import ClusterStatus, ThreatCluster


def _article_ts(row: dict[str, Any]) -> int:
    return int(
        row.get("article_ts")
        or (row.get("evidence") or {}).get("published_ts")
        or row.get("assigned_at")
        or 0
    )


def _ordered(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            _article_ts(row),
            int(row.get("assigned_at") or 0),
            str(row.get("article_id") or ""),
        ),
    )


def _unit_article_vector(article: dict[str, Any], cluster: ThreatCluster) -> tuple[float, ...]:
    vector = article.get("similarity_embedding_vector")
    if (
        not isinstance(vector, (list, tuple))
        or len(vector) != cluster.embedding_signature.dimensions
    ):
        raise ValueError("article has no compatible similarity embedding")
    if str(article.get("similarity_embedding_model") or "") != cluster.embedding_signature.model:
        raise ValueError("article embedding model does not match cluster")
    if (
        str(article.get("similarity_embedding_instruction") or "")
        != cluster.embedding_signature.instruction
    ):
        raise ValueError("article embedding instruction does not match cluster")
    values = tuple(float(value) for value in vector)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("article embedding contains invalid values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        raise ValueError("article embedding has zero norm")
    return tuple(value / norm for value in values)


def _indicators(rows: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(value).strip().casefold()
                for row in rows
                for value in row.get("strong_indicators") or ()
                if str(value).strip()
            }
        )
    )


def _operation_id(
    profile_id: str,
    article_id: str,
    source_id: str,
    target_id: str,
    source_revision: int,
    target_revision: int | None,
) -> str:
    identity = ":".join(
        (
            "feedsummary:cluster-membership-edit",
            profile_id,
            article_id,
            source_id,
            target_id,
            str(source_revision),
            str(target_revision),
        )
    )
    return f"membership_edit_{uuid.uuid5(uuid.NAMESPACE_URL, identity).hex}"


def build_cluster_membership_edit(
    source_cluster: dict[str, Any],
    source_memberships: Iterable[dict[str, Any]],
    *,
    article: dict[str, Any],
    article_id: str,
    edited_at: int,
    edited_by: str,
    comment: str = "",
    target_cluster: dict[str, Any] | None = None,
    target_memberships: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Build a move to an existing cluster or a split into a new singleton."""

    reviewer = " ".join(str(edited_by).split())
    article_id = str(article_id).strip()
    if not article_id or edited_at < 1 or not reviewer:
        raise ValueError("article, edit timestamp and editor are required")
    source = ThreatCluster.from_document(source_cluster)
    if source.superseded_by_cluster_id or source.status is ClusterStatus.CLOSED:
        raise ValueError("members cannot be edited on a closed or superseded cluster")
    source_rows = _ordered(source_memberships)
    if len(source_rows) != source.member_count:
        raise ValueError("source member count does not match persisted memberships")
    moved = next(
        (row for row in source_rows if str(row.get("article_id") or "") == article_id), None
    )
    if moved is None:
        raise ValueError("article is not a member of the source cluster")
    vector = _unit_article_vector(article, source)

    target: ThreatCluster | None = None
    target_rows = _ordered(target_memberships)
    if target_cluster is not None:
        target = ThreatCluster.from_document(target_cluster)
        if target.id == source.id:
            raise ValueError("source and target cluster must differ")
        if (
            target.profile_id != source.profile_id
            or target.embedding_signature != source.embedding_signature
            or target.tag_scope != source.tag_scope
        ):
            raise ValueError("source and target must share profile, embedding and tag scope")
        if target.superseded_by_cluster_id or target.status not in {
            ClusterStatus.ACTIVE,
            ClusterStatus.DORMANT,
        }:
            raise ValueError("target cluster must be active or dormant")
        if len(target_rows) != target.member_count:
            raise ValueError("target member count does not match persisted memberships")
        if any(str(row.get("article_id") or "") == article_id for row in target_rows):
            raise ValueError("article is already a member of the target cluster")
        target_id = target.id
        action = "move"
    else:
        if source.member_count <= 1:
            raise ValueError("a singleton article is already separate")
        target_id = stable_cluster_id(
            source.profile_id,
            article_id,
            f"manual-split-v1:{source.id}:{source.membership_revision}",
        )
        action = "split"

    operation_id = _operation_id(
        source.profile_id,
        article_id,
        source.id,
        target_id,
        source.membership_revision,
        target.membership_revision if target else None,
    )
    remaining = [row for row in source_rows if str(row.get("article_id") or "") != article_id]

    def revised_memberships(
        rows: list[dict[str, Any]], cluster_id: str, base: int
    ) -> list[dict[str, Any]]:
        result = []
        for index, row in enumerate(_ordered(rows), start=1):
            previous_id = str(row.get("cluster_id") or "")
            lineage = [str(value) for value in row.get("cluster_lineage") or ()]
            if previous_id != cluster_id and previous_id and previous_id not in lineage:
                lineage.append(previous_id)
            row.update(
                {
                    "cluster_id": cluster_id,
                    "cluster_lineage": lineage,
                    "cluster_membership_revision": base + index,
                    "membership_edit_id": operation_id,
                    "membership_edit_action": action,
                    "manually_edited_at": edited_at,
                    "manually_edited_by": reviewer,
                }
            )
            if previous_id != cluster_id:
                row["previous_cluster_id"] = previous_id
            result.append(row)
        return result

    if remaining:
        source_sum = tuple(total - value for total, value in zip(source.vector_sum, vector))
        # Tiny floating point residuals are harmless but make zero-ish values stable.
        source_sum = tuple(0.0 if abs(value) < 1e-12 else value for value in source_sum)
        source_final = replace(
            source,
            vector_sum=source_sum,
            centroid=tuple(value / len(remaining) for value in source_sum),
            member_count=len(remaining),
            membership_revision=source.membership_revision + len(remaining),
            first_seen_ts=min(_article_ts(row) for row in remaining),
            last_seen_ts=max(_article_ts(row) for row in remaining),
            strong_indicators=_indicators(remaining),
            strict_cve_identity=all(membership_is_strict_cve_record(row) for row in remaining),
            summarized_revision=source.membership_revision,
            latest_snapshot_id=None,
            last_summarized_at=None,
        ).to_document()
        final_source_rows = revised_memberships(remaining, source.id, source.membership_revision)
    else:
        source_final = replace(
            source,
            status=ClusterStatus.CLOSED,
            vector_sum=tuple(0.0 for _ in source.vector_sum),
            member_count=0,
            membership_revision=source.membership_revision + 1,
            superseded_by_cluster_id=target_id,
            reconciliation_id=operation_id,
            reconciled_at=edited_at,
        ).to_document()
        final_source_rows = []

    if target is not None:
        combined = target_rows + [moved]
        target_sum = tuple(total + value for total, value in zip(target.vector_sum, vector))
        target_final = replace(
            target,
            vector_sum=target_sum,
            centroid=tuple(value / len(combined) for value in target_sum),
            member_count=len(combined),
            membership_revision=target.membership_revision + len(combined),
            first_seen_ts=min(_article_ts(row) for row in combined),
            last_seen_ts=max(_article_ts(row) for row in combined),
            status=ClusterStatus.ACTIVE,
            strong_indicators=_indicators(combined),
            strict_cve_identity=all(membership_is_strict_cve_record(row) for row in combined),
            summarized_revision=target.membership_revision,
            latest_snapshot_id=None,
            last_summarized_at=None,
        ).to_document()
        target_base = target.membership_revision
    else:
        target_final = ThreatCluster(
            id=target_id,
            profile_id=source.profile_id,
            embedding_signature=source.embedding_signature,
            centroid=vector,
            vector_sum=vector,
            member_count=1,
            membership_revision=1,
            first_seen_ts=_article_ts(moved),
            last_seen_ts=_article_ts(moved),
            algorithm_version="manual-split-v1",
            strong_indicators=_indicators([moved]),
            strict_cve_identity=membership_is_strict_cve_record(moved),
            tag_scope=source.tag_scope,
        ).to_document()
        combined = [moved]
        target_base = 0
    final_target_rows = revised_memberships(combined, target_id, target_base)

    audit = {
        "membership_edit_id": operation_id,
        "last_membership_edited_at": edited_at,
        "last_membership_edited_by": reviewer,
        "last_membership_edit_comment": str(comment).strip(),
    }
    source_final.update(audit, updated_at=edited_at)
    target_final.update(audit, updated_at=edited_at)
    expected_memberships = {
        str(row["article_id"]): str(row["cluster_id"]) for row in source_rows + target_rows
    }
    return validate_cluster_membership_edit(
        {
            "id": operation_id,
            "profile_id": source.profile_id,
            "article_id": article_id,
            "action": action,
            "source_cluster_id": source.id,
            "target_cluster_id": target_id,
            "expected_source_membership_revision": source.membership_revision,
            "expected_target_membership_revision": target.membership_revision if target else None,
            "expected_memberships": expected_memberships,
            "source_cluster": source_final,
            "target_cluster": target_final,
            "memberships": final_source_rows + final_target_rows,
            "edited_at": edited_at,
            "edited_by": reviewer,
            "comment": str(comment).strip(),
            "status": "approved",
            "snapshot_policy": "preserve_history_rebuild_changed_clusters",
        }
    )


def validate_cluster_membership_edit(operation: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete edit document before persistence mutates state."""

    document = dict(operation or {})
    required = (
        "id",
        "profile_id",
        "article_id",
        "action",
        "source_cluster_id",
        "target_cluster_id",
        "expected_source_membership_revision",
        "expected_memberships",
        "source_cluster",
        "target_cluster",
        "memberships",
        "edited_at",
        "edited_by",
        "snapshot_policy",
    )
    if any(document.get(field) is None for field in required):
        raise ValueError("cluster membership edit is incomplete")
    if document["action"] not in {"move", "split"}:
        raise ValueError("unsupported cluster membership edit")
    source = ThreatCluster.from_document(dict(document["source_cluster"]))
    target = ThreatCluster.from_document(dict(document["target_cluster"]))
    profile_id = str(document["profile_id"])
    source_id = str(document["source_cluster_id"])
    target_id = str(document["target_cluster_id"])
    if source.id != source_id or target.id != target_id or source_id == target_id:
        raise ValueError("membership edit cluster identities are invalid")
    if source.profile_id != profile_id or target.profile_id != profile_id:
        raise ValueError("membership edit profiles do not match")
    if source.tag_scope != target.tag_scope:
        raise ValueError("membership edit tag scopes do not match")
    rows = [dict(row) for row in document["memberships"]]
    if any(str(row.get("tag_scope") or "") != source.tag_scope for row in rows):
        raise ValueError("membership edit rows must share the cluster tag scope")
    article_ids = [str(row.get("article_id") or "") for row in rows]
    if not all(article_ids) or len(article_ids) != len(set(article_ids)):
        raise ValueError("membership edit article ids must be unique")
    expected = {
        str(key): str(value) for key, value in dict(document["expected_memberships"]).items()
    }
    if set(expected) != set(article_ids):
        raise ValueError("membership edit must cover every affected membership")
    counts = {
        source_id: sum(str(row.get("cluster_id") or "") == source_id for row in rows),
        target_id: sum(str(row.get("cluster_id") or "") == target_id for row in rows),
    }
    if counts[source_id] != source.member_count or counts[target_id] != target.member_count:
        raise ValueError("membership edit counts do not match cluster documents")
    if (
        next(
            (row for row in rows if str(row.get("article_id")) == str(document["article_id"])), {}
        ).get("cluster_id")
        != target_id
    ):
        raise ValueError("edited article must end in target cluster")
    if document["snapshot_policy"] != "preserve_history_rebuild_changed_clusters":
        raise ValueError("unsupported membership edit snapshot policy")
    document.update(
        {
            "id": str(document["id"]),
            "profile_id": profile_id,
            "article_id": str(document["article_id"]),
            "source_cluster_id": source_id,
            "target_cluster_id": target_id,
            "expected_source_membership_revision": int(
                document["expected_source_membership_revision"]
            ),
            "expected_target_membership_revision": (
                int(document["expected_target_membership_revision"])
                if document.get("expected_target_membership_revision") is not None
                else None
            ),
            "expected_memberships": expected,
            "source_cluster": source.to_document()
            | {
                key: value
                for key, value in dict(document["source_cluster"]).items()
                if key not in source.to_document()
            },
            "target_cluster": target.to_document()
            | {
                key: value
                for key, value in dict(document["target_cluster"]).items()
                if key not in target.to_document()
            },
            "memberships": rows,
            "edited_at": int(document["edited_at"]),
            "edited_by": " ".join(str(document["edited_by"]).split()),
        }
    )
    return document
