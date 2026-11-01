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

"""Pure state transitions for explicit human review of ambiguous clusters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from feedsummary_core.long_term.models import ClusterStatus, ThreatCluster
from feedsummary_core.long_term.reconciliation import (
    build_cluster_merge_operation,
    validate_cluster_merge_operation,
)

REVIEW_DECISIONS = frozenset({"keep_separate", "exclude", "merge"})


def _audit_fields(
    *,
    decision: str,
    reviewed_at: int,
    reviewed_by: str,
    comment: str,
    target_cluster_id: str | None = None,
) -> dict[str, Any]:
    normalized_decision = str(decision or "").strip()
    normalized_reviewer = " ".join(str(reviewed_by or "").split())
    normalized_comment = " ".join(str(comment or "").split())
    if normalized_decision not in REVIEW_DECISIONS:
        raise ValueError("unsupported cluster review decision")
    if reviewed_at < 1 or not normalized_reviewer:
        raise ValueError("cluster review requires timestamp and reviewer")
    if len(normalized_reviewer) > 120 or len(normalized_comment) > 2000:
        raise ValueError("cluster review audit metadata is too long")
    if normalized_decision == "merge" and not str(target_cluster_id or "").strip():
        raise ValueError("merge review requires a target cluster")
    return {
        "review_decision": normalized_decision,
        "reviewed_at": int(reviewed_at),
        "reviewed_by": normalized_reviewer,
        "review_comment": normalized_comment or None,
        "reviewed_target_cluster_id": (
            str(target_cluster_id).strip() if target_cluster_id else None
        ),
    }


def build_cluster_review_resolution(
    cluster_document: dict[str, Any],
    *,
    decision: str,
    reviewed_at: int,
    reviewed_by: str,
    comment: str = "",
) -> dict[str, Any]:
    """Build an optimistic non-merge resolution for one review cluster."""

    cluster = ThreatCluster.from_document(cluster_document)
    if cluster.status is not ClusterStatus.NEEDS_REVIEW or cluster.review_decision:
        raise ValueError("cluster is not awaiting review")
    if decision not in {"keep_separate", "exclude"}:
        raise ValueError("non-merge review decision must keep or exclude the cluster")
    status = (
        ClusterStatus.ACTIVE if decision == "keep_separate" else ClusterStatus.CLOSED
    )
    resolved = replace(
        cluster,
        status=status,
        **_audit_fields(
            decision=decision,
            reviewed_at=reviewed_at,
            reviewed_by=reviewed_by,
            comment=comment,
        ),
    ).to_document()
    document = dict(cluster_document)
    document.update(resolved)
    document["updated_at"] = int(reviewed_at)
    return document


def _review_reconciliation_id(
    review: ThreatCluster, target: ThreatCluster
) -> str:
    identity = {
        "policy": "cluster-review-v1",
        "profile_id": review.profile_id,
        "review_cluster_id": review.id,
        "review_revision": review.membership_revision,
        "target_cluster_id": target.id,
        "target_revision": target.membership_revision,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"threat_review_{digest[:32]}"


def build_cluster_review_merge_operation(
    review_cluster_document: dict[str, Any],
    target_cluster_document: dict[str, Any],
    memberships_by_cluster: dict[str, Iterable[dict[str, Any]]],
    *,
    allowed_candidate_cluster_ids: Iterable[str],
    reviewed_at: int,
    reviewed_by: str,
    comment: str = "",
) -> dict[str, Any]:
    """Build an auditable merge after a reviewer selects an allowed candidate."""

    review = ThreatCluster.from_document(review_cluster_document)
    target = ThreatCluster.from_document(target_cluster_document)
    allowed = {str(value).strip() for value in allowed_candidate_cluster_ids}
    if review.status is not ClusterStatus.NEEDS_REVIEW or review.review_decision:
        raise ValueError("cluster is not awaiting review")
    if target.id not in allowed:
        raise ValueError("target cluster is not an approved review candidate")
    if target.status not in {ClusterStatus.ACTIVE, ClusterStatus.DORMANT}:
        raise ValueError("target cluster is not open for review merge")
    if review.profile_id != target.profile_id:
        raise ValueError("review and target cluster profiles differ")

    audit = _audit_fields(
        decision="merge",
        reviewed_at=reviewed_at,
        reviewed_by=reviewed_by,
        comment=comment,
        target_cluster_id=target.id,
    )
    operation = build_cluster_merge_operation(
        [review_cluster_document, target_cluster_document],
        memberships_by_cluster,
        primary_cluster_id=target.id,
        reconciliation_id=_review_reconciliation_id(review, target),
        reconciled_at=reviewed_at,
    )
    for tombstone in operation["superseded_clusters"]:
        if str(tombstone.get("id") or "") == review.id:
            tombstone.update(audit)
    operation["review_resolution"] = {
        "review_cluster_id": review.id,
        "target_cluster_id": target.id,
        **audit,
    }
    return validate_cluster_merge_operation(operation)


def validate_cluster_review_resolution(document: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete document accepted by atomic store transitions."""

    cluster = ThreatCluster.from_document(document)
    if cluster.review_decision not in {"keep_separate", "exclude"}:
        raise ValueError("cluster review resolution is incomplete")
    if cluster.review_decision == "keep_separate" and cluster.status is not ClusterStatus.ACTIVE:
        raise ValueError("kept cluster must be active")
    if cluster.review_decision == "exclude" and cluster.status is not ClusterStatus.CLOSED:
        raise ValueError("excluded cluster must be closed")
    return dict(document)
