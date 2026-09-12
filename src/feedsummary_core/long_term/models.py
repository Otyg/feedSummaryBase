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

"""Backend-neutral domain models for long-term threat analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ClusterStatus(str, Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    CLOSED = "closed"
    NEEDS_REVIEW = "needs_review"


ACTIVE_CLUSTER_STATUSES = frozenset({ClusterStatus.ACTIVE, ClusterStatus.DORMANT})


@dataclass(frozen=True)
class EmbeddingSignature:
    """Identity of one embedding vector space."""

    model: str
    dimensions: int
    instruction: str

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("embedding model must not be empty")
        if self.dimensions < 1:
            raise ValueError("embedding dimensions must be positive")

    @property
    def key(self) -> str:
        return f"{self.model}\x1f{self.dimensions}\x1f{self.instruction.strip()}"


@dataclass(frozen=True)
class ThreatCluster:
    """The deterministic portion of an event/campaign cluster."""

    id: str
    profile_id: str
    embedding_signature: EmbeddingSignature
    centroid: tuple[float, ...]
    vector_sum: tuple[float, ...]
    member_count: int
    membership_revision: int
    first_seen_ts: int
    last_seen_ts: int
    status: ClusterStatus = ClusterStatus.ACTIVE
    algorithm_version: str = "online-centroid-v1"
    strong_indicators: tuple[str, ...] = ()
    strict_cve_identity: bool = True
    summarized_revision: int = 0
    latest_snapshot_id: str | None = None
    last_summarized_at: int | None = None
    reopened_count: int = 0
    last_reopened_at: int | None = None
    superseded_by_cluster_id: str | None = None
    reconciliation_id: str | None = None
    reconciled_at: int | None = None
    review_decision: str | None = None
    reviewed_at: int | None = None
    reviewed_by: str | None = None
    review_comment: str | None = None
    reviewed_target_cluster_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.profile_id:
            raise ValueError("cluster id and profile id must not be empty")
        superseded = self.superseded_by_cluster_id is not None
        if superseded:
            if self.member_count != 0 or self.status is not ClusterStatus.CLOSED:
                raise ValueError("a superseded cluster must be closed and empty")
            if not self.reconciliation_id or self.reconciled_at is None:
                raise ValueError("a superseded cluster requires reconciliation lineage")
        elif self.member_count < 1:
            raise ValueError("cluster member_count must be positive")
        if self.membership_revision < self.member_count:
            raise ValueError("membership_revision cannot be lower than member_count")
        if not 0 <= self.summarized_revision <= self.membership_revision:
            raise ValueError("summarized_revision must cover an existing revision")
        if self.reopened_count < 0:
            raise ValueError("reopened_count cannot be negative")
        if (self.reopened_count == 0) != (self.last_reopened_at is None):
            raise ValueError("reopened_count and last_reopened_at must be set together")
        if self.last_reopened_at is not None and not (
            self.first_seen_ts <= self.last_reopened_at <= self.last_seen_ts
        ):
            raise ValueError("last_reopened_at must be within the cluster interval")
        if self.review_decision not in {None, "keep_separate", "exclude", "merge"}:
            raise ValueError("unsupported cluster review decision")
        if self.review_decision is None:
            if self.reviewed_at is not None or self.reviewed_by is not None:
                raise ValueError("unresolved cluster cannot have review audit metadata")
        elif self.reviewed_at is None or not str(self.reviewed_by or "").strip():
            raise ValueError("resolved cluster review requires timestamp and reviewer")
        if self.review_decision == "keep_separate" and self.status is ClusterStatus.NEEDS_REVIEW:
            raise ValueError("a separate reviewed cluster must leave needs_review")
        if self.review_decision == "exclude" and self.status is not ClusterStatus.CLOSED:
            raise ValueError("an excluded reviewed cluster must be closed")
        if self.review_decision == "merge" and not self.reviewed_target_cluster_id:
            raise ValueError("a merged reviewed cluster requires its target cluster")
        if len(self.centroid) != self.embedding_signature.dimensions:
            raise ValueError("centroid does not match the embedding signature")
        if len(self.vector_sum) != self.embedding_signature.dimensions:
            raise ValueError("vector_sum does not match the embedding signature")
        if self.first_seen_ts < 1 or self.last_seen_ts < self.first_seen_ts:
            raise ValueError("invalid cluster observation interval")

    def to_document(self) -> dict[str, Any]:
        """Flatten the model into the backend-neutral persistence document."""

        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "status": self.status.value,
            "first_seen_ts": self.first_seen_ts,
            "last_seen_ts": self.last_seen_ts,
            "embedding_model": self.embedding_signature.model,
            "embedding_dimension": self.embedding_signature.dimensions,
            "embedding_instruction": self.embedding_signature.instruction,
            "centroid": list(self.centroid),
            "vector_sum": list(self.vector_sum),
            "member_count": self.member_count,
            "membership_revision": self.membership_revision,
            "algorithm_version": self.algorithm_version,
            "strong_indicators": list(self.strong_indicators),
            "strict_cve_identity": self.strict_cve_identity,
            "summarized_revision": self.summarized_revision,
            "latest_snapshot_id": self.latest_snapshot_id,
            "last_summarized_at": self.last_summarized_at,
            "reopened_count": self.reopened_count,
            "last_reopened_at": self.last_reopened_at,
            "superseded_by_cluster_id": self.superseded_by_cluster_id,
            "reconciliation_id": self.reconciliation_id,
            "reconciled_at": self.reconciled_at,
            "review_decision": self.review_decision,
            "reviewed_at": self.reviewed_at,
            "reviewed_by": self.reviewed_by,
            "review_comment": self.review_comment,
            "reviewed_target_cluster_id": self.reviewed_target_cluster_id,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> ThreatCluster:
        """Validate and rebuild a cluster from a persistence document."""

        return cls(
            id=str(document["id"]),
            profile_id=str(document["profile_id"]),
            embedding_signature=EmbeddingSignature(
                model=str(document["embedding_model"]),
                dimensions=int(document["embedding_dimension"]),
                instruction=str(document.get("embedding_instruction") or ""),
            ),
            centroid=tuple(float(value) for value in document["centroid"]),
            vector_sum=tuple(float(value) for value in document["vector_sum"]),
            member_count=int(document["member_count"]),
            membership_revision=int(document["membership_revision"]),
            first_seen_ts=int(document["first_seen_ts"]),
            last_seen_ts=int(document["last_seen_ts"]),
            status=ClusterStatus(str(document.get("status") or ClusterStatus.ACTIVE.value)),
            algorithm_version=str(document.get("algorithm_version") or "online-centroid-v1"),
            strong_indicators=tuple(
                sorted(
                    {
                        str(value).strip().casefold()
                        for value in document.get("strong_indicators") or []
                        if str(value).strip()
                    }
                )
            ),
            # Missing means conservative legacy behaviour. Reconciliation can
            # derive a more precise value from persisted membership evidence.
            strict_cve_identity=bool(document.get("strict_cve_identity", True)),
            summarized_revision=int(document.get("summarized_revision") or 0),
            latest_snapshot_id=(
                str(document["latest_snapshot_id"])
                if document.get("latest_snapshot_id")
                else None
            ),
            last_summarized_at=(
                int(document["last_summarized_at"])
                if document.get("last_summarized_at") is not None
                else None
            ),
            reopened_count=int(document.get("reopened_count") or 0),
            last_reopened_at=(
                int(document["last_reopened_at"])
                if document.get("last_reopened_at") is not None
                else None
            ),
            superseded_by_cluster_id=(
                str(document["superseded_by_cluster_id"])
                if document.get("superseded_by_cluster_id")
                else None
            ),
            reconciliation_id=(
                str(document["reconciliation_id"])
                if document.get("reconciliation_id")
                else None
            ),
            reconciled_at=(
                int(document["reconciled_at"])
                if document.get("reconciled_at") is not None
                else None
            ),
            review_decision=(
                str(document["review_decision"])
                if document.get("review_decision")
                else None
            ),
            reviewed_at=(
                int(document["reviewed_at"])
                if document.get("reviewed_at") is not None
                else None
            ),
            reviewed_by=(
                str(document["reviewed_by"])
                if document.get("reviewed_by")
                else None
            ),
            review_comment=(
                str(document["review_comment"])
                if document.get("review_comment") is not None
                else None
            ),
            reviewed_target_cluster_id=(
                str(document["reviewed_target_cluster_id"])
                if document.get("reviewed_target_cluster_id")
                else None
            ),
        )
