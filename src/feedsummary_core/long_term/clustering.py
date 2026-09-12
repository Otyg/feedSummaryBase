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

"""Pure, deterministic event-clustering operations without persistence or LLM calls."""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import Enum

from feedsummary_core.long_term.models import (
    ACTIVE_CLUSTER_STATUSES,
    ClusterStatus,
    EmbeddingSignature,
    ThreatCluster,
)


class VectorValidationError(ValueError):
    """Raised when an embedding cannot safely participate in clustering."""


class AssignmentAction(str, Enum):
    MATCH = "match"
    NEW_CLUSTER = "new_cluster"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True)
class ClusteringSettings:
    similarity_threshold: float = 0.80
    ambiguity_margin: float = 0.03
    candidate_window_days: int = 60

    def __post_init__(self) -> None:
        if not -1.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1 and 1")
        if not 0.0 <= self.ambiguity_margin <= 2.0:
            raise ValueError("ambiguity_margin must be between 0 and 2")
        if self.candidate_window_days < 1:
            raise ValueError("candidate_window_days must be positive")


@dataclass(frozen=True)
class AssignmentDecision:
    action: AssignmentAction
    cluster_id: str | None
    similarity: float | None
    second_similarity: float | None
    reason: str
    best_candidate_cluster_id: str | None = None
    second_candidate_cluster_id: str | None = None


def _validated_vector(
    values: Sequence[float], dimensions: int | None = None
) -> tuple[float, ...]:
    if not values:
        raise VectorValidationError("embedding vector must not be empty")
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise VectorValidationError("embedding vector contains a non-number") from exc
    if dimensions is not None and len(vector) != dimensions:
        raise VectorValidationError(
            f"embedding dimension mismatch: expected {dimensions}, got {len(vector)}"
        )
    if not all(math.isfinite(value) for value in vector):
        raise VectorValidationError("embedding vector contains NaN or infinity")
    if math.sqrt(sum(value * value for value in vector)) <= 0.0:
        raise VectorValidationError("embedding vector has zero norm")
    return vector


def _unit_vector(
    values: Sequence[float], dimensions: int | None = None
) -> tuple[float, ...]:
    vector = _validated_vector(values, dimensions)
    norm = math.sqrt(sum(value * value for value in vector))
    return tuple(value / norm for value in vector)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity after rejecting malformed and zero-norm vectors."""

    left_vector = _validated_vector(left)
    right_vector = _validated_vector(right, len(left_vector))
    left_norm = math.sqrt(sum(value * value for value in left_vector))
    right_norm = math.sqrt(sum(value * value for value in right_vector))
    return sum(a * b for a, b in zip(left_vector, right_vector)) / (
        left_norm * right_norm
    )


def stable_cluster_id(
    profile_id: str,
    first_article_id: str,
    algorithm_version: str = "online-centroid-v1",
) -> str:
    """Create the same cluster ID when creation is retried for the same first article."""

    if not profile_id.strip() or not first_article_id.strip() or not algorithm_version.strip():
        raise ValueError("profile, article and algorithm version must not be empty")
    identity = f"feedsummary:threat-cluster:{profile_id}:{first_article_id}:{algorithm_version}"
    return f"threat_cluster_{uuid.uuid5(uuid.NAMESPACE_URL, identity).hex}"


def create_cluster(
    *,
    profile_id: str,
    article_id: str,
    article_ts: int,
    embedding: Sequence[float],
    signature: EmbeddingSignature,
    strong_indicators: Iterable[str] = (),
    strict_cve_identity: bool = True,
    algorithm_version: str = "online-centroid-v1",
) -> ThreatCluster:
    """Create a one-member cluster from a validated, normalized article vector."""

    if article_ts < 1:
        raise ValueError("article_ts must be positive")
    normalized = _unit_vector(embedding, signature.dimensions)
    return ThreatCluster(
        id=stable_cluster_id(profile_id, article_id, algorithm_version),
        profile_id=profile_id,
        embedding_signature=signature,
        centroid=normalized,
        vector_sum=normalized,
        member_count=1,
        membership_revision=1,
        first_seen_ts=article_ts,
        last_seen_ts=article_ts,
        status=ClusterStatus.ACTIVE,
        algorithm_version=algorithm_version,
        strong_indicators=tuple(
            sorted({str(value).strip().casefold() for value in strong_indicators if str(value).strip()})
        ),
        strict_cve_identity=bool(strict_cve_identity),
    )


def add_cluster_member(
    cluster: ThreatCluster,
    *,
    article_ts: int,
    embedding: Sequence[float],
    strong_indicators: Iterable[str] = (),
    strict_cve_identity: bool = True,
) -> ThreatCluster:
    """Return an updated cluster using an exact incremental arithmetic centroid."""

    normalized = _unit_vector(embedding, cluster.embedding_signature.dimensions)
    vector_sum = tuple(a + b for a, b in zip(cluster.vector_sum, normalized))
    member_count = cluster.member_count + 1
    centroid = tuple(value / member_count for value in vector_sum)
    reopened = cluster.status is ClusterStatus.DORMANT
    return replace(
        cluster,
        centroid=centroid,
        vector_sum=vector_sum,
        member_count=member_count,
        membership_revision=cluster.membership_revision + 1,
        first_seen_ts=min(cluster.first_seen_ts, article_ts),
        last_seen_ts=max(cluster.last_seen_ts, article_ts),
        status=ClusterStatus.ACTIVE,
        reopened_count=cluster.reopened_count + int(reopened),
        last_reopened_at=article_ts if reopened else cluster.last_reopened_at,
        strong_indicators=tuple(
            sorted(
                set(cluster.strong_indicators)
                | {
                    str(value).strip().casefold()
                    for value in strong_indicators
                    if str(value).strip()
                }
            )
        ),
        strict_cve_identity=(
            cluster.strict_cve_identity and bool(strict_cve_identity)
        ),
    )


def cluster_status_at(
    cluster: ThreatCluster,
    *,
    now_ts: int,
    dormant_after_days: int,
    close_after_days: int,
) -> ClusterStatus:
    """Derive lifecycle status without mutating cluster history."""

    if now_ts < 1:
        raise ValueError("now_ts must be positive")
    if dormant_after_days < 1:
        raise ValueError("dormant_after_days must be positive")
    if close_after_days < dormant_after_days:
        raise ValueError("close_after_days cannot be lower than dormant_after_days")
    if cluster.status in {ClusterStatus.CLOSED, ClusterStatus.NEEDS_REVIEW}:
        return cluster.status
    age_seconds = max(0, now_ts - cluster.last_seen_ts)
    if age_seconds >= close_after_days * 86400:
        return ClusterStatus.CLOSED
    if age_seconds >= dormant_after_days * 86400:
        return ClusterStatus.DORMANT
    return ClusterStatus.ACTIVE


def assign_article(
    *,
    profile_id: str,
    article_ts: int,
    embedding: Sequence[float],
    signature: EmbeddingSignature,
    candidates: Iterable[ThreatCluster],
    strong_indicators: Iterable[str] = (),
    strict_cve_identity: bool = True,
    settings: ClusteringSettings | None = None,
) -> AssignmentDecision:
    """Choose a compatible recent cluster or return a deterministic non-match decision."""

    settings = settings or ClusteringSettings()
    vector = _validated_vector(embedding, signature.dimensions)
    earliest_candidate_ts = article_ts - settings.candidate_window_days * 86400
    article_indicator_set = {
        str(value).strip().casefold() for value in strong_indicators if str(value).strip()
    }
    article_cves = {
        value for value in article_indicator_set if value.startswith("cve:")
    }
    scored = []
    conflicting_cves = False
    for cluster in candidates:
        if cluster.profile_id != profile_id:
            continue
        if cluster.embedding_signature != signature:
            continue
        if cluster.status not in ACTIVE_CLUSTER_STATUSES:
            continue
        if cluster.last_seen_ts < earliest_candidate_ts:
            continue
        cluster_indicator_set = set(cluster.strong_indicators)
        cluster_cves = {
            value for value in cluster_indicator_set if value.startswith("cve:")
        }
        if (
            strict_cve_identity
            and cluster.strict_cve_identity
            and article_cves
            and cluster_cves
            and article_cves.isdisjoint(cluster_cves)
        ):
            conflicting_cves = True
            continue
        score = cosine_similarity(vector, cluster.centroid)
        overlap = len(article_indicator_set.intersection(cluster_indicator_set))
        time_distance = abs(article_ts - cluster.last_seen_ts)
        scored.append((score, overlap, time_distance, cluster.id))

    if not scored:
        return AssignmentDecision(
            AssignmentAction.NEW_CLUSTER,
            None,
            None,
            None,
            "conflicting_cve_indicators"
            if conflicting_cves
            else "no_compatible_candidate",
        )

    scored.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    best_similarity, best_overlap, _, best_id = scored[0]
    second_similarity = scored[1][0] if len(scored) > 1 else None
    second_id = scored[1][3] if len(scored) > 1 else None
    if best_similarity < settings.similarity_threshold:
        return AssignmentDecision(
            AssignmentAction.NEW_CLUSTER,
            None,
            best_similarity,
            second_similarity,
            "below_similarity_threshold",
            best_candidate_cluster_id=best_id,
            second_candidate_cluster_id=second_id,
        )
    if (
        second_similarity is not None
        and second_similarity >= settings.similarity_threshold
        and best_similarity - second_similarity < settings.ambiguity_margin
    ):
        return AssignmentDecision(
            AssignmentAction.NEEDS_REVIEW,
            None,
            best_similarity,
            second_similarity,
            "ambiguous_candidates",
            best_candidate_cluster_id=best_id,
            second_candidate_cluster_id=second_id,
        )
    return AssignmentDecision(
        AssignmentAction.MATCH,
        best_id,
        best_similarity,
        second_similarity,
        "similarity_match_with_indicator_support"
        if best_overlap
        else "similarity_match",
        best_candidate_cluster_id=best_id,
        second_candidate_cluster_id=second_id,
    )
