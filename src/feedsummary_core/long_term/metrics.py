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

"""Deterministic rolling metrics and coverage warnings for threat landscapes."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Protocol

_DAY_SECONDS = 86400
_STATUSES = ("active", "dormant", "closed", "needs_review")


class LandscapeMetricsStore(Protocol):
    def list_threat_clusters(
        self, profile_id: str, **kwargs: Any
    ) -> list[dict[str, Any]]: ...

    def list_cluster_memberships(
        self, cluster_id: str, *, limit: int = 10000
    ) -> list[dict[str, Any]]: ...

    def get_articles_by_ids(self, article_ids: list[str]) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class LandscapeMetricSettings:
    windows_days: tuple[int, ...] = (7, 30, 90)
    bucket_days: int = 7
    min_independent_events_for_trend: int = 3
    source_concentration_threshold: float = 0.75
    source_mix_change_threshold: float = 0.35
    max_clusters: int = 10000
    max_memberships_per_cluster: int = 10000
    min_snapshot_member_count: int = 1

    def __post_init__(self) -> None:
        if not self.windows_days or any(days < 1 for days in self.windows_days):
            raise ValueError("windows_days must contain positive values")
        if len(set(self.windows_days)) != len(self.windows_days):
            raise ValueError("windows_days must not contain duplicates")
        if self.bucket_days < 1:
            raise ValueError("bucket_days must be positive")
        if self.min_independent_events_for_trend < 1:
            raise ValueError("min_independent_events_for_trend must be positive")
        for value, name in (
            (self.source_concentration_threshold, "source_concentration_threshold"),
            (self.source_mix_change_threshold, "source_mix_change_threshold"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.max_clusters < 1 or self.max_memberships_per_cluster < 1:
            raise ValueError("input limits must be positive")
        if self.min_snapshot_member_count < 1:
            raise ValueError("min_snapshot_member_count must be positive")


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _cluster_member_count(cluster: dict[str, Any]) -> int:
    """Read member count while retaining compatibility with older documents."""

    return _safe_int(cluster.get("member_count")) or _safe_int(
        cluster.get("membership_revision")
    )


def _in_period(value: Any, start_ts: int, end_ts: int) -> bool:
    timestamp = _safe_int(value)
    return timestamp > 0 and start_ts <= timestamp < end_ts


def _membership_time(membership: dict[str, Any]) -> int:
    evidence = membership.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    return _safe_int(
        membership.get("article_ts")
        or evidence.get("published_ts")
        or membership.get("assigned_at")
    )


def _membership_source(
    membership: dict[str, Any], article_sources: dict[str, str]
) -> str:
    evidence = membership.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    article_id = str(membership.get("article_id") or "")
    return str(
        evidence.get("source")
        or membership.get("source")
        or article_sources.get(article_id)
        or ""
    ).strip()


def _source_mix_distance(current: Counter[str], previous: Counter[str]) -> float | None:
    current_total = sum(current.values())
    previous_total = sum(previous.values())
    if not current_total or not previous_total:
        return None
    sources = set(current).union(previous)
    distance = sum(
        abs(current[source] / current_total - previous[source] / previous_total)
        for source in sources
    ) / 2.0
    return round(distance, 6)


def _period_values(
    memberships: list[dict[str, Any]],
    clusters_by_id: dict[str, dict[str, Any]],
    article_sources: dict[str, str],
    *,
    start_ts: int,
    end_ts: int,
) -> dict[str, Any]:
    selected = [
        row for row in memberships if start_ts <= _membership_time(row) < end_ts
    ]
    event_ids = sorted(
        {
            str(row.get("cluster_id") or "")
            for row in selected
            if str(row.get("cluster_id") or "") in clusters_by_id
        }
    )
    event_sources: dict[str, set[str]] = defaultdict(set)
    source_counts: Counter[str] = Counter()
    unknown_source_articles = 0
    for row in selected:
        cluster_id = str(row.get("cluster_id") or "")
        source = _membership_source(row, article_sources)
        if source:
            source_counts[source] += 1
            event_sources[cluster_id].add(source)
        else:
            unknown_source_articles += 1
    known_source_articles = sum(source_counts.values())
    dominant_source_share = (
        round(max(source_counts.values()) / known_source_articles, 6)
        if known_source_articles
        else None
    )
    status_counts = {status: 0 for status in _STATUSES}
    for cluster_id in event_ids:
        status = str(clusters_by_id[cluster_id].get("status") or "")
        if status in status_counts:
            status_counts[status] += 1
    return {
        "period_start_ts": start_ts,
        "period_end_ts": end_ts,
        "independent_event_count": len(event_ids),
        "new_event_count": sum(
            _in_period(clusters_by_id[cluster_id].get("first_seen_ts"), start_ts, end_ts)
            for cluster_id in event_ids
        ),
        "reopened_event_count": sum(
            _in_period(
                clusters_by_id[cluster_id].get("last_reopened_at"), start_ts, end_ts
            )
            for cluster_id in event_ids
        ),
        "article_count": len(selected),
        "event_cluster_ids": event_ids,
        "status_counts": status_counts,
        "source_coverage": {
            "unique_source_count": len(source_counts),
            "known_source_article_count": known_source_articles,
            "unknown_source_article_count": unknown_source_articles,
            "source_article_counts": dict(sorted(source_counts.items())),
            "dominant_source_share": dominant_source_share,
            "multi_source_event_count": sum(
                len(sources) > 1 for sources in event_sources.values()
            ),
            "unique_sources_per_event": {
                cluster_id: len(event_sources.get(cluster_id, set()))
                for cluster_id in event_ids
            },
        },
        "_source_counts": source_counts,
    }


def _weekly_buckets(
    memberships: list[dict[str, Any]],
    clusters_by_id: dict[str, dict[str, Any]],
    article_sources: dict[str, str],
    *,
    start_ts: int,
    end_ts: int,
    bucket_days: int,
) -> list[dict[str, Any]]:
    bucket_seconds = bucket_days * _DAY_SECONDS
    bucket_count = math.ceil((end_ts - start_ts) / bucket_seconds)
    buckets = []
    for reverse_index in range(bucket_count - 1, -1, -1):
        bucket_end = end_ts - reverse_index * bucket_seconds
        bucket_start = max(start_ts, bucket_end - bucket_seconds)
        values = _period_values(
            memberships,
            clusters_by_id,
            article_sources,
            start_ts=bucket_start,
            end_ts=bucket_end,
        )
        buckets.append(
            {
                "start_ts": bucket_start,
                "end_ts": bucket_end,
                "independent_event_count": values["independent_event_count"],
                "new_event_count": values["new_event_count"],
                "article_count": values["article_count"],
                "event_cluster_ids": values["event_cluster_ids"],
                "unique_source_count": values["source_coverage"]["unique_source_count"],
                "partial": bucket_end - bucket_start < bucket_seconds,
            }
        )
    return buckets


def compute_landscape_metrics(
    *,
    profile_id: str,
    period_end_ts: int,
    clusters: list[dict[str, Any]],
    memberships: list[dict[str, Any]],
    articles: list[dict[str, Any]] | None = None,
    settings: LandscapeMetricSettings | None = None,
    input_truncated: bool = False,
) -> dict[str, Any]:
    """Build reproducible event-level metrics for one frozen report timestamp."""

    settings = settings or LandscapeMetricSettings()
    profile_id = str(profile_id or "").strip()
    if not profile_id or period_end_ts < 1:
        raise ValueError("profile_id and a positive period_end_ts are required")
    profile_clusters = [
        dict(row)
        for row in clusters
        if str(row.get("profile_id") or "") == profile_id
        and str(row.get("id") or "")
    ]
    unresolved_review_cluster_ids = {
        str(row["id"])
        for row in profile_clusters
        if str(row.get("status") or "") == "needs_review"
        and not row.get("review_decision")
    }
    excluded_review_cluster_ids = {
        str(row["id"])
        for row in profile_clusters
        if str(row.get("review_decision") or "") == "exclude"
    }
    ignored_review_cluster_ids = (
        unresolved_review_cluster_ids | excluded_review_cluster_ids
    )
    clusters_by_id = {
        str(row["id"]): row
        for row in profile_clusters
        if str(row["id"]) not in ignored_review_cluster_ids
    }
    article_sources = {
        str(row.get("id")): str(row.get("source") or "").strip()
        for row in articles or []
        if str(row.get("id") or "")
    }
    valid_memberships = []
    invalid_membership_count = 0
    seen_memberships: set[tuple[str, str]] = set()
    duplicate_membership_count = 0
    ignored_review_membership_count = 0
    for row in sorted(
        memberships,
        key=lambda item: (
            _membership_time(item),
            str(item.get("cluster_id") or ""),
            str(item.get("article_id") or ""),
        ),
    ):
        cluster_id = str(row.get("cluster_id") or "")
        article_id = str(row.get("article_id") or "")
        if cluster_id in ignored_review_cluster_ids:
            ignored_review_membership_count += 1
            continue
        if cluster_id not in clusters_by_id or not article_id or _membership_time(row) < 1:
            invalid_membership_count += 1
            continue
        identity = (cluster_id, article_id)
        if identity in seen_memberships:
            duplicate_membership_count += 1
            continue
        seen_memberships.add(identity)
        valid_memberships.append(dict(row))

    windows = []
    global_warnings: set[str] = set()
    for days in settings.windows_days:
        seconds = days * _DAY_SECONDS
        start_ts = period_end_ts - seconds
        current = _period_values(
            valid_memberships,
            clusters_by_id,
            article_sources,
            start_ts=start_ts,
            end_ts=period_end_ts,
        )
        previous = _period_values(
            valid_memberships,
            clusters_by_id,
            article_sources,
            start_ts=start_ts - seconds,
            end_ts=start_ts,
        )
        for values in (current, previous):
            snapshot_eligible = [
                cluster_id
                for cluster_id in values["event_cluster_ids"]
                if _cluster_member_count(clusters_by_id[cluster_id])
                >= settings.min_snapshot_member_count
            ]
            values["snapshot_eligible_cluster_ids"] = snapshot_eligible
            values["atomic_observation_cluster_count"] = (
                len(values["event_cluster_ids"]) - len(snapshot_eligible)
            )
        buckets = _weekly_buckets(
            valid_memberships,
            clusters_by_id,
            article_sources,
            start_ts=start_ts,
            end_ts=period_end_ts,
            bucket_days=settings.bucket_days,
        )
        nonempty_buckets = sum(bucket["article_count"] > 0 for bucket in buckets)
        missing_buckets = len(buckets) - nonempty_buckets
        source_mix_distance = _source_mix_distance(
            current.pop("_source_counts"), previous.pop("_source_counts")
        )
        warnings = []
        if current["independent_event_count"] == 0:
            warnings.append("no_independent_events")
        elif (
            current["independent_event_count"]
            < settings.min_independent_events_for_trend
        ):
            warnings.append("insufficient_independent_events")
        if len(buckets) > 1 and nonempty_buckets < 2:
            warnings.append("insufficient_time_buckets")
        if missing_buckets:
            warnings.append("missing_time_buckets")
        if current["source_coverage"]["known_source_article_count"] == 0:
            warnings.append("no_known_sources")
        if current["source_coverage"]["unknown_source_article_count"]:
            warnings.append("unknown_source_articles")
        dominant_share = current["source_coverage"]["dominant_source_share"]
        if (
            dominant_share is not None
            and dominant_share >= settings.source_concentration_threshold
        ):
            warnings.append("high_source_concentration")
        if previous["independent_event_count"] == 0:
            warnings.append("missing_comparison_events")
        if previous["source_coverage"]["known_source_article_count"] == 0:
            warnings.append("missing_comparison_sources")
        if (
            source_mix_distance is not None
            and source_mix_distance >= settings.source_mix_change_threshold
        ):
            warnings.append("source_mix_shift")
        pending_snapshot_count = sum(
            _safe_int(clusters_by_id[cluster_id].get("summarized_revision"))
            < _safe_int(clusters_by_id[cluster_id].get("membership_revision"))
            for cluster_id in current["snapshot_eligible_cluster_ids"]
        )
        if pending_snapshot_count:
            warnings.append("pending_cluster_snapshots")
        trend_eligible = (
            current["independent_event_count"]
            >= settings.min_independent_events_for_trend
            and nonempty_buckets >= 2
            and missing_buckets == 0
            and current["source_coverage"]["known_source_article_count"] > 0
            and previous["independent_event_count"] > 0
            and not pending_snapshot_count
            and dominant_share is not None
            and dominant_share < settings.source_concentration_threshold
            and source_mix_distance is not None
            and source_mix_distance < settings.source_mix_change_threshold
        )
        global_warnings.update(warnings)
        windows.append(
            {
                "days": days,
                **current,
                "weekly_buckets": buckets,
                "comparison": {
                    **previous,
                    "independent_event_delta": current["independent_event_count"]
                    - previous["independent_event_count"],
                    "new_event_delta": current["new_event_count"]
                    - previous["new_event_count"],
                    "article_delta": current["article_count"] - previous["article_count"],
                    "source_mix_distance": source_mix_distance,
                },
                "coverage": {
                    "nonempty_time_bucket_count": nonempty_buckets,
                    "missing_time_bucket_count": missing_buckets,
                    "pending_snapshot_count": pending_snapshot_count,
                    "trend_eligible": trend_eligible,
                    "warnings": warnings,
                },
            }
        )
    if invalid_membership_count:
        global_warnings.add("invalid_memberships_ignored")
    if duplicate_membership_count:
        global_warnings.add("duplicate_memberships_ignored")
    if input_truncated:
        global_warnings.add("input_truncated")
    if unresolved_review_cluster_ids:
        global_warnings.add("unresolved_cluster_reviews_ignored")
    if excluded_review_cluster_ids:
        global_warnings.add("excluded_review_clusters_ignored")
    return {
        "schema_version": 1,
        "profile_id": profile_id,
        "period_end_ts": int(period_end_ts),
        "settings": {
            "windows_days": list(settings.windows_days),
            "bucket_days": settings.bucket_days,
            "min_independent_events_for_trend": (
                settings.min_independent_events_for_trend
            ),
            "source_concentration_threshold": settings.source_concentration_threshold,
            "source_mix_change_threshold": settings.source_mix_change_threshold,
            "min_snapshot_member_count": settings.min_snapshot_member_count,
        },
        "windows": windows,
        "quality": {
            "input_cluster_count": len(clusters_by_id),
            "input_membership_count": len(valid_memberships),
            "invalid_membership_count": invalid_membership_count,
            "duplicate_membership_count": duplicate_membership_count,
            "unresolved_review_cluster_count": len(unresolved_review_cluster_ids),
            "excluded_review_cluster_count": len(excluded_review_cluster_ids),
            "ignored_review_membership_count": ignored_review_membership_count,
            "input_truncated": bool(input_truncated),
            "warning_codes": sorted(global_warnings),
        },
    }


def build_landscape_metrics(
    store: LandscapeMetricsStore,
    *,
    profile_id: str,
    period_end_ts: int,
    settings: LandscapeMetricSettings | None = None,
) -> dict[str, Any]:
    """Load bounded persistence input and delegate to the pure metric calculation."""

    settings = settings or LandscapeMetricSettings()
    earliest_ts = period_end_ts - 2 * max(settings.windows_days) * _DAY_SECONDS
    clusters = store.list_threat_clusters(
        profile_id,
        min_last_seen_ts=earliest_ts,
        limit=settings.max_clusters + 1,
    )
    input_truncated = len(clusters) > settings.max_clusters
    clusters = clusters[: settings.max_clusters]
    memberships = []
    for cluster in sorted(clusters, key=lambda row: str(row.get("id") or "")):
        rows = store.list_cluster_memberships(
            str(cluster.get("id") or ""),
            limit=settings.max_memberships_per_cluster + 1,
        )
        if len(rows) > settings.max_memberships_per_cluster:
            input_truncated = True
        memberships.extend(rows[: settings.max_memberships_per_cluster])
    missing_source_ids = sorted(
        {
            str(row.get("article_id") or "")
            for row in memberships
            if str(row.get("article_id") or "")
            and not _membership_source(row, {})
        }
    )
    articles = store.get_articles_by_ids(missing_source_ids) if missing_source_ids else []
    return compute_landscape_metrics(
        profile_id=profile_id,
        period_end_ts=period_end_ts,
        clusters=clusters,
        memberships=memberships,
        articles=articles,
        settings=settings,
        input_truncated=input_truncated,
    )
