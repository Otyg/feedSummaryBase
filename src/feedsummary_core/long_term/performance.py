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

"""Deterministic, network-free capacity benchmark for long-term analysis."""

from __future__ import annotations

import tracemalloc
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from feedsummary_core.long_term.clustering import (
    ClusteringSettings,
    assign_article,
    create_cluster,
)
from feedsummary_core.long_term.metrics import (
    LandscapeMetricSettings,
    build_landscape_metrics,
)
from feedsummary_core.long_term.models import EmbeddingSignature, ThreatCluster
from feedsummary_core.long_term.reduce_analysis import ReduceSettings, select_report_snapshots
from feedsummary_core.long_term.reporting import render_landscape_report_messages
from feedsummary_core.summarizer.token_budget import messages_to_text

_DAY_SECONDS = 86400


@dataclass(frozen=True)
class PerformanceBenchmarkSettings:
    days: int
    events_per_day: int = 4
    embedding_dimensions: int = 64
    candidate_probes: int = 100
    period_end_ts: int = 1_800_000_000

    def __post_init__(self) -> None:
        if self.days not in {90, 180}:
            raise ValueError("benchmark days must be 90 or 180")
        if min(
            self.events_per_day,
            self.embedding_dimensions,
            self.candidate_probes,
            self.period_end_ts,
        ) < 1:
            raise ValueError("benchmark settings must be positive")


class _SyntheticStore:
    def __init__(
        self,
        clusters: list[dict[str, Any]],
        memberships: list[dict[str, Any]],
        articles: list[dict[str, Any]],
        snapshots: list[dict[str, Any]],
    ) -> None:
        self.clusters = clusters
        self.memberships_by_cluster: dict[str, list[dict[str, Any]]] = {}
        for membership in memberships:
            self.memberships_by_cluster.setdefault(
                str(membership["cluster_id"]), []
            ).append(membership)
        self.articles_by_id = {str(article["id"]): article for article in articles}
        self.snapshots = snapshots

    def list_threat_clusters(
        self, profile_id: str, *, limit: int = 10000, **_kwargs: Any
    ) -> list[dict[str, Any]]:
        return [
            cluster
            for cluster in self.clusters
            if cluster["profile_id"] == profile_id
        ][:limit]

    def list_cluster_memberships(
        self, cluster_id: str, *, limit: int = 10000
    ) -> list[dict[str, Any]]:
        return self.memberships_by_cluster.get(cluster_id, [])[:limit]

    def get_articles_by_ids(self, article_ids: list[str]) -> list[dict[str, Any]]:
        return [
            self.articles_by_id[article_id]
            for article_id in article_ids
            if article_id in self.articles_by_id
        ]

    def list_cluster_snapshots(
        self,
        profile_id: str,
        *,
        cluster_id: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        rows = [row for row in self.snapshots if row["profile_id"] == profile_id]
        if cluster_id is not None:
            rows = [row for row in rows if row["cluster_id"] == cluster_id]
        return rows[:limit]


def _embedding(index: int, dimensions: int) -> list[float]:
    return [float(((index + 1) * (dimension + 3)) % 97 + 1) for dimension in range(dimensions)]


def _synthetic_dataset(settings: PerformanceBenchmarkSettings) -> tuple[
    list[ThreatCluster],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    signature = EmbeddingSignature(
        "benchmark-embedding-v1",
        settings.embedding_dimensions,
        "represent cyber incident identity",
    )
    clusters = []
    memberships = []
    articles = []
    snapshots = []
    for day in range(settings.days):
        for event in range(settings.events_per_day):
            index = day * settings.events_per_day + event
            article_id = f"benchmark-article-{index:06d}"
            timestamp = settings.period_end_ts - day * _DAY_SECONDS - event
            cluster = create_cluster(
                profile_id="benchmark-profile",
                article_id=article_id,
                article_ts=timestamp,
                embedding=_embedding(index, settings.embedding_dimensions),
                signature=signature,
            )
            cluster_doc = cluster.to_document()
            clusters.append(cluster)
            memberships.append(
                {
                    "profile_id": "benchmark-profile",
                    "cluster_id": cluster.id,
                    "article_id": article_id,
                    "article_ts": timestamp,
                    "assigned_at": timestamp,
                    "evidence": {
                        "source": f"source-{index % 8}",
                        "published_ts": timestamp,
                    },
                }
            )
            articles.append(
                {
                    "id": article_id,
                    "source": f"source-{index % 8}",
                    "published_ts": timestamp,
                }
            )
            snapshots.append(
                {
                    "id": f"benchmark-snapshot-{index:06d}",
                    "profile_id": "benchmark-profile",
                    "cluster_id": cluster.id,
                    "membership_revision": 1,
                    "created_at": timestamp,
                    "payload": {
                        "profile_id": "benchmark-profile",
                        "cluster_id": cluster.id,
                        "summary": f"Synthetic incident {index}",
                    },
                }
            )
            clusters[-1] = ThreatCluster.from_document(cluster_doc)
    return clusters, [row.to_document() for row in clusters], memberships, articles, snapshots


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


def run_long_term_performance_benchmark(
    settings: PerformanceBenchmarkSettings,
) -> dict[str, Any]:
    """Measure deterministic analysis stages over a reproducible synthetic dataset."""

    tracemalloc.start()
    total_started = perf_counter()

    build_started = perf_counter()
    cluster_objects, cluster_docs, memberships, articles, snapshots = _synthetic_dataset(
        settings
    )
    dataset_build_ms = _elapsed_ms(build_started)
    event_count = len(cluster_objects)
    store = _SyntheticStore(cluster_docs, memberships, articles, snapshots)

    candidate_started = perf_counter()
    candidate_settings = ClusteringSettings(
        similarity_threshold=0.8,
        ambiguity_margin=0.03,
        candidate_window_days=settings.days,
    )
    probe_count = min(settings.candidate_probes, event_count)
    for probe in range(probe_count):
        source = cluster_objects[(probe * event_count) // probe_count]
        assign_article(
            profile_id="benchmark-profile",
            article_ts=settings.period_end_ts,
            embedding=source.centroid,
            signature=source.embedding_signature,
            candidates=cluster_objects,
            strong_indicators=(),
            settings=candidate_settings,
        )
    candidate_selection_ms = _elapsed_ms(candidate_started)

    metrics_started = perf_counter()
    metric_settings = LandscapeMetricSettings(
        windows_days=(7, 30, 90),
        max_clusters=event_count,
        max_memberships_per_cluster=2,
    )
    metrics = build_landscape_metrics(
        store,
        profile_id="benchmark-profile",
        period_end_ts=settings.period_end_ts + 1,
        settings=metric_settings,
    )
    metrics_ms = _elapsed_ms(metrics_started)

    snapshot_started = perf_counter()
    reduce_settings = ReduceSettings(
        max_context_tokens=1_000_000,
        max_output_tokens=2500,
        max_snapshot_records=event_count,
    )
    selected_snapshots = select_report_snapshots(
        store,
        profile_id="benchmark-profile",
        metrics=metrics,
        settings=reduce_settings,
    )
    snapshot_selection_ms = _elapsed_ms(snapshot_started)

    render_started = perf_counter()
    messages = render_landscape_report_messages(
        {
            "system": "Return benchmark JSON.",
            "user_template": "metrics={metrics} snapshots={cluster_snapshots}",
            "output_schema": {"type": "object"},
        },
        profile_context={"id": "benchmark-profile"},
        metrics=metrics,
        snapshots=selected_snapshots,
        previous_report=None,
    )
    reduce_input_chars = len(messages_to_text(messages))
    reduce_input_ms = _elapsed_ms(render_started)

    total_ms = _elapsed_ms(total_started)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "schema_version": 1,
        "dataset": {
            "days": settings.days,
            "events_per_day": settings.events_per_day,
            "event_count": event_count,
            "embedding_dimensions": settings.embedding_dimensions,
            "candidate_probes": probe_count,
        },
        "counts": {
            "cluster_count": len(cluster_docs),
            "membership_count": len(memberships),
            "snapshot_count": len(snapshots),
            "selected_snapshot_count": len(selected_snapshots),
            "reduce_input_chars": reduce_input_chars,
        },
        "timings_ms": {
            "dataset_build": dataset_build_ms,
            "candidate_selection": candidate_selection_ms,
            "metrics": metrics_ms,
            "snapshot_selection": snapshot_selection_ms,
            "reduce_input": reduce_input_ms,
            "total": total_ms,
        },
        "peak_memory_mb": round(peak_bytes / (1024 * 1024), 3),
    }
