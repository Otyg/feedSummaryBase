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

"""Evidence-bound Map updates for long-term threat clusters."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from typing import Any, Protocol

from feedsummary_core.long_term.models import ThreatCluster
from feedsummary_core.long_term.snapshot_validation import (
    SnapshotValidationError,
    parse_snapshot_json,
    validate_cluster_snapshot,
)
from feedsummary_core.summarizer.token_budget import estimate_tokens, messages_to_text


class MapStore(Protocol):
    def get_threat_cluster(self, cluster_id: str) -> dict[str, Any] | None: ...

    def list_cluster_memberships(
        self, cluster_id: str, *, limit: int = 10000
    ) -> list[dict[str, Any]]: ...

    def get_articles_by_ids(self, article_ids: list[str]) -> list[dict[str, Any]]: ...

    def list_cluster_snapshots(
        self, profile_id: str, *, cluster_id: str | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]: ...

    def save_cluster_snapshot_revision(
        self,
        cluster_doc: dict[str, Any],
        snapshot_doc: dict[str, Any],
        *,
        expected_membership_revision: int,
        expected_summarized_revision: int,
    ) -> bool: ...


class MapLLM(Protocol):
    async def chat(
        self, messages: list[dict[str, str]], *, temperature: float = 0.0
    ) -> str: ...


class PromptBudgetError(ValueError):
    """Even the minimum structured Map input does not fit the configured budget."""


class SnapshotRevisionConflict(RuntimeError):
    """Cluster membership changed while its Map snapshot was generated."""


@dataclass(frozen=True)
class MapSettings:
    min_pending_articles: int = 3
    max_update_delay_hours: int = 24
    max_articles_per_call: int = 12
    article_clip_chars: int = 2500
    max_context_tokens: int = 8192
    max_output_tokens: int = 1200
    safety_margin_tokens: int = 512
    format_repair_attempts: int = 1

    def __post_init__(self) -> None:
        positive = (
            self.min_pending_articles,
            self.max_update_delay_hours,
            self.max_articles_per_call,
            self.article_clip_chars,
            self.max_context_tokens,
            self.max_output_tokens,
        )
        if any(value < 1 for value in positive):
            raise ValueError("Map limits must be positive")
        if self.safety_margin_tokens < 0:
            raise ValueError("safety_margin_tokens cannot be negative")
        if self.format_repair_attempts not in {0, 1}:
            raise ValueError("format_repair_attempts must be zero or one")


@dataclass(frozen=True)
class MapUpdateResult:
    cluster_id: str
    action: str
    previous_revision: int
    summarized_revision: int
    snapshot_id: str | None
    input_article_ids: tuple[str, ...]
    repair_attempted: bool
    estimated_prompt_tokens: int


def cluster_needs_map_update(
    cluster: ThreatCluster, *, now_ts: int, settings: MapSettings
) -> bool:
    pending = cluster.membership_revision - cluster.summarized_revision
    if pending <= 0:
        return False
    if pending >= settings.min_pending_articles:
        return True
    reference = cluster.last_summarized_at or cluster.last_seen_ts
    return now_ts - reference >= settings.max_update_delay_hours * 3600


def _membership_revision(membership: dict[str, Any], fallback: int) -> int:
    return int(membership.get("cluster_membership_revision") or fallback)


def _article_material(article: dict[str, Any], clip_chars: int) -> dict[str, Any]:
    body = str(
        article.get("text") or article.get("content") or article.get("summary") or ""
    ).strip()
    return {
        "id": str(article.get("id") or ""),
        "title": str(article.get("title") or "").strip(),
        "source": str(article.get("source") or "").strip(),
        "published_ts": int(article.get("published_ts") or article.get("fetched_at") or 0),
        "text": body[:clip_chars],
    }


def _render_messages(
    prompt_package: dict[str, Any],
    *,
    cluster: ThreatCluster,
    target_revision: int,
    allowed_article_ids: list[str],
    previous_snapshot: dict[str, Any] | None,
    articles: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system = str(prompt_package.get("system") or "").strip()
    template = str(prompt_package.get("user_template") or "").strip()
    schema = prompt_package.get("output_schema")
    if not system or not template or not isinstance(schema, dict):
        raise ValueError("cluster prompt package is incomplete")
    user = template.format(
        profile_id=cluster.profile_id,
        cluster_id=cluster.id,
        membership_revision=target_revision,
        allowed_article_ids=json.dumps(allowed_article_ids, ensure_ascii=False),
        previous_snapshot=json.dumps(previous_snapshot, ensure_ascii=False),
        new_articles=json.dumps(articles, ensure_ascii=False),
    )
    user += "\n\nOUTPUT_SCHEMA:\n" + json.dumps(schema, ensure_ascii=False)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _snapshot_id(cluster_id: str, revision: int, prompt_version: str) -> str:
    identity = f"feedsummary:cluster-snapshot:{cluster_id}:{revision}:{prompt_version}"
    return f"threat_snapshot_{uuid.uuid5(uuid.NAMESPACE_URL, identity).hex}"


async def update_cluster_map_snapshot(
    store: MapStore,
    llm: MapLLM,
    *,
    cluster_id: str,
    prompt_package: dict[str, Any],
    now_ts: int,
    settings: MapSettings | None = None,
    force: bool = False,
) -> MapUpdateResult:
    """Generate, validate and atomically persist one bounded cluster snapshot."""

    settings = settings or MapSettings()
    document = store.get_threat_cluster(cluster_id)
    if document is None:
        raise ValueError(f"unknown threat cluster: {cluster_id}")
    cluster = ThreatCluster.from_document(document)
    if not force and not cluster_needs_map_update(cluster, now_ts=now_ts, settings=settings):
        return MapUpdateResult(
            cluster.id,
            "skipped",
            cluster.summarized_revision,
            cluster.summarized_revision,
            cluster.latest_snapshot_id,
            (),
            False,
            0,
        )

    memberships = store.list_cluster_memberships(cluster.id)
    indexed_memberships = list(enumerate(memberships, start=1))
    ordered_memberships = [
        item
        for fallback_revision, item in sorted(
            indexed_memberships,
            key=lambda pair: (
                _membership_revision(pair[1], pair[0]),
                str(pair[1].get("article_id") or ""),
            ),
        )
    ]
    pending = [
        membership
        for index, membership in enumerate(ordered_memberships, start=1)
        if _membership_revision(membership, index) > cluster.summarized_revision
    ][: settings.max_articles_per_call]
    if not pending:
        return MapUpdateResult(
            cluster.id,
            "skipped",
            cluster.summarized_revision,
            cluster.summarized_revision,
            cluster.latest_snapshot_id,
            (),
            False,
            0,
        )
    selected_ids = [str(item["article_id"]) for item in pending]
    target_revision = max(
        _membership_revision(item, index)
        for index, item in enumerate(pending, start=cluster.summarized_revision + 1)
    )
    articles_by_id = {
        str(article.get("id") or ""): article
        for article in store.get_articles_by_ids(selected_ids)
    }
    if set(selected_ids).difference(articles_by_id):
        raise ValueError("one or more cluster articles are missing from persistence")
    article_material = [
        _article_material(articles_by_id[article_id], settings.article_clip_chars)
        for article_id in selected_ids
    ]
    snapshots = store.list_cluster_snapshots(
        cluster.profile_id, cluster_id=cluster.id, limit=1
    )
    previous_snapshot = snapshots[0].get("payload") if snapshots else None
    allowed_ids = [
        str(item["article_id"])
        for index, item in enumerate(ordered_memberships, start=1)
        if _membership_revision(item, index) <= target_revision
    ]
    messages = _render_messages(
        prompt_package,
        cluster=cluster,
        target_revision=target_revision,
        allowed_article_ids=allowed_ids,
        previous_snapshot=previous_snapshot,
        articles=article_material,
    )
    budget = settings.max_context_tokens - settings.max_output_tokens - settings.safety_margin_tokens
    estimated = estimate_tokens(messages_to_text(messages))
    while estimated > budget and len(article_material) > 1:
        article_material.pop()
        selected_ids.pop()
        pending.pop()
        target_revision = _membership_revision(pending[-1], cluster.summarized_revision + len(pending))
        allowed_ids = [
            str(item["article_id"])
            for index, item in enumerate(ordered_memberships, start=1)
            if _membership_revision(item, index) <= target_revision
        ]
        messages = _render_messages(
            prompt_package,
            cluster=cluster,
            target_revision=target_revision,
            allowed_article_ids=allowed_ids,
            previous_snapshot=previous_snapshot,
            articles=article_material,
        )
        estimated = estimate_tokens(messages_to_text(messages))
    if estimated > budget:
        raise PromptBudgetError(
            f"minimum cluster update requires about {estimated} tokens; budget is {budget}"
        )

    raw = await llm.chat(
        messages,
        temperature=float(prompt_package.get("temperature", 0.0)),
    )
    repair_attempted = False
    try:
        payload = validate_cluster_snapshot(
            parse_snapshot_json(raw),
            profile_id=cluster.profile_id,
            cluster_id=cluster.id,
            membership_revision=target_revision,
            allowed_article_ids=set(allowed_ids),
        )
    except SnapshotValidationError as error:
        if settings.format_repair_attempts < 1:
            raise
        repair_attempted = True
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "Reparera endast JSON-formatet och schemafelen. Lägg inte till fakta "
                    "eller evidens. Returnera endast det korrigerade JSON-objektet."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "validation_error": str(error),
                        "allowed_article_ids": allowed_ids,
                        "output_schema": prompt_package["output_schema"],
                        "invalid_response": raw[:12000],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        repaired = await llm.chat(repair_messages, temperature=0.0)
        payload = validate_cluster_snapshot(
            parse_snapshot_json(repaired),
            profile_id=cluster.profile_id,
            cluster_id=cluster.id,
            membership_revision=target_revision,
            allowed_article_ids=set(allowed_ids),
        )

    prompt_version = str(prompt_package.get("prompt_version") or "").strip()
    if not prompt_version:
        raise ValueError("cluster prompt package has no prompt_version")
    snapshot_id = _snapshot_id(cluster.id, target_revision, prompt_version)
    snapshot = {
        "id": snapshot_id,
        "profile_id": cluster.profile_id,
        "cluster_id": cluster.id,
        "membership_revision": target_revision,
        "prompt_version": prompt_version,
        "created_at": int(now_ts),
        "input_article_ids": selected_ids,
        "payload": payload,
    }
    updated_cluster = replace(
        cluster,
        summarized_revision=target_revision,
        latest_snapshot_id=snapshot_id,
        last_summarized_at=int(now_ts),
    )
    if not store.save_cluster_snapshot_revision(
        updated_cluster.to_document(),
        snapshot,
        expected_membership_revision=cluster.membership_revision,
        expected_summarized_revision=cluster.summarized_revision,
    ):
        raise SnapshotRevisionConflict(
            f"cluster changed while snapshot was generated: {cluster.id}"
        )
    return MapUpdateResult(
        cluster.id,
        "saved",
        cluster.summarized_revision,
        target_revision,
        snapshot_id,
        tuple(selected_ids),
        repair_attempted,
        estimated,
    )
