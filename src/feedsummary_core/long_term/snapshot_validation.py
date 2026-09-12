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

"""Strict validation for LLM-produced cluster snapshots."""

from __future__ import annotations

import json
import re
from typing import Any

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "profile_id",
        "cluster_id",
        "membership_revision",
        "title",
        "summary",
        "confidence",
        "insufficient_evidence",
        "facts",
        "timeline",
        "mitre_techniques",
        "uncertainties",
    }
)
_CONFIDENCE = frozenset({"low", "medium", "high"})
_FACT_STATUS = frozenset({"active", "disputed", "superseded"})
_MITRE_PATTERN = re.compile(r"^T\d{4}(?:\.\d{3})?$")


class SnapshotValidationError(ValueError):
    """A cluster snapshot is malformed or cites evidence outside its input."""


def parse_snapshot_json(raw: str) -> dict[str, Any]:
    """Parse a bare JSON object; wrappers and Markdown fences are rejected."""

    try:
        payload = json.loads(str(raw or "").strip())
    except json.JSONDecodeError as exc:
        raise SnapshotValidationError(f"snapshot is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise SnapshotValidationError("snapshot must be a JSON object")
    return payload


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise SnapshotValidationError(
            f"{field} must be a non-empty string; got {type(value).__name__}"
        )
    if not value.strip():
        raise SnapshotValidationError(f"{field} must be a non-empty string; got blank")
    return value.strip()


def _evidence_ids(
    value: Any,
    *,
    field: str,
    allowed_article_ids: frozenset[str],
) -> list[str]:
    if not isinstance(value, list) or not value:
        raise SnapshotValidationError(f"{field} must contain evidence article IDs")
    result = [_non_empty_string(item, field) for item in value]
    if len(result) != len(set(result)):
        raise SnapshotValidationError(f"{field} contains duplicate article IDs")
    unknown = sorted(set(result).difference(allowed_article_ids))
    if unknown:
        raise SnapshotValidationError(f"{field} contains unknown article IDs: {unknown}")
    return result


def _validate_evidenced_items(
    items: Any,
    *,
    field: str,
    allowed_article_ids: frozenset[str],
    extra_fields: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise SnapshotValidationError(f"{field} must be an array")
    validated = []
    allowed_fields = {"statement", "evidence_article_ids", *extra_fields}
    for index, raw_item in enumerate(items):
        item_field = f"{field}[{index}]"
        if not isinstance(raw_item, dict):
            raise SnapshotValidationError(f"{item_field} must be an object")
        unknown_fields = set(raw_item).difference(allowed_fields)
        if unknown_fields:
            raise SnapshotValidationError(
                f"{item_field} contains unknown fields: {sorted(unknown_fields)}"
            )
        item = dict(raw_item)
        item["statement"] = _non_empty_string(item.get("statement"), f"{item_field}.statement")
        item["evidence_article_ids"] = _evidence_ids(
            item.get("evidence_article_ids"),
            field=f"{item_field}.evidence_article_ids",
            allowed_article_ids=allowed_article_ids,
        )
        validated.append(item)
    return validated


def validate_cluster_snapshot(
    payload: dict[str, Any],
    *,
    profile_id: str,
    cluster_id: str,
    membership_revision: int,
    allowed_article_ids: set[str] | frozenset[str],
) -> dict[str, Any]:
    """Return a normalized snapshot after identity, schema and evidence checks."""

    if not isinstance(payload, dict):
        raise SnapshotValidationError("snapshot must be an object")
    unknown_fields = set(payload).difference(_TOP_LEVEL_FIELDS)
    missing_fields = _TOP_LEVEL_FIELDS.difference(payload)
    if unknown_fields:
        raise SnapshotValidationError(f"snapshot contains unknown fields: {sorted(unknown_fields)}")
    if missing_fields:
        raise SnapshotValidationError(f"snapshot is missing fields: {sorted(missing_fields)}")
    if payload.get("schema_version") != 1:
        raise SnapshotValidationError("schema_version must equal 1")
    if payload.get("profile_id") != profile_id or payload.get("cluster_id") != cluster_id:
        raise SnapshotValidationError("snapshot identity does not match its input")
    if payload.get("membership_revision") != membership_revision:
        raise SnapshotValidationError("snapshot membership_revision does not match its input")
    allowed = frozenset(str(value) for value in allowed_article_ids)
    result = dict(payload)
    result["title"] = _non_empty_string(payload.get("title"), "title")
    result["summary"] = _non_empty_string(payload.get("summary"), "summary")
    if payload.get("confidence") not in _CONFIDENCE:
        raise SnapshotValidationError("confidence must be low, medium or high")
    if not isinstance(payload.get("insufficient_evidence"), bool):
        raise SnapshotValidationError("insufficient_evidence must be boolean")

    facts = _validate_evidenced_items(
        payload.get("facts"),
        field="facts",
        allowed_article_ids=allowed,
        extra_fields=frozenset({"status"}),
    )
    for index, fact in enumerate(facts):
        if fact.get("status") not in _FACT_STATUS:
            raise SnapshotValidationError(
                f"facts[{index}].status must be active, disputed or superseded"
            )
    result["facts"] = facts
    result["timeline"] = _validate_evidenced_items(
        payload.get("timeline"),
        field="timeline",
        allowed_article_ids=allowed,
        extra_fields=frozenset({"event_time"}),
    )
    techniques = _validate_evidenced_items(
        payload.get("mitre_techniques"),
        field="mitre_techniques",
        allowed_article_ids=allowed,
        extra_fields=frozenset({"technique_id"}),
    )
    for index, technique in enumerate(techniques):
        technique_id = _non_empty_string(
            technique.get("technique_id"), f"mitre_techniques[{index}].technique_id"
        )
        if not _MITRE_PATTERN.fullmatch(technique_id):
            raise SnapshotValidationError(
                f"mitre_techniques[{index}].technique_id is invalid"
            )
        technique["technique_id"] = technique_id
    result["mitre_techniques"] = techniques
    uncertainties = payload.get("uncertainties")
    if not isinstance(uncertainties, list):
        raise SnapshotValidationError("uncertainties must be an array")
    result["uncertainties"] = [
        _non_empty_string(value, f"uncertainties[{index}]")
        for index, value in enumerate(uncertainties)
    ]
    return result
