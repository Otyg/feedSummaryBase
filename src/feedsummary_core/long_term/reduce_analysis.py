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

"""Bounded direct/segmented Reduce execution for threat-landscape reports."""

from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from feedsummary_core.long_term.lease import LeaseGuard
from feedsummary_core.long_term.reporting import (
    ReportValidationError,
    build_landscape_report_document,
    build_landscape_summary_document,
    parse_landscape_report_json,
    render_landscape_report_messages,
    validate_landscape_report,
)
from feedsummary_core.summarizer.token_budget import estimate_tokens, messages_to_text

_SEGMENT_FIELDS = frozenset({"schema_version", "profile_id", "segment_id", "snapshots"})
_SEGMENT_SNAPSHOT_FIELDS = frozenset(
    {"id", "profile_id", "cluster_id", "summary", "key_facts", "uncertainties"}
)


class ReduceStore(Protocol):
    def list_cluster_snapshots(
        self, profile_id: str, *, cluster_id: str | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]: ...

    def list_threat_landscape_reports(
        self, profile_id: str, *, limit: int = 100
    ) -> list[dict[str, Any]]: ...

    def get_threat_landscape_report(self, report_id: str) -> dict[str, Any] | None: ...

    def save_threat_landscape_report(self, report_doc: dict[str, Any]) -> bool: ...

    def get_summary_doc(self, summary_doc_id: str) -> dict[str, Any] | None: ...

    def save_summary_doc(self, summary_doc: dict[str, Any]) -> Any: ...


class ReduceLLM(Protocol):
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> str: ...


class ReduceBudgetError(ValueError):
    """The Reduce input cannot fit even after one deterministic segmentation pass."""


class ReportMirrorError(RuntimeError):
    """A published report could not be safely mirrored to summary_docs."""


@dataclass(frozen=True)
class ReduceSettings:
    max_context_tokens: int = 8192
    max_output_tokens: int = 2500
    safety_margin_tokens: int = 512
    max_snapshots_per_segment: int = 12
    max_snapshot_records: int = 10000
    format_repair_attempts: int = 1

    def __post_init__(self) -> None:
        positive = (
            self.max_context_tokens,
            self.max_output_tokens,
            self.max_snapshots_per_segment,
            self.max_snapshot_records,
        )
        if any(value < 1 for value in positive):
            raise ValueError("Reduce limits must be positive")
        if self.safety_margin_tokens < 0:
            raise ValueError("safety_margin_tokens cannot be negative")
        if self.format_repair_attempts not in {0, 1}:
            raise ValueError("format_repair_attempts must be zero or one")
        if self.input_budget < 1:
            raise ValueError("Reduce output reservation leaves no input budget")

    @property
    def input_budget(self) -> int:
        return self.max_context_tokens - self.max_output_tokens - self.safety_margin_tokens


@dataclass(frozen=True)
class ReduceResult:
    action: str
    report_id: str
    input_snapshot_ids: tuple[str, ...]
    segment_count: int
    llm_call_count: int
    repair_attempted: bool
    estimated_final_prompt_tokens: int
    mirror_action: str = "disabled"


def _non_empty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReportValidationError(f"{field} must be a non-empty string")
    return value.strip()


def mirror_landscape_report(
    store: ReduceStore, report_document: dict[str, Any]
) -> str:
    """Idempotently mirror a canonical report without overwriting conflicts."""

    mirror = build_landscape_summary_document(report_document)
    mirror_id = str(mirror["id"])
    existing = store.get_summary_doc(mirror_id)
    if existing is not None:
        if existing != mirror:
            raise ReportMirrorError(f"conflicting summary mirror already exists: {mirror_id}")
        return "existing"

    try:
        store.save_summary_doc(mirror)
    except Exception as error:
        raise ReportMirrorError(f"summary mirror write failed: {mirror_id}") from error
    persisted = store.get_summary_doc(mirror_id)
    if persisted != mirror:
        raise ReportMirrorError(f"summary mirror could not be verified: {mirror_id}")
    return "saved"


def _event_cluster_ids(metrics: dict[str, Any]) -> set[str]:
    return {
        str(cluster_id)
        for window in metrics.get("windows") or []
        if isinstance(window, dict)
        for cluster_id in window.get("event_cluster_ids") or []
        if str(cluster_id)
    }


def select_report_snapshots(
    store: ReduceStore,
    *,
    profile_id: str,
    metrics: dict[str, Any],
    settings: ReduceSettings,
) -> list[dict[str, Any]]:
    """Select one latest snapshot per relevant cluster at the frozen report cutoff."""

    relevant = _event_cluster_ids(metrics)
    if not relevant:
        return []
    period_end_ts = int(metrics.get("period_end_ts") or 0)
    rows = store.list_cluster_snapshots(
        profile_id,
        limit=settings.max_snapshot_records + 1,
    )
    if len(rows) > settings.max_snapshot_records:
        raise ReduceBudgetError("snapshot input exceeds max_snapshot_records")
    eligible = [
        dict(row)
        for row in rows
        if str(row.get("cluster_id") or "") in relevant
        and int(row.get("created_at") or 0) <= period_end_ts
    ]
    eligible.sort(
        key=lambda row: (
            str(row.get("cluster_id") or ""),
            -int(row.get("created_at") or 0),
            str(row.get("id") or ""),
        )
    )
    selected = {}
    for row in eligible:
        selected.setdefault(str(row.get("cluster_id") or ""), row)
    return [selected[cluster_id] for cluster_id in sorted(selected)]


def _apply_snapshot_coverage(
    metrics: dict[str, Any], snapshots: list[dict[str, Any]]
) -> dict[str, Any]:
    result = copy.deepcopy(metrics)
    available = {str(row.get("cluster_id") or "") for row in snapshots}
    quality = result.setdefault("quality", {})
    warning_codes = {
        str(value) for value in quality.get("warning_codes") or [] if str(value)
    }
    for window in result.get("windows") or []:
        if not isinstance(window, dict):
            continue
        missing = sorted(
            {
                str(value)
                for value in window.get("event_cluster_ids") or []
                if str(value) and str(value) not in available
            }
        )
        coverage = window.setdefault("coverage", {})
        coverage["missing_snapshot_cluster_ids"] = missing
        warnings = [str(value) for value in coverage.get("warnings") or [] if str(value)]
        if missing:
            coverage["trend_eligible"] = False
            if "missing_report_snapshots" not in warnings:
                warnings.append("missing_report_snapshots")
            warning_codes.add("missing_report_snapshots")
        coverage["warnings"] = warnings
    quality["warning_codes"] = sorted(warning_codes)
    return result


def _previous_report(
    store: ReduceStore, profile_id: str, period_end_ts: int
) -> tuple[str | None, dict[str, Any] | None]:
    rows = store.list_threat_landscape_reports(profile_id, limit=100)
    eligible = [
        row
        for row in rows
        if int(row.get("period_end_ts") or 0) < period_end_ts
        and str(row.get("status") or "") == "published"
    ]
    if not eligible:
        return None, None
    eligible.sort(
        key=lambda row: (-int(row.get("period_end_ts") or 0), str(row.get("id") or ""))
    )
    row = eligible[0]
    report_id = str(row.get("id") or "") or None
    material = {
        "id": report_id,
        "period_start_ts": row.get("period_start_ts"),
        "period_end_ts": row.get("period_end_ts"),
        "analysis": row.get("analysis"),
        "forecast": row.get("forecast"),
        "quality": row.get("quality"),
    }
    return report_id, material


def _segment_id(profile_id: str, snapshots: list[dict[str, Any]]) -> str:
    identity = ":".join(str(row.get("id") or "") for row in snapshots)
    value = f"feedsummary:landscape-segment:{profile_id}:{identity}"
    return f"threat_segment_{uuid.uuid5(uuid.NAMESPACE_URL, value).hex}"


def _segment_messages(
    prompt_package: dict[str, Any],
    *,
    profile_id: str,
    segment_id: str,
    snapshots: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system = _non_empty(prompt_package.get("system"), "segment_prompt.system")
    template = _non_empty(
        prompt_package.get("user_template"), "segment_prompt.user_template"
    )
    schema = prompt_package.get("output_schema")
    if not isinstance(schema, dict):
        raise ReportValidationError("segment_prompt.output_schema must be an object")
    user = template.format(
        profile_id=profile_id,
        segment_id=segment_id,
        cluster_snapshots=json.dumps(
            snapshots,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    user += "\n\nOUTPUT_SCHEMA:\n" + json.dumps(
        schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _exact_fields(value: dict[str, Any], expected: frozenset[str], field: str) -> None:
    unknown = set(value).difference(expected)
    missing = expected.difference(value)
    if unknown or missing:
        raise ReportValidationError(
            f"{field} fields differ; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def validate_landscape_segment(
    payload: dict[str, Any],
    *,
    profile_id: str,
    segment_id: str,
    snapshots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate a one-to-one compact representation of input snapshots."""

    if not isinstance(payload, dict):
        raise ReportValidationError("segment must be an object")
    _exact_fields(payload, _SEGMENT_FIELDS, "segment")
    if (
        payload.get("schema_version") != 1
        or payload.get("profile_id") != profile_id
        or payload.get("segment_id") != segment_id
    ):
        raise ReportValidationError("segment identity does not match its input")
    raw_snapshots = payload.get("snapshots")
    if not isinstance(raw_snapshots, list):
        raise ReportValidationError("segment.snapshots must be an array")
    expected = {
        str(snapshot.get("id") or ""): str(snapshot.get("cluster_id") or "")
        for snapshot in snapshots
    }
    normalized = {}
    for index, raw in enumerate(raw_snapshots):
        field = f"segment.snapshots[{index}]"
        if not isinstance(raw, dict):
            raise ReportValidationError(f"{field} must be an object")
        _exact_fields(raw, _SEGMENT_SNAPSHOT_FIELDS, field)
        snapshot_id = _non_empty(raw.get("id"), f"{field}.id")
        cluster_id = _non_empty(raw.get("cluster_id"), f"{field}.cluster_id")
        if snapshot_id not in expected or expected[snapshot_id] != cluster_id:
            raise ReportValidationError(f"{field} identity is not in the segment input")
        if raw.get("profile_id") != profile_id or snapshot_id in normalized:
            raise ReportValidationError(f"{field} profile or identity is invalid")
        for key in ("key_facts", "uncertainties"):
            values = raw.get(key)
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ReportValidationError(f"{field}.{key} must contain strings")
        normalized[snapshot_id] = {
            "id": snapshot_id,
            "profile_id": profile_id,
            "cluster_id": cluster_id,
            "payload": {
                "profile_id": profile_id,
                "cluster_id": cluster_id,
                "summary": _non_empty(raw.get("summary"), f"{field}.summary"),
                "key_facts": [value.strip() for value in raw["key_facts"]],
                "uncertainties": [value.strip() for value in raw["uncertainties"]],
            },
        }
    if set(normalized) != set(expected):
        raise ReportValidationError("segment output must contain every input snapshot exactly once")
    return [normalized[snapshot_id] for snapshot_id in sorted(normalized)]


def _segment_groups(
    prompt_package: dict[str, Any],
    *,
    profile_id: str,
    snapshots: list[dict[str, Any]],
    settings: ReduceSettings,
) -> list[list[dict[str, Any]]]:
    groups = []
    current: list[dict[str, Any]] = []
    for snapshot in snapshots:
        candidate = [*current, snapshot]
        segment_id = _segment_id(profile_id, candidate)
        messages = _segment_messages(
            prompt_package,
            profile_id=profile_id,
            segment_id=segment_id,
            snapshots=candidate,
        )
        estimate = estimate_tokens(messages_to_text(messages))
        if (
            current
            and (
                len(candidate) > settings.max_snapshots_per_segment
                or estimate > settings.input_budget
            )
        ):
            groups.append(current)
            current = [snapshot]
            single_messages = _segment_messages(
                prompt_package,
                profile_id=profile_id,
                segment_id=_segment_id(profile_id, current),
                snapshots=current,
            )
            if estimate_tokens(messages_to_text(single_messages)) > settings.input_budget:
                raise ReduceBudgetError("one cluster snapshot cannot fit a segment prompt")
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


async def _repair_json(
    llm: ReduceLLM,
    *,
    raw: str,
    error: Exception,
    output_schema: dict[str, Any],
    identity: dict[str, Any],
    max_output_tokens: int,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Reparera endast JSON-format, schema och ID-fält. Lägg inte till fakta "
                "eller evidens. Returnera endast det korrigerade JSON-objektet."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "validation_error": str(error),
                    "required_identity": identity,
                    "output_schema": output_schema,
                    "invalid_response": raw[:16000],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]
    return await llm.chat(
        messages,
        temperature=0.0,
        max_output_tokens=max_output_tokens,
    )


async def run_landscape_reduce(
    store: ReduceStore,
    llm: ReduceLLM,
    *,
    profile_id: str,
    profile_context: dict[str, Any],
    metrics: dict[str, Any],
    final_prompt_package: dict[str, Any],
    segment_prompt_package: dict[str, Any],
    model: str,
    now_ts: int,
    forecast_horizon_days: frozenset[int] = frozenset({30, 90}),
    settings: ReduceSettings | None = None,
    mirror_to_summary_docs: bool = False,
    lease_guard: LeaseGuard | None = None,
) -> ReduceResult:
    """Generate, validate and idempotently save one frozen landscape report."""

    settings = settings or ReduceSettings()
    profile_id = _non_empty(profile_id, "profile_id")
    if now_ts < 1:
        raise ValueError("now_ts must be positive")
    snapshots = select_report_snapshots(
        store,
        profile_id=profile_id,
        metrics=metrics,
        settings=settings,
    )
    frozen_metrics = _apply_snapshot_coverage(metrics, snapshots)
    period_end_ts = int(frozen_metrics.get("period_end_ts") or 0)
    previous_report_id, previous = _previous_report(store, profile_id, period_end_ts)
    final_snapshots = snapshots
    messages = render_landscape_report_messages(
        final_prompt_package,
        profile_context=profile_context,
        metrics=frozen_metrics,
        snapshots=final_snapshots,
        previous_report=previous,
    )
    final_estimate = estimate_tokens(messages_to_text(messages))
    segment_count = 0
    llm_call_count = 0
    repair_attempted = False
    if final_estimate > settings.input_budget:
        compact = []
        groups = _segment_groups(
            segment_prompt_package,
            profile_id=profile_id,
            snapshots=snapshots,
            settings=settings,
        )
        for group in groups:
            segment_count += 1
            segment_id = _segment_id(profile_id, group)
            segment_messages = _segment_messages(
                segment_prompt_package,
                profile_id=profile_id,
                segment_id=segment_id,
                snapshots=group,
            )
            raw_segment = await llm.chat(
                segment_messages,
                temperature=float(segment_prompt_package.get("temperature", 0.0)),
                max_output_tokens=settings.max_output_tokens,
            )
            llm_call_count += 1
            try:
                segment_payload = parse_landscape_report_json(raw_segment)
                compact.extend(
                    validate_landscape_segment(
                        segment_payload,
                        profile_id=profile_id,
                        segment_id=segment_id,
                        snapshots=group,
                    )
                )
            except ReportValidationError as error:
                if settings.format_repair_attempts < 1:
                    raise
                repair_attempted = True
                repaired = await _repair_json(
                    llm,
                    raw=raw_segment,
                    error=error,
                    output_schema=segment_prompt_package["output_schema"],
                    identity={
                        "profile_id": profile_id,
                        "segment_id": segment_id,
                        "snapshot_ids": sorted(str(row["id"]) for row in group),
                    },
                    max_output_tokens=settings.max_output_tokens,
                )
                llm_call_count += 1
                compact.extend(
                    validate_landscape_segment(
                        parse_landscape_report_json(repaired),
                        profile_id=profile_id,
                        segment_id=segment_id,
                        snapshots=group,
                    )
                )
        final_snapshots = sorted(compact, key=lambda row: str(row["cluster_id"]))
        messages = render_landscape_report_messages(
            final_prompt_package,
            profile_context=profile_context,
            metrics=frozen_metrics,
            snapshots=final_snapshots,
            previous_report=previous,
        )
        final_estimate = estimate_tokens(messages_to_text(messages))
        if final_estimate > settings.input_budget:
            raise ReduceBudgetError("segmented Reduce input still exceeds final prompt budget")

    raw_report = await llm.chat(
        messages,
        temperature=float(final_prompt_package.get("temperature", 0.0)),
        max_output_tokens=settings.max_output_tokens,
    )
    llm_call_count += 1
    try:
        report = validate_landscape_report(
            parse_landscape_report_json(raw_report),
            profile_id=profile_id,
            metrics=frozen_metrics,
            snapshots=snapshots,
            forecast_horizon_days=forecast_horizon_days,
        )
    except ReportValidationError as error:
        if settings.format_repair_attempts < 1:
            raise
        repair_attempted = True
        repaired = await _repair_json(
            llm,
            raw=raw_report,
            error=error,
            output_schema=final_prompt_package["output_schema"],
            identity={
                "profile_id": profile_id,
                "period_end_ts": period_end_ts,
                "cluster_ids": sorted(str(row["cluster_id"]) for row in snapshots),
            },
            max_output_tokens=settings.max_output_tokens,
        )
        llm_call_count += 1
        report = validate_landscape_report(
            parse_landscape_report_json(repaired),
            profile_id=profile_id,
            metrics=frozen_metrics,
            snapshots=snapshots,
            forecast_horizon_days=forecast_horizon_days,
        )

    prompt_version = _non_empty(
        final_prompt_package.get("prompt_version"), "final_prompt.prompt_version"
    )
    document = build_landscape_report_document(
        report,
        metrics=frozen_metrics,
        snapshots=snapshots,
        prompt_version=prompt_version,
        model=model,
        created_at=now_ts,
        previous_report_id=previous_report_id,
        segment_prompt_version=(
            _non_empty(
                segment_prompt_package.get("prompt_version"),
                "segment_prompt.prompt_version",
            )
            if segment_count
            else None
        ),
    )
    if lease_guard is not None:
        await lease_guard.ensure_owned()
    saved = store.save_threat_landscape_report(document)
    canonical = document
    if not saved:
        existing = store.get_threat_landscape_report(document["id"])
        if existing is None:
            raise RuntimeError(
                "landscape report persistence failed without an existing report"
            )
        canonical = existing
    mirror_action = (
        mirror_landscape_report(store, canonical)
        if mirror_to_summary_docs
        else "disabled"
    )
    return ReduceResult(
        action="saved" if saved else "existing",
        report_id=str(document["id"]),
        input_snapshot_ids=tuple(document["input_snapshot_ids"]),
        segment_count=segment_count,
        llm_call_count=llm_call_count,
        repair_attempted=repair_attempted,
        estimated_final_prompt_tokens=final_estimate,
        mirror_action=mirror_action,
    )
