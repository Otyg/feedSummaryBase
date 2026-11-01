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

"""Strict Reduce-report contracts and deterministic Markdown rendering."""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "profile_id",
        "period_start_ts",
        "period_end_ts",
        "executive_summary",
        "observations",
        "changes",
        "alternative_explanations",
        "data_gaps",
        "forecast",
    }
)
_ASSESSMENT_FIELDS = frozenset({"statement", "confidence", "evidence_cluster_ids"})
_CHANGE_FIELDS = frozenset(
    {"statement", "direction", "confidence", "evidence_cluster_ids"}
)
_FORECAST_FIELDS = frozenset(
    {
        "hypothesis",
        "horizon_days",
        "confidence",
        "rationale",
        "evidence_cluster_ids",
        "leading_indicators",
        "invalidation_conditions",
    }
)
_CONFIDENCE = frozenset({"low", "medium", "high"})
_FORECAST_CONFIDENCE = frozenset({"low", "medium"})
_DIRECTIONS = frozenset({"increasing", "decreasing", "shifting", "stable", "uncertain"})


class ReportValidationError(ValueError):
    """The Reduce response cannot safely become a threat-landscape report."""


def parse_landscape_report_json(raw: str) -> dict[str, Any]:
    """Parse exactly one bare JSON object without accepting Markdown wrappers."""

    if not isinstance(raw, str) or not raw.strip():
        raise ReportValidationError("report response is empty")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReportValidationError(f"report is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ReportValidationError("report must be a JSON object")
    return payload


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReportValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _exact_fields(item: dict[str, Any], expected: frozenset[str], field: str) -> None:
    unknown = set(item).difference(expected)
    missing = expected.difference(item)
    if unknown:
        raise ReportValidationError(f"{field} contains unknown fields: {sorted(unknown)}")
    if missing:
        raise ReportValidationError(f"{field} is missing fields: {sorted(missing)}")


def _evidence_ids(
    value: Any,
    *,
    field: str,
    allowed_cluster_ids: frozenset[str],
    minimum: int,
) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ReportValidationError(f"{field} must contain at least {minimum} cluster IDs")
    values = [_non_empty_string(item, field) for item in value]
    if len(values) != len(set(values)):
        raise ReportValidationError(f"{field} contains duplicate cluster IDs")
    unknown = set(values).difference(allowed_cluster_ids)
    if unknown:
        raise ReportValidationError(f"{field} contains unknown cluster IDs: {sorted(unknown)}")
    return values


def _string_list(value: Any, field: str, *, minimum: int = 0) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ReportValidationError(f"{field} must contain at least {minimum} values")
    return [_non_empty_string(item, f"{field}[{index}]") for index, item in enumerate(value)]


def _assessment_items(
    value: Any,
    *,
    field: str,
    allowed_cluster_ids: frozenset[str],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ReportValidationError(f"{field} must be an array")
    result = []
    for index, raw in enumerate(value):
        item_field = f"{field}[{index}]"
        if not isinstance(raw, dict):
            raise ReportValidationError(f"{item_field} must be an object")
        _exact_fields(raw, _ASSESSMENT_FIELDS, item_field)
        confidence = raw.get("confidence")
        if confidence not in _CONFIDENCE:
            raise ReportValidationError(f"{item_field}.confidence is invalid")
        result.append(
            {
                "statement": _non_empty_string(raw.get("statement"), f"{item_field}.statement"),
                "confidence": confidence,
                "evidence_cluster_ids": _evidence_ids(
                    raw.get("evidence_cluster_ids"),
                    field=f"{item_field}.evidence_cluster_ids",
                    allowed_cluster_ids=allowed_cluster_ids,
                    minimum=1,
                ),
            }
        )
    return result


def _metric_contract(metrics: dict[str, Any], profile_id: str) -> tuple[int, int, bool, int]:
    if not isinstance(metrics, dict) or metrics.get("profile_id") != profile_id:
        raise ReportValidationError("metrics do not match the report profile")
    period_end_ts = int(metrics.get("period_end_ts") or 0)
    windows = metrics.get("windows")
    if period_end_ts < 1 or not isinstance(windows, list) or not windows:
        raise ReportValidationError("metrics have no valid analysis windows")
    starts = [int(window.get("period_start_ts") or 0) for window in windows if isinstance(window, dict)]
    if len(starts) != len(windows) or any(value < 1 for value in starts):
        raise ReportValidationError("metrics contain an invalid analysis period")
    trend_eligible = any(
        isinstance(window.get("coverage"), dict)
        and window["coverage"].get("trend_eligible") is True
        for window in windows
        if isinstance(window, dict)
    )
    settings = metrics.get("settings")
    settings = settings if isinstance(settings, dict) else {}
    minimum = int(settings.get("min_independent_events_for_trend") or 3)
    if minimum < 1:
        raise ReportValidationError("metrics contain an invalid event threshold")
    return min(starts), period_end_ts, trend_eligible, minimum


def _coverage_warnings(metrics: dict[str, Any]) -> list[str]:
    warnings = set()
    quality = metrics.get("quality")
    if isinstance(quality, dict):
        warnings.update(str(value) for value in quality.get("warning_codes") or [] if str(value))
    for window in metrics.get("windows") or []:
        coverage = window.get("coverage") if isinstance(window, dict) else None
        if isinstance(coverage, dict):
            warnings.update(str(value) for value in coverage.get("warnings") or [] if str(value))
    return sorted(warnings)


def _evidence_spans_time_buckets(
    cluster_ids: list[str], metrics: dict[str, Any]
) -> bool:
    evidence = set(cluster_ids)
    for window in metrics.get("windows") or []:
        if not isinstance(window, dict):
            continue
        coverage = window.get("coverage")
        if not isinstance(coverage, dict) or coverage.get("trend_eligible") is not True:
            continue
        matching_buckets = 0
        for bucket in window.get("weekly_buckets") or []:
            if isinstance(bucket, dict) and evidence.intersection(
                str(value) for value in bucket.get("event_cluster_ids") or []
            ):
                matching_buckets += 1
        if matching_buckets >= 2:
            return True
    return False


def _allowed_clusters(snapshots: list[dict[str, Any]], profile_id: str) -> frozenset[str]:
    allowed = set()
    snapshot_ids = set()
    for snapshot in snapshots:
        if not isinstance(snapshot, dict) or snapshot.get("profile_id") != profile_id:
            raise ReportValidationError("snapshot does not match the report profile")
        snapshot_id = _non_empty_string(snapshot.get("id"), "snapshot.id")
        cluster_id = _non_empty_string(snapshot.get("cluster_id"), f"{snapshot_id}.cluster_id")
        if snapshot_id in snapshot_ids or cluster_id in allowed:
            raise ReportValidationError("report input contains duplicate snapshot identities")
        payload = snapshot.get("payload")
        if not isinstance(payload, dict):
            raise ReportValidationError(f"{snapshot_id}.payload must be an object")
        if payload.get("profile_id") != profile_id or payload.get("cluster_id") != cluster_id:
            raise ReportValidationError(f"{snapshot_id}.payload identity does not match")
        snapshot_ids.add(snapshot_id)
        allowed.add(cluster_id)
    return frozenset(allowed)


def validate_landscape_report(
    payload: dict[str, Any],
    *,
    profile_id: str,
    metrics: dict[str, Any],
    snapshots: list[dict[str, Any]],
    forecast_horizon_days: frozenset[int] = frozenset({30, 90}),
) -> dict[str, Any]:
    """Validate identity, evidence and conservative trend/forecast gates."""

    if not isinstance(payload, dict):
        raise ReportValidationError("report must be an object")
    _exact_fields(payload, _REPORT_FIELDS, "report")
    period_start_ts, period_end_ts, trend_eligible, minimum_events = _metric_contract(
        metrics, profile_id
    )
    if payload.get("schema_version") != 1 or payload.get("profile_id") != profile_id:
        raise ReportValidationError("report schema or profile identity does not match")
    if (
        payload.get("period_start_ts") != period_start_ts
        or payload.get("period_end_ts") != period_end_ts
    ):
        raise ReportValidationError("report period does not match its frozen metrics")
    allowed_cluster_ids = _allowed_clusters(snapshots, profile_id)
    result = dict(payload)
    result["executive_summary"] = _non_empty_string(
        payload.get("executive_summary"), "executive_summary"
    )
    result["observations"] = _assessment_items(
        payload.get("observations"),
        field="observations",
        allowed_cluster_ids=allowed_cluster_ids,
    )
    result["alternative_explanations"] = _assessment_items(
        payload.get("alternative_explanations"),
        field="alternative_explanations",
        allowed_cluster_ids=allowed_cluster_ids,
    )

    raw_changes = payload.get("changes")
    if not isinstance(raw_changes, list):
        raise ReportValidationError("changes must be an array")
    if raw_changes and not trend_eligible:
        raise ReportValidationError("changes are forbidden when no metric window is trend eligible")
    changes = []
    for index, raw in enumerate(raw_changes):
        field = f"changes[{index}]"
        if not isinstance(raw, dict):
            raise ReportValidationError(f"{field} must be an object")
        _exact_fields(raw, _CHANGE_FIELDS, field)
        if raw.get("direction") not in _DIRECTIONS:
            raise ReportValidationError(f"{field}.direction is invalid")
        if raw.get("confidence") not in _CONFIDENCE:
            raise ReportValidationError(f"{field}.confidence is invalid")
        evidence_cluster_ids = _evidence_ids(
            raw.get("evidence_cluster_ids"),
            field=f"{field}.evidence_cluster_ids",
            allowed_cluster_ids=allowed_cluster_ids,
            minimum=minimum_events,
        )
        if not _evidence_spans_time_buckets(evidence_cluster_ids, metrics):
            raise ReportValidationError(f"{field} evidence does not span two time buckets")
        changes.append(
            {
                "statement": _non_empty_string(raw.get("statement"), f"{field}.statement"),
                "direction": raw["direction"],
                "confidence": raw["confidence"],
                "evidence_cluster_ids": evidence_cluster_ids,
            }
        )
    result["changes"] = changes

    data_gaps = _string_list(payload.get("data_gaps"), "data_gaps")
    if _coverage_warnings(metrics) and not data_gaps:
        raise ReportValidationError("coverage warnings require at least one data gap")
    result["data_gaps"] = data_gaps

    raw_forecast = payload.get("forecast")
    if not isinstance(raw_forecast, list) or len(raw_forecast) > 3:
        raise ReportValidationError("forecast must be an array with at most three items")
    if raw_forecast and not trend_eligible:
        raise ReportValidationError("forecast is forbidden when no metric window is trend eligible")
    forecast = []
    for index, raw in enumerate(raw_forecast):
        field = f"forecast[{index}]"
        if not isinstance(raw, dict):
            raise ReportValidationError(f"{field} must be an object")
        _exact_fields(raw, _FORECAST_FIELDS, field)
        if raw.get("horizon_days") not in forecast_horizon_days:
            raise ReportValidationError(f"{field}.horizon_days is not configured")
        if raw.get("confidence") not in _FORECAST_CONFIDENCE:
            raise ReportValidationError(f"{field}.confidence must be low or medium")
        evidence_cluster_ids = _evidence_ids(
            raw.get("evidence_cluster_ids"),
            field=f"{field}.evidence_cluster_ids",
            allowed_cluster_ids=allowed_cluster_ids,
            minimum=minimum_events,
        )
        if not _evidence_spans_time_buckets(evidence_cluster_ids, metrics):
            raise ReportValidationError(f"{field} evidence does not span two time buckets")
        forecast.append(
            {
                "hypothesis": _non_empty_string(raw.get("hypothesis"), f"{field}.hypothesis"),
                "horizon_days": raw["horizon_days"],
                "confidence": raw["confidence"],
                "rationale": _non_empty_string(raw.get("rationale"), f"{field}.rationale"),
                "evidence_cluster_ids": evidence_cluster_ids,
                "leading_indicators": _string_list(
                    raw.get("leading_indicators"), f"{field}.leading_indicators", minimum=1
                ),
                "invalidation_conditions": _string_list(
                    raw.get("invalidation_conditions"),
                    f"{field}.invalidation_conditions",
                    minimum=1,
                ),
            }
        )
    result["forecast"] = forecast
    return result


def render_landscape_report_messages(
    prompt_package: dict[str, Any],
    *,
    profile_context: dict[str, Any],
    metrics: dict[str, Any],
    snapshots: list[dict[str, Any]],
    previous_report: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Render deterministic Reduce messages from frozen, structured inputs."""

    system = _non_empty_string(prompt_package.get("system"), "prompt.system")
    template = _non_empty_string(prompt_package.get("user_template"), "prompt.user_template")
    schema = prompt_package.get("output_schema")
    if not isinstance(schema, dict):
        raise ReportValidationError("prompt.output_schema must be an object")
    ordered_snapshots = sorted(
        snapshots,
        key=lambda row: (str(row.get("cluster_id") or ""), str(row.get("id") or "")),
    )
    starts = [
        int(window.get("period_start_ts") or 0)
        for window in metrics.get("windows") or []
        if isinstance(window, dict)
    ]
    period = {
        "period_start_ts": min(starts) if starts else 0,
        "period_end_ts": int(metrics.get("period_end_ts") or 0),
    }
    values = {
        "profile_context": profile_context,
        "analysis_period": period,
        "metrics": metrics,
        "coverage_warnings": _coverage_warnings(metrics),
        "cluster_snapshots": ordered_snapshots,
        "previous_report": previous_report,
    }
    serialized = {
        key: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for key, value in values.items()
    }
    user = template.format(**serialized)
    user += "\n\nOUTPUT_SCHEMA:\n" + json.dumps(
        schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _markdown_text(value: str) -> str:
    text = " ".join(str(value).split())
    for character in ("\\", "`", "*", "_", "[", "]", "<", ">", "#"):
        text = text.replace(character, f"\\{character}")
    return text


def _evidence_text(cluster_ids: list[str]) -> str:
    return ", ".join(f"`{value.replace('`', '')}`" for value in cluster_ids)


def _date(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()


def render_landscape_markdown(
    report: dict[str, Any], *, metrics: dict[str, Any]
) -> str:
    """Render validated structured analysis without asking an LLM for Markdown."""

    lines = [
        "# Långtidsanalys av hotlandskapet",
        "",
        f"Period: {_date(report['period_start_ts'])}–{_date(report['period_end_ts'])} (UTC)",
        "",
        "## Sammanfattning",
        "",
        _markdown_text(report["executive_summary"]),
    ]
    sections = (
        ("Observationer", "observations", None),
        ("Förändringar", "changes", "direction"),
        ("Alternativa förklaringar", "alternative_explanations", None),
    )
    for title, key, extra in sections:
        lines.extend(["", f"## {title}", ""])
        if not report[key]:
            lines.append("- Inga belagda poster.")
            continue
        for item in report[key]:
            metadata = [f"säkerhet: {item['confidence']}"]
            if extra:
                metadata.append(f"riktning: {item[extra]}")
            metadata.append(f"evidens: {_evidence_text(item['evidence_cluster_ids'])}")
            lines.append(f"- {_markdown_text(item['statement'])} _({'; '.join(metadata)})_")
    lines.extend(["", "## Dataluckor", ""])
    lines.extend(
        [f"- {_markdown_text(value)}" for value in report["data_gaps"]]
        or ["- Inga uttryckliga dataluckor."]
    )
    lines.extend(["", "## Prognoshypoteser", ""])
    if not report["forecast"]:
        lines.append("- Inga prognoshypoteser med tillräckligt underlag.")
    for item in report["forecast"]:
        lines.extend(
            [
                f"### {_markdown_text(item['hypothesis'])}",
                "",
                f"Horisont: {item['horizon_days']} dagar. Säkerhet: {item['confidence']}.",
                "",
                _markdown_text(item["rationale"]),
                "",
                f"Evidens: {_evidence_text(item['evidence_cluster_ids'])}",
                "",
                "Indikatorer att bevaka:",
                *[f"- {_markdown_text(value)}" for value in item["leading_indicators"]],
                "",
                "Villkor som talar emot hypotesen:",
                *[f"- {_markdown_text(value)}" for value in item["invalidation_conditions"]],
            ]
        )
    warnings = _coverage_warnings(metrics)
    lines.extend(["", "## Kvalitetsvarningar", ""])
    lines.extend([f"- `{value}`" for value in warnings] or ["- Inga maskinella varningar."])
    return "\n".join(lines).rstrip() + "\n"


def build_landscape_report_document(
    report: dict[str, Any],
    *,
    metrics: dict[str, Any],
    snapshots: list[dict[str, Any]],
    prompt_version: str,
    model: str,
    created_at: int,
    previous_report_id: str | None = None,
    segment_prompt_version: str | None = None,
) -> dict[str, Any]:
    """Build the immutable persistence document after successful validation."""

    prompt_version = _non_empty_string(prompt_version, "prompt_version")
    model = _non_empty_string(model, "model")
    segment_prompt_version = (
        _non_empty_string(segment_prompt_version, "segment_prompt_version")
        if segment_prompt_version is not None
        else None
    )
    if created_at < 1:
        raise ReportValidationError("created_at must be positive")
    identity = (
        f"feedsummary:landscape-report:{report['profile_id']}:"
        f"{report['period_end_ts']}:{prompt_version}:"
        f"{segment_prompt_version or 'direct'}:{model}"
    )
    snapshot_ids = sorted(str(snapshot["id"]) for snapshot in snapshots)
    input_signature = hashlib.sha256(
        json.dumps(
            {"metrics": metrics, "snapshot_ids": snapshot_ids},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    report_id = f"threat_report_{uuid.uuid5(uuid.NAMESPACE_URL, f'{identity}:{input_signature}').hex}"
    markdown = render_landscape_markdown(report, metrics=metrics)
    return {
        "id": report_id,
        "profile_id": report["profile_id"],
        "period_start_ts": report["period_start_ts"],
        "period_end_ts": report["period_end_ts"],
        "created_at": int(created_at),
        "status": "published",
        "previous_report_id": previous_report_id,
        "input_snapshot_ids": snapshot_ids,
        "input_signature": input_signature,
        "metrics": metrics,
        "analysis": {key: value for key, value in report.items() if key != "forecast"},
        "forecast": report["forecast"],
        "markdown": markdown,
        "quality": {
            "validation": "passed",
            "warning_codes": _coverage_warnings(metrics),
        },
        "prompt_version": prompt_version,
        "segment_prompt_version": segment_prompt_version,
        "model": model,
    }


def build_landscape_summary_document(
    report_document: dict[str, Any],
) -> dict[str, Any]:
    """Build a deterministic UI mirror of one immutable published report."""

    if not isinstance(report_document, dict):
        raise ReportValidationError("report_document must be an object")
    report_id = _non_empty_string(report_document.get("id"), "report_document.id")
    profile_id = _non_empty_string(
        report_document.get("profile_id"), "report_document.profile_id"
    )
    if report_document.get("status") != "published":
        raise ReportValidationError("only published landscape reports may be mirrored")
    markdown = report_document.get("markdown")
    _non_empty_string(markdown, "report_document.markdown")
    created = int(report_document.get("created_at") or 0)
    period_start_ts = int(report_document.get("period_start_ts") or 0)
    period_end_ts = int(report_document.get("period_end_ts") or 0)
    if min(created, period_start_ts, period_end_ts) < 1:
        raise ReportValidationError("report mirror timestamps must be positive")
    if period_start_ts > period_end_ts:
        raise ReportValidationError("report mirror period is reversed")

    return {
        "id": f"summary_{report_id}",
        "created": created,
        "kind": "threat_landscape",
        "title": f"Långtidsanalys av hotlandskapet – {profile_id}",
        "summary": markdown,
        "source_report_id": report_id,
        "selection": {
            "prompt_package": "long_term/landscape_report",
            "profile_id": profile_id,
            "period_start_ts": period_start_ts,
            "period_end_ts": period_end_ts,
        },
        "meta": {
            "mirror_schema_version": 1,
            "authoritative_collection": "threat_landscape_reports",
            "authoritative_id": report_id,
            "quality": copy.deepcopy(report_document.get("quality") or {}),
            "prompt_version": report_document.get("prompt_version"),
            "segment_prompt_version": report_document.get("segment_prompt_version"),
            "model": report_document.get("model"),
        },
    }
