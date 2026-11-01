# LICENSE HEADER MANAGED BY add-license-header
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Martin Vesterlund

"""Identity helpers shared by online clustering and reconciliation."""

from __future__ import annotations

import re
from typing import Any, Iterable

CVE_PATTERN = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
_STRICT_CVE_TITLE_PATTERN = re.compile(
    r"^\s*(CVE-\d{4}-\d{4,7})\s+(?:[-\u2013\u2014:]\s*)\S.+$",
    re.IGNORECASE,
)


def normalized_cves(values: Iterable[str]) -> frozenset[str]:
    """Return canonical CVE identifiers found in arbitrary strings."""

    return frozenset(
        match.group(0).casefold()
        for value in values
        for match in CVE_PATTERN.finditer(str(value or ""))
    )


def is_strict_cve_record(article: dict[str, Any]) -> bool:
    """Identify a single-CVE catalogue record, not a narrative incident article.

    The deliberately narrow test keeps disjoint CVEs as a hard boundary only when
    the title has catalogue-record form and the whole article names no other CVE.
    """

    title = str(article.get("title") or "")
    match = _STRICT_CVE_TITLE_PATTERN.match(title)
    if match is None:
        return False
    text_values = [
        title,
        str(article.get("text") or ""),
        str(article.get("content") or ""),
        str(article.get("summary") or ""),
    ]
    return normalized_cves(text_values) == frozenset({match.group(1).casefold()})


def membership_is_strict_cve_record(membership: dict[str, Any]) -> bool:
    """Classify persisted evidence, including documents written before the flag."""

    explicit = membership.get("strict_cve_identity")
    if explicit is not None:
        return bool(explicit)
    evidence = membership.get("evidence") or {}
    title = str(evidence.get("title") or "")
    match = _STRICT_CVE_TITLE_PATTERN.match(title)
    if match is None:
        return False
    cves = normalized_cves(membership.get("strong_indicators") or ())
    return cves == frozenset({match.group(1).casefold()})
