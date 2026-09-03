"""Shared validation for relations between tags."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

PARENT_CHILD_RELATION = "parent_child"


class TagRelationError(ValueError):
    """Raised when a requested tag relation would violate an invariant."""


def normalize_tag_ids(values: Iterable[Any] | None) -> set[int] | None:
    """Normalize an optional list of positive tag identifiers."""
    if values is None:
        return None
    if isinstance(values, (str, bytes, Mapping)):
        raise TagRelationError("tag relations must be supplied as lists of tag IDs")

    normalized: set[int] = set()
    for value in values:
        if isinstance(value, bool):
            raise TagRelationError("tag relation IDs must be positive integers")
        try:
            tag_id = int(value)
        except (TypeError, ValueError) as exc:
            raise TagRelationError("tag relation IDs must be positive integers") from exc
        if tag_id <= 0:
            raise TagRelationError("tag relation IDs must be positive integers")
        normalized.add(tag_id)
    return normalized


def proposed_parent_child_edges(
    tag_id: int,
    *,
    parent_ids: Iterable[Any] | None,
    child_ids: Iterable[Any] | None,
    tags_by_id: Mapping[int, Mapping[str, Any]],
    existing_edges: Iterable[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Build and validate the complete edge set after replacing either side of a tag."""
    tag_id = int(tag_id)
    if tag_id not in tags_by_id:
        raise TagRelationError(f"tag not found: {tag_id}")

    parents = normalize_tag_ids(parent_ids)
    children = normalize_tag_ids(child_ids)
    if parents is None and children is None:
        raise TagRelationError("at least one of parents or children must be supplied")

    related_ids = (parents or set()) | (children or set())
    if tag_id in related_ids:
        raise TagRelationError("a tag cannot be related to itself")

    missing = sorted(related_ids - set(tags_by_id))
    if missing:
        raise TagRelationError(f"related tag not found: {missing[0]}")

    category = str(tags_by_id[tag_id].get("category") or "GENERAL")
    cross_category = sorted(
        related_id
        for related_id in related_ids
        if str(tags_by_id[related_id].get("category") or "GENERAL") != category
    )
    if cross_category:
        raise TagRelationError("parent-child relations cannot cross categories")

    edges = {(int(parent), int(child)) for parent, child in existing_edges}
    if parents is not None:
        edges = {edge for edge in edges if edge[1] != tag_id}
        edges.update((parent_id, tag_id) for parent_id in parents)
    if children is not None:
        edges = {edge for edge in edges if edge[0] != tag_id}
        edges.update((tag_id, child_id) for child_id in children)

    _validate_acyclic(edges)
    return edges


def _validate_acyclic(edges: Iterable[tuple[int, int]]) -> None:
    children_by_parent: dict[int, set[int]] = {}
    nodes: set[int] = set()
    for parent_id, child_id in edges:
        nodes.update((parent_id, child_id))
        children_by_parent.setdefault(parent_id, set()).add(child_id)

    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(tag_id: int) -> None:
        if tag_id in visiting:
            raise TagRelationError("parent-child relations cannot contain cycles")
        if tag_id in visited:
            return
        visiting.add(tag_id)
        for child_id in children_by_parent.get(tag_id, ()):
            visit(child_id)
        visiting.remove(tag_id)
        visited.add(tag_id)

    for tag_id in nodes:
        visit(tag_id)
