# LICENSE HEADER MANAGED BY add-license-header
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Martin Vesterlund
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse, promote, or sell copies of
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

"""
Integration of tagging functionality with the summarization pipeline.

This module provides helper functions to integrate article tagging into
the summarization workflow.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from feedsummary_core.llm_client import LLMClient
from feedsummary_core.persistence import NewsStore
from feedsummary_core.summarizer.tagging import TagManager

logger = logging.getLogger(__name__)


async def tag_articles(
    store: NewsStore,
    llm_client: LLMClient,
    article_ids: List[str],
    config: Dict[str, Any],
    max_tags_per_article: int = 5,
    skip_if_already_tagged: bool = True,
    enable_embedding_matching: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Tag multiple articles using the LLM and tag priority system with embedding support.

    Args:
        store: NewsStore instance
        llm_client: LLM client for generating tags and embeddings
        article_ids: List of article IDs to tag
        config: Configuration dictionary
        max_tags_per_article: Maximum tags per article
        skip_if_already_tagged: If True, skip articles that already have tags
        enable_embedding_matching: If True, use embedding-based tag matching when available

    Returns:
        Dict mapping article_id to list of assigned tags
    """
    # Pass llm_client to TagManager to enable embedding-based matching
    tag_manager = TagManager(store, llm_client=llm_client if enable_embedding_matching else None)
    results: Dict[str, List[Dict[str, Any]]] = {}

    for article_id in article_ids:
        try:
            # Check if already tagged
            if skip_if_already_tagged:
                existing_tags = store.get_article_tags(article_id)
                if existing_tags:
                    logger.info(f"Article {article_id} already tagged, skipping")
                    results[article_id] = existing_tags
                    continue

            # Get article
            article = store.get_article(article_id)
            if not article:
                logger.warning(f"Article {article_id} not found")
                continue

            # Generate tags
            tags = await tag_manager.generate_tags_for_article(
                llm_client=llm_client,
                article=article,
                config=config,
                max_tags=max_tags_per_article,
            )

            if tags:
                # Store tags with reasoning
                tag_entries = []
                for t in tags:
                    if "id" in t:
                        entry = {"tag_id": t["id"]}
                        # Include reasoning if present
                        if t.get("reasoning"):
                            entry["reasoning"] = t["reasoning"]
                        tag_entries.append(entry)
                
                store.add_article_tags(article_id, tag_entries)

                # Update article with tags
                article["tags"] = [t["name"] for t in tags]
                store.upsert_article(article)

                results[article_id] = tags
                logger.info(
                    f"Tagged article {article_id} with {len(tags)} tags: "
                    f"{', '.join(t['name'] for t in tags)}"
                )
            else:
                logger.warning(f"No tags generated for article {article_id}")
                results[article_id] = []

        except Exception as e:
            logger.error(f"Error tagging article {article_id}: {e}")
            results[article_id] = []

    return results


async def tag_article(
    store: NewsStore,
    llm_client: LLMClient,
    article_id: str,
    config: Dict[str, Any],
    max_tags: int = 5,
) -> List[Dict[str, Any]]:
    """
    Tag a single article.

    Args:
        store: NewsStore instance
        llm_client: LLM client
        article_id: Article ID to tag
        config: Configuration
        max_tags: Maximum tags

    Returns:
        List of assigned tags
    """
    result = await tag_articles(
        store=store,
        llm_client=llm_client,
        article_ids=[article_id],
        config=config,
        max_tags_per_article=max_tags,
        skip_if_already_tagged=False,
    )
    return result.get(article_id, [])


def get_article_tags_for_display(
    store: NewsStore,
    article_id: str,
) -> List[Dict[str, str]]:
    """
    Get article tags formatted for display.

    Args:
        store: NewsStore instance
        article_id: Article ID

    Returns:
        List of tags with 'name' and 'category' fields
    """
    tags = store.get_article_tags(article_id)
    return [
        {
            "name": t.get("name", ""),
            "category": t.get("category", "GENERAL"),
        }
        for t in tags
    ]


def add_predefined_tags(
    store: NewsStore,
    tags: List[Dict[str, str]],
) -> int:
    """
    Add a set of predefined tags to the database.

    Useful for bulk-loading common tags.

    Args:
        store: NewsStore instance
        tags: List of dicts with 'name', 'category' (optional), 'description' (optional)

    Returns:
        Number of tags added
    """
    tag_manager = TagManager(store)
    count = 0

    for tag in tags:
        if not isinstance(tag, dict) or "name" not in tag:
            continue

        tag_id = tag_manager.add_tag(
            name=tag["name"],
            category=tag.get("category", "GENERAL"),
            description=tag.get("description"),
        )
        if tag_id:
            count += 1
            logger.debug(f"Added tag: {tag['name']} (ID: {tag_id})")

    return count


def get_articles_by_tag(
    store: NewsStore,
    tag_name: str,
) -> List[str]:
    """
    Get all article IDs tagged with a specific tag.

    Note: This requires a SQL query not yet exposed by the store protocol.
    This is a placeholder for future enhancement.

    Args:
        store: NewsStore instance
        tag_name: Tag name to search for

    Returns:
        List of article IDs
    """
    logger.warning("get_articles_by_tag requires direct DB access - not yet implemented")
    return []


def cleanup_old_tags(
    store: NewsStore,
    days: int = 30,
) -> int:
    """
    Remove tags that haven't been used in X days.

    Args:
        store: NewsStore instance
        days: Number of days of inactivity

    Returns:
        Number of tags removed
    """
    return store.cleanup_unused_tags(days=days)


async def tag_articles_safe(
    store: NewsStore,
    llm_client: LLMClient,
    article_ids: List[str],
    config: Dict[str, Any],
    job_id: Optional[int] = None,
    max_tags_per_article: int = 5,
) -> int:
    """
    Safely tag articles, ignoring errors.

    This function is designed to be called automatically during the
    summarization pipeline and won't interrupt the pipeline if tagging fails.

    Args:
        store: NewsStore instance
        llm_client: LLM client
        article_ids: List of article IDs to tag
        config: Configuration dictionary
        job_id: Optional job ID for logging
        max_tags_per_article: Maximum tags per article

    Returns:
        Number of articles successfully tagged
    """
    if not article_ids:
        return 0

    logger.info(
        f"[Job {job_id}] Starting automatic tagging of {len(article_ids)} articles..."
    )

    try:
        result = await tag_articles(
            store=store,
            llm_client=llm_client,
            article_ids=article_ids,
            config=config,
            max_tags_per_article=max_tags_per_article,
            skip_if_already_tagged=True,
        )

        success_count = sum(1 for tags in result.values() if tags)
        logger.info(
            f"[Job {job_id}] Tagging complete: {success_count}/{len(article_ids)} articles tagged"
        )
        return success_count

    except Exception as e:
        logger.error(f"[Job {job_id}] Error during automatic tagging: {e}", exc_info=True)
        return 0


async def generate_summary_from_tags(
    store: NewsStore,
    llm_client: LLMClient,
    tag_names: List[str],
    config: Dict[str, Any],
    match_mode: str = "any",
) -> Optional[Dict[str, Any]]:
    """
    Generate a summary from articles tagged with specified tags.

    Args:
        store: NewsStore instance
        llm_client: LLM client for summarization
        tag_names: List of tag names to search for
        config: Configuration dictionary
        match_mode: "any" (OR) or "all" (AND)

    Returns:
        Summary dict or None if error
    """
    try:
        logger.info(f"Fetching articles tagged with: {tag_names} (mode: {match_mode})")

        # Get articles with these tags
        articles = store.get_articles_by_tags(
            tag_names=tag_names,
            match_mode=match_mode,
        )

        if not articles:
            logger.warning(f"No articles found with tags: {tag_names}")
            return None

        logger.info(f"Found {len(articles)} articles with tags: {tag_names}")

        # Import here to avoid circular dependency
        from feedsummary_core.summarizer.summarizer import summarize_batches_then_meta_with_stats

        # Generate summary
        meta_text, stats = await summarize_batches_then_meta_with_stats(
            config, articles, llm=llm_client, store=store, job_id=None
        )

        if not meta_text:
            logger.warning("Failed to generate summary from articles")
            return None

        # Build summary document
        import time
        from datetime import datetime

        created_ts = int(time.time())
        dt = datetime.fromtimestamp(created_ts)
        summary_doc_id = dt.strftime("tag_sum_%Y%m%d_%H%M")

        pts = [int(a.get("published_ts", a.get("fetched_at", 0))) for a in articles]
        pts_valid = [p for p in pts if p > 0]
        from_ts = min(pts_valid) if pts_valid else 0
        to_ts = max(pts_valid) if pts_valid else 0

        summary_doc = {
            "id": summary_doc_id,
            "title": f"Summary: {', '.join(tag_names)}",
            "created": created_ts,
            "kind": "tag-based-summary",
            "tags_used": tag_names,
            "match_mode": match_mode,
            "article_count": len(articles),
            "source_article_ids": [a.get("id") for a in articles],
            "summary": meta_text,
            "from": from_ts,
            "to": to_ts,
            "meta": {
                "batch_total": int(stats.get("batch_total") or 0),
                "trims": int(stats.get("trims") or 0),
                "drops": int(stats.get("drops") or 0),
            },
        }

        logger.info(f"Generated summary for tags {tag_names} (ID: {summary_doc_id})")
        return summary_doc

    except Exception as e:
        logger.error(f"Error generating summary from tags: {e}", exc_info=True)
        return None
