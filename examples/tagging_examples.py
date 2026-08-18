#!/usr/bin/env python3
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
Example usage of the tagging system.

This script demonstrates how to use the tagging functionality to:
1. Add predefined tags
2. Generate tags for articles using LLM
3. Retrieve and display tags
4. Clean up unused tags
"""

import asyncio
import logging
from typing import Any, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def example_basic_tagging():
    """Example 1: Basic tagging workflow"""
    logger.info("=== Example 1: Basic Tagging Workflow ===")

    from feedsummary_core.persistence import create_store
    from feedsummary_core.summarizer.tagging import TagManager

    # Initialize store
    store = create_store("sqlite://example_tags.db")
    tag_manager = TagManager(store)

    # Add some predefined tags
    logger.info("Adding predefined tags...")
    tag_manager.add_tag("cybersecurity", category="GENERAL")
    tag_manager.add_tag("vulnerability", category="GENERAL")
    tag_manager.add_tag("malware", category="GENERAL")
    tag_manager.add_tag("threat-intelligence", category="GENERAL")

    # Get all tags
    all_tags = tag_manager.get_all_tags()
    logger.info(f"Total tags in database: {len(all_tags)}")
    for tag in all_tags:
        logger.info(f"  - {tag['name']} ({tag['category']})")


async def example_tag_selection_logic():
    """Example 2: Demonstrate tag selection priority logic"""
    logger.info("\n=== Example 2: Tag Selection Priority Logic ===")

    from feedsummary_core.persistence import create_store
    from feedsummary_core.summarizer.tagging import TagManager

    store = create_store("sqlite://example_tags.db")
    tag_manager = TagManager(store)

    # Add some existing tags
    tag_manager.add_tag("security", category="GENERAL")
    tag_manager.add_tag("network-security", category="GENERAL")
    tag_manager.add_tag("CVE-2024-12345", category="VULNERABILITY")

    # Demonstrate tag selection
    candidates = [
        "security",  # Should match existing "security"
        "web security",  # Should match "security" (substring)
        "network attack",  # Should match "network-security"
        "CVE-2024-54321",  # Should create new VULNERABILITY
        "random-term",  # Should skip (not domain entity, not existing)
    ]

    logger.info(f"Candidate tags: {candidates}")
    selected = tag_manager.select_tags_for_article(
        article_id="example_1",
        candidate_tags=candidates,
        allow_new_tags=True,
    )

    logger.info(f"Selected tags:")
    for tag in selected:
        logger.info(f"  - {tag['name']} (ID: {tag['id']}, Category: {tag['category']})")


async def example_article_tagging():
    """Example 3: Tag a sample article"""
    logger.info("\n=== Example 3: Article Tagging ===")

    from feedsummary_core.persistence import create_store
    from feedsummary_core.summarizer.tagging import TagManager

    store = create_store("sqlite://example_tags.db")
    tag_manager = TagManager(store)

    # Create a sample article
    sample_article = {
        "id": "article_001",
        "title": "Critical CVE-2024-50123 Affects Windows Systems",
        "content": """
        A critical vulnerability (CVE-2024-50123) has been discovered
        in Windows. The threat actor known as APT28 has been seen
        exploiting this in targeted attacks in Eastern Europe.
        This vulnerability allows remote code execution and affects
        Windows 10 and later versions.
        """,
        "source": "security-news.com",
        "published_ts": 1692374400,
    }

    logger.info(f"Sample article: {sample_article['title']}")

    # Manual tag selection (simulating LLM extraction)
    candidate_tags = [
        "vulnerability",
        "windows",
        "CVE-2024-50123",
        "APT28",
        "Eastern Europe",
        "remote-code-execution",
    ]

    selected_tags = tag_manager.select_tags_for_article(
        article_id=sample_article["id"],
        candidate_tags=candidate_tags,
        allow_new_tags=True,
    )

    logger.info(f"Generated {len(selected_tags)} tags:")
    for tag in selected_tags:
        logger.info(f"  - {tag['name']} ({tag['category']})")

    # Store article with tags in database
    store.upsert_article(sample_article)
    if selected_tags:
        tag_ids = [t["id"] for t in selected_tags]
        store.add_article_tags(sample_article["id"], tag_ids)

        # Retrieve tags
        retrieved_tags = store.get_article_tags(sample_article["id"])
        logger.info(f"Stored {len(retrieved_tags)} tags for article {sample_article['id']}")


async def example_domain_entity_detection():
    """Example 4: Domain entity detection"""
    logger.info("\n=== Example 4: Domain Entity Detection ===")

    from feedsummary_core.summarizer.tagging import TagManager
    from feedsummary_core.persistence import create_store

    store = create_store("sqlite://example_tags.db")
    tag_manager = TagManager(store)

    test_cases = [
        ("CVE-2024-12345", True, "CVE pattern"),
        ("APT-28 campaign", True, "Threat actor"),
        ("Russia-based attacker", True, "Region"),
        ("zero-day exploit", True, "Vulnerability keyword"),
        ("general security update", False, "General tag"),
        ("patch management", False, "General tag"),
    ]

    logger.info("Testing domain entity detection:")
    for text, expected_entity, reason in test_cases:
        is_entity = tag_manager._is_domain_entity(text)
        status = "✓" if is_entity == expected_entity else "✗"
        logger.info(f"  {status} '{text}' -> {is_entity} ({reason})")


async def example_similarity_matching():
    """Example 5: Tag similarity matching"""
    logger.info("\n=== Example 5: Tag Similarity Matching ===")

    from feedsummary_core.persistence import create_store
    from feedsummary_core.summarizer.tagging import TagManager

    store = create_store("sqlite://example_tags.db")
    tag_manager = TagManager(store)

    # Add some tags
    tag_manager.add_tag("cybersecurity", category="GENERAL")
    tag_manager.add_tag("network-security", category="GENERAL")
    tag_manager.add_tag("application-security", category="GENERAL")

    # Find similar tags
    queries = [
        "cyber",
        "security",
        "network",
        "application",
        "physical-security",
    ]

    logger.info("Finding similar tags:")
    for query in queries:
        similar = tag_manager._find_similar_existing_tags(query, similarity_threshold=0.5)
        logger.info(f"  '{query}': {[t['name'] for t in similar]}")


async def main():
    """Run all examples"""
    try:
        await example_basic_tagging()
        await example_tag_selection_logic()
        await example_article_tagging()
        await example_domain_entity_detection()
        await example_similarity_matching()

        logger.info("\n✓ All examples completed successfully!")

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
