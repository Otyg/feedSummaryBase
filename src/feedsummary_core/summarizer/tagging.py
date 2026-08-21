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
Tagging system for articles.

Provides functionality to:
- Store tags in a database
- Extract and assign tags to articles
- Prioritize existing tags over new ones
- Prefer general tags over specific ones
- Allow creating new tags for relevant entities and categories:
  * Domain entities (threat actors, regions)
  * Vulnerability identifiers (CVEs)
  * Broad categories (data protection, ransomware, security)
  * Named entities (companies, products, people, locations)
  * Multi-word phrases that indicate specific topics
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from feedsummary_core.llm_client import LLMClient, get_client_embedding_model
from feedsummary_core.persistence import NewsStore
from feedsummary_core.summarizer.batching import (
    cached_embedding,
    embedding_source_hash,
    group_articles_by_similarity,
)
from feedsummary_core.tagging_rules import (
    CVE_PATTERN,
    VULNERABILITY_TAG_CATEGORY,
    extract_cve_ids,
    is_cve_tag,
)

logger = logging.getLogger(__name__)

# Patterns and keywords for relevant tags that can be created automatically
# These include domain entities + general categories that are valuable for tagging
THREAT_ACTOR_KEYWORDS = {
    'APT', 'group', 'gang', 'campaign', 'threat actor', 'hacker',
    'collective', 'organization', 'state-sponsored',
    # Known ransomware families
    'clop', 'lockbit', 'conti', 'revil', 'blackmatter', 'darkside',
    'sodinokibi', 'ragnar', 'netwalker', 'egregor', 'maze', 'ryuk',
}

REGION_KEYWORDS = {
    'region', 'country', 'continent', 'area', 'nation', 'territory',
    'Russia', 'China', 'USA', 'Europe', 'Asia', 'Africa', 'Americas',
}

VULNERABILITY_KEYWORDS = {
    'vulnerability', 'exploit', 'zero-day', '0-day', 'flaw', 'hole',
    'breach', 'bug', 'issue', 'threat'
}

# Broad categories that are relevant for new tags
CATEGORY_KEYWORDS = {
    'dataskydd', 'privacy', 'data protection', 'gdpr', 'compliance',
    'juridik', 'legal', 'lag', 'regulations', 'regulatory', 'law',
    'ransomware', 'malware', 'virus', 'worm', 'trojan', 'spyware',
    'sårbarhet', 'vulnerability', 'cve', 'zero-day',
    'säkerhet', 'security', 'incident', 'breach', 'attack', 'hack',
    'compromise', 'infection', 'exploit', 'exfiltration',
    'data theft', 'data_theft', 'data_breach', 'exposed_data', 'extortion',
    'prison', 'sentence', 'criminal', 'crime', 'police', 'arrest',
    'company', 'corporation', 'business', 'employer', 'employee',
    'account', 'credential', 'password', 'access', 'authentication',
}

# Known organizations and companies (often appear as single-word tags)
# Allows automatic creation of tags for well-known entities
KNOWN_ORGANIZATIONS = {
    'apple', 'microsoft', 'google', 'amazon', 'facebook', 'meta',
    'ibm', 'cisco', 'vmware', 'oracle', 'salesforce', 'adobe',
    'nhs', 'fbi', 'cia', 'nsa', 'dhs', 'fcc', 'ftc', 'sec',
    'interpol', 'europol', 'nato', 'un', 'eu',
    'tesla', 'uber', 'airbnb', 'netflix', 'twitter', 'linkedin',
    'walmart', 'target', 'home-depot', 'equifax', 'anthem',
}

# Keywords that indicate named entities (companies, products, people, locations)
# These should generally be created as new tags if not found
NAMED_ENTITY_INDICATORS = {
    'corp', 'corporation', 'company', 'ltd', 'inc', 'gmbh', 'ag', 'plc',
    'product', 'software', 'service', 'platform', 'tool',
    'system', 'application', 'framework', 'library',
    'city', 'country', 'state', 'province', 'region', 'continent',
    'apple', 'microsoft', 'google', 'amazon', 'facebook', 'meta',
}


class TagManager:
    """Manages article tags with priority-based tag selection and embedding support."""

    TAG_CATEGORY_GENERAL = "GENERAL"
    TAG_CATEGORY_DOMAIN_ENTITY = "DOMAIN_ENTITY"
    TAG_CATEGORY_VULNERABILITY = VULNERABILITY_TAG_CATEGORY

    def __init__(self, store: NewsStore, llm_client: Optional[Any] = None):
        """
        Initialize the TagManager with a NewsStore and optional LLM client for embeddings.
        
        Args:
            store: NewsStore instance
            llm_client: Optional LLM client with embed() method for embedding-based matching
        """
        self.store = store
        self.llm_client = llm_client
        self._embedding_cache: Dict[str, List[float]] = {}

    def init_tag_tables(self) -> None:
        """Initialize tag tables in the database (idempotent)."""
        try:
            # Try to get a tag to see if tables exist
            self.get_all_tags()
        except Exception as e:
            logger.warning(f"Tag tables might not exist yet: {e}")
            # Tables will be created by the store during initialization

    def add_tag(
        self,
        name: str,
        category: str = TAG_CATEGORY_GENERAL,
        description: Optional[str] = None,
    ) -> Optional[int]:
        """
        Add a new tag to the database.

        Args:
            name: Tag name (normalized to lowercase)
            category: Name of a database-defined tag category
            description: Optional description

        Returns:
            Tag ID if successful, None otherwise
        """
        if not name or not isinstance(name, str):
            return None

        name = name.strip().lower()
        if not name:
            return None

        # CVE identifiers always belong to the vulnerability category, regardless
        # of the category proposed by the caller or the LLM.
        if is_cve_tag(name):
            category = self.TAG_CATEGORY_VULNERABILITY

        # Non-CVE categories must resolve to a category defined by the active
        # database. Stores without category support retain the legacy defaults.
        if not is_cve_tag(name):
            resolved_category = self._resolve_tag_category(category)
            if resolved_category is None:
                logger.debug(
                    f"[TagValidate] Rejected tag '{name}': unknown category {category!r}"
                )
                return None
            category = resolved_category

        # Validate tag name length (max 2 words for GENERAL, unlimited otherwise)
        if not self._is_valid_tag_name(name, category):
            logger.debug(f"[TagValidate] Rejected tag '{name}': exceeds word limit for category {category}")
            return None

        try:
            fn = getattr(self.store, "add_tag", None)
            if callable(fn):
                tag_id = fn(name=name, category=category, description=description)
                embedding = self._embedding_cache.get(name)
                update_embedding = getattr(self.store, "update_tag_embedding", None)
                if tag_id and embedding and callable(update_embedding):
                    update_embedding(
                        tag_id,
                        embedding,
                        model=get_client_embedding_model(self.llm_client),
                        source_hash=embedding_source_hash(name),
                    )
                return tag_id
        except Exception as e:
            logger.error(f"Error adding tag: {e}")
        return None

    def get_tag_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a tag by name (case-insensitive)."""
        if not name or not isinstance(name, str):
            return None

        try:
            fn = getattr(self.store, "get_tag_by_name", None)
            if callable(fn):
                return fn(name.strip().lower())
        except Exception as e:
            logger.error(f"Error getting tag: {e}")
        return None

    def get_all_tags(self) -> List[Dict[str, Any]]:
        """Get all tags."""
        try:
            fn = getattr(self.store, "get_all_tags", None)
            if callable(fn):
                return fn()
        except Exception as e:
            logger.error(f"Error getting all tags: {e}")
        return []

    def get_all_categories(self) -> List[Dict[str, Any]]:
        """Get all tag categories defined by the active store."""
        try:
            fn = getattr(self.store, "get_all_categories", None)
            if callable(fn):
                categories = fn()
                if isinstance(categories, list):
                    return [category for category in categories if isinstance(category, dict)]
        except Exception as e:
            logger.error(f"Error getting tag categories: {e}")
        return []

    def _resolve_tag_category(
        self,
        proposed_category: Any,
        fallback_category: str = TAG_CATEGORY_GENERAL,
        categories: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """Resolve a proposed category to its canonical database-defined name."""
        available = self.get_all_categories() if categories is None else categories
        if not available:
            legacy_categories = {
                self.TAG_CATEGORY_GENERAL,
                self.TAG_CATEGORY_DOMAIN_ENTITY,
                self.TAG_CATEGORY_VULNERABILITY,
            }
            proposed = str(proposed_category or "").strip().upper()
            if proposed in legacy_categories:
                return proposed
            if fallback_category in legacy_categories:
                return fallback_category
            return self.TAG_CATEGORY_GENERAL

        category_names = {
            str(category.get("name") or "").strip().casefold():
            str(category.get("name") or "").strip()
            for category in available
            if str(category.get("name") or "").strip()
        }
        for candidate in (proposed_category, fallback_category, self.TAG_CATEGORY_GENERAL):
            normalized = str(candidate or "").strip().casefold()
            if normalized in category_names:
                return category_names[normalized]

        logger.warning(
            "No database-defined tag category matches proposed=%r or fallback=%r",
            proposed_category,
            fallback_category,
        )
        return None

    def _ensure_cve_category(self, tag: Dict[str, Any]) -> Dict[str, Any]:
        """Persist and return the canonical category for an existing CVE tag."""
        name = str(tag.get("name") or "").strip()
        if (
            not is_cve_tag(name)
            or tag.get("category") == self.TAG_CATEGORY_VULNERABILITY
        ):
            return tag

        normalized = tag.copy()
        normalized["category"] = self.TAG_CATEGORY_VULNERABILITY

        update_tag = getattr(self.store, "update_tag", None)
        tag_id = tag.get("id")
        if callable(update_tag) and tag_id:
            try:
                updated = update_tag(
                    tag_id,
                    category=self.TAG_CATEGORY_VULNERABILITY,
                )
                if isinstance(updated, dict):
                    normalized.update(updated)
                    normalized["category"] = self.TAG_CATEGORY_VULNERABILITY
            except Exception as e:
                logger.warning(
                    "Could not update category for CVE tag '%s': %s",
                    name,
                    e,
                )

        return normalized

    def tag_article(
        self,
        article_id: str,
        tag_ids: List[int],
    ) -> bool:
        """
        Tag an article with multiple tags.

        Args:
            article_id: Article ID
            tag_ids: List of tag IDs to assign

        Returns:
            True if successful, False otherwise
        """
        if not article_id or not tag_ids:
            return False

        # Deduplicate tag IDs (remove duplicates while preserving order)
        seen: Set[int] = set()
        unique_tag_ids: List[int] = []
        for tag_id in tag_ids:
            if tag_id not in seen:
                seen.add(tag_id)
                unique_tag_ids.append(tag_id)
        
        if len(unique_tag_ids) < len(tag_ids):
            logger.debug(f"[TagArticle] Deduplicated tag IDs: {len(tag_ids)} → {len(unique_tag_ids)}")

        try:
            fn = getattr(self.store, "add_article_tags", None)
            if callable(fn):
                fn(article_id=article_id, tag_ids=unique_tag_ids)
                return True
        except Exception as e:
            logger.error(f"Error tagging article: {e}")
        return False

    def get_article_tags(self, article_id: str) -> List[Dict[str, Any]]:
        """Get all tags for an article."""
        try:
            fn = getattr(self.store, "get_article_tags", None)
            if callable(fn):
                return fn(article_id=article_id)
        except Exception as e:
            logger.error(f"Error getting article tags: {e}")
        return []

    async def ensure_similar_articles_share_tags(
        self,
        articles: List[Dict[str, Any]],
        *,
        similarity_threshold: float = 0.78,
        embedding_text_chars: int = 2000,
        embedding_max_concurrency: int = 4,
        max_shared_tags: int = 1,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Add a shared existing tag to similar articles that have no tag overlap.

        The method never removes or replaces tags. A tag already assigned to at
        least one article in the similarity group is selected, preferring tags
        used by more group members and general tags over narrower entities.
        """
        if (
            len(articles) < 2
            or not self.llm_client
            or not hasattr(self.llm_client, "embed")
            or max_shared_tags <= 0
        ):
            return {}

        groups = await group_articles_by_similarity(
            articles,
            self.llm_client.embed,
            embedding_text_chars=max(1, embedding_text_chars),
            similarity_threshold=min(1.0, max(0.0, similarity_threshold)),
            max_concurrency=max(1, embedding_max_concurrency),
            store=self.store,
            embedding_model=get_client_embedding_model(self.llm_client),
        )
        add_tag_to_article = getattr(self.store, "add_tag_to_article", None)
        if not callable(add_tag_to_article):
            logger.warning("Store doesn't support adding individual tags to articles")
            return {}

        additions: Dict[str, List[Dict[str, Any]]] = {}
        category_priority = {
            self.TAG_CATEGORY_GENERAL: 0,
            self.TAG_CATEGORY_VULNERABILITY: 1,
            self.TAG_CATEGORY_DOMAIN_ENTITY: 2,
        }

        for group in groups:
            if len(group) < 2:
                continue

            tags_by_article: Dict[str, List[Dict[str, Any]]] = {}
            article_by_id: Dict[str, Dict[str, Any]] = {}
            for article in group:
                article_id = str(article.get("id") or "").strip()
                if not article_id:
                    continue
                article_by_id[article_id] = article
                tags_by_article[article_id] = self.get_article_tags(article_id)

            if len(tags_by_article) < 2:
                continue

            tag_id_sets = [
                {tag.get("id") for tag in tags if tag.get("id")}
                for tags in tags_by_article.values()
            ]
            if tag_id_sets and set.intersection(*tag_id_sets):
                continue

            tag_counts: Dict[int, int] = {}
            tags_by_id: Dict[int, Dict[str, Any]] = {}
            for tags in tags_by_article.values():
                for tag in tags:
                    tag_id = tag.get("id")
                    if not isinstance(tag_id, int) or tag_id <= 0:
                        continue
                    tag_counts[tag_id] = tag_counts.get(tag_id, 0) + 1
                    tags_by_id[tag_id] = tag

            candidates = sorted(
                tags_by_id.values(),
                key=lambda tag: (
                    -tag_counts[int(tag["id"])],
                    category_priority.get(str(tag.get("category") or ""), 3),
                    str(tag.get("name") or "").lower(),
                ),
            )[:max_shared_tags]

            for tag in candidates:
                tag_id = int(tag["id"])
                tag_name = str(tag.get("name") or "")
                for article_id, existing_tags in tags_by_article.items():
                    if any(existing.get("id") == tag_id for existing in existing_tags):
                        continue
                    if not add_tag_to_article(article_id, tag_id):
                        continue

                    shared_tag = dict(tag)
                    shared_tag["reasoning"] = "Delad från semantiskt liknande artikel."
                    existing_tags.append(shared_tag)
                    additions.setdefault(article_id, []).append(shared_tag)

                    article = article_by_id[article_id]
                    article_tag_names = list(article.get("tags") or [])
                    if tag_name and tag_name not in article_tag_names:
                        article["tags"] = article_tag_names + [tag_name]
                        self.store.upsert_article(article)

        if additions:
            logger.info(
                "[TagConsistency] Added %d shared tag associations to %d articles.",
                sum(len(tags) for tags in additions.values()),
                len(additions),
            )
        return additions

    def _is_relevant_new_tag(self, text: str) -> bool:
        """
        Check if text represents a relevant tag that should be created as a new tag.
        
        This includes:
        - Domain entities (CVEs, threat actors, regions, vulnerabilities)
        - Ransomware families and threat actors (clop, lockbit, etc.)
        - Known organizations (apple, nhs, microsoft, etc.)
        - Broad categories (data protection, ransomware, legal terms, etc.)
        - Named entities (companies, products, people, locations)
        - Tags that are 2+ words (typically more specific and valuable)
        
        Returns False for very generic terms that shouldn't be new tags.
        """
        if not text or len(text) < 2:
            return False
            
        text_lower = text.lower()
        
        # Normalize underscores and hyphens to spaces for word counting
        normalized = text.replace('_', ' ').replace('-', ' ')
        word_count = len(normalized.split())
        
        # CVE pattern - always relevant
        if CVE_PATTERN.search(text):
            return True
        
        # Check for threat actor keywords (includes ransomware families)
        for keyword in THREAT_ACTOR_KEYWORDS:
            if keyword.lower() in text_lower:
                return True
        
        # Check for region keywords
        for keyword in REGION_KEYWORDS:
            if keyword.lower() in text_lower:
                return True
        
        # Check for vulnerability keywords
        for keyword in VULNERABILITY_KEYWORDS:
            if keyword.lower() in text_lower:
                return True
        
        # Check for broad categories (dataskydd, ransomware, legal terms, etc.)
        for keyword in CATEGORY_KEYWORDS:
            if keyword.lower() in text_lower:
                return True
        
        # Check if it's a known organization (even single-word tags)
        # This allows tags like "apple", "nhs", "microsoft" to be created automatically
        if text_lower in KNOWN_ORGANIZATIONS:
            return True
        
        # Check for named entity indicators (companies, products, etc.)
        # These are relevant if they're multi-word tags (e.g., "Microsoft Corporation")
        # or if they contain named entity keywords
        if word_count >= 2:
            for keyword in NAMED_ENTITY_INDICATORS:
                if keyword.lower() in text_lower:
                    return True
            # Multi-word tags are generally specific enough to be valuable
            # e.g., "ransomware family", "security vendor", "threat actor",
            # "data theft", "account info", "employer law"
            return True
        
        # Single-word tags: only if they match our keyword lists or known organizations
        # (avoid creating too generic single-word tags)
        return False

    def _find_similar_existing_tags(
        self,
        tag_name: str,
        similarity_threshold: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Find existing tags that are semantically similar to the given tag name.

        Uses embedding-based similarity if LLM client is available, otherwise falls back
        to simple string similarity metrics and exact substring matching.

        Also checks if the tag_name matches any existing tag's synonyms, and returns
        the main tag if a synonym match is found.

        Args:
            tag_name: Tag name to find matches for
            similarity_threshold: Minimum similarity score (0.0-1.0)

        Returns:
            List of similar existing tags, ordered by similarity
        """
        existing_tags = self.get_all_tags()
        if not existing_tags:
            return []

        tag_lower = tag_name.lower()

        # CVE identifiers are unique IDs and must never be fuzzy-matched to a
        # different CVE merely because most digits happen to be the same.
        if is_cve_tag(tag_name):
            return [
                tag
                for tag in existing_tags
                if str(tag.get("name") or "").strip().lower() == tag_lower
            ][:1]

        # An exact name must always win over semantic matching. Otherwise an
        # embedding result in another category can replace a tag that already
        # exists under precisely the requested name.
        for tag in existing_tags:
            existing_name = str(tag.get("name") or "").strip().lower()
            if existing_name == tag_lower:
                logger.debug(
                    f"[TagMatch] Exact string match: '{tag_name}' == '{existing_name}'"
                )
                return [tag]

        # Check if tag_name matches any existing tag's synonyms (synonym-to-main-tag replacement)
        for tag in existing_tags:
            tag_synonyms = tag.get("synonyms", [])
            if tag_synonyms and isinstance(tag_synonyms, list):
                # Normalize synonyms to lowercase for comparison
                synonyms_lower = [s.lower() if isinstance(s, str) else str(s).lower() for s in tag_synonyms]
                if tag_lower in synonyms_lower:
                    logger.info(f"[TagMatch] Synonym match: '{tag_name}' is a synonym of main tag '{tag.get('name')}'")
                    return [tag]

        # Try embedding-based matching first if LLM client is available
        if self.llm_client and hasattr(self.llm_client, 'embed'):
            try:
                logger.debug(f"[TagMatch] Attempting embedding-based matching for '{tag_name}'")
                
                # Get embedding for the candidate tag
                candidate_embedding = self._get_or_compute_embedding(tag_name)
                
                if candidate_embedding:
                    # Use the store's embedding similarity method if available
                    fn = getattr(self.store, "get_tags_by_embedding_similarity", None)
                    if callable(fn):
                        try:
                            similar_by_embedding = fn(
                                candidate_embedding,
                                similarity_threshold=max(0.6, similarity_threshold - 0.1),
                                limit=5,
                                model=get_client_embedding_model(self.llm_client),
                            )
                        except TypeError:
                            # Compatibility with external stores implementing the older API.
                            similar_by_embedding = fn(
                                candidate_embedding,
                                similarity_threshold=max(0.6, similarity_threshold - 0.1),
                                limit=5,
                            )
                        
                        if similar_by_embedding:
                            # Log the embedding matches found
                            match_details = "; ".join([
                                f"{t.get('name')} (similarity: {t.get('_similarity_score', 'N/A'):.3f})"
                                for t in similar_by_embedding
                            ])
                            logger.info(f"[TagMatch] Embedding-based match for '{tag_name}': {match_details}")
                            
                            # Similarity is the primary ordering criterion;
                            # category is only a tie-breaker.
                            similar_by_embedding.sort(
                                key=lambda x: (
                                    -float(x.get("_similarity_score", 0.0)),
                                    x.get("category") != self.TAG_CATEGORY_GENERAL,
                                )
                            )
                            return similar_by_embedding
                        else:
                            logger.debug(f"[TagMatch] No embedding matches found for '{tag_name}' (threshold={max(0.6, similarity_threshold - 0.1):.2f})")
                else:
                    logger.debug(f"[TagMatch] Failed to compute embedding for '{tag_name}'")
            except Exception as e:
                logger.warning(f"[TagMatch] Embedding-based tag matching failed for '{tag_name}', falling back to string similarity: {e}")

        # Fallback to string-based similarity matching
        logger.debug(f"[TagMatch] Using string-based matching for '{tag_name}'")
        matches: List[Tuple[Dict[str, Any], float]] = []

        for tag in existing_tags:
            tag_name_str = tag.get("name", "").lower()
            if not tag_name_str:
                continue

            # Phrase containment is useful for tags such as "password" and
            # "password spraying", but only at token boundaries. Raw
            # substring matching incorrectly considered "ray" part of
            # "spraying" and "mfa" part of "comfast".
            if self._contains_complete_term(tag_lower, tag_name_str):
                logger.debug(
                    f"[TagMatch] Whole-term match: '{tag_name}' vs '{tag_name_str}'"
                )
                matches.append((tag, 0.9))
                continue

            # Levenshtein-based similarity (much better than char-set similarity)
            # Only accept matches with very high similarity (0.75+) to avoid false positives
            similarity = self._simple_similarity(tag_lower, tag_name_str)
            if similarity >= max(0.75, similarity_threshold):
                logger.debug(f"[TagMatch] Levenshtein similarity: '{tag_name}' vs '{tag_name_str}' = {similarity:.3f}")
                matches.append((tag, similarity))

        # Sort by similarity (descending) and prefer GENERAL category
        matches.sort(
            key=lambda x: (
                -x[1],  # Higher similarity first
                x[0].get("category") != self.TAG_CATEGORY_GENERAL,  # GENERAL first
            )
        )

        if matches:
            match_details = "; ".join([
                f"{t.get('name')} (similarity: {s:.3f})"
                for t, s in matches[:3]  # Show top 3
            ])
            logger.info(f"[TagMatch] String-based match for '{tag_name}': {match_details}")

        return [tag for tag, _ in matches]

    @staticmethod
    def _contains_complete_term(first: str, second: str) -> bool:
        """Return True when either tag contains the other as complete tokens."""
        if not first or not second:
            return False

        def contains(container: str, term: str) -> bool:
            pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
            return re.search(pattern, container) is not None

        return contains(first, second) or contains(second, first)

    async def _get_or_compute_embedding_async(self, text: str) -> Optional[List[float]]:
        """Compute an embedding without blocking the active event loop."""
        if not text or not self.llm_client:
            return None

        text_lower = text.lower()
        if text_lower in self._embedding_cache:
            return self._embedding_cache[text_lower]

        model = get_client_embedding_model(self.llm_client)
        get_tag = getattr(self.store, "get_tag_by_name", None)
        existing_tag = get_tag(text) if callable(get_tag) else None
        if isinstance(existing_tag, dict):
            persisted = cached_embedding(existing_tag, text, model)
            if persisted is not None:
                self._embedding_cache[text_lower] = persisted
                return persisted

        try:
            embedding = await self.llm_client.embed(text)
            if embedding:
                normalized = [float(value) for value in embedding]
                self._embedding_cache[text_lower] = normalized
                update_embedding = getattr(self.store, "update_tag_embedding", None)
                if existing_tag and callable(update_embedding):
                    update_embedding(
                        int(existing_tag["id"]),
                        normalized,
                        model=model,
                        source_hash=embedding_source_hash(text),
                    )
                return normalized
            return None
        except Exception as e:
            logger.warning(f"[TagMatch] Failed to compute embedding for '{text}': {e}")
            return None

    async def _cache_candidate_embeddings(self, candidate_tags: List) -> None:
        """Populate the sync matcher's cache while its caller can await I/O."""
        if not self.llm_client or not hasattr(self.llm_client, "embed"):
            return

        existing_tags = self.get_all_tags()
        exact_names = {
            str(tag.get("name") or "").strip().lower()
            for tag in existing_tags
        }
        synonym_names: Set[str] = set()
        for tag in existing_tags:
            synonyms = tag.get("synonyms", [])
            if isinstance(synonyms, list):
                synonym_names.update(
                    str(synonym).strip().lower() for synonym in synonyms
                )

        seen: Set[str] = set()
        for candidate in candidate_tags:
            if isinstance(candidate, dict):
                tag_name = str(candidate.get("name") or "").strip().lower()
            else:
                tag_name = str(candidate or "").strip().lower()

            if (
                not tag_name
                or tag_name in seen
                or tag_name in exact_names
                or tag_name in synonym_names
                or is_cve_tag(tag_name)
            ):
                continue

            seen.add(tag_name)
            await self._get_or_compute_embedding_async(tag_name)

    def _get_or_compute_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding or compute and cache it.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if not available
        """
        if not text or not self.llm_client:
            return None
        
        text_lower = text.lower()
        
        # Check cache
        if text_lower in self._embedding_cache:
            return self._embedding_cache[text_lower]
        
        # Compute synchronously only when no event loop is active. Async callers
        # populate this cache through _cache_candidate_embeddings first.
        try:
            import asyncio
            try:
                asyncio.get_running_loop()
                logger.debug(
                    f"[TagMatch] Embedding for '{text}' was not precomputed in async context"
                )
                return None
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    embedding = loop.run_until_complete(self.llm_client.embed(text))
                    if embedding:
                        self._embedding_cache[text_lower] = embedding
                    return embedding
                finally:
                    loop.close()
        except Exception as e:
            logger.debug(f"Failed to compute embedding for '{text}': {e}")
            return None

    @staticmethod
    def _simple_similarity(s1: str, s2: str) -> float:
        """
        Calculate similarity between two strings using Levenshtein distance.
        This is much better than char-set similarity for tag matching.
        
        Returns a score between 0.0 (completely different) and 1.0 (identical).
        """
        if not s1 or not s2:
            return 0.0
        
        if s1 == s2:
            return 1.0
        
        # Levenshtein distance implementation
        len1, len2 = len(s1), len(s2)
        
        # Create a matrix to store distances
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        # Initialize first column and row
        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j
        
        # Fill the matrix
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if s1[i - 1] == s2[j - 1]:
                    matrix[i][j] = matrix[i - 1][j - 1]
                else:
                    matrix[i][j] = 1 + min(
                        matrix[i - 1][j],      # deletion
                        matrix[i][j - 1],      # insertion
                        matrix[i - 1][j - 1]   # substitution
                    )
        
        # Calculate similarity from distance
        distance = matrix[len1][len2]
        max_len = max(len1, len2)
        
        # Convert distance to similarity score (0.0-1.0)
        # Higher score = more similar
        similarity = 1.0 - (distance / max_len)
        return max(0.0, similarity)

    @staticmethod
    def _is_valid_tag_name(tag_name: str, category: str = TAG_CATEGORY_GENERAL) -> bool:
        """
        Validate tag name based on category.
        
        Rules:
        - GENERAL tags: max 2 words (e.g., "data breach" is ok, "security incident response" is not)
        - DOMAIN_ENTITY tags: unlimited words (e.g., "John Smith", "Apple Inc", "SQL Injection Attack")
        
        Args:
            tag_name: Tag name to validate
            category: Tag category
            
        Returns:
            True if valid, False otherwise
        """
        if not tag_name or not isinstance(tag_name, str):
            return False
        
        tag_name = tag_name.strip()
        if not tag_name:
            return False
        
        # Count words (split by whitespace)
        word_count = len(tag_name.split())
        
        # GENERAL tags: max 2 words
        if category == TagManager.TAG_CATEGORY_GENERAL:
            return word_count <= 2
        
        # DOMAIN_ENTITY tags: unlimited
        return True

    def select_tags_for_article(
        self,
        article_id: str,
        candidate_tags: List,  # Can be List[str] or List[Dict[str, str]]
        allow_new_tags: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Select the best tags for an article from a list of candidates.

        Priority:
        1. Use existing tags that match candidates (prefer GENERAL over DOMAIN_ENTITY)
        2. For NAMED_ENTITY tags from LLM: Always create if not found (organizations, groups, actors)
        3. For CATEGORY tags from LLM: Only create if they match keyword lists
        4. For backward compatibility with string tags: Use _is_relevant_new_tag() heuristic

        Args:
            article_id: Article ID (for context/validation)
            candidate_tags: List of candidate tags (strings or dicts with 'name', 'type', and optional 'reasoning')
            allow_new_tags: If True, allow creating new relevant tags

        Returns:
            List of selected tags as dicts with 'id', 'name', 'category', and optional 'reasoning' fields
        """
        selected_tags: List[Dict[str, Any]] = []
        processed_tags: Set[str] = set()
        available_categories = self.get_all_categories()
        available_category_names = {
            str(category.get("name") or "").strip().casefold():
            str(category.get("name") or "").strip()
            for category in available_categories
            if str(category.get("name") or "").strip()
        }

        for candidate in candidate_tags:
            if not candidate:
                continue
            
            # Handle both string and dict formats
            if isinstance(candidate, dict):
                tag_name = candidate.get("name", "").strip().lower()
                tag_type = candidate.get("type", "CATEGORY").upper()
                proposed_category = candidate.get("category")
                tag_reasoning = candidate.get("reasoning", "")
            else:
                tag_name = str(candidate).strip().lower()
                tag_type = "CATEGORY"  # Default for backward compatibility
                proposed_category = None
                tag_reasoning = ""
            
            if not tag_name or tag_name in processed_tags:
                continue

            processed_tags.add(tag_name)

            # Try to find existing similar tags
            similar_tags = self._find_similar_existing_tags(tag_name)

            if similar_tags:
                # Use the best match (already sorted by priority)
                tag_dict = self._ensure_cve_category(similar_tags[0].copy())
                tag_dict["reasoning"] = tag_reasoning
                logger.debug(f"[TagSelect] Using existing tag '{tag_dict['name']}' for candidate '{tag_name}'")
                selected_tags.append(tag_dict)
            elif allow_new_tags:
                # Decision logic based on tag type
                should_create = False
                default_category = (
                    self.TAG_CATEGORY_DOMAIN_ENTITY
                    if tag_type == "NAMED_ENTITY"
                    else self.TAG_CATEGORY_GENERAL
                )
                explicit_category = available_category_names.get(
                    str(proposed_category or "").strip().casefold()
                )
                resolved_category = self._resolve_tag_category(
                    proposed_category,
                    fallback_category=default_category,
                    categories=available_categories,
                )
                
                if tag_type == "NAMED_ENTITY":
                    # NAMED_ENTITY tags (organizations, groups, actors) are always created
                    # This allows LLM-identified entities like "Apple", "Clop", "APT28" to be tagged automatically
                    should_create = True
                elif tag_type == "CATEGORY":
                    # CATEGORY tags only created if they match our keyword heuristics
                    should_create = self._is_relevant_new_tag(tag_name)

                # An explicit valid database category is enough evidence to create
                # a new tag even if the legacy keyword heuristic does not know it.
                if explicit_category:
                    should_create = True
                
                if should_create:
                    # Validate tag name (max 2 words for GENERAL, unlimited otherwise)
                    if is_cve_tag(tag_name):
                        tag_category = self.TAG_CATEGORY_VULNERABILITY
                    else:
                        tag_category = resolved_category

                    if tag_category is None:
                        logger.debug(
                            f"[TagSelect] Rejected tag '{tag_name}': no valid database category"
                        )
                        continue
                    
                    if not self._is_valid_tag_name(tag_name, tag_category):
                        logger.debug(f"[TagSelect] Rejected tag '{tag_name}': exceeds word limit for {tag_type}")
                        continue
                    
                    tag_id = self.add_tag(tag_name, category=tag_category)
                    if tag_id:
                        logger.debug(f"[TagSelect] Created new tag '{tag_name}' (type={tag_type})")
                        selected_tags.append({
                            "id": tag_id,
                            "name": tag_name,
                            "category": tag_category,
                            "reasoning": tag_reasoning,
                        })
                    else:
                        # Tag might already exist, try to get it
                        existing = self.get_tag_by_name(tag_name)
                        if existing:
                            existing["reasoning"] = tag_reasoning
                            selected_tags.append(existing)
                else:
                    # Tag not relevant for automatic creation
                    logger.debug(f"Skipping tag (not classified as relevant): {tag_name} (type={tag_type})")

        # Deduplicate by tag ID (prevent same tag being assigned twice)
        seen_tag_ids: Set[int] = set()
        deduplicated_tags: List[Dict[str, Any]] = []
        for tag_dict in selected_tags:
            tag_id = tag_dict.get("id")
            if tag_id and tag_id not in seen_tag_ids:
                seen_tag_ids.add(tag_id)
                deduplicated_tags.append(tag_dict)
            elif not tag_id:
                # No ID (shouldn't happen), add anyway
                deduplicated_tags.append(tag_dict)
        
        if len(deduplicated_tags) < len(selected_tags):
            logger.info(f"[TagSelect] Deduplicated tags: {len(selected_tags)} → {len(deduplicated_tags)}")

        return deduplicated_tags

    async def select_tags_for_article_async(
        self,
        article_id: str,
        candidate_tags: List,
        allow_new_tags: bool = True,
    ) -> List[Dict[str, Any]]:
        """Async selection path that precomputes embeddings before matching."""
        await self._cache_candidate_embeddings(candidate_tags)
        return self.select_tags_for_article(
            article_id=article_id,
            candidate_tags=candidate_tags,
            allow_new_tags=allow_new_tags,
        )

    def extract_tags_from_llm_response(self, response: str) -> List[Dict[str, str]]:
        """
        Extract tags from LLM response.

        Expects JSON format with per-tag reasoning:
        {
            "tags": [
                {"tag": "tag1", "type": "NAMED_ENTITY", "category": "ORGANIZATION", "reasoning": "Why this tag..."},
                {"tag": "tag2", "type": "CATEGORY", "category": "THREAT", "reasoning": "Why this tag..."},
                ...
            ]
        }
        
        For backward compatibility, also supports old formats:
        {
            "tags": [
                {"tag": "tag1", "type": "NAMED_ENTITY"},
                {"tag": "tag2", "type": "CATEGORY"},
                ...
            ],
            "reasoning": "General explanation"
        }
        
        Or simply:
        {
            "tags": ["tag1", "tag2", ...],
            "reasoning": "..."
        }

        Args:
            response: LLM response text

        Returns:
            List of tags as dicts with 'name', 'type', 'category', and optional
            'reasoning' keys.
            If reasoning is not provided, defaults to empty string.
        """
        if not response or not isinstance(response, str):
            return []

        try:
            import re
            
            # Try to extract tags array using regex
            # This avoids json.loads() issues with control characters
            tags_match = re.search(r'"tags"\s*:\s*\[(.*?)\]', response, re.DOTALL)
            if tags_match:
                tags_str = tags_match.group(1)
                tags = []
                
                # Try to parse as new format: [{"tag": "...", "type": "..."}, ...]
                if '{' in tags_str:
                    # Extract objects
                    obj_matches = re.findall(r'\{[^}]*\}', tags_str)
                    for obj_match in obj_matches:
                        try:
                            # Clean and parse
                            obj_match_clean = ''.join(
                                c for c in obj_match
                                if ord(c) >= 32 or c in '\n\r\t'
                            )
                            obj = json.loads(obj_match_clean)
                            if isinstance(obj, dict):
                                tag_name = obj.get("tag") or obj.get("name")
                                tag_type = obj.get("type", "CATEGORY")
                                tag_category = obj.get("category")
                                tag_reasoning = obj.get("reasoning", "")
                                if tag_name:
                                    tags.append({
                                        "name": str(tag_name).strip().lower(),
                                        "type": tag_type.upper() if tag_type else "CATEGORY",
                                        "category": str(tag_category).strip() if tag_category else "",
                                        "reasoning": str(tag_reasoning).strip() if tag_reasoning else ""
                                    })
                        except Exception:
                            pass
                else:
                    # Fallback: Extract as simple quoted strings (old format)
                    tag_matches = re.findall(r'"([^"]*)"', tags_str)
                    for tag in tag_matches:
                        cleaned_tag = ''.join(
                            c for c in tag
                            if ord(c) >= 32 or c in '\n\r\t'
                        ).strip()
                        if cleaned_tag:
                            tags.append({
                                "name": cleaned_tag.lower(),
                                "type": "CATEGORY",  # Default type for old format
                                "reasoning": ""  # No reasoning in old format
                            })
                
                if tags:
                    return tags
            
            # Fallback: Try full JSON parsing
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                try:
                    data = json.loads(json_str)
                    if isinstance(data, dict):
                        tags_data = data.get("tags", [])
                        if isinstance(tags_data, list):
                            tags = []
                            for t in tags_data:
                                if isinstance(t, dict):
                                    tag_name = t.get("tag") or t.get("name")
                                    tag_type = t.get("type", "CATEGORY")
                                    tag_category = t.get("category")
                                    tag_reasoning = t.get("reasoning", "")
                                    if tag_name:
                                        tags.append({
                                            "name": str(tag_name).strip().lower(),
                                            "type": tag_type.upper() if tag_type else "CATEGORY",
                                            "category": str(tag_category).strip() if tag_category else "",
                                            "reasoning": str(tag_reasoning).strip() if tag_reasoning else ""
                                        })
                                elif isinstance(t, str):
                                    tags.append({
                                        "name": str(t).strip().lower(),
                                        "type": "CATEGORY"
                                    })
                            return tags
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            logger.debug(f"Error extracting tags from LLM response: {e}")

        return []

    async def generate_tags_for_article(
        self,
        llm_client: LLMClient,
        article: Dict[str, Any],
        config: Dict[str, Any],
        max_tags: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Generate and select tags for an article using an LLM.

        Args:
            llm_client: LLM client for generating tags
            article: Article dict with 'title', 'content', etc.
            config: Configuration dict
            max_tags: Maximum number of non-CVE tags to select

        Returns:
            List of selected tags
        """
        if not article:
            return []

        title = article.get("title", "").strip()
        content = article.get("content", "").strip()
        summary = article.get("summary", "").strip()

        # Combine available text for context
        text_context = " ".join([title, content or summary])
        if not text_context:
            return []

        # CVE identifiers are deterministic domain entities. Extract them directly
        # instead of relying on the LLM to include every identifier in its response.
        cve_candidates = [
            {
                "name": cve_id,
                "type": "NAMED_ENTITY",
                "category": self.TAG_CATEGORY_VULNERABILITY,
                "reasoning": "CVE identifier mentioned explicitly in the article.",
            }
            for cve_id in extract_cve_ids(
                " ".join(part for part in (title, content, summary) if part)
            )
        ]

        # Prepare prompt for tag generation
        prompt = self._build_tagging_prompt(
            text_context,
            max_tags,
            categories=self.get_all_categories(),
        )

        try:
            # Use chat() method instead of generate() for compatibility with FallbackLLMClient
            response = await llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.3
            )
            candidate_tags = self.extract_tags_from_llm_response(response)
        except Exception as e:
            logger.error(f"Error generating LLM tags: {e}")
            candidate_tags = []

        try:
            # CVEs are prepended so all directly extracted identifiers are processed
            # even when the LLM returns more candidates than the regular-tag budget.
            regular_limit = max(0, max_tags)
            regular_candidates = [
                candidate
                for candidate in candidate_tags
                if not is_cve_tag(candidate.get("name"))
            ]
            selected = await self.select_tags_for_article_async(
                article_id=article.get("id", ""),
                candidate_tags=cve_candidates + regular_candidates[:regular_limit * 2],
                allow_new_tags=True,
            )

            # Preserve selection order while applying the maximum only to non-CVE
            # tags. Every CVE mentioned in the article remains in the result.
            limited: List[Dict[str, Any]] = []
            regular_count = 0
            for tag in selected:
                if is_cve_tag(tag.get("name")):
                    limited.append(tag)
                elif regular_count < regular_limit:
                    limited.append(tag)
                    regular_count += 1
            return limited
        except Exception as e:
            logger.error(f"Error generating tags: {e}")
            return []

    def _build_tagging_prompt(
        self,
        article_text: str,
        max_tags: int = 5,
        categories: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build a prompt for the LLM to generate tags."""
        # Truncate article text if too long
        if len(article_text) > 2000:
            article_text = article_text[:2000] + "..."

        category_lines = []
        category_names = []
        for category in categories or []:
            name = str(category.get("name") or "").strip()
            if not name:
                continue
            category_names.append(name)
            label = str(category.get("label") or "").strip()
            description = str(category.get("description") or "").strip()
            details = ": ".join(part for part in (label, description) if part)
            category_lines.append(f"- {name}" + (f": {details}" if details else ""))
        if not category_names:
            category_names = ["GENERAL", "DOMAIN_ENTITY", "VULNERABILITY"]
            category_lines = [f"- {name}" for name in category_names]
        category_list = "\n".join(category_lines)
        category_lookup = {name.casefold(): name for name in category_names}

        def example_category(*preferred: str) -> str:
            for name in preferred:
                if name.casefold() in category_lookup:
                    return category_lookup[name.casefold()]
            return category_names[0]

        entity_category = example_category("ORGANIZATION", "DOMAIN_ENTITY", "GENERAL")
        topic_category = example_category("THREAT", "GENERAL")
        product_category = example_category("PRODUCT", "DOMAIN_ENTITY", "GENERAL")

        return f"""Analyze the following article and extract up to {max_tags} relevant non-CVE tags.
Also include every CVE identifier mentioned in the article.
CVE identifiers do not count toward this limit.

CLASSIFICATION RULES:
Tags should be classified as one of two types:

1. NAMED_ENTITY: Names of real-world actors, organizations, or products
   - Ransomware groups/families (Clop, LockBit, Conti, Emotet, etc.)
   - APT groups (APT28, Lazarus, etc.)
   - Companies/Organizations (Apple, Microsoft, NHS, etc.)
   - Threat actors (FIN7, Wizard Spider, etc.)
   - Specific products/services mentioned
   - CVE numbers
   - Regions/Countries mentioned in security context

2. CATEGORY: General security topics or concepts
   - Threat types (ransomware, malware, spyware, exploit)
   - Legal/Regulatory (GDPR, compliance, privacy, legal)
   - Incident types (data breach, account takeover, phishing)
   - Security concepts (zero-day, vulnerability, authentication)
   - Any conceptual or descriptive tags

AVAILABLE DATABASE CATEGORIES:
{category_list}

IMPORTANT:
- Return tags in lowercase
- Keep tags concise (1-3 words max)
- Focus on the main topics and entities mentioned
- Include every CVE identifier, even when this makes the total exceed {max_tags} tags
- Set "category" to exactly one name from AVAILABLE DATABASE CATEGORIES
- Choose the most specific suitable database category; use GENERAL when none is suitable
- For NAMED_ENTITY tags: include ANY organization/group/actor/product name mentioned, even if uncommon
- For CATEGORY tags: only include if they are central to the article
- Provide a brief explanation for each tag (1-2 sentences max)

Article:
{article_text}

Respond in JSON format:
{{
    "tags": [
        {{"tag": "tag1", "type": "NAMED_ENTITY", "category": "{entity_category}", "reasoning": "Brief explanation why this tag is relevant"}},
        {{"tag": "tag2", "type": "CATEGORY", "category": "{topic_category}", "reasoning": "Brief explanation why this tag is relevant"}},
        {{"tag": "tag3", "type": "NAMED_ENTITY", "category": "{product_category}", "reasoning": "Brief explanation why this tag is relevant"}}
    ]
}}"""

    async def reclassify_article_with_existing_tags(
        self,
        llm_client: LLMClient,
        article: Dict[str, Any],
        current_article_tags: List[Dict[str, Any]] = None,
        max_tags: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Reclassify an article by suggesting only from existing tags in the database.
        
        This method presents all available tags (excluding the article's current tags)
        to the LLM and asks which ones are relevant. No new tags are created.
        
        Args:
            llm_client: LLM client for generating suggestions
            article: Article dict with 'title', 'content', etc.
            current_article_tags: List of tags already on the article (to exclude)
            max_tags: Maximum number of tags to suggest
        
        Returns:
            List of suggested tags with 'id', 'name', 'category', 'reasoning'
        """
        if not article:
            return []

        title = article.get("title", "").strip()
        content = article.get("content", "").strip()
        summary = article.get("summary", "").strip()

        # Combine available text for context
        text_context = " ".join([title, content or summary])
        if not text_context:
            return []

        # Get all tags from database
        all_tags = self.get_all_tags()
        if not all_tags:
            logger.warning("No tags available in database for reclassification")
            return []

        # Get current tag names to exclude (case-insensitive)
        current_tag_names = set()
        if current_article_tags:
            current_tag_names = {t.get("name", "").lower() for t in current_article_tags}

        # Filter tags: exclude current tags and organize for the prompt
        available_tags = []
        for tag in all_tags:
            tag_name = tag.get("name", "").lower()
            if tag_name and tag_name not in current_tag_names:
                available_tags.append({
                    "name": tag.get("name", ""),
                    "id": tag.get("id"),
                    "category": tag.get("category", "GENERAL"),
                })

        if not available_tags:
            logger.info(f"[Reclassify] No available tags (article already has all tags)")
            return []

        # Build the reclassification prompt
        prompt = self._build_reclassification_prompt(text_context, available_tags, max_tags)

        try:
            # Use LLM to suggest relevant tags
            response = await llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.3
            )

            # Extract suggested tag names
            suggested_tag_names = self._extract_tag_suggestions_from_response(response)
            logger.debug(f"[Reclassify] LLM suggested: {suggested_tag_names}")

            # Match suggested names to actual tags
            suggested_tags = []
            for suggestion in suggested_tag_names:
                # Find matching tag (case-insensitive)
                match = next(
                    (t for t in available_tags if t["name"].lower() == suggestion.lower()),
                    None
                )
                if match:
                    suggested_tags.append({
                        "id": match["id"],
                        "name": match["name"],
                        "category": match["category"],
                    })

            return suggested_tags[:max_tags]
        except Exception as e:
            logger.error(f"Error reclassifying article: {e}")
            return []

    def _build_reclassification_prompt(
        self,
        article_text: str,
        available_tags: List[Dict[str, str]],
        max_tags: int = 5,
    ) -> str:
        """
        Build a prompt for reclassifying an article using only existing tags.
        
        Args:
            article_text: Article title + content
            available_tags: List of dicts with 'name', 'id', 'category'
            max_tags: Maximum suggestions
        
        Returns:
            Prompt string
        """
        # Truncate article text if too long
        if len(article_text) > 2000:
            article_text = article_text[:2000] + "..."

        # Organize tags by category for clarity
        tags_by_category = {}
        for tag in available_tags:
            cat = tag.get("category", "GENERAL")
            if cat not in tags_by_category:
                tags_by_category[cat] = []
            tags_by_category[cat].append(tag["name"])

        # Format tags for the prompt
        tags_list = ""
        for category in sorted(tags_by_category.keys()):
            tags_list += f"\n{category}:\n"
            for tag_name in sorted(tags_by_category[category]):
                tags_list += f"  - {tag_name}\n"

        return f"""Analyze the following article and suggest relevant tags from the provided list.

IMPORTANT INSTRUCTIONS:
1. You MUST ONLY suggest tags from the provided list below
2. Do NOT suggest tags that are not in the list
3. Do NOT create new tags or suggest new tag names
4. A tag is ONLY relevant if it is DIRECTLY relevant to the article content
5. It is PERFECTLY OK to suggest NO tags if none are directly relevant
6. Provide your reasoning for each suggested tag

Available tags organized by category:
{tags_list}

SUGGESTION RULES:
- Suggest up to {max_tags} tags maximum
- Only include tags that are directly and clearly related to the article
- Do not include marginal or loosely related tags
- If you are unsure whether a tag applies, do NOT include it
- It is better to suggest too few tags than too many

Article:
{article_text}

Respond in JSON format with the EXACT tag names from the list above:
{{
    "suggested_tags": [
        {{"tag": "exact_tag_name", "reasoning": "Why this tag is relevant"}},
        {{"tag": "another_tag", "reasoning": "Why this tag is relevant"}}
    ]
}}

If NO tags are relevant, respond with:
{{
    "suggested_tags": []
}}"""

    def _extract_tag_suggestions_from_response(self, response: str) -> List[str]:
        """
        Extract suggested tag names from LLM response.
        
        Args:
            response: LLM response string
        
        Returns:
            List of suggested tag names
        """
        try:
            # Try to parse as JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response_json = json.loads(json_match.group())
                suggested = response_json.get("suggested_tags", [])

                if isinstance(suggested, list):
                    tag_names = []
                    for item in suggested:
                        if isinstance(item, dict):
                            tag_name = item.get("tag", "").strip()
                        else:
                            tag_name = str(item).strip()

                        if tag_name:
                            tag_names.append(tag_name)

                    logger.debug(f"[ReclassifyExtract] Extracted {len(tag_names)} suggestions: {tag_names}")
                    return tag_names
        except Exception as e:
            logger.debug(f"Error parsing reclassification response: {e}")

        return []

    async def generate_embeddings_for_all_tags(self) -> int:
        """
        Generate and cache embeddings for all tags without embeddings.
        
        This method helps populate the embedding vectors for efficient similarity matching.
        Call this periodically to improve tag matching performance.
        
        Returns:
            Number of tags processed
        """
        if not self.llm_client or not hasattr(self.llm_client, 'embed'):
            logger.warning("LLM client not available or doesn't support embeddings")
            return 0
        
        existing_tags = self.get_all_tags()
        if not existing_tags:
            return 0
        
        processed_count = 0
        embedding_model = get_client_embedding_model(self.llm_client)
        
        for tag in existing_tags:
            try:
                tag_id = tag.get("id")
                tag_name = tag.get("name", "")
                
                # Skip only when both the source text and model still match.
                if cached_embedding(tag, tag_name, embedding_model) is not None:
                    continue
                
                if not tag_id or not tag_name:
                    continue
                
                # Generate embedding
                embedding = await self.llm_client.embed(tag_name)
                
                if embedding:
                    # Update tag with embedding
                    fn = getattr(self.store, "update_tag_embedding", None)
                    if callable(fn):
                        success = fn(
                            tag_id,
                            embedding,
                            model=embedding_model,
                            source_hash=embedding_source_hash(tag_name),
                        )
                        if success:
                            processed_count += 1
                            logger.info(f"Generated embedding for tag '{tag_name}'")
                
            except Exception as e:
                logger.warning(f"Failed to generate embedding for tag '{tag.get('name')}': {e}")
                continue
        
        logger.info(f"Generated embeddings for {processed_count} tags")
        return processed_count

    async def tag_article_by_content(
        self,
        article_id: str,
        article_content: str,
        max_tags: int = 5,
        similarity_threshold: float = 0.70,
    ) -> List[Dict[str, Any]]:
        """
        Tag an article by matching its content embedding to existing tag embeddings.
        
        This provides an alternative/supplementary tagging approach:
        - Compute embedding for article content
        - Find tags with similar embeddings
        - Return top matching tags
        
        Args:
            article_id: Article ID
            article_content: Article summary or content text
            max_tags: Maximum number of tags to assign
            similarity_threshold: Minimum similarity score (0.0-1.0)
            
        Returns:
            List of matched tags
        """
        if not self.llm_client or not hasattr(self.llm_client, 'embed'):
            logger.warning("LLM client not available for content-based tagging")
            return []
        
        if not article_content or not article_content.strip():
            return []
        
        try:
            embedding_model = get_client_embedding_model(self.llm_client)
            article = self.store.get_article(article_id) or {}
            content_embedding = cached_embedding(article, article_content, embedding_model)
            if content_embedding is None:
                content_embedding = await self.llm_client.embed(article_content)
                update_embedding = getattr(self.store, "update_article_embedding", None)
                if content_embedding and callable(update_embedding):
                    update_embedding(
                        article_id,
                        content_embedding,
                        model=embedding_model,
                        source_hash=embedding_source_hash(article_content),
                    )
            
            if not content_embedding:
                logger.warning(f"Failed to embed content for article {article_id}")
                return []
            
            # Find similar tags by embedding
            fn = getattr(self.store, "get_tags_by_embedding_similarity", None)
            if not callable(fn):
                logger.warning("Store doesn't support embedding similarity search")
                return []
            
            try:
                similar_tags = fn(
                    content_embedding,
                    similarity_threshold=similarity_threshold,
                    limit=max_tags,
                    model=embedding_model,
                )
            except TypeError:
                similar_tags = fn(
                    content_embedding,
                    similarity_threshold=similarity_threshold,
                    limit=max_tags,
                )
            
            logger.info(f"Found {len(similar_tags)} tags matching content for article {article_id}")
            
            return similar_tags
            
        except Exception as e:
            logger.error(f"Content-based tagging failed for article {article_id}: {e}")
            return []

    def cleanup_old_unused_tags(self, days: int = 30) -> int:
        """
        Remove tags that haven't been used in X days.

        Args:
            days: Number of days of inactivity before removal

        Returns:
            Number of tags removed
        """
        try:
            fn = getattr(self.store, "cleanup_unused_tags", None)
            if callable(fn):
                return fn(days=days)
        except Exception as e:
            logger.error(f"Error cleaning up tags: {e}")
        return 0
