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
  * Domain entities (CVEs, vulnerabilities, threat actors, regions)
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

from feedsummary_core.llm_client import LLMClient
from feedsummary_core.persistence import NewsStore

logger = logging.getLogger(__name__)

# Patterns and keywords for relevant tags that can be created automatically
# These include domain entities + general categories that are valuable for tagging
CVE_PATTERN = re.compile(r'CVE-\d{4}-\d{4,5}', re.IGNORECASE)

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
            category: Tag category (GENERAL or DOMAIN_ENTITY)
            description: Optional description

        Returns:
            Tag ID if successful, None otherwise
        """
        if not name or not isinstance(name, str):
            return None

        name = name.strip().lower()
        if not name:
            return None

        # Validate category
        if category not in (self.TAG_CATEGORY_GENERAL, self.TAG_CATEGORY_DOMAIN_ENTITY):
            category = self.TAG_CATEGORY_GENERAL

        # Validate tag name length (max 2 words for GENERAL, unlimited for DOMAIN_ENTITY)
        if not self._is_valid_tag_name(name, category):
            logger.debug(f"[TagValidate] Rejected tag '{name}': exceeds word limit for category {category}")
            return None

        try:
            fn = getattr(self.store, "add_tag", None)
            if callable(fn):
                return fn(name=name, category=category, description=description)
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
                        similar_by_embedding = fn(
                            candidate_embedding,
                            similarity_threshold=max(0.6, similarity_threshold - 0.1),
                            limit=5
                        )
                        
                        if similar_by_embedding:
                            # Log the embedding matches found
                            match_details = "; ".join([
                                f"{t.get('name')} (similarity: {t.get('_similarity_score', 'N/A'):.3f})"
                                for t in similar_by_embedding
                            ])
                            logger.info(f"[TagMatch] Embedding-based match for '{tag_name}': {match_details}")
                            
                            # Prefer GENERAL tags over DOMAIN_ENTITY
                            similar_by_embedding.sort(
                                key=lambda x: x.get("category") == self.TAG_CATEGORY_GENERAL,
                                reverse=True
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

            # Exact match
            if tag_name_str == tag_lower:
                logger.debug(f"[TagMatch] Exact string match: '{tag_name}' == '{tag_name_str}'")
                matches.append((tag, 1.0))
                continue

            # Substring match
            if tag_lower in tag_name_str or tag_name_str in tag_lower:
                logger.debug(f"[TagMatch] Substring match: '{tag_name}' ⊂ '{tag_name_str}'")
                matches.append((tag, 0.9))
                continue

            # Simple Levenshtein-like similarity
            similarity = self._simple_similarity(tag_lower, tag_name_str)
            if similarity >= similarity_threshold:
                logger.debug(f"[TagMatch] Char-set similarity: '{tag_name}' vs '{tag_name_str}' = {similarity:.3f}")
                matches.append((tag, similarity))

        # Sort by similarity (descending) and prefer GENERAL category
        matches.sort(
            key=lambda x: (
                -x[1],  # Higher similarity first
                x[0].get("category") == self.TAG_CATEGORY_GENERAL,  # GENERAL first
            )
        )

        if matches:
            match_details = "; ".join([
                f"{t.get('name')} (similarity: {s:.3f})"
                for t, s in matches[:3]  # Show top 3
            ])
            logger.info(f"[TagMatch] String-based match for '{tag_name}': {match_details}")

        return [tag for tag, _ in matches]

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
        
        # Compute embedding (synchronously in async context)
        try:
            import asyncio
            # Try to get running event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context
                task = loop.create_task(self.llm_client.embed(text))
                # This is risky - we can't await in a sync function
                # We'll just skip caching in sync context
                return None
            except RuntimeError:
                # No running loop - try to create one temporarily
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
        """Calculate a simple similarity score between two strings (0.0-1.0)."""
        if not s1 or not s2:
            return 0.0

        # Common character ratio
        common = len(set(s1) & set(s2))
        total = len(set(s1) | set(s2))
        if total == 0:
            return 0.0

        return common / total

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

        for candidate in candidate_tags:
            if not candidate:
                continue
            
            # Handle both string and dict formats
            if isinstance(candidate, dict):
                tag_name = candidate.get("name", "").strip().lower()
                tag_type = candidate.get("type", "CATEGORY").upper()
                tag_reasoning = candidate.get("reasoning", "")
            else:
                tag_name = str(candidate).strip().lower()
                tag_type = "CATEGORY"  # Default for backward compatibility
                tag_reasoning = ""
            
            if not tag_name or tag_name in processed_tags:
                continue

            processed_tags.add(tag_name)

            # Try to find existing similar tags
            similar_tags = self._find_similar_existing_tags(tag_name)

            if similar_tags:
                # Use the best match (already sorted by priority)
                tag_dict = similar_tags[0].copy()
                tag_dict["reasoning"] = tag_reasoning
                logger.debug(f"[TagSelect] Using existing tag '{tag_dict['name']}' for candidate '{tag_name}'")
                selected_tags.append(tag_dict)
            elif allow_new_tags:
                # Decision logic based on tag type
                should_create = False
                
                if tag_type == "NAMED_ENTITY":
                    # NAMED_ENTITY tags (organizations, groups, actors) are always created
                    # This allows LLM-identified entities like "Apple", "Clop", "APT28" to be tagged automatically
                    should_create = True
                elif tag_type == "CATEGORY":
                    # CATEGORY tags only created if they match our keyword heuristics
                    should_create = self._is_relevant_new_tag(tag_name)
                
                if should_create:
                    # Validate tag name (max 2 words for GENERAL, unlimited for DOMAIN_ENTITY)
                    tag_category = (
                        self.TAG_CATEGORY_DOMAIN_ENTITY 
                        if tag_type == "NAMED_ENTITY" 
                        else self.TAG_CATEGORY_GENERAL
                    )
                    
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

    def extract_tags_from_llm_response(self, response: str) -> List[Dict[str, str]]:
        """
        Extract tags from LLM response.

        Expects JSON format with per-tag reasoning:
        {
            "tags": [
                {"tag": "tag1", "type": "NAMED_ENTITY", "reasoning": "Why this tag..."},
                {"tag": "tag2", "type": "CATEGORY", "reasoning": "Why this tag..."},
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
            List of tags as dicts with 'name', 'type', and optional 'reasoning' keys.
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
                                tag_reasoning = obj.get("reasoning", "")
                                if tag_name:
                                    tags.append({
                                        "name": str(tag_name).strip().lower(),
                                        "type": tag_type.upper() if tag_type else "CATEGORY",
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
                                    tag_reasoning = t.get("reasoning", "")
                                    if tag_name:
                                        tags.append({
                                            "name": str(tag_name).strip().lower(),
                                            "type": tag_type.upper() if tag_type else "CATEGORY",
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
            max_tags: Maximum number of tags to select

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

        # Prepare prompt for tag generation
        prompt = self._build_tagging_prompt(text_context, max_tags)

        try:
            # Use chat() method instead of generate() for compatibility with FallbackLLMClient
            response = await llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.3
            )
            candidate_tags = self.extract_tags_from_llm_response(response)

            # Select best tags using priority logic
            selected = self.select_tags_for_article(
                article_id=article.get("id", ""),
                candidate_tags=candidate_tags[:max_tags * 2],  # Get more than needed
                allow_new_tags=True,
            )

            return selected[:max_tags]  # Limit to max_tags
        except Exception as e:
            logger.error(f"Error generating tags: {e}")
            return []

    def _build_tagging_prompt(self, article_text: str, max_tags: int = 5) -> str:
        """Build a prompt for the LLM to generate tags."""
        # Truncate article text if too long
        if len(article_text) > 2000:
            article_text = article_text[:2000] + "..."

        return f"""Analyze the following article and extract up to {max_tags} relevant tags.

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

IMPORTANT:
- Return tags in lowercase
- Keep tags concise (1-3 words max)
- Focus on the main topics and entities mentioned
- For NAMED_ENTITY tags: include ANY organization/group/actor/product name mentioned, even if uncommon
- For CATEGORY tags: only include if they are central to the article
- Provide a brief explanation for each tag (1-2 sentences max)

Article:
{article_text}

Respond in JSON format:
{{
    "tags": [
        {{"tag": "tag1", "type": "NAMED_ENTITY", "reasoning": "Brief explanation why this tag is relevant"}},
        {{"tag": "tag2", "type": "CATEGORY", "reasoning": "Brief explanation why this tag is relevant"}},
        {{"tag": "tag3", "type": "NAMED_ENTITY", "reasoning": "Brief explanation why this tag is relevant"}}
    ]
}}"""

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
        
        for tag in existing_tags:
            try:
                tag_id = tag.get("id")
                tag_name = tag.get("name", "")
                
                # Skip if already has embedding
                if tag.get("embedding_vector"):
                    continue
                
                if not tag_id or not tag_name:
                    continue
                
                # Generate embedding
                embedding = await self.llm_client.embed(tag_name)
                
                if embedding:
                    # Update tag with embedding
                    fn = getattr(self.store, "update_tag_embedding", None)
                    if callable(fn):
                        success = fn(tag_id, embedding)
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
            # Get article content embedding
            content_embedding = await self.llm_client.embed(article_content)
            
            if not content_embedding:
                logger.warning(f"Failed to embed content for article {article_id}")
                return []
            
            # Find similar tags by embedding
            fn = getattr(self.store, "get_tags_by_embedding_similarity", None)
            if not callable(fn):
                logger.warning("Store doesn't support embedding similarity search")
                return []
            
            similar_tags = fn(
                content_embedding,
                similarity_threshold=similarity_threshold,
                limit=max_tags
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
