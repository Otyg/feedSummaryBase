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
#    contributors may be used to endorse or promote products derived from
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
# ----------------------------
import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlsplit

import aiohttp
import feedparser
import trafilatura
from aiolimiter import AsyncLimiter
from lxml import html as lxml_html
from lxml.etree import _Element, tostring
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from feedsummary_core.persistence import NewsStore
from feedsummary_core.summarizer.helpers import (
    RateLimitError,
    compute_content_hash,
    entry_published_ts,
    load_feeds_into_config,
    parse_lookback_to_seconds,
    set_job,
    stable_id,
)

logger = logging.getLogger(__name__)


def _norm_cat(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _entry_categories(entry: feedparser.FeedParserDict) -> Set[str]:
    cats: Set[str] = set()

    try:
        c = getattr(entry, "category", None)
        if isinstance(c, str) and c.strip():
            cats.add(_norm_cat(c))
    except Exception:
        pass

    try:
        cs = getattr(entry, "categories", None)
        if isinstance(cs, list):
            for x in cs:
                if isinstance(x, str) and x.strip():
                    cats.add(_norm_cat(x))
                elif isinstance(x, dict):
                    term = x.get("term") or x.get("label") or x.get("name")
                    if isinstance(term, str) and term.strip():
                        cats.add(_norm_cat(term))
    except Exception:
        pass

    try:
        tags = getattr(entry, "tags", None)
        if isinstance(tags, list):
            for t in tags:
                if isinstance(t, dict):
                    term = t.get("term") or t.get("label") or t.get("name")
                    if isinstance(term, str) and term.strip():
                        cats.add(_norm_cat(term))
    except Exception:
        pass

    return cats


def _passes_category_filter(entry: feedparser.FeedParserDict, feed_cfg: Dict[str, Any]) -> bool:
    inc = feed_cfg.get("category_include") or feed_cfg.get("categories_include")
    exc = feed_cfg.get("category_exclude") or feed_cfg.get("categories_exclude")

    inc_set: Set[str] = set()
    exc_set: Set[str] = set()

    if isinstance(inc, list):
        inc_set = {_norm_cat(str(x)) for x in inc if str(x).strip()}
    elif isinstance(inc, str) and inc.strip():
        inc_set = {_norm_cat(inc)}

    if isinstance(exc, list):
        exc_set = {_norm_cat(str(x)) for x in exc if str(x).strip()}
    elif isinstance(exc, str) and exc.strip():
        exc_set = {_norm_cat(exc)}

    if not inc_set and not exc_set:
        return True

    cats = _entry_categories(entry)

    if exc_set and (cats & exc_set):
        return False

    if inc_set and not (cats & inc_set):
        return False

    return True


async def fetch_rss(feed_url: str, session: aiohttp.ClientSession) -> feedparser.FeedParserDict:
    """Download and parse one RSS or Atom feed."""

    async with session.get(feed_url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
        resp.raise_for_status()
        content = await resp.read()
        logger.info(f"{feed_url} hämtad")
    return feedparser.parse(content)


def _normalize_extraction_domain(value: str) -> str:
    """Normalize a configured domain or URL hostname for exact matching."""
    candidate = str(value or "").strip().lower().rstrip(".")
    if "://" in candidate:
        candidate = (urlsplit(candidate).hostname or "").lower().rstrip(".")
    if candidate.startswith("www."):
        candidate = candidate[4:]
    return candidate


def _domain_extraction_rule(
    url: str,
    extraction_config: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the exact normalized domain rule for a URL, if configured."""
    if not isinstance(extraction_config, dict):
        return None
    domains = extraction_config.get("domains")
    if not isinstance(domains, dict):
        return None

    hostname = _normalize_extraction_domain(urlsplit(url).hostname or "")
    for configured_domain, rule in domains.items():
        if (
            _normalize_extraction_domain(str(configured_domain)) == hostname
            and isinstance(rule, dict)
        ):
            return rule
    return None


def _select_html_fragment(html: str, xpath: str, url: str) -> Optional[str]:
    """Select and serialize the first element matched by a configured XPath."""
    try:
        tree = lxml_html.fromstring(html, base_url=url)
        matches = tree.xpath(xpath)
    except Exception as error:
        logger.warning("Invalid content_xpath for %s (%s): %s", url, xpath, error)
        return None

    elements = [match for match in matches if isinstance(match, _Element)]
    if not elements:
        logger.warning("content_xpath matched no elements for %s: %s", url, xpath)
        return None
    if len(elements) > 1:
        logger.warning(
            "content_xpath matched %d elements for %s; using the first: %s",
            len(elements),
            url,
            xpath,
        )
    return tostring(elements[0], encoding="unicode", method="html")


def extract_text_from_html(
    html: str,
    url: str,
    extraction_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Extract readable text, optionally preselecting a domain-specific fragment."""
    rule = _domain_extraction_rule(url, extraction_config)
    extract_kwargs: Dict[str, Any] = {
        "url": url,
        "include_comments": False,
        "include_tables": False,
    }
    if rule:
        for option in (
            "include_comments",
            "include_tables",
            "include_links",
            "include_images",
            "favor_precision",
            "favor_recall",
            "deduplicate",
            "target_language",
            "prune_xpath",
        ):
            if option in rule:
                extract_kwargs[option] = rule[option]

        xpath = str(rule.get("content_xpath") or "").strip()
        fragment = _select_html_fragment(html, xpath, url) if xpath else None
        if fragment:
            extracted = trafilatura.extract(fragment, **extract_kwargs)
            if extracted and extracted.strip():
                logger.debug("Used domain-specific content_xpath for %s: %s", url, xpath)
                return extracted.strip()
            logger.warning(
                "Trafilatura returned no text for selected fragment at %s; using full page",
                url,
            )

    extracted = trafilatura.extract(html, **extract_kwargs)
    return (extracted or "").strip()


def extract_text_from_rss_entry(entry: feedparser.FeedParserDict) -> str:
    """Extract text from RSS entry (summary or description) as fallback."""

    # Try summary first
    summary = getattr(entry, "summary", None) or getattr(entry, "description", None)
    if isinstance(summary, str) and summary.strip():
        # Remove HTML tags if present
        text = re.sub(r"<[^>]+>", " ", summary)
        text = " ".join(text.split())  # Normalize whitespace
        return text.strip()

    # Try content field (some feeds use this)
    content = getattr(entry, "content", None)
    if isinstance(content, list) and content:
        for item in content:
            if isinstance(item, dict):
                value = item.get("value")
                if isinstance(value, str) and value.strip():
                    text = re.sub(r"<[^>]+>", " ", value)
                    text = " ".join(text.split())
                    return text.strip()

    return ""


async def fetch_article_html(url: str, session: aiohttp.ClientSession, timeout_s: int) -> str:
    """Fetch raw article HTML and raise :class:`RateLimitError` on HTTP 429."""

    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
        if resp.status == 429:
            ra = resp.headers.get("Retry-After")
            retry_after = float(ra) if ra and ra.isdigit() else None
            body = await resp.text(errors="ignore")
            raise RateLimitError(429, retry_after=retry_after, body=body[:500])

        # Non-retriable client errors should be skipped immediately to avoid wasted retries
        if 400 <= resp.status < 500:
            body = await resp.text(errors="ignore")
            raise aiohttp.ClientResponseError(
                request_info=resp.request_info,
                history=resp.history,
                status=resp.status,
                message=f"HTTP {resp.status} client error",
                headers=resp.headers,
            )

        resp.raise_for_status()
        return await resp.text(errors="ignore")


def _is_transient_article_error(exc: BaseException) -> bool:
    """Retry only on transient fetch errors, not permanent client errors like 404."""
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, aiohttp.ClientResponseError):
        return 500 <= exc.status < 600
    return isinstance(exc, aiohttp.ClientError)


@retry(
    wait=wait_exponential_jitter(initial=1, max=30),
    stop=stop_after_attempt(6),
    retry=retry_if_exception(_is_transient_article_error),
    reraise=True,
)
async def guarded_fetch_article(url: str, session: aiohttp.ClientSession, timeout_s: int) -> str:
    """Fetch an article with retry handling for rate limits and transient HTTP issues."""

    try:
        return await fetch_article_html(url, session, timeout_s)
    except RateLimitError as e:
        if e.retry_after:
            await asyncio.sleep(min(e.retry_after, 60))
        raise


async def gather_articles_to_store(
    config: Dict[str, Any],
    store: NewsStore,
    job_id: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Ingest (RSS -> article text -> store)

    - config.ingest.lookback: "24h", "3d", "2w", "90m" osv.
    - max_items_per_feed används fortfarande som safety cap.
    - Per-feed filter:
        category_include / category_exclude (matchar entry.tags[].term m.fl.)

    ✅ Artikel-store ska bara hålla artiklar:
      - vi sätter inte summarized/summarized_at här längre
    """
    config = load_feeds_into_config(config, base_config_path="config.yaml")
    feeds = config.get("feeds", [])
    ingest_cfg = config.get("ingest") or {}
    extraction_config = ingest_cfg.get("extraction")
    lookback = ingest_cfg.get("lookback")
    max_items = int(ingest_cfg.get("max_items_per_feed", config.get("max_items_per_feed", 8)))

    timeout_s = int(config.get("article_timeout_s", 20))
    http_limiter = AsyncLimiter(max_rate=6, time_period=1)
    headers = {"User-Agent": "news-summarizer/2.1 (personal; rate-limited)"}

    inserted = 0
    updated = 0
    connector = aiohttp.TCPConnector(limit=50, ttl_dns_cache=300)
    async with aiohttp.ClientSession(
        headers=headers,
        connector=connector,
        max_line_size=16384,
        max_field_size=32768,
    ) as session:
        for f in feeds:
            name = f["name"]
            feed_url = f["url"]
            set_job(f"Läser RSS: {name}", job_id, store)

            try:
                logger.info(f"Hämtar RSS: {name}")
                feed = await fetch_rss(feed_url, session)
            except Exception as e:
                logger.warning(f"Kunde inte läsa RSS: {name} ({feed_url}) -> {e}")
                continue

            entries = list(feed.entries or [])

            cutoff_ts: Optional[int] = None
            if lookback:
                now = int(time.time())
                cutoff_ts = now - parse_lookback_to_seconds(str(lookback))

                filtered: List[feedparser.FeedParserDict] = []
                for entry in entries:
                    ts = entry_published_ts(entry)
                    if ts is None:
                        continue
                    if ts >= cutoff_ts:
                        filtered.append(entry)

                filtered.sort(key=lambda e: entry_published_ts(e) or 0, reverse=True)
                if max_items and len(filtered) > max_items:
                    filtered = filtered[:max_items]

                logger.info(
                    "RSS %s: %d entries (filtered=%d, lookback=%s, cap=%s)",
                    name,
                    len(entries),
                    len(filtered),
                    lookback,
                    max_items,
                )
                entries_to_process = filtered
            else:
                entries_to_process = entries[:max_items]
                logger.info(
                    "RSS %s: %d entries (cap=%s, lookback=none)",
                    name,
                    len(entries),
                    max_items,
                )

            # category filter per feed
            before_cat = len(entries_to_process)
            if isinstance(f, dict) and (
                f.get("category_include")
                or f.get("categories_include")
                or f.get("category_exclude")
                or f.get("categories_exclude")
            ):
                tmp: List[feedparser.FeedParserDict] = []
                for e in entries_to_process:
                    if _passes_category_filter(e, f):
                        tmp.append(e)
                entries_to_process = tmp

                logger.info(
                    "RSS %s: category filter applied (kept=%d/%d include=%s exclude=%s)",
                    name,
                    len(entries_to_process),
                    before_cat,
                    f.get("category_include") or f.get("categories_include"),
                    f.get("category_exclude") or f.get("categories_exclude"),
                )

            for entry in entries_to_process:
                link = getattr(entry, "link", None)
                if not link:
                    continue

                aid = stable_id(link)
                title = (getattr(entry, "title", "") or "").strip()
                published = getattr(entry, "published", "") or getattr(entry, "updated", "") or ""

                existing = store.get_article(aid)

                text = None
                fetch_error = None

                # Try to fetch original article
                try:
                    async with http_limiter:
                        html = await guarded_fetch_article(link, session, timeout_s)
                    text = extract_text_from_html(html, link, extraction_config)
                except Exception as e:
                    fetch_error = e
                    logger.debug(f"Kunde inte hämta artikel {link}: {e}")

                # If original article fetch failed, try RSS content as fallback
                if not text or len(text) < 200:
                    rss_text = extract_text_from_rss_entry(entry)
                    if rss_text and len(rss_text) >= 100:
                        text = rss_text
                        logger.info(f"Använder RSS-innehål för: {title}")

                # Skip if we still don't have sufficient content
                if not text or len(text) < 100:
                    if fetch_error:
                        if isinstance(fetch_error, RetryError) and fetch_error.last_attempt:
                            last = fetch_error.last_attempt.exception()
                            logger.warning(
                                "Artikel hoppades över: %s -> %s (efter flera försök, sista=%s, no RSS fallback)",
                                link,
                                fetch_error,
                                last,
                            )
                        else:
                            logger.warning(
                                "Artikel hoppades över: %s -> %s (no RSS fallback)",
                                link,
                                fetch_error,
                            )
                    else:
                        logger.debug(f"Artikel hoppades över (otillräckligt innehål): {link}")
                    continue

                try:
                    chash = compute_content_hash(title, link, text)
                    ts = entry_published_ts(entry)
                    doc = {
                        "id": aid,
                        "source": name,
                        "title": title,
                        "url": link,
                        "published": published,
                        "published_ts": ts or 0,
                        "fetched_at": int(time.time()),
                        "text": text,
                        "content_hash": chash,
                    }

                    if existing is None:
                        store.upsert_article(doc)
                        inserted += 1
                    else:
                        if existing.get("content_hash") != chash:
                            store.upsert_article(doc)
                            updated += 1

                except Exception as e:
                    logger.warning("Fel vid sparande av artikel: %s -> %s", link, e)
    logger.info(f"Articles: {str(inserted)} inserted, {str(updated)} updated")
    return inserted, updated
