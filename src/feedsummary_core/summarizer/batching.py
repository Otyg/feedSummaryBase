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
#

import asyncio
import hashlib
import logging
import math
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from feedsummary_core.summarizer.helpers import text_clip

logger = logging.getLogger(__name__)


def embedding_source_hash(text: str) -> str:
    """Return a stable fingerprint for the exact text sent to the embedding model."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def cached_embedding(
    document: dict,
    text: str,
    embedding_model: str = "",
) -> Optional[List[float]]:
    """Return a persisted embedding only when its model and source text still match."""
    vector = document.get("embedding_vector")
    if not isinstance(vector, list) or not vector:
        return None
    if not all(isinstance(value, (int, float)) for value in vector):
        return None
    if document.get("embedding_source_hash") != embedding_source_hash(text):
        return None
    if str(document.get("embedding_model") or "") != str(embedding_model or ""):
        return None
    return [float(value) for value in vector]


def article_embedding_text(article: dict, embedding_text_chars: int = 2000) -> str:
    """Build the canonical text used for persisted article embeddings."""
    title = str(article.get("title", "") or "").strip()
    article_body = (
        article.get("text", "") or article.get("content", "") or article.get("summary", "")
    )
    body = text_clip(article_body, max(1, int(embedding_text_chars))).strip()
    return "\n\n".join(part for part in (title, body) if part)


async def ensure_article_embedding(
    article: dict,
    embed: Callable[[str], Awaitable[List[float]]],
    *,
    store: Any,
    embedding_model: str,
    embedding_text_chars: int = 2000,
) -> Optional[List[float]]:
    """Reuse or create the canonical embedding and persist it on the article."""
    text = article_embedding_text(article, embedding_text_chars)
    if not text:
        return None
    persisted = cached_embedding(article, text, embedding_model)
    if persisted is not None:
        return persisted
    try:
        vector = await embed(text)
        if not vector or not all(isinstance(value, (int, float)) for value in vector):
            return None
        normalized = [float(value) for value in vector]
        article_id = str(article.get("id") or "").strip()
        source_hash = embedding_source_hash(text)
        update_embedding = getattr(store, "update_article_embedding", None)
        if article_id and callable(update_embedding):
            update_embedding(
                article_id,
                normalized,
                model=embedding_model,
                source_hash=source_hash,
            )
        article.update(
            {
                "embedding_vector": normalized,
                "embedding_model": embedding_model,
                "embedding_source_hash": source_hash,
            }
        )
        return normalized
    except Exception as exc:
        logger.warning(
            "Kunde inte skapa eller spara artikel-embedding för %s: %s",
            article.get("id"),
            exc,
        )
        return None


def batch_articles(
    articles: List[dict],
    max_chars_per_batch: int,
    max_articles_per_batch: int,
    article_clip_chars: int = 2500,
) -> List[List[dict]]:
    """Split article docs into bounded batches after clipping each article body."""

    batches: List[List[dict]] = []
    current: List[dict] = []
    current_chars = 0

    for a in articles:
        per_article_text = text_clip(a.get("text", ""), article_clip_chars)
        estimated = len(per_article_text) + len(a.get("title", "")) + len(a.get("url", "")) + 200

        if current and (
            current_chars + estimated > max_chars_per_batch
            or len(current) >= max_articles_per_batch
        ):
            batches.append(current)
            current = []
            current_chars = 0

        a2 = dict(a)
        a2["text"] = per_article_text
        current.append(a2)
        current_chars += estimated

    if current:
        batches.append(current)

    return batches


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return -1.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return -1.0
    return dot / (left_norm * right_norm)


def _similarity_components(
    embeddings: Sequence[Optional[Sequence[float]]], threshold: float
) -> List[List[int]]:
    """Return stable, transitive groups of articles whose embeddings are similar."""
    parents = list(range(len(embeddings)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(embeddings)):
        if not embeddings[left]:
            continue
        for right in range(left + 1, len(embeddings)):
            if (
                embeddings[right]
                and _cosine_similarity(embeddings[left], embeddings[right]) >= threshold
            ):
                union(left, right)

    grouped: Dict[int, List[int]] = {}
    for index in range(len(embeddings)):
        grouped.setdefault(find(index), []).append(index)
    return sorted(grouped.values(), key=lambda group: group[0])


def _batch_similarity_groups(
    groups: Sequence[Sequence[dict]],
    *,
    max_chars_per_batch: int,
    max_articles_per_batch: int,
) -> List[List[dict]]:
    """Pack similarity groups together whenever the configured limits allow it."""
    batches: List[List[dict]] = []
    current: List[dict] = []

    def flush() -> None:
        nonlocal current
        if current:
            batches.append(current)
            current = []

    for group_items in groups:
        group = list(group_items)
        group_chars = sum(_estimate_article_chars(article) for article in group)
        group_fits = (
            not max_articles_per_batch or len(group) <= max_articles_per_batch
        ) and group_chars <= max_chars_per_batch

        if group_fits:
            combined_fits = (
                not max_articles_per_batch or len(current) + len(group) <= max_articles_per_batch
            ) and _batch_chars(current) + group_chars <= max_chars_per_batch
            if current and not combined_fits:
                flush()
            current.extend(group)
            continue

        # A similarity group can itself exceed a hard limit. Keep as much of it
        # together as possible, splitting only where the normal batcher must.
        flush()
        for article in group:
            if current and not _can_fit_in_batch(
                current,
                article,
                max_chars_per_batch=max_chars_per_batch,
                max_articles_per_batch=max_articles_per_batch,
            ):
                flush()
            current.append(article)

    flush()
    return batches


async def group_articles_by_similarity(
    articles: List[dict],
    embed: Callable[[str], Awaitable[List[float]]],
    *,
    embedding_text_chars: int = 2000,
    similarity_threshold: float = 0.78,
    max_concurrency: int = 4,
    store: Any = None,
    embedding_model: str = "",
) -> List[List[dict]]:
    """Embed articles and return stable groups that likely describe the same story."""
    embedding_inputs: List[str] = []
    for article in articles:
        embedding_inputs.append(article_embedding_text(article, embedding_text_chars))

    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def embed_one(article: dict, text: str) -> Optional[List[float]]:
        if not text:
            return None
        persisted = cached_embedding(article, text, embedding_model)
        if persisted is not None:
            return persisted
        try:
            async with semaphore:
                vector = await embed(text)
            if vector and all(isinstance(value, (int, float)) for value in vector):
                normalized = [float(value) for value in vector]
                article_id = str(article.get("id") or "").strip()
                update_embedding = getattr(store, "update_article_embedding", None)
                if article_id and callable(update_embedding):
                    try:
                        source_hash = embedding_source_hash(text)
                        update_embedding(
                            article_id,
                            normalized,
                            model=embedding_model,
                            source_hash=source_hash,
                        )
                        article.update(
                            {
                                "embedding_vector": normalized,
                                "embedding_model": embedding_model,
                                "embedding_source_hash": source_hash,
                            }
                        )
                    except Exception as exc:
                        logger.warning(
                            "Kunde inte spara artikel-embedding för %s: %s",
                            article_id,
                            exc,
                        )
                return normalized
        except Exception as exc:
            logger.warning("Kunde inte skapa artikel-embedding: %s", exc)
        return None

    embeddings = await asyncio.gather(
        *(embed_one(article, text) for article, text in zip(articles, embedding_inputs))
    )
    usable = sum(vector is not None for vector in embeddings)
    if usable < 2:
        logger.warning(
            "Semantisk gruppering hoppades över: endast %d/%d embeddings kunde skapas.",
            usable,
            len(articles),
        )
        return [[article] for article in articles]

    components = _similarity_components(embeddings, similarity_threshold)
    logger.info(
        "Semantisk gruppering placerade %d artiklar i %d ämnesgrupper (tröskel %.2f).",
        len(articles),
        len(components),
        similarity_threshold,
    )
    return [[articles[index] for index in component] for component in components]


async def batch_articles_by_similarity(
    articles: List[dict],
    embed: Callable[[str], Awaitable[List[float]]],
    *,
    max_chars_per_batch: int,
    max_articles_per_batch: int,
    article_clip_chars: int = 2500,
    embedding_text_chars: int = 2000,
    similarity_threshold: float = 0.78,
    max_concurrency: int = 4,
    store: Any = None,
    embedding_model: str = "",
) -> List[List[dict]]:
    """Embed articles, group likely duplicate stories, and create bounded batches."""
    clipped: List[dict] = []
    for article in articles:
        item = dict(article)
        item["text"] = text_clip(item.get("text", ""), article_clip_chars)
        clipped.append(item)

    groups = await group_articles_by_similarity(
        clipped,
        embed,
        embedding_text_chars=embedding_text_chars,
        similarity_threshold=similarity_threshold,
        max_concurrency=max_concurrency,
        store=store,
        embedding_model=embedding_model,
    )
    return _batch_similarity_groups(
        groups,
        max_chars_per_batch=max_chars_per_batch,
        max_articles_per_batch=max_articles_per_batch,
    )


# ----------------------------
# Prompt-too-long helpers + stable resume helpers
# ----------------------------
class PromptTooLongStructural(Exception):
    """Raised when a prompt overflow requires structural batch changes, not trimming."""

    def __init__(self, overflow_tokens: int):
        super().__init__(f"prompt too long (structural), overflow={overflow_tokens}")
        self.overflow_tokens = overflow_tokens


def _choose_trim_action(overflow_tokens: int, structural_threshold: int) -> str:
    if overflow_tokens <= 200:
        return "word_trim"
    if overflow_tokens <= structural_threshold:
        return "drop_one_article"
    return "drop_multiple_articles"


def trim_text_tail_by_words(text: str, remove_tokens: int, *, chars_per_token: float) -> str:
    """
    Tar bort från slutet men alltid på whitespace (ordgräns).
    """
    s = text or ""
    if not s:
        return s

    remove_chars = int(max(1, remove_tokens) * chars_per_token)
    if remove_chars >= len(s):
        return ""

    target = max(0, len(s) - remove_chars)
    cut = max(s.rfind(" ", 0, target), s.rfind("\n", 0, target), s.rfind("\t", 0, target))
    if cut <= 0:
        cut = target

    return s[:cut].rstrip() + "\n\n[TRUNCATED FOR CONTEXT WINDOW]\n"


def _trim_last_user_word_boundary(
    messages: List[Dict[str, str]], remove_tokens: int, *, chars_per_token: float
) -> List[Dict[str, str]]:
    out = [dict(m) for m in messages]
    idx = None
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            idx = i
            break
    if idx is None:
        return out
    content = out[idx].get("content") or ""
    out[idx]["content"] = trim_text_tail_by_words(
        content, remove_tokens, chars_per_token=chars_per_token
    )
    return out


def _estimate_article_chars(a: dict) -> int:
    return (
        len(a.get("text", "") or "")
        + len(a.get("title", "") or "")
        + len(a.get("url", "") or "")
        + 200
    )


def _batch_chars(batch: List[dict]) -> int:
    return sum(_estimate_article_chars(x) for x in batch)


def _can_fit_in_batch(
    batch: List[dict], a: dict, *, max_chars_per_batch: int, max_articles_per_batch: int
) -> bool:
    if max_articles_per_batch and len(batch) >= max_articles_per_batch:
        return False
    return (_batch_chars(batch) + _estimate_article_chars(a)) <= max_chars_per_batch


def _move_article_to_tail_batch(
    batches: List[List[dict]],
    a: dict,
    *,
    max_chars_per_batch: int,
    max_articles_per_batch: int,
    avoid_batch: Optional[List[dict]] = None,
) -> None:
    """
    Flytta till sista batch om plats, annars ny batch.
    Viktigt: undvik att lägga tillbaka i samma batch (tail-loop).
    """
    if not batches:
        batches.append([a])
        return

    last = batches[-1]
    if avoid_batch is not None and last is avoid_batch:
        batches.append([a])
        return

    if _can_fit_in_batch(
        last,
        a,
        max_chars_per_batch=max_chars_per_batch,
        max_articles_per_batch=max_articles_per_batch,
    ):
        last.append(a)
    else:
        batches.append([a])


def _batch_article_ids_map(batches_local: List[List[dict]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for i, b in enumerate(batches_local, start=1):
        out[str(i)] = [str(a.get("id", "")) for a in b if a.get("id")]
    return out


def _done_batches_payload(
    done_map_local: Dict[int, str], batches_local: List[List[dict]]
) -> Dict[str, Dict[str, Any]]:
    ids_map = _batch_article_ids_map(batches_local)
    payload: Dict[str, Dict[str, Any]] = {}
    for k, v in sorted(done_map_local.items()):
        sk = str(k)
        payload[sk] = {"article_ids": ids_map.get(sk, []), "summary": v}
    return payload


def _done_map_from_done_batches(cp_done_batches: Dict[str, Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if not isinstance(cp_done_batches, dict):
        return out
    for k, entry in cp_done_batches.items():
        try:
            idx = int(k)
        except Exception:
            continue
        if isinstance(entry, dict) and isinstance(entry.get("summary"), str):
            out[idx] = entry["summary"]
    return out


def _build_batches_from_checkpoint(
    batch_article_ids: Dict[str, Any],
    all_articles: List[dict],
    *,
    clip_chars: int,
) -> List[List[dict]]:
    """
    Återskapa batch-indelning EXAKT från checkpointens batch_article_ids.
    """
    by_id: Dict[str, dict] = {}
    for a in all_articles:
        aid = a.get("id")
        if aid:
            by_id[str(aid)] = a

    def key_int(k: str) -> int:
        try:
            return int(k)
        except Exception:
            return 10**9

    rebuilt: List[List[dict]] = []
    missing: List[str] = []

    for k in sorted(batch_article_ids.keys(), key=key_int):
        ids = batch_article_ids.get(k)
        if not isinstance(ids, list):
            continue

        batch: List[dict] = []
        for aid in ids:
            aid_s = str(aid)
            a = by_id.get(aid_s)
            if not a:
                missing.append(aid_s)
                continue
            a2 = dict(a)
            a2["text"] = text_clip(a2.get("text", ""), clip_chars)
            batch.append(a2)

        if batch:
            rebuilt.append(batch)

    if missing:
        logger.warning(
            "Resume: %d article_ids saknas i store (hoppas över). Ex: %s",
            len(missing),
            ", ".join(missing[:5]),
        )

    if not rebuilt:
        raise RuntimeError("Resume: batch_article_ids fanns men inga batcher kunde återskapas.")

    return rebuilt


def _budgeted_meta_user(
    *,
    prompts: Dict[str, str],
    batch_summaries: List[Tuple[int, str]],
    sources_text: str,
    budget_tokens: int,
    chars_per_token: float,
    lookback: str,
) -> str:
    """
    Bygg meta-user inom en *explicit* tokenbudget.
    Skalar ner via clip-levels, käll-clip och decimering.
    """

    def est_tokens(s: str) -> int:
        return max(1, int(len(s) / chars_per_token))

    def render(batch_block: str, src: str) -> str:
        return prompts["meta_user_template"].format(
            batch_summaries=batch_block,
            sources_list=src,
            lookback=lookback or "okänt tidsfönster",
        )

    # Aggressivare stegar än innan (särskilt decimations)
    sources_levels = [len(sources_text), 6000, 3500, 2000, 1200, 700]
    clip_levels = [4200, 3200, 2400, 1800, 1200, 900, 700, 500, 350, 250]
    decimations = [1, 2, 3, 4, 6, 8, 12]  # 1=alla batcher, 2=varannan, ...

    summaries_desc = sorted(batch_summaries, key=lambda x: x[0], reverse=True)

    for src_lim in sources_levels:
        src2 = (
            sources_text
            if len(sources_text) <= src_lim
            else (sources_text[:src_lim].rstrip() + "…")
        )

        for dec in decimations:
            subset = summaries_desc[::dec]

            for clip_n in clip_levels:
                parts: List[str] = []

                for i, s in subset:
                    s2 = (s or "").strip()
                    if len(s2) > clip_n:
                        s2 = s2[:clip_n].rstrip() + "…"

                    candidate_parts = parts + [f"Batch {i}:\n{s2}"]
                    candidate_block = "\n\n====================\n\n".join(candidate_parts)
                    candidate_user = render(candidate_block, src2)

                    if est_tokens(candidate_user) <= budget_tokens:
                        parts = candidate_parts
                    else:
                        if parts:
                            break
                        continue

                if parts:
                    return render("\n\n====================\n\n".join(parts), src2)

    # Sista utväg: utan batch-summaries
    return render(
        "[Inga batch-summaries kunde inkluderas inom context-budget.]",
        (sources_text[:700].rstrip() + "…") if len(sources_text) > 700 else sources_text,
    )


def _est_user_tokens(s: str, chars_per_token: float) -> int:
    return max(1, int(len(s) / chars_per_token))


def _compact_article_block(a: dict, *, idx: int) -> str:
    """
    Token-cheap per-article block.
    - One-line header with optional source + url
    - No 'Publicerad:' line (saves tokens)
    - Keep body as-is (already clipped upstream in batch_articles())
    """
    title = str(a.get("title", "") or "").strip()
    source = str(a.get("source", "") or "").strip()
    url = str(a.get("url", "") or "").strip()
    text = str(a.get("text", "") or "").strip()

    head = f"[{idx}] {title}" if title else f"[{idx}] (utan titel)"
    # Keep header compact but informative
    if source:
        head += f" ({source})"
    if url:
        head += f" {url}"

    if text:
        return f"{head}\n{text}"
    return head


def build_messages_for_batch(
    *,
    prompts: Dict[str, str],
    batch_index: int,
    batch_total: int,
    batch_items: List[dict],
) -> List[Dict[str, str]]:
    """
    Shared batch prompt builder (used by summarizer + prompt_lab).

    Returns a chat message list compatible with LLMClient.chat.
    """
    parts: List[str] = []
    for i, a in enumerate(batch_items, start=1):
        parts.append(_compact_article_block(a, idx=i))

    corpus = "\n\n---\n\n".join(parts)
    user_content = prompts["batch_user_template"].format(
        batch_index=batch_index,
        batch_total=batch_total,
        articles_corpus=corpus,
    )

    return [
        {"role": "system", "content": prompts["batch_system"]},
        {"role": "user", "content": user_content},
    ]
