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

from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml

from feedsummary_core.llm_client import create_llm_client, get_primary_llm_config
from feedsummary_core.persistence import NewsStore, create_store
from feedsummary_core.summarizer.helpers import (
    setup_logging,
    load_prompts,
    parse_lookback_to_seconds,
    load_feeds_into_config,
    _checkpoint_key,
    _checkpoint_path,
    _meta_ckpt_path,
    _load_checkpoint,
    _atomic_write_json,
)
from feedsummary_core.summarizer.ingest import gather_articles_to_store
from feedsummary_core.summarizer.summarizer import (
    summarize_batches_then_meta_with_stats,
    super_meta_from_topic_sections_with_stats,
)

setup_logging()
logger = logging.getLogger(__name__)


def _published_ts(a: dict) -> int:
    ts = a.get("published_ts")
    if isinstance(ts, int) and ts > 0:
        return ts
    fa = a.get("fetched_at")
    if isinstance(fa, int) and fa > 0:
        return fa
    return 0


def _primary_llm_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = get_primary_llm_config(config)
    return cfg if isinstance(cfg, dict) else {}


def _summary_doc_id(created_ts: int, job_id: Optional[int]) -> str:
    dt = datetime.fromtimestamp(created_ts)
    base = dt.strftime("sum_%Y%m%d_%H%M")
    return f"{base}_job{job_id}" if job_id is not None else base


def _persist_summary_doc(store: NewsStore, doc: Dict[str, Any]) -> Any:
    fn = getattr(store, "save_summary_doc", None)
    if not callable(fn):
        raise RuntimeError("Store saknar save_summary_doc() för summary_docs.")
    return fn(doc)


def _render_prompt_template(template_text: str, values: Dict[str, Any]) -> str:
    out = str(template_text or "")
    for k, v in values.items():
        out = out.replace("{" + k + "}", str(v))
    return out


async def _run_prompt_package_step_on_text(
    *,
    config: Dict[str, Any],
    llm,
    package_name: str,
    step: str,
    summary_text: str,
    lookback: str,
    from_ts: int,
    to_ts: int,
) -> str:
    """
    Befintlig helper som återanvänds av compose_summary_docs().

    Stöd:
      - step='title'      => title_system + title_user_template
      - step='ingress'    => super_meta_system + super_meta_user_template
      - step='proofread'  => proofread_system + proofread_user_template
      - step='revise'     => revise_system + revise_user_template
    """
    prompts = load_prompts(config, package=package_name)

    if step == "title":
        sys_key = "title_system"
        user_key = "title_user_template"
    elif step == "ingress":
        sys_key = "super_meta_system"
        user_key = "super_meta_user_template"
    elif step == "proofread":
        sys_key = "proofread_system"
        user_key = "proofread_user_template"
    elif step == "revise":
        sys_key = "revise_system"
        user_key = "revise_user_template"
    else:
        raise ValueError(f"Okänt promptsteg: {step}")

    sys_p = str(prompts.get(sys_key) or "").strip()
    user_t = str(prompts.get(user_key) or "").strip()
    if not sys_p or not user_t:
        return ""

    from_date = datetime.fromtimestamp(int(from_ts)).strftime("%Y-%m-%d") if from_ts else ""
    to_date = datetime.fromtimestamp(int(to_ts)).strftime("%Y-%m-%d") if to_ts else ""

    user = _render_prompt_template(
        user_t,
        {
            "summary": summary_text,
            "batch_summaries": summary_text,
            "topic_summaries": summary_text,
            "lookback": lookback,
            "from_date": from_date,
            "to_date": to_date,
        },
    )

    out = await llm.chat(
        [{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    return str(out or "").strip()


def _extract_summary_doc_parts(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(doc.get("id") or ""),
        "title": str(doc.get("title") or "").strip(),
        "summary": str(doc.get("summary") or "").strip(),
        "sources": list(doc.get("sources") or []),
        "sources_snapshots": list(doc.get("sources_snapshots") or []),
        "from": int(doc.get("from") or 0),
        "to": int(doc.get("to") or 0),
        "selection": dict(doc.get("selection") or {}),
    }


def _build_composed_summary_text(
    *,
    sections: List[Dict[str, Any]],
    ingress: Optional[str],
) -> str:
    parts: List[str] = []
    if ingress and ingress.strip():
        parts.append(ingress.strip())
        parts.append("")

    for sec in sections:
        tag = str(sec.get("tag") or "").strip()
        body = str(sec.get("summary") or "").strip()
        if not body:
            continue
        parts.append(f"## {tag}")
        parts.append("")
        parts.append(body)
        parts.append("")

    return "\n".join(parts).strip()


def _prepend_ingress(summary_text: str, ingress: Optional[str]) -> str:
    body = str(summary_text or "").strip()
    intro = str(ingress or "").strip()
    if intro and body:
        return f"{intro}\n\n{body}"
    if intro:
        return intro
    return body


def _dedupe_keep_order(items: List[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for item in items:
        key = repr(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _get_config_sources(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    feeds = config.get("feeds")
    if isinstance(feeds, list) and all(isinstance(x, dict) for x in feeds):
        return feeds  # type: ignore[return-value]
    return []


def _set_config_sources(config: Dict[str, Any], sources: List[Dict[str, Any]]) -> None:
    config["feeds"] = sources


def _name_of(s: Dict[str, Any]) -> str:
    return str(s.get("name") or s.get("title") or s.get("label") or "").strip()


def _topics_of(s: Dict[str, Any]) -> List[str]:
    t = s.get("topics")
    if isinstance(t, list):
        out = [str(x).strip() for x in t if str(x).strip()]
        return out
    if isinstance(t, str) and t.strip():
        return [t.strip()]

    t2 = s.get("topic")
    if isinstance(t2, str) and t2.strip():
        return [t2.strip()]

    return []


def _source_topics_map(config: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for s in _get_config_sources(config):
        n = _name_of(s)
        if not n:
            continue
        out[n] = _topics_of(s)
    return out


def _apply_overrides(config: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not overrides:
        return config

    cfg = copy.deepcopy(config)

    lookback = overrides.get("lookback")
    if isinstance(lookback, str) and lookback.strip():
        ingest = cfg.setdefault("ingest", {})
        if isinstance(ingest, dict):
            ingest["lookback"] = lookback.strip()

    selected = overrides.get("sources")
    if isinstance(selected, list) and selected:
        selected_set = {str(x) for x in selected if str(x).strip()}
        all_sources = _get_config_sources(cfg)
        filtered = [s for s in all_sources if _name_of(s) in selected_set]
        _set_config_sources(cfg, filtered)

    selected_topics = overrides.get("topics")
    if (
        (not (isinstance(selected, list) and selected))
        and isinstance(selected_topics, list)
        and selected_topics
    ):
        wanted = {str(t).strip() for t in selected_topics if str(t).strip()}
        if wanted:
            all_sources = _get_config_sources(cfg)

            def has_topic(s: Dict[str, Any]) -> bool:
                ts = set(_topics_of(s))
                return bool(ts.intersection(wanted))

            filtered = [s for s in all_sources if has_topic(s)]
            _set_config_sources(cfg, filtered)

    # OBS:
    # Tag-stöd antas finnas längre ned i stacken enligt nuvarande design.
    # Om urvalet inte redan hanteras där behöver denna funktion utökas
    # med motsvarande filtrering för overrides["tags"].

    prompt_pkg = overrides.get("prompt_package")
    if isinstance(prompt_pkg, str) and prompt_pkg.strip():
        p = cfg.setdefault("prompts", {})
        if isinstance(p, dict):
            p["selected"] = prompt_pkg.strip()

    return cfg


def _selected_source_names(config: Dict[str, Any]) -> List[str]:
    srcs = _get_config_sources(config)
    out: List[str] = []
    for s in srcs:
        n = _name_of(s)
        if n:
            out.append(n)
    return out


def _selected_topics_from_config(config: Dict[str, Any]) -> List[str]:
    topics: List[str] = []
    seen = set()
    for s in _get_config_sources(config):
        for t in _topics_of(s):
            if t not in seen:
                seen.add(t)
                topics.append(t)
    topics.sort(key=lambda x: x.lower())
    return topics


def _selected_prompt_package(config: Dict[str, Any]) -> str:
    p = config.get("prompts") or {}
    if isinstance(p, dict):
        return str(p.get("selected") or "").strip()
    return ""


def _selection_doc(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "lookback": str((config.get("ingest") or {}).get("lookback") or ""),
        "sources": _selected_source_names(config),
        "topics": _selected_topics_from_config(config),
        "prompt_package": _selected_prompt_package(config),
    }


def _select_articles_for_summary(
    config: Dict[str, Any], store: NewsStore, *, limit: int = 2000
) -> List[dict]:
    ingest = config.get("ingest") or {}
    lookback = str(ingest.get("lookback") or "").strip()
    sources = _selected_source_names(config)

    now = int(time.time())
    since_ts = 0
    if lookback:
        since_ts = now - parse_lookback_to_seconds(lookback)

    list_by_filter = getattr(store, "list_articles_by_filter", None)
    if callable(list_by_filter) and since_ts > 0 and sources:
        rows = list_by_filter(sources=sources, since_ts=since_ts, until_ts=now, limit=limit)
        rows.sort(key=_published_ts)  # type: ignore
        return rows  # type: ignore

    list_articles = getattr(store, "list_articles", None)
    if callable(list_articles):
        rows = list_articles(limit=limit)
    else:
        rows = store.list_unsummarized_articles(limit=limit)

    if sources:
        srcset = set(sources)
        rows = [a for a in rows if a.get("source") in srcset]  # type: ignore
    if since_ts > 0:
        rows = [a for a in rows if _published_ts(a) >= since_ts]  # type: ignore

    rows.sort(key=_published_ts)  # type: ignore
    return rows[:limit]  # type: ignore


def _primary_topic_for_article(a: Dict[str, Any], topic_map: Dict[str, List[str]]) -> str:
    src = str(a.get("source") or "").strip()
    ts = topic_map.get(src) or []
    if ts:
        return ts[0]
    return "Okategoriserat"


def _group_articles_by_primary_topic(
    articles: List[dict],
    topic_map: Dict[str, List[str]],
) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for a in articles:
        t = _primary_topic_for_article(a, topic_map)
        groups.setdefault(t, []).append(a)
    for t, items in groups.items():
        items.sort(key=_published_ts)
        groups[t] = items
    return groups


def _topic_order(groups: Dict[str, List[dict]]) -> List[str]:
    def key(t: str) -> Tuple[int, int, str]:
        if t == "Okategoriserat":
            return (999999, 1, t.lower())
        return (len(groups.get(t) or []) * -1, 0, t.lower())

    return sorted(list(groups.keys()), key=key)


def _fmt_dt_hm(ts: int) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")


def _build_sources_appendix_markdown(snapshots: List[Dict[str, Any]]) -> str:
    if not snapshots:
        return ""

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in snapshots:
        src = str(s.get("source") or "").strip() or "Okänd källa"
        groups[src].append(s)

    if not groups:
        return ""

    out: List[str] = []
    out.append("## Källor")
    out.append("")

    for src in sorted(groups.keys(), key=lambda x: x.lower()):
        items = groups[src]
        items = sorted(items, key=lambda x: int(x.get("published_ts") or 0), reverse=True)

        out.append(f"### {src}")
        out.append("")

        for it in items:
            title = str(it.get("title") or "").strip() or "(utan titel)"
            url = str(it.get("url") or "").strip()
            pts = int(it.get("published_ts") or 0)
            dt = _fmt_dt_hm(pts) if pts else ""

            line = title
            if dt:
                line = f"{line} — {dt}"

            out.append(f"- {line}")
            if url:
                out.append(f"  {url}")

        out.append("")

    return "\n".join(out).strip() + "\n"


def _strip_sources_appendix_from_summary(summary_text: str) -> str:
    """
    Tar bort den lokala källsektionen från ett summary-dokument.

    Vi utgår från att källappendix byggs deterministiskt sist i dokumentet
    med rubriken '## Källor'.
    """
    text = str(summary_text or "").rstrip()

    marker = "\n## Källor\n"
    idx = text.find(marker)
    if idx >= 0:
        return text[:idx].rstrip()

    if text.startswith("## Källor\n"):
        return ""

    return text


def _snapshot_topic_map_for_job(
    store: NewsStore,
    job_id: int,
    *,
    topic_map: Dict[str, List[str]],
    selection: Dict[str, Any],
    overrides: Optional[Dict[str, Any]],
) -> None:
    try:
        store.update_job(
            job_id,
            selection=selection,
            source_topics_map=topic_map,
            overrides=overrides or {},
        )
    except Exception as e:
        logger.warning("Kunde inte spara job context (selection/topic_map) i store: %s", e)


def _load_job_context(store: NewsStore, job_id: int) -> Dict[str, Any]:
    try:
        j = store.get_job(job_id) or {}
        if isinstance(j, dict):
            return j
    except Exception:
        pass
    return {}


def _write_job_corpus_checkpoint(
    config: Dict[str, Any], store: NewsStore, job_id: int, articles: List[dict]
) -> None:
    try:
        key = _checkpoint_key(job_id, [])
        cp_path = _checkpoint_path(config, key)
        payload = {
            "kind": "job_corpus",
            "created_at": int(time.time()),
            "job_id": int(job_id),
            "checkpoint_key": key,
            "article_ids": [str(a.get("id") or "") for a in articles if a.get("id")],
        }
        _atomic_write_json(cp_path, payload)
    except Exception as e:
        logger.warning("Kunde inte skriva job-corpus checkpoint (resume kan bli instabilt): %s", e)


def _load_ordered_articles_from_job_checkpoint(
    config: Dict[str, Any], store: NewsStore, job_id: int
) -> Tuple[List[str], List[dict]]:
    key = _checkpoint_key(job_id, [])
    cp_path = _checkpoint_path(config, key)
    meta_path = _meta_ckpt_path(config, key)

    cp = None
    if meta_path.exists():
        cp = _load_checkpoint(meta_path)
    if not cp and cp_path.exists():
        cp = _load_checkpoint(cp_path)

    if not cp:
        raise RuntimeError(f"Ingen checkpoint hittades för job {job_id} ({cp_path} / {meta_path})")

    article_ids = cp.get("article_ids") or []
    if not article_ids:
        raise RuntimeError(f"Checkpoint saknar article_ids för job {job_id}")

    articles = store.get_articles_by_ids(article_ids)
    if not articles:
        raise RuntimeError("Kunde inte ladda artiklar från store för checkpointens article_ids")

    by_id = {str(a.get("id")): a for a in articles if a.get("id")}
    ordered = [by_id[i] for i in article_ids if i in by_id]
    return article_ids, ordered


def _default_summary_title(*, lookback: str, from_ts: int, to_ts: int) -> str:
    lb = (lookback or "").strip()
    if from_ts and to_ts:
        a = datetime.fromtimestamp(int(from_ts)).strftime("%Y-%m-%d")
        b = datetime.fromtimestamp(int(to_ts)).strftime("%Y-%m-%d")
        span = a if a == b else f"{a}–{b}"
        return f"Nyhetssammanfattning {span}" + (f" ({lb})" if lb else "")
    return "Nyhetssammanfattning" + (f" ({lb})" if lb else "")


async def _generate_summary_title(
    *,
    config: Dict[str, Any],
    llm,
    summary_text: str,
    from_ts: int,
    to_ts: int,
    selection: Dict[str, Any],
) -> str:
    lookback = str((selection or {}).get("lookback") or "").strip()
    fallback = _default_summary_title(lookback=lookback, from_ts=from_ts, to_ts=to_ts)

    prompts = load_prompts(config)
    sys_p = str(prompts.get("title_system") or "").strip()
    user_t = str(prompts.get("title_user_template") or "").strip()
    if not sys_p or not user_t:
        return fallback

    from_date = datetime.fromtimestamp(int(from_ts)).strftime("%Y-%m-%d") if from_ts else ""
    to_date = datetime.fromtimestamp(int(to_ts)).strftime("%Y-%m-%d") if to_ts else ""

    try:
        user = user_t.format(
            summary=(summary_text or "").strip(),
            lookback=lookback,
            from_date=from_date,
            to_date=to_date,
        )
    except Exception:
        return fallback

    try:
        out = await llm.chat(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        title = str(out or "").strip()
    except Exception:
        return fallback

    if not title:
        return fallback

    title = title.splitlines()[0].strip().strip('"').strip("'").strip()
    if not title:
        return fallback

    if len(title) > 120:
        title = title[:120].rstrip() + "…"
    return title


def _topic_concurrency(config: Dict[str, Any], topic_count: int) -> int:
    b = config.get("batching", {}) or {}
    raw = b.get("topic_max_workers", b.get("topic_workers", None))
    try:
        n = int(raw) if raw is not None else 4
    except Exception:
        n = 4

    if n < 1:
        n = 1
    if n > 16:
        n = 16
    return min(n, max(1, int(topic_count)))


async def _summarize_and_persist_like_refresh(
    *,
    config: Dict[str, Any],
    store: NewsStore,
    llm,
    job_id: Optional[int],
    articles: List[dict],
    topic_map: Dict[str, List[str]],
    selection: Dict[str, Any],
) -> Any:
    groups = _group_articles_by_primary_topic(articles, topic_map)
    topics = _topic_order(groups)
    created_ts = int(time.time())

    if len(topics) <= 1:
        meta_text, stats = await summarize_batches_then_meta_with_stats(
            config, articles, llm=llm, store=store, job_id=job_id
        )

        ids = [a.get("id") for a in articles if a.get("id")]

        pts = [_published_ts(a) for a in articles]
        pts2 = [p for p in pts if p > 0]
        from_ts = min(pts2) if pts2 else 0
        to_ts = max(pts2) if pts2 else 0

        snapshots = [
            {
                "id": a.get("id"),
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "source": a.get("source", ""),
                "published_ts": _published_ts(a),
                "content_hash": a.get("content_hash", ""),
            }
            for a in articles
        ]

        appendix = _build_sources_appendix_markdown(snapshots)
        if appendix:
            meta_text = (meta_text or "").rstrip() + "\n\n" + appendix

        title = await _generate_summary_title(
            config=config,
            llm=llm,
            summary_text=meta_text or "",
            from_ts=from_ts,
            to_ts=to_ts,
            selection=selection,
        )

        summary_doc: Dict[str, Any] = {
            "id": _summary_doc_id(created_ts, job_id),
            "title": title,
            "created": created_ts,
            "kind": "summary",
            "llm": {
                "provider": _primary_llm_cfg(config).get("provider", "unknown"),
                "model": _primary_llm_cfg(config).get("model", "unknown"),
                "temperature": 0.2,
                "max_output_tokens": int(_primary_llm_cfg(config).get("max_output_tokens") or 0),
            },
            "prompts": load_prompts(config),
            "batching": config.get("batching", {}) or {},
            "sources": ids,
            "sources_snapshots": snapshots,
            "from": from_ts,
            "to": to_ts,
            "summary": meta_text,
            "meta": {
                "batch_total": int(stats.get("batch_total") or 0),
                "trims": int(stats.get("trims") or 0),
                "drops": int(stats.get("drops") or 0),
                "meta_budget_tokens": int(stats.get("meta_budget_tokens") or 0),
            },
            "selection": dict(selection or {}),
        }

        return _persist_summary_doc(store, summary_doc)

    stitched_parts: List[str] = []
    lookback_str = str((config.get("ingest") or {}).get("lookback") or "").strip()

    stitched_parts.append("# Sammanfattning per ämnesområde")
    if lookback_str:
        stitched_parts.append(f"_Tidsfönster: {lookback_str}_")
    stitched_parts.append("")

    pts_all = [_published_ts(a) for a in articles]
    pts_all2 = [p for p in pts_all if p > 0]
    overall_from = min(pts_all2) if pts_all2 else 0
    overall_to = max(pts_all2) if pts_all2 else 0

    topic_items: List[Tuple[int, str, List[dict]]] = []
    for i, topic in enumerate(topics, start=1):
        items = groups.get(topic) or []
        if not items:
            continue
        topic_items.append((i, topic, items))

    max_workers = _topic_concurrency(config, len(topic_items))
    sem = asyncio.Semaphore(max_workers)

    async def _run_one_topic(i: int, topic: str, items: List[dict]) -> Dict[str, Any]:
        async with sem:
            if job_id is not None:
                try:
                    store.update_job(
                        job_id,
                        message=f"Summerar ämnesområden parallellt ({max_workers} workers). "
                        f"Startar {i}/{len(topics)}: {topic} ({len(items)} artiklar)...",
                    )
                except Exception:
                    pass

            topic_meta, topic_stats = await summarize_batches_then_meta_with_stats(
                config, items, llm=llm, store=store, job_id=job_id
            )

            ids = [a.get("id") for a in items if a.get("id")]

            snaps = [
                {
                    "id": a.get("id"),
                    "title": a.get("title", ""),
                    "url": a.get("url", ""),
                    "source": a.get("source", ""),
                    "published_ts": _published_ts(a),
                    "content_hash": a.get("content_hash", ""),
                    "topic": topic,
                }
                for a in items
            ]

            pts = [_published_ts(a) for a in items]
            pts2 = [p for p in pts if p > 0]
            from_ts = min(pts2) if pts2 else 0
            to_ts = max(pts2) if pts2 else 0

            return {
                "topic": topic,
                "order_i": i,
                "from": from_ts,
                "to": to_ts,
                "sources": ids,
                "sources_snapshots": snaps,
                "summary": topic_meta,
                "stats": topic_stats,
            }

    tasks = [
        asyncio.create_task(_run_one_topic(i, topic, items)) for (i, topic, items) in topic_items
    ]
    results = await asyncio.gather(*tasks)

    by_topic: Dict[str, Dict[str, Any]] = {r["topic"]: r for r in results}
    sections: List[Dict[str, Any]] = []
    all_ids: List[str] = []
    all_snaps: List[Dict[str, Any]] = []

    for topic in topics:
        r = by_topic.get(topic)
        if not r:
            continue

        topic_stats = r.get("stats") or {}
        ids = r.get("sources") or []
        snaps = r.get("sources_snapshots") or []
        topic_meta = r.get("summary") or ""

        all_ids.extend(ids)
        all_snaps.extend(snaps)

        sections.append(
            {
                "topic": topic,
                "from": int(r.get("from") or 0),
                "to": int(r.get("to") or 0),
                "sources": ids,
                "sources_snapshots": snaps,
                "summary": topic_meta,
                "meta": {
                    "batch_total": int(topic_stats.get("batch_total") or 0),
                    "trims": int(topic_stats.get("trims") or 0),
                    "drops": int(topic_stats.get("drops") or 0),
                    "meta_budget_tokens": int(topic_stats.get("meta_budget_tokens") or 0),
                },
            }
        )

        stitched_parts.append(f"## {topic}")
        stitched_parts.append("")
        stitched_parts.append(str(topic_meta).strip())
        stitched_parts.append("")

    stitched_summary = "\n".join(stitched_parts).strip() + "\n"

    overview_text = ""
    overview_stats: Dict[str, Any] = {
        "super_meta_budget_tokens": 0,
        "super_meta_enabled": 0,
    }
    try:
        overview_text, overview_stats = await super_meta_from_topic_sections_with_stats(
            config=config, sections=sections, llm=llm, store=store, job_id=job_id
        )
    except KeyError:
        overview_text = ""
        overview_stats = {"super_meta_budget_tokens": 0, "super_meta_enabled": 0}

    final_summary = stitched_summary
    if overview_text.strip():
        final_summary = overview_text.strip() + "\n\n" + stitched_summary.strip() + "\n"

    appendix = _build_sources_appendix_markdown(all_snaps)
    if appendix:
        final_summary = final_summary.rstrip() + "\n\n" + appendix

    title = await _generate_summary_title(
        config=config,
        llm=llm,
        summary_text=final_summary or "",
        from_ts=overall_from,
        to_ts=overall_to,
        selection=selection,
    )

    summary_doc = {
        "id": _summary_doc_id(created_ts, job_id),
        "title": title,
        "created": created_ts,
        "kind": "summary",
        "llm": {
            "provider": _primary_llm_cfg(config).get("provider", "unknown"),
            "model": _primary_llm_cfg(config).get("model", "unknown"),
            "temperature": 0.2,
            "max_output_tokens": int(_primary_llm_cfg(config).get("max_output_tokens") or 0),
        },
        "prompts": load_prompts(config),
        "batching": config.get("batching", {}) or {},
        "sources": list(dict.fromkeys([x for x in all_ids if x])),
        "sources_snapshots": all_snaps,
        "from": overall_from,
        "to": overall_to,
        "overview": overview_text,
        "summary": final_summary,
        "sections": sections,
        "meta": {
            "super_meta_enabled": int(overview_stats.get("super_meta_enabled") or 0),
            "super_meta_budget_tokens": int(overview_stats.get("super_meta_budget_tokens") or 0),
        },
        "selection": dict(selection or {}),
    }

    return _persist_summary_doc(store, summary_doc)


def _load_summary_doc(store: NewsStore, summary_id: str) -> Dict[str, Any]:
    fn = getattr(store, "get_summary_doc", None)
    if not callable(fn):
        raise RuntimeError("Store saknar get_summary_doc()")
    doc = fn(summary_id)
    if not isinstance(doc, dict):
        raise RuntimeError(f"Kunde inte läsa summary_doc {summary_id}")
    return doc


async def compose_summary_docs(
    *,
    config: Dict[str, Any],
    store: NewsStore,
    llm,
    job_id: Optional[int],
    name: str,
    sections: List[Dict[str, str]],
    proofread_package: Optional[str] = None,
    ingress_package: Optional[str] = None,
    title_package: Optional[str] = None,
) -> str:
    """
    Sammanfogar redan genererade summary_docs utan att göra ny ingest eller ny artikel-summering.

    Tänkt att användas av worker-lagret efter att flera 'triggered'-scheman körts som vanliga jobb.
    """
    loaded_sections: List[Dict[str, Any]] = []
    all_sources: List[str] = []
    all_snapshots: List[Dict[str, Any]] = []
    from_candidates: List[int] = []
    to_candidates: List[int] = []

    lookback = str((config.get("ingest") or {}).get("lookback") or "").strip()

    for spec in sections:
        summary_id = str(spec.get("summary_id") or "").strip()
        if not summary_id:
            continue

        doc = _load_summary_doc(store, summary_id)
        parts = _extract_summary_doc_parts(doc)

        heading = (
            str(spec.get("schedule") or "").strip()
            or str(spec.get("tag") or "").strip()
            or str(spec.get("promptpackage") or "").strip()
        )

        body_text = _strip_sources_appendix_from_summary(str(parts.get("summary") or ""))

        loaded_sections.append(
            {
                "tag": heading,
                "summary_id": summary_id,
                "schedule": str(spec.get("schedule") or "").strip(),
                "promptpackage": str(spec.get("promptpackage") or "").strip(),
                "summary": body_text,
            }
        )

        all_sources.extend(list(parts.get("sources") or []))
        all_snapshots.extend(list(parts.get("sources_snapshots") or []))

        if int(parts.get("from") or 0) > 0:
            from_candidates.append(int(parts["from"]))
        if int(parts.get("to") or 0) > 0:
            to_candidates.append(int(parts["to"]))

    if not loaded_sections:
        raise RuntimeError("compose_summary_docs: inga summary docs att sammanfoga")

    overall_from = min(from_candidates) if from_candidates else 0
    overall_to = max(to_candidates) if to_candidates else 0

    merged_without_ingress = _build_composed_summary_text(
        sections=loaded_sections,
        ingress=None,
    )

    revised_summary_body = merged_without_ingress
    proofread_applied = False
    revise_applied = False
    if proofread_package:
        proofread_text = ""
        with contextlib.suppress(Exception):
            proofread_text = await _run_prompt_package_step_on_text(
                config=config,
                llm=llm,
                package_name=proofread_package,
                step="proofread",
                summary_text=revised_summary_body,
                lookback=lookback,
                from_ts=overall_from,
                to_ts=overall_to,
            )
        if proofread_text:
            revised_summary_body = proofread_text
            proofread_applied = True

        revised_text = ""
        with contextlib.suppress(Exception):
            revised_text = await _run_prompt_package_step_on_text(
                config=config,
                llm=llm,
                package_name=proofread_package,
                step="revise",
                summary_text=revised_summary_body,
                lookback=lookback,
                from_ts=overall_from,
                to_ts=overall_to,
            )
        if revised_text:
            revised_summary_body = revised_text
            revise_applied = True

    ingress_text = ""
    if ingress_package:
        with contextlib.suppress(Exception):
            ingress_text = await _run_prompt_package_step_on_text(
                config=config,
                llm=llm,
                package_name=ingress_package,
                step="ingress",
                summary_text=revised_summary_body,
                lookback=lookback,
                from_ts=overall_from,
                to_ts=overall_to,
            )

    final_summary_body = _prepend_ingress(revised_summary_body, ingress_text or None)

    title_text = ""
    if title_package:
        with contextlib.suppress(Exception):
            title_text = await _run_prompt_package_step_on_text(
                config=config,
                llm=llm,
                package_name=title_package,
                step="title",
                summary_text=final_summary_body,
                lookback=lookback,
                from_ts=overall_from,
                to_ts=overall_to,
            )
    if not title_text:
        title_text = _default_summary_title(
            lookback=lookback,
            from_ts=overall_from,
            to_ts=overall_to,
        )

    created_ts = int(time.time())
    appendix = _build_sources_appendix_markdown(_dedupe_keep_order(all_snapshots))

    final_summary = final_summary_body
    if appendix:
        final_summary = final_summary.rstrip() + "\n\n" + appendix
    summary_doc: Dict[str, Any] = {
        "id": _summary_doc_id(created_ts, job_id),
        "title": title_text,
        "created": created_ts,
        "kind": "summary",
        "llm": {
            "provider": _primary_llm_cfg(config).get("provider", "unknown"),
            "model": _primary_llm_cfg(config).get("model", "unknown"),
            "temperature": 0.2,
            "max_output_tokens": int(_primary_llm_cfg(config).get("max_output_tokens") or 0),
        },
        "prompts": {"_package": "composed"},
        "batching": config.get("batching", {}) or {},
        "sources": _dedupe_keep_order([x for x in all_sources if x]),
        "sources_snapshots": _dedupe_keep_order(all_snapshots),
        "from": overall_from,
        "to": overall_to,
        "overview": ingress_text or "",
        "summary": final_summary,
        "sections": [
            {
                "tag": s["tag"],
                "schedule": s["schedule"],
                "promptpackage": s["promptpackage"],
                "summary_id": s["summary_id"],
            }
            for s in loaded_sections
        ],
        "meta": {
            "composed": True,
            "proofread_package": proofread_package or "",
            "ingress_package": ingress_package or "",
            "title_package": title_package or "",
            "proofread_applied": int(proofread_applied),
            "revise_applied": int(revise_applied),
        },
        "selection": {
            "name": name,
            "contents": sections,
            "lookback": lookback,
        },
    }

    return str(_persist_summary_doc(store, summary_doc))


async def run_pipeline(
    config_path: str = "config.yaml",
    job_id: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    llm=None,
) -> Optional[Any]:
    try:
        if config_dict is None:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        else:
            config = config_dict

        config = load_feeds_into_config(config, base_config_path=config_path)
        config = _apply_overrides(config, overrides)

        store = create_store(config.get("store", {}))
        if llm is None:
            llm = create_llm_client(config)

        if job_id is not None:
            store.update_job(
                job_id,
                status="running",
                started_at=int(time.time()),
                message="Startar ingest...",
            )

        ins, upd = await gather_articles_to_store(config, store, job_id=job_id)

        if job_id is not None:
            store.update_job(
                job_id,
                message=f"Ingest klart. Inserted={ins}, Updated={upd}. Förbereder summering...",
            )

        to_sum = _select_articles_for_summary(config, store, limit=2000)
        if not to_sum:
            if job_id is not None:
                store.update_job(
                    job_id,
                    status="done",
                    finished_at=int(time.time()),
                    message="Klart: inga artiklar matchade urvalet (lookback/källor/ämnen).",
                )
            return None

        topic_map = _source_topics_map(config)
        selection = _selection_doc(config)

        if job_id is not None:
            _snapshot_topic_map_for_job(
                store,
                job_id,
                topic_map=topic_map,
                selection=selection,
                overrides=overrides,
            )
            _write_job_corpus_checkpoint(config, store, int(job_id), to_sum)

        summary_doc_id = await _summarize_and_persist_like_refresh(
            config=config,
            store=store,
            llm=llm,
            job_id=job_id,
            articles=to_sum,
            topic_map=topic_map,
            selection=selection,
        )

        if job_id is not None:
            store.update_job(
                job_id,
                status="done",
                finished_at=int(time.time()),
                message=f"Klart: summerade {len(to_sum)} artiklar.",
                summary_id=str(summary_doc_id),
            )

        return summary_doc_id

    except Exception as e:
        if job_id is not None:
            try:
                store.update_job(
                    job_id,
                    status="failed",
                    finished_at=int(time.time()),
                    message=f"Fel: {e}",
                )
            except Exception:
                pass
        raise


async def run_resume_job(
    *,
    config: Dict[str, Any],
    store: NewsStore,
    llm,
    job_id: int,
) -> str:
    jid = int(job_id)
    try:
        ctx = _load_job_context(store, jid)

        selection = ctx.get("selection")
        if not isinstance(selection, dict):
            selection = _selection_doc(config)

        topic_map = ctx.get("source_topics_map")
        if not isinstance(topic_map, dict):
            topic_map = _source_topics_map(config)

        _article_ids, ordered_articles = _load_ordered_articles_from_job_checkpoint(
            config, store, jid
        )

        summary_doc_id = await _summarize_and_persist_like_refresh(
            config=config,
            store=store,
            llm=llm,
            job_id=jid,
            articles=ordered_articles,
            topic_map=topic_map,
            selection=selection,
        )

        try:
            store.update_job(
                jid,
                status="done",
                finished_at=int(time.time()),
                message=f"Resume klart: summerade {len(ordered_articles)} artiklar.",
                summary_id=str(summary_doc_id),
            )
        except Exception:
            pass

        return str(summary_doc_id)

    except Exception as e:
        try:
            store.update_job(
                jid,
                status="failed",
                finished_at=int(time.time()),
                message=f"Resume fel: {e}",
            )
        except Exception:
            pass
        raise


if __name__ == "__main__":
    asyncio.run(run_pipeline("config.yaml"))
