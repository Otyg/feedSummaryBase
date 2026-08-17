# Tagging System - Automatic Integration Guide

## Overview

The tagging system is now **automatically integrated** into the summarization pipeline. No additional configuration needed beyond the normal setup.

## What Happens Automatically

When you run summarization:

```
Pipeline Start
  ↓
Ingest articles
  ↓
Summarize articles ← LLM processes all articles
  ↓
**Tag articles** ← New! Automatic tagging starts
  ↓
  For each article:
    - Extract tags via LLM
    - Match against existing tags (prefer existing/general)
    - Allow new tags for domain entities only
    - Store tags in database
  ↓
Update job status (includes tag count)
  ↓
Pipeline Complete
```

## How to Use

### 1. Run Normal Summarization Pipeline

Just run the pipeline as usual. Tagging happens automatically:

```python
from feedsummary_core.summarizer.main import run_pipeline

# Tagging runs automatically after summarization
summary_id = await run_pipeline(
    config_path="config.yaml",
    job_id=123  # Optional job tracking
)
```

### 2. Monitor Tagging Progress

The tagging process logs progress:

```
[Job 123] Starting automatic tagging of 50 articles...
[Job 123] Tagging complete: 45/50 articles tagged
```

And job status includes tagging info:

```
"Klart: summerade 50 artiklar. Taggade: 45."
```

### 3. Pre-populate Common Tags (Optional)

For better tag matching, add common tags beforehand:

```python
from feedsummary_core.persistence import create_store
from feedsummary_core.summarizer.tagging_integration import add_predefined_tags

store = create_store(config.get("store", {}))

common_tags = [
    {"name": "cybersecurity", "category": "GENERAL"},
    {"name": "vulnerability", "category": "GENERAL"},
    {"name": "malware", "category": "GENERAL"},
    {"name": "threat-intelligence", "category": "GENERAL"},
    {"name": "data-breach", "category": "GENERAL"},
    {"name": "network-security", "category": "GENERAL"},
]

count = add_predefined_tags(store, common_tags)
print(f"Added {count} predefined tags")
```

## What Works with Both Backends

The tagging system works with **both** persistence backends:

### SqliteStore (SQL-based)
- Tables: `tags`, `article_tags`
- Indexes on `name` and `category`
- Recommended for production
- Automatic table creation on first use

### TinyDbStore (JSON-based)  
- Tables: `tags`, `article_tags`
- JSON format in `news_docs.json`
- Good for development
- Automatic table creation on first use

## Configuration

### Adjust Tagging Behavior

Edit default values in `tag_articles_safe()` call in `main.py`:

```python
# Current defaults in main.py
tagged_count = await tag_articles_safe(
    store=store,
    llm_client=llm,
    article_ids=article_ids,
    config=config,
    job_id=job_id,
    max_tags_per_article=5,  # ← Change this for more/fewer tags
)
```

### Skip Previously Tagged Articles

Currently enabled by default - articles with existing tags are skipped:

```python
# In tagging_integration.py
result = await tag_articles(
    ...
    skip_if_already_tagged=True,  # Skip already tagged
    ...
)
```

## Manual Tagging (Advanced)

For custom tagging outside the pipeline:

```python
from feedsummary_core.summarizer.tagging_integration import tag_articles

# Tag specific articles manually
results = await tag_articles(
    store=store,
    llm_client=llm_client,
    article_ids=["art_123", "art_456"],
    config=config,
    max_tags_per_article=5,
)

for article_id, tags in results.items():
    print(f"{article_id}: {[t['name'] for t in tags]}")
```

## Database Queries

### Check tagged articles

```python
import sqlite3

con = sqlite3.connect("news.db")
con.row_factory = sqlite3.Row

# Get articles with their tags
articles_tags = con.execute("""
    SELECT DISTINCT at.article_id, COUNT(t.id) as tag_count
    FROM article_tags at
    JOIN tags t ON at.tag_id = t.id
    GROUP BY at.article_id
    ORDER BY tag_count DESC
""").fetchall()

for row in articles_tags:
    print(f"Article {row['article_id']}: {row['tag_count']} tags")
```

### Most used tags

```python
tags_usage = con.execute("""
    SELECT t.name, t.category, COUNT(at.article_id) as usage
    FROM tags t
    LEFT JOIN article_tags at ON t.id = at.tag_id
    GROUP BY t.id
    ORDER BY usage DESC
    LIMIT 20
""").fetchall()

for row in tags_usage:
    print(f"{row['name']} ({row['category']}): {row['usage']} uses")
```

### Find articles by tag

```python
articles = con.execute("""
    SELECT DISTINCT at.article_id
    FROM article_tags at
    JOIN tags t ON at.tag_id = t.id
    WHERE t.name = 'cybersecurity'
""").fetchall()
```

## Error Handling

Tagging is designed to not interrupt the pipeline:

- If LLM fails → article is skipped with warning
- If database error → logged but pipeline continues
- If parsing fails → logged but pipeline continues

Check logs for any tagging-related issues:

```python
import logging
logging.getLogger("feedsummary_core.summarizer.tagging").setLevel(logging.DEBUG)
```

## Performance Notes

- **First run**: May be slower due to LLM calls and tag creation
- **Subsequent runs**: Faster as existing tags are reused
- **Batch size**: Each article creates one LLM call (can be optimized)
- **Database**: Indexes ensure fast lookups

## Troubleshooting

### Tags not being created

1. Check LLM is returning proper JSON format
2. Verify database has been initialized (tables created)
3. Check logs for specific errors

### Too many duplicate tags

- Add predefined common tags before running
- Lower similarity threshold (advanced)
- Run tag cleanup to remove old tags

### Performance is slow

- Ensure database indexes are present
- Consider running tagging in off-peak times
- Check LLM response time

## Next Steps

1. **Start using**: Just run the normal pipeline - tagging works!
2. **Optimize**: Add predefined tags for your domain
3. **Monitor**: Check logs and database queries
4. **Extend**: Build custom UI to browse and manage tags

For more details, see:
- [TAGGING_SYSTEM.md](TAGGING_SYSTEM.md) - Technical reference
- [TAGGING_QUICKSTART.md](TAGGING_QUICKSTART.md) - Quick start
