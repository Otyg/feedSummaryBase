# Tagging System - Quick Start Guide

## ⚡ Automatic Tagging

**Tagging now runs automatically during the summarization pipeline!**

When you run `run_pipeline()` or `run_resume_job()`, articles are automatically tagged after summarization completes. No additional setup required.

### What Happens Automatically

1. Articles are summarized as usual
2. **After summarization**, each article is automatically tagged
3. Tags are stored in the database
4. Job status includes tagging information

### Checking Tag Progress

Tags are logged during the pipeline:
```
[Job 123] Starting automatic tagging of 50 articles...
[Job 123] Tagging complete: 45/50 articles tagged
```

Job status message will show: `"Klart: summerade 50 artiklar. Taggade: 45."`

## Installation & Setup

The tagging system is already integrated into the codebase. No additional dependencies are needed.

### 1. Initialize the System

```python
from feedsummary_core.persistence import create_store
from feedsummary_core.summarizer.tagging import TagManager

# Initialize store (tables created automatically)
store = create_store("sqlite://news.db")
tag_manager = TagManager(store)
```

### 2. Add Predefined Tags (Optional)

Pre-populate common tags to improve matching:

```python
from feedsummary_core.summarizer.tagging_integration import add_predefined_tags

tags = [
    {"name": "cybersecurity", "category": "GENERAL"},
    {"name": "vulnerability", "category": "GENERAL"},
    {"name": "malware", "category": "GENERAL"},
    {"name": "threat-intelligence", "category": "GENERAL"},
]

add_predefined_tags(store, tags)
```

## Run Summarization (Tagging Included)

```python
import asyncio
from feedsummary_core.summarizer.tagging_integration import tag_article

async def tag_one_article():
    tags = await tag_article(
        store=store,
        llm_client=llm_client,
        article_id="article_123",
        config=config,
        max_tags=5
    )
    print(f"Tagged with: {[t['name'] for t in tags]}")

asyncio.run(tag_one_article())
```

### Tagging Multiple Articles

```python
from feedsummary_core.summarizer.tagging_integration import tag_articles

async def tag_multiple():
    results = await tag_articles(
        store=store,
        llm_client=llm_client,
        article_ids=["art_1", "art_2", "art_3"],
        config=config,
        max_tags_per_article=5,
        skip_if_already_tagged=True
    )
    
    for article_id, tags in results.items():
        print(f"{article_id}: {[t['name'] for t in tags]}")

asyncio.run(tag_multiple())
```

### Retrieving Article Tags

```python
# Get raw tags
tags = store.get_article_tags("article_123")

# Get formatted for display
from feedsummary_core.summarizer.tagging_integration import get_article_tags_for_display
display_tags = get_article_tags_for_display(store, "article_123")
```

### Cleaning Up Old Tags

```python
from feedsummary_core.summarizer.tagging_integration import cleanup_old_tags

# Remove tags unused for 30+ days
removed_count = cleanup_old_tags(store, days=30)
print(f"Cleaned up {removed_count} tags")
```

## How It Works

### The Priority System

When tagging an article, the system:

1. **Extracts candidates** from the article using LLM
2. **Searches for existing tags** that match the candidates
3. **Prioritizes as follows:**
   - ✓ Use existing GENERAL tags (preferred)
   - ✓ Use existing DOMAIN_ENTITY tags
   - ✓ Create NEW tags ONLY for domain entities (CVEs, threat actors, regions, vulnerabilities)
   - ✗ Skip new regular tags (to keep database clean and focused)

### Domain Entities (Exceptions)

The system automatically recognizes and allows new tags for:

- **CVE Numbers**: `CVE-2024-1234`, `CVE-2023-5678`
- **Threat Actors**: APT-28, Lazarus Group, FIN7
- **Geographic Regions**: Russia, North Korea, Eastern Europe
- **Vulnerabilities**: "zero-day", "supply chain", "privilege escalation"

### Similarity Matching

The system finds existing tags that match candidates through:

- Exact matching: `"security"` → `"security"`
- Substring matching: `"network security"` → `"network-security"`
- Character similarity: `"cyber"` + `"security"` → `"cybersecurity"`

## Configuration

### Adjust Similarity Threshold

```python
tag_manager = TagManager(store)

# Higher threshold = stricter matching
similar = tag_manager._find_similar_existing_tags(
    "my_tag",
    similarity_threshold=0.8  # Default is 0.6
)
```

### Customize Domain Entity Detection

Edit `tagging.py` to modify patterns:

```python
# Add CVE variants
CVE_PATTERN = re.compile(r'CVE-\d{4}-\d{4,5}|GHSA-\w{4}-\w{4}-\w{4}')

# Add threat actor keywords
THREAT_ACTOR_KEYWORDS = {
    'APT', 'group', 'gang', 'campaign', 'threat actor',
    'hacker', 'collective', 'organization', 'state-sponsored',
    'syndicate', 'team'  # Add more...
}
```

## Integration with Summarizer

Add tagging to your main summarization workflow:

```python
# In main.py or wherever you summarize articles
from feedsummary_core.summarizer.tagging_integration import tag_articles

async def process_articles():
    # ... summarize articles ...
    
    # Tag the summarized articles
    article_ids = [a["id"] for a in articles]
    await tag_articles(
        store=store,
        llm_client=llm_client,
        article_ids=article_ids,
        config=config,
        max_tags_per_article=5
    )
```

## Troubleshooting

### Tags aren't being created

1. **Check if LLM is returning JSON:** Inspect LLM response format
   ```python
   response = await llm_client.generate(prompt, config)
   # Should contain: {"tags": ["tag1", "tag2"], "reasoning": "..."}
   ```

2. **Verify existing tags:** Make sure database has been initialized
   ```python
   all_tags = store.get_all_tags()
   print(f"Total tags: {len(all_tags)}")
   ```

### Too many similar tags created

- Lower the similarity threshold to find more existing matches
- Add predefined tags for your domain
- Manually clean up duplicates

### Performance issues

- Index check: Database should have indexes on `tags.name` and `article_tags.article_id`
- Consider batch processing: Use `tag_articles()` instead of `tag_article()` in loops

## Database Queries (Manual)

Access the database directly for advanced queries:

```python
import sqlite3

con = sqlite3.connect("news.db")
con.row_factory = sqlite3.Row

# Get most used tags
tags = con.execute("""
    SELECT t.name, COUNT(at.article_id) as usage_count
    FROM tags t
    LEFT JOIN article_tags at ON t.id = at.tag_id
    GROUP BY t.id
    ORDER BY usage_count DESC
    LIMIT 10
""").fetchall()

# Get articles by tag
articles = con.execute("""
    SELECT DISTINCT at.article_id
    FROM article_tags at
    JOIN tags t ON at.tag_id = t.id
    WHERE t.name = 'cybersecurity'
""").fetchall()
```

## Monitoring

### Check tag health

```python
tag_manager = TagManager(store)
all_tags = tag_manager.get_all_tags()

general_count = sum(1 for t in all_tags if t['category'] == 'GENERAL')
domain_count = sum(1 for t in all_tags if t['category'] == 'DOMAIN_ENTITY')

print(f"Tags: {len(all_tags)} total")
print(f"  - GENERAL: {general_count}")
print(f"  - DOMAIN_ENTITY: {domain_count}")
```

### Log tagging statistics

```python
import asyncio
from feedsummary_core.summarizer.tagging_integration import tag_articles

results = await tag_articles(store, llm_client, article_ids, config)

successful = sum(1 for tags in results.values() if tags)
print(f"Tagged {successful}/{len(results)} articles")
```

## Further Reading

- See [TAGGING_SYSTEM.md](TAGGING_SYSTEM.md) for complete documentation
- See [examples/tagging_examples.py](examples/tagging_examples.py) for runnable examples
- Check [src/feedsummary_core/summarizer/tagging.py](src/feedsummary_core/summarizer/tagging.py) for API reference
