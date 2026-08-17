# Tag-Based Summary Generation

## Overview

Du kan nu skapa sammanfattningar baserat på taggar. Det är perfekt för att:

- Skapa fokuserade sammanfattningar på specifika ämnen
- Spela in alla artiklar från en viss säkerhetskategori
- Generera rapporter för specifika hot-aktörer eller CVE:er
- Samla och sammanfatta relaterade artiklar

## How It Works

1. **Skapa en tagg-baserad sammanfattning**
   - Specificera en eller flera taggar
   - Välj matchningsmetod (ANY eller ALL)
   - Systemet hämtar alla matchande artiklar
   - En ny sammanfattning genereras från dessa artiklar

2. **Lagra sammanfattningen**
   - Sammanfattningen sparas i databasen
   - Källartiklarna lagras som referenser
   - Full metadata inkluderas (tid, tag-information, etc.)

## Basic Usage

### Generate Summary from Single Tag

```python
from feedsummary_core.persistence import create_store
from feedsummary_core.llm_client import create_llm_client
from feedsummary_core.summarizer.tagging_integration import generate_summary_from_tags
import asyncio

async def main():
    store = create_store("sqlite://news.db")
    llm_client = create_llm_client(config)
    
    # Generate summary for all articles tagged "cybersecurity"
    summary = await generate_summary_from_tags(
        store=store,
        llm_client=llm_client,
        tag_names=["cybersecurity"],
        config=config,
        match_mode="any"
    )
    
    if summary:
        print(f"Summary ID: {summary['id']}")
        print(f"Title: {summary['title']}")
        print(f"Articles: {summary['article_count']}")
        print(f"\n{summary['summary']}")
    
        # Save to database (optional)
        store.save_summary_doc(summary)

asyncio.run(main())
```

### Generate Summary from Multiple Tags

```python
# Find articles with ANY of these tags
summary = await generate_summary_from_tags(
    store=store,
    llm_client=llm_client,
    tag_names=["CVE-2024-1234", "vulnerability", "zero-day"],
    config=config,
    match_mode="any"  # Articles with any of these tags
)
```

### Find Articles with ALL Tags

```python
# Find articles with ALL of these tags
summary = await generate_summary_from_tags(
    store=store,
    llm_client=llm_client,
    tag_names=["malware", "Russia"],
    config=config,
    match_mode="all"  # Articles with all these tags
)
```

## API Reference

### `generate_summary_from_tags()`

```python
async def generate_summary_from_tags(
    store: NewsStore,
    llm_client: LLMClient,
    tag_names: List[str],
    config: Dict[str, Any],
    match_mode: str = "any",
) -> Optional[Dict[str, Any]]:
```

**Args:**
- `store`: NewsStore instance
- `llm_client`: LLM client for summarization
- `tag_names`: List of tag names to search for
- `config`: Configuration dictionary
- `match_mode`: 
  - `"any"` - Articles with ANY of the tags (OR)
  - `"all"` - Articles with ALL of the tags (AND)

**Returns:**
- Dictionary with summary document if successful
- None if no articles found or error

**Summary Document Structure:**
```python
{
    "id": "tag_sum_20260814_1430",  # Auto-generated ID
    "title": "Summary: cybersecurity, vulnerability",
    "created": 1723617000,
    "kind": "tag-based-summary",
    "tags_used": ["cybersecurity", "vulnerability"],
    "match_mode": "any",
    "article_count": 23,
    "source_article_ids": ["art_1", "art_2", ...],
    "summary": "Full summary text...",
    "from": 1723600000,  # Earliest article timestamp
    "to": 1723620000,    # Latest article timestamp
    "meta": {
        "batch_total": 4,
        "trims": 2,
        "drops": 0
    }
}
```

### `store.get_articles_by_tags()`

Get articles directly without generating summary:

```python
articles = store.get_articles_by_tags(
    tag_names=["malware", "APT28"],
    match_mode="any"
)

for article in articles:
    print(f"{article['title']} - {article['source']}")
```

## Workflow Examples

### Example 1: Weekly Security Report

```python
import asyncio
from datetime import datetime

async def generate_weekly_security_report():
    store = create_store(config.get("store", {}))
    llm_client = create_llm_client(config)
    
    # Generate summaries for key security topics
    topics = [
        ["CVE", "vulnerability"],
        ["malware"],
        ["data-breach"],
        ["threat-intelligence"],
    ]
    
    summaries = []
    for topic_tags in topics:
        summary = await generate_summary_from_tags(
            store=store,
            llm_client=llm_client,
            tag_names=topic_tags,
            config=config,
            match_mode="all"
        )
        
        if summary:
            summaries.append(summary)
            store.save_summary_doc(summary)
    
    print(f"Generated {len(summaries)} topic summaries for weekly report")
    return summaries

asyncio.run(generate_weekly_security_report())
```

### Example 2: Threat Actor Report

```python
async def generate_threat_actor_report(threat_actor_name: str):
    summary = await generate_summary_from_tags(
        store=store,
        llm_client=llm_client,
        tag_names=[threat_actor_name],
        config=config,
        match_mode="any"
    )
    
    if summary:
        # Save to database
        doc_id = store.save_summary_doc(summary)
        
        # Create report file
        with open(f"threat_report_{threat_actor_name}.md", "w") as f:
            f.write(f"# {summary['title']}\n\n")
            f.write(f"Generated: {datetime.fromtimestamp(summary['created'])}\n\n")
            f.write(f"Articles: {summary['article_count']}\n\n")
            f.write(summary['summary'])
        
        return doc_id
```

### Example 3: CVE Summary

```python
async def summarize_cve(cve_id: str):
    """Create summary of all articles mentioning a specific CVE"""
    
    summary = await generate_summary_from_tags(
        store=store,
        llm_client=llm_client,
        tag_names=[cve_id],
        config=config,
        match_mode="any"
    )
    
    if summary:
        print(f"\n=== {cve_id} Summary ===")
        print(f"Articles: {summary['article_count']}")
        print(f"Date range: {summary['from']} - {summary['to']}")
        print(f"\n{summary['summary']}")
```

## Database Queries

### Find Tags by Usage

```sql
SELECT name, COUNT(*) as usage
FROM tags
LEFT JOIN article_tags ON tags.id = article_tags.tag_id
GROUP BY tags.id
ORDER BY usage DESC
LIMIT 20;
```

### Articles by Multiple Tags

```sql
SELECT DISTINCT article_id
FROM article_tags at
WHERE tag_id IN (
    SELECT id FROM tags WHERE name IN ('cybersecurity', 'vulnerability')
)
GROUP BY article_id
HAVING COUNT(*) = 2;  -- Has all tags
```

### Tag Distribution by Category

```sql
SELECT category, COUNT(*) as tag_count
FROM tags
GROUP BY category;
```

## Performance Notes

- **First call**: May be slower due to LLM summarization
- **Multiple calls**: Same tags return consistent results (deterministic summarization)
- **Large result sets**: 100+ articles might take longer
- **Database**: Indexed queries ensure fast tag lookups

## Error Handling

```python
try:
    summary = await generate_summary_from_tags(...)
except Exception as e:
    logger.error(f"Failed to generate summary: {e}")
    summary = None
```

Common errors:
- **No articles found**: Check if articles are properly tagged
- **LLM timeout**: Increase timeout in config
- **Empty summary**: Check if articles have content
- **Tag not found**: Verify tag names are lowercase

## Next Steps

1. **Browse tags**: Query `get_all_tags()` to see available tags
2. **Generate summaries**: Use `generate_summary_from_tags()` for topics
3. **Compare reports**: Generate same summary at different times
4. **Export**: Save summaries and export to markdown/PDF

For more information:
- See [TAGGING_SYSTEM.md](TAGGING_SYSTEM.md) for tagging reference
- See [TAGGING_INTEGRATION_GUIDE.md](TAGGING_INTEGRATION_GUIDE.md) for pipeline integration
