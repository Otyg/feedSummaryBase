# feedSummaryBase

Core library for intelligent RSS feed summarization with LLM support and advanced article tagging.

## Features

- **RSS Feed Ingestion**: Parse and fetch articles from RSS feeds with robust error handling
- **Intelligent Summarization**: Generate summaries using local or cloud-based LLMs (Ollama)
- **Article Tagging System**: 
  - Automatic tag extraction and assignment using LLMs
  - Smart tag prioritization (prefer existing tags over new ones, general over specific)
  - Domain entity detection (CVE patterns, threat actors, regions, vulnerabilities)
  - Similarity matching for tag deduplication
- **Tag-Based Summary Generation**: Create focused summaries from articles filtered by tags
- **Multi-Backend Persistence**:
  - SQLite (production-grade SQL storage)
  - TinyDB (JSON-based storage for development)
- **Job Management**: Track and resume summarization jobs
- **Batching & Chunking**: Intelligent content batching with token budget management
- **Proofread & Revise**: LLM-based quality improvement for generated summaries

## Quick Start

### Installation

```bash
pip install feedsummary-core
```

### Basic Usage

```python
import asyncio
from feedsummary_core.persistence import create_store
from feedsummary_core.llm_client import create_llm_client
from feedsummary_core.summarizer.main import run_pipeline

async def main():
    # Run the full pipeline: fetch RSS feeds → summarize → tag
    await run_pipeline("config.yaml")

asyncio.run(main())
```

## Core Modules

### Persistence Layer (`persistence/`)

**NewsStore Protocol** - Abstract interface for persistence backends

**SqliteStore** - Production-grade SQL backend
- Indexed queries for optimal performance
- Support for articles, summaries, jobs, and tags
- WAL mode for concurrent access
- Automatic table initialization

```python
from feedsummary_core.persistence import create_store

store = create_store({"type": "sqlite", "path": "news.db"})
articles = store.get_articles()
tags = store.get_all_tags()
```

**TinyDbStore** - JSON-based backend (development)
- File-based storage (news_docs.json)
- No external dependencies
- Complete feature parity with SqliteStore

### LLM Client (`llm_client/`)

Unified interface for LLM interaction with support for multiple backends:

- **OllamaLocal** - Local Ollama instance
- **OllamaCloud** - Remote Ollama API
- **FallbackClient** - Chain multiple LLM clients with fallback logic

```python
from feedsummary_core.llm_client import create_llm_client

config = {
    "llm": {
        "provider": "ollama_local",
        "model": "mistral:latest",
        "temperature": 0.3,
    }
}

llm = create_llm_client(config)
response = await llm.chat(messages)
```

### Summarization (`summarizer/`)

**main.py** - Primary orchestration
- `run_pipeline()` - Full end-to-end workflow
- `run_resume_job()` - Resume interrupted jobs
- `compose_summary_docs()` - Generate composite summaries

**summarizer.py** - Core summarization engine
- `summarize_batches_then_meta_with_stats()` - Batch processing
- `super_meta_from_topic_sections_with_stats()` - Topic aggregation
- `_proofread_and_revise_meta_with_stats()` - Quality improvement

**tagging.py** - TagManager class
- Intelligent tag extraction and prioritization
- Domain entity detection (CVE, threat actors, regions, vulnerabilities)
- Similarity matching with configurable thresholds
- LLM-based tag generation with structured prompts

**tagging_integration.py** - Pipeline integration
- `tag_articles()` - Main tagging function
- `tag_articles_safe()` - Non-blocking wrapper for pipeline
- `generate_summary_from_tags()` - Create summaries from filtered articles
- Automatic pipeline integration with error handling

**ingest.py** - RSS feed handling
- `gather_articles_to_store()` - Fetch and process feeds
- `fetch_article_html()` - Full-content extraction
- Retry logic with exponential backoff

## Configuration

Create a `config.yaml` file:

```yaml
feeds:
  - url: "https://example.com/feed.xml"
    lookback: "48h"
    enabled: true

store:
  type: "sqlite"      # or "tinydb"
  path: "news.db"

llm:
  provider: "ollama_local"  # or "ollama_cloud", "fallback"
  model: "mistral:latest"
  temperature: 0.3
  base_url: "http://localhost:11434"

summarizer:
  max_tokens_per_article: 2000
  batch_size: 5
  max_retries: 3
  final_summary_lang: "svenska"

tagging:
  enabled: true
  max_tags_per_article: 5
  use_similarity_matching: true
  similarity_threshold: 0.6
```

## Usage Examples

### Summarization Pipeline

```python
import asyncio
from feedsummary_core.summarizer.main import run_pipeline

# Run complete pipeline: fetch RSS → summarize → tag → save
async def main():
    await run_pipeline("config.yaml")

asyncio.run(main())
```

### Article Tagging

```python
import asyncio
from feedsummary_core.persistence import create_store
from feedsummary_core.llm_client import create_llm_client
from feedsummary_core.summarizer.tagging_integration import tag_articles

async def main():
    store = create_store(config.get("store", {}))
    llm = create_llm_client(config)
    
    # Tag existing articles
    result = await tag_articles(
        store=store,
        llm_client=llm,
        article_ids=["article_1", "article_2"],
        config=config
    )
    
    print(f"Tagged articles: {sum(1 for tags in result.values() if tags)}")

asyncio.run(main())
```

### Tag-Based Summary Generation

```python
import asyncio
from feedsummary_core.persistence import create_store
from feedsummary_core.llm_client import create_llm_client
from feedsummary_core.summarizer.tagging_integration import generate_summary_from_tags

async def main():
    store = create_store(config.get("store", {}))
    llm = create_llm_client(config)
    
    # Generate summary for all articles with "cybersecurity" tag
    summary = await generate_summary_from_tags(
        store=store,
        llm_client=llm,
        tag_names=["cybersecurity"],
        config=config,
        match_mode="any"  # "any" (OR) or "all" (AND)
    )
    
    if summary:
        print(f"Generated summary: {summary['title']}")
        print(f"Articles: {summary['article_count']}")
        store.save_summary_doc(summary)

asyncio.run(main())
```

### Job Management

```python
import asyncio
from feedsummary_core.summarizer.main import run_pipeline, run_resume_job

async def main():
    # Start new job
    job_id = await run_pipeline("config.yaml")
    
    # Resume interrupted job
    await run_resume_job(job_id, "config.yaml")

asyncio.run(main())
```

## Architecture

### Data Flow

```
RSS Feeds
   ↓
[Ingest] → Fetch articles, extract content
   ↓
[Store] → Save to persistence layer
   ↓
[Summarize] → Batch processing, LLM summarization
   ↓
[Tag] → Extract & assign tags (automatic in pipeline)
   ↓
[Refine] → Proofread and improve summaries
   ↓
[Persist] → Save summaries with metadata
```

### Module Dependencies

```
persistence/          - Abstract store interface
  ├── SqliteStore
  └── TinyDbStore

llm_client/          - LLM abstraction
  ├── OllamaLocal
  ├── OllamaCloud
  └── FallbackClient

summarizer/
  ├── main.py       - Pipeline orchestration
  ├── summarizer.py - Summarization engine
  ├── tagging.py    - Tag management
  ├── tagging_integration.py - Pipeline integration
  ├── ingest.py     - Feed fetching
  ├── batching.py   - Chunk management
  └── helpers.py    - Utilities
```

## Documentation

For detailed information, see:

- [TAGGING_SYSTEM.md](TAGGING_SYSTEM.md) - Complete tagging system reference
- [TAGGING_INTEGRATION_GUIDE.md](TAGGING_INTEGRATION_GUIDE.md) - Automatic tagging in pipeline
- [TAGGING_QUICKSTART.md](TAGGING_QUICKSTART.md) - Quick tagging examples
- [TAG_BASED_SUMMARY.md](TAG_BASED_SUMMARY.md) - Tag-based summary generation guide

## Requirements

- Python >= 3.10
- SQLite >= 3.x (for SqliteStore)
- Local Ollama or cloud Ollama access
- RSS feed URLs

## Dependencies

Key dependencies (see `pyproject.toml` for full list):

- `aiohttp` - Async HTTP client
- `feedparser` - RSS feed parsing
- `tenacity` - Retry logic
- `tinydb` - JSON storage
- `PyYAML` - Configuration files
- `trafilatura` - Content extraction
- `ollama` - Ollama client

## License

BSD 3-Clause License - See LICENSE file for details
