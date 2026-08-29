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
  - MongoDB (shared, production-grade document storage)
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

**MongoDBStore** - MongoDB document backend
- Indexed collections for articles, summaries, jobs, tags, and categories
- Atomic integer IDs for compatibility with the existing store interface
- Complete feature parity with TinyDB and SQLite

Install the MongoDB driver and configure the backend:

```bash
pip install "feedsummary-core[mongodb]"
```

```python
store = create_store({
    "provider": "mongodb",
    "uri": "mongodb://localhost:27017",
    "database": "feedsummary",
})
```

Migrate an existing TinyDB database by validating it first and then running the
idempotent import:

```bash
feedsummary-migrate-tinydb-mongodb news_docs.json --dry-run

export FEEDSUMMARY_MONGODB_URI="mongodb://localhost:27017"
export FEEDSUMMARY_MONGODB_DATABASE="feedsummary"
feedsummary-migrate-tinydb-mongodb news_docs.json
```

The migration preserves article, job, tag, and category IDs along with tag
relations and embedding caches. Unique URL or name collisions stop the import by
default; use `--conflict-policy keep-existing` to map relations to the existing
MongoDB records. The command can also be run as
`python -m feedsummary_core.persistence.migrate_tinydb_to_mongodb`.

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
  - name: "Example feed"
    url: "https://example.com/feed.xml"
    # Optional per-feed minimum TLS version, applied to the feed and its articles.
    # Supported values: "1.2" and "1.3".
    tls_min_version: "1.3"
    # Use native curl for servers that reject aiohttp's TLS fingerprint.
    http_client: "curl"

store:
  provider: "sqlite"  # sqlite, tinydb, or mongodb
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
  similarity_consistency:
    enabled: true
    similarity_threshold: 0.78
    max_shared_tags: 1
```

### Embedding-based ML tagging

The production ML path uses a classifier selected in configuration with
persisted article embeddings. Both the classifier and the category scope are
configuration-driven; `categories` accepts one or more database category names.
On each tagging run, a corpus fingerprint detects changed article-tag
associations, classifier settings, or category scope and atomically retrains the
artifact when needed. Full retraining correctly handles tags that were removed
or corrected.

```yaml
tagging:
  # Existing tags in these categories may be selected by the LLM, but the LLM
  # may not create new tags in them.
  llm_new_tag_excluded_categories: [DOMAIN_ENTITY]
  ml:
    enabled: false
    classifier: logistic_regression  # logistic_regression or sgd
    representation: embedding
    categories: [DOMAIN_ENTITY]
    embedding_model: ""  # inherit from the local embedding provider
    embedding_text_chars: 2000
    model_path: data/tagging_ml/embedding_classifier.joblib
    min_label_support: 3
    min_training_articles: 30
    max_tags_per_article: 5
    threshold: 0.8014432738078289
    regularization_c: 1.0  # logistic_regression only
    alpha: 0.0001          # sgd only
    max_iter: 2000
    tolerance: 0.0001
    random_state: 42
    auto_retrain: true
```

To change or extend the scope, edit only the category list, for example
`categories: [DOMAIN_ENTITY, LOCATION]`. The artifact is rejected and retrained
when its configured classifier or category list no longer matches.

When a compatible embedding or model is unavailable, tagging safely falls back
to the existing LLM path. `llm_new_tag_excluded_categories` still applies to
that fallback, so it may reuse an existing `DOMAIN_ENTITY` tag but cannot add a
new one. Joblib artifacts must only be loaded from trusted local paths.

ML tagging emits searchable JSON payloads after stable event names:

- `ml_tagging.model_ready` records the classifier, categories, threshold, model
  version, label count, and training corpus metadata.
- `ml_tagging.predictions` records the article ID and every accepted ML tag with
  category and probability. An empty `suggestions` list means that the model ran
  but no score met the configured threshold.
- `ml_tagging.skipped` records incompatible or missing article embeddings, while
  `ml_tagging.model_unavailable` records initialization and training failures.
- `ml_tagging.below_threshold` records the five strongest rejected candidates
  at DEBUG level.

No article text is included in these events. Accepted ML assignments also keep
the classifier and rounded probability in the persisted `motivering` field.

### Benchmark classical ML tagging

The first ML tagging stage is an offline, read-only benchmark against historical
MongoDB tag assignments. It does not change the production LLM tagging flow.
Install the optional dependencies and add an ML section to the normal config:

```bash
pip install "feedsummary-core[ml]"
feedsummary-benchmark-tags --config config.yaml --output-dir artifacts/tag-benchmark
```

```yaml
store:
  provider: mongodb
  uri: mongodb://localhost:27017
  database: feedsummary

tagging:
  ml:
    categories: [GENERAL]
    min_label_support: 10
    max_tags_per_article: 5
    max_text_chars: 20000
    random_seed: 42
    n_jobs: 1
    max_category_combinations: 63
```

Each run writes a JSON report, a compact Markdown report, and the winning
scikit-learn pipeline. Historical tags are treated as labels and may contain
LLM-generated noise. Only load the generated Joblib model from a trusted source.
When multiple categories are configured, every non-empty category combination
is evaluated. Ranking uses chronological validation micro-F1 adjusted for tag
assignment coverage: tags excluded by `min_label_support` count as unpredicted.
The report shows raw F1 alongside label and assignment coverage, so a category
with one easy, well-supported tag cannot win merely because its many rare tags
were filtered out. The limit above prevents an accidental exponential benchmark
when too many categories are supplied.

After all articles in a run have been tagged, similarity consistency checks
embedding-based article groups for tag overlap. If a group has no tag shared by
all members, the most common existing general tag is added to the missing
articles. Existing tags are never removed or replaced.

### Similarity-based batching

When the LLM chain contains an `ollama_local` provider, articles are embedded with
that provider's configured `embedding_model`. Articles above the similarity
threshold are kept in the same summary batch whenever the hard article and
character limits allow it.

```yaml
batching:
  similarity_enabled: true
  similarity_threshold: 0.78
  embedding_text_chars: 2000
  embedding_max_concurrency: 4

llm:
  provider: ollama_local
  model: qwen2.5:7b
  embedding_model: embeddinggemma:latest
```

Set `similarity_enabled: false` to retain sequential batching. If embeddings
cannot be generated, the summarizer automatically falls back to sequential
batching.

Article and tag embeddings are persisted by every storage backend. Cache entries
contain `embedding_vector`, `embedding_model`, `embedding_source_hash`, and
`embedding_updated_at`; they are reused until the embedded text or model changes.

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
