# Artikel-taggningssystem

## Översikt

Taggningssystemet analyserar och taggar artiklar **automatiskt** under sammanfattningsprocessen. Det använder:

1. **Databaskvotering**: Sparade taggar lagras i SQLite eller TinyDB
2. **LLM-baserad taggextrahering**: Intelligenta taggar från artikelinnehål
3. **Prioriteringssystem**: Föredrar befintliga taggar framför nya
4. **Domän-specifika undantag**: Tillåter nya taggar för CVE:er, hot-aktörer, regioner, etc.

## ⚡ Automatisk Taggning

**Taggning körs automatiskt som en del av sammanfattningspipelinen.**

### How It Works

1. `run_pipeline()` eller `run_resume_job()` initierar summarizeringen
2. Artiklar sammanfattas normalt
3. **EFTER** summarizeringen är klar, körs taggning automatiskt på alla artiklar
4. Taggar lagras i databasen
5. Job-status uppdateras med tagging-information

### Ingen Åtgärd Krävs

Du behöver inte göra något speciellt - taggning körs automatiskt!

```python
from feedsummary_core.summarizer.main import run_pipeline

# Taggning körs automatiskt
summary_id = await run_pipeline(
    config_path="config.yaml",
    job_id=123
)
# Job status: "Klart: summerade 50 artiklar. Taggade: 45."
```

## Arkitektur

### Moduler

| Modul | Syfte |
|-------|--------|
| `tagging.py` | Huvudsaklig TagManager-klass med prioriteringslogik |
| `tagging_integration.py` | Integrationshjälpare: `tag_articles()`, `tag_articles_safe()`, etc. |
| `SqliteStore.py` (utökad) | Databaskvotering för taggar (SQLite-backend) |
| `TinyDbStore.py` (utökad) | Databaskvotering för taggar (TinyDB/JSON-backend) |
| `main.py` (integrerad) | Automatisk taggning under pipeline |

### Stöd för Alla Persistansmoduler

Taggning fungerar med **båda** available persistence-backends:

**SqliteStore** (SQL-based):
- Använder SQLite-tabeller för prestanda
- Stöder full SQL-optimering
- Rekommenderas för produktion

**TinyDbStore** (JSON-based):
- Använder TinyDB för enkelhets skull
- Sparas i JSON-fil
- Bra för utveckling och små deployment

Båda implementerar samma interface definierat i `NewsStore` protokoll.

### Databaskvotering

```sql
CREATE TABLE tags (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT UNIQUE NOT NULL,
    category   TEXT DEFAULT 'GENERAL',
    description TEXT,
    created_at INTEGER,
    updated_at INTEGER
);

CREATE TABLE article_tags (
    article_id TEXT NOT NULL,
    tag_id     INTEGER NOT NULL,
    created_at INTEGER,
    PRIMARY KEY (article_id, tag_id)
);
```

## Taggkategorier

### GENERAL (Standard)
- Normala ämnestaggar
- Prioriteras från befintliga taggar
- Får inte skapas nya automatiskt (om de redan finns)

### DOMAIN_ENTITY (Domänspecifik)
Undantag där nya taggar ALLTID tillåts:
- **CVE:er**: `CVE-2024-1234` mönster
- **Hot-aktörer**: APT-grupper, kampanjnamn (t.ex. "APT28", "Lazarus Group")
- **Geografiska regioner**: Länder, kontinenter (t.ex. "Russia", "North Korea")
- **Sårbarheter**: Specifika sårbarhetsnamn och typer

## Taggprioriteringsalgoritm

```
1. Extrahera kandidattaggar från artikel (via LLM)
2. För varje kandidattagg:
   a. Sök efter befintliga liknande taggar
   b. Om befintlig tagg finns → använd den
   c. Om domänspecifik entitet → skapa ny DOMAIN_ENTITY tagg
   d. Annars → hoppa över (föredra allmän framför specifik)
3. Lagra valda taggar i databasen
```

### Likhetsmätning

Systemet använder flera strategier för att hitta likartade befintliga taggar:

1. **Exakt överensstämmelse**: `tag == candidate`
2. **Substring-matchning**: `tag contains candidate` eller vice versa
3. **Teckenuppsättningsöverlapn**: Beräknad likhet (0.0-1.0)

Matchningar sorteras efter:
- Likhetsscore (högre först)
- Kategori (GENERAL före DOMAIN_ENTITY)

## Användning

### Grundläggande taggning av en artikel

```python
from feedsummary_core.summarizer.tagging import TagManager
from feedsummary_core.persistence import create_store
from feedsummary_core.llm_client import create_llm_client

store = create_store("sqlite://news.db")
tag_manager = TagManager(store)
llm_client = create_llm_client(config)

# Tagga en artikel
article = store.get_article("article_123")
tags = await tag_manager.generate_tags_for_article(
    llm_client=llm_client,
    article=article,
    config=config,
    max_tags=5
)

# Spara taggar
tag_ids = [t["id"] for t in tags]
store.add_article_tags("article_123", tag_ids)
```

### Taggning av flera artiklar

```python
from feedsummary_core.summarizer.tagging_integration import tag_articles

results = await tag_articles(
    store=store,
    llm_client=llm_client,
    article_ids=["art_1", "art_2", "art_3"],
    config=config,
    max_tags_per_article=5,
    skip_if_already_tagged=True
)

# results: {
#     "art_1": [{"id": 1, "name": "cybersecurity", "category": "GENERAL"}, ...],
#     "art_2": [{"id": 42, "name": "CVE-2024-1234", "category": "DOMAIN_ENTITY"}, ...],
#     ...
# }
```

### Hantering av fördefinierade taggar

```python
from feedsummary_core.summarizer.tagging_integration import add_predefined_tags

# Lägg till vanliga taggar
common_tags = [
    {"name": "cybersecurity", "category": "GENERAL"},
    {"name": "vulnerability", "category": "GENERAL"},
    {"name": "malware", "category": "GENERAL"},
    {"name": "threat-intelligence", "category": "GENERAL"},
]

count = add_predefined_tags(store, common_tags)
print(f"Added {count} tags")
```

### Hämta taggar för en artikel

```python
# Alla taggar för en artikel
tags = store.get_article_tags("article_123")
# Returnerar: [{"id": 1, "name": "security", "category": "GENERAL"}, ...]

# Formaterade för visning
from feedsummary_core.summarizer.tagging_integration import get_article_tags_for_display

display_tags = get_article_tags_for_display(store, "article_123")
# Returnerar: [{"name": "security", "category": "GENERAL"}, ...]
```

### Rensning av oanvända taggar

```python
from feedsummary_core.summarizer.tagging_integration import cleanup_old_tags

removed = cleanup_old_tags(store, days=30)
print(f"Removed {removed} unused tags")
```

## LLM-prompt för taggextrahering

Systemet använder följande prompt-mall för att få LLM att extrahera taggar:

```
Analyze the following article and extract up to {max_tags} relevant tags.

IMPORTANT RULES:
1. Prefer existing, general tags over creating new, specific tags
2. CVE numbers (e.g., CVE-2024-1234), threat actors, regions, and 
   vulnerability names are EXCEPTIONS - these should be new tags
3. Return tags in lowercase
4. Keep tags concise (1-3 words max)
5. Focus on the main topics and entities mentioned

Article:
{article_text}

Respond in JSON format:
{
    "tags": ["tag1", "tag2", "tag3"],
    "reasoning": "Brief explanation of why these tags were chosen"
}
```

Systemet extraherar sedan JSON från svar och tillämpar prioriteringslogiken.

## Domänentitetsdetektering

Systemet identifierar domänspecifika entiteter genom:

1. **Regex-mönster**: CVE format (`CVE-\d{4}-\d{4,5}`)
2. **Nyckelordsmatchning**:
   - Hot-aktörer: "APT", "group", "threat actor", "collective"
   - Regioner: Länder, kontinenter, geografiska termer
   - Sårbarheter: "vulnerability", "exploit", "zero-day", "breach"

## Integration med summarizer

För att integrera taggning i sammanfattningsflödet:

```python
# I main.py eller summarizer.py
from feedsummary_core.summarizer.tagging_integration import tag_articles

# Efter att artiklar har sammanfattats
articles = store.list_unsummarized_articles()
article_ids = [a["id"] for a in articles]

# Tagga artiklar
await tag_articles(
    store=store,
    llm_client=llm_client,
    article_ids=article_ids,
    config=config,
    max_tags_per_article=5
)
```

## Integration med summarizer

Taggning är **automatiskt integrerad** i summarization pipeline.

### Automatisk Körning

I `main.py`:

**run_pipeline()** - Efter summarizeringen:
```python
# Artiklar sammanfattas
summary_doc_id = await _summarize_and_persist_like_refresh(...)

# Taggning körs automatiskt
article_ids = [a.get("id") for a in to_sum if a.get("id")]
if article_ids:
    tagged_count = await tag_articles_safe(...)
```

**run_resume_job()** - Efter resumeering:
```python
# Resumerade artiklar sammanfattas
summary_doc_id = await _summarize_and_persist_like_refresh(...)

# Taggning körs automatiskt
article_ids = [a.get("id") for a in ordered_articles if a.get("id")]
if article_ids:
    tagged_count = await tag_articles_safe(...)
```

### Säker Körning

Taggning använder `tag_articles_safe()` vilket:
- Ignorerar fel utan att avbryta pipeline
- Loggar framsteg och statistik
- Returnerar antal framgångsrikt taggade artiklar
- Inkluderas i job-status-meddelande

### För att integrera med summarizer manuellt (avancerat)

Om du bygger egen pipeline:

```python
from feedsummary_core.summarizer.tagging_integration import tag_articles_safe

# Efter summarizeringen
articles = [...]
article_ids = [a["id"] for a in articles]

await tag_articles_safe(
    store=store,
    llm_client=llm_client,
    article_ids=article_ids,
    config=config,
    job_id=job_id,
    max_tags_per_article=5
)
```

## Gammal Dokumentation: Manual Tagging

För manuell taggning utan automatisk integration, se gamla anvndningsexempel nedan (deprecated, men fortfarande funktionell):

### Databaskvotering
- **Indexering**: Taggar indexeras på `name` och `category` för snabba sökningar
- **Foreign keys**: `article_tags` refererar till `tags` med ON DELETE CASCADE
- **Unikhet**: Taggnamn är unika (case-insensitive)

### LLM-anrop
- Varje artikel skapar ett LLM-anrop för taggextrahering
- Textlängd begränsas till 2000 tecken för att hålla prompten hanterbar
- Resultat cachelagras i databasen

### Minnesanvändning
- TagManager behåller inte alla taggar i minnet - hämtar från DB vid behov
- Likhetssökning görs på alla befintliga taggar (kan optimeras med indexering)

## Anpassning

### Ändra likhetströskeln

```python
tag_manager = TagManager(store)
similar_tags = tag_manager._find_similar_existing_tags(
    "my_tag",
    similarity_threshold=0.7  # Högre = striktare matchning
)
```

### Ändra domänentitetsdetektering

Redigera regex-mönster och nyckelord i `tagging.py`:

```python
CVE_PATTERN = re.compile(r'CVE-\d{4}-\d{4,5}', re.IGNORECASE)
THREAT_ACTOR_KEYWORDS = {'APT', 'group', ...}  # Lägg till fler ord
```

### Anpassad LLM-prompt

Redigera `_build_tagging_prompt()` i `TagManager` eller skapa en underclass.

## Framtida förbättringar

- [ ] Fulle-textökning efter artiklar för en given tagg
- [ ] Batch-taggning med flera artiklar i en LLM-anrop
- [ ] Tagg-relationer (synonymer, hypernymer/hyponymer)
- [ ] Konfigurerbara domänentitets-mönster
- [ ] Tagganvändningsstatistik och analytics
- [ ] Autocomplete för befintliga taggar i UI
