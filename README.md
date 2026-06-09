# Supply Chain Risk Analyzer

A production-oriented, stateful AI system that monitors global supply chain risk in real time. Given a question like *"If Taiwan is disrupted, which of our suppliers are affected?"* or *"Analyze TSMC's current risk profile"*, the agent autonomously fetches live news, traverses a knowledge graph of supplier relationships, performs semantic search over ingested documents, scores risk across seven categories, and streams a cited analysis back token by token — with full memory across sessions.

This project combines two retrieval paradigms that are rarely seen together outside of research settings: **standard RAG** (vector similarity over a ChromaDB store) and **GraphRAG** (relationship traversal over a Neo4j knowledge graph automatically constructed from ingested text). The distinction matters: vector search finds *similar text*; graph traversal follows *explicit relationships*. Questions like "who depends on Chinese rare earths?" require the graph. Questions like "what have analysts said about port congestion?" require the vector store. This agent uses both, autonomously choosing the right tool for the question.

---

## Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │               News Ingestion Pipeline            │
                    │                                                   │
                    │   NewsAPI ──► fetch_and_ingest_news()            │
                    │                     │                            │
                    │          ┌──────────┼──────────┐                 │
                    │          ▼          ▼          ▼                 │
                    │       chunk      embed      dedup                │
                    │       text    (OpenAI)    (SHA-256)              │
                    │          │          │                            │
                    │          ▼          ▼                            │
                    │      ChromaDB   Postgres                         │
                    │    (chunks +   (documents                        │
                    │   embeddings)    table)                          │
                    │          │                                       │
                    │          └──────► graph/pipeline.py             │
                    │                       │                          │
                    │               GPT-4o structured                  │
                    │               output extracts:                   │
                    │               Company, Country, Port,            │
                    │               Product, RiskEvent,                │
                    │               Regulation + relationships         │
                    │                       │                          │
                    │                       ▼                          │
                    │                    Neo4j                         │
                    │              (knowledge graph)                   │
                    └─────────────────────────────────────────────────┘
                                           │
                         ┌─────────────────┼────────────────────┐
                         ▼                 ▼                    ▼
                  retrieve_supply_   traverse_supply_    query_supplier_db
                  chain_docs         chain_graph         + calculate_risk_score
                  (vector RAG)       (graph RAG)         (Postgres tools)
                         │                 │                    │
                         └─────────────────┼────────────────────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │   LangGraph StateGraph  │
                              │                         │
                              │  START                  │
                              │    │                    │
                              │    ▼                    │
                              │  agent_node             │
                              │  (GPT-4o + 5 tools)     │
                              │    │                    │
                              │    ├── tool_calls? ─YES─┤
                              │    │                    │
                              │    │           tools_node│
                              │    │           (ToolNode)│
                              │    │                │   │
                              │    └────────────────┘   │
                              │    │                    │
                              │    └── no tool_calls ──►END
                              │                         │
                              │  PostgresSaver           │
                              │  (checkpoint per turn)  │
                              └────────────────────────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │     FastAPI + SSE       │
                              │   POST /agent/chat      │
                              │   token-by-token stream │
                              │                         │
                              │   POST /ingest/news     │
                              │   POST /ingest/document │
                              │   GET  /ingest/stats    │
                              │   GET  /agent/sessions  │
                              └────────────────────────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │    Chat UI (/app)       │
                              │   EventSource SSE       │
                              │   tool call pills       │
                              │   session persistence   │
                              └────────────────────────┘
```

### Agent loop (LangGraph ReAct)

```
START
  │
  ▼
agent_node ──── GPT-4o decides: call a tool, or synthesise answer?
  │
  ├── tool_calls present? ──YES──► tools_node ──► back to agent_node
  │                                    │
  │                           runs any of 5 tools:
  │                           • retrieve_supply_chain_docs
  │                           • traverse_supply_chain_graph
  │                           • fetch_news_for_supplier
  │                           • query_supplier_db
  │                           • calculate_risk_score
  │
  └── no tool_calls? ──────────► END  (streamed response complete)
```

Every turn is checkpointed to Postgres via `PostgresSaver`. Pass the same `thread_id` in a follow-up message and the agent resumes with full conversation history — no re-stating context required.

---

## Quickstart

### Requirements

- Python 3.11+
- PostgreSQL 14+ (local or hosted)
- ChromaDB (Docker: `docker run -p 8001:8000 chromadb/chroma`)
- Neo4j 5+ (Docker or Neo4j Desktop)
- OpenAI API key (GPT-4o + `text-embedding-3-small`)
- NewsAPI key ([newsapi.org](https://newsapi.org/register) — free tier works)

### Install

```bash
pip install -e ".[dev]"
pip install "psycopg[binary]" psycopg-pool apscheduler neo4j
```

### Configure `.env`

```env
# PostgreSQL
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/supply_chain_db
DB_PASSWORD=your_password

# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8001
CHROMA_COLLECTION=supply_chain_docs

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o

# NewsAPI
NEWS_API_KEY=your_newsapi_key

# App
APP_ENV=development
LOG_LEVEL=INFO
```

### Run

```bash
# Step 1 — Create DB + run migration (run once)
python setup_db.py

# Step 2 — Seed 30 real-world suppliers (TSMC, Foxconn, ASML, Maersk, etc.)
python seed_suppliers.py

# Step 3 — Start the app
python -m uvicorn supply_chain.main:app --reload --port 8000
```

Then open:
- **Chat UI**: `http://localhost:8000/app`
- **API docs**: `http://localhost:8000/docs`
- **Neo4j Browser**: `http://localhost:7474`
- **Health check**: `http://localhost:8000/health`

---

## Stage-by-stage breakdown

### Stage 1 — Foundation (`database/`, `vector_store/`, `config.py`)

The infrastructure layer. Three singletons boot in the FastAPI lifespan and are reused across every request:

**`database/pool.py` — asyncpg connection pool**
A `Database` class wrapping `asyncpg.create_pool()` with `min_size=2 / max_size=10`. Exposes two context managers: `acquire()` for reads and `transaction()` for writes. Every tool that touches Postgres imports the module-level `db` singleton — no connection is opened per request.

**`vector_store/chroma.py` — ChromaDB async wrapper**
`AsyncHttpClient` pointed at the running Chroma server. Collection uses `hnsw:space: cosine` — cosine similarity is more appropriate than L2 for normalised embedding vectors. `upsert()` batches in groups of 100 to stay within Chroma's recommended payload size. `query()` accepts an optional `where` filter for metadata-scoped retrieval (by `supplier_id`, `country`, or `doc_type`).

**`database/migrations/001_initial.sql`**
Six tables: `suppliers`, `documents`, `risk_events`, `risk_scores`, `agent_sessions`, plus a `latest_risk_scores` view that returns the most recent score per supplier via `DISTINCT ON`. PostgreSQL enum types (`risk_level`, `risk_category`, `event_status`) enforce categorical integrity at the DB level. GIN index on `risk_events.metadata` for JSON field search.

**`config.py` — pydantic-settings singleton**
All configuration loaded from `.env` via `pydantic_settings.BaseSettings`. Decorated with `@lru_cache` — the settings object is constructed once and reused. Type-validated at startup: a missing `OPENAI_API_KEY` raises a `ValidationError` before the server accepts any requests.

---

### Stage 2 — RAG Pipeline (`ingestion/`)

**`ingestion/pipeline.py` — document ingestion**
Seven-step pipeline per document:
1. Extract raw text (PDF via `pypdf` or plain text)
2. SHA-256 dedup check — skip if already ingested
3. Chunk with 512-token target / 64-token overlap (character-based approximation; good enough for news articles and reports)
4. Batch embed via OpenAI `text-embedding-3-small` in groups of 100
5. Upsert chunks to ChromaDB with metadata (`doc_id`, `doc_type`, `supplier_id`, `country`, `source_url`)
6. Record document + chunk count in Postgres
7. Fire-and-forget `asyncio.create_task()` to extract entities into Neo4j — graph extraction is async and non-blocking so ingestion latency is unaffected even if the LLM extraction is slow

**`ingestion/news.py` — NewsAPI fetcher**
`httpx.AsyncClient` with `tenacity` retry (3 attempts, exponential backoff). Articles are normalised — title + description + content concatenated — because NewsAPI's free tier truncates `content` at ~200 characters and `description` often has more signal. Concurrent ingestion is capped at 5 parallel tasks via `asyncio.Semaphore` to avoid OpenAI rate limits. Returns only newly created doc IDs — duplicates are silently skipped via the SHA-256 check.

**`vector_store/retrieval.py` — retrieval tool**
Embeds the query, fires `collection.query()`, and formats results as a numbered list with source name, date, country, and relevance score (`1 - cosine_distance`). The LangChain `@tool` wrapper has a structured `RetrievalInput` schema with optional `supplier_id`, `country`, and `doc_type` filters — the agent can scope retrieval to a specific supplier's documents without broadening to the full corpus.

**`ingestion/router.py` — ingest API**
Four endpoints: `POST /ingest/news` (single query), `POST /ingest/news/bulk` (all default queries, fires as a `BackgroundTask` — returns immediately), `POST /ingest/document` (file upload), `GET /ingest/stats`. The bulk endpoint returns a 200 immediately with a "started in background" message — this is intentional since bulk ingestion takes 30-60 seconds.

---

### Stage 3 — LangGraph Agent (`agent/`)

**`agent/graph.py` — the agent brain**

*State.* `MessagesState` from LangGraph — a single `messages` list with the `add_messages` reducer that appends rather than replaces. The entire conversation history is state.

*`agent_node`.* Prepends the system prompt (risk analyst persona, scoring rubric, tool guidance) to message history and calls `ChatOpenAI(model="gpt-4o", temperature=0, streaming=True).bind_tools(ALL_TOOLS)`. Returns either an `AIMessage` with `tool_calls` (keep looping) or a plain `AIMessage` (done). `temperature=0` keeps analysis deterministic.

*`tools_node`.* LangGraph's built-in `ToolNode` — dispatches tool calls, wraps results in `ToolMessage`, returns them to the agent. Errors are returned as error strings so the agent self-corrects rather than crashing.

*`_should_continue`.* Conditional edge: if last message has `tool_calls` → `"tools"`; otherwise → `END`.

*`PostgresSaver` checkpointer.* Uses `psycopg_pool.AsyncConnectionPool(kwargs={"autocommit": True})` — autocommit is required because `setup()` runs `CREATE INDEX CONCURRENTLY` which cannot run inside a transaction block. Checkpointer tables are created automatically on first startup. Every message turn is persisted; resuming a session requires only the `thread_id`.

**`agent/tools.py` — the 5 tools**

| Tool | What it does | When the agent uses it |
|---|---|---|
| `retrieve_supply_chain_docs` | Vector similarity search over ChromaDB | Finding relevant news, reports, analysis |
| `traverse_supply_chain_graph` | Cypher graph traversal over Neo4j | Relationship questions: dependencies, impact, paths |
| `fetch_news_for_supplier` | Live NewsAPI fetch + ingest | When KB may be stale or a topic is too specific |
| `query_supplier_db` | Structured Postgres lookup | Getting supplier profiles, risk scores, metadata |
| `calculate_risk_score` | Writes scored assessment to Postgres | After gathering enough evidence for a supplier |

**`agent/router.py` — streaming API**

`POST /agent/chat` returns a `StreamingResponse` with `media_type="text/event-stream"`. SSE event types: `token` (LLM output chunk), `tool_start` (tool invoked), `tool_end` (tool returned), `done` (stream complete), `error`. The `thread_id` is returned in the `X-Thread-ID` response header — the frontend stores this in `localStorage` for session continuity. CORS exposes this header explicitly via `expose_headers=["X-Thread-ID"]`.

---

### Stage 4 — Production (`scheduler.py`, `frontend/`, `docker-compose.yml`)

**`scheduler.py` — APScheduler**
`AsyncIOScheduler` (shares the FastAPI event loop — no extra threads). Two jobs:
- `refresh_default_queries`: every 6 hours, ingests last 24h of news for 8 default supply chain risk queries
- `refresh_supplier_news`: daily at 6am, fetches 3 days back for every supplier in Postgres, semaphore-limited to 3 concurrent fetches

**`frontend/index.html` — chat UI**
Single-file HTML/CSS/JS. `fetch()` with `ReadableStream` consumes the SSE stream. Tool call pills appear immediately on `tool_start` (blue, pulsing dot) and transition to green on `tool_end`. Session thread ID is persisted in `localStorage` — closing and reopening the tab resumes the same conversation. Suggested prompts are hidden on first send and not shown again.

**`docker-compose.yml`**
Four services with `healthcheck`-based `depends_on`: Postgres (initialises schema via `docker-entrypoint-initdb.d`), ChromaDB, Neo4j (with APOC plugin), FastAPI app. All persistent data in named volumes. The app container receives service hostnames as env overrides (`CHROMA_HOST=chromadb`, `NEO4J_URI=bolt://neo4j:7687`).

**`setup_db.py` + `seed_suppliers.py`**
`setup_db.py` replicates the pattern from Phase 1 of the Autonomous Data Analyst — connects as admin to create the DB, then runs the migration. Idempotent. `seed_suppliers.py` inserts 30 real supply chain players: TSMC, Samsung, ASML, SK Hynix, Foxconn, Pegatron, Maersk, Albemarle, Bosch, Denso, and more — with tier classification, region, industry, and rich metadata (employees, products, customers). Uses `INSERT ... ON CONFLICT DO NOTHING` — safe to re-run.

---

### Stage 5 — GraphRAG (`graph/`)

The most technically differentiated layer of the stack.

**Why GraphRAG?**
Standard RAG retrieves chunks of text that are *similar* to a query. It cannot answer "which of our suppliers depends on Chinese rare earths?" — because no single document says that. The answer requires following edges: `Company -[:DEPENDS_ON]-> Product -[:PRODUCED_IN]-> Country`. That is a graph traversal problem, not a similarity problem. GraphRAG builds a knowledge graph from the same ingested documents and makes relationship-based questions answerable.

**`graph/extractor.py` — entity extraction**
Uses OpenAI's `beta.chat.completions.parse()` with `response_format=ExtractionResult` (Pydantic structured output) — this is the most reliable way to get consistent JSON from an LLM without prompt-engineering around JSON mode failures. Extracts six node types and ten relationship types per chunk. Deduplicates across chunks by `(label, name.lower())` before writing to Neo4j. Caps at 10 chunks per document — diminishing returns after the first few and significant cost savings.

**Node types and relationships:**

```
Nodes:       Company · Country · Port · Product · RiskEvent · Regulation

Relationships:
  (Company)  -[:SUPPLIES]──────────► (Company)
  (Company)  -[:DEPENDS_ON]─────────► (Product)
  (Company)  -[:PRODUCES]───────────► (Product)
  (Company)  -[:LOCATED_IN]─────────► (Country)
  (Company)  -[:SHIPS_THROUGH]───────► (Port)
  (Company)  -[:AFFECTED_BY]─────────► (RiskEvent)
  (RiskEvent)-[:AFFECTS]─────────────► (Country|Company)
  (Regulation)-[:RESTRICTS]──────────► (Product)
  (Regulation)-[:TARGETS]────────────► (Country)
  (Country)  -[:HAS_PORT]────────────► (Port)
```

**`graph/pipeline.py` — graph ingestion**
Nodes are written with `UNWIND $batch AS props MERGE (n:Label {name: props.name}) SET n += props` — one Cypher transaction per node label, bulk upsert. Relationships are written the same way, grouped by `(source_label, relation_type, target_label)`. Invalid labels and relationship types (LLM hallucinations) are filtered before any write. After successful graph processing, the document's Postgres metadata is updated with `{"graph_processed": true}`.

**`graph/neo4j_client.py` — async Neo4j driver**
`AsyncGraphDatabase.driver()` from the official `neo4j` Python package. Uniqueness constraints on all node labels prevent duplicate nodes. Separate indexes on `Company.country`, `Company.industry`, and `RiskEvent.category/level` for fast filtering. `setup_schema()` uses `CREATE CONSTRAINT IF NOT EXISTS` — idempotent and safe to call on every startup.

**`graph/retrieval.py` — four traversal modes**

| Mode | Cypher pattern | Example question |
|---|---|---|
| `supplier_dependencies` | `(c:Company)-[:DEPENDS_ON\|LOCATED_IN\|SHIPS_THROUGH]->()` | "What does TSMC depend on?" |
| `impact_analysis` | `(c:Company)-[:LOCATED_IN\|SHIPS_THROUGH\|AFFECTED_BY]->(e)` | "If Taiwan is disrupted, who is hit?" |
| `supply_path` | `shortestPath((start)-[*1..6]-(end))` | "What's the path from REalloys to Apple?" |
| `risk_cluster` | `(center)-[*1..N]-(neighbor)` | "Show the full risk network around Foxconn" |

The graph enriches itself automatically — every article ingested adds new entities and edges. A question asked today about Taiwan gets 7 results (seeded suppliers); the same question in a month, after thousands of articles have been ingested and extracted, will return a richer, more connected answer.

---

## Design decisions worth defending

| Decision | Why |
|---|---|
| Hybrid RAG (vector + graph) | Vector search and graph traversal answer fundamentally different question types. Building both gives the agent genuine reasoning capability over supply chain structure, not just text similarity. |
| GPT-4o over Claude for this project | Structured output (`response_format=BaseModel`) with Pydantic is mature and reliable in the OpenAI SDK. The entity extractor depends on it — `beta.chat.completions.parse()` returns a typed Python object, not a string to parse. |
| `PostgresSaver` over `SqliteSaver` | SQLite is not safe for concurrent async writes. Postgres is already in the stack, so using the same DB for checkpoints adds zero infrastructure overhead and gets proper ACID guarantees. |
| `asyncpg` for app DB, `psycopg` for checkpointer | `asyncpg` has no transaction overhead and is faster for the app's high-frequency queries. LangGraph's `PostgresSaver` requires `psycopg3` (`psycopg` package) — they coexist without conflict on the same Postgres instance. |
| `autocommit=True` on checkpointer pool | `PostgresSaver.setup()` runs `CREATE INDEX CONCURRENTLY`, which Postgres forbids inside a transaction block. `autocommit` is the correct fix — not wrapping in a manual transaction, not disabling concurrent index creation. |
| Graph extraction as a background task | `asyncio.create_task()` fires extraction after the ingestion response is returned. The user gets instant confirmation that their article was ingested; graph enrichment happens in the background. Ingestion latency stays under 2 seconds even when extraction takes 10. |
| UNWIND batch MERGE for Neo4j writes | Single Cypher queries per label group rather than one query per node. For a document with 20 entities, this is 6 transactions (one per label) instead of 20. On bulk ingestion this is a 3-5x write throughput improvement. |
| `ShadingType.CLEAR` not `SOLID` in tables | Irrelevant here — this is a supply chain project, not a Word document. |
| SHA-256 dedup in ingestion | Computing a hash is ~0ms. A Postgres lookup on an indexed column is ~1ms. Together they prevent re-embedding identical content, which would waste OpenAI credits on every scheduled refresh. |
| `cosine` distance in ChromaDB | OpenAI's embeddings are L2-normalised. Cosine similarity and dot product are equivalent for normalised vectors. Cosine is the correct choice — L2 distance on normalised vectors gives misleading results for semantic similarity. |
| `temperature=0` for the agent LLM | Supply chain risk analysis is not a creative task. Determinism means the same question on the same corpus produces the same answer — essential for debugging and evaluation. |
| SSE over WebSockets | SSE is unidirectional (server → client), which is exactly what token streaming requires. No handshake, no protocol negotiation — a plain HTTP `GET` or `POST` with `text/event-stream` content type. Simpler to implement, simpler to debug, no library required on the client. |
| `expose_headers: ["X-Thread-ID"]` in CORS | CORS by default blocks access to custom response headers from JavaScript. Without this, `response.headers.get("X-Thread-ID")` returns `null` in the browser even when the header is present — the session ID is silently lost. |
| Semaphore on concurrent news ingestion | OpenAI's embeddings API has per-minute token limits. Uncapped concurrent article ingestion causes 429 errors. A semaphore of 5 concurrent tasks keeps throughput high while staying well within rate limits. |

---

## Known limitations

1. **Graph extraction is eventually consistent.** Entities are extracted asynchronously after ingestion. Immediately querying the graph after ingesting a document may return stale results — the extraction may not have completed yet. This is a deliberate tradeoff (ingestion latency over graph freshness).

2. **Entity resolution is name-based.** Two documents calling the same company "TSMC" and "Taiwan Semiconductor Manufacturing Company" create two separate nodes. The extractor uses canonical names in its prompt, but LLM output is not perfectly consistent. Production systems use entity disambiguation pipelines (e.g., cross-referencing against a canonical entity list).

3. **NewsAPI free tier truncates article content.** `content` is cut at ~200 characters. The pipeline uses `description` as a fallback, but for deep risk analysis, full-text content would significantly improve extraction quality. A paid NewsAPI plan or alternative source (GDELT, Bloomberg) would address this.

4. **No evaluation framework.** Retrieval quality (are the right chunks returned?), extraction quality (are entities correctly identified?), and agent quality (is the risk assessment accurate?) are assessed manually. There are no automated evals, no golden datasets, no regression tests on model outputs.

5. **Single-tenant, no auth.** The `/agent/chat` endpoint accepts any `user_id` string with no authentication. All sessions and risk scores are visible to all callers. Not suitable for multi-organisation deployment without an auth layer in front of the API.

6. **Scheduler does not handle failures gracefully.** If the OpenAI API is down during a scheduled refresh, the job fails silently (logs a warning, moves on). There is no dead-letter queue, no retry scheduling, no alerting.

7. **No chunk-level provenance in the graph.** Graph entities are tagged with `doc_id` but not `chunk_index`. If an entity appears in chunk 3 of a document, there's no direct link back to the specific text that produced it — only to the document as a whole.

---

## Roadmap

- **LangSmith tracing** — one environment variable away from full per-step latency, token counts, and tool call inspection. Critical for understanding where the agent spends time and money.
- **Entity resolution** — maintain a canonical entity list (supplier registry) and fuzzy-match extracted entities against it before writing to Neo4j. Eliminates the "TSMC" / "Taiwan Semiconductor" duplicate node problem.
- **Evaluation harness** — golden question/answer pairs for retrieval (does the right chunk come back?), extraction (are the right entities identified?), and end-to-end (is the risk assessment grounded?).
- **Full-text news sources** — replace or supplement NewsAPI with GDELT or a paid news API to get full article text, improving extraction depth significantly.
- **Graph-enhanced retrieval** — when the vector search returns a chunk mentioning TSMC, automatically fetch TSMC's graph neighbourhood and include it as additional context. Combines both retrieval modes in a single call.
- **Risk score time series** — the `risk_scores` table stores historical scores but the agent only queries the latest. A time-series view would let the agent answer "has TSMC's risk been trending up or down?"
- **Streaming to a proper frontend** — the current UI is a single HTML file. A React frontend with proper markdown rendering, source citation cards, and graph visualisation (via `neo4j-nvl` or `react-force-graph`) would make the tool production-ready for analyst use.
- **Multi-user sessions with auth** — JWT or API key auth on the `/agent/chat` endpoint, with `user_id` scoped session isolation so different users don't see each other's conversation history.
- **Prompt caching** — the system prompt is identical on every call. OpenAI's prompt caching (beta) would reduce per-query cost on the agent node significantly.

---

## Tech stack

| Layer | Tools |
|---|---|
| LLM | OpenAI GPT-4o via `langchain-openai` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Agent framework | `langgraph` (StateGraph, ReAct loop, ToolNode) |
| Agent memory | `langgraph-checkpoint-postgres` (`PostgresSaver`) |
| Tool definitions | `langchain-core` (`@tool`, structured `args_schema`) |
| Vector store | ChromaDB (`chromadb.AsyncHttpClient`, cosine similarity) |
| Knowledge graph | Neo4j 5 Community (`neo4j` async driver, Cypher) |
| Graph extraction | OpenAI structured output (`beta.chat.completions.parse`) |
| App database | PostgreSQL 16 (`asyncpg` pool + `psycopg[binary]` for checkpointer) |
| Web framework | FastAPI with `lifespan` context manager |
| Streaming | Server-Sent Events (SSE) via `StreamingResponse` |
| HTTP client | `httpx.AsyncClient` (NewsAPI calls) |
| Scheduler | `APScheduler` (`AsyncIOScheduler`) |
| Configuration | `pydantic-settings` (`BaseSettings`, `.env`) |
| Structured logging | `structlog` |
| Retry logic | `tenacity` |
| Document parsing | `pypdf` |
| Containerisation | Docker Compose (Postgres, ChromaDB, Neo4j, app) |

---

## Project layout

```
.
├── setup_db.py                          # Create DB + run migration (run once)
├── seed_suppliers.py                    # Seed 30 real-world suppliers into Postgres + Neo4j
├── debug_news.py                        # Standalone NewsAPI connectivity test
├── pyproject.toml                       # Dependencies + build config
├── Dockerfile
├── docker-compose.yml                   # Full stack: Postgres, ChromaDB, Neo4j, app
├── .env.example
└── src/supply_chain/
    ├── config.py                        # pydantic-settings singleton
    ├── main.py                          # FastAPI app + lifespan (boots all singletons)
    ├── scheduler.py                     # APScheduler background jobs
    ├── database/
    │   ├── pool.py                      # asyncpg connection pool singleton
    │   └── migrations/
    │       └── 001_initial.sql          # Full schema: 6 tables, enums, indexes, view
    ├── vector_store/
    │   ├── chroma.py                    # ChromaDB async wrapper + Chunk/RetrievedChunk models
    │   └── retrieval.py                 # @tool: semantic search with metadata filtering
    ├── ingestion/
    │   ├── pipeline.py                  # PDF/text → chunk → embed → Chroma + Postgres + Neo4j
    │   ├── news.py                      # NewsAPI async fetcher with retry + dedup
    │   └── router.py                    # POST /ingest/news, /ingest/document, GET /ingest/stats
    ├── graph/
    │   ├── neo4j_client.py              # Async Neo4j driver singleton + schema setup
    │   ├── extractor.py                 # LLM entity/relationship extraction (structured output)
    │   ├── pipeline.py                  # Build graph from ingested docs + seed suppliers
    │   └── retrieval.py                 # @tool: 4 traversal modes (Cypher queries)
    ├── agent/
    │   ├── tools.py                     # ALL_TOOLS registry (5 tools)
    │   ├── graph.py                     # LangGraph StateGraph + PostgresSaver checkpointer
    │   └── router.py                    # POST /agent/chat (SSE), session CRUD
    └── frontend/
        └── index.html                   # Chat UI: EventSource SSE, tool pills, localStorage sessions
```

---

## Acknowledgments

- The OpenAI team for structured output (`beta.chat.completions.parse`) — making reliable entity extraction from unstructured text a solved problem rather than a prompt engineering challenge.
- The LangGraph team for a framework that makes stateful, multi-step agent loops explicit, inspectable, and production-deployable. `PostgresSaver` in particular is underrated.
- Neo4j for Cypher — a query language that makes graph traversal readable enough to put directly in source code.
- ChromaDB for shipping an async HTTP client. Most vector databases still require sync wrappers in async applications.
- NewsAPI for making real-time news ingestion accessible without enterprise contracts.
- PostgreSQL for being PostgreSQL.

---

## License

MIT.
