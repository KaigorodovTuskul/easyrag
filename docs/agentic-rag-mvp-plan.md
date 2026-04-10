# Agentic RAG MVP Plan

## Goal

Build an MVP for local-first agentic RAG focused on DOCX ingestion, including tables, with Ollama as the primary provider and OpenRouter as the temporary debug fallback.

## Core constraints

- Primary inference endpoint: `http://10.32.2.36:11434`
- Available models must be discovered dynamically from `GET /api/tags`
- The active loaded model must be preferred when present via `GET /api/ps`
- If no active model is reported, default to `gemma4:26b`
- Default embedding model: `qwen3-embedding:8b`
- Temporary fallback provider: OpenRouter
- MVP UI: Streamlit
- Narrow future optimization target: high-precision retrieval for DOCX and tables

## Recommended MVP architecture

### 1. Provider abstraction

Implement a provider layer with a single internal contract:

- `list_models()`
- `get_active_model()`
- `generate()`
- `embed()`
- `healthcheck()`

Providers:

- `OllamaProvider`
- `OpenRouterProvider`

Selection logic:

1. Try Ollama healthcheck
2. If reachable, pull `tags` and `ps`
3. If `ps` reports a loaded model and that model exists in `tags`, use it
4. Else use `gemma4:26b` if present
5. Else choose the closest available chat-capable model from `tags`
6. If Ollama is unreachable, fall back to OpenRouter

### 2. Agentic orchestration

Do not start with a fully autonomous agent loop. Use a bounded agentic pipeline:

1. Query classification
2. Query rewrite set generation
3. Tool selection
4. Multi-retrieval
5. Evidence validation
6. Answer synthesis with citations

Tools:

- `search_chunks`
- `search_tables`
- `search_exact`
- `load_context_by_id`
- `answer_from_context`

Guardrails:

- max 2 retrieval iterations
- max 1 rewrite round
- force citation-backed answer
- refuse unsupported claims when evidence is weak

### 3. Ingestion pipeline

For DOCX, extract structured content first and only then build searchable chunks.

Document elements:

- headings
- paragraphs
- lists
- tables
- table rows
- table cells

Recommended element schema:

- `doc_id`
- `source_path`
- `element_id`
- `element_type`
- `section_path`
- `page_hint`
- `table_id`
- `row_id`
- `col_id`
- `text`
- `normalized_text`
- `metadata_json`

Storage strategy:

- paragraph chunks
- table-level chunks
- row-level chunks
- optional cell-level records for exact lookup

### 4. Retrieval stack

For your target use case, even with agentic control, retrieval must stay hybrid and exact-first.

Order:

1. Exact lexical retrieval
2. Table-aware row retrieval
3. Dense retrieval
4. Lightweight rerank
5. Context assembly

Recommended scoring:

- phrase / exact match boost
- metadata filter boost
- row-header overlap boost
- semantic similarity as secondary signal

### 5. Storage

MVP:

- SQLite for metadata
- local FAISS for embeddings

Next step:

- PostgreSQL + `pgvector` + full text search

Reason:

- fast MVP bootstrap now
- clean migration path later
- exact-match retrieval is better served long-term in Postgres than in vector-only stores

### 6. Streamlit MVP

Tabs:

- Chat
- Documents
- Retrieval Trace
- Settings

The Retrieval Trace view is mandatory for debugging agent behavior. Show:

- chosen provider
- chosen chat model
- chosen embedding model
- tool calls
- rewritten queries
- retrieved chunks
- scores

## Delivery phases

### Phase 1. Bootstrap

- project layout
- config loader
- provider abstraction
- Ollama/OpenRouter clients
- healthcheck and model-selection logic

### Phase 2. Ingestion

- DOCX parser
- table extraction
- normalization
- chunking
- indexing into SQLite + FAISS

### Phase 3. Retrieval

- exact search
- vector search
- table-aware retrieval
- score fusion

### Phase 4. Agent layer

- query classifier
- bounded planner
- tool loop
- evidence validator
- answer composer

### Phase 5. UI

- Streamlit document upload
- indexing controls
- chat interface
- retrieval trace

### Phase 6. Precision tuning for your domain

- custom exact-match heuristics
- domain synonym controls
- row/header linking improvements
- answer extraction templates

## Suggested repository layout

```text
easyrag/
  app/
    agents/
    core/
    ingestion/
    providers/
    retrieval/
    storage/
    ui/
  data/
    docs/
    index/
  docs/
  scripts/
  tools/
  tests/
```

## Technical decisions for MVP

- Python: portable local runtime for bootstrap
- UI: Streamlit
- LLM routing: provider abstraction with Ollama-first policy
- Embeddings: default `qwen3-embedding:8b`
- Chat model: active Ollama model if loaded, else `gemma4:26b`
- Fallback: OpenRouter
- DOCX extraction: start with `python-docx`, add a richer parser only if needed
- Agent pattern: bounded and inspectable, not autonomous open-ended loops

## What is intentionally not in MVP

- multi-agent orchestration
- graph RAG
- knowledge graph extraction
- autonomous background crawlers
- aggressive query rewriting loops

These features increase complexity faster than they improve precision for your current problem.
