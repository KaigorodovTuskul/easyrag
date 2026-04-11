# EasyRAG

Local-first agentic RAG for DOCX documents.

<p>
  <b>Exact search</b> + <b>BM25</b> + <b>hybrid retrieval</b> + <b>Ollama/OpenRouter routing</b> + <b>Streamlit chat UI</b>
</p>

## Demo

![EasyRAG demo](demo.gif)

## Overview

EasyRAG builds a local workspace for one uploaded DOCX document, caches parsed fragments and embeddings, then lets the user ask questions in a chat interface.

RU: локальный RAG для DOCX-документов с кэшем базы, exact/BM25/hybrid retrieval и чат-интерфейсом.

## Setup: Portable Python

1. Run `bootstrap_env.bat`
2. Copy `.env.example` to `.env`
3. Set `OLLAMA_BASE_URL` for Ollama
4. If you use OpenRouter fallback, fill `OPENROUTER_API_KEY`
5. Start with `start_app.bat`

## Setup: venv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts\run_streamlit.ps1
```

## Setup: System Python

```powershell
python -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts\run_streamlit.ps1
```

## `.env`

Start from `.env.example`:

```env
APP_LANGUAGE=ru

OLLAMA_BASE_URL=http://10.32.2.36:11434
OLLAMA_DEFAULT_MODEL=gemma4:26b
OLLAMA_DEFAULT_EMBED_MODEL=qwen3-embedding:8b
OLLAMA_DEFAULT_VISION_MODEL=
OLLAMA_CONTROL_TIMEOUT_SECONDS=2
OLLAMA_INFERENCE_TIMEOUT_SECONDS=300

OPENROUTER_API_KEY=
OPENROUTER_MODEL=google/gemma-4-26b-a4b-it
OPENROUTER_EMBED_MODEL=qwen/qwen3-embedding-8b
OPENROUTER_VISION_MODEL=

EMBEDDING_BATCH_SIZE=16
EMBEDDING_RECORD_TYPES=paragraph,table_row
```

`APP_LANGUAGE` supports `ru` and `en`. The language can also be switched in the UI.
`EMBEDDING_RECORD_TYPES` controls which record types get vector embeddings. The default skips table cells because exact/BM25 search already covers them and table rows preserve enough semantic context for most table questions. Use `paragraph,table_row,table_cell` if you need the previous broader embedding behavior.
`OLLAMA_DEFAULT_VISION_MODEL` and `OPENROUTER_VISION_MODEL` are optional defaults for DOCX formula extraction from embedded images. Leave them empty to default to `None` in the UI; when `None` is selected, EasyRAG will try the selected answer model as the vision model.

Formula images are no longer processed by local OCR. EasyRAG now sends embedded DOCX images to the selected multimodal model during indexing and stores the recognized formula text in the search index when extraction succeeds.

## Commands

```powershell
tools\python-portable\python.exe scripts\provider_status.py
powershell -ExecutionPolicy Bypass -File scripts\run_streamlit.ps1
```
