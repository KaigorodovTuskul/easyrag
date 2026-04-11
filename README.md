# EasyRAG

Local-first RAG for DOCX documents with exact search, BM25, hybrid retrieval, multimodal formula extraction, and a Streamlit chat UI.

## Overview

EasyRAG builds a local workspace for one uploaded DOCX document, caches parsed fragments and embeddings, and then answers questions in a chat interface.

Key points:

- DOCX parsing with paragraph and table extraction
- Exact search + BM25 + hybrid retrieval
- Ollama and OpenRouter support
- Separate answer, embedding, and vision model selection
- Formula extraction from embedded DOCX images through a multimodal model
- Inline display of formula images in relevant answers
- Deterministic answer builders for norms, terms, codes, and composition questions

RU: локальный RAG для DOCX-документов с кэшем workspace, exact/BM25/hybrid retrieval, multimodal-извлечением формул и чат-интерфейсом.

## Demo

![EasyRAG demo](demo.gif)

## Requirements

- Windows
- Python 3.11+ recommended
- Microsoft Office / PowerPoint installed if the DOCX contains `WMF` or `EMF` formulas

Why Office matters:

- many Word formulas in regulatory documents are embedded as `WMF` / `EMF`
- EasyRAG renders those vector objects through PowerPoint first, then sends the resulting image to the selected vision-capable model
- without Office, some vector formulas may remain unavailable for recognition and preview

## Setup: Portable Python

1. Run `bootstrap_env.bat`
2. Copy `.env.example` to `.env`
3. Fill provider settings you actually use
4. Start with `start_app.bat`

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

## Configuration

Start from `.env.example`:

```env
APP_ENV=dev
APP_HOST=0.0.0.0
APP_PORT=8501
APP_LANGUAGE=ru
LLM_PROVIDER=openrouter

OLLAMA_BASE_URL=http://10.32.2.36:11434
OLLAMA_DEFAULT_MODEL=gemma4:26b
OLLAMA_DEFAULT_EMBED_MODEL=qwen3-embedding:8b
OLLAMA_DEFAULT_VISION_MODEL=
OLLAMA_TAGS_PATH=/api/tags
OLLAMA_PS_PATH=/api/ps
OLLAMA_CONTROL_TIMEOUT_SECONDS=2
OLLAMA_INFERENCE_TIMEOUT_SECONDS=300

OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=google/gemma-4-26b-a4b-it
OPENROUTER_EMBED_MODEL=qwen/qwen3-embedding-8b
OPENROUTER_VISION_MODEL=

EMBEDDING_BATCH_SIZE=16
EMBEDDING_RECORD_TYPES=paragraph,table_row
VECTOR_BACKEND=local
VECTOR_COLLECTION=easyrag_chunks
RERANKER_ENABLED=false
TRACE_AGENT_STEPS=true
```

Notes:

- `APP_LANGUAGE` supports `ru` and `en`
- `OLLAMA_DEFAULT_VISION_MODEL` and `OPENROUTER_VISION_MODEL` are optional
- if the vision model is left empty, the UI defaults to `None`
- when `None` is selected, EasyRAG tries to use the selected answer model as the vision model if that model supports image input
- `EMBEDDING_RECORD_TYPES=paragraph,table_row` is the default because exact/BM25 already covers many table questions well

## Formula Handling

EasyRAG no longer uses local OCR or `pix2tex`.

Current pipeline:

1. DOCX parser extracts formula images and stores them in workspace assets
2. If a vision-capable model is available during indexing or later enrichment, EasyRAG sends the image to that model
3. Recognized formula text is written back into the search index
4. If recognition is unavailable, the image asset is still preserved and can be shown in the chat answer

Important limitations:

- if a formula exists only as an embedded Word image/object, text-only retrieval can still answer from surrounding prose without reconstructing the exact fraction
- exact visual reconstruction of `WMF` / `EMF` formulas depends on Office/PowerPoint availability on the machine

## Current UI Behavior

- answer model, embedding model, and vision model are selected separately in the sidebar
- there are no extra "custom model" text fields in the current UI
- formula images are shown only for formula/norm answers where they are locally relevant
- deterministic term/composition answers intentionally do not auto-attach unrelated formula images

## Commands

```powershell
tools\python-portable\python.exe scripts\provider_status.py
powershell -ExecutionPolicy Bypass -File scripts\run_streamlit.ps1
```
