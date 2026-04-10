# EasyRAG

MVP agentic RAG for local DOCX search.

## Features

- DOCX upload and local workspace cache
- exact + hybrid retrieval
- Ollama-first routing, OpenRouter fallback
- Ollama models from `/api/tags`, active model from `/api/ps`
- Streamlit chat UI

## Setup

1. Run `bootstrap_env.bat`
2. Create `.env`
3. Start app with `start_app.bat`

## `.env`

For Ollama, set your local Ollama endpoint in `OLLAMA_BASE_URL`.

For OpenRouter fallback, fill `OPENROUTER_API_KEY`.

```env
OLLAMA_BASE_URL=http://10.32.2.36:11434
OLLAMA_DEFAULT_MODEL=gemma4:26b
OLLAMA_DEFAULT_EMBED_MODEL=qwen3-embedding:8b
OLLAMA_CONTROL_TIMEOUT_SECONDS=2
OLLAMA_INFERENCE_TIMEOUT_SECONDS=300

OPENROUTER_API_KEY=
OPENROUTER_MODEL=google/gemma-4-26b-a4b-it
OPENROUTER_EMBED_MODEL=qwen/qwen3-embedding-8b
```

## Commands

```powershell
tools\python-portable\python.exe scripts\provider_status.py
powershell -ExecutionPolicy Bypass -File scripts\run_streamlit.ps1
```
