# EasyRAG

MVP agentic RAG:

- `Ollama` first, `OpenRouter` fallback
- models from `/api/tags`
- active model from `/api/ps`
- `Streamlit` UI

## Start

1. Put your key into `.env`
2. Run `bootstrap_env.bat`
3. Check routing: `tools\\python-portable\\python.exe scripts\\provider_status.py`
4. Start UI: `powershell -ExecutionPolicy Bypass -File scripts\\run_streamlit.ps1`
