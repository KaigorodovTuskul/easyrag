from __future__ import annotations

from app.core.config import AppConfig
from app.providers.router import ProviderRouter

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Streamlit is not installed. Install dependencies before running the UI.") from exc


def _render_sidebar(config: AppConfig) -> None:
    st.sidebar.header("Config")
    st.sidebar.code(
        "\n".join(
            [
                f"APP_ENV={config.app_env}",
                f"OLLAMA_BASE_URL={config.ollama_base_url}",
                f"OPENROUTER_BASE_URL={config.openrouter_base_url}",
                f"TRACE_AGENT_STEPS={config.trace_agent_steps}",
            ]
        ),
        language="bash",
    )


def main() -> None:
    config = AppConfig.load()
    router = ProviderRouter(config)
    provider, selection = router.resolve()

    st.set_page_config(page_title="EasyRAG MVP", layout="wide")
    st.title("EasyRAG Agentic MVP")
    st.caption("Ollama-first provider routing with OpenRouter fallback")

    _render_sidebar(config)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Provider")
        st.write(
            {
                "provider": provider.name,
                "reachable": selection.reachable,
                "reason": selection.reason,
                "chat_model": selection.chat_model,
                "embed_model": selection.embed_model,
                "active_model": selection.active_model,
            }
        )

    with col2:
        st.subheader("Available Models")
        st.write([model.name for model in selection.available_models] or ["No models discovered"])

    st.subheader("MVP Scope")
    st.markdown(
        """
        - `Provider routing`: detect live Ollama, prefer active loaded model from `/api/ps`
        - `Fallback`: use OpenRouter when Ollama is unavailable
        - `Next`: DOCX ingestion, table-aware retrieval, bounded agent loop
        """
    )


if __name__ == "__main__":
    main()
