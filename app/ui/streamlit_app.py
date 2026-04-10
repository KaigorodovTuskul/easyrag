from __future__ import annotations

from app.core.config import AppConfig
from app.providers.router import ProviderRouter
from app.retrieval.exact import search_exact
from app.retrieval.records import build_search_records
from app.storage.files import save_parsed_payload, save_uploaded_docx

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


def _render_docx_ingestion() -> None:
    st.subheader("DOCX Ingestion")
    uploaded_file = st.file_uploader("Upload DOCX", type=["docx"])

    if uploaded_file is None:
        st.info("Upload a .docx file to inspect extracted paragraphs and tables.")
        return

    try:
        from app.ingestion.docx_parser import ParsedDocx, parse_docx_bytes
    except ImportError:
        st.error("DOCX parser dependencies are not installed. Install requirements and restart the app.")
        return

    content = uploaded_file.getvalue()
    parsed = parse_docx_bytes(uploaded_file.name, content)

    doc_path = save_uploaded_docx(uploaded_file.name, content)
    json_path = save_parsed_payload(uploaded_file.name, parsed.to_dict())

    st.write(
        {
            "source_name": parsed.source_name,
            "paragraph_count": parsed.paragraph_count,
            "table_count": parsed.table_count,
            "saved_docx": str(doc_path),
            "saved_json": str(json_path),
        }
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Paragraphs", "Tables", "Exact Search"])

    with tab1:
        _render_doc_overview(parsed)

    with tab2:
        _render_paragraphs(parsed)

    with tab3:
        _render_tables(parsed)

    with tab4:
        _render_exact_search(parsed)


def _render_doc_overview(parsed) -> None:
    st.markdown("**Sections**")
    section_paths = []
    for paragraph in parsed.paragraphs:
        if paragraph.style.startswith("Heading"):
            section_paths.append(" / ".join(paragraph.section_path))

    st.write(section_paths or ["No headings found"])


def _render_paragraphs(parsed) -> None:
    items = [
        {
            "id": paragraph.element_id,
            "style": paragraph.style,
            "section_path": " / ".join(paragraph.section_path),
            "text": paragraph.text,
        }
        for paragraph in parsed.paragraphs
    ]
    st.dataframe(items, use_container_width=True, hide_index=True)


def _render_tables(parsed) -> None:
    if not parsed.tables:
        st.info("No tables found in the document.")
        return

    for table in parsed.tables:
        st.markdown(f"**{table.element_id}**")
        st.caption(f"Section: {' / '.join(table.section_path) if table.section_path else 'Root'}")
        rows = [row.values for row in table.rows]
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_exact_search(parsed) -> None:
    records = build_search_records(parsed)
    query = st.text_input("Search parsed document", placeholder="Try: 8580, Н1.1, ипотечным ссудам")

    if not query:
        st.caption(f"Searchable records: {len(records)}")
        return

    results = search_exact(records, query=query, limit=20)
    st.caption(f"Searchable records: {len(records)}. Results: {len(results)}")

    if not results:
        st.warning("No exact/keyword matches found.")
        return

    rows = [
        {
            "score": round(result.score, 2),
            "id": result.record.record_id,
            "type": result.record.record_type,
            "section": " / ".join(result.record.section_path),
            "matched": ", ".join(result.matched_terms[:8]),
            "snippet": result.snippet,
        }
        for result in results
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


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
        - `Now`: DOCX ingestion, table extraction, JSON debug artifacts
        """
    )

    _render_docx_ingestion()


if __name__ == "__main__":
    main()
