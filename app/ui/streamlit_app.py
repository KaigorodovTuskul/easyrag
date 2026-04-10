from __future__ import annotations

from app.agents.answer import build_answer_context
from app.agents.embedding_indexer import build_embedding_index
from app.core.config import AppConfig
from app.providers.base import ProviderError
from app.providers.router import ProviderRouter
from app.retrieval.exact import search_exact
from app.retrieval.hybrid import search_hybrid
from app.retrieval.records import build_search_records
from app.storage.files import save_parsed_payload, save_uploaded_docx
from app.storage.embeddings import load_embeddings, replace_embeddings
from app.storage.index import get_index_summary, load_index_records, replace_document_records

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
    records = build_search_records(parsed)
    replace_document_records(parsed.source_name, records)

    st.write(
        {
            "source_name": parsed.source_name,
            "paragraph_count": parsed.paragraph_count,
            "table_count": parsed.table_count,
            "indexed_records": len(records),
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
    query = st.text_input("Search parsed document", placeholder="Try: 8580, N1.1, 47405")

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


def _render_persistent_search(provider, selection) -> None:
    st.subheader("Saved Index Search")
    summary = get_index_summary()
    st.write(summary)

    records = load_index_records()
    embedding_records = load_embeddings()

    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"Exact records: {len(records)}")
    with col2:
        st.caption(f"Embedding records: {len(embedding_records)}")

    if st.button("Build embedding index", disabled=not records):
        try:
            with st.spinner("Building embeddings..."):
                embeddings = build_embedding_index(provider, records, model=selection.embed_model)
                replace_embeddings(embeddings)
        except ProviderError as exc:
            st.error(f"Embedding provider error: {exc}")
        except Exception as exc:
            st.error(f"Embedding indexing failed: {exc}")
        else:
            st.success(f"Embedding index built: {len(embeddings)} records")
            embedding_records = embeddings

    mode = st.radio("Search mode", ["exact", "hybrid"], horizontal=True)
    query = st.text_input("Search saved index", placeholder="Try: 8580, N1.1, 47405", key="saved_index_query")

    if not query:
        return

    if mode == "hybrid":
        query_vector = []
        if embedding_records:
            try:
                query_vector = provider.embed(query, model=selection.embed_model).vector
            except ProviderError as exc:
                st.error(f"Embedding provider error: {exc}")
            except Exception as exc:
                st.error(f"Query embedding failed: {exc}")
        results, trace = search_hybrid(records, embedding_records, query=query, query_vector=query_vector, limit=30)
        st.caption(
            f"Indexed records: {len(records)}. Results: {len(results)}. "
            f"Trace: exact={trace.exact_count}, vector={trace.vector_count}, fused={trace.fused_count}"
        )
    else:
        results = search_exact(records, query=query, limit=30)
        st.caption(f"Indexed records: {len(records)}. Results: {len(results)}")

    if not results:
        st.warning("No exact/keyword matches found in saved index.")
        return

    rows = [
        {
            "score": round(result.score, 2),
            "source": result.record.source_name,
            "id": result.record.record_id,
            "type": result.record.record_type,
            "section": " / ".join(result.record.section_path),
            "matched": ", ".join(result.matched_terms[:8]),
            "snippet": result.snippet,
        }
        for result in results
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("Answer generation", expanded=True):
        answer_context = build_answer_context(query, results)
        st.caption(f"Using model: {selection.chat_model}")
        st.json(answer_context.citations)

        if st.button("Generate answer", key="generate_saved_index_answer"):
            try:
                generated = provider.generate(answer_context.prompt, model=selection.chat_model)
            except ProviderError as exc:
                st.error(f"Provider error: {exc}")
            except Exception as exc:
                st.error(f"Answer generation failed: {exc}")
            else:
                st.markdown(generated.text or "Empty answer returned by provider.")


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
        - `Now`: DOCX ingestion, table extraction, saved exact-search index
        """
    )

    _render_persistent_search(provider, selection)
    _render_docx_ingestion()


if __name__ == "__main__":
    main()
