from __future__ import annotations

from app.agents.answer import build_answer_context
from app.agents.controller import run_agent_retrieval
from app.agents.embedding_indexer import build_embedding_index
from app.core.config import AppConfig
from app.eval.runner import run_eval
from app.ingestion.batch import ingest_input_folder
from app.providers.base import ProviderError
from app.providers.router import ProviderRouter
from app.retrieval.evidence import validate_evidence
from app.retrieval.exact import search_exact
from app.retrieval.hybrid import search_hybrid
from app.retrieval.records import build_search_records
from app.storage.embeddings import load_embeddings, replace_embeddings
from app.storage.files import save_parsed_payload, save_uploaded_docx
from app.storage.index import get_index_summary, load_index_records, replace_document_records

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Streamlit is not installed. Install dependencies before running the UI.") from exc


def main() -> None:
    st.set_page_config(page_title="EasyRAG", layout="wide")

    config = AppConfig.load()
    router = ProviderRouter(config)
    provider, selection = router.resolve()

    st.title("EasyRAG")
    st.caption("Поиск и ответы по DOCX-документам с учетом таблиц")

    _render_sidebar(config, provider.name, selection)
    _render_status_cards(selection)
    _render_quick_start()

    tab_search, tab_docs, tab_quality, tab_debug = st.tabs(
        ["Поиск и ответ", "Документы", "Проверка качества", "Технические детали"]
    )

    with tab_search:
        _render_agent_search(provider, selection)
        _render_manual_search(provider, selection)

    with tab_docs:
        _render_index_actions(provider, selection)
        _render_docx_upload()

    with tab_quality:
        _render_eval()

    with tab_debug:
        _render_debug(provider.name, selection)


def _render_sidebar(config: AppConfig, provider_name: str, selection) -> None:
    st.sidebar.header("Настройки")
    st.sidebar.write(
        {
            "провайдер": provider_name,
            "модель ответа": selection.chat_model,
            "модель embeddings": selection.embed_model,
            "Ollama": config.ollama_base_url,
        }
    )
    st.sidebar.caption("Секреты из .env здесь не показываются.")


def _render_status_cards(selection) -> None:
    summary = get_index_summary()
    embeddings = load_embeddings()

    col1, col2, col3 = st.columns(3)
    col1.metric("Документов в индексе", summary.get("source_count", 0))
    col2.metric("Фрагментов для exact-поиска", summary.get("record_count", 0))
    col3.metric("Фрагментов с embeddings", len(embeddings))

    if not summary.get("record_count"):
        st.warning("Индекс пустой. Перейдите во вкладку «Документы» и загрузите DOCX или нажмите «Индексировать input/*.docx».")
    elif not embeddings:
        st.info("Exact-поиск уже работает. Для semantic/hybrid поиска можно построить embedding-индекс во вкладке «Документы».")

    with st.expander("Текущие модели"):
        st.write(
            {
                "chat_model": selection.chat_model,
                "embed_model": selection.embed_model,
                "active_model": selection.active_model,
                "reason": selection.reason,
            }
        )


def _render_quick_start() -> None:
    with st.expander("Что делать сначала?", expanded=False):
        st.markdown(
            """
            1. Откройте вкладку **Документы**.
            2. Нажмите **Индексировать input/*.docx** или загрузите один DOCX вручную.
            3. Вернитесь во вкладку **Поиск и ответ**.
            4. Введите запрос, например `8580`, `Н1.1` или `Что такое код 8580?`.
            5. Нажмите **Сформировать ответ**, если нужен ответ модели по найденному контексту.
            """
        )


def _render_agent_search(provider, selection) -> None:
    st.subheader("1. Задайте вопрос")
    records = load_index_records()
    embedding_records = load_embeddings()
    query = st.text_input("Вопрос или точное значение", placeholder="Например: Что такое код 8580?", key="agent_query")

    if not query:
        st.caption("Агент сам выберет exact или hybrid retrieval и покажет найденный контекст.")
        return

    agent_result = run_agent_retrieval(
        provider=provider,
        records=records,
        embedding_records=embedding_records,
        query=query,
        embed_model=selection.embed_model,
        limit=30,
    )

    st.write(
        {
            "тип запроса": agent_result.query_type,
            "режим поиска": agent_result.mode,
            "найдено фрагментов": len(agent_result.results),
            "уверенность": agent_result.evidence.confidence,
            "evidence": agent_result.evidence.reason,
        }
    )

    if not agent_result.results:
        st.warning("Подходящий контекст не найден.")
        return

    _render_results_table(agent_result.results)
    _render_answer_box(provider, selection, query, agent_result.results, agent_result.evidence, key_prefix="agent")

    with st.expander("Trace агента"):
        st.json([{"name": step.name, "details": step.details} for step in agent_result.steps])


def _render_manual_search(provider, selection) -> None:
    with st.expander("Ручной поиск exact/hybrid"):
        records = load_index_records()
        embedding_records = load_embeddings()
        mode = st.radio("Режим", ["exact", "hybrid"], horizontal=True, key="manual_mode")
        query = st.text_input("Ручной запрос", placeholder="Например: 8580, N1.1, 47405", key="manual_query")

        if not query:
            return

        if mode == "hybrid":
            query_vector = []
            if embedding_records:
                try:
                    query_vector = provider.embed(query, model=selection.embed_model).vector
                except Exception as exc:
                    st.error(f"Не удалось получить embedding запроса: {exc}")
            results, trace = search_hybrid(records, embedding_records, query=query, query_vector=query_vector, limit=30)
            st.caption(f"Trace: exact={trace.exact_count}, vector={trace.vector_count}, fused={trace.fused_count}")
        else:
            results = search_exact(records, query=query, limit=30)

        if not results:
            st.warning("Ничего не найдено.")
            return

        evidence = validate_evidence(results)
        _render_results_table(results)
        _render_answer_box(provider, selection, query, results, evidence, key_prefix="manual")


def _render_answer_box(provider, selection, query: str, results, evidence, key_prefix: str) -> None:
    st.subheader("2. Ответ по найденному контексту")
    st.write({"можно отвечать": evidence.ok, "уверенность": evidence.confidence, "причина": evidence.reason})
    answer_context = build_answer_context(query, results, evidence=evidence)

    with st.expander("Используемые источники"):
        st.json(answer_context.citations)

    if st.button("Сформировать ответ", key=f"{key_prefix}_generate_answer"):
        if not evidence.ok:
            st.warning("Evidence слабый. Ответ может быть пропущен или должен быть сформулирован как «не найдено».")
        try:
            with st.spinner("Модель формирует ответ..."):
                generated = provider.generate(answer_context.prompt, model=selection.chat_model)
        except ProviderError as exc:
            st.error(f"Ошибка провайдера: {exc}")
        except Exception as exc:
            st.error(f"Не удалось сформировать ответ: {exc}")
        else:
            st.markdown(generated.text or "Модель вернула пустой ответ.")


def _render_results_table(results) -> None:
    st.subheader("Найденный контекст")
    rows = [
        {
            "score": round(result.score, 2),
            "документ": result.record.source_name,
            "id": result.record.record_id,
            "тип": result.record.record_type,
            "раздел": " / ".join(result.record.section_path),
            "совпадения": ", ".join(result.matched_terms[:8]),
            "фрагмент": result.snippet,
        }
        for result in results
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_index_actions(provider, selection) -> None:
    st.subheader("1. Индексация документов")
    st.caption("Файлы из папки input/ не пушатся в Git и используются только локально.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Индексировать input/*.docx", type="primary"):
            try:
                results = ingest_input_folder()
            except Exception as exc:
                st.error(f"Индексация не удалась: {exc}")
            else:
                st.success(f"Проиндексировано документов: {len(results)}")
                st.dataframe([result.__dict__ for result in results], use_container_width=True, hide_index=True)

    with col2:
        records = load_index_records()
        if st.button("Построить embedding-индекс", disabled=not records):
            try:
                with st.spinner("Строю embeddings. Это может занять время..."):
                    embeddings = build_embedding_index(provider, records, model=selection.embed_model)
                    replace_embeddings(embeddings)
            except Exception as exc:
                st.error(f"Embedding-индекс не построен: {exc}")
            else:
                st.success(f"Embedding-индекс построен: {len(embeddings)} фрагментов.")


def _render_docx_upload() -> None:
    st.subheader("2. Загрузить один DOCX вручную")
    uploaded_file = st.file_uploader("Выберите DOCX-файл", type=["docx"])

    if uploaded_file is None:
        return

    try:
        from app.ingestion.docx_parser import parse_docx_bytes
    except ImportError:
        st.error("Не установлены зависимости для DOCX. Запустите bootstrap_env.bat.")
        return

    content = uploaded_file.getvalue()
    parsed = parse_docx_bytes(uploaded_file.name, content)
    doc_path = save_uploaded_docx(uploaded_file.name, content)
    json_path = save_parsed_payload(uploaded_file.name, parsed.to_dict())
    records = build_search_records(parsed)
    replace_document_records(parsed.source_name, records)

    st.success("Документ загружен и проиндексирован.")
    st.write(
        {
            "документ": parsed.source_name,
            "абзацев": parsed.paragraph_count,
            "таблиц": parsed.table_count,
            "фрагментов": len(records),
            "сохранен DOCX": str(doc_path),
            "сохранен JSON": str(json_path),
        }
    )

    with st.expander("Посмотреть извлеченные таблицы и абзацы"):
        _render_doc_overview(parsed)
        _render_tables(parsed)


def _render_doc_overview(parsed) -> None:
    st.markdown("**Разделы**")
    sections = sorted({" / ".join(paragraph.section_path) for paragraph in parsed.paragraphs if paragraph.section_path})
    st.write(sections or ["Разделы не найдены"])


def _render_tables(parsed) -> None:
    if not parsed.tables:
        st.info("Таблицы не найдены.")
        return

    for table in parsed.tables:
        st.markdown(f"**{table.element_id}**")
        st.caption(f"Раздел: {' / '.join(table.section_path) if table.section_path else 'Корень документа'}")
        st.dataframe([row.values for row in table.rows], use_container_width=True, hide_index=True)


def _render_eval() -> None:
    st.subheader("Проверка качества поиска")
    st.caption("Проверяет, попадает ли ожидаемый record id в top-1/top-3.")

    if st.button("Запустить проверку качества"):
        try:
            results = run_eval()
        except Exception as exc:
            st.error(f"Проверка не запустилась: {exc}")
            return

        hit_at_1 = sum(1 for result in results if result.hit_at_1)
        hit_at_3 = sum(1 for result in results if result.hit_at_3)
        st.write({"кейсов": len(results), "hit@1": hit_at_1, "hit@3": hit_at_3})
        st.dataframe([result.__dict__ for result in results], use_container_width=True, hide_index=True)


def _render_debug(provider_name: str, selection) -> None:
    st.subheader("Технические детали")
    st.write(get_index_summary())
    st.write(
        {
            "provider": provider_name,
            "chat_model": selection.chat_model,
            "embed_model": selection.embed_model,
            "active_model": selection.active_model,
            "reason": selection.reason,
        }
    )
    st.caption("Секреты из .env не отображаются.")


if __name__ == "__main__":
    main()
