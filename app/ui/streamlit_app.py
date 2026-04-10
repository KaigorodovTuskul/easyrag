from __future__ import annotations

from dataclasses import asdict

from app.agents.answer import build_answer_context
from app.agents.controller import run_agent_retrieval
from app.agents.embedding_indexer import build_embedding_index
from app.core.config import AppConfig
from app.eval.runner import run_eval
from app.ingestion.batch import ingest_input_folder
from app.providers.base import ProviderError
from app.providers.router import ProviderRouter
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
    provider, selection = ProviderRouter(config).resolve()

    st.title("EasyRAG")
    st.caption("Простой поиск и ответы по DOCX-документам")

    _render_index_step(provider, selection)
    st.divider()
    _render_question_step(provider, selection)
    st.divider()
    _render_diagnostics(provider.name, selection)


def _render_index_step(provider, selection) -> None:
    st.subheader("1. Подготовьте документы")
    summary = get_index_summary()
    records_count = int(summary.get("record_count", 0) or 0)
    docs_count = int(summary.get("source_count", 0) or 0)
    embeddings_count = len(load_embeddings())

    col_status, col_actions = st.columns([1, 2])
    with col_status:
        st.metric("Документов", docs_count)
        st.metric("Фрагментов", records_count)
        st.metric("Embeddings", embeddings_count)

    with col_actions:
        uploaded_file = st.file_uploader("Загрузить DOCX", type=["docx"], label_visibility="collapsed")
        if uploaded_file is not None:
            _index_uploaded_docx(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Индексировать папку input", type="primary", use_container_width=True):
                _batch_ingest_input()

        with col2:
            if st.button("Построить embeddings", use_container_width=True, disabled=not records_count):
                _build_embeddings(provider, selection)

        st.caption("Минимум: загрузите DOCX или нажмите «Индексировать папку input». Embeddings нужны только для hybrid/semantic поиска.")


def _render_question_step(provider, selection) -> None:
    st.subheader("2. Задайте вопрос")

    records = load_index_records()
    if not records:
        st.info("Сначала проиндексируйте документы в шаге 1.")
        return

    query = st.text_input(
        "Вопрос",
        placeholder="Например: Что такое код 8580? Или просто: Н1.1",
        label_visibility="collapsed",
    )

    if not query:
        return

    agent_result = run_agent_retrieval(
        provider=provider,
        records=records,
        embedding_records=load_embeddings(),
        query=query,
        embed_model=selection.embed_model,
        limit=12,
    )

    if not agent_result.results:
        st.warning("Ничего не найдено. Попробуйте точную формулировку, код или номер счета.")
        _render_trace(agent_result)
        return

    st.subheader("3. Проверьте найденный контекст")
    _render_best_match(agent_result.results[0])

    with st.expander("Показать еще найденные фрагменты"):
        _render_results_table(agent_result.results)

    st.subheader("4. Получите ответ")
    st.write(
        {
            "уверенность": agent_result.evidence.confidence,
            "основание": agent_result.evidence.reason,
        }
    )

    answer_context = build_answer_context(query, agent_result.results, evidence=agent_result.evidence)

    if st.button("Сформировать ответ", type="primary"):
        try:
            with st.spinner("Формирую ответ..."):
                generated = provider.generate(answer_context.prompt, model=selection.chat_model)
        except ProviderError as exc:
            st.error(f"Ошибка модели: {exc}")
        except Exception as exc:
            st.error(f"Не удалось сформировать ответ: {exc}")
        else:
            st.markdown(generated.text or "Модель вернула пустой ответ.")

    with st.expander("Источники и trace"):
        st.json(answer_context.citations)
        _render_trace(agent_result)


def _render_best_match(result) -> None:
    record = result.record
    st.markdown(f"**Лучшее совпадение:** `{record.record_id}` · `{record.record_type}` · score `{round(result.score, 2)}`")
    if record.section_path:
        st.caption(" / ".join(record.section_path))
    st.info(result.snippet)


def _render_results_table(results) -> None:
    rows = [
        {
            "score": round(result.score, 2),
            "id": result.record.record_id,
            "тип": result.record.record_type,
            "документ": result.record.source_name,
            "раздел": " / ".join(result.record.section_path),
            "фрагмент": result.snippet,
        }
        for result in results
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_trace(agent_result) -> None:
    st.json([{"шаг": step.name, "детали": step.details} for step in agent_result.steps])


def _index_uploaded_docx(uploaded_file) -> None:
    try:
        from app.ingestion.docx_parser import parse_docx_bytes
    except ImportError:
        st.error("Не установлены зависимости для DOCX. Запустите bootstrap_env.bat.")
        return

    content = uploaded_file.getvalue()
    parsed = parse_docx_bytes(uploaded_file.name, content)
    save_uploaded_docx(uploaded_file.name, content)
    save_parsed_payload(uploaded_file.name, parsed.to_dict())
    records = build_search_records(parsed)
    replace_document_records(parsed.source_name, records)
    st.success(f"Документ проиндексирован: {parsed.source_name}. Фрагментов: {len(records)}")


def _batch_ingest_input() -> None:
    try:
        results = ingest_input_folder()
    except Exception as exc:
        st.error(f"Не удалось проиндексировать input: {exc}")
        return

    if not results:
        st.warning("В папке input нет DOCX-файлов.")
        return

    st.success(f"Проиндексировано документов: {len(results)}")
    with st.expander("Подробности индексации"):
        st.dataframe([asdict(result) for result in results], use_container_width=True, hide_index=True)


def _build_embeddings(provider, selection) -> None:
    records = load_index_records()
    try:
        with st.spinner("Строю embeddings..."):
            embeddings = build_embedding_index(provider, records, model=selection.embed_model)
            replace_embeddings(embeddings)
    except Exception as exc:
        st.error(f"Не удалось построить embeddings: {exc}")
        return

    st.success(f"Embeddings построены: {len(embeddings)}")


def _render_diagnostics(provider_name: str, selection) -> None:
    with st.expander("Диагностика"):
        st.write("Индекс")
        st.json(get_index_summary())

        st.write("Модели")
        st.json(
            {
                "provider": provider_name,
                "chat_model": selection.chat_model,
                "embed_model": selection.embed_model,
                "active_model": selection.active_model,
                "reason": selection.reason,
            }
        )

        if st.button("Запустить проверку качества"):
            try:
                results = run_eval()
            except Exception as exc:
                st.error(f"Проверка не запустилась: {exc}")
                return

            hit_at_1 = sum(1 for result in results if result.hit_at_1)
            hit_at_3 = sum(1 for result in results if result.hit_at_3)
            st.write({"кейсов": len(results), "hit@1": hit_at_1, "hit@3": hit_at_3})
            st.dataframe([asdict(result) for result in results], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
