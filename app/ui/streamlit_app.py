from __future__ import annotations

from dataclasses import asdict

from app.agents.answer import build_answer_context
from app.agents.controller import run_agent_retrieval
from app.agents.embedding_indexer import build_embedding_index
from app.core.config import AppConfig
from app.eval.runner import run_eval
from app.providers.base import ProviderError
from app.providers.router import ProviderRouter
from app.retrieval.records import SearchRecord, build_search_records
from app.storage.files import save_parsed_payload, save_uploaded_docx
from app.storage.workspaces import (
    estimate_tokens,
    file_hash_for_content,
    list_workspaces,
    load_workspace_embeddings,
    load_workspace_info,
    load_workspace_records,
    save_workspace_embeddings,
    save_workspace_records,
    workspace_id_for,
)

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Streamlit is not installed. Install dependencies before running the UI.") from exc


def main() -> None:
    st.set_page_config(page_title="EasyRAG", layout="wide")

    config = AppConfig.load()
    provider, selection = ProviderRouter(config).resolve()

    st.title("EasyRAG")
    st.caption("Один документ - одна локальная база. Загрузили один раз, потом можно быстро открыть готовую базу.")

    workspace_id = _render_workspace_step(provider, selection)
    st.divider()
    _render_question_step(provider, selection, workspace_id)
    st.divider()
    _render_diagnostics(provider.name, selection, workspace_id)


def _render_workspace_step(provider, selection) -> str | None:
    st.subheader("1. Выберите базу или загрузите документ")

    workspaces = list_workspaces()
    selected_workspace_id = st.session_state.get("workspace_id")

    if workspaces:
        labels = [_workspace_label(item) for item in workspaces]
        ids = [item.workspace_id for item in workspaces]
        selected_index = ids.index(selected_workspace_id) if selected_workspace_id in ids else 0
        selected_label = st.selectbox("Готовые локальные базы", labels, index=selected_index)
        selected_workspace_id = ids[labels.index(selected_label)]
        st.session_state["workspace_id"] = selected_workspace_id
        _render_workspace_summary(selected_workspace_id)
    else:
        st.info("Готовых баз пока нет. Загрузите DOCX ниже.")

    uploaded_file = st.file_uploader("Загрузить DOCX и автоматически подготовить базу", type=["docx"])
    if uploaded_file is not None:
        selected_workspace_id = _prepare_uploaded_workspace(uploaded_file, provider, selection)
        st.session_state["workspace_id"] = selected_workspace_id

    return selected_workspace_id


def _render_question_step(provider, selection, workspace_id: str | None) -> None:
    st.subheader("2. Задайте вопрос")

    if not workspace_id:
        st.info("Сначала выберите готовую базу или загрузите DOCX.")
        return

    records = load_workspace_records(workspace_id)
    embeddings = load_workspace_embeddings(workspace_id)
    if not records:
        st.warning("В выбранной базе нет индекса. Загрузите документ заново.")
        return

    query = st.text_input("Вопрос", placeholder="Например: Что такое код 8580? Или просто: Н1.1", label_visibility="collapsed")
    if not query:
        return

    agent_result = run_agent_retrieval(
        provider=provider,
        records=records,
        embedding_records=embeddings,
        query=query,
        embed_model=selection.embed_model,
        limit=12,
    )

    if not agent_result.results:
        st.warning("Ничего не найдено. Попробуйте точную формулировку, код или номер счета.")
        with st.expander("Trace"):
            _render_trace(agent_result)
        return

    st.subheader("3. Найденный контекст")
    _render_best_match(agent_result.results[0])

    with st.expander("Показать остальные совпадения"):
        _render_results_table(agent_result.results)

    st.subheader("4. Ответ")
    st.caption(f"Уверенность: {agent_result.evidence.confidence}. Основание: {agent_result.evidence.reason}.")
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


def _prepare_uploaded_workspace(uploaded_file, provider, selection) -> str:
    try:
        from app.ingestion.docx_parser import parse_docx_bytes
    except ImportError:
        st.error("Не установлены зависимости для DOCX. Запустите bootstrap_env.bat.")
        return st.session_state.get("workspace_id")

    content = uploaded_file.getvalue()
    file_hash = file_hash_for_content(content)
    workspace_id = workspace_id_for(uploaded_file.name, file_hash)
    info = load_workspace_info(workspace_id)

    if info and info.embedding_count > 0 and info.embed_model == selection.embed_model:
        st.success("Готовая локальная база уже есть. Она загружена из кэша.")
        _render_workspace_summary(workspace_id)
        return workspace_id

    st.info("Готовлю локальную базу для выбранного документа.")
    parse_progress = st.progress(0, text="Индексация документа...")

    parsed = parse_docx_bytes(uploaded_file.name, content)
    save_uploaded_docx(uploaded_file.name, content)
    save_parsed_payload(uploaded_file.name, parsed.to_dict())
    records = build_search_records(parsed)
    save_workspace_records(workspace_id, parsed.source_name, file_hash, records)
    parse_progress.progress(1.0, text=f"Индексация завершена: {len(records)} фрагментов, ~{_records_tokens(records)} токенов")

    _build_workspace_embeddings(workspace_id, provider, selection, records)
    _render_workspace_summary(workspace_id)
    return workspace_id


def _build_workspace_embeddings(workspace_id: str, provider, selection, records: list[SearchRecord]) -> None:
    info = load_workspace_info(workspace_id)
    if info and info.embedding_count > 0 and info.embed_model == selection.embed_model:
        st.success("Embeddings уже построены для этой модели. Повторно не считаю.")
        return

    total_records = len([record for record in records if record.record_type != "table" and record.text.strip()])
    total_tokens = sum(estimate_tokens(record.text) for record in records if record.record_type != "table" and record.text.strip())
    progress = st.progress(0, text=f"Строю embeddings: 0/{total_records}, ~0/{total_tokens} токенов")
    log_box = st.empty()
    log_lines: list[str] = []
    processed_tokens = 0

    def on_progress(index: int, total: int, record: SearchRecord, token_count: int) -> None:
        nonlocal processed_tokens
        processed_tokens += token_count
        progress.progress(
            index / max(total, 1),
            text=f"Строю embeddings: {index}/{total}, ~{processed_tokens}/{total_tokens} токенов",
        )
        log_lines.append(f"{index}/{total}: {record.record_id} ({record.record_type}), ~{token_count} tokens")
        log_box.code("\n".join(log_lines[-12:]), language="text")

    try:
        embeddings = build_embedding_index(
            provider=provider,
            records=records,
            model=selection.embed_model,
            progress_callback=on_progress,
        )
        save_workspace_embeddings(workspace_id, embeddings, selection.embed_model)
    except Exception as exc:
        st.warning(f"Индекс exact готов, но embeddings не построены: {exc}")
        return

    st.success(f"Embeddings построены: {len(embeddings)} фрагментов, примерно {processed_tokens} токенов.")


def _render_workspace_summary(workspace_id: str) -> None:
    info = load_workspace_info(workspace_id)
    if info is None:
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Документ", info.source_name)
    col2.metric("Фрагментов", info.record_count)
    col3.metric("Embeddings", info.embedding_count)
    col4.metric("Токенов ~", info.token_count)


def _workspace_label(info) -> str:
    embed_status = "embeddings есть" if info.embedding_count else "только exact"
    return f"{info.source_name} | {info.record_count} фрагм. | {embed_status}"


def _records_tokens(records: list[SearchRecord]) -> int:
    return sum(estimate_tokens(record.text) for record in records)


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


def _render_diagnostics(provider_name: str, selection, workspace_id: str | None) -> None:
    with st.expander("Диагностика"):
        st.write("Текущая база")
        st.json(asdict(load_workspace_info(workspace_id)) if workspace_id and load_workspace_info(workspace_id) else {})

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
