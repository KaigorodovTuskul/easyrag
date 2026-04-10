from __future__ import annotations

from dataclasses import asdict
import re

from app.agents.answer import build_answer_context
from app.agents.code_lookup import try_build_code_lookup_answer
from app.agents.controller import run_agent_retrieval
from app.agents.embedding_indexer import build_embedding_index
from app.agents.norm_lookup import try_build_norm_lookup_answer
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
    st.caption("Загрузите DOCX и задавайте вопросы в чате.")

    tab_chat, tab_debug = st.tabs(["Чат", "Отладка"])

    with tab_chat:
        workspace_id = _render_workspace_picker(provider, selection)
        _render_chat(provider, selection, workspace_id)

    with tab_debug:
        _render_debug(provider.name, selection, st.session_state.get("workspace_id"))


def _render_workspace_picker(provider, selection) -> str | None:
    workspaces = list_workspaces()
    selected_workspace_id = st.session_state.get("workspace_id")

    with st.container(border=True):
        col_left, col_right = st.columns([1, 1])

        with col_left:
            uploaded_file = st.file_uploader("Загрузить DOCX", type=["docx"])
            if uploaded_file is not None:
                selected_workspace_id = _prepare_uploaded_workspace(uploaded_file, provider, selection)
                st.session_state["workspace_id"] = selected_workspace_id

        with col_right:
            if workspaces:
                labels = [_workspace_label(item) for item in workspaces]
                ids = [item.workspace_id for item in workspaces]
                selected_index = ids.index(selected_workspace_id) if selected_workspace_id in ids else 0
                selected_label = st.selectbox("Или открыть готовую базу", labels, index=selected_index)
                selected_workspace_id = ids[labels.index(selected_label)]
                st.session_state["workspace_id"] = selected_workspace_id
            else:
                st.info("Готовых баз пока нет.")

        if selected_workspace_id:
            _render_workspace_summary(selected_workspace_id)

    return selected_workspace_id


def _render_chat(provider, selection, workspace_id: str | None) -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if not workspace_id:
        st.info("Сначала загрузите DOCX. После подготовки базы чат станет доступен.")
        return

    if st.button("Очистить диалог"):
        _clear_chat_context()
        st.rerun()

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Напишите вопрос по документу")
    if not prompt:
        return

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer = _answer_question(provider, selection, workspace_id, prompt)
        st.markdown(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})


def _answer_question(provider, selection, workspace_id: str, prompt: str) -> str:
    records = load_workspace_records(workspace_id)
    embeddings = load_workspace_embeddings(workspace_id)
    effective_query = _build_effective_query(prompt)
    st.session_state["last_effective_query"] = effective_query

    agent_result = run_agent_retrieval(
        provider=provider,
        records=records,
        embedding_records=embeddings,
        query=effective_query,
        embed_model=selection.embed_model,
        limit=12,
    )

    st.session_state["last_debug"] = {
        "query": prompt,
        "effective_query": effective_query,
        "workspace_id": workspace_id,
        "query_type": agent_result.query_type,
        "mode": agent_result.mode,
        "evidence": asdict(agent_result.evidence),
        "steps": [{"name": step.name, "details": step.details} for step in agent_result.steps],
        "results": [_result_to_debug(result) for result in agent_result.results],
    }

    if not agent_result.results:
        return "Не нашел подходящий фрагмент в выбранном документе. Попробуйте точный код, номер счета или другую формулировку."

    code_answer = try_build_code_lookup_answer(effective_query, agent_result.results)
    if code_answer is not None:
        st.session_state["last_debug"]["answer_mode"] = "deterministic_code_lookup"
        return code_answer.answer

    norm_answer = try_build_norm_lookup_answer(effective_query, agent_result.results)
    if norm_answer is not None:
        st.session_state["last_debug"]["answer_mode"] = "deterministic_norm_lookup"
        return norm_answer.answer

    answer_context = build_answer_context(effective_query, agent_result.results, evidence=agent_result.evidence)
    st.session_state["last_debug"]["citations"] = answer_context.citations

    try:
        generated = provider.generate(answer_context.prompt, model=selection.chat_model)
    except ProviderError as exc:
        return f"Нашел контекст, но модель ответа недоступна: {exc}"
    except Exception as exc:
        return f"Нашел контекст, но не смог сформировать ответ: {exc}"

    return generated.text or "Модель вернула пустой ответ."


def _build_effective_query(prompt: str) -> str:
    target = _extract_query_target(prompt)
    if target:
        st.session_state["last_user_query"] = prompt
        st.session_state["last_query_target"] = target
        return prompt

    previous_user_message = st.session_state.get("last_user_query") or _previous_user_message()
    if not previous_user_message:
        st.session_state["last_user_query"] = prompt
        return prompt

    st.session_state["last_user_query"] = previous_user_message
    return f"{previous_user_message}\nУточнение: {prompt}"


def _previous_user_message() -> str | None:
    # Current prompt has already been appended to session_state messages.
    for message in reversed(st.session_state.get("messages", [])[:-1]):
        if message.get("role") == "user":
            return message.get("content")
    return None


def _extract_query_target(prompt: str) -> str | None:
    norm_match = re.search(r"\b[НN]\s*\d+(?:\.\d+)?\b", prompt, flags=re.IGNORECASE)
    if norm_match:
        return norm_match.group(0).replace(" ", "").upper()

    numeric_match = re.search(r"\b\d{3,}(?:\.\d+)?\b", prompt)
    if numeric_match:
        return numeric_match.group(0)

    return None


def _clear_chat_context() -> None:
    for key in [
        "messages",
        "last_debug",
        "last_user_query",
        "last_query_target",
        "last_effective_query",
    ]:
        st.session_state.pop(key, None)


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
        st.success("База уже готова. Открываю из кэша.")
        return workspace_id

    st.info("Готовлю базу. Это нужно сделать один раз для документа.")
    parse_progress = st.progress(0, text="Индексация документа...")

    parsed = parse_docx_bytes(uploaded_file.name, content)
    save_uploaded_docx(uploaded_file.name, content)
    save_parsed_payload(uploaded_file.name, parsed.to_dict())
    records = build_search_records(parsed)
    save_workspace_records(workspace_id, parsed.source_name, file_hash, records)
    parse_progress.progress(1.0, text=f"Индексация готова: {len(records)} фрагментов, ~{_records_tokens(records)} токенов")

    _build_workspace_embeddings(workspace_id, provider, selection, records)
    st.session_state["messages"] = []
    return workspace_id


def _build_workspace_embeddings(workspace_id: str, provider, selection, records: list[SearchRecord]) -> None:
    info = load_workspace_info(workspace_id)
    if info and info.embedding_count > 0 and info.embed_model == selection.embed_model:
        st.success("Embeddings уже есть. Повторно не считаю.")
        return

    selected = [record for record in records if record.record_type != "table" and record.text.strip()]
    total_records = len(selected)
    total_tokens = sum(estimate_tokens(record.text) for record in selected)
    progress = st.progress(0, text=f"Embeddings: 0/{total_records}, ~0/{total_tokens} токенов")
    log_box = st.empty()
    log_lines: list[str] = []
    processed_tokens = 0

    def on_progress(index: int, total: int, record: SearchRecord, token_count: int) -> None:
        nonlocal processed_tokens
        processed_tokens += token_count
        progress.progress(index / max(total, 1), text=f"Embeddings: {index}/{total}, ~{processed_tokens}/{total_tokens} токенов")
        log_lines.append(f"{index}/{total}: {record.record_id} ({record.record_type}), ~{token_count} tokens")
        log_box.code("\n".join(log_lines[-8:]), language="text")

    try:
        embeddings = build_embedding_index(
            provider=provider,
            records=records,
            model=selection.embed_model,
            progress_callback=on_progress,
        )
        save_workspace_embeddings(workspace_id, embeddings, selection.embed_model)
    except Exception as exc:
        st.warning(f"Exact-поиск готов, но embeddings не построены: {exc}")
        return

    st.success(f"Embeddings готовы: {len(embeddings)} фрагментов, примерно {processed_tokens} токенов.")


def _render_workspace_summary(workspace_id: str) -> None:
    info = load_workspace_info(workspace_id)
    if info is None:
        return
    st.caption(f"Текущая база: {info.source_name} · фрагментов: {info.record_count} · embeddings: {info.embedding_count}")


def _render_debug(provider_name: str, selection, workspace_id: str | None) -> None:
    st.subheader("Отладка")
    st.caption("Эта вкладка для суперпользователя: источники, trace, модели, eval.")

    info = load_workspace_info(workspace_id) if workspace_id else None
    st.write("Текущая база")
    st.json(asdict(info) if info else {})

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

    st.write("Последний запрос")
    st.json(st.session_state.get("last_debug", {}))

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


def _workspace_label(info) -> str:
    embed_status = "embeddings есть" if info.embedding_count else "только exact"
    return f"{info.source_name} | {info.record_count} фрагм. | {embed_status}"


def _records_tokens(records: list[SearchRecord]) -> int:
    return sum(estimate_tokens(record.text) for record in records)


def _result_to_debug(result) -> dict:
    return {
        "score": round(result.score, 2),
        "record_id": result.record.record_id,
        "record_type": result.record.record_type,
        "source_name": result.record.source_name,
        "section_path": result.record.section_path,
        "matched_terms": result.matched_terms,
        "snippet": result.snippet,
    }


if __name__ == "__main__":
    main()
