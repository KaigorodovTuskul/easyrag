from __future__ import annotations

from dataclasses import asdict
import re
import time

from app.agents.answer import build_answer_context
from app.agents.code_lookup import try_build_code_lookup_answer, try_build_code_topic_answer
from app.agents.controller import run_agent_retrieval
from app.agents.embedding_indexer import build_embedding_index, select_embedding_records
from app.agents.norm_lookup import try_build_norm_lookup_answer
from app.agents.query_understanding import QueryUnderstanding, build_query_suggestions
from app.agents.term_lookup import try_build_term_lookup_answer
from app.core.config import AppConfig
from app.core.i18n import SUPPORTED_LANGUAGES, normalize_language, t
from app.ingestion.formula_enrichment import enrich_formula_records
from app.providers.base import BaseProvider, ProviderError
from app.providers.ollama import OllamaProvider
from app.providers.openrouter import OpenRouterProvider
from app.providers.router import ProviderRouter
from app.retrieval.records import SearchRecord, build_search_records
from app.storage.conversations import clear_conversation, load_conversation, save_conversation
from app.storage.files import save_parsed_payload, save_uploaded_docx
from app.storage.formula_images import (
    load_workspace_formula_images,
    read_formula_image_for_display,
    save_workspace_formula_images,
)
from app.storage.entities import save_workspace_entities
from app.storage.workspaces import (
    delete_workspace,
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
    language = _render_language_settings(config)
    provider, selection = _render_provider_settings(config, language)
    workspace_id = _render_sidebar_workspace(config, provider, selection, language)

    st.title("EasyRAG")
    st.caption(t("app.caption", language))

    _render_chat(provider, selection.chat_model, selection.embed_model, workspace_id, language)


def _render_language_settings(config: AppConfig) -> str:
    if "language" not in st.session_state:
        st.session_state["language"] = normalize_language(config.app_language)

    configured = normalize_language(st.session_state["language"])
    st.sidebar.header(t("language.header", configured))
    labels = {"ru": "RU", "en": "EN"}
    selected = st.sidebar.radio(
        t("language.label", configured),
        list(SUPPORTED_LANGUAGES),
        index=list(SUPPORTED_LANGUAGES).index(configured),
        format_func=lambda value: labels[value],
        horizontal=True,
        key="language",
    )
    return selected


def _render_provider_settings(config: AppConfig, language: str):
    st.sidebar.header(t("model.header", language))
    mode = st.sidebar.radio(t("provider.label", language), ["auto", "ollama", "openrouter"], horizontal=True)

    provider, selection = _resolve_provider(config, mode, language)
    available_model_names = [model.name for model in selection.available_models]

    selection.chat_model = _render_model_selectbox(
        label=t("chat_model.label", language),
        available=available_model_names,
        current=selection.chat_model,
        key="chat_model_select",
    )
    selection.embed_model = _render_model_selectbox(
        label=t("embed_model.label", language),
        available=available_model_names,
        current=selection.embed_model,
        key="embed_model_select",
    )
    selection.vision_model = _render_model_selectbox(
        label=t("vision_model.label", language),
        available=available_model_names,
        current=selection.vision_model,
        key="vision_model_select",
        allow_none=True,
    )

    with st.sidebar.expander(t("provider_status.title", language)):
        st.write(
            {
                t("provider_status.provider", language): provider.name,
                t("provider_status.reachable", language): selection.reachable,
                t("provider_status.active_model", language): selection.active_model,
                t("provider_status.vision_model", language): selection.vision_model or "None",
                t("provider_status.reason", language): _localize_reason(selection.reason, language),
            }
        )

    return provider, selection


def _resolve_provider(config: AppConfig, mode: str, language: str | None = None) -> tuple[BaseProvider, object]:
    if mode == "ollama":
        provider = OllamaProvider(config)
        try:
            return provider, provider.resolve_selection()
        except ProviderError as exc:
            st.sidebar.warning(t("ollama_unavailable", normalize_language(language), error=exc))
            fallback = OpenRouterProvider(config)
            return fallback, fallback.resolve_selection()

    if mode == "openrouter":
        provider = OpenRouterProvider(config)
        return provider, provider.resolve_selection()

    return ProviderRouter(config).resolve()


def _model_options(available: list[str], current: str | None) -> list[str]:
    options = [current] if current else []
    options.extend(name for name in available if name and name not in options)
    return options


def _render_model_selectbox(
    label: str,
    available: list[str],
    current: str | None,
    key: str,
    allow_none: bool = False,
) -> str | None:
    none_option = "__none__"
    options = _model_options(available, current)
    choice_options = ([none_option] if allow_none else []) + options

    if current is None and allow_none:
        selected_choice = none_option
    elif current:
        selected_choice = current
    elif options:
        selected_choice = options[0]
    else:
        selected_choice = none_option if allow_none else None

    if selected_choice is None:
        return None

    selected = st.sidebar.selectbox(
        label,
        choice_options,
        index=choice_options.index(selected_choice),
        format_func=lambda value: _format_model_option(value, st.session_state.get("language", "ru")),
        key=key,
    )

    if allow_none and selected == none_option:
        return None
    return selected


def _format_model_option(value: str, language: str) -> str:
    if value == "__none__":
        return "None"
    return value


def _localize_reason(reason: str, language: str) -> str:
    replacements = {
        "Active model from /api/ps is selected": t("reason.active_ollama_model", language),
        "Default model selection applied": t("reason.default_model", language),
        "OPENROUTER_API_KEY is not set": t("reason.openrouter_key_missing", language),
        "Configured OpenRouter fallback": t("reason.openrouter_fallback", language),
        "Ollama unavailable": t("reason.ollama_unavailable", language),
    }
    localized = reason
    for source, target in replacements.items():
        localized = localized.replace(source, target)
    return localized


def _render_sidebar_workspace(config: AppConfig, provider, selection, language: str) -> str | None:
    st.sidebar.header(t("document.header", language))
    workspaces = list_workspaces()
    selected_workspace_id = st.session_state.get("workspace_id")

    if workspaces:
        labels = [_workspace_label(item, language) for item in workspaces]
        ids = [item.workspace_id for item in workspaces]
        selected_index = ids.index(selected_workspace_id) if selected_workspace_id in ids else 0
        selected_label = st.sidebar.selectbox(t("workspace.select", language), labels, index=selected_index)
        selected_workspace_id = ids[labels.index(selected_label)]
        _set_workspace(selected_workspace_id)
    else:
        st.sidebar.info(t("workspace.empty", language))

    uploaded_file = st.sidebar.file_uploader(t("upload.label", language), type=["docx"])
    if uploaded_file is not None:
        selected_workspace_id = _prepare_uploaded_workspace(uploaded_file, provider, selection, config, language)
        _set_workspace(selected_workspace_id)

    if selected_workspace_id:
        _render_workspace_summary(selected_workspace_id, language)
        _render_formula_enrichment_controls(selected_workspace_id, provider, selection, config, language)
        if st.sidebar.button(t("clear_chat", language)):
            _clear_chat_context(selected_workspace_id)
            st.rerun()
        if st.sidebar.button(t("workspace.delete", language), type="secondary"):
            _delete_workspace(selected_workspace_id, language)
            st.rerun()

    _render_debug(provider.name, selection, selected_workspace_id, language)
    return selected_workspace_id


def _set_workspace(workspace_id: str) -> None:
    if st.session_state.get("workspace_id") == workspace_id:
        return
    st.session_state["workspace_id"] = workspace_id
    st.session_state["messages"] = load_conversation(workspace_id)
    st.session_state.pop("last_debug", None)
    st.session_state.pop("last_user_query", None)
    st.session_state.pop("last_query_target", None)
    st.session_state.pop("last_effective_query", None)


def _render_chat(provider, chat_model: str, embed_model: str, workspace_id: str | None, language: str) -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = load_conversation(workspace_id) if workspace_id else []

    if not workspace_id:
        st.info(t("no_workspace", language))
        return

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            _render_formula_images(message.get("formula_images"))

    prompt = st.chat_input(t("chat.placeholder", language))
    if not prompt:
        return

    st.session_state["messages"].append({"role": "user", "content": prompt})
    save_conversation(workspace_id, st.session_state["messages"])

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer, formula_images = _answer_question(provider, chat_model, embed_model, workspace_id, prompt, language)
        st.write_stream(_stream_text(answer))
        _render_formula_images(formula_images)

    assistant_message = {"role": "assistant", "content": answer}
    if formula_images:
        assistant_message["formula_images"] = formula_images
    st.session_state["messages"].append(assistant_message)
    save_conversation(workspace_id, st.session_state["messages"])


def _answer_question(provider, chat_model: str, embed_model: str, workspace_id: str, prompt: str, language: str) -> tuple[str, list[dict]]:
    records = load_workspace_records(workspace_id)
    embeddings = load_workspace_embeddings(workspace_id)
    entities = save_workspace_entities(workspace_id, records)
    formula_images_by_id = load_workspace_formula_images(workspace_id)

    if _is_document_summary_query(prompt):
        summary_answer = _build_document_summary(records, language)
        st.session_state["last_debug"] = {
            "query": prompt,
            "effective_query": prompt,
            "query_resolution": "document_summary",
            "workspace_id": workspace_id,
            "answer_mode": "deterministic_document_summary",
        }
        return summary_answer, []

    effective_query, agent_result, query_resolution = _resolve_query_with_context(
        provider=provider,
        records=records,
        embeddings=embeddings,
        entities=entities,
        prompt=prompt,
        embed_model=embed_model,
    )
    st.session_state["last_effective_query"] = effective_query

    st.session_state["last_debug"] = {
        "query": prompt,
        "effective_query": effective_query,
        "query_resolution": query_resolution,
        "workspace_id": workspace_id,
        "query_type": agent_result.query_type,
        "query_entity": agent_result.entity,
        "mode": agent_result.mode,
        "evidence": asdict(agent_result.evidence),
        "steps": [{"name": step.name, "details": step.details} for step in agent_result.steps],
        "results": [_result_to_debug(result) for result in agent_result.results],
    }

    if not agent_result.results:
        return t("no_results", language), []

    code_answer = try_build_code_lookup_answer(effective_query, agent_result.results, language=language)
    if code_answer is not None:
        st.session_state["last_debug"]["answer_mode"] = "deterministic_code_lookup"
        return code_answer.answer, _collect_formula_images_for_code_answer(code_answer, agent_result.results, formula_images_by_id)

    code_topic_answer = try_build_code_topic_answer(effective_query, records, language=language)
    if code_topic_answer is not None:
        st.session_state["last_debug"]["answer_mode"] = "deterministic_code_topic_lookup"
        return code_topic_answer.answer, _collect_formula_images_for_record_ids(agent_result.results, set(code_topic_answer.source_record_ids), formula_images_by_id)

    norm_answer = try_build_norm_lookup_answer(effective_query, agent_result.results, language=language)
    if norm_answer is not None:
        st.session_state["last_debug"]["answer_mode"] = "deterministic_norm_lookup"
        return norm_answer.answer, _collect_formula_images(agent_result.results, formula_images_by_id, agent_result.query_type, agent_result.entity)

    term_answer = try_build_term_lookup_answer(
        QueryUnderstanding(intent=agent_result.query_type, entity=agent_result.entity),
        agent_result.results,
        records,
        language=language,
    )
    if term_answer is not None:
        st.session_state["last_debug"]["answer_mode"] = "deterministic_term_lookup"
        return term_answer.answer, []

    if not agent_result.evidence.ok:
        st.session_state["last_debug"]["answer_mode"] = "weak_evidence_refusal"
        return _with_suggestions(t("no_results", language), prompt, agent_result, language), []

    answer_context = build_answer_context(
        effective_query,
        agent_result.results,
        max_results=_answer_context_limit(agent_result.results, agent_result.query_type, agent_result.entity),
        evidence=agent_result.evidence,
        language=language,
    )
    st.session_state["last_debug"]["citations"] = answer_context.citations

    try:
        generated = provider.generate(answer_context.prompt, model=chat_model)
    except ProviderError as exc:
        return t("model_unavailable", language, error=exc), []
    except Exception as exc:
        return t("answer_failed", language, error=exc), []

    answer_text = generated.text or t("empty_answer", language)
    return _with_suggestions(answer_text, prompt, agent_result, language), _collect_formula_images_for_citations(
        answer_context.citations,
        agent_result.results,
        formula_images_by_id,
        query_type=agent_result.query_type,
        question=effective_query,
    )


def _resolve_query_with_context(provider, records: list[SearchRecord], embeddings, entities: list[str], prompt: str, embed_model: str):
    standalone_result = run_agent_retrieval(
        provider=provider,
        records=records,
        embedding_records=embeddings,
        query=prompt,
        embed_model=embed_model,
        entity_names=entities,
        limit=12,
    )

    target = _extract_query_target(prompt)
    if target:
        st.session_state["last_user_query"] = prompt
        st.session_state["last_query_target"] = target
        return prompt, standalone_result, "standalone_explicit_target"

    if standalone_result.results:
        st.session_state["last_user_query"] = prompt
        st.session_state.pop("last_query_target", None)
        return prompt, standalone_result, "standalone_found_results"

    previous_user_message = st.session_state.get("last_user_query") or _previous_user_message()
    if not previous_user_message:
        st.session_state["last_user_query"] = prompt
        return prompt, standalone_result, "standalone_no_previous_context"

    st.session_state["last_user_query"] = previous_user_message
    contextual_query = f"{previous_user_message}\n{t('clarification.prefix', st.session_state.get('language', 'ru'))}: {prompt}"
    contextual_result = run_agent_retrieval(
        provider=provider,
        records=records,
        embedding_records=embeddings,
        query=contextual_query,
        embed_model=embed_model,
        entity_names=entities,
        limit=12,
    )
    return contextual_query, contextual_result, "contextual_after_empty_standalone"


def _previous_user_message() -> str | None:
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


def _answer_context_limit(results: list, query_type: str, entity: str | None = None) -> int:
    if query_type in {"definition", "composition", "formula", "norm"} or entity:
        return 12
    if any(result.record.record_type in {"table_cell", "table_row"} for result in results[:12]):
        return 12
    return 6


def _clear_chat_context(workspace_id: str) -> None:
    clear_conversation(workspace_id)
    for key in [
        "messages",
        "last_debug",
        "last_user_query",
        "last_query_target",
        "last_effective_query",
    ]:
        st.session_state.pop(key, None)


def _delete_workspace(workspace_id: str, language: str) -> None:
    try:
        delete_workspace(workspace_id)
    except Exception as exc:
        st.sidebar.warning(t("workspace.delete_failed", language, error=exc))
        return

    for key in [
        "workspace_id",
        "messages",
        "last_debug",
        "last_user_query",
        "last_query_target",
        "last_effective_query",
    ]:
        st.session_state.pop(key, None)
    st.sidebar.success(t("workspace.deleted", language))


def _prepare_uploaded_workspace(uploaded_file, provider, selection, config: AppConfig, language: str) -> str:
    try:
        from app.ingestion.docx_parser import parse_docx_bytes
    except ImportError:
        st.sidebar.error(t("docx_deps_missing", language))
        return st.session_state.get("workspace_id")

    content = uploaded_file.getvalue()
    file_hash = file_hash_for_content(content)
    workspace_id = workspace_id_for(
        uploaded_file.name,
        _workspace_cache_hash(
            content,
            config,
        ),
    )
    info = load_workspace_info(workspace_id)

    if info and info.embedding_count > 0 and info.embed_model == selection.embed_model:
        st.sidebar.success(t("workspace_cached", language))
        return workspace_id

    st.sidebar.info(t("workspace_preparing", language))
    parse_progress = st.sidebar.progress(0, text=t("indexing.progress", language))

    parsed = parse_docx_bytes(
        uploaded_file.name,
        content,
    )
    save_uploaded_docx(uploaded_file.name, content)
    save_parsed_payload(uploaded_file.name, parsed.to_dict())
    save_workspace_formula_images(workspace_id, parsed.formula_images)
    records = build_search_records(parsed)
    records = _enrich_records_with_formulas(records, workspace_id, provider, selection, config, language)
    save_workspace_records(workspace_id, parsed.source_name, file_hash, records)
    save_workspace_entities(workspace_id, records)
    parse_progress.progress(1.0, text=t("indexing.done", language, records=len(records), tokens=_records_tokens(records)))

    _build_workspace_embeddings(workspace_id, provider, selection, records, config, language)
    st.session_state["messages"] = []
    save_conversation(workspace_id, [])
    return workspace_id


def _workspace_cache_hash(
    content: bytes,
    config: AppConfig,
) -> str:
    parser_signature = (
        "docx-parser:formula-v4:assets-only:"
        f"embedding-types={','.join(config.embedding_record_types)}"
    ).encode("utf-8")
    return file_hash_for_content(content + parser_signature)


def _build_workspace_embeddings(
    workspace_id: str,
    provider,
    selection,
    records: list[SearchRecord],
    config: AppConfig,
    language: str,
) -> None:
    info = load_workspace_info(workspace_id)
    if info and info.embedding_count > 0 and info.embed_model == selection.embed_model:
        st.sidebar.success(t("embeddings.cached", language))
        return

    selected = select_embedding_records(records, record_types=config.embedding_record_types)
    total_records = len(selected)
    total_tokens = sum(estimate_tokens(record.text) for record in selected)
    progress = st.sidebar.progress(0, text=t("embeddings.progress", language, index=0, total=total_records, processed=0, tokens=total_tokens))
    log_box = st.sidebar.empty()
    log_lines: list[str] = []
    processed_tokens = 0

    def on_progress(index: int, total: int, record: SearchRecord, token_count: int) -> None:
        nonlocal processed_tokens
        processed_tokens += token_count
        progress.progress(
            index / max(total, 1),
            text=t("embeddings.progress", language, index=index, total=total, processed=processed_tokens, tokens=total_tokens),
        )
        log_lines.append(f"{index}/{total}: {record.record_id} ({record.record_type}), ~{token_count} tokens")
        log_box.code("\n".join(log_lines[-8:]), language="text")

    try:
        embeddings = build_embedding_index(
            provider=provider,
            records=records,
            model=selection.embed_model,
            batch_size=config.embedding_batch_size,
            record_types=config.embedding_record_types,
            progress_callback=on_progress,
        )
        save_workspace_embeddings(workspace_id, embeddings, selection.embed_model)
    except Exception as exc:
        st.sidebar.warning(t("embeddings.failed", language, error=exc))
        return

    st.sidebar.success(t("embeddings.done", language, records=len(embeddings), tokens=processed_tokens))


def _render_workspace_summary(workspace_id: str, language: str) -> None:
    info = load_workspace_info(workspace_id)
    if info is None:
        return
    st.sidebar.caption(t("workspace.current", language, source=info.source_name))
    st.sidebar.caption(t("workspace.stats", language, records=info.record_count, embeddings=info.embedding_count))


def _render_formula_enrichment_controls(workspace_id: str, provider, selection, config: AppConfig, language: str) -> None:
    if not st.sidebar.button(t("formula.enrich", language), key=f"formula_enrich_{workspace_id}"):
        return

    added_records, added_embeddings = _run_formula_enrichment(workspace_id, provider, selection, config)
    if added_records:
        st.sidebar.success(
            t(
                "formula.enrich_done",
                language,
                records=added_records,
                embeddings=added_embeddings,
            )
        )
        if st.session_state.get("workspace_id") == workspace_id:
            st.session_state["messages"] = load_conversation(workspace_id)
    else:
        st.sidebar.info(t("formula.enrich_empty", language))


def _run_formula_enrichment(workspace_id: str, provider, selection, config: AppConfig) -> tuple[int, int]:
    records = load_workspace_records(workspace_id)
    formula_images = load_workspace_formula_images(workspace_id)
    effective_model = _effective_formula_model(selection)
    if not effective_model:
        return 0, 0

    enrichment = enrich_formula_records(
        provider=provider,
        records=records,
        formula_images=formula_images,
        model=effective_model,
        embed_model=selection.embed_model,
        config=config,
    )
    if not enrichment.created_records:
        return 0, 0

    info = load_workspace_info(workspace_id)
    if info is None:
        return 0, 0

    combined_records = [*records, *enrichment.created_records]
    save_workspace_records(workspace_id, info.source_name, info.file_hash, combined_records)
    save_workspace_entities(workspace_id, combined_records)

    existing_embeddings = load_workspace_embeddings(workspace_id)
    save_workspace_embeddings(workspace_id, [*existing_embeddings, *enrichment.created_embeddings], selection.embed_model)
    return len(enrichment.created_records), len(enrichment.created_embeddings)


def _enrich_records_with_formulas(records: list[SearchRecord], workspace_id: str, provider, selection, config: AppConfig, language: str) -> list[SearchRecord]:
    if not _can_attempt_formula_vision(selection):
        return records

    enrichment = enrich_formula_records(
        provider=provider,
        records=records,
        formula_images=load_workspace_formula_images(workspace_id),
        model=_effective_formula_model(selection),
        embed_model=None,
        config=config,
    )
    if enrichment.created_records:
        st.sidebar.caption(
            t(
                "formula.auto_enrich",
                language,
                records=len(enrichment.created_records),
            )
        )
    return [*records, *enrichment.created_records]


def _render_debug(provider_name: str, selection, workspace_id: str | None, language: str) -> None:
    with st.sidebar.expander(t("debug.title", language)):
        info = load_workspace_info(workspace_id) if workspace_id else None
        st.write(t("debug.workspace", language))
        st.json(asdict(info) if info else {})

        st.write(t("debug.models", language))
        st.json(
            {
                "provider": provider_name,
                "chat_model": selection.chat_model,
                "embed_model": selection.embed_model,
                "vision_model": selection.vision_model,
                "active_model": selection.active_model,
                "reason": selection.reason,
            }
        )

        st.write(t("debug.last_query", language))
        st.json(st.session_state.get("last_debug", {}))


def _workspace_label(info, language: str) -> str:
    embed_status = t("workspace.embeddings_ready", language) if info.embedding_count else t("workspace.exact_only", language)
    return f"{info.source_name} | {info.record_count} {t('workspace.fragments_short', language)} | {embed_status}"


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


def _with_suggestions(answer: str, prompt: str, agent_result, language: str) -> str:
    should_suggest = (not agent_result.results) or (not agent_result.evidence.ok) or (
        agent_result.query_type in {"definition", "composition", "formula", "norm"} and not agent_result.entity
    )
    if not should_suggest:
        return answer

    suggestions = build_query_suggestions(prompt, agent_result.entity, agent_result.query_type, language=language)
    if not suggestions:
        return answer

    header = "Maybe you meant:" if language == "en" else "Может, вы имели в виду:"
    lines = "\n".join(f"- {item}" for item in suggestions)
    if not answer.strip():
        return f"{header}\n{lines}"
    return f"{answer}\n\n{header}\n{lines}"


def _stream_text(text: str):
    for chunk in _stream_chunks(text):
        yield chunk
        time.sleep(0.01)


def _stream_chunks(text: str):
    buffer = ""
    for char in text:
        buffer += char
        if char.isspace() or char in ".,;:!?)]}":
            yield buffer
            buffer = ""
    if buffer:
        yield buffer


def _is_document_summary_query(prompt: str) -> bool:
    normalized = prompt.lower().replace("ё", "е")
    return any(
        marker in normalized
        for marker in [
            "о чем документ",
            "о чем этот документ",
            "расскажи вкратце",
            "кратко о документе",
            "краткое содержание",
            "summary of the document",
            "summarize the document",
            "what is this document about",
        ]
    )


def _build_document_summary(records: list[SearchRecord], language: str) -> str:
    title_parts = [record.text.strip() for record in records[:12] if record.record_type == "paragraph" and record.text.strip()]
    title = " ".join(title_parts[:8]).strip()
    headings: list[str] = []
    for record in records:
        if record.record_type != "paragraph":
            continue
        if not record.section_path:
            continue
        section = record.section_path[-1].strip()
        if section and section not in headings:
            headings.append(section)
        if len(headings) >= 4:
            break

    appendix_rows = [record for record in records if record.record_type == "table_row" and "Приложение 1" in record.section_path]
    if language == "en":
        lines = [f"The document is about banking mandatory ratios and related calculation codes."]
        if title:
            lines.append(f"Title context: {title}.")
        if headings:
            lines.append(f"Main sections: {', '.join(headings)}.")
        if appendix_rows:
            lines.append("It also contains Appendix 1 with codes used in ratio calculations and risk-related indicators.")
    else:
        lines = ["Документ посвящен обязательным нормативам банков и кодам, которые используются при их расчете."]
        if title:
            lines.append(f"Контекст заголовка: {title}.")
        if headings:
            lines.append(f"Основные разделы: {', '.join(headings)}.")
        if appendix_rows:
            lines.append("В документе также есть Приложение 1 с кодами, используемыми при расчете нормативов и связанных рисков.")
    return "\n\n".join(lines)


def _collect_formula_images(results, formula_images_by_id: dict[str, object], query_type: str | None = None, entity: str | None = None, limit: int = 6) -> list[dict]:
    if query_type in {"formula", "norm"} and entity:
        prioritized = _collect_formula_images_from_anchor_window(results, formula_images_by_id, before=0, after=3, limit=limit)
        if prioritized:
            return prioritized

    collected: list[dict] = []
    seen: set[str] = set()
    for result in results:
        asset_ids = result.record.metadata.get("formula_image_ids", [])
        if not isinstance(asset_ids, list):
            continue
        for asset_id in asset_ids:
            if asset_id in seen or asset_id not in formula_images_by_id:
                continue
            asset = formula_images_by_id[asset_id]
            collected.append({"asset_id": asset.asset_id, "filename": asset.filename, "path": asset.relative_path})
            seen.add(asset_id)
            if len(collected) >= limit:
                return collected
    return collected


def _collect_formula_images_for_citations(
    citations: list[dict[str, str | float]],
    results,
    formula_images_by_id: dict[str, object],
    query_type: str | None = None,
    question: str | None = None,
    limit: int = 6,
) -> list[dict]:
    normalized_question = (question or "").lower().replace("ё", "е")
    asks_formula = query_type in {"formula", "norm"} or any(
        token in normalized_question for token in ["формул", "рассчит", "calculate", "formula", "how is"]
    )
    if not asks_formula:
        return []

    cited_ids = {str(item.get("id")) for item in citations if item.get("id")}
    if not cited_ids:
        return []
    return _collect_formula_images_for_record_ids(results, cited_ids, formula_images_by_id, limit=limit)


def _collect_formula_images_for_code_answer(code_answer, results, formula_images_by_id: dict[str, object], limit: int = 6) -> list[dict]:
    direct = _collect_formula_images_for_record_ids(results, {code_answer.source_record_id}, formula_images_by_id, limit=limit)
    if direct:
        return direct

    if not code_answer.asks_calculation or not code_answer.related_norms:
        return []

    norm_related = [
        result
        for result in results
        if result.record.record_type == "paragraph"
        and any(_record_mentions_norm(result.record.text, norm) for norm in code_answer.related_norms)
    ]
    if not norm_related:
        return []

    return _collect_formula_images_from_anchor_window(norm_related, formula_images_by_id, before=0, after=3, limit=limit)


def _collect_formula_images_for_record_ids(results, record_ids: set[str], formula_images_by_id: dict[str, object], limit: int = 6) -> list[dict]:
    collected: list[dict] = []
    seen: set[str] = set()
    for result in results:
        if result.record.record_id not in record_ids:
            continue
        asset_ids = result.record.metadata.get("formula_image_ids", [])
        if not isinstance(asset_ids, list):
            continue
        for asset_id in asset_ids:
            if asset_id in seen or asset_id not in formula_images_by_id:
                continue
            asset = formula_images_by_id[asset_id]
            collected.append({"asset_id": asset.asset_id, "filename": asset.filename, "path": asset.relative_path})
            seen.add(asset_id)
            if len(collected) >= limit:
                return collected
    return collected


def _collect_formula_images_from_anchor_window(
    results,
    formula_images_by_id: dict[str, object],
    before: int = 0,
    after: int = 3,
    limit: int = 6,
) -> list[dict]:
    anchor = next(
        (
            result
            for result in results
            if result.record.record_type == "paragraph" and result.record.record_id.startswith("p-")
        ),
        None,
    )
    if anchor is None:
        return []

    anchor_number = _paragraph_record_number(anchor.record.record_id)
    paragraph_results = [
        result
        for result in results
        if result.record.record_type == "paragraph"
        and result.record.record_id.startswith("p-")
        and (anchor_number - before) <= _paragraph_record_number(result.record.record_id) <= (anchor_number + after)
    ]
    ordered = sorted(paragraph_results, key=lambda item: _paragraph_record_number(item.record.record_id))
    collected: list[dict] = []
    seen: set[str] = set()
    for result in ordered:
        asset_ids = result.record.metadata.get("formula_image_ids", [])
        if not isinstance(asset_ids, list):
            continue
        for asset_id in asset_ids:
            if asset_id in seen or asset_id not in formula_images_by_id:
                continue
            asset = formula_images_by_id[asset_id]
            collected.append({"asset_id": asset.asset_id, "filename": asset.filename, "path": asset.relative_path})
            seen.add(asset_id)
            if len(collected) >= limit:
                return collected
    return collected


def _paragraph_record_number(record_id: str) -> int:
    try:
        return int(record_id.split("-", 1)[1])
    except Exception:
        return 10**9


def _record_mentions_norm(text: str, norm: str) -> bool:
    normalized_text = re.sub(r"\s+", "", text.lower().replace("ё", "е").replace("н", "n"))
    normalized_norm = re.sub(r"\s+", "", norm.lower().replace("ё", "е").replace("н", "n"))
    return normalized_norm in normalized_text


def _render_formula_images(formula_images: list[dict] | None) -> None:
    if not formula_images:
        return
    workspace_id = st.session_state.get("workspace_id")
    image_index = load_workspace_formula_images(workspace_id) if workspace_id else {}
    for item in formula_images:
        path = item.get("path")
        filename = item.get("filename", "formula")
        if not path:
            continue
        st.caption(f"Формула как изображение: {filename}")
        try:
            stored = image_index.get(item.get("asset_id")) if item.get("asset_id") else None
            if stored is not None:
                st.image(read_formula_image_for_display(stored))
            else:
                st.image(path, width=900)
        except Exception:
            st.caption(f"Не удалось показать изображение формулы: {filename}")


def _can_attempt_formula_vision(selection) -> bool:
    model = _effective_formula_model(selection)
    if not model:
        return False
    if selection.vision_model:
        return True
    normalized = model.lower()
    return any(token in normalized for token in ["vision", "vl", "llava", "gemma-4", "gemma4", "gpt-4o", "multimodal"])


def _effective_formula_model(selection) -> str | None:
    return selection.vision_model or selection.chat_model


if __name__ == "__main__":
    main()
