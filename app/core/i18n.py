from __future__ import annotations

Language = str

SUPPORTED_LANGUAGES = ("ru", "en")

TRANSLATIONS: dict[str, dict[str, str]] = {
    "app.caption": {
        "ru": "Чат по выбранному DOCX-документу",
        "en": "Chat with the selected DOCX document",
    },
    "language.header": {"ru": "Язык", "en": "Language"},
    "language.label": {"ru": "Интерфейс", "en": "Interface"},
    "model.header": {"ru": "Модель", "en": "Model"},
    "provider.label": {"ru": "Провайдер", "en": "Provider"},
    "chat_model.label": {"ru": "Модель ответа", "en": "Answer model"},
    "embed_model.label": {"ru": "Модель embeddings", "en": "Embedding model"},
    "provider_status.title": {"ru": "Статус провайдера", "en": "Provider status"},
    "ollama_unavailable": {"ru": "Ollama недоступна: {error}", "en": "Ollama is unavailable: {error}"},
    "document.header": {"ru": "Документ", "en": "Document"},
    "workspace.select": {"ru": "Готовая база", "en": "Saved workspace"},
    "workspace.empty": {"ru": "Готовых баз пока нет.", "en": "No saved workspaces yet."},
    "upload.label": {"ru": "Загрузить DOCX", "en": "Upload DOCX"},
    "clear_chat": {"ru": "Очистить диалог", "en": "Clear chat"},
    "no_workspace": {
        "ru": "Загрузите DOCX или выберите готовую базу в левой панели.",
        "en": "Upload a DOCX file or select a saved workspace in the sidebar.",
    },
    "chat.placeholder": {"ru": "Напишите вопрос по документу", "en": "Ask a question about the document"},
    "no_results": {
        "ru": "Не нашел подходящий фрагмент в выбранном документе. Попробуйте точный код, номер счета или другую формулировку.",
        "en": "No relevant fragment was found in the selected document. Try an exact code, account number, or another wording.",
    },
    "model_unavailable": {
        "ru": "Нашел контекст, но модель ответа недоступна: {error}",
        "en": "Context was found, but the answer model is unavailable: {error}",
    },
    "answer_failed": {
        "ru": "Нашел контекст, но не смог сформировать ответ: {error}",
        "en": "Context was found, but the answer could not be generated: {error}",
    },
    "empty_answer": {"ru": "Модель вернула пустой ответ.", "en": "The model returned an empty answer."},
    "clarification.prefix": {"ru": "Уточнение", "en": "Clarification"},
    "docx_deps_missing": {
        "ru": "Не установлены зависимости для DOCX. Запустите bootstrap_env.bat.",
        "en": "DOCX dependencies are not installed. Run bootstrap_env.bat.",
    },
    "workspace_cached": {"ru": "База уже готова. Открываю из кэша.", "en": "Workspace is ready. Opening from cache."},
    "workspace_preparing": {
        "ru": "Готовлю базу. Это нужно сделать один раз для документа.",
        "en": "Preparing workspace. This is done once per document.",
    },
    "indexing.progress": {"ru": "Индексация документа...", "en": "Indexing document..."},
    "indexing.done": {
        "ru": "Индексация готова: {records} фрагментов, ~{tokens} токенов",
        "en": "Indexing complete: {records} fragments, ~{tokens} tokens",
    },
    "embeddings.cached": {
        "ru": "Embeddings уже есть. Повторно не считаю.",
        "en": "Embeddings already exist. Skipping rebuild.",
    },
    "embeddings.progress": {
        "ru": "Embeddings: {index}/{total}, ~{processed}/{tokens} токенов",
        "en": "Embeddings: {index}/{total}, ~{processed}/{tokens} tokens",
    },
    "embeddings.failed": {
        "ru": "Exact-поиск готов, но embeddings не построены: {error}",
        "en": "Exact search is ready, but embeddings were not built: {error}",
    },
    "embeddings.done": {
        "ru": "Embeddings готовы: {records} фрагментов, примерно {tokens} токенов.",
        "en": "Embeddings ready: {records} fragments, about {tokens} tokens.",
    },
    "workspace.current": {"ru": "Текущая база: {source}", "en": "Current workspace: {source}"},
    "workspace.stats": {
        "ru": "Фрагментов: {records}; embeddings: {embeddings}",
        "en": "Fragments: {records}; embeddings: {embeddings}",
    },
    "debug.title": {"ru": "Отладка", "en": "Debug"},
    "debug.workspace": {"ru": "Текущая база", "en": "Current workspace"},
    "debug.models": {"ru": "Модели", "en": "Models"},
    "debug.last_query": {"ru": "Последний запрос", "en": "Last query"},
    "debug.run_eval": {"ru": "Запустить проверку качества", "en": "Run quality check"},
    "debug.eval_failed": {"ru": "Проверка не запустилась: {error}", "en": "Quality check failed: {error}"},
    "debug.cases": {"ru": "кейсов", "en": "cases"},
    "workspace.embeddings_ready": {"ru": "embeddings есть", "en": "embeddings ready"},
    "workspace.exact_only": {"ru": "только exact", "en": "exact only"},
    "workspace.fragments_short": {"ru": "фрагм.", "en": "fragments"},
}


def normalize_language(language: str | None) -> Language:
    if language in SUPPORTED_LANGUAGES:
        return language
    return "ru"


def t(key: str, language: Language, **kwargs: object) -> str:
    template = TRANSLATIONS.get(key, {}).get(normalize_language(language), key)
    return template.format(**kwargs) if kwargs else template
