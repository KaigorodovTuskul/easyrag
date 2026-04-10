from __future__ import annotations

import json

from app.core.config import AppConfig
from app.providers.router import ProviderRouter


def main() -> None:
    config = AppConfig.load()
    provider, selection = ProviderRouter(config).resolve()

    payload = {
        "provider": provider.name,
        "reachable": selection.reachable,
        "reason": selection.reason,
        "chat_model": selection.chat_model,
        "embed_model": selection.embed_model,
        "active_model": selection.active_model,
        "available_models": [model.name for model in selection.available_models],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
