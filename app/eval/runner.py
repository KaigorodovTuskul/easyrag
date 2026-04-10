from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.retrieval.exact import search_exact
from app.storage.index import load_index_records


EVAL_PATH = Path("eval") / "cases.json"


@dataclass(slots=True)
class EvalCaseResult:
    query: str
    expected_record_id: str
    hit_at_1: bool
    hit_at_3: bool
    top_record_id: str | None


def load_eval_cases(path: Path = EVAL_PATH) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def run_eval(path: Path = EVAL_PATH) -> list[EvalCaseResult]:
    records = load_index_records()
    results: list[EvalCaseResult] = []

    for case in load_eval_cases(path):
        query = case["query"]
        expected = case["expected_record_id"]
        found = search_exact(records, query, limit=3)
        top_record_id = found[0].record.record_id if found else None
        top3 = [result.record.record_id for result in found]
        results.append(
            EvalCaseResult(
                query=query,
                expected_record_id=expected,
                hit_at_1=top_record_id == expected,
                hit_at_3=expected in top3,
                top_record_id=top_record_id,
            )
        )

    return results
