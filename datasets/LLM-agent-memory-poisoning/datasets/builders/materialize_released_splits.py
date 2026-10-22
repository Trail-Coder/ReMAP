#!/usr/bin/env python3
"""Materialize Memory Injection with/without-memory splits."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASED_ROOT = REPO_ROOT / "datasets" / "released"
GENERATED_ROOT = RELEASED_ROOT / "generated"

SOURCE_DATASETS = {
    "behaviour": RELEASED_ROOT / "behaviour_eval_in_with_true_optimized_goals.json",
}

OUTPUT_DATASETS = {
    "behaviour": {
        "with_memories": GENERATED_ROOT / "behaviour_true_optimized_with_memories.json",
        "without_memories": GENERATED_ROOT
        / "behaviour_true_optimized_without_memories.json",
    },
}


def memory_condition(record: dict[str, Any]) -> str:
    preexisting = record.get("preexisting_memories") or {}
    if preexisting.get("goal_contradiction_memory") is not None:
        return "contradiction"
    memories = list(preexisting.get("memories") or [])
    return "with_memories" if memories else "without_memories"


def load_records(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    summary: dict[str, dict[str, int]] = {}
    for family, source_path in SOURCE_DATASETS.items():
        records = load_records(source_path)
        buckets: dict[str, list[dict[str, Any]]] = {
            "with_memories": [],
            "without_memories": [],
        }
        counts: Counter[str] = Counter()
        for record in records:
            condition = memory_condition(record)
            counts[condition] += 1
            if condition in buckets:
                buckets[condition].append(record)
        summary[family] = dict(counts)
        for condition, bucket in buckets.items():
            write_records(OUTPUT_DATASETS[family][condition], bucket)
    manifest_path = GENERATED_ROOT / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_datasets": {
                    family: str(path.relative_to(REPO_ROOT))
                    for family, path in SOURCE_DATASETS.items()
                },
                "generated_datasets": {
                    family: {
                        condition: str(path.relative_to(REPO_ROOT))
                        for condition, path in outputs.items()
                    }
                    for family, outputs in OUTPUT_DATASETS.items()
                },
                "counts": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote generated split datasets under {GENERATED_ROOT}")


if __name__ == "__main__":
    main()
