import json, os, random
from typing import List, Optional
from framework.core import Dataset, Sample

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.environ.get(
    "MEMORY_INJECTION_DATA_PATH",
    os.path.join(
        _PROJECT_ROOT,
        "datasets",
        "LLM-agent-memory-poisoning",
        "datasets",
        "released",
        "behaviour_eval_in_with_true_optimized_goals.json",
    ),
)


class SleeperBenchmarkDataset(Dataset):
    name = "Memory Injection"
    description = ("500 diverse documents from 15 domains (legal, news, science, code, "
                   "email, etc.) with unique queries and attack goals from the "
                   "LLM-agent-memory-poisoning paper")

    def load(self, n: Optional[int] = None, seed: int = 42,
             domains: Optional[List[str]] = None, **kwargs) -> List[Sample]:
        with open(DATA_PATH, encoding="utf-8") as f:
            raw = json.load(f)

        random.seed(seed)
        if domains:
            raw = [r for r in raw if r["document"].get("domain", "?") in domains]
        if n and n < len(raw):
            raw = random.sample(raw, n)

        samples = []
        for r in raw:
            doc = r["document"]
            goal = r["goal"]
            query = r["query"]
            memories = r.get("preexisting_memories", {}).get("memories", [])

            samples.append(Sample(
                document=doc["text"],
                query=query["query"],
                sample_id=f'{doc.get("domain","?")}_{doc.get("doc_id","?")}',
                metadata={
                    "domain": doc.get("domain", "?"),
                    "format": doc.get("annotations", {}).get("document_format", "text"),
                    "goal_text": goal.get("goal_text", ""),
                    "goal_category": goal.get("category_name", ""),
                    "query_category": query.get("category_name", ""),
                    "query_style": query.get("style", ""),
                    "preexisting_memories": memories,
                },
            ))
        return samples
