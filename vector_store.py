from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class VectorRecord:
    id: str
    source: str
    chunk_index: int
    text: str
    embedding: list[float]


@dataclass
class SearchResult:
    record: VectorRecord
    score: float


class LocalVectorStore:
    """Tiny JSON-backed vector store for teaching the RAG concept.

    This is intentionally simple for the first RAG build. Learners can inspect
    the stored chunks and embeddings in data/vector_store.json.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records: list[VectorRecord] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.records = []
            return

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.records = [VectorRecord(**item) for item in payload.get("records", [])]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": [asdict(record) for record in self.records]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def clear(self) -> None:
        self.records = []
        self.save()

    def add_records(self, records: Iterable[VectorRecord]) -> int:
        new_records = list(records)
        if not new_records:
            return 0

        # Replace records for the same source so re-ingesting a file stays clean.
        sources = {record.source for record in new_records}
        self.records = [record for record in self.records if record.source not in sources]
        self.records.extend(new_records)
        self.save()
        return len(new_records)

    def search(self, query_embedding: list[float], top_k: int = 4) -> list[SearchResult]:
        results: list[SearchResult] = []

        for record in self.records:
            score = cosine_similarity(query_embedding, record.embedding)
            results.append(SearchResult(record=record, score=score))

        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    def count(self) -> int:
        return len(self.records)

    def sources(self) -> list[str]:
        return sorted({record.source for record in self.records})


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)
