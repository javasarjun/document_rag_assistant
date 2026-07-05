from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from config import settings
from content_sanitizer import sanitize_content
from document_loader import iter_supported_files, load_document
from llm_service import LLMService
from security_policy import (
    SAFE_FALLBACK,
    ChunkDecision,
    apply_security_policy,
    classify_trust,
)
from source_validator import validate_source
from text_splitter import split_text
from vector_store import LocalVectorStore, SearchResult, VectorRecord


@dataclass
class IngestionResult:
    source: str
    chunks_added: int
    trusted: bool = False


@dataclass
class RagAnswer:
    answer: str
    results: list[SearchResult]
    quarantined: list[ChunkDecision] = field(default_factory=list)
    sensitive: bool = False
    blocked: bool = False


class DocumentRAGPipeline:
    """End-to-end RAG pipeline: load, chunk, embed, retrieve, answer."""

    def __init__(self) -> None:
        self.llm = LLMService()
        self.vector_store = LocalVectorStore(settings.vector_store_path)

    def ingest_file(self, file_path: str | Path) -> IngestionResult:
        # Section 6 defense #1: reject untrusted sources before loading.
        path = validate_source(file_path)
        # Section 6 defense #4: record whether this source is a trusted
        # authority. Uploads are ingestible but default to untrusted.
        is_trusted = classify_trust(path) == "trusted"
        text = load_document(path)
        chunks = split_text(
            text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        records: list[VectorRecord] = []
        for index, chunk in enumerate(chunks):
            embedding = self.llm.embed_text(chunk)
            record = VectorRecord(
                id=f"{path.name}:{index}",
                source=str(path),
                chunk_index=index,
                text=chunk,
                embedding=embedding,
                trusted=is_trusted,
            )
            records.append(record)

        added = self.vector_store.add_records(records)
        return IngestionResult(source=str(path), chunks_added=added, trusted=is_trusted)

    def ingest_folder(self, folder_path: str | Path) -> list[IngestionResult]:
        results: list[IngestionResult] = []
        for file_path in iter_supported_files(folder_path):
            results.append(self.ingest_file(file_path))
        return results

    def ask(self, question: str, top_k: int | None = None) -> RagAnswer:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if self.vector_store.count() == 0:
            raise RuntimeError("No documents have been ingested yet.")

        query_embedding = self.llm.embed_text(question)
        results = self.vector_store.search(query_embedding, top_k=top_k or settings.top_k)

        # Section 6 defense #4: enforce the security policy BEFORE generation.
        # Chunks that contain prompt injection, or that come from an untrusted
        # source for a sensitive question, are quarantined and never reach the
        # model.
        outcome = apply_security_policy(results, question)

        if not outcome.has_context:
            # No usable context survived enforcement -> safe fallback, no model
            # call. We refuse rather than repeat unverified/injected content.
            return RagAnswer(
                answer=SAFE_FALLBACK,
                results=[decision.result for decision in outcome.decisions],
                quarantined=outcome.quarantined,
                sensitive=outcome.sensitive,
                blocked=True,
            )

        context = build_context(outcome.allowed)
        answer = self.llm.generate_answer(question=question, context=context)
        return RagAnswer(
            answer=answer,
            results=outcome.allowed,
            quarantined=outcome.quarantined,
            sensitive=outcome.sensitive,
            blocked=False,
        )

    def clear_index(self) -> None:
        self.vector_store.clear()


def build_context(results: list[SearchResult]) -> str:
    context_blocks: list[str] = []
    for number, result in enumerate(results, start=1):
        record = result.record
        context_blocks.append(
            "\n".join(
                [
                    f"[Source {number}]",
                    f"File: {record.source}",
                    f"Chunk: {record.chunk_index}",
                    f"Similarity: {result.score:.4f}",
                    "Text:",
                    sanitize_content(record.text),
                ]
            )
        )

    return "\n\n".join(context_blocks)
