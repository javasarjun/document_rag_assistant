"""End-to-end enforcement test using a stubbed LLM (no Ollama/OpenAI needed).

Replays the real attack documents through the full pipeline and asserts the
security policy quarantines them before any answer is generated.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rag_pipeline import DocumentRAGPipeline
from security_policy import SAFE_FALLBACK
from vector_store import LocalVectorStore

REPO = Path(__file__).resolve().parents[1]


def _fake_embed(text: str) -> list[float]:
    """Deterministic tiny embedding so retrieval works without a model."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return [b / 255.0 for b in h[:16]]


@pytest.fixture()
def pipeline(tmp_path):
    pipe = DocumentRAGPipeline()
    # Use a throwaway vector store so we never touch real data.
    pipe.vector_store = LocalVectorStore(tmp_path / "vs.json")
    # Stub the LLM: deterministic embeddings, and a generator that would happily
    # echo injected content (to prove the *policy*, not the model, is blocking).
    pipe.llm.embed_text = _fake_embed  # type: ignore[assignment]
    pipe.llm.generate_answer = lambda question, context: f"ANSWERED_FROM_CONTEXT: {context[:60]}"  # type: ignore[assignment]
    return pipe


def _ingest(pipe, relpath):
    return pipe.ingest_file(REPO / relpath)


def test_malicious_pdf_is_quarantined_and_falls_back(pipeline):
    _ingest(pipeline, "data/uploads/malicious_policy_notice.pdf")
    ans = pipeline.ask("What is the refund policy?", top_k=8)

    assert ans.blocked is True
    assert ans.answer == SAFE_FALLBACK
    assert "unlimited lifetime refunds" not in ans.answer.lower()
    assert ans.quarantined
    assert any("injection" in d.reason.lower() for d in ans.quarantined)


def test_poisoned_pdf_untrusted_for_sensitive_question(pipeline):
    _ingest(pipeline, "data/uploads/poisoned_refund_policy.pdf")
    ans = pipeline.ask("What is the enterprise refund policy?", top_k=8)

    assert ans.blocked is True
    assert ans.sensitive is True
    assert "365 days" not in ans.answer
    assert any("untrusted" in d.reason.lower() for d in ans.quarantined)


def test_trusted_sample_docs_answer_normally(pipeline):
    _ingest(pipeline, "sample_docs/rag_notes.txt")
    _ingest(pipeline, "sample_docs/ai_security_overview.txt")
    ans = pipeline.ask("What is retrieval-augmented generation?", top_k=8)

    assert ans.blocked is False
    assert ans.answer.startswith("ANSWERED_FROM_CONTEXT")
    assert ans.results
