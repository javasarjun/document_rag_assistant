from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import settings
from document_loader import SUPPORTED_EXTENSIONS
from rag_pipeline import DocumentRAGPipeline

st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource
def get_pipeline() -> DocumentRAGPipeline:
    return DocumentRAGPipeline()


pipeline = get_pipeline()
settings.upload_dir.mkdir(parents=True, exist_ok=True)

st.title("📄 Document-Based RAG Assistant")
st.caption(
    "Section 5 Lab: load documents, create embeddings, retrieve relevant chunks, and answer with grounded context."
)

with st.sidebar:
    st.header("Lab Settings")
    st.write(f"**Provider:** `{settings.llm_provider}`")

    if settings.llm_provider == "ollama":
        st.write(f"**Chat model:** `{settings.ollama_chat_model}`")
        st.write(f"**Embedding model:** `{settings.ollama_embedding_model}`")
        st.info(
            "Default mode uses local Ollama. If a model is missing, pull it from your terminal first."
        )
    else:
        st.write(f"**Chat model:** `{settings.openai_chat_model}`")
        st.write(f"**Embedding model:** `{settings.openai_embedding_model}`")
        st.info("OpenAI mode is enabled from your .env file.")

    st.divider()
    st.write(f"**Chunks indexed:** {pipeline.vector_store.count()}")
    sources = pipeline.vector_store.sources()
    if sources:
        st.write("**Sources:**")
        for source in sources:
            st.caption(Path(source).name)

    if st.button("Clear local vector store"):
        pipeline.clear_index()
        st.cache_resource.clear()
        st.success("Vector store cleared. Refreshing app...")
        st.rerun()

left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.subheader("1. Ingest documents")

    st.markdown(
        "Use the sample docs first, then upload your own `.txt`, `.md`, or `.pdf` files."
    )

    if st.button("Ingest sample_docs folder"):
        with st.spinner("Chunking documents and creating embeddings..."):
            try:
                results = pipeline.ingest_folder("sample_docs")
                for result in results:
                    badge = "trusted" if result.trusted else "untrusted"
                    st.success(
                        f"Indexed {result.chunks_added} chunks from "
                        f"{Path(result.source).name} ({badge} source)"
                    )
                st.rerun()
            except Exception as exc:  # noqa: BLE001 - beginner lab UI should show clear errors
                st.error(str(exc))

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Ingest uploaded files"):
        with st.spinner("Saving files, chunking text, and creating embeddings..."):
            for uploaded_file in uploaded_files:
                destination = settings.upload_dir / uploaded_file.name
                destination.write_bytes(uploaded_file.getbuffer())
                try:
                    result = pipeline.ingest_file(destination)
                    badge = "trusted" if result.trusted else "untrusted (upload)"
                    st.success(
                        f"Indexed {result.chunks_added} chunks from "
                        f"{uploaded_file.name} — {badge}"
                    )
                    if not result.trusted:
                        st.warning(
                            f"{uploaded_file.name} is treated as an UNTRUSTED source. "
                            "It will not be used to answer sensitive questions "
                            "(refunds, policy, legal, finance, etc.)."
                        )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not ingest {uploaded_file.name}: {exc}")
            st.rerun()

with right:
    st.subheader("2. Ask questions")

    default_question = "What is retrieval-augmented generation?"
    question = st.text_input("Question", value=default_question)
    top_k = st.slider("Number of retrieved chunks", min_value=1, max_value=8, value=settings.top_k)

    if st.button("Ask RAG assistant", type="primary"):
        with st.spinner("Embedding question, retrieving chunks, and generating answer..."):
            try:
                rag_answer = pipeline.ask(question, top_k=top_k)
                st.markdown("### Answer")
                if rag_answer.blocked:
                    st.error(rag_answer.answer)
                else:
                    st.write(rag_answer.answer)

                if rag_answer.sensitive:
                    st.info(
                        "This was treated as a **sensitive question**, so only "
                        "trusted sources are allowed to answer it."
                    )

                if rag_answer.quarantined:
                    st.markdown("### 🛡️ Quarantined chunks (blocked before generation)")
                    for decision in rag_answer.quarantined:
                        record = decision.result.record
                        st.warning(
                            f"**{Path(record.source).name}** (chunk {record.chunk_index}, "
                            f"{decision.trust}, injection score {decision.injection_score}) — "
                            f"{decision.reason}"
                        )

                st.markdown("### Retrieved context used for the answer")
                if not rag_answer.results or (rag_answer.blocked):
                    st.caption("No context was allowed into the model for this answer.")
                for index, result in enumerate(rag_answer.results, start=1):
                    record = result.record
                    trust_badge = "✅ trusted" if getattr(record, "trusted", False) else "⚠️ untrusted"
                    with st.expander(
                        f"Source {index}: {Path(record.source).name} | Chunk {record.chunk_index} | "
                        f"Score {result.score:.4f} | {trust_badge}"
                    ):
                        st.write(record.text)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

st.divider()
st.markdown(
    "**Teaching boundary:** This lab builds the RAG app. Section 6 will attack and secure it with malicious PDFs, context isolation, sanitization, and trusted source validation."
)
