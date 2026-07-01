# Document-Based RAG Assistant

Section 5 lab for the **AI Security Masterclass**.

This lab builds a fresh RAG application from scratch. It is intentionally separate from the earlier vulnerable chat assistant and prompt-injection labs.

## What this app does

The app lets learners:

1. Load text, markdown, and PDF documents.
2. Split documents into smaller chunks.
3. Convert chunks into embeddings.
4. Store embeddings in a simple local JSON vector store.
5. Search for the chunks most similar to a user's question.
6. Ask an LLM to answer using only retrieved context.

## Why the vector store is simple

This first RAG lab uses `data/vector_store.json` instead of a production vector database. That makes the retrieval flow easier to inspect and teach.

In later lessons, this can evolve into Chroma, FAISS, pgvector, Pinecone, Weaviate, OpenSearch, or another production-ready vector store.

## Setup

```bash
cd "/Users/arjunvaid/Desktop/AI Security Master Class/document_rag_assistant"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Default: Ollama

This lab uses Ollama by default.

Pull the default models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Run the app:

```bash
streamlit run app.py
```

Then click **Ingest sample_docs folder** and ask:

```text
What is retrieval-augmented generation?
```

## Switch to OpenAI

Edit `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Then restart Streamlit.

## Project structure

```text
document_rag_assistant/
├── app.py                  # Streamlit UI
├── config.py               # Environment settings
├── document_loader.py      # TXT, Markdown, and PDF loading
├── text_splitter.py        # Chunking logic
├── llm_service.py          # Ollama/OpenAI chat + embeddings
├── vector_store.py         # Simple JSON vector store
├── rag_pipeline.py         # Load → chunk → embed → retrieve → answer
├── sample_docs/            # Starter documents for the lab
├── data/                   # Local vector store and uploads
├── tests/                  # Small tests for splitter/vector logic
└── LAB_GUIDE.md            # Instructor-friendly lab flow
```

## Teaching boundary

This lab is only for building RAG.

Do not add RAG security controls here. Keep these for Section 6:

- malicious PDF handling
- prompt injection in retrieved documents
- trusted source validation
- context isolation
- retrieved content sanitization
- malicious context testing
