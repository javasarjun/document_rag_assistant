# Lab Guide: Build a Document-Based RAG Assistant from Scratch

## Course placement

**Section 5: Building RAG Applications**

This lab comes after these theory lectures:

1. What is Retrieval-Augmented Generation (RAG)?
2. How RAG Applications Work
3. Understanding Embeddings and Vector Search

This lab should feel like a new build from scratch, not a continuation of the earlier prompt-injection chatbot.

---

## Lab outcome

By the end of this lab, learners will have a working document-based RAG assistant that can:

- read `.txt`, `.md`, and `.pdf` files
- split documents into chunks
- create embeddings
- store embeddings locally
- retrieve relevant chunks for a question
- generate an answer grounded in retrieved context

---

## Instructor framing

Say this before starting:

> So far, our assistant mostly answered from the model's own knowledge and prompt. Now we are building a new kind of AI application: a document-based assistant. Instead of asking the model to remember everything, we will retrieve the right document chunks first and then ask the model to answer using that context.

Also make the Section 5 versus Section 6 boundary clear:

> In this section, we are building RAG. In the next section, we will attack and secure RAG.

---

## Step 1: Show the app structure

Files to explain:

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface |
| `document_loader.py` | Loads text, markdown, and PDF files |
| `text_splitter.py` | Splits long documents into smaller chunks |
| `llm_service.py` | Calls Ollama or OpenAI for embeddings and answers |
| `vector_store.py` | Saves and searches embeddings in local JSON |
| `rag_pipeline.py` | Connects loading, chunking, embedding, retrieval, and answering |

---

## Step 2: Configure the provider

Default provider is Ollama:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.2:3b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

To switch to OpenAI:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Teaching point:

> The rest of the RAG pipeline does not care whether embeddings come from Ollama or OpenAI. That provider detail is hidden inside `llm_service.py`.

---

## Step 3: Load documents

Open `document_loader.py` and explain:

- `.txt` and `.md` files are read directly.
- `.pdf` files are processed page by page.
- Unsupported file types are rejected.

Teaching point:

> RAG starts before the model. If the document loader is bad, retrieval will be bad, and the final answer will also be bad.

---

## Step 4: Split documents into chunks

Open `text_splitter.py` and explain:

- Long documents are too large to send directly into every prompt.
- Chunks make search more focused.
- Overlap prevents important context from being cut off between chunks.

Teaching point:

> Chunking is one of the most important design decisions in RAG. Too large, and retrieval becomes noisy. Too small, and answers may miss context.

---

## Step 5: Create embeddings

Open `llm_service.py` and explain:

- The app creates embeddings for document chunks.
- It also creates an embedding for the user's question.
- Similar meanings should produce similar vectors.

Teaching point:

> An embedding is not the answer. It is the searchable meaning representation that helps the system find the right chunks.

---

## Step 6: Store and search vectors

Open `vector_store.py` and explain:

- The lab stores records in `data/vector_store.json`.
- Each record has source, chunk text, and embedding.
- Retrieval uses cosine similarity.

Teaching point:

> A vector database is basically helping us answer this question: which stored chunks are closest in meaning to the user's question?

---

## Step 7: Ask a grounded question

Run:

```bash
streamlit run app.py
```

In the UI:

1. Click **Ingest sample_docs folder**.
2. Ask: `What is retrieval-augmented generation?`
3. Expand the retrieved context.
4. Show which chunk was used.

Teaching point:

> The model is not answering alone. It is answering with retrieved document context.

---

## Step 8: Try retrieval behavior

Good questions:

```text
What are the main steps in a RAG pipeline?
```

```text
How is vector search different from keyword search?
```

```text
Why does AI security need trust boundaries?
```

Vague question:

```text
Tell me about this.
```

Missing-information question:

```text
What is the refund policy for this course?
```

Teaching point:

> Good retrieval depends on good documents, good chunks, good embeddings, and good questions.

---

## Section 6 setup

End the lab with this transition:

> This app works, but it trusts the documents too much. In the next section, we will place malicious instructions inside documents and see how retrieved context can attack the assistant.

Do not implement the Section 6 defenses in this lab.
