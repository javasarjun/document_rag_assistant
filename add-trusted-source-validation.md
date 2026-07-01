feature: add-trusted-source-validation
goal: In this lesson you'll add a trusted source validation layer that controls which documents are allowed into the RAG index.
subtitle: Control what gets in before it ever becomes retrievable context
conclusion: And that's trusted source validation - the first Section 6 defense. The assistant now rejects documents from untrusted locations, unknown file types, and oversized files before they reach the index. Next we'll add context isolation to separate instructions from retrieved content.
sequence: source_validator.py -> config.py -> rag_pipeline.py
target_learner: Intermediate

## new: source_validator.py

A brand-new module that checks every document against four controls (exists, allowed type, trusted location, size cap) before it can be ingested. Anything that fails is rejected with `UntrustedSourceError`.

## modified: config.py

Add settings that define the trusted directories and the maximum document size, plus validation for them.

```python
    # Section 6 defense #1 - Trusted source validation.
    # Only documents that live under a trusted directory are allowed into the
    # RAG index, and only up to a maximum file size.
    trusted_source_dirs: tuple[str, ...] = tuple(
        part.strip()
        for part in os.getenv("TRUSTED_SOURCE_DIRS", "sample_docs,data/uploads").split(",")
        if part.strip()
    )
    max_document_mb: float = float(os.getenv("MAX_DOCUMENT_MB", "10"))
```

```python
        if not self.trusted_source_dirs:
            raise ValueError("At least one TRUSTED_SOURCE_DIRS entry is required.")

        if self.max_document_mb <= 0:
            raise ValueError("MAX_DOCUMENT_MB must be greater than 0.")
```

## modified: rag_pipeline.py

Import the validator and call it at the very top of `ingest_file`, so an untrusted source is rejected before the document is loaded, chunked, or embedded.

```python
from source_validator import validate_source
```

```python
    def ingest_file(self, file_path: str | Path) -> IngestionResult:
        # Section 6 defense #1: reject untrusted sources before loading.
        path = validate_source(file_path)
        text = load_document(path)
```
