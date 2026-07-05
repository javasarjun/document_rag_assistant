from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "4"))

    vector_store_path: Path = Path(os.getenv("VECTOR_STORE_PATH", "data/vector_store.json"))
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", "data/uploads"))

    # Section 6 defense #1 - Trusted source validation (ingestion gate).
    # Documents may only be *ingested* from these directories (blocks path
    # traversal / arbitrary disk reads), up to a maximum file size. Being
    # ingestible does NOT make a document authoritative - see below.
    trusted_source_dirs: tuple[str, ...] = tuple(
        part.strip()
        for part in os.getenv("TRUSTED_SOURCE_DIRS", "sample_docs,data/uploads").split(",")
        if part.strip()
    )
    max_document_mb: float = float(os.getenv("MAX_DOCUMENT_MB", "10"))

    # Section 6 defense #4 - Enforcement layer.
    # Trust authority: only documents under these directories are treated as
    # trusted/approved sources. Everything else (notably user uploads) defaults
    # to UNTRUSTED and cannot be used to answer sensitive questions. Keep this a
    # strict subset of the ingest dirs so uploads can be indexed but not trusted.
    trusted_authority_dirs: tuple[str, ...] = tuple(
        part.strip()
        for part in os.getenv("TRUSTED_AUTHORITY_DIRS", "sample_docs").split(",")
        if part.strip()
    )
    # A retrieved chunk scoring at/above this many injection points is
    # quarantined and never sent to the model.
    injection_block_threshold: int = int(os.getenv("INJECTION_BLOCK_THRESHOLD", "5"))

    def validate(self) -> None:
        if self.llm_provider not in {"ollama", "openai"}:
            raise ValueError("LLM_PROVIDER must be either 'ollama' or 'openai'.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")

        if not self.trusted_source_dirs:
            raise ValueError("At least one TRUSTED_SOURCE_DIRS entry is required.")

        if not self.trusted_authority_dirs:
            raise ValueError("At least one TRUSTED_AUTHORITY_DIRS entry is required.")

        if self.injection_block_threshold <= 0:
            raise ValueError("INJECTION_BLOCK_THRESHOLD must be greater than 0.")

        if self.max_document_mb <= 0:
            raise ValueError("MAX_DOCUMENT_MB must be greater than 0.")


settings = Settings()
settings.validate()
