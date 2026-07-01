"""Trusted source validation for the RAG ingestion pipeline.

Section 6 defense #1. A RAG assistant trusts its documents implicitly: whatever
gets ingested becomes retrievable context the model will treat as truth. So the
first line of defense is controlling *what is allowed in*.

Before a document is loaded, chunked, or embedded, ``validate_source`` checks
that it:

1. exists and is a regular file,
2. has an allowed file type,
3. lives under a trusted directory (no path-traversal into the rest of disk),
4. is within the maximum allowed size.

Anything that fails is rejected with :class:`UntrustedSourceError` before it can
reach the index.
"""

from __future__ import annotations

from pathlib import Path

from config import settings
from document_loader import SUPPORTED_EXTENSIONS


class UntrustedSourceError(ValueError):
    """Raised when a document fails trusted source validation."""


def _trusted_roots() -> list[Path]:
    """Resolve the configured trusted directories to absolute paths."""
    return [Path(root).resolve() for root in settings.trusted_source_dirs]


def _is_within(path: Path, root: Path) -> bool:
    """Return True if ``path`` is located inside ``root``."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_source(file_path: str | Path) -> Path:
    """Validate that a document comes from a trusted source.

    Returns the resolved path on success; raises ``UntrustedSourceError`` on the
    first failed control so the caller never ingests an untrusted document.
    """
    path = Path(file_path).resolve()

    # 1. Must be a real, existing file.
    if not path.is_file():
        raise UntrustedSourceError(f"Not a readable file: {path}")

    # 2. File type must be on the allowlist.
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UntrustedSourceError(
            f"Blocked file type '{path.suffix}'. Allowed types: {allowed}."
        )

    # 3. Location must be inside a trusted directory (blocks path traversal).
    roots = _trusted_roots()
    if not any(_is_within(path, root) for root in roots):
        allowed = ", ".join(str(root) for root in roots)
        raise UntrustedSourceError(
            f"Untrusted source location: {path}. Allowed directories: {allowed}."
        )

    # 4. Size must be within the configured cap.
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > settings.max_document_mb:
        raise UntrustedSourceError(
            f"Document is {size_mb:.1f} MB, exceeding the "
            f"{settings.max_document_mb:.1f} MB limit."
        )

    return path
