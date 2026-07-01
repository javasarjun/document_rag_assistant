import re


def normalize_text(text: str) -> str:
    """Clean repeated whitespace while preserving paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> list[str]:
    """Split text into overlapping character chunks.

    This beginner-friendly splitter intentionally uses characters instead of
    tokenizer-specific logic. In later production lessons, this can evolve into
    token-aware, section-aware, or AST-aware chunking.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    cleaned = normalize_text(text)
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        candidate = cleaned[start:end]

        # Prefer ending near a paragraph or sentence boundary when possible.
        if end < len(cleaned):
            boundary = max(candidate.rfind("\n\n"), candidate.rfind(". "), candidate.rfind("? "), candidate.rfind("! "))
            if boundary > chunk_size * 0.55:
                end = start + boundary + 1
                candidate = cleaned[start:end]

        chunk = candidate.strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(cleaned):
            break

        start = max(0, end - chunk_overlap)

    return chunks
