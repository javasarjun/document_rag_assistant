from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


class UnsupportedDocumentTypeError(ValueError):
    pass


def load_document(path: str | Path) -> str:
    """Load text from a supported document file."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentTypeError(
            f"Unsupported file type: {suffix}. Supported types: {SUPPORTED_EXTENSIONS}"
        )

    if suffix in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        return _load_pdf(file_path)

    raise UnsupportedDocumentTypeError(f"Unsupported file type: {suffix}")


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"\n\n--- Page {page_number} ---\n{page_text.strip()}")

    return "\n".join(pages).strip()


def iter_supported_files(folder: str | Path) -> Iterable[Path]:
    """Yield supported files from a folder recursively."""
    folder_path = Path(folder)
    for file_path in sorted(folder_path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield file_path
