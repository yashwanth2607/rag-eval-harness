"""Document loading: pdf / txt / md -> list of {text, source} records."""

from pathlib import Path


def load_documents(docs_dir: str) -> list[dict]:
    """Load every supported file under docs_dir.

    Returns a list of {"text": ..., "source": <filename>} dicts.
    """
    records: list[dict] = []
    for path in Path(docs_dir).glob("*"):
        if path.suffix == ".pdf":
            records.append({"text": _read_pdf(path), "source": path.name})
        elif path.suffix in {".txt", ".md"}:
            records.append({"text": path.read_text(encoding="utf-8"), "source": path.name})
    return [r for r in records if r["text"].strip()]


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)
