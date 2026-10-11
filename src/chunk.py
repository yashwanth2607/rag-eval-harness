"""Chunking strategies. Chunking is the cheapest lever on retrieval quality."""

from .config import ChunkConfig


def chunk_text(text: str, cfg: ChunkConfig) -> list[str]:
    if cfg.strategy == "fixed":
        return _fixed(text, cfg.chunk_size, cfg.chunk_overlap)
    if cfg.strategy == "recursive":
        return _recursive(text, cfg.chunk_size, cfg.chunk_overlap)
    if cfg.strategy == "semantic":
        return _semantic(text, cfg)  # TODO: implement embedding-based boundaries
    raise ValueError(f"unknown chunking strategy: {cfg.strategy}")


def _fixed(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks, step = [], max(size - overlap, 1)
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i : i + size]))
    return chunks


def _recursive(text: str, size: int, overlap: int) -> list[str]:
    """Split on paragraph, then sentence, then word boundaries."""
    import re

    separators = ["\n\n", "\n", ". ", " "]
    chunks: list[str] = []

    def _split(piece: str, seps: list[str]) -> list[str]:
        if len(piece.split()) <= size or not seps:
            return [piece]
        sep = seps[0]
        parts, current = [], ""
        for part in piece.split(sep):
            candidate = current + sep + part if current else part
            if len(candidate.split()) > size and current:
                parts.append(current)
                current = part
            else:
                current = candidate
        if current:
            parts.append(current)
        out = []
        for p in parts:
            out.extend(_split(p, seps[1:]) if len(p.split()) > size else [p])
        return out

    for c in _split(text, separators):
        chunks.append(c.strip())
    # apply overlap
    if overlap and chunks:
        overlapped = [chunks[0]]
        for c in chunks[1:]:
            prev_words = overlapped[-1].split()
            glue = " ".join(prev_words[-overlap:]) if len(prev_words) > overlap else ""
            overlapped.append((glue + " " + c).strip())
        return overlapped
    _ = re  # reserved for sentence-aware splitting
    return chunks


def _semantic(text: str, cfg: ChunkConfig) -> list[str]:
    raise NotImplementedError("semantic chunking is on the roadmap")
