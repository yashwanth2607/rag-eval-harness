"""Top-k retrieval, with an optional rerank stage."""

from pymilvus import Collection

from .config import RetrievalConfig
from .embed_store import embed_texts


def retrieve(
    col: Collection, query: str, embed_model: str, cfg: RetrievalConfig
) -> list[dict]:
    """Return top-k chunks as [{"text", "source", "score"}]."""
    q_emb = embed_texts([query], embed_model)[0]
    results = col.search(
        [q_emb],
        "embedding",
        {"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=cfg.top_k,
        output_fields=["text", "source"],
    )[0]

    hits = [
        {"text": h.entity.get("text"), "source": h.entity.get("source"), "score": h.score}
        for h in results
    ]
    if cfg.rerank:
        hits = _rerank(query, hits)  # TODO: cross-encoder reranker
    return hits


def _rerank(query: str, hits: list[dict]) -> list[dict]:
    raise NotImplementedError("reranking is on the roadmap")
