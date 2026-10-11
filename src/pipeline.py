"""CLI orchestration: ingest -> chunk -> embed -> Milvus."""

import argparse
import json

from .chunk import chunk_text
from .config import Config
from .embed_store import connect, embed_texts, get_or_create_collection, upsert_chunks
from .ingest import load_documents


def cmd_ingest(docs_dir: str, cfg: Config) -> None:
    docs = load_documents(docs_dir)
    print(f"loaded {len(docs)} documents")

    all_chunks, all_sources = [], []
    for d in docs:
        for c in chunk_text(d["text"], cfg.chunk):
            all_chunks.append(c)
            all_sources.append(d["source"])
    print(f"produced {len(all_chunks)} chunks ({cfg.chunk.strategy})")

    connect(cfg.milvus)
    col = get_or_create_collection(cfg.milvus)
    embeddings = embed_texts(all_chunks, cfg.embed.model, cfg.embed.batch_size)
    upsert_chunks(col, all_chunks, all_sources, embeddings)
    print(f"upserted into Milvus collection '{cfg.milvus.collection}'")


def cmd_eval(questions_path: str, cfg: Config) -> None:
    """Run the eval set through retrieve (+ generate) and print metric summaries."""
    # TODO: wire retrieve -> generate -> evaluate over a JSONL eval set.
    # Eval set format (one JSON per line):
    #   {"question": "...", "relevant_sources": ["doc.pdf"], "reference_answer": "..."}
    print(f"eval over {questions_path} — not yet wired. See examples/run_eval.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG eval harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest docs into Milvus")
    p_ingest.add_argument("--docs", required=True)

    p_eval = sub.add_parser("eval", help="run eval set")
    p_eval.add_argument("--questions", required=True)

    args = parser.parse_args()
    cfg = Config()
    if args.cmd == "ingest":
        cmd_ingest(args.docs, cfg)
    elif args.cmd == "eval":
        cmd_eval(args.questions, cfg)


if __name__ == "__main__":
    main()
