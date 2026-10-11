"""End-to-end eval run: retrieve -> generate -> score.

Usage:
    python examples/run_eval.py --questions ./data/eval_questions.jsonl

Eval set format (JSONL, one object per line):
    {"question": "...", "relevant_sources": ["doc.pdf"], "reference_answer": "..."}
"""

import argparse
import json

from src.chunk import chunk_text  # noqa: F401  (kept for discoverability)
from src.config import Config
from src.embed_store import connect, embed_texts, get_or_create_collection
from src.evaluate import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from src.retrieve import retrieve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    args = parser.parse_args()

    cfg = Config()
    connect(cfg.milvus)
    col = get_or_create_collection(cfg.milvus)

    # TODO: plug in your LLM client here (OpenAI / Anthropic / local).
    def llm_judge(prompt: str) -> str:
        raise NotImplementedError("wire up an LLM judge client")

    def generate(question: str, contexts: list[str]) -> str:
        raise NotImplementedError("wire up a generation call")

    scores = []
    with open(args.questions) as f:
        for line in f:
            item = json.loads(line)
            hits = retrieve(col, item["question"], cfg.embed.model, cfg.retrieval)
            contexts = [h["text"] for h in hits]
            answer = generate(item["question"], contexts)

            scores.append(
                {
                    "question": item["question"],
                    "context_precision": context_precision(
                        [h["source"] for h in hits],
                        set(item.get("relevant_sources", [])),
                        cfg.retrieval.top_k,
                    ),
                    "faithfulness": faithfulness(answer, contexts, llm_judge),
                    "answer_relevancy": answer_relevancy(
                        item["question"],
                        answer,
                        lambda t: embed_texts(t, cfg.embed.model),
                    ),
                }
            )

    for s in scores:
        print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
