"""Evaluation metrics, split by component.

Retriever metrics need ground-truth relevant chunk ids per question.
Generator metrics need (question, answer, retrieved context) triples.
"""

from dataclasses import dataclass


@dataclass
class EvalResult:
    context_precision: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None


def context_precision(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved chunks that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for i in top_k if i in relevant_ids) / len(top_k)


def context_recall(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Fraction of relevant chunks that were retrieved."""
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids) & relevant_ids) / len(relevant_ids)


def faithfulness(answer: str, contexts: list[str], llm_judge) -> float:
    """LLM-as-judge: is every claim in the answer supported by the context?

    llm_judge: callable(prompt) -> "yes" | "no" per claim. Split the answer
    into atomic claims, verify each against the contexts, return the fraction
    supported. Swap in any judge (OpenAI, Anthropic, local model).
    """
    claims = _split_claims(answer, llm_judge)
    if not claims:
        return 0.0
    supported = sum(
        1 for c in claims if _claim_supported(c, contexts, llm_judge)
    )
    return supported / len(claims)


def answer_relevancy(question: str, answer: str, embed_fn) -> float:
    """Cosine similarity between question and answer embeddings (proxy)."""
    import numpy as np

    q, a = embed_fn([question])[0], embed_fn([answer])[0]
    return float(np.dot(q, a) / (np.linalg.norm(q) * np.linalg.norm(a) + 1e-9))


def _split_claims(answer: str, llm_judge) -> list[str]:
    prompt = (
        "Split the following answer into atomic factual claims, one per line. "
        "No commentary.\n\nAnswer:\n" + answer
    )
    return [l.strip() for l in llm_judge(prompt).splitlines() if l.strip()]


def _claim_supported(claim: str, contexts: list[str], llm_judge) -> bool:
    context_block = "\n---\n".join(contexts)
    prompt = (
        "Given the context below, is the claim supported? Reply with only "
        f"'yes' or 'no'.\n\nContext:\n{context_block}\n\nClaim: {claim}"
    )
    return llm_judge(prompt).strip().lower().startswith("yes")
