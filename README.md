# RAG Evaluation Harness

A lightweight, extensible harness for evaluating Retrieval-Augmented Generation pipelines —
built to answer the question every RAG project eventually faces: *is the retriever or the generator at fault?*

## Why this exists

Most RAG evals conflate retrieval quality with generation quality. This harness separates the two:
score the retriever (context precision/recall) and the generator (faithfulness, answer relevancy)
independently, so you know exactly which component to fix.

## Architecture

```
PDFs / docs ──▶ ingest ──▶ chunk ──▶ embed ──▶ Milvus ──▶ retrieve ──▶ generate ──▶ evaluate
                                                        (top-k)        (LLM)       (metrics)
```

## Metrics

| Component  | Metric            | What it tells you                          |
|------------|-------------------|--------------------------------------------|
| Retriever  | Context precision | Are the retrieved chunks actually relevant? |
| Retriever  | Context recall    | Did we retrieve everything needed?          |
| Generator  | Faithfulness      | Is the answer grounded in the context?      |
| Generator  | Answer relevancy  | Does the answer address the question?       |

## Quickstart

```bash
pip install -r requirements.txt
# start Milvus (standalone)
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest
# ingest documents
python -m src.pipeline ingest --docs ./data
# run evaluation
python examples/run_eval.py --questions ./data/eval_questions.jsonl
```

## Project layout

```
src/
  config.py       # all knobs: chunk size, top-k, models, Milvus connection
  ingest.py       # document loading (pdf, txt, md)
  chunk.py        # chunking strategies (fixed, recursive, semantic)
  embed_store.py  # embeddings + Milvus collection management
  retrieve.py     # top-k retrieval with optional reranking
  evaluate.py     # metric implementations
  pipeline.py     # CLI orchestration
examples/
  run_eval.py     # end-to-end eval run
data/             # put public datasets here (not committed)
```

## Roadmap

- [ ] Reranker support (cross-encoder)
- [ ] Semantic chunking
- [ ] RAGAS / DeepEval adapter for metric comparison
- [ ] CI with a golden eval set (regression-guard your prompts)

## Notes

Built as a clean-room side project to explore RAG evaluation patterns used in production
GenAI platforms. Public datasets only — no proprietary code or data.
