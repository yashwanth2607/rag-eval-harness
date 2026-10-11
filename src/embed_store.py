"""Embeddings + Milvus collection management."""

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from .config import MilvusConfig


def connect(cfg: MilvusConfig) -> None:
    connections.connect(host=cfg.host, port=cfg.port)


def get_or_create_collection(cfg: MilvusConfig) -> Collection:
    if utility.has_collection(cfg.collection):
        return Collection(cfg.collection)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=cfg.dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
    ]
    schema = CollectionSchema(fields, description="RAG document chunks")
    col = Collection(cfg.collection, schema)
    col.create_index(
        "embedding",
        {"index_type": cfg.index_type, "metric_type": cfg.metric_type, "params": {"nlist": 128}},
    )
    return col


def embed_texts(texts: list[str], model_name: str, batch_size: int = 64) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True).tolist()


def upsert_chunks(
    col: Collection, chunks: list[str], sources: list[str], embeddings: list[list[float]]
) -> None:
    col.insert([embeddings, chunks, sources])
    col.flush()
    col.load()
