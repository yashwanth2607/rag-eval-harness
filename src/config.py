"""Central configuration — every tuning knob lives here."""

from dataclasses import dataclass, field


@dataclass
class ChunkConfig:
    strategy: str = "recursive"  # fixed | recursive | semantic
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class EmbedConfig:
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64


@dataclass
class MilvusConfig:
    host: str = "localhost"
    port: str = "19530"
    collection: str = "rag_docs"
    dim: int = 384  # must match embedding model output
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"


@dataclass
class RetrievalConfig:
    top_k: int = 5
    rerank: bool = False


@dataclass
class EvalConfig:
    # thresholds for pass/fail on the golden set
    min_context_precision: float = 0.7
    min_faithfulness: float = 0.8


@dataclass
class Config:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
