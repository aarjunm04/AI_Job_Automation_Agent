"""RAG system — resume ingestion, embedding, and query pipeline."""

from rag_systems.rag_pipeline import RAGPipeline
from rag_systems.resume_engine import ResumeEngine, get_default_engine
from rag_systems.chromadb_store import ChromaStore, ChromaStoreConfig
try:
    from rag_systems.ingestion import ingest_all_resumes, ingest_single_resume
except Exception:  # pragma: no cover - optional deps for ingestion
    def ingest_all_resumes(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("rag_systems.ingestion dependencies are not available in this environment")

    def ingest_single_resume(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("rag_systems.ingestion dependencies are not available in this environment")

__all__ = [
    "RAGPipeline",
    "ResumeEngine",
    "get_default_engine",
    "ChromaStore",
    "ChromaStoreConfig",
    "ingest_all_resumes",
    "ingest_single_resume",
]
