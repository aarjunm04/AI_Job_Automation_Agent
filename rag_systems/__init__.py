"""
rag_systems — public API.

Single ingestion entry point: ingest_all_resumes()
All calls to ingest resumes must go through this function.
"""
from rag_systems.ingestion import ingest_all_resumes, ingest_single_resume

__all__ = ["ingest_all_resumes", "ingest_single_resume"]
