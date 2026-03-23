# rag_systems/rag_pipeline.py
"""
Production RAG Pipeline with Gemini Embeddings API
Handles text chunking, embedding generation, and semantic search
"""

from __future__ import annotations
import os
import re
import time
import httpx
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = [
    "chunk_text",
    "EmbeddingProvider",
    "GeminiEmbedder",
    "NVIDIANIMEmbedder",
    "EmbeddingService",
    "RAGPipeline",
    "create_default_pipeline",
]


# ============================================================
# CHUNKING (single authoritative implementation)
# ============================================================


def chunk_text(
    text: str,
    chunk_size: int = 450,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split text into overlapping chunks for embedding.

    Args:
        text: Raw input text to chunk.
        chunk_size: Maximum tokens per chunk (default 400 — safely under NIM 512 limit).
        chunk_overlap: Token overlap between consecutive chunks (default 80).

    Returns:
        List of text chunk strings, minimum 1 chunk always returned.
    """
    tokens = text.split()
    if not tokens:
        return [text] if text else [""]

    step = max(1, chunk_size - chunk_overlap)
    chunks: list[str] = []
    for i in range(0, len(tokens), step):
        chunk = " ".join(tokens[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks if chunks else [text]


# ============================================================
# EMBEDDING PROVIDERS
# ============================================================

class EmbeddingProvider:
    """Base class for embedding providers"""
    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError


@dataclass
class GeminiEmbedder(EmbeddingProvider):
    """
    Gemini embedding provider using Gemini embeddings API.
    Supports 768, 1536, or 3072 dimensions with MRL
    """
    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model: str = field(default_factory=lambda: "models/text-embedding-004")
    output_dimensionality: int = 768  # Match NVIDIA NIM dimension for consistent vector space
    task_type: str = "RETRIEVAL_DOCUMENT"  # For resume indexing
    timeout_seconds: float = 30

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings using Gemini API
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY in environment variables")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"{self.model}:embedContent?key={self.api_key}"
        )

        payload = {
            "model": self.model,
            "content": {
                "parts": [{"text": text}]
            },
            "taskType": self.task_type,
            "outputDimensionality": self.output_dimensionality
        }

        try:
            resp = httpx.post(url, json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            
            # Extract embedding from response
            # Response structure: {"embedding": {"values": [0.1, 0.2, ...]}}
            if "embedding" in data and "values" in data["embedding"]:
                embedding = data["embedding"]["values"]
                
                # Validate embedding dimension — discard and signal None on mismatch
                if len(embedding) != self.output_dimensionality:
                    logger.error(
                        "Embedding dimension mismatch from Gemini: got %d, "
                        "expected %d — discarding, will not write to ChromaDB",
                        len(embedding),
                        self.output_dimensionality,
                    )
                    return None

                # Normalize for smaller dimensions (768, 1536)
                if self.output_dimensionality < 3072:
                    embedding = self._normalize(embedding)

                logger.debug("Generated %d-dim Gemini embedding", len(embedding))
                return embedding
            else:
                raise RuntimeError(f"Unexpected API response structure: {data}")

        except httpx.HTTPStatusError as e:
            logger.error(f"Gemini API HTTP error: {e.response.status_code} - {e.response.text}")
            if e.response.status_code == 404:
                raise RuntimeError(
                    "Gemini embedding endpoint/model not found (404). "
                    f"Configured model='{self.model}'. "
                    "Set GEMINI_EMBEDDING_MODEL=models/text-embedding-004 "
                    "or verify model access for your API key."
                ) from e
            raise RuntimeError(f"Gemini API request failed: {e}") from e
        except httpx.RequestError as e:
            logger.error(f"Gemini API connection error: {e}")
            raise RuntimeError("Failed to connect to Gemini API") from e
        except Exception as e:
            logger.exception("Gemini embedding error: %s", e)
            raise RuntimeError("Gemini embedding request failed") from e

    @staticmethod
    def _normalize(embedding: List[float]) -> List[float]:
        """
        Normalize embedding vector to unit length
        Required for dimensions < 3072 for accurate similarity
        """
        vec = np.array(embedding)
        norm = np.linalg.norm(vec)
        if norm > 0:
            return (vec / norm).tolist()
        return embedding

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query (job description) with RETRIEVAL_QUERY task type.
        """
        original_task = self.task_type
        self.task_type = "RETRIEVAL_QUERY"

        try:
            return self.embed_text(text)
        finally:
            self.task_type = original_task


@dataclass
class NVIDIANIMEmbedder(EmbeddingProvider):
    """
    Primary embedding provider using NVIDIA NIM OpenAI-compatible embeddings API.
    """

    base_url: str = field(
        default_factory=lambda: os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    )
    api_key: str = field(default_factory=lambda: os.getenv("NVIDIA_NIM_API_KEY", ""))
    model: str = "nvidia/nv-embedqa-e5-v5"
    dimensions: int = 1024
    timeout_seconds: float = 30.0

    def embed_text(self, text: str, input_type: str = "passage") -> List[float]:
        """
        Generate embeddings using NVIDIA NIM OpenAI-compatible embeddings API.
        """
        if not self.api_key:
            raise RuntimeError("Missing NVIDIA_NIM_API_KEY in environment variables")

        url = f"{self.base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": text,
            "encoding_format": "float",
            "input_type": input_type,
            "truncate": "END",
        }
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=self.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()

            # Expected OpenAI-compatible response: {"data":[{"embedding":[...]}], ...}
            embeddings = data.get("data") or []
            if not embeddings or "embedding" not in embeddings[0]:
                raise RuntimeError(f"Unexpected NVIDIA NIM response structure: {data}")

            embedding: List[float] = embeddings[0]["embedding"]
            if len(embedding) != self.dimensions:
                logger.error(
                    "Embedding dimension mismatch from NVIDIA NIM: got %d, "
                    "expected %d — discarding, will not write to ChromaDB",
                    len(embedding),
                    self.dimensions,
                )
                return None

            logger.debug("Generated %d-dim NVIDIA NIM embedding", len(embedding))
            return embedding
        except httpx.HTTPStatusError as e:
            logger.error(
                "NVIDIA NIM API HTTP error: %s - %s",
                e.response.status_code,
                e.response.text,
            )
            raise RuntimeError(f"NVIDIA NIM API request failed: {e}") from e
        except httpx.RequestError as e:
            logger.error("NVIDIA NIM API connection error: %s", e)
            raise RuntimeError("Failed to connect to NVIDIA NIM API") from e
        except Exception as e:  # noqa: BLE001
            logger.exception("NVIDIA NIM embedding error: %s", e)
            raise RuntimeError("NVIDIA NIM embedding request failed") from e


@dataclass
class EmbeddingService:
    """
    Embedding service that tries NVIDIA NIM first, then falls back to Gemini.

    All external embedding calls are wrapped with retry (max 3) and
    exponential backoff (time.sleep(2**attempt)).
    """

    primary: EmbeddingProvider = field(default_factory=NVIDIANIMEmbedder)
    fallback: EmbeddingProvider = field(default_factory=GeminiEmbedder)
    max_retries: int = 3

    def embed_text(self, text: str) -> List[float]:
        return self._embed(text, is_query=False)

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text, is_query=True)

    def _embed(self, text: str, is_query: bool) -> List[float]:
        last_error: Optional[Exception] = None

        for provider_name, provider in (
            ("NVIDIA_NIM", self.primary),
            ("Gemini", self.fallback),
        ):
            for attempt in range(self.max_retries):
                try:
                    vector = None
                    if isinstance(provider, NVIDIANIMEmbedder):
                        input_type = "query" if is_query else "passage"
                        vector = provider.embed_text(text, input_type=input_type)
                    elif isinstance(provider, GeminiEmbedder):
                        if is_query:
                            vector = provider.embed_query(text)
                        else:
                            vector = provider.embed_text(text)

                    if vector is None:
                        raise ValueError(
                            f"{provider_name} embed returned None (dimension mismatch at provider level)"
                        )

                    # This validation is complex because providers have different dimensions.
                    # We check against the provider's own configured dimension.
                    expected_dim = 0
                    if isinstance(provider, NVIDIANIMEmbedder):
                        expected_dim = provider.dimensions
                    elif isinstance(provider, GeminiEmbedder):
                        expected_dim = provider.output_dimensionality

                    if expected_dim > 0 and len(vector) != expected_dim:
                        logger.warning(
                            f"{provider_name} returned {len(vector)}-dim vector, expected {expected_dim}. "
                            "This may cause issues if the ChromaDB collection has a fixed dimension."
                        )

                    return vector
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    logger.error(
                        "Embedding error with %s on attempt %d: %s",
                        provider_name,
                        attempt + 1,
                        exc,
                    )
                    # Exponential backoff: 1, 2, 4 seconds between retries
                    if attempt < self.max_retries - 1:
                        time.sleep(2**attempt)
            logger.warning(
                "Provider %s failed after %d attempts, trying next fallback",
                provider_name,
                self.max_retries,
            )

        error_message = (
            "All embedding providers failed. "
            f"Last error: {last_error!r}" if last_error else "All embedding providers failed."
        )
        logger.error(error_message)
        raise RuntimeError(error_message)


# ============================================================
# RAG PIPELINE
# ============================================================

@dataclass
class RAGPipeline:
    """
    Main RAG orchestration pipeline.
    Handles embedding, retrieval, and context building.
    """
    store: Any  # ChromaStore instance
    embedder: EmbeddingProvider
    embedding_service: EmbeddingService = field(default_factory=EmbeddingService)

    def clean_job_text(self, text: str) -> str:
        """Clean and normalize job description text"""
        return re.sub(r"\s+", " ", text or "").strip()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query with proper preprocessing
        
        Args:
            text: Query text (job description)
            
        Returns:
            Query embedding vector
        """
        cleaned = self.clean_job_text(text)
        return self.embedding_service.embed_query(cleaned)

    def get_top_resume_anchors(
        self, 
        embedding: List[float], 
        k: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top K resume-level anchor embeddings
        
        Args:
            embedding: Query embedding
            k: Number of anchors to retrieve
            
        Returns:
            List of anchor matches with metadata
        """
        return self.store.query_anchor(query_embedding=embedding, n_results=k)

    def get_relevant_chunks(
        self, 
        embedding: List[float], 
        filters: Optional[Dict[str, Any]] = None, 
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top K relevant chunks with optional filtering
        
        Args:
            embedding: Query embedding
            filters: Optional metadata filters (e.g., {"resume_id": "xyz"})
            k: Number of chunks to retrieve
            
        Returns:
            List of chunk matches with metadata and distances
        """
        return self.store.query(
            query_embeddings=[embedding], 
            n_results=k, 
            where=filters
        )

    def build_grounding_context(
        self, 
        text: str, 
        top_k_chunks: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build complete grounding context for a query
        
        Args:
            text: Query text
            top_k_chunks: Number of chunks to retrieve
            filters: Optional metadata filters
            
        Returns:
            Dictionary with query embedding and retrieved chunks
        """
        vec = self.embed_query(text)
        chunks = self.get_relevant_chunks(vec, filters, top_k_chunks)
        
        return {
            "query_text": text,
            "query_embedding": vec,
            "top_chunks": chunks,
            "num_chunks": len(chunks)
        }


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def create_default_pipeline(
    chroma_store: Any,
    use_local_embedder: bool = False
) -> RAGPipeline:
    """
    Create RAG pipeline with default configuration
    
    Args:
        chroma_store: ChromaStore instance
        use_local_embedder: Kept for backwards compatibility (ignored).

    Returns:
        Configured RAGPipeline instance with NVIDIA NIM primary
        and Gemini fallback embedding service.
    """
    embedding_service = EmbeddingService()
    # Preserve existing signature by still passing an embedder instance,
    # while routing all queries through EmbeddingService.
    return RAGPipeline(
        store=chroma_store,
        embedder=embedding_service.primary,
        embedding_service=embedding_service,
    )
