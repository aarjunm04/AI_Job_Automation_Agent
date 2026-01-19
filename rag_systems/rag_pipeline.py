# rag_systems/rag_pipeline.py
"""
Production RAG Pipeline with Gemini Embeddings API
Handles text chunking, embedding generation, and semantic search
"""

from __future__ import annotations
import os
import re
import httpx
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    Gemini embedding provider using text-embedding-004
    Supports 768, 1536, or 3072 dimensions with MRL
    """
    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY_RAG", ""))
    model: str = "models/text-embedding-004"
    output_dimensionality: int = 768  # Recommended for storage efficiency
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
            raise RuntimeError("Missing GEMINI_API_KEY_RAG in environment variables")

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
                
                # Validate embedding dimension
                if len(embedding) != self.output_dimensionality:
                    logger.warning(
                        f"Expected {self.output_dimensionality} dims, got {len(embedding)}"
                    )
                
                # Normalize for smaller dimensions (768, 1536)
                if self.output_dimensionality < 3072:
                    embedding = self._normalize(embedding)
                
                logger.debug(f"Generated {len(embedding)}-dim embedding")
                return embedding
            else:
                raise RuntimeError(f"Unexpected API response structure: {data}")

        except httpx.HTTPStatusError as e:
            logger.error(f"Gemini API HTTP error: {e.response.status_code} - {e.response.text}")
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
        Embed a query (job description) with RETRIEVAL_QUERY task type
        
        Args:
            text: Query text
            
        Returns:
            Query embedding vector
        """
        # Temporarily switch to query mode
        original_task = self.task_type
        self.task_type = "RETRIEVAL_QUERY"
        
        try:
            embedding = self.embed_text(text)
            return embedding
        finally:
            self.task_type = original_task


@dataclass
class LocalDeterministicEmbedder(EmbeddingProvider):
    """
    Fallback embedder using deterministic hashing
    For development/testing without API keys
    """
    dim: int = 768
    
    def embed_text(self, text: str) -> List[float]:
        """Generate deterministic embedding from text hash"""
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        vec = np.frombuffer(h * 4, dtype=np.uint8)[:self.dim].astype(float)
        normalized = vec / (np.linalg.norm(vec) or 1.0)
        return normalized.tolist()


# ============================================================
# CHUNKING
# ============================================================

def _sentence_split(text: str) -> List[str]:
    """Split text into sentences"""
    return re.split(r"(?<=[.!?])\s+", text.strip())


def chunk_text(
    text: str, 
    max_tokens: int = 450, 
    overlap: int = 80
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks for better retrieval
    
    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk (approximate by word count)
        overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of chunk dictionaries with metadata
    """
    import uuid
    
    sents = _sentence_split(text)
    chunks = []
    buf, count, offset = [], 0, 0

    for s in sents:
        wc = len(s.split())
        
        # Create chunk if buffer exceeds max_tokens
        if count + wc > max_tokens and buf:
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": " ".join(buf),
                "offset": offset,
                "token_count": count,
                "section_heading": None
            })

            # Keep overlap for context continuity
            if overlap > 0:
                keep, kept = [], 0
                for prev in reversed(buf):
                    w = len(prev.split())
                    if kept >= overlap:
                        break
                    keep.insert(0, prev)
                    kept += w
                buf = keep
                count = sum(len(x.split()) for x in buf)
            else:
                buf = []
                count = 0

        buf.append(s)
        count += wc
        offset += 1

    # Add remaining buffer as final chunk
    if buf:
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "text": " ".join(buf),
            "offset": offset,
            "token_count": count,
            "section_heading": None
        })

    logger.info(f"Chunked text into {len(chunks)} chunks (max_tokens={max_tokens}, overlap={overlap})")
    return chunks


# ============================================================
# RAG PIPELINE
# ============================================================

@dataclass
class RAGPipeline:
    """
    Main RAG orchestration pipeline
    Handles embedding, retrieval, and context building
    """
    store: Any  # ChromaStore instance
    embedder: EmbeddingProvider

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
        
        # Use query-specific embedding if available
        if isinstance(self.embedder, GeminiEmbedder):
            return self.embedder.embed_query(cleaned)
        else:
            return self.embedder.embed_text(cleaned)

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
        use_local_embedder: If True, use local deterministic embedder
        
    Returns:
        Configured RAGPipeline instance
    """
    if use_local_embedder or not os.getenv("GEMINI_API_KEY_RAG"):
        logger.warning("Using LocalDeterministicEmbedder (no API key found)")
        embedder = LocalDeterministicEmbedder(dim=768)
    else:
        logger.info("Using GeminiEmbedder with text-embedding-004")
        embedder = GeminiEmbedder(
            output_dimensionality=768,
            task_type="RETRIEVAL_DOCUMENT"
        )
    
    return RAGPipeline(store=chroma_store, embedder=embedder)

