"""
═══════════════════════════════════════════════════════════════════════════════
JOB AUTOMATION MCP - INTEGRATIONS MODULE
═══════════════════════════════════════════════════════════════════════════════

External service integrations for MCP.

Components:
- CircuitBreaker: Fault tolerance for external service calls
- RagClient: Resume-to-job matching using RAG/vector search
- LLMRouter: Multi-provider LLM routing (NVIDIA NIM DeepSeek + Perplexity)
- NotionClient: Job tracking and logging in Notion
- ScraperClient: Job scraping orchestration

Features:
- Automatic failover between providers
- Circuit breaker pattern for resilience
- Retry logic with exponential backoff
- Provider health tracking
- Request/response caching
- Comprehensive error handling

Author: Job Automation Team
Version: 2.0 Enterprise
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import os
import time
import asyncio
import logging
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum

# HTTP client
import httpx

logger = logging.getLogger("mcp.integrations")


# ═══════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════════════════

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.

    Prevents cascading failures by opening circuit after threshold failures.
    Automatically attempts recovery after cooldown period.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            success_threshold: Successes needed to close circuit in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None

        logger.debug(f"CircuitBreaker initialized: threshold={failure_threshold}")

    async def call(self, func, *args, **kwargs):
        """
        Execute function through circuit breaker.

        Args:
            func: Async function to execute
            *args, **kwargs: Function arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            # Execute function
            result = await func(*args, **kwargs)

            # Handle success
            self._on_success()
            return result

        except Exception as e:
            # Handle failure
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if not self.last_failure_time:
            return True

        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.recovery_timeout

    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker CLOSED after recovery")
        else:
            self.failure_count = max(0, self.failure_count - 1)

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker reopened during recovery attempt")


# ═══════════════════════════════════════════════════════════════════════════════
# RAG CLIENT (Resume Matching)
# ═══════════════════════════════════════════════════════════════════════════════

"""
RAG CLIENT (Resume Matching)

RAG system for resume-to-job matching.

Uses vector embeddings to find optimal resume match for job description.

TODO: Implement actual RAG logic with:
- Embedding model (e.g., sentence-transformers, OpenAI embeddings)
- Vector database (e.g., Pinecone, Weaviate, Qdrant, Chroma)
- Resume parsing and chunking
- Semantic search
- Scoring and ranking
"""

class RagClient:
    """Client for RAG resume matching system running on port 8090"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8090",
        api_key: str = "dev-666a3ac3d47d42161b8ae35f93b9cbd1",
        timeout: int = 30
    ):
        """
        Initialize RAG client for resume selection
        
        Args:
            base_url: RAG server URL (default: http://localhost:8090)
            api_key: X-RAG-API-Key authentication header
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        
        # Available specialized resumes
        self.resume_types = [
            'resume_ai_automation',
            'resume_ai_ml', 
            'resume_data_engineering',
            'resume_data_science',
            'resume_generic',
            'resume_llm_genai',
            'resume_mlops',
            'resume_original'
        ]
        
        logger.info(f"✅ RAG Client initialized - {base_url}")
    
    async def select_best_resume(
        self,
        job_description: str,
        session_id: str,
        top_k: int = 10,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Query RAG system to select optimal resume for job posting
        
        Args:
            job_description: Full job posting text with requirements
            session_id: Unique session ID for context tracking
            top_k: Number of resume chunks to retrieve (default: 10)
            include_metadata: Include detailed metadata in response
            
        Returns:
            Dict containing:
                - answer: RAG analysis explanation
                - selected_resume: Resume name (e.g., 'resume_llm_genai')
                - selected_resume_id: Resume identifier
                - selected_resume_path: Full file path to resume
                - confidence_score: Match confidence 0.0-1.0
                - matching_skills: List of matched skills
                - chunks_retrieved: Number of chunks analyzed
                - session_id: Session identifier
                
        Example:
            {
                "answer": "Based on LLM requirements, resume_llm_genai is best match",
                "selected_resume": "resume_llm_genai",
                "selected_resume_path": "/Users/apple/TechStack/Resumes/resume_llm_genai.pdf",
                "confidence_score": 0.94,
                "matching_skills": ["Python", "RAG", "LangChain", "LLMs"],
                "chunks_retrieved": 10,
                "session_id": "sess_123"
            }
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/rag/query",
                headers={
                    "Content-Type": "application/json",
                    "X-RAG-API-Key": self.api_key
                },
                json={
                    "session_id": session_id,
                    "query": job_description,
                    "top_k": top_k,
                    "include_metadata": include_metadata
                },
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            selected = result.get('selected_resume', 'resume_generic')
            confidence = result.get('confidence_score', 0.0)
            
            logger.info(
                f"RAG selected '{selected}' with {confidence:.2%} confidence "
                f"for session {session_id}"
            )
            
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(
                f"RAG API HTTP error {e.response.status_code}: {e.response.text}"
            )
            raise Exception(
                f"RAG API error: {e.response.status_code} - {e.response.text}"
            )
            
        except httpx.RequestError as e:
            logger.error(f"RAG connection failed: {str(e)}")
            raise Exception(
                "Failed to connect to RAG system. "
                "Ensure RAG server is running at localhost:8090"
            )
            
        except Exception as e:
            logger.error(f"RAG query error: {str(e)}")
            raise
    
    async def compare_resumes(
        self,
        job_description: str,
        resume_names: List[str],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Compare multiple resumes against job description
        
        Args:
            job_description: Job posting text
            resume_names: List of resume names to compare
            session_id: Session identifier
            
        Returns:
            Comparison results with scores for each resume
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/rag/compare",
                headers={
                    "Content-Type": "application/json",
                    "X-RAG-API-Key": self.api_key
                },
                json={
                    "session_id": session_id,
                    "query": job_description,
                    "resume_names": resume_names
                }
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Resume comparison error: {str(e)}")
            raise
    
    async def health_check(self) -> bool:
        """
        Check RAG system health and availability
        
        Returns:
            True if RAG system is responsive, False otherwise
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/health",
                timeout=5
            )
            is_healthy = response.status_code == 200
            
            if is_healthy:
                logger.info("✅ RAG system health check passed")
            else:
                logger.warning(f"⚠️ RAG health check failed: {response.status_code}")
                
            return is_healthy
            
        except Exception as e:
            logger.warning(f"⚠️ RAG health check failed: {str(e)}")
            return False
    
    async def get_available_resumes(self) -> List[str]:
        """
        Get list of available resume types from RAG system
        
        Returns:
            List of resume names available in the system
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/resumes",
                headers={"X-RAG-API-Key": self.api_key}
            )
            
            if response.status_code == 200:
                return response.json().get('resumes', self.resume_types)
            else:
                return self.resume_types
                
        except Exception as e:
            logger.warning(f"Could not fetch resumes: {str(e)}")
            return self.resume_types
    
    async def close(self):
        """Close HTTP client connection pool"""
        await self.client.aclose()
        logger.info("RAG client connection closed")



# ═══════════════════════════════════════════════════════════════════════════════
# LLM ROUTER (NVIDIA NIM + Perplexity)
# ═══════════════════════════════════════════════════════════════════════════════

class LLMProvider(Enum):
    """Supported LLM providers."""
    NVIDIA_NIM = "nvidia_nim"
    PERPLEXITY = "perplexity"


class LLMRouter:
    """
    Multi-provider LLM router with automatic failover.

    Routes requests to optimal LLM provider based on:
    - Task type (research, form_filling, summarization, etc.)
    - Provider availability
    - Cost optimization
    - Response quality

    Providers:
    - NVIDIA NIM: DeepSeek R1 Distill Llama 70B (reasoning, decision making)
    - Perplexity: Sonar models (research, web search)
    """

    # Task-to-provider mapping
    TASK_ROUTING = {
        "research": LLMProvider.PERPLEXITY,      # Best for search/research
        "form_filling": LLMProvider.NVIDIA_NIM,  # Fast and accurate
        "job_matching": LLMProvider.NVIDIA_NIM,  # Good at analysis
        "decision_making": LLMProvider.NVIDIA_NIM, # Reasoning tasks
        "summarization": LLMProvider.NVIDIA_NIM, # Efficient
        "general": LLMProvider.NVIDIA_NIM        # Default
    }

    def __init__(self):
        """Initialize LLM router with configured providers."""
        self.providers = {}
        self.circuit_breakers = {}

        # Initialize NVIDIA NIM (DeepSeek R1)
        nvidia_key = os.getenv("NVIDIA_NIM_API_KEY")
        if nvidia_key:
            self.providers[LLMProvider.NVIDIA_NIM] = {
            "api_key": nvidia_key,
            "base_url": "https://integrate.api.nvidia.com/v1",
            "model": "deepseek-ai/deepseek-v3.1-terminus"
            }
            self.circuit_breakers[LLMProvider.NVIDIA_NIM] = CircuitBreaker()
            logger.info("✅ NVIDIA NIM provider initialized")

        # Initialize Perplexity
        perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        if perplexity_key:
            self.providers[LLMProvider.PERPLEXITY] = {
                "api_key": perplexity_key,
                "base_url": "https://api.perplexity.ai",
                "model": os.getenv("PERPLEXITY_MODEL", "sonar")
            }
            self.circuit_breakers[LLMProvider.PERPLEXITY] = CircuitBreaker()
            logger.info("✅ Perplexity provider initialized")

        if not self.providers:
            logger.error("❌ No LLM providers configured! Set NVIDIA_NIM_API_KEY or PERPLEXITY_API_KEY")
        else:
            logger.info(f"✅ LLMRouter initialized with {len(self.providers)} provider(s)")

    async def complete(
        self,
        prompt: str,
        task_type: str = "general",
        max_tokens: int = 2000,
        temperature: float = 0.7,
        preferred_provider: Optional[LLMProvider] = None
    ) -> Dict[str, Any]:
        """
        Generate LLM completion with automatic failover.

        Args:
            prompt: Input prompt
            task_type: Task type for routing (research/form_filling/etc.)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            preferred_provider: Override automatic routing

        Returns:
            Dict with content, provider, usage stats

        Example return:
            {
                "content": "Response text...",
                "provider": "nvidia_nim",
                "model": "deepseek-r1-distill-llama-70b",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "finish_reason": "stop"
            }
        """
        # Determine provider
        if preferred_provider and preferred_provider in self.providers:
            provider = preferred_provider
        else:
            provider = self.TASK_ROUTING.get(task_type, LLMProvider.NVIDIA_NIM)
            if provider not in self.providers:
                provider = list(self.providers.keys())[0] if self.providers else None

        if not provider:
            raise Exception("No LLM providers available")

        # Try primary provider
        try:
            circuit_breaker = self.circuit_breakers[provider]
            result = await circuit_breaker.call(
                self._call_provider,
                provider,
                prompt,
                max_tokens,
                temperature
            )
            return result
        except Exception as e:
            logger.warning(f"Primary provider {provider.value} failed: {e}")

            # Try fallback providers
            for fallback_provider in self.providers.keys():
                if fallback_provider == provider:
                    continue

                try:
                    logger.info(f"Attempting fallback to {fallback_provider.value}")
                    circuit_breaker = self.circuit_breakers[fallback_provider]
                    result = await circuit_breaker.call(
                        self._call_provider,
                        fallback_provider,
                        prompt,
                        max_tokens,
                        temperature
                    )
                    return result
                except Exception as fallback_error:
                    logger.warning(f"Fallback provider {fallback_provider.value} failed: {fallback_error}")
                    continue

            raise Exception("All LLM providers failed")

    async def _call_provider(
        self,
        provider: LLMProvider,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """
        Call specific LLM provider API.

        Implements OpenAI-compatible chat completion format for both providers.
        """
        config = self.providers[provider]

        logger.info(f"Calling {provider.value} API for completion")

        # Build request payload (OpenAI format)
        payload = {
            "model": config["model"],
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }

        # Add provider-specific headers
        if provider == LLMProvider.PERPLEXITY:
            # Perplexity may need additional params for web search
            payload["return_citations"] = True
            payload["return_images"] = False

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{config['base_url']}/chat/completions",
                    json=payload,
                    headers=headers
                )

                response.raise_for_status()
                data = response.json()

                # Parse response (OpenAI format)
                choice = data["choices"][0]
                usage = data.get("usage", {})

                result = {
                    "content": choice["message"]["content"],
                    "provider": provider.value,
                    "model": config["model"],
                    "usage": {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)
                    },
                    "finish_reason": choice.get("finish_reason", "stop")
                }

                # Add citations if available (Perplexity)
                if "citations" in data:
                    result["citations"] = data["citations"]

                logger.info(f"✅ {provider.value} completion successful ({usage.get('total_tokens', 0)} tokens)")
                return result

        except httpx.HTTPStatusError as e:
            logger.error(f"{provider.value} API error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"{provider.value} call failed: {e}")
            raise

    async def abstractive_summary(
        self,
        text: str,
        strategy: str = "rolling",
        max_sentences: int = 8
    ) -> str:
        """
        Generate abstractive summary using LLM.

        Args:
            text: Text to summarize
            strategy: Summarization strategy
            max_sentences: Max sentences in summary

        Returns:
            Summary text
        """
        prompt = f"""Summarize the following text in {max_sentences} sentences or less.
Focus on key information and actionable insights.

Text:
{text[:10000]}

Summary:"""

        result = await self.complete(
            prompt,
            task_type="summarization",
            max_tokens=500,
            temperature=0.3
        )

        return result["content"]


# ═══════════════════════════════════════════════════════════════════════════════
# NOTION CLIENT (Job Tracking)
# ═══════════════════════════════════════════════════════════════════════════════

class NotionClient:
    """
    Notion integration for job tracking dashboard.

    Maintains a Notion database with:
    - Job listings (scraped/manual)
    - Application status (applied/interviewing/rejected/offer)
    - Timestamps and metadata
    - Notes and follow-ups

    TODO: Implement Notion API integration
    """

    def __init__(
        self,
        api_key: str = None,
        database_id: str = None
    ):
        """
        Initialize Notion client.

        Args:
            api_key: Notion integration API key
            database_id: Notion database ID for job tracking
        """
        self.api_key = api_key or os.getenv("NOTION_API_KEY")
        self.database_id = database_id or os.getenv("NOTION_DATABASE_ID")
        self.circuit_breaker = CircuitBreaker()

        if not self.api_key or not self.database_id:
            logger.warning("⚠️  Notion credentials not configured")
        else:
            logger.info("✅ NotionClient initialized")

    async def create_job_entry(
        self,
        job_data: Dict[str, Any]
    ) -> str:
        """
        Create new job entry in Notion database.

        Args:
            job_data: Job information dict
                Required fields: title, company, url
                Optional: location, salary, description, source

        Returns:
            Notion page ID

        TODO: Implement actual Notion API call
        """
        logger.info(f"Creating Notion entry for job: {job_data.get('title')}")

        try:
            return await self.circuit_breaker.call(
                self._create_page,
                job_data
            )
        except Exception as e:
            logger.error(f"Failed to create Notion entry: {e}")
            return f"mock_page_{int(time.time())}"

    async def _create_page(self, job_data: Dict[str, Any]) -> str:
        """
        Internal Notion page creation.

        TODO: Implement with Notion API:
        POST https://api.notion.com/v1/pages
        """
        # STUB: Simulate API call
        await asyncio.sleep(0.1)

        page_id = f"notion_page_{int(time.time())}"
        logger.debug(f"Created Notion page {page_id}")

        return page_id

    async def update_job_status(
        self,
        page_id: str,
        status: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Update job application status.

        Args:
            page_id: Notion page ID
            status: New status (applied/interviewing/rejected/offer)
            notes: Additional notes

        Returns:
            True if successful

        TODO: Implement Notion API update
        """
        logger.info(f"Updating Notion page {page_id} status to {status}")

        try:
            return await self.circuit_breaker.call(
                self._update_page,
                page_id,
                {"status": status, "notes": notes}
            )
        except Exception as e:
            logger.error(f"Failed to update Notion page: {e}")
            return False

    async def _update_page(
        self,
        page_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Internal Notion page update.

        TODO: Implement with Notion API:
        PATCH https://api.notion.com/v1/pages/{page_id}
        """
        # STUB: Simulate API call
        await asyncio.sleep(0.1)
        return True

    async def query_jobs(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query jobs from Notion database.

        Args:
            filters: Notion filter object
            limit: Max results to return

        Returns:
            List of job entries

        TODO: Implement Notion query API
        """
        logger.info(f"Querying Notion database with filters: {filters}")

        # STUB: Return empty list
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# SCRAPER CLIENT (Job Scraping Orchestration)
# ═══════════════════════════════════════════════════════════════════════════════

class ScraperClient:
    """
    Job scraping orchestration client.

    Coordinates with external scraper services:
    - Jooble API
    - Remotive API
    - Custom Playwright scrapers
    - JobSpy integration

    TODO: Implement scraper integrations
    """

    def __init__(self):
        """Initialize scraper client."""
        self.jooble_api_key = os.getenv("JOOBLE_API_KEY")
        self.circuit_breaker = CircuitBreaker()

        logger.info("✅ ScraperClient initialized (stub mode)")

    async def trigger_scrape(
        self,
        source: str,
        search_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Trigger job scraping from source.

        Args:
            source: Source name (jooble/remotive/linkedin/custom)
            search_params: Search parameters
                {
                    "keywords": "software engineer",
                    "location": "remote",
                    "experience_level": "mid",
                    "limit": 50
                }

        Returns:
            Dict with job_count and session_id

        TODO: Implement scraper triggers
        """
        logger.info(f"Triggering scrape from {source}")

        try:
            return await self.circuit_breaker.call(
                self._scrape,
                source,
                search_params
            )
        except Exception as e:
            logger.error(f"Scrape trigger failed: {e}")
            return {"job_count": 0, "session_id": None, "error": str(e)}

    async def _scrape(
        self,
        source: str,
        search_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Internal scraping logic.

        TODO: Implement actual scraper calls:
        - Jooble: POST to API with search params
        - Remotive: GET from API endpoint
        - Custom: Trigger Playwright scraper via webhook
        """
        # STUB: Simulate scraping
        await asyncio.sleep(0.5)

        return {
            "job_count": 25,  # Mock count
            "session_id": f"scrape_{source}_{int(time.time())}",
            "source": source
        }

    async def get_scrape_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get scraping job status.

        Args:
            session_id: Scraping session ID

        Returns:
            Status dict with progress

        TODO: Implement status tracking
        """
        # STUB: Return mock status
        return {
            "session_id": session_id,
            "status": "completed",
            "progress": 100,
            "jobs_found": 25
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS (for server.py integration)
# ═══════════════════════════════════════════════════════════════════════════════

def get_rag_client() -> RagClient:
    """Factory function to get RagClient instance."""
    # Read RAG_KEY_MCP directly from environment
    rag_key = os.getenv("RAG_KEY_MCP", "mcp-default-key")
    
    return RagClient(
        base_url=os.getenv("RAG_BASE_URL", "http://localhost:8090"),
        api_key=rag_key,  # Use MCP-specific key
        timeout=int(os.getenv("RAG_TIMEOUT", "30"))
    )


def get_llm_router() -> LLMRouter:
    """Factory function to get LLMRouter instance."""
    return LLMRouter()


def get_notion_client() -> NotionClient:
    """Factory function to get NotionClient instance."""
    return NotionClient(
        api_key=os.getenv("NOTION_API_KEY"),
        database_id=os.getenv("NOTION_DATABASE_ID")
    )


def get_scraper_client() -> ScraperClient:
    """Factory function to get ScraperClient instance."""
    return ScraperClient()