import json
import logging
import unittest
from unittest.mock import patch, MagicMock

# Import the tools to be tested
from tools.rag_tools import (
    query_rag,
    get_resume_context,
    embed_job_description,
)
from tools.postgres_tools import _fetch_user_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared mock payload for users.user_settings JSONB
# ---------------------------------------------------------------------------

_MOCK_USER_SETTINGS: dict = {
    "default_resume": "AarjunGen.pdf",
    "dry_run": False,
    "auto_apply_enabled": True,
    "jobs_per_run_target": 150,
    "search_query": "AI ML Data Science Automation Engineer remote",
}

_MOCK_PLATFORM_SETTINGS: dict = {
    "platform_limits": {
        "linkedin": {"max_per_run": 40, "rate_limit_per_request_seconds": 3},
    },
    "scoring_weights": {
        "skills_match": 0.35,
        "experience_match": 0.25,
        "title_match": 0.20,
        "location_match": 0.10,
        "company_score": 0.10,
    },
}


# ---------------------------------------------------------------------------
# Existing tests — mocks updated for JSONB architecture
# ---------------------------------------------------------------------------


@patch("tools.rag_tools.rag_api.select_resume")
def test_query_resume_match_success(mock_select_resume: MagicMock) -> None:
    """Test that a successful RAG call returns the expected resume payload."""
    mock_select_resume.return_value = {
        "top_resume_id": "AarjunGen.pdf",
        "top_score": 0.88,
        "candidates": [
            {
                "final_score": 0.88,
                "recommended_chunks": [
                    {"document": "5+ years formatting LLM structured output"},
                    {"document": "Led backend architecture for multi-agent systems"},
                ],
            }
        ],
    }

    result_str = query_rag.func(
        job_description="We are looking for a Senior AI Agent Engineer with LLM architecture experience.",
        job_title="Senior AI Agent Engineer",
        required_skills="Python, LLM, AgentOps, Postgres",
    )

    result = json.loads(result_str)

    assert result["resume_suggested"] == "AarjunGen.pdf"
    assert result["similarity_score"] == 0.88
    assert result["fit_score"] == 0.88
    assert isinstance(result["talking_points"], list)
    assert len(result["talking_points"]) == 2
    assert "LLM structured output" in result["talking_points"][0]


@patch("tools.rag_tools._fetch_user_config")
@patch("tools.rag_tools.rag_api.select_resume")
def test_query_resume_match_fallback_on_error(
    mock_select_resume: MagicMock,
    mock_fetch: MagicMock,
) -> None:
    """Test that RAG failure triggers the DB fallback path for default_resume."""
    mock_fetch.return_value = {
        "user_settings": _MOCK_USER_SETTINGS,
        "platform_settings": _MOCK_PLATFORM_SETTINGS,
    }
    mock_select_resume.side_effect = Exception("ChromaDB connection timeout")

    result_str = query_rag.func(
        job_description="Description",
        job_title="Title",
        required_skills="Skills",
    )

    result = json.loads(result_str)

    assert result["resume_suggested"] == "AarjunGen.pdf"
    assert result["match_reasoning"] == "rag_unavailable_db_fallback"
    assert result["fit_score"] == 0.0
    mock_fetch.assert_called_once()


@patch("tools.rag_tools.rag_api.get_rag_context")
def test_get_resume_context(mock_get_rag_context: MagicMock) -> None:
    """Test get_resume_context formats chunks correctly."""
    mock_get_rag_context.return_value = {
        "top_chunks": [
            {"document": "Extensive experience with FastAPI and Postgres."},
            {"document": "Deployed applications using Docker and AWS."},
        ]
    }

    context = get_resume_context.func(
        resume_filename="ignore_me.pdf",
        job_description="Looking for Python backend with Docker.",
    )

    assert "[CHUNK 1] Extensive experience" in context
    assert "[CHUNK 2] Deployed applications" in context


# ---------------------------------------------------------------------------
# FIX B2: New test class for RAG → DB fallback paths
# ---------------------------------------------------------------------------


class TestRAGFallbackDBPath(unittest.TestCase):
    """Tests for query_resume_match DB fallback when RAG is unavailable.

    Validates that the production code fetches ``default_resume`` from
    ``users.user_settings`` JSONB when ChromaDB is unreachable.
    """

    def setUp(self) -> None:
        """Start patchers for RAG API and _fetch_user_config."""
        self.patcher_rag = patch("tools.rag_tools.rag_api.select_resume")
        self.mock_rag = self.patcher_rag.start()

        self.patcher_fetch = patch("tools.rag_tools._fetch_user_config")
        self.mock_fetch = self.patcher_fetch.start()

    def tearDown(self) -> None:
        """Stop all patchers."""
        self.patcher_fetch.stop()
        self.patcher_rag.stop()

    def test_query_resume_match_rag_unavailable_uses_db_fallback(self) -> None:
        """RAG down → DB fallback returns default_resume from user_settings."""
        self.mock_rag.side_effect = Exception("connection_refused")
        self.mock_fetch.return_value = {
            "user_settings": {"default_resume": "AarjunGen.pdf"},
            "platform_settings": {},
        }

        result_str = query_rag.func(
            job_description="Python developer job",
            job_title="ML Engineer",
            required_skills="Python",
        )
        result = json.loads(result_str)

        assert result["resume_suggested"] == "AarjunGen.pdf"
        assert result["match_reasoning"] == "rag_unavailable_db_fallback"
        assert result["fit_score"] == 0.0

    def test_query_resume_match_rag_and_db_both_fail(self) -> None:
        """RAG down AND DB down → hardcoded AarjunGen.pdf fallback, no raise."""
        self.mock_rag.side_effect = Exception("chromadb_crash")
        self.mock_fetch.side_effect = RuntimeError("db_config_fetch_failed")

        result_str = query_rag.func(
            job_description="Some job",
            job_title="Engineer",
            required_skills="Python",
        )
        result = json.loads(result_str)

        assert result["resume_suggested"] == "AarjunGen.pdf"
        assert result["fit_score"] == 0.0

    def test_query_resume_match_rag_success_skips_db(self) -> None:
        """When RAG succeeds, _fetch_user_config must NOT be called."""
        self.mock_rag.return_value = {
            "top_resume_id": "AarjunGen.pdf",
            "top_score": 0.92,
            "candidates": [
                {
                    "final_score": 0.92,
                    "recommended_chunks": [
                        {"document": "Expert in transformer architectures"},
                    ],
                }
            ],
        }

        result_str = query_rag.func(
            job_description="Build LLM pipelines",
            job_title="AI Engineer",
            required_skills="Python, LLM",
        )
        result = json.loads(result_str)

        self.mock_fetch.assert_not_called()
        assert result["fit_score"] > 0.0


if __name__ == "__main__":
    unittest.main()
