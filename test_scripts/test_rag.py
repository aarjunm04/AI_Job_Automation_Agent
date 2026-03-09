import pytest
import json
from unittest.mock import patch, MagicMock

# Import the tools to be tested
from tools.rag_tools import (
    query_resume_match,
    get_resume_context,
    embed_job_description
)

@patch('tools.rag_tools.rag_api.select_resume')
def test_query_resume_match_success(mock_select_resume):
    # Mock the RAG engine's return payload
    mock_select_resume.return_value = {
        "top_resume_id": "AarjunGen.pdf",
        "top_score": 0.88,
        "candidates": [
            {
                "final_score": 0.88,
                "recommended_chunks": [
                    {"document": "5+ years formatting LLM structured output"},
                    {"document": "Led backend architecture for multi-agent systems"}
                ]
            }
        ]
    }
    
    result_str = query_resume_match.func(
        job_description="We are looking for a Senior AI Agent Engineer with LLM architecture experience.",
        job_title="Senior AI Agent Engineer",
        required_skills="Python, LLM, AgentOps, Postgres"
    )
    
    result = json.loads(result_str)
    
    # Verify the structure matches expected schema
    assert result["resume_suggested"] == "AarjunGen.pdf"
    assert result["similarity_score"] == 0.88
    assert result["fit_score"] == 0.88
    assert isinstance(result["talking_points"], list)
    assert len(result["talking_points"]) == 2
    assert "LLM structured output" in result["talking_points"][0]

@patch('tools.rag_tools.rag_api.select_resume')
@patch('tools.rag_tools.os.getenv')
def test_query_resume_match_fallback_on_error(mock_getenv, mock_select_resume):
    # Mock environment default and raise error on RAG
    mock_getenv.return_value = "Aarjun_Fallback.pdf"
    mock_select_resume.side_effect = Exception("ChromaDB connection timeout")
    
    result_str = query_resume_match.func(
        job_description="Description",
        job_title="Title",
        required_skills="Skills"
    )
    
    result = json.loads(result_str)
    
    # Verify it falls back safely without crashing
    assert "error" in result
    assert result["error"] == "rag_unavailable"
    assert result["fallback_resume"] == "Aarjun_Fallback.pdf"

@patch('tools.rag_tools.rag_api.get_rag_context')
def test_get_resume_context(mock_get_rag_context):
    mock_get_rag_context.return_value = {
        "top_chunks": [
            {"document": "Extensive experience with FastAPI and Postgres."},
            {"document": "Deployed applications using Docker and AWS."}
        ]
    }
    
    context = get_resume_context.func(
        resume_filename="ignore_me.pdf",
        job_description="Looking for Python backend with Docker."
    )
    
    assert "[CHUNK 1] Extensive experience" in context
    assert "[CHUNK 2] Deployed applications" in context
