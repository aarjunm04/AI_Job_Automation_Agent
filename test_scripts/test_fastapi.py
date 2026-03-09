import pytest
import os
from unittest.mock import patch, MagicMock

# Set environment variables for testing so local DB tries to connect 
# without failing if env is missing
os.environ["FASTAPI_API_KEY"] = "test-token"
os.environ["RAG_SERVER_API_KEY"] = "test-api"
os.environ["LOCAL_POSTGRES_URL"] = "postgresql://dummy"
os.environ["ACTIVE_DB"] = "local"

from fastapi.testclient import TestClient
from api.api_server import app

# We initialize the client outside, but we need to mock lifespan dependencies sometimes.
# Since app lifespan wraps startup/shutdown, we'll test without triggering DB errors.

client = TestClient(app)

@patch('api.api_server.get_run_stats')
def test_status_endpoint_healthy(mock_get_run_stats):
    mock_get_run_stats.return_value = '{}'
    
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["db_connected"] is True

@patch('api.api_server.get_run_stats')
def test_status_endpoint_degraded(mock_get_run_stats):
    mock_get_run_stats.side_effect = Exception("Database connection failed")
    
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["db_connected"] is False

@patch('api.api_server.get_run_stats')
@patch('api.api_server.update_application_status')
@patch('api.api_server.log_event')
def test_manual_apply_queue(mock_log, mock_update, mock_get_stats):
    mock_get_stats.return_value = "{}"
    
    payload = {
        "job_post_id": "job-1234",
        "user_id": "user-456",
        "resume_filename": "AarjunGen.pdf",
        "notes": "Good opportunity"
    }
    
    response = client.post(
        "/apply/manual",
        json=payload,
        headers={"X-API-Key": "test-api"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["queued"] is True
    assert data["job_post_id"] == "job-1234"
    
    # Verify the background update and log calls happened
    mock_update.assert_called_once_with(
        application_id="job-1234",
        new_status="manual_queued",
        resume_used="AarjunGen.pdf",
        notes="Good opportunity"
    )
    mock_log.assert_called_once()
