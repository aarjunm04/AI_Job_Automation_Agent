import pytest
import json
from unittest.mock import patch, MagicMock

# Import the tools to be tested
from tools.postgres_tools import (
    upsert_job_post,
    get_run_stats,
    get_platform_config,
    create_run_batch
)

@patch('tools.postgres_tools._get_conn')
def test_upsert_job_post_success(mock_get_conn):
    # Setup mock connection and cursor
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_get_conn.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor
    
    # Mock the database response
    mock_cursor.fetchone.return_value = {"id": "1234-abcd", "inserted": True}
    
    # Call the tool function (using .func to bypass CrewAI wrapper if present)
    result_str = upsert_job_post.func(
        run_batch_id="batch-123",
        source_platform="test_platform",
        title="Test Job",
        company="Test Company",
        url="http://test.com/job"
    )
    
    result = json.loads(result_str)
    
    # Assertions
    assert "error" not in result
    assert result["job_post_id"] == "1234-abcd"
    assert result["action"] == "inserted"
    mock_cursor.execute.assert_called_once()
    mock_conn.commit.assert_called_once()

@patch('tools.postgres_tools._get_conn')
def test_get_platform_config_found(mock_get_conn):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_get_conn.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor
    
    # Mock existing config
    mock_cursor.fetchone.return_value = {
        "platform": "ycombinator",
        "max_per_run": 30,
        "max_per_day": 100,
        "max_concurrent_sessions": 2
    }
    
    result_str = get_platform_config.func(platform="ycombinator")
    config = json.loads(result_str)
    
    assert config["platform"] == "ycombinator"
    assert config["max_per_run"] == 30
    assert config["max_concurrent_sessions"] == 2

@patch('tools.postgres_tools._get_conn')
def test_get_platform_config_default(mock_get_conn):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_get_conn.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor
    
    # Mock no existing config
    mock_cursor.fetchone.return_value = None
    
    result_str = get_platform_config.func(platform="unknown_platform")
    config = json.loads(result_str)
    
    assert config["platform"] == "unknown_platform"
    assert config["max_per_run"] == 50  # Default fallback
    assert config["max_per_day"] == 100
    
@patch('tools.postgres_tools._get_conn')
def test_create_run_batch(mock_get_conn):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_get_conn.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor
    
    mock_cursor.fetchone.return_value = {"id": "batch-1", "run_date": "2026-03-09"}
    
    result_str = create_run_batch.func(run_index_in_week=1)
    result = json.loads(result_str)
    
    assert result["run_batch_id"] == "batch-1"
    assert result["run_date"] == "2026-03-09"
    mock_conn.commit.assert_called_once()
