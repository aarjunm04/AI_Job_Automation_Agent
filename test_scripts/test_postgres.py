import json
import logging
import time
import unittest
from unittest.mock import patch, MagicMock, call

import psycopg2

# Import the tools to be tested
from tools.postgres_tools import (
    upsert_job_post,
    get_run_stats,
    get_platform_config,
    create_run_batch,
    _fetch_user_config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared mock data for the new Postgres JSONB architecture
# ---------------------------------------------------------------------------

_MOCK_PLATFORM_SETTINGS: dict = {
    "platform_limits": {
        "linkedin": {"max_per_run": 40, "rate_limit_per_request_seconds": 3},
        "indeed": {"max_per_run": 40, "rate_limit_per_request_seconds": 3},
        "remoteok": {"max_per_run": 30, "rate_limit_per_request_seconds": 2},
        "himalayas": {"max_per_run": 30, "rate_limit_per_request_seconds": 2},
        "wellfound": {"max_per_run": 20, "rate_limit_per_request_seconds": 4},
        "wwr": {"max_per_run": 20, "rate_limit_per_request_seconds": 4},
        "yc": {"max_per_run": 20, "rate_limit_per_request_seconds": 3},
        "arc": {"max_per_run": 15, "rate_limit_per_request_seconds": 4},
        "turing": {"max_per_run": 15, "rate_limit_per_request_seconds": 4},
        "crossover": {"max_per_run": 10, "rate_limit_per_request_seconds": 5},
    },
    "scoring_weights": {
        "skills_match": 0.35,
        "experience_match": 0.25,
        "title_match": 0.20,
        "location_match": 0.10,
        "company_score": 0.10,
    },
    "job_filters": {
        "min_fit_score": 0.40,
        "exclude_keywords": ["senior 10+ years", "principal", "director"],
        "required_keywords": ["remote", "python"],
        "job_types": ["fulltime", "contract"],
    },
}

_MOCK_USER_SETTINGS: dict = {
    "default_resume": "AarjunGen.pdf",
    "dry_run": False,
    "auto_apply_enabled": True,
    "jobs_per_run_target": 150,
    "jobs_per_run_minimum": 100,
    "search_query": "AI ML Data Science Automation Engineer remote",
    "log_level": "INFO",
}


# ---------------------------------------------------------------------------
# Existing tests — mocks updated for JSONB architecture
# ---------------------------------------------------------------------------


@patch("tools.postgres_tools._get_conn")
def test_upsert_job_post_success(mock_get_conn: MagicMock) -> None:
    """Test that upsert_job_post inserts a job and returns the correct payload."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_get_conn.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor

    mock_cursor.fetchone.return_value = {"id": "1234-abcd", "inserted": True}

    result_str = upsert_job_post.func(
        run_batch_id="batch-123",
        source_platform="test_platform",
        title="Test Job",
        company="Test Company",
        url="http://test.com/job",
    )

    result = json.loads(result_str)

    assert "error" not in result
    assert result["job_post_id"] == "1234-abcd"
    assert result["action"] == "inserted"
    mock_cursor.execute.assert_called_once()
    mock_conn.commit.assert_called_once()


@patch("tools.postgres_tools._fetch_user_config")
def test_get_platform_config_found(mock_fetch: MagicMock) -> None:
    """Test get_platform_config returns correct JSONB data for a known platform."""
    mock_fetch.return_value = {
        "user_settings": _MOCK_USER_SETTINGS,
        "platform_settings": _MOCK_PLATFORM_SETTINGS,
    }

    result_str = get_platform_config.func(platform="linkedin")
    config = json.loads(result_str)

    assert config["max_per_run"] == 40
    assert config["rate_limit_per_request_seconds"] == 3


@patch("tools.postgres_tools._fetch_user_config")
def test_get_platform_config_default(mock_fetch: MagicMock) -> None:
    """Test get_platform_config returns safe defaults for an unknown platform."""
    mock_fetch.return_value = {
        "user_settings": _MOCK_USER_SETTINGS,
        "platform_settings": _MOCK_PLATFORM_SETTINGS,
    }

    result_str = get_platform_config.func(platform="unknown_platform")
    config = json.loads(result_str)

    assert config["max_per_run"] == 20
    assert config["rate_limit_per_request_seconds"] == 3


@patch("tools.postgres_tools._fetch_user_config")
def test_get_platform_config_crossover(mock_fetch: MagicMock) -> None:
    """Test get_platform_config returns correct data for crossover platform."""
    mock_fetch.return_value = {
        "user_settings": _MOCK_USER_SETTINGS,
        "platform_settings": _MOCK_PLATFORM_SETTINGS,
    }

    result_str = get_platform_config.func(platform="crossover")
    config = json.loads(result_str)

    assert config["max_per_run"] == 10
    assert config["rate_limit_per_request_seconds"] == 5


@patch("tools.postgres_tools._get_conn")
def test_create_run_batch(mock_get_conn: MagicMock) -> None:
    """Test create_run_batch inserts a run session and returns batch info."""
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


# ---------------------------------------------------------------------------
# FIX A2: New test class for _fetch_user_config helper
# ---------------------------------------------------------------------------


class TestFetchUserConfig(unittest.TestCase):
    """Tests for the _fetch_user_config() Postgres JSONB helper.

    All tests validate the retry and fail-soft semantics introduced as part
    of the SSoT migration.
    """

    def setUp(self) -> None:
        """Start patcher for _get_conn and time.sleep."""
        self.patcher_conn = patch("tools.postgres_tools._get_conn")
        self.mock_get_conn = self.patcher_conn.start()
        self.patcher_sleep = patch("tools.postgres_tools.time.sleep")
        self.mock_sleep = self.patcher_sleep.start()

    def tearDown(self) -> None:
        """Stop all patchers."""
        self.patcher_sleep.stop()
        self.patcher_conn.stop()

    def test_fetch_user_config_success(self) -> None:
        """Happy path: returns user_settings and platform_settings from DB."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        self.mock_get_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchone.return_value = {
            "user_settings": _MOCK_USER_SETTINGS,
            "platform_settings": _MOCK_PLATFORM_SETTINGS,
        }

        result = _fetch_user_config()

        assert "user_settings" in result
        assert "platform_settings" in result
        assert result["user_settings"]["default_resume"] == "AarjunGen.pdf"
        assert result["platform_settings"]["scoring_weights"]["skills_match"] == 0.35

    def test_fetch_user_config_db_failure_retries(self) -> None:
        """Retries on OperationalError and succeeds on the 3rd attempt."""
        mock_conn_ok = MagicMock()
        mock_cursor = MagicMock()
        mock_conn_ok.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = {
            "user_settings": _MOCK_USER_SETTINGS,
            "platform_settings": _MOCK_PLATFORM_SETTINGS,
        }

        # First 2 calls raise OperationalError, 3rd succeeds
        self.mock_get_conn.side_effect = [
            psycopg2.OperationalError("conn refused"),
            psycopg2.OperationalError("conn refused"),
            mock_conn_ok,
        ]

        result = _fetch_user_config()

        assert self.mock_get_conn.call_count == 3
        assert result["user_settings"]["default_resume"] == "AarjunGen.pdf"

    def test_fetch_user_config_all_retries_exhausted(self) -> None:
        """All 3 retries fail — returns empty dict (fail-soft, never raises)."""
        self.mock_get_conn.side_effect = psycopg2.OperationalError("conn refused")

        result = _fetch_user_config()

        assert self.mock_get_conn.call_count == 3
        assert result == {}

    def test_fetch_user_config_null_response(self) -> None:
        """DB row returns None — returns empty dict, does not raise."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        self.mock_get_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchone.return_value = None

        result = _fetch_user_config()

        assert result == {}


if __name__ == "__main__":
    unittest.main()
