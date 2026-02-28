"""
Notion API client for AI Job Application Agent.

This module provides a low-level Notion API client for creating and managing
pages in the Job Tracker and Applications databases. Used exclusively by
notion_tools.py.

All Notion property names and types are aligned with the existing Chrome
extension implementation.
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional

import requests

# Module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Notion API configuration
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_APPLICATIONS_DB_ID = os.getenv("NOTION_APPLICATIONS_DB_ID")
NOTION_JOB_TRACKER_DB_ID = os.getenv("NOTION_JOB_TRACKER_DB_ID")
NOTION_API_VERSION = "2022-06-28"
NOTION_BASE_URL = "https://api.notion.com/v1"

__all__ = ["NotionClient"]


class NotionClient:
    """
    Low-level Notion API client.

    Provides methods for creating pages in Job Tracker and Applications databases,
    updating page properties, and querying databases with pagination support.
    """

    def __init__(self) -> None:
        """Initialize the Notion client."""
        if not NOTION_API_KEY:
            raise RuntimeError(
                "NOTION_API_KEY not configured. Set it in narad.env to use Notion integration."
            )
        logger.info("NotionClient initialized")

    def _headers(self) -> Dict[str, str]:
        """
        Get HTTP headers for Notion API requests.

        Returns:
            Dict with Authorization, Content-Type, and Notion-Version headers.

        Raises:
            RuntimeError: If NOTION_API_KEY is not set.
        """
        if not NOTION_API_KEY:
            raise RuntimeError("NOTION_API_KEY not configured")

        return {
            "Authorization": f"Bearer {NOTION_API_KEY}",
            "Content-Type": "application/json",
            "Notion-Version": NOTION_API_VERSION,
        }

    def _request(
        self, method: str, endpoint: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the Notion API with retry logic.

        Args:
            method: HTTP method (GET, POST, PATCH).
            endpoint: API endpoint path (e.g., "/pages").
            payload: Request body as dictionary.

        Returns:
            Response JSON as dictionary.

        Raises:
            RuntimeError: If request fails after all retries.
        """
        url = f"{NOTION_BASE_URL}{endpoint}"
        max_retries = 3
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=self._headers(),
                    json=payload,
                    timeout=30,
                )

                # Handle rate limiting
                if response.status_code == 429:
                    logger.warning(
                        f"Notion API rate limit hit (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(5)
                    continue

                # Handle non-200 responses
                if response.status_code != 200:
                    error_body = response.text
                    logger.error(
                        f"Notion API error {response.status_code}: {error_body}"
                    )
                    raise RuntimeError(
                        f"Notion API returned {response.status_code}: {error_body}"
                    )

                return response.json()

            except requests.RequestException as e:
                last_exception = e
                if attempt < max_retries - 1:
                    sleep_time = 2**attempt
                    logger.warning(
                        f"Notion API request failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {sleep_time}s..."
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        f"Notion API request failed after {max_retries} attempts: {e}"
                    )

        raise RuntimeError(f"Notion API request failed: {last_exception}")

    def create_job_tracker_page(
        self,
        title: str,
        company: str,
        job_url: str,
        stage: str,
        date_applied: str,
        platform: str,
        applied_via: str,
        ctc: str,
        notes: str,
        job_type: str,
        location: str,
        resume_used: str,
    ) -> Dict[str, Any]:
        """
        Create a new page in the Job Tracker database.

        Args:
            title: Job title.
            company: Company name.
            job_url: Job posting URL.
            stage: Application stage (Applied, In Review, Interview, Offer, Rejected).
            date_applied: Date applied in ISO format (YYYY-MM-DD).
            platform: Job platform name.
            applied_via: Application method (Auto or Manual).
            ctc: Compensation/salary information.
            notes: Additional notes.
            job_type: Job type (full-time, contract, etc.).
            location: Job location.
            resume_used: Resume variant used.

        Returns:
            Full Notion API response dictionary.
        """
        if not NOTION_JOB_TRACKER_DB_ID:
            raise RuntimeError("NOTION_JOB_TRACKER_DB_ID not configured")

        payload = {
            "parent": {"database_id": NOTION_JOB_TRACKER_DB_ID},
            "properties": {
                "Job Title": {"title": [{"text": {"content": title[:2000]}}]},
                "Company": {"rich_text": [{"text": {"content": company[:2000]}}]},
                "Job URL": {"url": job_url[:2000] if job_url else None},
                "Stage": {"select": {"name": stage}},
                "Date Applied": {"date": {"start": date_applied}},
                "Platform": {"select": {"name": platform}},
                "Applied Via": {"select": {"name": applied_via}},
                "CTC": {"rich_text": [{"text": {"content": ctc[:2000]}}]},
                "Notes": {"rich_text": [{"text": {"content": notes[:2000]}}]},
                "Job Type": {"select": {"name": job_type}},
                "Location": {"rich_text": [{"text": {"content": location[:2000]}}]},
                "Resume Used": {"rich_text": [{"text": {"content": resume_used[:2000]}}]},
            },
        }

        logger.info(f"Creating Job Tracker page: {company} - {title}")
        return self._request("POST", "/pages", payload)

    def create_applications_page(
        self,
        title: str,
        company: str,
        job_url: str,
        deadline: str,
        platform: str,
        status: str,
        ctc: str,
        priority: str,
        fit_score: float,
        job_type: str,
        location: str,
        notes: str,
        resume_suggested: str,
    ) -> Dict[str, Any]:
        """
        Create a new page in the Applications database (manual queue).

        Args:
            title: Job title.
            company: Company name.
            job_url: Job posting URL.
            deadline: Application deadline in ISO format (YYYY-MM-DD).
            platform: Job platform name.
            status: Application status (Queued, In Review, Applied, Skipped).
            ctc: Compensation/salary information.
            priority: Priority level (High, Medium, Low).
            fit_score: Fit score (0.0 to 1.0).
            job_type: Job type (full-time, contract, etc.).
            location: Job location.
            notes: Additional notes.
            resume_suggested: Suggested resume variant.

        Returns:
            Full Notion API response dictionary.
        """
        if not NOTION_APPLICATIONS_DB_ID:
            raise RuntimeError("NOTION_APPLICATIONS_DB_ID not configured")

        # Build properties - handle optional deadline
        properties: Dict[str, Any] = {
            "Job Title": {"title": [{"text": {"content": title[:2000]}}]},
            "Company": {"rich_text": [{"text": {"content": company[:2000]}}]},
            "Job URL": {"url": job_url[:2000] if job_url else None},
            "Platform": {"select": {"name": platform}},
            "Status": {"select": {"name": status}},
            "CTC": {"rich_text": [{"text": {"content": ctc[:2000]}}]},
            "Priority": {"select": {"name": priority}},
            "Fit Score": {"number": round(fit_score, 2)},
            "Job Type": {"select": {"name": job_type}},
            "Location": {"rich_text": [{"text": {"content": location[:2000]}}]},
            "Notes": {"rich_text": [{"text": {"content": notes[:2000]}}]},
            "Resume Suggested": {"rich_text": [{"text": {"content": resume_suggested[:2000]}}]},
        }

        # Only add deadline if provided
        if deadline:
            properties["Application Deadline"] = {"date": {"start": deadline}}

        payload = {
            "parent": {"database_id": NOTION_APPLICATIONS_DB_ID},
            "properties": properties,
        }

        logger.info(f"Creating Applications page: {company} - {title}")
        return self._request("POST", "/pages", payload)

    def update_page_status(self, page_id: str, status: str) -> Dict[str, Any]:
        """
        Update the Status property of a Notion page.

        Args:
            page_id: Notion page ID.
            status: New status value.

        Returns:
            Updated page response dictionary.
        """
        payload = {"properties": {"Status": {"select": {"name": status}}}}

        logger.info(f"Updating page {page_id} status to: {status}")
        return self._request("PATCH", f"/pages/{page_id}", payload)

    def query_database(
        self,
        database_id: str,
        filter_payload: Optional[Dict[str, Any]] = None,
        page_size: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Query a Notion database with automatic pagination.

        Args:
            database_id: Notion database ID.
            filter_payload: Optional filter object (Notion filter syntax).
            page_size: Number of results per page (max 100).

        Returns:
            Flat list of all page objects matching the query.
        """
        all_results: List[Dict[str, Any]] = []
        has_more = True
        start_cursor = None

        payload: Dict[str, Any] = {"page_size": min(page_size, 100)}

        if filter_payload:
            payload["filter"] = filter_payload

        while has_more:
            if start_cursor:
                payload["start_cursor"] = start_cursor

            response = self._request("POST", f"/databases/{database_id}/query", payload)

            results = response.get("results", [])
            all_results.extend(results)

            has_more = response.get("has_more", False)
            start_cursor = response.get("next_cursor")

            logger.debug(
                f"Query page fetched: {len(results)} results, has_more={has_more}"
            )

        logger.info(f"Query completed: {len(all_results)} total results")
        return all_results

    def health_check(self) -> Dict[str, Any]:
        """
        Check Notion API connection health.

        Returns:
            Dictionary with connection status, bot name, and error (if any).
            Never raises - always returns a dict.
        """
        try:
            response = self._request("GET", "/users/me", None)

            bot_name = response.get("name", "Unknown")
            logger.info(f"Notion health check successful: {bot_name}")

            return {"connected": True, "bot_name": bot_name, "error": None}

        except Exception as e:
            logger.error(f"Notion health check failed: {e}")
            return {"connected": False, "bot_name": None, "error": str(e)}
