"""
Notion API client for AI Job Application Agent.

This module provides a low-level Notion API client for creating and managing
pages in the Job Tracker and Applications databases. Used exclusively by
notion_tools.py.

All Notion property names and types are aligned with the existing Chrome
extension implementation.

Includes async methods for run reports and alerts.
"""

from __future__ import annotations

import asyncio
import os
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import httpx
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

__all__ = ["NotionClient", "FinalReport"]


@dataclass
class FinalReport:
    """Data structure for pipeline run report."""
    run_batch_id: str
    run_date: str
    jobs_discovered: int
    jobs_scored: int
    jobs_auto_applied: int
    jobs_manual_queued: int
    jobs_failed: int
    total_cost_usd: float
    duration_minutes: float
    success: bool
    error_summary: Optional[str] = None
    top_applied_jobs: Optional[List[Dict[str, Any]]] = None


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
                "NOTION_API_KEY not configured. Set it in java.env to use Notion integration."
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
                "Applied Via": {"select": {"name": applied_via if applied_via else "Auto-Apply"}},
                "CTC": {"rich_text": [{"text": {"content": str(ctc)[:2000] if ctc else ""}}]},
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

    # =========================================================================
    # ASYNC METHODS — For use by main.py and async pipeline components
    # =========================================================================

    async def _async_request(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Make an async HTTP request to the Notion API with retry logic.

        Args:
            method: HTTP method (GET, POST, PATCH).
            endpoint: API endpoint path (e.g., "/pages").
            payload: Request body as dictionary.
            retries: Number of retry attempts.

        Returns:
            Response JSON as dictionary.

        Raises:
            RuntimeError: If request fails after all retries.
        """
        url = f"{NOTION_BASE_URL}{endpoint}"
        headers = self._headers()
        last_exception: Optional[Exception] = None

        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    if method == "GET":
                        response = await client.get(url, headers=headers)
                    elif method == "POST":
                        response = await client.post(url, headers=headers, json=payload)
                    elif method == "PATCH":
                        response = await client.patch(url, headers=headers, json=payload)
                    else:
                        raise ValueError(f"Unsupported method: {method}")

                    # Handle rate limiting
                    if response.status_code == 429:
                        wait_time = int(response.headers.get("Retry-After", 5))
                        logger.warning(f"Notion rate limit hit, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status_code not in (200, 201):
                        raise RuntimeError(
                            f"Notion API {response.status_code}: {response.text}"
                        )

                    return response.json()

            except Exception as e:
                last_exception = e
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        f"Notion request failed (attempt {attempt + 1}/{retries}): {e}"
                    )
                    await asyncio.sleep(wait)

        raise RuntimeError(f"Notion API request failed: {last_exception}")

    async def post_run_report(self, report: FinalReport) -> Dict[str, Any]:
        """
        Post a complete pipeline run report to Notion.

        Creates a page in the Run Reports database with full statistics
        and a markdown table of top applied jobs.

        Args:
            report: FinalReport dataclass with all run statistics.

        Returns:
            Notion API response dictionary.
        """
        database_id = NOTION_RUN_REPORTS_DB_ID or NOTION_JOB_TRACKER_DB_ID
        if not database_id:
            raise RuntimeError("No Notion database configured for run reports")

        # Build status
        status = "Success" if report.success else "Failed"
        if report.error_summary:
            status = "Failed"

        # Build title: "Run 2026-03-13 — 45/150 applied"
        title = f"Run {report.run_date} — {report.jobs_auto_applied}/{report.jobs_discovered} applied"

        # Build body content with markdown table
        body_blocks: List[Dict[str, Any]] = []

        # Summary paragraph
        summary = (
            f"**Pipeline Run Summary**\n\n"
            f"- Jobs Discovered: {report.jobs_discovered}\n"
            f"- Jobs Scored: {report.jobs_scored}\n"
            f"- Jobs Applied: {report.jobs_auto_applied}\n"
            f"- Jobs Queued: {report.jobs_manual_queued}\n"
            f"- Jobs Failed: {report.jobs_failed}\n"
            f"- Total Cost: ${report.total_cost_usd:.4f}\n"
            f"- Duration: {report.duration_minutes:.1f} minutes\n"
        )

        body_blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": summary}}]
            }
        })

        # Add top jobs table if available
        if report.top_applied_jobs:
            # Table header
            body_blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "Top Applied Jobs"}}]
                }
            })

            # Build markdown-style table as text (Notion tables are complex)
            table_text = "| Title | Company | Score | Platform |\n|-------|---------|-------|----------|\n"
            for job in report.top_applied_jobs[:10]:
                table_text += (
                    f"| {job.get('title', 'N/A')[:30]} | "
                    f"{job.get('company', 'N/A')[:20]} | "
                    f"{job.get('fit_score', 0):.2f} | "
                    f"{job.get('platform', 'N/A')} |\n"
                )

            body_blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": table_text}}]
                }
            })

        # Add error summary if any
        if report.error_summary:
            body_blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "Errors"}}]
                }
            })
            body_blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": report.error_summary}}]
                }
            })

        payload = {
            "parent": {"database_id": database_id},
            "properties": {
                "Job Title": {"title": [{"text": {"content": title}}]},
                "Stage": {"select": {"name": status}},
                "Date Applied": {"date": {"start": report.run_date}},
                "Platform": {"select": {"name": "Auto Agent"}},
                "Applied Via": {"select": {"name": "Auto-Apply"}},
                "Notes": {"rich_text": [{"text": {"content": f"RunID: {report.run_batch_id}"}}]},
            },
            "children": body_blocks,
        }

        logger.info(f"Posting run report: {title}")
        return await self._async_request("POST", "/pages", payload)

    async def post_run_report_simple(
        self,
        run_batch_id: str,
        jobs_discovered: int,
        jobs_applied: int,
        jobs_queued: int,
        jobs_failed: int,
        cost_usd: float,
        duration_mins: float,
    ) -> Dict[str, Any]:
        """
        Post a simplified run report to Notion.

        Args:
            run_batch_id: UUID of the run batch.
            jobs_discovered: Total jobs found.
            jobs_applied: Jobs successfully applied.
            jobs_queued: Jobs sent to manual queue.
            jobs_failed: Jobs that failed.
            cost_usd: Total LLM cost.
            duration_mins: Run duration in minutes.

        Returns:
            Notion API response.
        """
        report = FinalReport(
            run_batch_id=run_batch_id,
            run_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            jobs_discovered=jobs_discovered,
            jobs_scored=jobs_discovered,
            jobs_auto_applied=jobs_applied,
            jobs_manual_queued=jobs_queued,
            jobs_failed=jobs_failed,
            total_cost_usd=cost_usd,
            duration_minutes=duration_mins,
            success=jobs_failed == 0,
        )
        return await self.post_run_report(report)

    async def post_alert(
        self,
        message: str,
        level: str = "WARNING",
        run_batch_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Post an alert to the Notion Alerts database.

        Args:
            message: Alert message text.
            level: Alert level (INFO, WARNING, CRITICAL).
            run_batch_id: Optional run batch ID for context.

        Returns:
            Notion API response dictionary.
        """
        database_id = NOTION_ALERTS_DB_ID
        if not database_id:
            logger.warning("NOTION_ALERTS_DB_ID not configured, skipping alert")
            return {"skipped": True, "reason": "no_alerts_db"}

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        title = f"🚨 {level}: Pipeline Alert"

        if run_batch_id:
            message = f"{message}\n\nRun ID: {run_batch_id}"

        payload = {
            "parent": {"database_id": database_id},
            "properties": {
                "Title": {"title": [{"text": {"content": title}}]},
                "Level": {"select": {"name": level}},
                "Timestamp": {"date": {"start": timestamp}},
                "Message": {"rich_text": [{"text": {"content": message[:2000]}}]},
            },
        }

        logger.info(f"Posting alert: {level} - {message[:100]}...")
        return await self._async_request("POST", "/pages", payload)

    async def async_health_check(self) -> Dict[str, Any]:
        """
        Async version of health check.

        Returns:
            Dictionary with connection status.
        """
        try:
            response = await self._async_request("GET", "/users/me")
            bot_name = response.get("name", "Unknown")
            logger.info(f"Notion async health check successful: {bot_name}")
            return {"connected": True, "bot_name": bot_name, "error": None}
        except Exception as e:
            logger.error(f"Notion async health check failed: {e}")
            return {"connected": False, "bot_name": None, "error": str(e)}
