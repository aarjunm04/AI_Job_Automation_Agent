"""LinkedIn Easy Apply — detection, metadata extraction, and reroute module.

LinkedIn Easy Apply is NOT automated due to LinkedIn Terms of Service
compliance. This module detects LinkedIn jobs, extracts all available
metadata from the page, checks for external apply links, and routes
the job accordingly:

- **Easy Apply jobs**: Route to manual queue with extracted metadata.
- **External apply jobs**: Return the external ATS URL in ``proof_value``
  so the pipeline can re-route to the correct platform module
  (Greenhouse, Lever, Workday, etc.).
- **Non-LinkedIn pages**: Return ``UNKNOWN_ATS`` error.

This module NEVER clicks the Easy Apply button and NEVER interacts with
LinkedIn's application flow. All DOM inspection is read-only.

User profile keys are accepted but not used for form filling — only
for metadata enrichment in the reroute reason string.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any

from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from auto_apply.platforms.base_platform import (
    BasePlatformApply,
    ApplyResult,
)

logger = logging.getLogger(__name__)

__all__ = ["LinkedInEasyApply"]


class LinkedInEasyApply(BasePlatformApply):
    """LinkedIn Easy Apply detection, metadata extraction, and reroute.

    This module does NOT automate LinkedIn applications. It serves as
    the routing decision point for LinkedIn-detected jobs:

    1. Extracts all available job metadata from the LinkedIn page.
    2. Checks for external apply links (redirects to Greenhouse, Lever,
       Workday, etc.).
    3. Routes Easy Apply jobs to the manual queue with a clear reason.

    Steps:
        1. Navigate + verify LinkedIn page.
        2. Detect apply button type + extract metadata + route.
    """

    PLATFORM_NAME: str = "linkedin"
    STEPS_TOTAL: int = 2

    async def apply(self) -> ApplyResult:
        """Detect LinkedIn job type and route appropriately.

        For Easy Apply jobs: extract metadata and route to manual queue.
        For external apply jobs: extract external URL and return it in
        ``proof_value`` so the pipeline can re-route to the correct
        platform module (``apply_tools.py`` reads ``proof_value`` and
        dispatches to the right platform class based on URL pattern).
        For non-LinkedIn pages: return ``UNKNOWN_ATS`` error.

        Returns:
            ApplyResult — always ``reroute_to_manual=True`` for Easy
            Apply. For external links: ``reroute_to_manual=False``,
            ``proof_type="success_url"``,
            ``proof_value=external_apply_url``.
        """
        self.steps_completed = 0

        # ── Step 1: Navigate + Verify LinkedIn ──
        step1_result: ApplyResult | None = await self._step_navigate()
        if step1_result is not None:
            return step1_result

        # ── Step 2: Detect + Extract + Route ──
        return await self._step_detect_and_route()

    # ------------------------------------------------------------------
    # Step 1: Navigate + Verify
    # ------------------------------------------------------------------

    async def _step_navigate(self) -> ApplyResult | None:
        """Navigate to the LinkedIn job page and verify URL.

        Returns:
            ApplyResult on failure, None on success to proceed.
        """
        try:
            job_url: str = self.job_meta.get("job_url", "")
            current_url: str = self.page.url
            if not current_url.startswith(job_url[:30]):
                await self.page.goto(
                    job_url,
                    wait_until="domcontentloaded",
                    timeout=20000,
                )
            await self.page.wait_for_load_state(
                "domcontentloaded", timeout=20000
            )
            await asyncio.sleep(1.5)  # LinkedIn SPA settle time

            # Verify LinkedIn URL
            if "linkedin.com" not in self.page.url:
                return self._build_result(
                    success=False,
                    error_code="UNKNOWN_ATS",
                    reroute_to_manual=True,
                    reroute_reason="Not a LinkedIn page",
                )

            self.steps_completed = 1
            return None

        except PlaywrightTimeoutError:
            return self._build_result(
                success=False,
                error_code="TIMEOUT",
                reroute_to_manual=True,
                reroute_reason="LinkedIn page load timeout",
            )
        except Exception as e:
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=f"LinkedIn navigation error: {str(e)}",
            )

    # ------------------------------------------------------------------
    # Step 2: Detect + Extract + Route
    # ------------------------------------------------------------------

    async def _step_detect_and_route(self) -> ApplyResult:
        """Detect apply button type, extract metadata, and route.

        Returns:
            ApplyResult with routing decision.
        """
        try:
            # Extract all available metadata from the job page
            metadata: Dict[str, str] = (
                await self._extract_linkedin_metadata()
            )
            self.logger.info(
                "LinkedIn job metadata extracted: title=%s company=%s",
                metadata.get("title", ""),
                metadata.get("company", ""),
            )

            # Check for external apply link first (higher priority)
            external_url: str = await self._get_external_apply_url()
            if external_url:
                self.logger.info(
                    "LinkedIn external apply URL found: %s", external_url
                )
                self.steps_completed = 2
                # proof_value=external_url is used by apply_tools.py
                # to re-route this job to the correct platform module.
                # ApplyAgent reads proof_value and dispatches to the
                # right platform apply class based on URL pattern.
                return self._build_result(
                    success=True,
                    proof_type="success_url",
                    proof_value=external_url,
                    proof_confidence=1.0,
                    reroute_to_manual=False,
                    reroute_reason=None,
                )

            # Check for Easy Apply button
            easy_apply = await self.page.query_selector(
                "button.jobs-apply-button[aria-label*='Easy Apply'], "
                "button[data-control-name='jobdetails_topcard_inapply'], "
                "div.jobs-easy-apply-content"
            )

            if easy_apply:
                self.steps_completed = 2
                self.logger.info(
                    "LinkedIn Easy Apply detected — routing to manual "
                    "queue (ToS compliance)"
                )
                return self._build_result(
                    success=False,
                    error_code=None,
                    reroute_to_manual=True,
                    reroute_reason=(
                        "LinkedIn Easy Apply not automated "
                        "(ToS compliance). "
                        f"Job: {metadata.get('title', 'Unknown')} at "
                        f"{metadata.get('company', 'Unknown')}. "
                        "Manual queue: use talking points from "
                        "/match endpoint."
                    ),
                )

            # Neither Easy Apply nor external link found
            self.steps_completed = 2
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=(
                    "LinkedIn page loaded but no apply button found. "
                    "Job may be expired or require LinkedIn Premium."
                ),
            )

        except Exception as e:
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=f"LinkedIn detection error: {str(e)}",
            )

    # ------------------------------------------------------------------
    # Metadata Extraction
    # ------------------------------------------------------------------

    async def _extract_linkedin_metadata(self) -> Dict[str, str]:
        """Extract all available job metadata from the LinkedIn page.

        Scrapes title, company, location, job type, seniority,
        description, posted date, and applicant count. All fields are
        non-fatal — missing fields return empty string.

        Returns:
            Dict with keys: ``title``, ``company``, ``location``,
            ``job_type``, ``posted_date``, ``applicant_count``,
            ``description``.
        """
        metadata: Dict[str, str] = {}

        async def _safe_text(selector: str) -> str:
            """Extract inner text from first matching element safely."""
            try:
                el = await self.page.query_selector(selector)
                if el:
                    return (await el.inner_text()).strip()
            except Exception:
                pass
            return ""

        metadata["title"] = await _safe_text(
            "h1.t-24, "
            "h1[class*='jobs-unified-top-card__job-title']"
        )
        metadata["company"] = await _safe_text(
            "a[class*='topcard__org-name-link'], "
            "span[class*='jobs-unified-top-card__company-name'] a"
        )
        metadata["location"] = await _safe_text(
            "span[class*='topcard__flavor--bullet'], "
            "span[class*='jobs-unified-top-card__bullet']"
        )
        metadata["job_type"] = await _safe_text(
            "span[class*='jobs-unified-top-card__workplace-type']"
        )
        metadata["posted_date"] = await _safe_text(
            "span[class*='jobs-unified-top-card__posted-date']"
        )
        metadata["applicant_count"] = await _safe_text(
            "span[class*='jobs-unified-top-card__applicant-count']"
        )

        # Description — extract first 1000 chars
        try:
            desc_el = await self.page.query_selector(
                "div.jobs-description__content, div#job-details"
            )
            if desc_el:
                full_text: str = await desc_el.inner_text()
                metadata["description"] = full_text.strip()[:1000]
            else:
                metadata["description"] = ""
        except Exception:
            metadata["description"] = ""

        return metadata

    # ------------------------------------------------------------------
    # External Apply URL Detection
    # ------------------------------------------------------------------

    async def _get_external_apply_url(self) -> str:
        """Check if the LinkedIn job has an external apply button.

        LinkedIn jobs that redirect to an external ATS have an ``<a>``
        tag with an external ``href`` on the apply button. This method
        performs **read-only DOM inspection only** — it never clicks
        any buttons.

        Returns:
            External apply URL string, or ``""`` if the job uses Easy
            Apply or no external link is found.
        """
        # External apply button is an <a> tag (not a <button>)
        external_selectors: list[str] = [
            "a.jobs-apply-button[href]",
            "a[data-control-name='jobdetails_topcard_inapply'][href]",
            "a[class*='jobs-apply-button'][href]",
        ]

        for selector in external_selectors:
            try:
                el = await self.page.query_selector(selector)
                if el:
                    href: str = await el.get_attribute("href") or ""
                    if href and href.startswith("http"):
                        # Confirm it's not a LinkedIn internal URL
                        if "linkedin.com" not in href:
                            return href
            except Exception:
                continue

        # Check for non-Easy-Apply button (external redirect pattern)
        try:
            apply_btn = await self.page.query_selector(
                "button.jobs-apply-button"
                ":not([aria-label*='Easy Apply'])"
            )
            if apply_btn:
                aria: str = (
                    await apply_btn.get_attribute("aria-label") or ""
                )
                if "easy apply" not in aria.lower():
                    self.logger.info(
                        "LinkedIn external apply button found "
                        "(non-Easy-Apply) — flagging for manual review"
                    )
                    # Do NOT click — just flag. Read-only inspection.
        except Exception:
            pass

        return ""
