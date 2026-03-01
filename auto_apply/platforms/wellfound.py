"""Wellfound (formerly AngelList Talent) platform apply module.

Requires an active login session in the Playwright browser context.
Handles both Easy Apply (one-click with profile data) and Full Apply
(multi-field modal form) variants.

Key characteristics:
    - Custom React UI with no standard ``form[action]`` tags.
    - Fields identified by ``aria-label`` and ``data-test`` attributes.
    - Resume upload via custom uploader widget.
    - Login session is a hard prerequisite — reroute immediately
      if not logged in.

User profile keys expected:
    ``first_name``, ``last_name``, ``email``, ``phone``,
    ``linkedin_url``, ``portfolio_url``, ``location``,
    ``years_experience``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any, Optional

from playwright.async_api import (
    ElementHandle,
    TimeoutError as PlaywrightTimeoutError,
)

from auto_apply.platforms.base_platform import (
    BasePlatformApply,
    ApplyResult,
)

logger = logging.getLogger(__name__)

__all__ = ["WellfoundApply"]


class WellfoundApply(BasePlatformApply):
    """Wellfound (AngelList Talent) Playwright apply module.

    Checks login status, detects Easy Apply vs Full Apply variant,
    and handles both flows from the application modal.

    Steps:
        1. Navigate + verify + check login + open modal + detect variant.
        2. Submit (Easy Apply one-click or Full Apply form fill + submit).
    """

    PLATFORM_NAME: str = "wellfound"
    STEPS_TOTAL: int = 2

    async def apply(self) -> ApplyResult:
        """Handle Wellfound job application.

        Checks login, detects Easy Apply vs Full Apply, and handles
        each variant. For Easy Apply: one-click submit with
        confirmation capture. For Full Apply: fills modal form
        fields and submits.

        Returns:
            ApplyResult with proof or reroute instruction.
        """
        self.steps_completed = 0

        # ── Step 1: Navigate + verify + check login + detect variant ──
        return await self._step_navigate_and_apply()

    # ------------------------------------------------------------------
    # Step 1: Navigate + Verify + Open + Detect + Handle
    # ------------------------------------------------------------------

    async def _step_navigate_and_apply(self) -> ApplyResult:
        """Navigate, verify Wellfound, check login, detect variant.

        Returns:
            ApplyResult from the appropriate variant handler.
        """
        try:
            job_url: str = self.job_meta.get("job_url", "")
            current_url: str = self.page.url
            if not current_url.startswith(job_url[:30]):
                await self.page.goto(
                    job_url,
                    wait_until="networkidle",
                    timeout=20000,
                )
            await self.page.wait_for_load_state(
                "networkidle", timeout=20000
            )
            await asyncio.sleep(1)

            if "wellfound.com" not in self.page.url:
                return self._build_result(
                    success=False,
                    error_code="UNKNOWN_ATS",
                    reroute_to_manual=True,
                    reroute_reason="Not a Wellfound page",
                )

            # Login check — mandatory
            login_el = await self.page.query_selector(
                "a[href='/login'], button:has-text('Sign In'), "
                "a:has-text('Sign in')"
            )
            user_menu = await self.page.query_selector(
                "div[data-test='user-menu'], "
                "button[data-test='user-avatar'], "
                "img[alt*='avatar' i]"
            )
            if login_el and not user_menu:
                return self._build_result(
                    success=False,
                    error_code="NAV_FAIL",
                    reroute_to_manual=True,
                    reroute_reason=(
                        "Wellfound: not logged in. "
                        "Login session required for Wellfound apply."
                    ),
                )

            if await self._detect_captcha():
                return self._build_result(
                    success=False,
                    error_code="CAPTCHA",
                    reroute_to_manual=True,
                    reroute_reason="CAPTCHA on Wellfound page",
                )

            # Click main apply button to open modal
            apply_btn = await self.page.query_selector(
                "button[data-test='apply-button'], "
                "button:has-text('Apply'), "
                "a[data-test='apply-button']"
            )
            if not apply_btn:
                return self._build_result(
                    success=False,
                    error_code="NAV_FAIL",
                    reroute_to_manual=True,
                    reroute_reason="Wellfound: apply button not found",
                )
            await apply_btn.click()
            await asyncio.sleep(1.5)

            # Detect variant: Easy Apply vs Full Apply
            easy_btn = await self.page.query_selector(
                "button[data-test='easy-apply-btn'], "
                "button:has-text('Easy Apply'), "
                "span:has-text('Easy Apply')"
            )
            modal = await self.page.query_selector(
                "div[data-test='application-modal'], "
                "div[class*='ApplicationModal']"
            )

            self.steps_completed = 1

            if easy_btn:
                return await self._handle_easy_apply(easy_btn)
            elif modal:
                return await self._handle_full_apply()
            else:
                return self._build_result(
                    success=False,
                    error_code="NAV_FAIL",
                    reroute_to_manual=True,
                    reroute_reason=(
                        "Wellfound: could not detect Easy Apply or "
                        "Full Apply modal after clicking apply button"
                    ),
                )

        except PlaywrightTimeoutError:
            return self._build_result(
                success=False,
                error_code="TIMEOUT",
                reroute_to_manual=True,
                reroute_reason="Wellfound page load timeout",
            )
        except Exception as e:
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=str(e),
            )

    # ------------------------------------------------------------------
    # Easy Apply Handler
    # ------------------------------------------------------------------

    async def _handle_easy_apply(
        self, easy_btn: ElementHandle
    ) -> ApplyResult:
        """Handle Wellfound Easy Apply one-click flow.

        Clicks the Easy Apply button and captures confirmation.
        In ``dry_run``: logs intent, returns success without clicking.

        Args:
            easy_btn: Playwright ElementHandle for Easy Apply button.

        Returns:
            ApplyResult with confirmation proof.
        """
        if self.dry_run:
            self.logger.info(
                "[DRY_RUN] Would click Wellfound Easy Apply for %s",
                self.job_meta.get("job_url", ""),
            )
            self.steps_completed = 2
            return self._build_result(
                success=True,
                proof_type="none",
                proof_value="DRY_RUN",
                proof_confidence=1.0,
            )

        try:
            await easy_btn.click()
            await asyncio.sleep(2)
            self.steps_completed = 2

            # Proof: confirmation message or modal change
            confirm = await self.page.query_selector(
                "div:has-text('Application submitted'), "
                "div:has-text('Applied successfully'), "
                "div[data-test='application-success'], "
                "span:has-text('Applied')"
            )
            if confirm:
                return self._build_result(
                    success=True,
                    proof_type="form_disappearance",
                    proof_value=(
                        f"Wellfound Easy Apply: {self.page.url}"
                    ),
                    proof_confidence=0.92,
                )

            # Check if apply button changed to "Applied" state
            applied_state = await self.page.query_selector(
                "button:has-text('Applied'), "
                "button[disabled]:has-text('Apply'), "
                "span.applied-badge"
            )
            if applied_state:
                return self._build_result(
                    success=True,
                    proof_type="form_disappearance",
                    proof_value="Button state: Applied",
                    proof_confidence=0.88,
                )

            return self._build_result(
                success=False,
                error_code="PROOF_FAIL",
                reroute_to_manual=True,
                reroute_reason=(
                    "Wellfound Easy Apply: no confirmation detected"
                ),
            )

        except Exception as e:
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=(
                    f"Wellfound Easy Apply error: {str(e)}"
                ),
            )

    # ------------------------------------------------------------------
    # Full Apply Handler
    # ------------------------------------------------------------------

    async def _handle_full_apply(self) -> ApplyResult:
        """Handle Wellfound Full Apply modal form.

        Fills all available form fields inside the application modal,
        uploads resume, and submits. In ``dry_run``: fills fields but
        skips submit click.

        Note: ``textarea[data-test="message"]`` (cover letter) is
        intentionally left blank — too personal for automation.

        Returns:
            ApplyResult with proof or reroute instruction.
        """
        profile: Dict[str, Any] = self.user_profile

        # Fill standard fields inside modal
        field_map: list[tuple[str, str]] = [
            (
                "input[aria-label='First name']",
                str(profile.get("first_name", "")),
            ),
            (
                "input[aria-label='Last name']",
                str(profile.get("last_name", "")),
            ),
            (
                "input[aria-label='Email']",
                str(profile.get("email", "")),
            ),
            (
                "input[aria-label='Phone']",
                str(profile.get("phone", "")),
            ),
            (
                "input[aria-label='LinkedIn URL']",
                str(profile.get("linkedin_url", "")),
            ),
            (
                "input[aria-label='GitHub URL']",
                str(profile.get("portfolio_url", "")),
            ),
            (
                "input[aria-label='Portfolio URL']",
                str(profile.get("portfolio_url", "")),
            ),
            (
                "input[aria-label='Website URL']",
                str(profile.get("portfolio_url", "")),
            ),
        ]

        for selector, value in field_map:
            if not value:
                continue
            try:
                el = await self.page.query_selector(selector)
                if el and await el.is_visible():
                    current: str = await el.input_value()
                    if not current.strip():
                        await self._fill_react_input(selector, value)
            except Exception as e:
                self.logger.debug(
                    "Wellfound field fill failed (%s): %s",
                    selector,
                    str(e),
                )

        # Resume upload
        await self._upload_wellfound_resume()

        # Dry-run check
        if self.dry_run:
            self.logger.info(
                "[DRY_RUN] Would submit Wellfound Full Apply for %s",
                self.job_meta.get("job_url", ""),
            )
            self.steps_completed = 2
            return self._build_result(
                success=True,
                proof_type="none",
                proof_value="DRY_RUN",
                proof_confidence=1.0,
            )

        if await self._detect_captcha():
            return self._build_result(
                success=False,
                error_code="CAPTCHA",
                reroute_to_manual=True,
                reroute_reason=(
                    "CAPTCHA before Wellfound Full Apply submit"
                ),
            )

        # Submit
        return await self._submit_full_apply()

    async def _upload_wellfound_resume(self) -> None:
        """Upload resume via Wellfound's uploader widget.

        Tries the ``data-test="resume-upload"`` area first (file
        chooser trigger), falls back to direct ``<input type="file">``.
        """
        try:
            upload_area = await self.page.query_selector(
                "div[data-test='resume-upload']"
            )
            if upload_area:
                async with self.page.expect_file_chooser(
                    timeout=5000
                ) as fc_info:
                    await upload_area.click()
                fc = await fc_info.value
                await fc.set_files(self.resume_path)
                self.logger.info(
                    "Wellfound resume uploaded via widget: %s",
                    self.resume_path,
                )
                return
        except Exception:
            pass

        # Fallback: direct file input
        try:
            file_input = await self.page.query_selector(
                "input[type='file']"
            )
            if file_input:
                await file_input.set_input_files(self.resume_path)
                self.logger.info(
                    "Wellfound resume uploaded via file input: %s",
                    self.resume_path,
                )
        except Exception as e:
            self.logger.warning(
                "Wellfound resume upload failed: %s", str(e)
            )

    async def _submit_full_apply(self) -> ApplyResult:
        """Click submit and capture proof for Full Apply.

        Returns:
            ApplyResult with proof or reroute.
        """
        try:
            submit_clicked: bool = await self._click_next_or_continue(
                [
                    "button[data-test='submit-application']",
                    "button[type='submit']",
                    "button:has-text('Submit Application')",
                    "button:has-text('Submit')",
                    "button:has-text('Apply')",
                ]
            )

            if not submit_clicked:
                return self._build_result(
                    success=False,
                    error_code="NAV_FAIL",
                    reroute_to_manual=True,
                    reroute_reason=(
                        "Wellfound Full Apply: submit button not found"
                    ),
                )

            await asyncio.sleep(2)
            self.steps_completed = 2

            # Proof capture
            confirm = await self.page.query_selector(
                "div[data-test='application-success'], "
                "div:has-text('Application submitted'), "
                "div:has-text('Applied successfully'), "
                "span:has-text('Applied')"
            )
            if confirm:
                return self._build_result(
                    success=True,
                    proof_type="form_disappearance",
                    proof_value=(
                        f"Wellfound Full Apply: {self.page.url}"
                    ),
                    proof_confidence=0.91,
                )

            return self._build_result(
                success=False,
                error_code="PROOF_FAIL",
                reroute_to_manual=True,
                reroute_reason=(
                    "Wellfound Full Apply: submit clicked "
                    "but no confirmation detected"
                ),
            )

        except Exception as e:
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=f"Wellfound submit error: {str(e)}",
            )
