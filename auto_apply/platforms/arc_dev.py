"""Arc.dev remote jobs platform apply module.

Arc.dev application flow opens a multi-step modal after clicking
"Apply". Arc.dev pre-fills fields from the user's Arc.dev profile —
this module fills only remaining empty fields and handles submission.

Key characteristics:
    - ``data-testid`` attributes are the most reliable selectors.
    - Standard HTML inputs rendered by React in a modal.
    - 3–4 modal steps: Basic Info → Links → Resume → Questions (opt).
    - Custom question textareas are left blank (too custom for auto-fill).
    - Resume upload checks if already uploaded before triggering chooser.

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

__all__ = ["ArcDevApply"]


class ArcDevApply(BasePlatformApply):
    """Arc.dev remote job platform Playwright apply module.

    Navigates a multi-step application modal, filling only empty
    fields (Arc.dev pre-fills from user profile). Handles up to
    5 modal steps with dynamic step detection.

    Steps:
        1. Navigate + verify + open modal.
        2–4. Fill fields per step + advance.
    """

    PLATFORM_NAME: str = "arc_dev"
    STEPS_TOTAL: int = 4
    MAX_MODAL_STEPS: int = 5

    async def apply(self) -> ApplyResult:
        """Handle Arc.dev remote job application modal.

        Navigates multi-step application modal, filling only empty
        fields (Arc.dev pre-fills from user profile). Handles up to
        5 modal steps with dynamic step detection.

        Returns:
            ApplyResult with proof or reroute instruction.
        """
        self.steps_completed = 0

        # ── Step 1: Navigate + verify + open modal ──
        step1_result: Optional[ApplyResult] = (
            await self._step_navigate_and_open()
        )
        if step1_result is not None:
            return step1_result

        # ── Modal step loop ──
        return await self._modal_step_loop()

    # ------------------------------------------------------------------
    # Step 1: Navigate + Open Modal
    # ------------------------------------------------------------------

    async def _step_navigate_and_open(self) -> Optional[ApplyResult]:
        """Navigate to Arc.dev page and open application modal.

        Returns:
            ApplyResult on failure, None on success to proceed.
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

            if "arc.dev" not in self.page.url:
                return self._build_result(
                    success=False,
                    error_code="UNKNOWN_ATS",
                    reroute_to_manual=True,
                    reroute_reason="Not an Arc.dev page",
                )

            if await self._detect_captcha():
                return self._build_result(
                    success=False,
                    error_code="CAPTCHA",
                    reroute_to_manual=True,
                    reroute_reason="CAPTCHA on Arc.dev page",
                )

            # Click apply button
            apply_btn = await self.page.query_selector(
                "button[data-testid='apply-btn'], "
                "button:has-text('Apply Now'), "
                "button:has-text('Apply')"
            )
            if not apply_btn:
                return self._build_result(
                    success=False,
                    error_code="NAV_FAIL",
                    reroute_to_manual=True,
                    reroute_reason="Arc.dev: apply button not found",
                )

            await apply_btn.click()
            await asyncio.sleep(1.5)

            # Verify modal opened
            modal = await self.page.query_selector(
                "div[data-testid='application-modal'], "
                "div[class*='ApplicationModal']"
            )
            if not modal:
                return self._build_result(
                    success=False,
                    error_code="NAV_FAIL",
                    reroute_to_manual=True,
                    reroute_reason=(
                        "Arc.dev: application modal did not open"
                    ),
                )

            self.steps_completed = 1
            return None  # proceed to modal loop

        except PlaywrightTimeoutError:
            return self._build_result(
                success=False,
                error_code="TIMEOUT",
                reroute_to_manual=True,
                reroute_reason=(
                    "Arc.dev page load or modal open timeout"
                ),
            )
        except Exception as e:
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=str(e),
            )

    # ------------------------------------------------------------------
    # Modal Step Loop
    # ------------------------------------------------------------------

    async def _modal_step_loop(self) -> ApplyResult:
        """Iterate through Arc.dev modal steps, filling and advancing.

        Returns:
            ApplyResult from submit handler or error.
        """
        try:
            for step_num in range(self.MAX_MODAL_STEPS):
                await asyncio.sleep(1)

                if await self._detect_captcha():
                    return self._build_result(
                        success=False,
                        error_code="CAPTCHA",
                        reroute_to_manual=True,
                        reroute_reason=(
                            f"CAPTCHA in Arc.dev modal step "
                            f"{step_num + 1}"
                        ),
                    )

                # Check for submit button — indicates final step
                submit_btn = await self.page.query_selector(
                    "button[data-testid='submit-btn'], "
                    "button[type='submit']:has-text('Submit')"
                )

                # Fill current step fields (empty ones only)
                await self._fill_arc_step_fields()
                self.steps_completed += 1

                if submit_btn and await submit_btn.is_visible():
                    return await self._handle_arc_submit(submit_btn)

                # Advance to next step
                advanced: bool = await self._click_next_or_continue(
                    [
                        "button[data-testid='next-step-btn']",
                        "button:has-text('Next')",
                        "button:has-text('Continue')",
                        "button[type='button']:has-text('Next')",
                    ]
                )

                if not advanced:
                    # Re-check for submit (some flows skip to submit)
                    submit_btn = await self.page.query_selector(
                        "button[data-testid='submit-btn']"
                    )
                    if submit_btn and await submit_btn.is_visible():
                        return await self._handle_arc_submit(submit_btn)
                    return self._build_result(
                        success=False,
                        error_code="NAV_FAIL",
                        reroute_to_manual=True,
                        reroute_reason=(
                            "Arc.dev: could not advance past "
                            f"step {step_num + 1}"
                        ),
                    )

            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason="Arc.dev: exceeded max modal steps",
            )

        except PlaywrightTimeoutError:
            return self._build_result(
                success=False,
                error_code="TIMEOUT",
                reroute_to_manual=True,
                reroute_reason="Arc.dev modal navigation timeout",
            )
        except Exception as e:
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=f"Arc.dev modal error: {str(e)}",
            )

    # ------------------------------------------------------------------
    # Step Field Fill
    # ------------------------------------------------------------------

    async def _fill_arc_step_fields(self) -> None:
        """Fill only empty fields on the current Arc.dev modal step.

        Arc.dev pre-fills from user profile — this method respects
        existing values and only fills genuinely empty fields.
        Each field failure is non-fatal.
        """
        profile: Dict[str, Any] = self.user_profile
        full_name: str = (
            f"{profile.get('first_name', '')} "
            f"{profile.get('last_name', '')}"
        ).strip()

        field_map: list[tuple[str, str]] = [
            # Basic info step
            (
                "input[data-testid='applicant-name'], "
                "input[placeholder*='name' i]",
                full_name,
            ),
            (
                "input[data-testid='applicant-email'], "
                "input[placeholder*='email' i]",
                str(profile.get("email", "")),
            ),
            (
                "input[data-testid='applicant-phone'], "
                "input[placeholder*='phone' i]",
                str(profile.get("phone", "")),
            ),
            # Links step
            (
                "input[data-testid='linkedin-url'], "
                "input[placeholder*='LinkedIn' i]",
                str(profile.get("linkedin_url", "")),
            ),
            (
                "input[data-testid='portfolio-url'], "
                "input[placeholder*='portfolio' i]",
                str(profile.get("portfolio_url", "")),
            ),
            (
                "input[data-testid='github-url'], "
                "input[placeholder*='GitHub' i]",
                str(profile.get("portfolio_url", "")),
            ),
        ]

        for selector, value in field_map:
            if not value:
                continue
            try:
                el = await self.page.query_selector(selector)
                if not el or not await el.is_visible():
                    continue
                current: str = await el.input_value()
                if current.strip():
                    continue  # Pre-filled — do not overwrite
                await self._fill_react_input(selector, value)
            except Exception as e:
                self.logger.debug(
                    "Arc.dev field fill failed (%s): %s",
                    selector,
                    str(e),
                )

        # Resume upload (only if dropzone visible and no resume uploaded)
        await self._upload_arc_resume()

        # Log custom questions (leave blank — too custom to auto-fill)
        try:
            questions = await self.page.query_selector_all(
                "textarea[data-testid*='question']"
            )
            if questions:
                self.logger.info(
                    "Arc.dev: %d custom question(s) found — left blank",
                    len(questions),
                )
        except Exception:
            pass

    async def _upload_arc_resume(self) -> None:
        """Upload resume via Arc.dev dropzone if not already uploaded.

        Checks for existing upload indicators before triggering the
        file chooser to avoid duplicate uploads.
        """
        try:
            dropzone = await self.page.query_selector(
                "div[data-testid='resume-dropzone']"
            )
            if not dropzone or not await dropzone.is_visible():
                return

            # Check if resume already uploaded
            uploaded_text: str = await dropzone.inner_text()
            upload_indicators: list[str] = [
                ".pdf",
                ".doc",
                "uploaded",
                "resume",
            ]
            if any(
                ind in uploaded_text.lower() for ind in upload_indicators
            ):
                self.logger.info(
                    "Arc.dev: resume already uploaded — skipping"
                )
                return

            async with self.page.expect_file_chooser(
                timeout=5000
            ) as fc_info:
                await dropzone.click()
            fc = await fc_info.value
            await fc.set_files(self.resume_path)
            self.logger.info(
                "Arc.dev resume uploaded: %s", self.resume_path
            )
        except Exception as e:
            self.logger.debug(
                "Arc.dev resume upload failed: %s", str(e)
            )

    # ------------------------------------------------------------------
    # Submit Handler
    # ------------------------------------------------------------------

    async def _handle_arc_submit(
        self, submit_btn: ElementHandle
    ) -> ApplyResult:
        """Handle Arc.dev final submit step.

        In ``dry_run``: logs intent, returns success without clicking.
        In live mode: clicks Submit, waits for confirmation.

        Args:
            submit_btn: Playwright ElementHandle for submit button.

        Returns:
            ApplyResult with proof or reroute.
        """
        if self.dry_run:
            self.logger.info(
                "[DRY_RUN] Would submit Arc.dev application for %s",
                self.job_meta.get("job_url", ""),
            )
            self.steps_completed += 1
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
                reroute_reason="CAPTCHA before Arc.dev submit",
            )

        try:
            await submit_btn.click()
            await asyncio.sleep(2.5)
            self.steps_completed += 1

            # Proof: confirmation element
            confirm = await self.page.query_selector(
                "div[data-testid='application-success'], "
                "h2:has-text('Application Submitted'), "
                "div:has-text('successfully applied'), "
                "div:has-text('application has been submitted')"
            )
            if confirm:
                return self._build_result(
                    success=True,
                    proof_type="form_disappearance",
                    proof_value=(
                        f"Arc.dev confirmation: {self.page.url}"
                    ),
                    proof_confidence=0.92,
                )

            # URL-based proof
            if any(
                x in self.page.url
                for x in [
                    "success",
                    "confirmation",
                    "applied",
                    "thank",
                ]
            ):
                return self._build_result(
                    success=True,
                    proof_type="success_url",
                    proof_value=self.page.url,
                    proof_confidence=0.88,
                )

            # Modal disappeared proof
            modal_gone: bool = not await self.page.query_selector(
                "div[data-testid='application-modal']"
            )
            if modal_gone:
                return self._build_result(
                    success=True,
                    proof_type="form_disappearance",
                    proof_value=self.page.url,
                    proof_confidence=0.72,
                )

            return self._build_result(
                success=False,
                error_code="PROOF_FAIL",
                reroute_to_manual=True,
                reroute_reason=(
                    "Arc.dev submit clicked but no confirmation "
                    "proof found"
                ),
            )

        except Exception as e:
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=f"Arc.dev submit error: {str(e)}",
            )
