"""Indeed platform apply module.

Handles both Indeed Apply (hosted wizard inside iframe) and External
Apply (redirect to company ATS) flows.

Indeed Apply wizard characteristics:
    - Loads inside an iframe (``smartapply.indeed.com``).
    - React-controlled inputs throughout.
    - 1–4 wizard steps with variable page count.
    - High CAPTCHA risk — reroute immediately on any detection.
    - Resume upload via Indeed's own file widget.

External apply jobs return the external URL in ``proof_value``
so the pipeline can re-route to the correct platform module.

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
    Page,
    Frame,
    TimeoutError as PlaywrightTimeoutError,
)

from auto_apply.platforms.base_platform import (
    BasePlatformApply,
    ApplyResult,
)

logger = logging.getLogger(__name__)

__all__ = ["IndeedApply"]


class IndeedApply(BasePlatformApply):
    """Indeed platform Playwright apply module.

    Detects whether the current Indeed job uses the hosted Indeed Apply
    wizard (iframe) or an external apply link, and handles each flow.

    Steps:
        1. Navigate + verify Indeed URL + detect flow type.
        2–4. Wizard pages (fill fields per page + advance).

    For external apply jobs, ``proof_value`` contains the external URL
    so ``apply_tools.py`` can re-route to the correct platform module.
    """

    PLATFORM_NAME: str = "indeed"
    STEPS_TOTAL: int = 4
    MAX_WIZARD_PAGES: int = 5

    async def apply(self) -> ApplyResult:
        """Handle Indeed job application.

        Detects flow type (Indeed Apply wizard vs external redirect),
        handles each accordingly. For wizard: navigates iframe-hosted
        multi-step form. For external: returns external URL as pipeline
        signal.

        Returns:
            ApplyResult with proof or reroute instruction.
        """
        self.steps_completed = 0

        # ── Step 1: Navigate + detect flow type ──
        step1_result: Optional[ApplyResult] = (
            await self._step_navigate_and_detect()
        )
        if step1_result is not None:
            return step1_result

        # ── Wizard navigation loop ──
        return await self._wizard_loop()

    # ------------------------------------------------------------------
    # Step 1: Navigate + Detect
    # ------------------------------------------------------------------

    async def _step_navigate_and_detect(self) -> Optional[ApplyResult]:
        """Navigate to Indeed page, detect flow type, open wizard.

        Returns:
            ApplyResult if terminal (external URL found, error, or no
            button). None if wizard opened successfully — proceed to
            wizard loop.
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
            await asyncio.sleep(1.5)

            # Verify Indeed URL
            if "indeed.com" not in self.page.url:
                return self._build_result(
                    success=False,
                    error_code="UNKNOWN_ATS",
                    reroute_to_manual=True,
                    reroute_reason="Not an Indeed page",
                )

            if await self._detect_captcha():
                return self._build_result(
                    success=False,
                    error_code="CAPTCHA",
                    reroute_to_manual=True,
                    reroute_reason="CAPTCHA on Indeed page load",
                )

            # Check for external apply link first
            external_url: str = await self._get_indeed_external_url()
            if external_url:
                self.logger.info(
                    "Indeed external apply URL found: %s", external_url
                )
                self.steps_completed = 1
                # proof_value=external_url is used by apply_tools.py
                # to re-route to the correct platform module.
                return self._build_result(
                    success=True,
                    proof_type="success_url",
                    proof_value=external_url,
                    proof_confidence=1.0,
                    reroute_to_manual=False,
                    reroute_reason=None,
                )

            # Check for Indeed Apply button
            ia_button = await self.page.query_selector(
                "button[data-testid='indeedApplyButton'], "
                "button.ia-IndeedApplyButton, "
                "span.ia-IndeedApplyButton, "
                "div[class*='indeed-apply-widget'] button"
            )
            if not ia_button:
                return self._build_result(
                    success=False,
                    error_code="NAV_FAIL",
                    reroute_to_manual=True,
                    reroute_reason=(
                        "Indeed: no apply button found "
                        "(job may be expired)"
                    ),
                )

            # Click Indeed Apply to open wizard
            await ia_button.click()
            await asyncio.sleep(2)
            self.steps_completed = 1
            return None  # proceed to wizard loop

        except PlaywrightTimeoutError:
            return self._build_result(
                success=False,
                error_code="TIMEOUT",
                reroute_to_manual=True,
                reroute_reason="Indeed page load timeout",
            )
        except Exception as e:
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=str(e),
            )

    # ------------------------------------------------------------------
    # External URL Detection
    # ------------------------------------------------------------------

    async def _get_indeed_external_url(self) -> str:
        """Check for an external apply link on the Indeed job page.

        Looks for apply buttons that are ``<a>`` tags with external
        ``href`` attributes. Returns the URL or ``""`` if none found.

        Returns:
            External apply URL string, or ``""`` if Indeed Apply.
        """
        selectors: list[str] = [
            "a[data-testid='applyButton'][href]",
            "a.jobsearch-IndeedApplyButton-contentWrapper[href]",
            "a[id*='applyButton'][href]",
        ]
        for selector in selectors:
            try:
                el = await self.page.query_selector(selector)
                if el:
                    href: str = await el.get_attribute("href") or ""
                    if (
                        href.startswith("http")
                        and "indeed.com" not in href
                    ):
                        return href
            except Exception:
                continue
        return ""

    # ------------------------------------------------------------------
    # Wizard Frame Detection
    # ------------------------------------------------------------------

    async def _get_wizard_frame(self) -> Optional[Frame]:
        """Locate the Indeed Apply wizard iframe Frame object.

        Indeed Apply wizard loads inside an iframe from
        ``smartapply.indeed.com``. Waits up to 10s for the iframe
        to appear, then resolves the Frame context.

        Returns:
            Playwright Frame object, or None if not found.
        """
        try:
            await self.page.wait_for_selector(
                "iframe[id*='indeedapply'], "
                "iframe[src*='smartapply'], "
                "iframe[name*='indeed']",
                timeout=10000,
            )
        except PlaywrightTimeoutError:
            return None

        # Search frames by URL pattern
        for frame in self.page.frames:
            frame_url: str = frame.url or ""
            if any(
                x in frame_url
                for x in ["smartapply", "indeedapply", "indeed.com/apply"]
            ):
                return frame

        # Fallback: resolve via iframe element
        try:
            iframe_el = await self.page.query_selector(
                "iframe[id*='indeedapply'], iframe[src*='smartapply']"
            )
            if iframe_el:
                return await iframe_el.content_frame()
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Wizard Loop
    # ------------------------------------------------------------------

    async def _wizard_loop(self) -> ApplyResult:
        """Navigate through Indeed Apply wizard pages.

        Locates the wizard iframe, iterates through pages, fills
        fields, and advances. Exits on submit page or error.

        Returns:
            ApplyResult from submit handler or error.
        """
        try:
            wizard_frame: Optional[Frame] = (
                await self._get_wizard_frame()
            )
            if wizard_frame is None:
                return self._build_result(
                    success=False,
                    error_code="NAV_FAIL",
                    reroute_to_manual=True,
                    reroute_reason=(
                        "Indeed Apply wizard iframe not found "
                        "after button click"
                    ),
                )

            for page_num in range(self.MAX_WIZARD_PAGES):
                await asyncio.sleep(1)

                if await self._detect_captcha():
                    return self._build_result(
                        success=False,
                        error_code="CAPTCHA",
                        reroute_to_manual=True,
                        reroute_reason=(
                            f"CAPTCHA in Indeed wizard page {page_num + 1}"
                        ),
                    )

                # Detect review/submit page
                is_review = await wizard_frame.query_selector(
                    "[data-testid='ia-ReviewPage'], "
                    "div[class*='ReviewPage'], "
                    "button[data-testid='ia-SubmitButton']"
                )
                if is_review:
                    return await self._handle_indeed_submit(wizard_frame)

                # Fill current wizard page
                await self._fill_indeed_wizard_page(wizard_frame)
                self.steps_completed += 1

                # Click continue
                continued: bool = await self._click_indeed_continue(
                    wizard_frame
                )
                if not continued:
                    # Check if submit appeared
                    submit_el = await wizard_frame.query_selector(
                        "button[data-testid='ia-SubmitButton']"
                    )
                    if submit_el:
                        return await self._handle_indeed_submit(
                            wizard_frame
                        )
                    return self._build_result(
                        success=False,
                        error_code="NAV_FAIL",
                        reroute_to_manual=True,
                        reroute_reason=(
                            "Indeed wizard: could not advance past "
                            f"page {page_num + 1}"
                        ),
                    )

            # Exceeded max pages
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=(
                    "Indeed wizard exceeded max page limit"
                ),
            )

        except PlaywrightTimeoutError:
            return self._build_result(
                success=False,
                error_code="TIMEOUT",
                reroute_to_manual=True,
                reroute_reason="Indeed wizard navigation timeout",
            )
        except Exception as e:
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=f"Indeed wizard error: {str(e)}",
            )

    # ------------------------------------------------------------------
    # Wizard Page Fill
    # ------------------------------------------------------------------

    async def _fill_indeed_wizard_page(self, frame: Frame) -> None:
        """Fill all fillable fields on the current Indeed wizard page.

        Operates on a Frame context (iframe). Uses ``name``-based and
        ``id``-based selectors. All individual fills are non-fatal.

        Args:
            frame: Playwright Frame object for the wizard iframe.
        """
        profile: Dict[str, Any] = self.user_profile
        full_name: str = (
            f"{profile.get('first_name', '')} "
            f"{profile.get('last_name', '')}"
        ).strip()

        field_map: list[tuple[str, str]] = [
            (
                "input[name='applicant.name'], "
                "input[id='input-applicant.name']",
                full_name,
            ),
            (
                "input[name='applicant.email'], "
                "input[id='input-applicant.email']",
                str(profile.get("email", "")),
            ),
            (
                "input[name='applicant.phoneNumber'], "
                "input[id='input-applicant.phoneNumber']",
                str(profile.get("phone", "")),
            ),
        ]

        for selector, value in field_map:
            if not value:
                continue
            try:
                el = await frame.query_selector(selector)
                if el and await el.is_visible():
                    current: str = await el.input_value()
                    if not current.strip():
                        await frame.fill(selector, value)
            except Exception as e:
                self.logger.debug(
                    "Indeed wizard field fill failed (%s): %s",
                    selector,
                    str(e),
                )

        # Resume upload in wizard
        try:
            file_trigger = await frame.query_selector(
                "div[data-testid='FileUpload-input'], "
                "button[data-testid='FileUpload-input']"
            )
            if file_trigger:
                async with frame.expect_file_chooser(
                    timeout=5000
                ) as fc_info:
                    await file_trigger.click()
                fc = await fc_info.value
                await fc.set_files(self.resume_path)
                self.logger.info(
                    "Indeed resume uploaded via wizard: %s",
                    self.resume_path,
                )
        except Exception:
            # Fallback: direct file input
            try:
                file_input = await frame.query_selector(
                    "input[type='file']"
                )
                if file_input:
                    await file_input.set_input_files(self.resume_path)
                    self.logger.info(
                        "Indeed resume uploaded via file input: %s",
                        self.resume_path,
                    )
            except Exception as e:
                self.logger.debug(
                    "Indeed resume upload failed: %s", str(e)
                )

    # ------------------------------------------------------------------
    # Wizard Navigation
    # ------------------------------------------------------------------

    async def _click_indeed_continue(self, frame: Frame) -> bool:
        """Click Indeed wizard Continue/Next button.

        Args:
            frame: Playwright Frame object for the wizard iframe.

        Returns:
            True if button found and clicked, False otherwise.
        """
        selectors: list[str] = [
            "button[data-testid='ia-continueButton']",
            "button[data-testid='continue-button']",
            "button:has-text('Continue')",
            "button:has-text('Next')",
            "button[type='submit']"
            ":not([data-testid='ia-SubmitButton'])",
        ]
        for selector in selectors:
            try:
                el = await frame.query_selector(selector)
                if el and await el.is_visible():
                    await el.click()
                    await asyncio.sleep(1.5)
                    return True
            except Exception:
                continue
        return False

    # ------------------------------------------------------------------
    # Submit Handler
    # ------------------------------------------------------------------

    async def _handle_indeed_submit(self, frame: Frame) -> ApplyResult:
        """Handle the Indeed Apply review/submit page.

        In ``dry_run``: logs intent, returns success without clicking.
        In live mode: clicks Submit, captures proof.

        Args:
            frame: Playwright Frame object for the wizard iframe.

        Returns:
            ApplyResult with proof or reroute.
        """
        if self.dry_run:
            self.logger.info(
                "[DRY_RUN] Would submit Indeed Apply for %s",
                self.job_meta.get("job_url", ""),
            )
            self.steps_completed += 1
            return self._build_result(
                success=True,
                proof_type="none",
                proof_value="DRY_RUN",
                proof_confidence=1.0,
            )

        try:
            submit_btn = await frame.query_selector(
                "button[data-testid='ia-SubmitButton']"
            )
            if not submit_btn:
                return self._build_result(
                    success=False,
                    error_code="NAV_FAIL",
                    reroute_to_manual=True,
                    reroute_reason=(
                        "Indeed submit button not found on review page"
                    ),
                )

            await submit_btn.click()
            await asyncio.sleep(3)
            self.steps_completed += 1

            # Proof: confirmation in frame
            confirm = await frame.query_selector(
                "div[data-testid='ia-ConfirmationPage'], "
                "div[class*='ConfirmationPage'], "
                "h1:has-text('Application sent'), "
                "div:has-text('application was sent')"
            )
            if confirm:
                return self._build_result(
                    success=True,
                    proof_type="form_disappearance",
                    proof_value=(
                        f"Indeed confirmation: {self.page.url}"
                    ),
                    proof_confidence=0.93,
                )

            # URL-based proof
            if any(
                x in self.page.url
                for x in ["confirmation", "applied", "success", "thank"]
            ):
                return self._build_result(
                    success=True,
                    proof_type="success_url",
                    proof_value=self.page.url,
                    proof_confidence=0.90,
                )

            return self._build_result(
                success=False,
                error_code="PROOF_FAIL",
                reroute_to_manual=True,
                reroute_reason=(
                    "Indeed submit clicked but no confirmation found"
                ),
            )

        except Exception as e:
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=f"Indeed submit error: {str(e)}",
            )
