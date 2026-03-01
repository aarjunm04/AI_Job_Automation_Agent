"""Workday ATS platform-specific Playwright apply module.

Handles the full multi-page application flow on Workday instances
(``{company}.wd{N}.myworkdayjobs.com``, ``wd{N}.myworkday.com``).

Key Workday challenges addressed:
    1. Multi-page flow (3–7 pages) with dynamic routing.
    2. ALL inputs are React-controlled — 3-tier fill strategy required.
    3. Proprietary ``data-automation-id`` attributes for reliable selectors.
    4. Workday-specific file upload widget (not standard ``<input type=file>``).
    5. Variable page ordering per company configuration.
    6. Voluntary disclosure pages handled via decline-all strategy.

User profile keys expected (split by caller before passing):
    ``first_name``, ``last_name``, ``email``, ``phone``,
    ``linkedin_url``, ``portfolio_url``, ``location``,
    ``years_experience``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any, List, Optional

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from auto_apply.platforms.base_platform import (
    BasePlatformApply,
    ApplyResult,
)

logger = logging.getLogger(__name__)

__all__ = ["WorkdayApply"]


# ---------------------------------------------------------------------------
# Question Answer Helper
# ---------------------------------------------------------------------------


def _get_workday_question_answer(
    label_text: str,
    user_profile: Dict[str, Any],
) -> str:
    """Map a Workday screening question label to an auto-answer.

    Returns empty string if no mapping found — the field will be skipped.
    Covers the most common Workday screening question patterns.

    Args:
        label_text: The full text of the question label.
        user_profile: Dict of user profile values.

    Returns:
        String answer or ``""`` to skip.
    """
    label_lower: str = label_text.lower()

    # Work authorisation
    if any(
        x in label_lower
        for x in [
            "authoris",
            "authoriz",
            "eligible to work",
            "work in",
            "legally authoris",
            "legally authoriz",
        ]
    ):
        return "Yes"

    # Visa sponsorship
    if any(
        x in label_lower
        for x in [
            "sponsor",
            "visa",
            "require sponsorship",
            "need sponsorship",
            "work permit",
        ]
    ):
        return "No"

    # Years of experience
    if any(
        x in label_lower
        for x in [
            "years of experience",
            "years experience",
            "how many years",
            "total experience",
        ]
    ):
        return str(user_profile.get("years_experience", "1"))

    # Remote work
    if any(
        x in label_lower
        for x in ["remote", "work from home", "hybrid", "onsite"]
    ):
        return "Yes"

    # Availability / start date
    if any(
        x in label_lower
        for x in [
            "available",
            "start date",
            "notice period",
            "when can you",
            "earliest",
        ]
    ):
        return "Immediately"

    # Salary / compensation
    if any(
        x in label_lower
        for x in [
            "salary",
            "compensation",
            "ctc",
            "expected pay",
            "desired salary",
        ]
    ):
        return "Open to discussion"

    # Relocation
    if any(
        x in label_lower
        for x in ["relocat", "willing to move", "open to relocation"]
    ):
        return "No"

    # Disability / veteran / gender — leave blank (privacy)
    if any(
        x in label_lower
        for x in [
            "disability",
            "veteran",
            "gender",
            "race",
            "ethnicity",
            "orientation",
        ]
    ):
        return ""

    # Referral source
    if any(
        x in label_lower
        for x in ["how did you hear", "referral", "source", "learn about"]
    ):
        return "Online job board"

    # LinkedIn
    if "linkedin" in label_lower:
        return str(user_profile.get("linkedin_url", ""))

    # Location
    if any(x in label_lower for x in ["location", "city", "where"]):
        return str(user_profile.get("location", ""))

    return ""


# ---------------------------------------------------------------------------
# Workday Apply Module
# ---------------------------------------------------------------------------


class WorkdayApply(BasePlatformApply):
    """Workday ATS multi-page Playwright apply module.

    Navigates through all Workday application pages dynamically,
    filling fields on each page and advancing via Next/Save & Continue
    buttons. Handles variable page counts (3–7 pages) and React inputs
    throughout using a 3-tier fill strategy.

    Pages handled:
        - My Information (name, email, phone, city)
        - My Experience (resume upload, LinkedIn, website)
        - Application Questions (screening/custom questions)
        - Self Identify / Voluntary Disclosures (decline-all)
        - Review & Submit (proof capture)

    Work experience and education sections are intentionally skipped —
    too complex for reliable automation.
    """

    PLATFORM_NAME: str = "workday"
    STEPS_TOTAL: int = 7

    MAX_PAGES: int = 8
    STEP_TIMEOUT: int = 20000
    FIELD_TIMEOUT: int = 8000

    def __init__(
        self,
        page: Page,
        job_meta: Dict[str, Any],
        user_profile: Dict[str, Any],
        dry_run: bool = False,
    ) -> None:
        super().__init__(page, job_meta, user_profile, dry_run)
        self._current_page_num: int = 0
        self._page_titles_visited: List[str] = []

    # ==================================================================
    # Main Apply Flow
    # ==================================================================

    async def apply(self) -> ApplyResult:
        """Execute the full Workday multi-page application flow.

        Navigates through all Workday pages dynamically, filling fields
        on each page and advancing via Next/Save & Continue buttons.
        Handles variable page counts (3–7 pages) and React inputs.

        Returns:
            ApplyResult with proof capture from final submission page.
        """
        self.steps_completed = 0
        self._current_page_num = 0
        self._page_titles_visited = []

        # ── Step 1: Navigate + Verify Workday fingerprint ──
        nav_result: Optional[ApplyResult] = await self._step_navigate_and_verify()
        if nav_result is not None:
            return nav_result

        # ── Multi-page loop ──
        return await self._multi_page_loop()

    # ------------------------------------------------------------------
    # Step 1: Navigate + Verify
    # ------------------------------------------------------------------

    async def _step_navigate_and_verify(self) -> Optional[ApplyResult]:
        """Navigate to job URL and verify Workday ATS fingerprint.

        Returns:
            ApplyResult on failure (reroute), None on success to proceed.
        """
        try:
            job_url: str = self.job_meta.get("job_url", "")
            if not self.page.url.startswith(job_url[:35]):
                await self.page.goto(
                    job_url,
                    wait_until="domcontentloaded",
                    timeout=25000,
                )
            await self.page.wait_for_load_state(
                "networkidle", timeout=20000
            )

            # Verify Workday fingerprint
            wd_body = await self.page.query_selector(
                "[data-automation-id='wd-Page-Body'], "
                "div[class*='WDUI'], "
                "div[data-uxi-widget-type]"
            )
            if not wd_body:
                return self._build_result(
                    success=False,
                    error_code="UNKNOWN_ATS",
                    reroute_to_manual=True,
                    reroute_reason=(
                        "Workday fingerprint not found — ATS mismatch"
                    ),
                )

            if await self._detect_captcha():
                return self._build_result(
                    success=False,
                    error_code="CAPTCHA",
                    reroute_to_manual=True,
                    reroute_reason="CAPTCHA on Workday page load",
                )

            self.steps_completed = 1
            self.logger.info(
                "Workday ATS verified for %s",
                self.job_meta.get("job_url", ""),
            )
            return None

        except PlaywrightTimeoutError:
            return self._build_result(
                success=False,
                error_code="TIMEOUT",
                reroute_to_manual=True,
                reroute_reason="Workday initial page load timeout",
            )
        except Exception as e:
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=str(e),
            )

    # ------------------------------------------------------------------
    # Multi-Page Loop
    # ------------------------------------------------------------------

    async def _multi_page_loop(self) -> ApplyResult:
        """Iterate through Workday pages, filling and advancing.

        Detects the current page title, routes to the appropriate handler,
        then clicks Next/Continue to advance. Exits on submit, error,
        or MAX_PAGES safety cap.

        Returns:
            ApplyResult from the review/submit page or an error result.
        """
        for page_iteration in range(self.MAX_PAGES):
            self._current_page_num = page_iteration + 1

            try:
                # Wait for page body to be visible
                await self.page.wait_for_selector(
                    "[data-automation-id='wd-Page-Body']",
                    state="visible",
                    timeout=self.STEP_TIMEOUT,
                )
                await asyncio.sleep(1)  # Workday React settle time

                # Identify current page
                page_title: str = await self._get_current_page_title()
                self._page_titles_visited.append(page_title)
                self.logger.info(
                    "Workday page %d: '%s'",
                    self._current_page_num,
                    page_title,
                )

                # CAPTCHA check on every page
                if await self._detect_captcha():
                    return self._build_result(
                        success=False,
                        error_code="CAPTCHA",
                        reroute_to_manual=True,
                        reroute_reason=(
                            f"CAPTCHA on Workday page: {page_title}"
                        ),
                    )

                # Route to appropriate page handler
                page_title_lower: str = page_title.lower()

                if any(
                    x in page_title_lower
                    for x in [
                        "my information",
                        "personal info",
                        "contact info",
                        "legal name",
                        "your information",
                    ]
                ):
                    await self._handle_my_information_page()

                elif any(
                    x in page_title_lower
                    for x in [
                        "my experience",
                        "experience",
                        "resume",
                        "work history",
                        "background",
                    ]
                ):
                    await self._handle_my_experience_page()

                elif any(
                    x in page_title_lower
                    for x in [
                        "application question",
                        "additional question",
                        "questionnaire",
                        "screening",
                    ]
                ):
                    await self._handle_application_questions_page()

                elif any(
                    x in page_title_lower
                    for x in [
                        "self identify",
                        "voluntary",
                        "disclosure",
                        "equal employment",
                        "eeoc",
                        "diversity",
                    ]
                ):
                    await self._handle_voluntary_disclosures_page()

                elif any(
                    x in page_title_lower
                    for x in ["review", "confirm", "summary", "preview"]
                ):
                    return await self._handle_review_and_submit_page()

                else:
                    # Unknown page — attempt to advance without filling
                    self.logger.warning(
                        "Unknown Workday page: '%s' — attempting to advance",
                        page_title,
                    )
                    advanced: bool = await self._click_next_button()
                    if not advanced:
                        return self._build_result(
                            success=False,
                            error_code="NAV_FAIL",
                            reroute_to_manual=True,
                            reroute_reason=(
                                f"Stuck on unknown Workday page: "
                                f"{page_title}"
                            ),
                        )
                    continue

                # Advance to next page
                self.steps_completed += 1
                await asyncio.sleep(1)
                advanced = await self._click_next_button()

                if not advanced:
                    # Check if already on review/submit page
                    new_title: str = await self._get_current_page_title()
                    if any(
                        x in new_title.lower()
                        for x in ["review", "confirm", "submit"]
                    ):
                        return await self._handle_review_and_submit_page()
                    return self._build_result(
                        success=False,
                        error_code="NAV_FAIL",
                        reroute_to_manual=True,
                        reroute_reason=(
                            f"Could not advance past Workday page: "
                            f"{page_title}"
                        ),
                    )

            except PlaywrightTimeoutError:
                last_title: str = (
                    self._page_titles_visited[-1]
                    if self._page_titles_visited
                    else "unknown"
                )
                return self._build_result(
                    success=False,
                    error_code="TIMEOUT",
                    reroute_to_manual=True,
                    reroute_reason=(
                        f"Timeout on Workday page "
                        f"{self._current_page_num}: {last_title}"
                    ),
                )
            except Exception as e:
                self.logger.error(
                    "Unexpected error on Workday page %d: %s",
                    self._current_page_num,
                    str(e),
                )
                return self._build_result(
                    success=False,
                    error_code="NAV_FAIL",
                    reroute_to_manual=True,
                    reroute_reason=(
                        f"Error on page {self._current_page_num}: "
                        f"{str(e)}"
                    ),
                )

        # Safety: exceeded MAX_PAGES
        return self._build_result(
            success=False,
            error_code="NAV_FAIL",
            reroute_to_manual=True,
            reroute_reason=(
                f"Workday exceeded MAX_PAGES ({self.MAX_PAGES}) — "
                f"pages visited: {self._page_titles_visited}"
            ),
        )

    # ==================================================================
    # Page Title Detection
    # ==================================================================

    async def _get_current_page_title(self) -> str:
        """Read the current Workday step/page title from DOM.

        Tries ``data-automation-id`` selectors first, falls back to
        generic heading elements.

        Returns:
            Page title string, or ``""`` if undetectable.
        """
        title_selectors: list[str] = [
            "[data-automation-id='stepTitle']",
            "h2[data-automation-id='pageHeaderTitle']",
            "[data-automation-id='pageHeader'] h2",
            "h2.css-1xdhyk6",
            "h1",
            "h2",
        ]
        for selector in title_selectors:
            try:
                el = await self.page.query_selector(selector)
                if el:
                    text: str = await el.inner_text()
                    if text.strip():
                        return text.strip()
            except Exception:
                continue
        return ""

    # ==================================================================
    # Page Handlers
    # ==================================================================

    async def _handle_my_information_page(self) -> None:
        """Fill the My Information page fields.

        Fills first name, last name, email (if empty), phone, and city.
        All fields are React-controlled — uses ``_fill_wd_field``
        throughout. Each field failure is non-fatal.
        """
        profile: Dict[str, Any] = self.user_profile

        # First name
        await self._fill_wd_field(
            "[data-automation-id='firstName']",
            str(profile.get("first_name", "")),
        )

        # Last name
        await self._fill_wd_field(
            "[data-automation-id='lastName']",
            str(profile.get("last_name", "")),
        )

        # Email — may be pre-filled from Workday account
        try:
            email_el = await self.page.query_selector(
                "[data-automation-id='email']"
            )
            if email_el:
                current: str = await email_el.input_value() or ""
                if not current.strip():
                    await self._fill_wd_field(
                        "[data-automation-id='email']",
                        str(profile.get("email", "")),
                    )
        except Exception:
            await self._fill_wd_field(
                "[data-automation-id='email']",
                str(profile.get("email", "")),
            )

        # Phone
        await self._fill_wd_field(
            "[data-automation-id='phone-number']",
            str(profile.get("phone", "")),
        )

        # City / location (optional — present in some configs)
        location: str = str(profile.get("location", ""))
        if location:
            city: str = (
                location.split(",")[0].strip()
                if "," in location
                else location
            )
            await self._fill_wd_field(
                "[data-automation-id='city']", city
            )

        self.logger.info("My Information page filled")

    async def _handle_my_experience_page(self) -> None:
        """Fill the My Experience page — resume upload + URLs.

        Skips work history and education sections (too complex for
        reliable automation). Uploads resume via Workday's proprietary
        file upload widget.
        """
        # Resume upload — Workday file upload widget
        upload_success: bool = False

        # Strategy 1: click the drop zone to trigger file chooser
        try:
            drop_zone = await self.page.query_selector(
                "[data-automation-id='file-upload-drop-zone']"
            )
            if drop_zone:
                async with self.page.expect_file_chooser(
                    timeout=8000
                ) as fc_info:
                    await drop_zone.click()
                fc = await fc_info.value
                await fc.set_files(self.resume_path)
                upload_success = True
                self.logger.info(
                    "Workday resume uploaded via drop zone: %s",
                    self.resume_path,
                )
        except Exception as e:
            self.logger.debug(
                "Drop zone upload failed: %s — trying file input", str(e)
            )

        # Strategy 2: direct hidden file input
        if not upload_success:
            try:
                file_input = await self.page.query_selector(
                    "input[data-automation-id='file-upload-input'], "
                    "input[type='file']"
                )
                if file_input:
                    await file_input.set_input_files(self.resume_path)
                    upload_success = True
                    self.logger.info(
                        "Workday resume uploaded via file input: %s",
                        self.resume_path,
                    )
            except Exception as e:
                self.logger.warning(
                    "Workday resume upload fallback failed: %s", str(e)
                )

        if not upload_success:
            self.logger.warning(
                "All Workday resume upload strategies failed for %s",
                self.job_meta.get("job_url", ""),
            )

        # LinkedIn URL
        await self._fill_wd_field(
            "[data-automation-id='linkedIn']",
            str(self.user_profile.get("linkedin_url", "")),
        )

        # Website / portfolio
        await self._fill_wd_field(
            "[data-automation-id='website']",
            str(self.user_profile.get("portfolio_url", "")),
        )

        self.logger.info("My Experience page filled")

    async def _handle_application_questions_page(self) -> None:
        """Handle Workday Application Questions / Screening Questions.

        Finds all question inputs by ``data-automation-id`` patterns and
        attempts keyword-based auto-answers. Skips questions it cannot
        map to a known answer pattern.
        """
        try:
            question_containers = await self.page.query_selector_all(
                "[data-automation-id='questionnaire'], "
                "[data-automation-id*='Question'], "
                "div[data-automation-id='formField']"
            )

            for container in question_containers:
                await self._handle_single_question(container)

        except Exception as e:
            self.logger.warning(
                "Application questions scan error: %s — continuing",
                str(e),
            )

        self.logger.info("Application Questions page processed")

    async def _handle_single_question(self, container: Any) -> None:
        """Process a single question container on the questions page.

        Args:
            container: Playwright ElementHandle for the question container.
        """
        try:
            # Get question label text
            label_el = await container.query_selector(
                "label, [data-automation-id='questionTitle'], "
                "div[class*='label']"
            )
            label_text: str = ""
            if label_el:
                label_text = (await label_el.inner_text()).strip()

            # Find input within container
            input_el = await container.query_selector(
                "input[type='text'], input[type='number'], "
                "textarea, select"
            )

            # Find yes/no radio elements
            yes_el = await container.query_selector(
                "[data-automation-id='yesLabel'], "
                "label:has-text('Yes'), [for*='yes' i]"
            )
            no_el = await container.query_selector(
                "[data-automation-id='noLabel'], "
                "label:has-text('No'), [for*='no' i]"
            )

            answer: str = _get_workday_question_answer(
                label_text, self.user_profile
            )
            if not answer:
                return

            # Yes/No question
            if yes_el and no_el and answer.lower() in ("yes", "no"):
                target = yes_el if answer.lower() == "yes" else no_el
                try:
                    await target.click()
                except Exception:
                    pass
                return

            # Text/number/select input
            if input_el:
                tag: str = await input_el.evaluate(
                    "el => el.tagName.toLowerCase()"
                )
                if tag == "select":
                    try:
                        await input_el.select_option(label=answer)
                    except Exception:
                        try:
                            await input_el.select_option(value=answer)
                        except Exception:
                            pass
                else:
                    input_id: str = (
                        await input_el.get_attribute("id") or ""
                    )
                    if input_id:
                        await self._fill_wd_field(f"#{input_id}", answer)

        except Exception as e:
            self.logger.debug(
                "Skipping question due to error: %s", str(e)
            )

    async def _handle_voluntary_disclosures_page(self) -> None:
        """Handle voluntary disclosures / self-identification pages.

        Uses a decline-all strategy: clicks "Decline to answer" or
        "Prefer not to say" options for all fields. If not available,
        fields are left blank and the page is advanced. This is the
        safest approach — demographic fields should never be auto-filled.
        """
        decline_selectors: list[str] = [
            "label:has-text('Decline to answer')",
            "label:has-text('Prefer not to answer')",
            "label:has-text('Prefer not to say')",
            "label:has-text('Do not wish to disclose')",
            "label:has-text('I do not wish')",
            "[data-automation-id='promptOption']:has-text('Decline')",
            "input[value*='decline' i] + label",
            "input[value*='prefer_not' i] + label",
        ]

        for selector in decline_selectors:
            try:
                elements = await self.page.query_selector_all(selector)
                for el in elements:
                    try:
                        await el.click()
                        await asyncio.sleep(0.2)
                    except Exception:
                        continue
            except Exception:
                continue

        self.logger.info(
            "Voluntary Disclosures page — decline-all strategy applied"
        )

    # ------------------------------------------------------------------
    # Review & Submit
    # ------------------------------------------------------------------

    async def _handle_review_and_submit_page(self) -> ApplyResult:
        """Handle the final Review + Submit page.

        In ``dry_run`` mode: logs intent and returns success without
        clicking Submit. In live mode: clicks Submit and captures proof
        via 3-tier strategy (confirmation text, URL change, page gone).

        Returns:
            ApplyResult with proof from submission.
        """
        self.steps_completed += 1

        if self.dry_run:
            self.logger.info(
                "[DRY_RUN] Would submit Workday application for %s",
                self.job_meta.get("job_url", ""),
            )
            return self._build_result(
                success=True,
                proof_type="none",
                proof_value="DRY_RUN",
                proof_confidence=1.0,
            )

        # Final CAPTCHA check
        if await self._detect_captcha():
            return self._build_result(
                success=False,
                error_code="CAPTCHA",
                reroute_to_manual=True,
                reroute_reason="CAPTCHA on Workday review page",
            )

        # Click Submit
        submit_selectors: list[str] = [
            "[data-automation-id='wd-CommandButton_wysiwyg_submit']",
            "button[data-automation-id*='submit' i]",
            "button[aria-label*='Submit' i]",
            "button:has-text('Submit')",
            "[data-automation-id='bottom-navigation-next-button']",
        ]

        submit_clicked: bool = False
        for selector in submit_selectors:
            try:
                await self.page.wait_for_selector(
                    selector, state="visible", timeout=5000
                )
                await self.page.click(selector)
                submit_clicked = True
                self.logger.info(
                    "Workday submit clicked via: %s", selector
                )
                break
            except Exception:
                continue

        if not submit_clicked:
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=(
                    "Workday submit button not found on review page"
                ),
            )

        # Wait for post-submission page
        try:
            await self.page.wait_for_load_state(
                "networkidle", timeout=20000
            )
        except PlaywrightTimeoutError:
            pass  # Some instances don't fire networkidle after submit

        await asyncio.sleep(2)

        # ── Proof Capture ──
        return await self._capture_proof()

    async def _capture_proof(self) -> ApplyResult:
        """Capture proof of successful Workday submission.

        Strategy 1: Thank you / confirmation text on page.
        Strategy 2: URL contains submission confirmation keywords.
        Strategy 3: Workday page body element disappeared.

        Returns:
            ApplyResult with proof on success, or PROOF_FAIL reroute.
        """
        # Strategy 1: confirmation text
        try:
            confirm_el = await self.page.query_selector(
                "h2:has-text('Thank you'), "
                "h1:has-text('Thank you'), "
                "div:has-text('Your application has been submitted'), "
                "[data-automation-id='confirmation'], "
                "div[class*='ThankYou'], "
                "p:has-text('successfully submitted')"
            )
            if confirm_el:
                confirm_text: str = await confirm_el.inner_text()
                return self._build_result(
                    success=True,
                    proof_type="form_disappearance",
                    proof_value=f"Confirmation: {confirm_text[:200]}",
                    proof_confidence=0.95,
                )
        except Exception:
            pass

        # Strategy 2: URL-based proof
        current_url: str = self.page.url
        if any(
            x in current_url
            for x in [
                "confirmation",
                "thank",
                "submitted",
                "success",
                "complete",
            ]
        ):
            return self._build_result(
                success=True,
                proof_type="success_url",
                proof_value=current_url,
                proof_confidence=0.92,
            )

        # Strategy 3: page body disappeared
        wd_body = await self.page.query_selector(
            "[data-automation-id='wd-Page-Body']"
        )
        if not wd_body:
            return self._build_result(
                success=True,
                proof_type="form_disappearance",
                proof_value=current_url,
                proof_confidence=0.70,
            )

        # No proof found — reroute
        return self._build_result(
            success=False,
            error_code="PROOF_FAIL",
            reroute_to_manual=True,
            reroute_reason=(
                "Workday submit clicked but no confirmation proof found. "
                f"Final URL: {current_url}"
            ),
        )

    # ==================================================================
    # Workday-Specific Field Fill (3-Tier Strategy)
    # ==================================================================

    async def _fill_wd_field(
        self,
        selector: str,
        value: str,
        timeout: int = 8000,
    ) -> bool:
        """Fill a Workday React-controlled input field.

        Workday inputs reject standard ``fill()`` — this method uses a
        3-tier strategy:

        Tier 1: ``page.evaluate`` native setter (React fiber bypass).
        Tier 2: Click + select-all + ``type()`` with delay (human sim).
        Tier 3: ``page.fill()`` as last resort.

        All tiers dispatch ``input``, ``change``, and ``blur`` events
        after setting the value.

        Args:
            selector: CSS selector for the Workday input element.
            value: String value to inject.
            timeout: Milliseconds to wait for element visibility.

        Returns:
            True on success, False if all tiers failed.
        """
        if not value:
            return True  # Nothing to fill — not a failure

        try:
            await self.page.wait_for_selector(
                selector, state="visible", timeout=timeout
            )
        except PlaywrightTimeoutError:
            self.logger.debug(
                "Workday field not visible (timeout): %s", selector
            )
            return False

        # Tier 1: React native setter via evaluate
        try:
            result = await self.page.evaluate(
                """
                (args) => {
                    const el = document.querySelector(args.selector);
                    if (!el) return {success: false, reason: 'not found'};

                    const proto = el.tagName === 'TEXTAREA'
                        ? HTMLTextAreaElement.prototype
                        : HTMLInputElement.prototype;
                    const descriptor = Object.getOwnPropertyDescriptor(
                        proto, 'value'
                    );
                    if (descriptor && descriptor.set) {
                        descriptor.set.call(el, args.value);
                        el.dispatchEvent(
                            new Event('input', {bubbles: true})
                        );
                        el.dispatchEvent(
                            new Event('change', {bubbles: true})
                        );
                        el.dispatchEvent(
                            new Event('blur', {bubbles: true})
                        );
                        return {success: true, method: 'native_setter'};
                    }
                    return {success: false, reason: 'no native setter'};
                }
            """,
                {"selector": selector, "value": value},
            )

            if result and result.get("success"):
                # Verify value was accepted by Workday's React state
                await asyncio.sleep(0.3)
                try:
                    actual: str = await self.page.input_value(selector)
                    if actual == value:
                        return True
                except Exception:
                    pass
                # Value not accepted — fall through to Tier 2
        except Exception as e:
            self.logger.debug(
                "Tier 1 fill failed for %s: %s", selector, str(e)
            )

        # Tier 2: click + select all + type with delay (human simulation)
        try:
            await self.page.click(selector)
            await asyncio.sleep(0.2)
            await self.page.keyboard.press("Control+a")
            await self.page.keyboard.press("Delete")
            await asyncio.sleep(0.1)
            await self.page.type(selector, value, delay=50)
            await self.page.keyboard.press("Tab")  # trigger blur/validation
            await asyncio.sleep(0.3)

            try:
                actual = await self.page.input_value(selector)
                if actual == value or actual.strip() == value.strip():
                    self.logger.debug(
                        "Tier 2 (type) succeeded for %s", selector
                    )
                    return True
            except Exception:
                pass
        except Exception as e:
            self.logger.debug(
                "Tier 2 fill failed for %s: %s", selector, str(e)
            )

        # Tier 3: standard fill() fallback
        try:
            await self.page.fill(selector, value)
            return True
        except Exception as e:
            self.logger.warning(
                "All fill tiers failed for Workday field %s: %s",
                selector,
                str(e),
            )
            return False

    # ==================================================================
    # Navigation
    # ==================================================================

    async def _click_next_button(self) -> bool:
        """Click the Workday Next / Save & Continue button.

        Tries all known Workday navigation button selectors. Scrolls
        the button into view first (Workday buttons are sometimes
        off-screen). Waits for page to settle after click.

        Returns:
            True if button was found and clicked, False otherwise.
        """
        next_selectors: list[str] = [
            "[data-automation-id='bottom-navigation-next-button']",
            "[data-automation-id='bottom-navigation-save-continue-button']",
            "[data-automation-id='pageFooter-continueButton']",
            "button[aria-label='Next']",
            "button:has-text('Save and Continue')",
            "button:has-text('Next')",
            "button:has-text('Continue')",
        ]

        for selector in next_selectors:
            try:
                await self.page.wait_for_selector(
                    selector, state="visible", timeout=4000
                )
                # Scroll button into view — Workday buttons sometimes off-screen
                await self.page.eval_on_selector(
                    selector,
                    "el => el.scrollIntoView("
                    "{behavior: 'smooth', block: 'center'})",
                )
                await asyncio.sleep(0.3)
                await self.page.click(selector)

                # Wait for Workday to load next page
                try:
                    await self.page.wait_for_load_state(
                        "networkidle", timeout=12000
                    )
                except PlaywrightTimeoutError:
                    pass  # Non-fatal — Workday sometimes skips networkidle

                await asyncio.sleep(1)  # Workday React re-render settle
                self.logger.debug("Workday next clicked via: %s", selector)
                return True
            except Exception:
                continue

        return False
