"""Lever ATS platform-specific Playwright apply module.

Handles the full application flow on Lever boards
(``jobs.lever.co/{company}/{job_id}``). Supports:

- Direct Lever pages and iframe-embedded application forms
- Single "name" field (full name) and standard contact fields
- Custom card-based questions (``cards[{card_id}][field{N}]``)
- Resume upload via data-qa button or direct file input
- Multi-proof capture (thanks text, URL change, form disappearance)
- ``dry_run`` mode (fills all fields, skips final submit click)

User profile keys expected (split by caller before passing):
    ``first_name``, ``last_name``, ``email``, ``phone``,
    ``linkedin_url``, ``portfolio_url``, ``location``,
    ``years_experience``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any, Union

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

__all__ = ["LeverApply"]


# Type alias for Playwright Page or Frame context
PageOrFrame = Union[Page, Frame]


# ---------------------------------------------------------------------------
# Custom Question Helper
# ---------------------------------------------------------------------------


def _get_lever_custom_answer(
    label_text: str, user_profile: Dict[str, Any]
) -> str:
    """Return the best auto-answer for a Lever custom question field.

    Checks ``label_text`` for common question patterns and maps to
    ``user_profile`` values. Returns empty string if no match — the field
    is left blank.

    Args:
        label_text: The label text of the custom question field.
        user_profile: Dict of user profile values from env.

    Returns:
        String answer or ``""`` if no mapping found.
    """
    label_lower: str = label_text.lower()

    if any(x in label_lower for x in
           ["years", "experience", "how long", "how many"]):
        return str(user_profile.get("years_experience", ""))

    if any(x in label_lower for x in ["location", "city", "where"]):
        return str(user_profile.get("location", ""))

    if any(x in label_lower for x in
           ["linkedin", "profile url", "linkedin url"]):
        return str(user_profile.get("linkedin_url", ""))

    if any(x in label_lower for x in
           ["website", "portfolio", "github", "personal url"]):
        return str(user_profile.get("portfolio_url", ""))

    if any(x in label_lower for x in
           ["salary", "ctc", "compensation", "expected"]):
        return "Open to discussion"

    if any(x in label_lower for x in
           ["sponsor", "visa", "authoris", "authoriz", "work auth"]):
        return "No"

    if any(x in label_lower for x in
           ["available", "start date", "notice", "join"]):
        return "Immediately"

    if any(x in label_lower for x in
           ["referral", "how did you hear", "how did you find"]):
        return "Online job board"

    # Cover letter / motivation — leave blank (too personal to auto-fill)
    if any(x in label_lower for x in
           ["cover letter", "why", "motivation"]):
        return ""

    return ""


# ---------------------------------------------------------------------------
# Lever Apply Module
# ---------------------------------------------------------------------------


class LeverApply(BasePlatformApply):
    """Lever ATS 2-step Playwright apply module with iframe support.

    Steps:
        1. Navigate + verify ATS + detect iframe + fill fields + upload.
        2. Submit + capture proof (thanks text / URL / form gone).

    Handles both direct ``jobs.lever.co`` pages and iframe-embedded
    application forms. All field operations on the detected work context
    (top-level page or inner frame).
    """

    PLATFORM_NAME: str = "lever"
    STEPS_TOTAL: int = 2

    def __init__(
        self,
        page: Page,
        job_meta: Dict[str, Any],
        user_profile: Dict[str, Any],
        dry_run: bool = False,
    ) -> None:
        super().__init__(page, job_meta, user_profile, dry_run)
        self._work_page: PageOrFrame | None = None

    # ------------------------------------------------------------------
    # Iframe Detection
    # ------------------------------------------------------------------

    async def _get_work_context(self) -> PageOrFrame:
        """Detect and cache the Playwright context for form operations.

        Checks if the Lever application form is embedded in an iframe.
        If so, returns the iframe's content frame. Otherwise returns the
        top-level page.

        Returns:
            Page or Frame to use for all fill/click operations.
        """
        if self._work_page is not None:
            return self._work_page

        try:
            iframe_el = await self.page.query_selector(
                "iframe[src*='lever.co'], iframe[src*='jobs.lever']"
            )
            if iframe_el:
                frame = await iframe_el.content_frame()
                if frame is not None:
                    self.logger.info("Lever form detected inside iframe")
                    self._work_page = frame
                    return self._work_page
        except Exception as e:
            self.logger.debug("Iframe detection error: %s", str(e))

        self._work_page = self.page
        return self._work_page

    # ------------------------------------------------------------------
    # Context-Aware Helpers
    # ------------------------------------------------------------------

    async def _fill_on_context(
        self,
        context: PageOrFrame,
        selector: str,
        value: str,
        timeout: int = 5000,
    ) -> bool:
        """Fill a text field on a given Playwright Page or Frame context.

        Same logic as ``BasePlatformApply._fill_text_field`` but accepts
        any page/frame context.

        Args:
            context: Playwright Page or Frame.
            selector: CSS selector for the target input.
            value: Value to fill.
            timeout: Milliseconds to wait for selector visibility.

        Returns:
            True on success, False on failure.
        """
        if not value:
            return False
        try:
            await context.wait_for_selector(
                selector, state="visible", timeout=timeout
            )
            await context.click(selector, click_count=3)
            await context.fill(selector, value)
            return True
        except PlaywrightTimeoutError:
            self.logger.debug(
                "Selector not found on context (timeout): %s", selector
            )
            return False
        except Exception as e:
            self.logger.debug(
                "fill() failed on context for %s: %s — trying type()",
                selector,
                str(e),
            )
            try:
                await context.click(selector)
                await context.type(selector, value, delay=30)
                return True
            except Exception as e2:
                self.logger.warning(
                    "type() also failed on context for %s: %s",
                    selector,
                    str(e2),
                )
                return False

    async def _upload_resume_on_context(
        self,
        context: PageOrFrame,
        selector: str,
        timeout: int = 10000,
    ) -> bool:
        """Upload the resume PDF on a given Playwright Page or Frame.

        Args:
            context: Playwright Page or Frame.
            selector: CSS selector for the file input.
            timeout: Milliseconds to wait for file chooser.

        Returns:
            True on success, False on failure.
        """
        try:
            file_input = await context.query_selector(selector)
            if file_input:
                await file_input.set_input_files(self.resume_path)
                self.logger.info(
                    "Resume uploaded via context input: %s",
                    self.resume_path,
                )
                return True
        except Exception as e:
            self.logger.debug(
                "Direct context file input failed: %s", str(e)
            )

        try:
            async with context.expect_file_chooser(
                timeout=timeout
            ) as fc_info:
                await context.click(selector)
            fc = await fc_info.value
            await fc.set_files(self.resume_path)
            self.logger.info(
                "Resume uploaded via context file chooser: %s",
                self.resume_path,
            )
            return True
        except Exception as e:
            self.logger.error(
                "Context resume upload failed for %s: %s", selector, str(e)
            )
            return False

    # ------------------------------------------------------------------
    # Main Apply Flow
    # ------------------------------------------------------------------

    async def apply(self) -> ApplyResult:
        """Execute the full Lever application flow.

        Returns:
            ApplyResult with outcome, proof, and routing decision.
        """
        # ── Step 1: Navigate + Verify + Fill + Upload ──
        result = await self._step_fill()
        if result is not None:
            return result

        # ── Step 2: Submit + Capture Proof ──
        return await self._step_submit_and_proof()

    # ------------------------------------------------------------------
    # Step 1: Navigate + Verify + Fill + Upload
    # ------------------------------------------------------------------

    async def _step_fill(self) -> ApplyResult | None:
        """Navigate, verify Lever fingerprint, fill fields, upload resume.

        Returns:
            ApplyResult on failure (reroute), None on success to proceed.
        """
        self.steps_completed = 0
        try:
            # Navigate if not already on the job page
            current_url: str = self.page.url
            job_url: str = self.job_meta.get("job_url", "")
            if not current_url.startswith(job_url[:30]):
                await self.page.goto(
                    job_url,
                    wait_until="domcontentloaded",
                    timeout=20000,
                )
            await self.page.wait_for_load_state(
                "networkidle", timeout=15000
            )

            # Detect iframe embedding
            work_page: PageOrFrame = await self._get_work_context()

            # Verify Lever DOM fingerprint on work context
            form_el = await work_page.query_selector(
                "div.application-form, form[action*='lever.co'], "
                "div.lever-job-posting"
            )
            if not form_el:
                return self._build_result(
                    success=False,
                    error_code="UNKNOWN_ATS",
                    reroute_to_manual=True,
                    reroute_reason="Lever fingerprint not found",
                )

            # CAPTCHA check (always on top-level page)
            if await self._detect_captcha():
                return self._build_result(
                    success=False,
                    error_code="CAPTCHA",
                    reroute_to_manual=True,
                    reroute_reason="CAPTCHA detected on page load",
                )

            # ── Fill fields ──

            # Full name (Lever uses single name field)
            full_name: str = (
                f"{self.user_profile.get('first_name', '')} "
                f"{self.user_profile.get('last_name', '')}"
            ).strip()
            await self._fill_on_context(
                work_page, "input[name='name']", full_name
            )

            # Contact fields
            await self._fill_on_context(
                work_page,
                "input[name='email']",
                self.user_profile.get("email", ""),
            )
            await self._fill_on_context(
                work_page,
                "input[name='phone']",
                self.user_profile.get("phone", ""),
            )

            # URL fields
            await self._fill_on_context(
                work_page,
                "input[name='urls[LinkedIn]']",
                self.user_profile.get("linkedin_url", ""),
            )
            await self._fill_on_context(
                work_page,
                "input[name='urls[GitHub]']",
                self.user_profile.get("portfolio_url", ""),
            )
            await self._fill_on_context(
                work_page,
                "input[name='urls[Portfolio]']",
                self.user_profile.get("portfolio_url", ""),
            )

            # Custom card-based questions
            await self._fill_card_questions(work_page)

            # Resume upload
            upload_ok: bool = await self._upload_lever_resume(work_page)
            if not upload_ok:
                self.logger.warning(
                    "Resume upload failed for Lever job: %s", job_url
                )

            self.steps_completed = 1
            self.logger.info(
                "Step 1 complete: Lever fields filled for %s", job_url
            )
            return None  # proceed to step 2

        except PlaywrightTimeoutError:
            return self._build_result(
                success=False,
                error_code="TIMEOUT",
                reroute_to_manual=True,
                reroute_reason="Lever form load timeout",
            )
        except Exception as e:
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=str(e),
            )

    async def _fill_card_questions(self, work_page: PageOrFrame) -> None:
        """Scan and fill Lever custom card-based question fields.

        Lever uses ``cards[{card_id}][field{N}]`` naming for custom
        questions within application forms.

        Args:
            work_page: Playwright Page or Frame containing the form.
        """
        try:
            card_inputs = await work_page.query_selector_all(
                "input[name*='cards['], textarea[name*='cards['], "
                "select[name*='cards[']"
            )
            for inp in card_inputs:
                try:
                    tag: str = await inp.evaluate(
                        "el => el.tagName.toLowerCase()"
                    )

                    # Skip already-filled fields
                    if tag != "select":
                        current_val: str = await inp.input_value()
                        if current_val:
                            continue

                    inp_name: str = await inp.get_attribute("name") or ""
                    if not inp_name:
                        continue

                    # Resolve label text
                    label: str = await work_page.evaluate(
                        """
                        (name) => {
                            const el = document.querySelector(
                                `[name="${name}"]`
                            );
                            if (!el) return '';
                            const group = el.closest(
                                '.application-question, .form-group, li'
                            );
                            if (!group) return '';
                            const lbl = group.querySelector(
                                'label, .question-basic-label'
                            );
                            return lbl ? lbl.innerText.trim() : '';
                        }
                    """,
                        inp_name,
                    )

                    answer: str = _get_lever_custom_answer(
                        label, self.user_profile
                    )
                    if not answer:
                        continue

                    if tag in ("input", "textarea"):
                        await self._fill_on_context(
                            work_page, f"[name='{inp_name}']", answer
                        )
                    elif tag == "select":
                        try:
                            await work_page.select_option(
                                f"[name='{inp_name}']", label=answer
                            )
                        except Exception:
                            pass  # non-fatal
                except Exception as card_err:
                    self.logger.debug(
                        "Card question fill error: %s", str(card_err)
                    )
                    continue
        except Exception as e:
            self.logger.warning(
                "Lever card questions scan error: %s — continuing", str(e)
            )

    async def _upload_lever_resume(self, work_page: PageOrFrame) -> bool:
        """Upload resume using Lever's upload button or file input.

        Tries the ``data-qa='btn-upload-resume'`` button first, then
        falls back to a direct ``<input type="file">``.

        Args:
            work_page: Playwright Page or Frame containing the form.

        Returns:
            True on success, False on failure.
        """
        # Try Lever's upload button (file chooser dialog)
        try:
            async with work_page.expect_file_chooser(
                timeout=5000
            ) as fc_info:
                await work_page.click(
                    "div[data-qa='btn-upload-resume'], "
                    "button[data-qa='btn-upload-resume']"
                )
            fc = await fc_info.value
            await fc.set_files(self.resume_path)
            self.logger.info(
                "Resume uploaded via Lever btn-upload-resume: %s",
                self.resume_path,
            )
            return True
        except Exception as e:
            self.logger.debug(
                "Lever upload button failed: %s — trying file input", str(e)
            )

        # Fallback: direct file input
        return await self._upload_resume_on_context(
            work_page, "input[type='file']"
        )

    # ------------------------------------------------------------------
    # Step 2: Submit + Capture Proof
    # ------------------------------------------------------------------

    async def _step_submit_and_proof(self) -> ApplyResult:
        """Submit the Lever form and capture proof of submission.

        Returns:
            ApplyResult with proof on success, or reroute on failure.
        """
        try:
            work_page: PageOrFrame = await self._get_work_context()

            # Dry-run: skip submit
            if self.dry_run:
                self.logger.info(
                    "[DRY_RUN] Would submit Lever form for %s",
                    self.job_meta.get("job_url", ""),
                )
                self.steps_completed = 2
                return self._build_result(
                    success=True,
                    proof_type="none",
                    proof_value="DRY_RUN",
                    proof_confidence=1.0,
                )

            # CAPTCHA check before submit
            if await self._detect_captcha():
                return self._build_result(
                    success=False,
                    error_code="CAPTCHA",
                    reroute_to_manual=True,
                    reroute_reason="CAPTCHA before Lever submit",
                )

            # Click submit on work context
            submit_clicked: bool = False
            submit_selectors: list[str] = [
                "button[data-qa='btn-submit']",
                "button[type='submit']",
                "input[type='submit']",
                "button:has-text('Submit Application')",
                "button:has-text('Apply')",
            ]
            for selector in submit_selectors:
                try:
                    await work_page.wait_for_selector(
                        selector, state="visible", timeout=3000
                    )
                    await work_page.click(selector)
                    submit_clicked = True
                    break
                except Exception:
                    continue

            if not submit_clicked:
                return self._build_result(
                    success=False,
                    error_code="NAV_FAIL",
                    reroute_to_manual=True,
                    reroute_reason="Lever submit button not found",
                )

            # Wait for confirmation response
            await asyncio.sleep(2)

            # Proof: "Thanks for applying!" text on work_page or self.page
            for check_ctx in [work_page, self.page]:
                try:
                    thanks = await check_ctx.query_selector(
                        "h1:has-text('Thanks'), div[class*='thanks'], "
                        "div[class*='confirmation'], div[class*='success']"
                    )
                    if thanks:
                        self.steps_completed = 2
                        return self._build_result(
                            success=True,
                            proof_type="form_disappearance",
                            proof_value=self.page.url,
                            proof_confidence=0.92,
                        )
                except Exception:
                    continue

            # URL-based proof
            current_url: str = self.page.url
            if any(
                x in current_url
                for x in ["confirmation", "thank", "applied", "success"]
            ):
                self.steps_completed = 2
                return self._build_result(
                    success=True,
                    proof_type="success_url",
                    proof_value=current_url,
                    proof_confidence=0.95,
                )

            # Form disappearance proof
            try:
                form_gone: bool = not await work_page.query_selector(
                    "div.application-form"
                )
            except Exception:
                form_gone = True  # context destroyed = form is gone

            if form_gone:
                self.steps_completed = 2
                return self._build_result(
                    success=True,
                    proof_type="form_disappearance",
                    proof_value=self.page.url,
                    proof_confidence=0.70,
                )

            # No proof found — reroute
            return self._build_result(
                success=False,
                error_code="PROOF_FAIL",
                reroute_to_manual=True,
                reroute_reason=(
                    "Lever submission unconfirmed — no proof found"
                ),
            )

        except PlaywrightTimeoutError:
            return self._build_result(
                success=False,
                error_code="TIMEOUT",
                reroute_to_manual=True,
                reroute_reason="Lever submit timeout",
            )
        except Exception as e:
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=f"Lever submit error: {str(e)}",
            )
