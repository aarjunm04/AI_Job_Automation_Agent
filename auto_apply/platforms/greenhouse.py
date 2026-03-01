"""Greenhouse ATS platform-specific Playwright apply module.

Handles the full application flow on Greenhouse boards
(``boards.greenhouse.io``, ``grnh.se`` shortlinks). Supports:

- Standard Greenhouse application forms (embed v1 + v2)
- Custom question fields (``job_application_answers_attributes_N``)
- Resume upload via ``<input type="file">``
- Multi-proof capture (confirmation div, URL change, form disappearance)
- ``dry_run`` mode (fills all fields, skips final submit click)

User profile keys expected (split by caller before passing):
    ``first_name``, ``last_name``, ``email``, ``phone``,
    ``linkedin_url``, ``portfolio_url``, ``location``,
    ``years_experience``.
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

__all__ = ["GreenhouseApply"]


# ---------------------------------------------------------------------------
# Custom Question Helper
# ---------------------------------------------------------------------------


def _get_custom_answer(label_text: str, user_profile: Dict[str, Any]) -> str:
    """Return the best auto-answer for a Greenhouse custom question field.

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

    return ""


# ---------------------------------------------------------------------------
# Greenhouse Apply Module
# ---------------------------------------------------------------------------


class GreenhouseApply(BasePlatformApply):
    """Greenhouse ATS 3-step Playwright apply module.

    Steps:
        1. Navigate + verify ATS fingerprint + CAPTCHA check.
        2. Fill all fields (basic + custom) + upload resume.
        3. Submit + capture proof (confirmation div / URL / form gone).

    Inherits shared helpers from ``BasePlatformApply``.
    """

    PLATFORM_NAME: str = "greenhouse"
    STEPS_TOTAL: int = 3

    async def apply(self) -> ApplyResult:
        """Execute the full Greenhouse application flow.

        Returns:
            ApplyResult with outcome, proof, and routing decision.
        """
        # ── Step 1: Navigate + Verify ATS ──
        result = await self._step_navigate_and_verify()
        if result is not None:
            return result

        # ── Step 2: Fill fields + upload resume ──
        result = await self._step_fill_fields()
        if result is not None:
            return result

        # ── Step 3: Submit + capture proof ──
        return await self._step_submit_and_proof()

    # ------------------------------------------------------------------
    # Step 1: Navigate + Verify
    # ------------------------------------------------------------------

    async def _step_navigate_and_verify(self) -> ApplyResult | None:
        """Navigate to the job page and verify Greenhouse fingerprint.

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

            # Verify Greenhouse DOM fingerprint
            app_body = await self.page.query_selector(
                "div#app_body, div#application"
            )
            if not app_body:
                return self._build_result(
                    success=False,
                    error_code="UNKNOWN_ATS",
                    reroute_to_manual=True,
                    reroute_reason=(
                        "Greenhouse fingerprint not found — ATS mismatch"
                    ),
                )

            # CAPTCHA check on load
            if await self._detect_captcha():
                return self._build_result(
                    success=False,
                    error_code="CAPTCHA",
                    reroute_to_manual=True,
                    reroute_reason="CAPTCHA detected on page load",
                )

            self.steps_completed = 1
            self.logger.info(
                "Step 1 complete: Greenhouse ATS verified for %s", job_url
            )
            return None  # proceed to step 2

        except PlaywrightTimeoutError:
            return self._build_result(
                success=False,
                error_code="TIMEOUT",
                reroute_to_manual=True,
                reroute_reason="Page load timeout",
            )
        except Exception as e:
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=str(e),
            )

    # ------------------------------------------------------------------
    # Step 2: Fill Fields + Upload Resume
    # ------------------------------------------------------------------

    async def _step_fill_fields(self) -> ApplyResult | None:
        """Fill all Greenhouse form fields and upload the resume.

        Returns:
            ApplyResult on failure (reroute), None on success to proceed.
        """
        try:
            # Wait for the application form
            await self.page.wait_for_selector(
                "form#application-form, div#application", timeout=10000
            )

            # Basic profile fields (non-fatal if any missing)
            await self._fill_text_field(
                "input#first_name",
                self.user_profile.get("first_name", ""),
            )
            await self._fill_text_field(
                "input#last_name",
                self.user_profile.get("last_name", ""),
            )
            await self._fill_text_field(
                "input#email",
                self.user_profile.get("email", ""),
            )
            await self._fill_text_field(
                "input#phone",
                self.user_profile.get("phone", ""),
            )

            # LinkedIn and website (optional, present in most GH forms)
            await self._fill_text_field(
                "input#linkedin_profile",
                self.user_profile.get("linkedin_url", ""),
            )
            await self._fill_text_field(
                "input#website",
                self.user_profile.get("portfolio_url", ""),
            )

            # Custom questions — dynamic IDs with index N
            await self._fill_custom_questions()

            # Resume upload
            upload_ok: bool = await self._upload_resume("input#resume")
            if not upload_ok:
                upload_ok = await self._upload_resume(
                    "input[type='file'][name*='resume'], input[type='file']"
                )
            if not upload_ok:
                self.logger.warning(
                    "Resume upload failed for %s — continuing without resume",
                    self.job_meta.get("job_url", ""),
                )

            self.steps_completed = 2
            self.logger.info(
                "Step 2 complete: fields filled for %s",
                self.job_meta.get("job_url", ""),
            )
            return None  # proceed to step 3

        except PlaywrightTimeoutError:
            return self._build_result(
                success=False,
                error_code="TIMEOUT",
                reroute_to_manual=True,
                reroute_reason=(
                    "Form fill timeout — Greenhouse form did not load"
                ),
            )
        except Exception as e:
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=f"Form fill error: {str(e)}",
            )

    async def _fill_custom_questions(self) -> None:
        """Scan and fill Greenhouse custom question fields.

        Finds all visible text/textarea inputs with IDs matching the
        Greenhouse pattern (``*text_value*``) and attempts to auto-answer
        based on label text keyword matching.
        """
        try:
            custom_inputs = await self.page.query_selector_all(
                "input[id*='text_value'], textarea[id*='text_value']"
            )
            for inp in custom_inputs:
                inp_id: str = await inp.get_attribute("id") or ""
                if not inp_id:
                    continue

                # Only fill empty fields
                current_val: str = await inp.input_value()
                if current_val:
                    continue

                # Resolve label text
                label: str = await self.page.evaluate(
                    """
                    (id) => {
                        const label = document.querySelector(
                            `label[for="${id}"]`
                        );
                        return label ? label.innerText.trim() : '';
                    }
                """,
                    inp_id,
                )

                answer: str = _get_custom_answer(label, self.user_profile)
                if answer:
                    await self._fill_react_input(f"#{inp_id}", answer)
        except Exception as e:
            self.logger.warning(
                "Custom questions scan error: %s — continuing", str(e)
            )

    # ------------------------------------------------------------------
    # Step 3: Submit + Capture Proof
    # ------------------------------------------------------------------

    async def _step_submit_and_proof(self) -> ApplyResult:
        """Submit the Greenhouse form and capture proof of submission.

        Returns:
            ApplyResult with proof on success, or reroute on failure.
        """
        try:
            # Dry-run: skip submit
            if self.dry_run:
                self.logger.info(
                    "[DRY_RUN] Would submit Greenhouse form for %s",
                    self.job_meta.get("job_url", ""),
                )
                self.steps_completed = 3
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
                    reroute_reason="CAPTCHA appeared before submit",
                )

            # Click submit
            submit_clicked: bool = await self._click_next_or_continue(
                [
                    "button[data-provides='submit-application']",
                    "input[type='submit']",
                    "button[type='submit']",
                    "button:has-text('Submit Application')",
                    "button:has-text('Submit')",
                ]
            )
            if not submit_clicked:
                return self._build_result(
                    success=False,
                    error_code="NAV_FAIL",
                    reroute_to_manual=True,
                    reroute_reason="Submit button not found",
                )

            # Wait for confirmation response
            await asyncio.sleep(2)

            # Proof strategy 1: confirmation div
            confirmation = await self.page.query_selector(
                "div#confirmation, div.confirmation, "
                "div[class*='confirmation']"
            )
            if confirmation:
                self.steps_completed = 3
                return self._build_result(
                    success=True,
                    proof_type="form_disappearance",
                    proof_value=self.page.url,
                    proof_confidence=0.90,
                )

            # Proof strategy 2: URL change to confirmation/thanks
            current_url: str = self.page.url
            if any(
                x in current_url
                for x in [
                    "confirmation",
                    "thank",
                    "thanks",
                    "submitted",
                    "success",
                ]
            ):
                self.steps_completed = 3
                return self._build_result(
                    success=True,
                    proof_type="success_url",
                    proof_value=current_url,
                    proof_confidence=0.95,
                )

            # Proof strategy 3: form disappeared from DOM
            form_still = await self.page.query_selector(
                "form#application-form"
            )
            if not form_still:
                self.steps_completed = 3
                return self._build_result(
                    success=True,
                    proof_type="form_disappearance",
                    proof_value=current_url,
                    proof_confidence=0.75,
                )

            # No proof found — reroute
            return self._build_result(
                success=False,
                error_code="PROOF_FAIL",
                reroute_to_manual=True,
                reroute_reason=(
                    "Could not confirm Greenhouse submission — "
                    "no proof found"
                ),
            )

        except PlaywrightTimeoutError:
            return self._build_result(
                success=False,
                error_code="TIMEOUT",
                reroute_to_manual=True,
                reroute_reason="Submit/confirmation timeout",
            )
        except Exception as e:
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=f"Submit error: {str(e)}",
            )
