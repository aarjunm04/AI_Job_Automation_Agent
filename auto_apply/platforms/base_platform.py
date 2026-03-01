"""Base class for all platform-specific apply modules.

Provides shared Playwright helpers for field filling (standard + React),
dropdown selection, file upload, CAPTCHA detection, navigation, and
standardised result/error dataclasses. All ATS platform modules
(greenhouse.py, lever.py, etc.) inherit from BasePlatformApply.
"""

from __future__ import annotations

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

logger = logging.getLogger(__name__)

__all__ = ["BasePlatformApply", "ApplyResult", "ApplyError"]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class ApplyResult:
    """Standardised return value from every platform ``apply()`` call.

    Attributes:
        success: Whether the application was successfully submitted.
        proof_type: Type of proof captured — ``"confirmation_number"``,
            ``"success_url"``, ``"form_disappearance"``, or ``"none"``.
        proof_value: Raw proof string (URL, confirmation number, etc.).
        proof_confidence: Confidence in the proof (0.0–1.0).
        error_code: Structured error code if failed, else None.
        reroute_to_manual: Whether to reroute to the manual apply queue.
        reroute_reason: Human-readable reason for rerouting.
        steps_completed: Number of steps completed before exit.
        steps_total: Total expected steps for this platform.
        platform: ATS platform name (e.g. ``"greenhouse"``).
        job_url: URL of the job that was being applied to.
    """

    success: bool
    proof_type: str
    proof_value: Optional[str]
    proof_confidence: float
    error_code: Optional[str]
    reroute_to_manual: bool
    reroute_reason: Optional[str]
    steps_completed: int
    steps_total: int
    platform: str
    job_url: str


@dataclass
class ApplyError:
    """Structured error from a platform apply step.

    Attributes:
        code: Error code (e.g. ``"TIMEOUT"``, ``"CAPTCHA"``,
            ``"UNKNOWN_ATS"``, ``"PROOF_FAIL"``, ``"UPLOAD_FAIL"``,
            ``"NAV_FAIL"``).
        message: Human-readable error description.
        step: Step number where the error occurred.
        reroute: Whether this error warrants manual reroute.
    """

    code: str
    message: str
    step: int
    reroute: bool = True


# ---------------------------------------------------------------------------
# Base Class
# ---------------------------------------------------------------------------


class BasePlatformApply(ABC):
    """Abstract base for all ATS-specific Playwright apply modules.

    Subclasses must set ``PLATFORM_NAME`` and ``STEPS_TOTAL`` class
    variables and implement the ``apply()`` coroutine.

    Constructor Args:
        page: Live Playwright async Page object.
        job_meta: Dict with keys: ``job_url``, ``title``, ``company``,
            ``platform``, ``fit_score``, ``resume_suggested``, ``run_id``.
        user_profile: Dict loaded from env user profile keys.
            Expected keys: ``first_name``, ``last_name``, ``email``,
            ``phone``, ``linkedin_url``, ``portfolio_url``, ``location``,
            ``years_experience``.
        dry_run: If True, fill all fields but do NOT click Submit.
    """

    PLATFORM_NAME: str = "base"
    STEPS_TOTAL: int = 1

    def __init__(
        self,
        page: Page,
        job_meta: Dict[str, Any],
        user_profile: Dict[str, Any],
        dry_run: bool = False,
    ) -> None:
        self.page = page
        self.job_meta = job_meta
        self.user_profile = user_profile
        self.dry_run = dry_run
        self.steps_completed: int = 0
        self.resume_path: str = self._resolve_resume_path()
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

    # ------------------------------------------------------------------
    # Resume Resolution
    # ------------------------------------------------------------------

    def _resolve_resume_path(self) -> str:
        """Resolve the absolute path to the suggested resume PDF.

        Priority:
            1. ``resume_suggested`` from ``job_meta`` in ``RESUME_DIR``.
            2. ``DEFAULT_RESUME`` env var in ``RESUME_DIR``.
            3. First ``.pdf`` found in ``RESUME_DIR``.

        Returns:
            Absolute path string. May point to a non-existent file if no
            PDF is found — the upload step will handle the error.
        """
        resume_dir: str = os.getenv("RESUME_DIR", "app/resumes")
        suggested: str = self.job_meta.get(
            "resume_suggested",
            os.getenv("DEFAULT_RESUME", "AarjunGen.pdf"),
        )
        full_path: str = os.path.join(resume_dir, suggested)
        if os.path.exists(full_path):
            return full_path

        # Fallback to default resume
        default: str = os.path.join(
            resume_dir, os.getenv("DEFAULT_RESUME", "AarjunGen.pdf")
        )
        if os.path.exists(default):
            self.logger.warning(
                "Suggested resume %s not found, using default: %s",
                suggested,
                default,
            )
            return default

        # Last resort: find any PDF in resume_dir
        try:
            pdfs: list[str] = [
                f for f in os.listdir(resume_dir) if f.endswith(".pdf")
            ]
            if pdfs:
                return os.path.join(resume_dir, pdfs[0])
        except OSError:
            pass

        self.logger.error("No resume PDF found in %s", resume_dir)
        return default

    # ------------------------------------------------------------------
    # Field Filling Helpers
    # ------------------------------------------------------------------

    async def _fill_text_field(
        self,
        selector: str,
        value: str,
        timeout: int = 5000,
        clear_first: bool = True,
    ) -> bool:
        """Fill a text input by CSS selector.

        Tries ``page.fill()``, falls back to triple-click + ``type()``
        for React-controlled inputs.

        Args:
            selector: CSS selector for the target input.
            value: Value to fill.
            timeout: Milliseconds to wait for selector visibility.
            clear_first: Whether to clear existing value before filling.

        Returns:
            True on success, False on failure.
        """
        if not value:
            return False
        try:
            await self.page.wait_for_selector(
                selector, state="visible", timeout=timeout
            )
            if clear_first:
                await self.page.triple_click(selector)
                await self.page.keyboard.press("Control+a")
                await self.page.keyboard.press("Delete")
            await self.page.fill(selector, value)
            return True
        except PlaywrightTimeoutError:
            self.logger.debug("Selector not found (timeout): %s", selector)
            return False
        except Exception as e:
            self.logger.debug(
                "fill() failed for %s: %s — trying type()", selector, str(e)
            )
            try:
                await self.page.click(selector)
                await self.page.keyboard.press("Control+a")
                await self.page.type(selector, value, delay=30)
                return True
            except Exception as e2:
                self.logger.warning(
                    "type() also failed for %s: %s", selector, str(e2)
                )
                return False

    async def _fill_react_input(
        self,
        selector: str,
        value: str,
        timeout: int = 5000,
    ) -> bool:
        """Fill a React-controlled input using native setter injection.

        Uses ``Object.getOwnPropertyDescriptor`` on the appropriate
        prototype to bypass React's synthetic event system.

        Args:
            selector: CSS selector for the target input.
            value: Value to inject.
            timeout: Milliseconds to wait for selector visibility.

        Returns:
            True on success, False on failure.
        """
        if not value:
            return False
        try:
            await self.page.wait_for_selector(
                selector, state="visible", timeout=timeout
            )
            await self.page.evaluate(
                """
                (args) => {
                    const el = document.querySelector(args.selector);
                    if (!el) return false;
                    const proto = el.tagName === 'TEXTAREA'
                        ? HTMLTextAreaElement.prototype
                        : HTMLInputElement.prototype;
                    const setter = Object.getOwnPropertyDescriptor(
                        proto, 'value'
                    ).set;
                    setter.call(el, args.value);
                    el.dispatchEvent(new Event('input', {bubbles: true}));
                    el.dispatchEvent(new Event('change', {bubbles: true}));
                    return true;
                }
            """,
                {"selector": selector, "value": value},
            )
            return True
        except Exception as e:
            self.logger.warning(
                "React fill failed for %s: %s", selector, str(e)
            )
            return False

    async def _select_option(
        self,
        selector: str,
        value: str,
        timeout: int = 5000,
    ) -> bool:
        """Select a dropdown option by value or label text.

        Args:
            selector: CSS selector for the ``<select>`` element.
            value: Option value or visible label text.
            timeout: Milliseconds to wait for selector visibility.

        Returns:
            True on success, False on failure.
        """
        try:
            await self.page.wait_for_selector(
                selector, state="visible", timeout=timeout
            )
            try:
                await self.page.select_option(selector, value=value)
                return True
            except Exception:
                await self.page.select_option(selector, label=value)
                return True
        except Exception as e:
            self.logger.warning(
                "select_option failed for %s: %s", selector, str(e)
            )
            return False

    # ------------------------------------------------------------------
    # File Upload
    # ------------------------------------------------------------------

    async def _upload_resume(
        self,
        file_input_selector: str,
        timeout: int = 10000,
    ) -> bool:
        """Upload the resume PDF to a file input.

        Handles both direct ``<input type="file">`` and file chooser
        dialog patterns.

        Args:
            file_input_selector: CSS selector for the file input or
                upload trigger button.
            timeout: Milliseconds to wait for file chooser.

        Returns:
            True on success, False on failure.
        """
        # Try direct file input first
        try:
            file_input = await self.page.query_selector(file_input_selector)
            if file_input:
                await file_input.set_input_files(self.resume_path)
                self.logger.info(
                    "Resume uploaded via direct input: %s", self.resume_path
                )
                return True
        except Exception as e:
            self.logger.debug(
                "Direct file input failed: %s — trying file chooser", str(e)
            )

        # Fallback: file chooser dialog trigger
        try:
            async with self.page.expect_file_chooser(
                timeout=timeout
            ) as fc_info:
                await self.page.click(file_input_selector)
            file_chooser = await fc_info.value
            await file_chooser.set_files(self.resume_path)
            self.logger.info(
                "Resume uploaded via file chooser: %s", self.resume_path
            )
            return True
        except Exception as e:
            self.logger.error(
                "Resume upload failed for selector %s: %s",
                file_input_selector,
                str(e),
            )
            return False

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    async def _click_next_or_continue(
        self,
        selectors: List[str],
        timeout: int = 5000,
    ) -> bool:
        """Try a list of selectors to click Next/Continue/Submit buttons.

        Iterates through ``selectors`` in order. Returns on the first
        successful click after waiting for ``networkidle``.

        Args:
            selectors: Ordered list of CSS selectors to try.
            timeout: Milliseconds to wait for each selector.

        Returns:
            True if any selector was clicked, False if none found.
        """
        for selector in selectors:
            try:
                await self.page.wait_for_selector(
                    selector, state="visible", timeout=timeout
                )
                await self.page.click(selector)
                await self.page.wait_for_load_state(
                    "networkidle", timeout=8000
                )
                return True
            except Exception:
                continue
        self.logger.warning(
            "No next/continue button found from selectors: %s", selectors
        )
        return False

    # ------------------------------------------------------------------
    # CAPTCHA Detection
    # ------------------------------------------------------------------

    async def _detect_captcha(self) -> bool:
        """Detect common CAPTCHA patterns on the current page.

        Checks for reCAPTCHA, hCaptcha, and generic CAPTCHA DOM indicators.

        Returns:
            True if a CAPTCHA element is detected, False otherwise.
        """
        captcha_indicators: list[str] = [
            "iframe[src*='recaptcha']",
            "iframe[src*='hcaptcha']",
            "div.g-recaptcha",
            "div[data-sitekey]",
            "#captcha",
            "div.captcha",
            "iframe[title*='challenge']",
        ]
        for selector in captcha_indicators:
            try:
                element = await self.page.query_selector(selector)
                if element:
                    self.logger.warning("CAPTCHA detected: %s", selector)
                    return True
            except Exception:
                continue
        return False

    # ------------------------------------------------------------------
    # Result Builder
    # ------------------------------------------------------------------

    def _build_result(
        self,
        success: bool,
        proof_type: str = "none",
        proof_value: Optional[str] = None,
        proof_confidence: float = 0.0,
        error_code: Optional[str] = None,
        reroute_to_manual: bool = False,
        reroute_reason: Optional[str] = None,
    ) -> ApplyResult:
        """Construct a standardised ``ApplyResult``.

        Args:
            success: Whether the application succeeded.
            proof_type: Type of proof captured.
            proof_value: Raw proof string.
            proof_confidence: Confidence score (0.0–1.0).
            error_code: Structured error code if failed.
            reroute_to_manual: Whether to reroute to manual queue.
            reroute_reason: Human-readable reroute reason.

        Returns:
            Populated ApplyResult dataclass.
        """
        return ApplyResult(
            success=success,
            proof_type=proof_type,
            proof_value=proof_value,
            proof_confidence=proof_confidence,
            error_code=error_code,
            reroute_to_manual=reroute_to_manual,
            reroute_reason=reroute_reason,
            steps_completed=self.steps_completed,
            steps_total=self.STEPS_TOTAL,
            platform=self.PLATFORM_NAME,
            job_url=self.job_meta.get("job_url", ""),
        )

    # ------------------------------------------------------------------
    # Abstract Interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def apply(self) -> ApplyResult:
        """Execute the full application flow for this platform.

        Returns:
            ApplyResult with outcome, proof, and routing decision.
        """
        ...
