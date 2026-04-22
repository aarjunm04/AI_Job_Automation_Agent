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
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import agentops
from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

try:
    from playwright_stealth import stealth_sync
    STEALTH_AVAILABLE: bool = True
except ImportError:
    STEALTH_AVAILABLE = False

logger = logging.getLogger(__name__)

__all__ = ["BasePlatformApply", "ApplyResult", "ApplyError", "_apply_page_stealth"]


# ---------------------------------------------------------------------------
# Stealth Configuration & Helpers
# ---------------------------------------------------------------------------
STEALTH_ENABLED: bool = os.getenv("PLAYWRIGHT_STEALTH_ENABLED", "true").lower() == "true"
PLAYWRIGHT_USER_AGENT: str = os.getenv(
    "PLAYWRIGHT_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


async def _apply_page_stealth(page: Page) -> bool:
    """Apply playwright-stealth configuration to a page with fail-soft behavior.
    
    Applies stealth_sync (if available), sets viewport size, and user agent.
    All exceptions are caught and logged as warnings — stealth application 
    never crashes the apply pipeline.
    
    Args:
        page: Playwright async Page object to apply stealth to.
        
    Returns:
        True if stealth applied successfully, False on failure (non-fatal).
    """
    if not STEALTH_ENABLED or not STEALTH_AVAILABLE:
        logger.debug(
            "_apply_page_stealth: stealth disabled or unavailable "
            "(STEALTH_ENABLED=%s, STEALTH_AVAILABLE=%s)",
            STEALTH_ENABLED,
            STEALTH_AVAILABLE,
        )
        return False

    try:
        # Apply stealth_sync (strips Playwright fingerprints)
        stealth_sync(page)
        logger.debug("_apply_page_stealth: stealth_sync applied successfully")
        
        # Set realistic viewport size (Chrome desktop window)
        try:
            await page.set_viewport_size({"width": 1920, "height": 1080})
            logger.debug("_apply_page_stealth: viewport set to 1920x1080")
        except Exception as vp_exc:
            logger.warning("_apply_page_stealth: viewport set failed: %s", vp_exc)
        
        # Set realistic user agent via extra HTTP headers
        try:
            await page.set_extra_http_headers({"User-Agent": PLAYWRIGHT_USER_AGENT})
            logger.debug("_apply_page_stealth: user agent header set")
        except Exception as ua_exc:
            logger.warning("_apply_page_stealth: user agent header failed: %s", ua_exc)
        
        return True
        
    except Exception as exc:
        logger.warning(
            "_apply_page_stealth: stealth application failed (continuing): %s",
            exc,
        )
        return False


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
        page: Optional["Page"] = None,
        job_meta: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
    ) -> None:
        # logger MUST be first — everything else may raise
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.page = page
        self.job_meta = job_meta or {}
        self.user_profile = user_profile or {}
        self.dry_run = dry_run or os.getenv("DRY_RUN").lower()
        self.steps_completed: int = 0
        self.resume_path: str = self._resolve_resume_path() if page else ""

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
            Absolute path string. Empty string if DEFAULT_RESUME is unset
            and no PDF can be found — the upload step will handle the error.
        """
        resume_dir: str = os.getenv("RESUME_DIR", "/app/resumes")
        default_resume: str = os.getenv("DEFAULT_RESUME", "")

        # Priority 1: job_meta suggested resume, resolved inside RESUME_DIR
        suggested: str = self.job_meta.get("resume_suggested", "")
        if suggested:
            full_path: str = os.path.join(resume_dir, suggested)
            if os.path.exists(full_path):
                return full_path
            self.logger.warning(
                "Suggested resume '%s' not found at %s — falling back to default",
                suggested,
                full_path,
            )

        # Priority 2: DEFAULT_RESUME from env
        if not default_resume:
            self.logger.warning(
                "DEFAULT_RESUME env var not set — "
                "resume upload will fail if no path provided"
            )
        else:
            resolved: str = os.path.join(resume_dir, default_resume)
            if os.path.isfile(resolved):
                return resolved
            self.logger.warning(
                "Default resume not found at %s — "
                "verify RESUME_DIR and DEFAULT_RESUME in java.env",
                resolved,
            )

        # Priority 3 (last resort): first PDF in resume_dir
        try:
            pdfs: list[str] = sorted(
                f for f in os.listdir(resume_dir) if f.endswith(".pdf")
            )
            if pdfs:
                fallback: str = os.path.join(resume_dir, pdfs[0])
                self.logger.warning(
                    "Using first available PDF as resume fallback: %s", fallback
                )
                return fallback
        except OSError:
            pass

        self.logger.error("No resume PDF found in %s", resume_dir)
        return ""

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
    # Dry-run Helpers
    # ------------------------------------------------------------------

    def _gate_dry_run_submit(self) -> bool:
        """Check if the final form submission should be skipped.

        Returns:
            ``True`` if submit must be skipped (dry_run is active).
            ``False`` if real submission should proceed.
        """
        if self.dry_run:
            self.logger.info(
                "[DRY RUN] %s — skipping form submit",
                self.PLATFORM_NAME,
            )
            return True
        return False

    def _dry_run_result(self, steps_completed: int = 0) -> "ApplyResult":
        """Build a successful dry-run ApplyResult.

        Args:
            steps_completed: Number of form steps completed before
                the submit gate was triggered.

        Returns:
            ApplyResult with ``proof_value="DRY_RUN"`` and
            ``success=True``.
        """
        self.steps_completed = steps_completed
        return self._build_result(
            success=True,
            proof_type="dry_run",
            proof_value="DRY_RUN",
            proof_confidence=1.0,
        )

    # ------------------------------------------------------------------
    # Abstract Interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def apply(self) -> "ApplyResult":
        """Execute the full application flow for this platform.

        Returns:
            ApplyResult with outcome, proof, and routing decision.
        """
        ...

BasePlatform = BasePlatformApply
