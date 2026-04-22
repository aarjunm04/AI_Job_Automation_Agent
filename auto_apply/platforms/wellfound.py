"""
auto_apply/platforms/wellfound.py
Production-grade Wellfound (AngelList Talent) job application automator.

Uses confirmed [data-test] attribute selectors from live production scripts
(ezeslucky/welfound.md gist, June 2025 — verified active).

Key flow:
  Login → /jobs page → LearnMoreButton → modal/slideIn → fill textarea
  → handle custom questions → handle relocation → SubmitButton → close modal

Author      : Perplexity (scaffold) + GitHub Copilot (implementation)
Standards   : IDE_STANDARDS.md — Python 3.11, async Playwright, fail-soft,
              DRY_RUN gate, retry+backoff, Google docstrings, mypy hints
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Union, Dict


from playwright.async_api import (
    Page,
    Browser,
    BrowserContext,
    async_playwright,
    TimeoutError as PlaywrightTimeout,
)

from auto_apply.platforms.base_platform import BasePlatformApply, ApplyResult

__all__ = ["WellfoundPlatform"]

logger = logging.getLogger(__name__)




# ─────────────────────────────────────────────────────────────────────────────
# CONFIRMED LIVE SELECTORS
# Source: ezeslucky/welfound.md gist (June 2025), live DOM inspection
# All selectors use Wellfound's stable [data-test] React component attributes.
# ─────────────────────────────────────────────────────────────────────────────
SEL = {
    # ── Auth ──────────────────────────────────────────────────────────────────
    "login_url"           : "https://wellfound.com/login",
    "email_input"         : "input[name='user[email]'], input[type='email'], "
                            "input[placeholder*='email' i]",
    "password_input"      : "input[name='user[password]'], input[type='password']",
    "login_submit"        : "input[type='submit'][value*='Log in' i], "
                            "button[type='submit']:has-text('Log in'), "
                            "button[type='submit']:has-text('Sign in')",
    "login_wall_check"    : ".sign-in, [class*='login'], [href*='/login']",
    "logged_in_check"     : "[data-test='UserMenu'], [class*='userAvatar'], "
                            "img[alt*='avatar' i]",

    # ── Jobs page ─────────────────────────────────────────────────────────────
    "jobs_url"            : "https://wellfound.com/jobs",
    "job_cards_container" : "[class*='JobsList'], [class*='jobList'], "
                            ".styles_component__fhNAA",

    # ── Job modal (slide-in) — CONFIRMED DATA-TEST SELECTORS ─────────────────
    # These are the exact attributes used by Wellfound's React components
    "learn_more_btn"      : "button[data-test='LearnMoreButton']",
    "apply_submit_btn"    : "button[data-test='JobDescriptionSlideIn--SubmitButton']",
    "close_modal_btn"     : "button[data-test='closeButton']",
    "modal_container"     : "[role='dialog'], [class*='SlideIn'], "
                            "[class*='Modal'], [class*='Drawer']",

    # ── Application form fields inside modal ──────────────────────────────────
    "cover_textarea"      : "textarea:not([disabled])",   # CONFIRMED from gist
    "custom_radio_groups" : "[data-test^='RadioGroup-customQuestionAnswers']",  # CONFIRMED
    "relocation_radio"    : "input[name='qualification.location.action']",      # CONFIRMED
    "location_dropdown"   : "#form-input--qualification\.location\.locationId "
                            ".select__control",                                   # CONFIRMED
    "location_menu_opt"   : ".select__menu-list div",

    # ── External ATS redirect detection ──────────────────────────────────────
    "ats_redirect_keywords": ["greenhouse", "lever", "workday", "taleo",
                               "icims", "jobvite", "brassring"],

    # ── Salary / compensation field ───────────────────────────────────────────
    "salary_field"        : "input[placeholder*='salary' i], "
                            "input[placeholder*='compensation' i], "
                            "input[name*='salary' i]",

    # ── Confirmation ──────────────────────────────────────────────────────────
    "confirm_check"       : "div:has-text('Application submitted' i), "
                            "div:has-text('Successfully applied' i), "
                            "div:has-text('applied' i)",
}

SCREENSHOT_DIR = Path("logs")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

async def _screenshot(page: Page, name: str) -> str:
    """Save screenshot to logs/ and return path string."""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = str(SCREENSHOT_DIR / f"{name}_{int(time.time())}.png")
    try:
        await page.screenshot(path=path, full_page=False)
        logger.info("Screenshot saved: %s", path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Screenshot failed: %s", exc)
    return path


async def _safe_fill(page: Page, selector: str, value: str,
                     retries: int = 3, label: str = "") -> bool:
    """Fill a field handling compound selectors, with retry+backoff."""
    for attempt in range(1, retries + 1):
        try:
            for sel in [s.strip() for s in selector.split(",")]:
                el = page.locator(sel).first
                if await el.count() > 0 and await el.is_visible():
                    await el.scroll_into_view_if_needed()
                    await el.fill(value)
                    logger.debug("Filled [%s] attempt %d", label or sel[:40], attempt)
                    return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fill attempt %d/%d [%s]: %s", attempt, retries, label, exc)
            if attempt < retries:
                await asyncio.sleep(2 ** attempt)
    return False


async def _react_fill(page: Page, selector: str, value: str) -> bool:
    """
    Fill a React-controlled textarea/input by dispatching the input event.
    Required for Wellfound — plain fill() doesn't trigger React state update.
    """
    try:
        for sel in [s.strip() for s in selector.split(",")]:
            el = page.locator(sel).first
            if await el.count() > 0 and await el.is_visible():
                await el.scroll_into_view_if_needed()
                await el.click()
                # Clear and fill via JavaScript to trigger React synthetic event
                await page.evaluate(
                    """(args) => {
                        const [selector, val] = args;
                        const el = document.querySelector(selector);
                        if (!el) return false;
                        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                            window.HTMLTextAreaElement.prototype, 'value'
                        )?.set || Object.getOwnPropertyDescriptor(
                            window.HTMLInputElement.prototype, 'value'
                        )?.set;
                        if (nativeInputValueSetter) nativeInputValueSetter.call(el, val);
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                        return true;
                    }""",
                    [sel, value],
                )
                logger.debug("React-filled [%s] (%d chars)", sel[:40], len(value))
                return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("React fill failed [%s]: %s", selector[:40], exc)
    return False


async def _handle_custom_questions(page: Page) -> int:
    """
    Handle Wellfound custom radio question groups inside the apply modal.

    Strategy (confirmed from live script):
      - 3-option groups → select middle option (Intermediate)
      - 2-option groups → select first option
      - Other lengths → select first option as safe default

    Returns count of radio groups handled.
    """
    count = 0
    try:
        groups = await page.locator(SEL["custom_radio_groups"]).all()
        for group in groups:
            try:
                radios = await group.locator("input[type='radio']").all()
                n = len(radios)
                if n == 0:
                    continue
                idx = 1 if n == 3 else 0  # middle for 3, first for 2+
                await radios[idx].click()
                count += 1
                logger.debug("Custom Q: %d-option group → selected idx %d", n, idx)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Custom Q group handler: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Custom Q locator: %s", exc)
    logger.info("[wellfound] Custom question groups handled: %d", count)
    return count


async def _handle_relocation(page: Page, profile: dict[str, Any]) -> bool:
    """
    Handle the Wellfound relocation questionnaire if it appears in the modal.
    Selects first radio option and first location dropdown option.
    Returns True if handled.
    """
    try:
        reloc = page.locator(SEL["relocation_radio"]).first
        if await reloc.count() > 0:
            await reloc.click()
            logger.info("[wellfound] Relocation radio clicked")
            await asyncio.sleep(0.5)

            # Open and select location dropdown
            dropdown = page.locator(SEL["location_dropdown"]).first
            if await dropdown.count() > 0:
                await dropdown.click()
                await asyncio.sleep(0.5)
                first_opt = page.locator(SEL["location_menu_opt"]).first
                if await first_opt.count() > 0:
                    await first_opt.click()
                    logger.info("[wellfound] Location option selected")
                    await asyncio.sleep(1)
            return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("[wellfound] Relocation handler: %s", exc)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PLATFORM CLASS
# ─────────────────────────────────────────────────────────────────────────────

class WellfoundPlatform(BasePlatformApply):
    """
    Wellfound (AngelList Talent) job application automator.

    Uses confirmed [data-test] attribute selectors from live production scripts.

    Handles:
      - Login with WELLFOUND_EMAIL / WELLFOUND_PASSWORD env vars
      - Browsing /jobs page and applying to individual job URLs
      - Cover letter / application text (React-controlled textarea)
      - Custom radio question groups (3-opt→middle, 2-opt→first)
      - Relocation questionnaire + location dropdown
      - Salary expectation field
      - ATS redirect detection (some Wellfound jobs embed external ATS)
      - DRY_RUN gate (fill but never submit)
      - Multi-job batch mode via apply_from_jobs_page()
    """

    platform_name: str = "wellfound"

    async def _inject_wellfound_cookies(self, context: BrowserContext) -> None:
        """Inject wellfound session cookie to bypass login (Issue 7)."""
        cookie_val = os.getenv("WELLFOUND_SESSION_COOKIE")
        if cookie_val:
            await context.add_cookies([
                {"name": "_angellist", "value": cookie_val, "domain": ".wellfound.com", "path": "/"}
            ])
            self.logger.info("[wellfound] Injected WELLFOUND_SESSION_COOKIE")

    # ── Public entry point — single job URL ───────────────────────────────────

    async def apply(
        self,
        job_url: Optional[str] = None,
        profile: Optional[Dict[str, Any]] = None,
    ) -> Union[ApplyResult, Dict[str, Any]]:
        """
        Apply to a single Wellfound job URL.

        Args:
            job_url : Wellfound job URL or /jobs browse URL.
            profile : User profile dict from user_profile.json.

        Returns:
            Result dict: status, platform, fields_filled,
            dry_run_stopped, proof_screenshot_path, error.
        """
        job_url = job_url or self.job_meta.get("url", self.job_meta.get("job_url", ""))
        profile = profile or self.user_profile

        if self.dry_run:
            self.logger.info(
                "[%s] DRY_RUN=True — submit BLOCKED",
                self.__class__.__name__,
            )
            return {
                "dry_run_stopped": True,
                "status": "dry_run_blocked",
                "platform": "wellfound",
                "fields_filled": 0,
                "proof_screenshot_path": "",
                "error": "",
            }
        dry_run = os.getenv("DRY_RUN").lower()
        fields_filled = 0
        proof_path = ""

        try:
            async with async_playwright() as pw:
                browser: Browser = await pw.chromium.launch(
                    headless=False,  # Wellfound blocks headless — always headed
                    slow_mo=int(os.getenv("PLAYWRIGHT_SLOW_MO", "400")),
                    args=["--disable-blink-features=AutomationControlled"],
                )
                context: BrowserContext = await browser.new_context(
                    viewport={"width": 1280, "height": 900},
                    user_agent=(
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/122.0.0.0 Safari/537.36"
                    ),
                    locale="en-US",
                )
                await self._inject_wellfound_cookies(context)
                page = await context.new_page()

                # ── 1. Login ──────────────────────────────────────────────────
                login_ok = await self.login(page, profile)
                if not login_ok:
                    return self._result("login_required", 0, False, "",
                                         "Login failed — check WELLFOUND_EMAIL/PASSWORD")

                proof_path = await _screenshot(page, "wellfound_01_logged_in")

                # ── 2. Navigate to job ────────────────────────────────────────
                logger.info("[wellfound] Navigating to: %s", job_url[:80])
                await page.goto(job_url, timeout=30_000)
                await page.wait_for_load_state("networkidle", timeout=20_000)
                await asyncio.sleep(1)
                proof_path = await _screenshot(page, "wellfound_02_job_page")

                # ── 3. Find and click LearnMore / Apply button ────────────────
                modal_opened = await self._open_apply_modal(page)
                if not modal_opened:
                    proof_path = await _screenshot(page, "wellfound_03_no_modal")
                    return self._result("error", 0, False, proof_path,
                                         "Could not open apply modal — no LearnMoreButton found")

                await asyncio.sleep(1.5)
                proof_path = await _screenshot(page, "wellfound_03_modal_open")

                # ── 4. Check apply button state ───────────────────────────────
                apply_btn = page.locator(SEL["apply_submit_btn"]).first
                if await apply_btn.count() == 0:
                    # Check for external ATS redirect
                    ats = await self._detect_ats_redirect(page)
                    if ats:
                        return self._result("redirected_to_ats", 0, False, proof_path,
                                             f"Redirected to {ats}")
                    return self._result("error", 0, False, proof_path,
                                         "SubmitButton not found in modal")

                is_disabled = await apply_btn.get_attribute("disabled")

                # ── 5. Handle relocation if apply btn disabled ────────────────
                if is_disabled is not None:
                    logger.info("[wellfound] Apply btn disabled — handling relocation Q")
                    await _handle_relocation(page, profile)
                    await asyncio.sleep(1)

                # ── 6. Handle custom radio questions ──────────────────────────
                custom_filled = await _handle_custom_questions(page)
                fields_filled += custom_filled

                # ── 7. Fill cover letter / application textarea ───────────────
                cover_text = profile.get("cover_letter_short", "")
                if cover_text:
                    filled = await _react_fill(page, SEL["cover_textarea"], cover_text)
                    if filled:
                        fields_filled += 1
                        logger.info("[wellfound] Cover letter filled (%d chars)", len(cover_text))
                    await asyncio.sleep(0.5)

                # ── 8. Fill salary if visible ─────────────────────────────────
                salary = str(profile.get("expected_salary_usd", ""))
                if salary:
                    if await _safe_fill(page, SEL["salary_field"], salary, label="salary"):
                        fields_filled += 1

                proof_path = await _screenshot(page, "wellfound_04_filled")
                logger.info("[wellfound] Fields filled: %d", fields_filled)

                # ── DRY_RUN GATE ──────────────────────────────────────────────
                if dry_run:
                    logger.info("[wellfound] DRY_RUN=true — stopping before submit")
                    # Close modal cleanly
                    await self._close_modal(page)
                    return self._result("dry_run", fields_filled, True, proof_path, None)

                # ── 9. Click Submit ───────────────────────────────────────────
                logger.info("[wellfound] Submitting application...")
                submitted = await self._click_submit(page)
                if not submitted:
                    return self._result("error", fields_filled, False, proof_path,
                                         "Submit button click failed")

                await asyncio.sleep(3)
                proof_path = await _screenshot(page, "wellfound_05_submitted")

                # ── 10. Close modal after submit ──────────────────────────────
                await self._close_modal(page)
                logger.info("[wellfound] Application submitted and modal closed")
                return self._result("applied", fields_filled, False, proof_path, None)

        except Exception as exc:  # noqa: BLE001
            logger.error("[wellfound] Fatal error: %s", exc, exc_info=True)
            try:
                proof_path = await _screenshot(page, "wellfound_error")  # type: ignore[name-defined]
            except Exception:
                pass
            return self._result("error", fields_filled, False, proof_path, str(exc))

    # ── Batch mode — apply to multiple jobs from /jobs page ──────────────────

    async def apply_from_jobs_page(
        self,
        profile: dict[str, Any],
        max_jobs: int = 20,
        scroll_limit: int = 10,
    ) -> dict[str, Any]:
        """
        Batch-apply to multiple jobs from wellfound.com/jobs.

        Scrolls the jobs page, clicks LearnMoreButton on each job card,
        fills and submits the modal, moves to next job.

        Args:
            profile    : User profile dict.
            max_jobs   : Maximum number of applications to submit.
            scroll_limit: Maximum number of page scrolls before stopping.

        Returns:
            Summary dict: applied_count, skipped_count, errors.
        """
        dry_run = os.getenv("DRY_RUN").lower()
        applied = 0
        skipped = 0
        errors: list[str] = []
        processed_btns: set[str] = set()

        try:
            async with async_playwright() as pw:
                browser: Browser = await pw.chromium.launch(
                    headless=False,
                    slow_mo=int(os.getenv("PLAYWRIGHT_SLOW_MO", "400")),
                    args=["--disable-blink-features=AutomationControlled"],
                )
                context = await browser.new_context(
                    viewport={"width": 1280, "height": 900},
                    user_agent=(
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/122.0.0.0 Safari/537.36"
                    ),
                )
                page = await context.new_page()

                # Login
                login_ok = await self.login(page, profile)
                if not login_ok:
                    return {"applied": 0, "skipped": 0, "errors": ["Login failed"]}

                await page.goto(SEL["jobs_url"], timeout=30_000)
                await page.wait_for_load_state("networkidle", timeout=20_000)
                await asyncio.sleep(2)
                await _screenshot(page, "wellfound_batch_01_jobs_page")

                scroll_count = 0

                while applied < max_jobs and scroll_count < scroll_limit:
                    # Get unprocessed LearnMore buttons
                    btns = await page.locator(SEL["learn_more_btn"]).all()
                    new_btns = []
                    for btn in btns:
                        try:
                            text = await btn.evaluate("el => el.outerHTML")
                            key = text[:100]
                            if key not in processed_btns:
                                new_btns.append((btn, key))
                        except Exception:
                            continue

                    if not new_btns:
                        # Scroll to load more
                        await page.evaluate("window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})")
                        scroll_count += 1
                        logger.info("[wellfound] Scroll %d/%d — no new jobs found", scroll_count, scroll_limit)
                        await asyncio.sleep(2)
                        continue

                    scroll_count = 0  # reset on finding new jobs

                    for btn, btn_key in new_btns:
                        if applied >= max_jobs:
                            break

                        processed_btns.add(btn_key)

                        try:
                            # Open modal
                            await btn.scroll_into_view_if_needed()
                            await asyncio.sleep(0.3)
                            await btn.click()
                            logger.info("[wellfound] Batch [%d/%d] Opened modal", applied + 1, max_jobs)

                            submit_btn = page.locator(SEL["apply_submit_btn"]).first
                            try:
                                await submit_btn.wait_for(state="visible", timeout=6_000)
                            except PlaywrightTimeout:
                                logger.warning("[wellfound] Modal load timeout — skipping")
                                await self._close_modal(page)
                                skipped += 1
                                continue

                            # Check if disabled (relocation required)
                            is_disabled = await submit_btn.get_attribute("disabled")
                            if is_disabled is not None:
                                await _handle_relocation(page, profile)
                                await asyncio.sleep(1)

                            # Fill custom questions
                            await _handle_custom_questions(page)

                            # Fill cover letter
                            cover = profile.get("cover_letter_short", "")
                            if cover:
                                await _react_fill(page, SEL["cover_textarea"], cover)
                                await asyncio.sleep(0.5)

                            await _screenshot(page, f"wellfound_batch_job_{applied + 1}")

                            # DRY_RUN gate
                            if dry_run:
                                logger.info("[wellfound] DRY_RUN=true — skipping submit for batch job %d", applied + 1)
                                await self._close_modal(page)
                                applied += 1
                                await asyncio.sleep(1)
                                continue

                            # Submit
                            await submit_btn.click()
                            await asyncio.sleep(3)
                            applied += 1
                            logger.info("[wellfound] Applied to job %d/%d", applied, max_jobs)

                        except Exception as exc:  # noqa: BLE001
                            logger.warning("[wellfound] Batch job error: %s", exc)
                            errors.append(str(exc))
                            skipped += 1
                        finally:
                            await self._close_modal(page)
                            await asyncio.sleep(1)

                await _screenshot(page, "wellfound_batch_complete")
                logger.info("[wellfound] Batch done — applied: %d, skipped: %d", applied, skipped)
                return {"applied": applied, "skipped": skipped, "errors": errors,
                        "dry_run": dry_run}

        except Exception as exc:  # noqa: BLE001
            logger.error("[wellfound] Batch fatal error: %s", exc, exc_info=True)
            return {"applied": applied, "skipped": skipped, "errors": [str(exc)]}

    # ── Login helper ──────────────────────────────────────────────────────────

    async def login(self, page: Page, profile: dict[str, Any]) -> bool:
        """
        Log in to Wellfound using env var credentials.

        Checks if already logged in first to avoid redundant login.

        Args:
            page    : Active Playwright page.
            profile : Profile dict (fallback email source).

        Returns:
            True if logged in successfully, False otherwise.
        """
        email    = os.getenv("WELLFOUND_EMAIL", profile.get("email", ""))
        password = os.getenv("WELLFOUND_PASSWORD", "")

        if not email or not password:
            logger.error("[wellfound] WELLFOUND_EMAIL or WELLFOUND_PASSWORD not set")
            return False

        # Check if already logged in
        try:
            await page.goto("https://wellfound.com", timeout=20_000)
            await page.wait_for_load_state("networkidle", timeout=15_000)
            already_in = await page.locator(SEL["logged_in_check"]).count() > 0
            if already_in:
                logger.info("[wellfound] Already logged in")
                return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("[wellfound] Pre-login check: %s", exc)

        logger.info("[wellfound] Logging in as %s", email[:30])
        for attempt in range(1, 4):
            try:
                await page.goto(SEL["login_url"], timeout=20_000)
                await page.wait_for_load_state("networkidle", timeout=15_000)
                await asyncio.sleep(0.5)

                _EMAIL_SELS = [
                    'input[name="user[email]"]',
                    'input[type="email"]',
                    'input[placeholder*="email" i]',
                    'input[id*="email" i]',
                    "#user_email",
                ]
                _email_el = None
                for _sel in _EMAIL_SELS:
                    try:
                        _email_el = page.locator(_sel).first
                        await _email_el.wait_for(state="visible", timeout=3000)
                        break
                    except Exception:
                        _email_el = None
                        continue
                if _email_el is None:
                    raise RuntimeError("Email field not found after trying all selectors")
                await _email_el.fill(os.getenv("WELLFOUND_EMAIL", ""))

                _PASS_SELS = [
                    'input[name="user[password]"]',
                    'input[type="password"]',
                    'input[placeholder*="password" i]',
                    'input[id*="password" i]',
                    "#user_password",
                ]
                _pass_el = None
                for _sel in _PASS_SELS:
                    try:
                        _pass_el = page.locator(_sel).first
                        await _pass_el.wait_for(state="visible", timeout=3000)
                        break
                    except Exception:
                        _pass_el = None
                        continue
                if _pass_el is None:
                    raise RuntimeError("Password field not found after trying all selectors")
                await _pass_el.fill(os.getenv("WELLFOUND_PASSWORD", ""))

                _SUBMIT_SELS = [
                    'button[type="submit"]',
                    'input[type="submit"]',
                    'button:has-text("Sign in")',
                    'button:has-text("Log in")',
                    'button:has-text("Continue")',
                ]
                _submit_el = None
                for _sel in _SUBMIT_SELS:
                    try:
                        _submit_el = page.locator(_sel).first
                        await _submit_el.wait_for(state="visible", timeout=3000)
                        break
                    except Exception:
                        _submit_el = None
                        continue
                if _submit_el is None:
                    raise RuntimeError("Submit button not found after trying all selectors")
                await _submit_el.click()

                await page.wait_for_load_state("networkidle", timeout=15_000)
                await asyncio.sleep(1)

                # Verify login success
                logged_in = await page.locator(SEL["logged_in_check"]).count() > 0
                not_on_login = "/login" not in page.url
                if logged_in or not_on_login:
                    logger.info("[wellfound] Login successful")
                    return True

                logger.warning("[wellfound] Login attempt %d failed — retrying", attempt)
                await asyncio.sleep(2 ** attempt)

            except Exception as exc:  # noqa: BLE001
                logger.warning("[wellfound] Login attempt %d error: %s", attempt, exc)
                await asyncio.sleep(2 ** attempt)

        logger.error("[wellfound] Login failed after 3 attempts")
        return False

    # ── Modal helpers ─────────────────────────────────────────────────────────

    async def _open_apply_modal(self, page: Page) -> bool:
        """Click LearnMoreButton or a direct Apply button to open the modal."""
        selectors = [
            SEL["learn_more_btn"],
            "button:has-text('Apply')",
            "a:has-text('Apply')",
            "button:has-text('Apply Now')",
        ]
        for sel in selectors:
            try:
                el = page.locator(sel).first
                if await el.count() > 0 and await el.is_visible():
                    await el.scroll_into_view_if_needed()
                    await el.click()
                    # Wait for modal to appear
                    try:
                        await page.wait_for_selector(
                            f"{SEL['apply_submit_btn']}, {SEL['modal_container']}",
                            timeout=8_000,
                        )
                    except PlaywrightTimeout:
                        pass
                    logger.info("[wellfound] Modal opened via: %s", sel[:50])
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.debug("[wellfound] Open modal [%s]: %s", sel[:40], exc)
        return False

    async def _close_modal(self, page: Page) -> None:
        """Close the apply modal/slideIn using the closeButton."""
        try:
            close = page.locator(SEL["close_modal_btn"]).first
            if await close.count() > 0 and await close.is_visible():
                await close.click()
                await asyncio.sleep(0.5)
                logger.debug("[wellfound] Modal closed")
        except Exception as exc:  # noqa: BLE001
            logger.debug("[wellfound] Modal close: %s", exc)

    async def _click_submit(self, page: Page) -> bool:
        """Click the submit button with retry."""
        for attempt in range(1, 4):
            try:
                btn = page.locator(SEL["apply_submit_btn"]).first
                if await btn.count() > 0 and await btn.is_visible():
                    is_disabled = await btn.get_attribute("disabled")
                    if is_disabled is not None:
                        logger.warning("[wellfound] Submit btn still disabled on attempt %d", attempt)
                        await asyncio.sleep(2 ** attempt)
                        continue
                    await btn.click()
                    logger.info("[wellfound] Submit clicked")
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.warning("[wellfound] Submit attempt %d: %s", attempt, exc)
                await asyncio.sleep(2 ** attempt)
        return False

    async def _detect_ats_redirect(self, page: Page) -> Optional[str]:
        """Check if the page has redirected to an external ATS."""
        url = page.url.lower()
        for kw in SEL["ats_redirect_keywords"]:
            if kw in url and "wellfound" not in url:
                return kw
        return None

    # ── Compat shim ───────────────────────────────────────────────────────────

    async def _apply_compat(self) -> "ApplyResult":
        """
        Backward-compatibility shim bridging BasePlatformApply.apply() → self.apply().
        Loads profile from config and delegates to the full apply() method.
        """
        try:
            import json
            profile_path = Path("config/user_profile.json")
            profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
            job_url = getattr(self, "job_url", "") or SEL["jobs_url"]
            result = await self.apply(job_url, profile)
            return ApplyResult(
                status=result.get("status", "error"),
                platform=result.get("platform", self.platform_name),
                error=result.get("error"),
                proof_screenshot_path=result.get("proof_screenshot_path"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[wellfound] _apply_compat error: %s", exc)
            return ApplyResult(status="error", platform=self.platform_name, error=str(exc))

    # ── Result builder ────────────────────────────────────────────────────────

    @staticmethod
    def _result(
        status: str,
        fields_filled: int,
        dry_run_stopped: bool,
        proof_screenshot_path: str,
        error: Optional[str],
    ) -> dict[str, Any]:
        """Build the standard result dict returned by apply()."""
        return {
            "status"               : status,
            "platform"             : "wellfound",
            "fields_filled"        : fields_filled,
            "dry_run_stopped"      : dry_run_stopped,
            "proof_screenshot_path": proof_screenshot_path,
            "error"                : error,
        }

WellfoundApply = WellfoundPlatform
