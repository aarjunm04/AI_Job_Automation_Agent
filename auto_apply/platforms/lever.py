"""
auto_apply/platforms/lever.py
Production-grade Lever ATS job application automator.

Lever uses standard HTML name/type attributes — no dynamic IDs.
Form renders inline on the job page (no modal, no redirect for most jobs).

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
from typing import Any, Dict, Optional, Union

from playwright.async_api import (
    Page,
    Browser,
    async_playwright,
    TimeoutError as PlaywrightTimeout,
)

from auto_apply.platforms.base_platform import BasePlatformApply, ApplyResult

__all__ = ["LeverPlatform"]

logger = logging.getLogger(__name__)

class LeverPlatform(BasePlatformApply):
    def __init__(
        self,
        job_meta: dict | None = None,
        user_profile: dict | None = None,
        **kwargs,
    ) -> None:
        """
        Flexible __init__ that satisfies BasePlatformApply's signature
        while allowing zero-arg instantiation for testing.

        Args:
            job_meta     : Job metadata dict (url, title, company, etc.)
            user_profile : User profile dict (name, email, resume_path, etc.)
        """
        self.job_meta    = job_meta    or {}
        self.user_profile = user_profile or {}
        # Expose job_url directly for compat shim
        self.job_url     = self.job_meta.get("url", self.job_meta.get("job_url", ""))
        # Call super only if it accepts these args to avoid MRO crash
        try:
            super().__init__(job_meta=self.job_meta, user_profile=self.user_profile, **kwargs)
        except TypeError:
            pass  # BasePlatformApply may not accept kwargs — fail soft

# ─────────────────────────────────────────────────────────────────────────────
# STABLE SELECTORS
# Source: live DOM inspection jobs.lever.co 2024-2026, multiple repos
# Lever forms use consistent name/placeholder attributes across all companies.
# ─────────────────────────────────────────────────────────────────────────────
SEL = {
    # ── Apply entry ───────────────────────────────────────────────────────────
    "apply_btn"        : "a:has-text('Apply for this job'), "
                         "button:has-text('Apply for this job'), "
                         "a:has-text('Apply'), button:has-text('Apply')",

    # ── Core personal fields (standard across ALL Lever forms) ────────────────
    "full_name"        : "input[name='name']",
    "email"            : "input[name='email']",
    "phone"            : "input[name='phone']",
    "org"              : "input[name='org']",           # current company
    "urls_linkedin"    : "input[name='urls[LinkedIn]'], "
                         "input[placeholder*='LinkedIn' i], "
                         "input[placeholder*='linkedin' i]",
    "urls_github"      : "input[name='urls[GitHub]'], "
                         "input[placeholder*='GitHub' i], "
                         "input[placeholder*='github' i]",
    "urls_portfolio"   : "input[name='urls[Portfolio]'], "
                         "input[placeholder*='Portfolio' i], "
                         "input[placeholder*='website' i], "
                         "input[placeholder*='personal site' i]",
    "urls_twitter"     : "input[name='urls[Twitter]'], input[placeholder*='Twitter' i]",
    "urls_other"       : "input[name='urls[Other]']",

    # ── Resume upload ─────────────────────────────────────────────────────────
    "resume_upload"    : "input[type='file'][name='resume'], "
                         "input[type='file']",

    # ── Cover letter / additional info ────────────────────────────────────────
    "cover_letter"     : "textarea[name='comments'], "
                         "textarea[placeholder*='cover letter' i], "
                         "textarea[placeholder*='additional' i], "
                         "textarea[placeholder*='anything else' i]",

    # ── Custom questions (Lever allows companies to add these) ────────────────
    "custom_text"      : ".application-question input[type='text']:visible, "
                         ".custom-question input[type='text']:visible",
    "custom_textarea"  : ".application-question textarea:visible, "
                         ".custom-question textarea:visible",
    "custom_select"    : ".application-question select:visible, "
                         ".custom-question select:visible",
    "custom_radio"     : ".application-question input[type='radio']:visible, "
                         ".custom-question input[type='radio']:visible",
    "custom_checkbox"  : ".application-question input[type='checkbox']:visible",

    # ── EEOC / Demographic (some Lever forms include these) ───────────────────
    "eeoc_fields"      : "select[name*='eeo' i], select[name*='eeoc' i], "
                         "select[name*='demographic' i], select[name*='gender' i], "
                         "select[name*='race' i], select[name*='veteran' i], "
                         "select[name*='disability' i]",
    "decline_options"  : [
        "Decline to identify",
        "Decline to Self Identify",
        "I don't wish to answer",
        "Prefer not to say",
        "Choose not to disclose",
        "No Response",
        "I do not wish to disclose",
        "I don't wish to identify",
    ],

    # ── Submit ────────────────────────────────────────────────────────────────
    "submit_btn"       : "input[type='submit'][value*='Submit' i], "
                         "button[type='submit'], "
                         "button:has-text('Submit Application'), "
                         "button:has-text('Submit')",

    # ── Redirect detection ────────────────────────────────────────────────────
    "ats_redirect_keywords": ["workday", "greenhouse", "taleo", "icims",
                               "successfactors", "brassring", "jobvite"],

    # ── Confirmation ─────────────────────────────────────────────────────────
    "confirm_text"     : "h1:has-text('Thank you'), h2:has-text('Thank you'), "
                         ".thanks, .confirmation, "
                         "div:has-text('application has been submitted' i)",
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
    """Fill a visible input field with retry+backoff. Returns True on success."""
    for attempt in range(1, retries + 1):
        try:
            # Handle compound selectors (comma-separated)
            for sel in [s.strip() for s in selector.split(",")]:
                el = page.locator(sel).first
                if await el.count() > 0 and await el.is_visible():
                    await el.scroll_into_view_if_needed()
                    await el.fill(value)
                    logger.debug("Filled [%s] attempt %d/%d", label or sel[:40], attempt, retries)
                    return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fill attempt %d/%d [%s]: %s", attempt, retries, label, exc)
            if attempt < retries:
                await asyncio.sleep(2 ** attempt)
    return False


async def _fill_eeoc_selects(page: Page) -> int:
    """Find EEOC/demographic selects and choose the first 'Decline' option."""
    count = 0
    decline_phrases = SEL["decline_options"]
    try:
        selects = await page.locator(SEL["eeoc_fields"]).all()
        for sel_el in selects:
            opts = await sel_el.locator("option").all()
            for opt in opts:
                text = (await opt.inner_text()).strip()
                if any(p.lower() in text.lower() for p in decline_phrases):
                    try:
                        await sel_el.select_option(label=text)
                        count += 1
                        logger.debug("EEOC: selected decline '%s'", text)
                        break
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("EEOC select fail: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.debug("EEOC handler error: %s", exc)
    return count


async def _handle_custom_questions(page: Page, profile: dict[str, Any]) -> int:
    """
    Handle Lever's company-specific custom application questions generically.
    Matches question text to profile values via keywords.
    Returns count filled.
    """
    filled = 0
    keyword_map: dict[str, str] = {
        "sponsor"       : "No",
        "visa"          : profile.get("visa_sponsorship_needed", "No"),
        "authoriz"      : "Yes",
        "relocat"       : profile.get("willing_to_relocate", "No"),
        "remote"        : "Yes",
        "salary"        : str(profile.get("expected_salary_usd", "")),
        "compensation"  : str(profile.get("expected_salary_usd", "")),
        "experience"    : str(profile.get("years_of_experience", "")),
        "year"          : str(profile.get("years_of_experience", "")),
        "linkedin"      : profile.get("linkedin_url", ""),
        "github"        : profile.get("github_url", ""),
        "portfolio"     : profile.get("portfolio_url", ""),
        "cover"         : profile.get("cover_letter_short", ""),
        "why"           : profile.get("why_join_statement",
                                       profile.get("cover_letter_short", "")),
        "tell us"       : profile.get("cover_letter_short", ""),
        "describe"      : profile.get("cover_letter_short", ""),
        "hear about"    : "LinkedIn",
        "how did"       : "LinkedIn",
        "notice"        : profile.get("notice_period", "Immediate"),
    }

    # Custom text inputs
    try:
        inputs = await page.locator(SEL["custom_text"]).all()
        for inp in inputs:
            try:
                placeholder = (await inp.get_attribute("placeholder") or "").lower()
                aria_label  = (await inp.get_attribute("aria-label") or "").lower()
                # Walk up DOM to get question label text
                q_text = await inp.evaluate(
                    "el => el.closest('.application-question')"
                    "?.querySelector('label,p,h4')?.innerText || ''")
                hint = (placeholder + " " + aria_label + " " + q_text).lower()
                for kw, val in keyword_map.items():
                    if kw in hint and val:
                        current = await inp.input_value()
                        if not current:
                            await inp.fill(val)
                            filled += 1
                            logger.debug("Custom Q text: filled [%s]", hint[:50])
                        break
            except Exception as exc:  # noqa: BLE001
                logger.debug("Custom Q text input skip: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Custom Q text locator: %s", exc)

    # Custom textareas
    try:
        textareas = await page.locator(SEL["custom_textarea"]).all()
        for ta in textareas:
            try:
                current = await ta.input_value()
                if current:
                    continue
                placeholder = (await ta.get_attribute("placeholder") or "").lower()
                q_text = await ta.evaluate(
                    "el => el.closest('.application-question')"
                    "?.querySelector('label,p,h4')?.innerText || ''")
                hint = (placeholder + " " + q_text).lower()
                text = profile.get("cover_letter_short", "")
                for kw, val in keyword_map.items():
                    if kw in hint and val:
                        text = val
                        break
                if text:
                    await ta.fill(text)
                    filled += 1
                    logger.debug("Custom Q textarea: filled [%s]", hint[:50])
            except Exception as exc:  # noqa: BLE001
                logger.debug("Custom Q textarea skip: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Custom Q textarea locator: %s", exc)

    # Custom selects — pick most relevant option
    try:
        selects = await page.locator(SEL["custom_select"]).all()
        for sel_el in selects:
            try:
                q_text = await sel_el.evaluate(
                    "el => el.closest('.application-question')"
                    "?.querySelector('label,p,h4')?.innerText || ''").lower()
                opts = await sel_el.locator("option").all()
                if len(opts) > 1:
                    # Try to find a "No" option for sponsor/visa questions
                    if "sponsor" in q_text or "visa" in q_text:
                        for opt in opts:
                            if "no" in (await opt.inner_text()).lower():
                                await sel_el.select_option(
                                    label=(await opt.inner_text()).strip())
                                filled += 1
                                break
                    else:
                        # Default: pick second option (avoid blank first option)
                        await sel_el.select_option(index=1)
                        filled += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("Custom Q select skip: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Custom Q select locator: %s", exc)

    # Custom radio buttons — safe defaults
    try:
        radios = await page.locator(SEL["custom_radio"]).all()
        for radio in radios:
            try:
                val = (await radio.get_attribute("value") or "").lower()
                q_text = await radio.evaluate(
                    "el => el.closest('.application-question')"
                    "?.innerText || ''").lower()
                should_check = False
                if ("sponsor" in q_text or "visa" in q_text) and val in ("no", "false"):
                    should_check = True
                elif ("authoriz" in q_text or "eligible" in q_text or
                      "remote" in q_text) and val in ("yes", "true"):
                    should_check = True
                if should_check:
                    await radio.check()
                    filled += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("Custom Q radio skip: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Custom Q radio locator: %s", exc)

    logger.info("[lever] Custom questions filled: %d", filled)
    return filled


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PLATFORM CLASS
# ─────────────────────────────────────────────────────────────────────────────

class LeverPlatform(BasePlatformApply):
    """
    Lever ATS job application automator.

    Handles:
      - Inline apply form (no login required for most Lever jobs)
      - Standard fields: name, email, phone, LinkedIn, GitHub, portfolio
      - Resume file upload
      - Cover letter textarea
      - Company-specific custom questions (generic keyword matching)
      - EEOC/demographic selects → always "Decline to identify"
      - ATS redirect detection (Workday/Greenhouse embedded in Lever)
      - DRY_RUN gate (fill but never submit)

    No login required — Lever is a public apply system.
    """

    platform_name: str = "lever"

    # ── Public entry point ────────────────────────────────────────────────────

    async def apply(
        self,
        job_url: Optional[str] = None,
        profile: Optional[Dict[str, Any]] = None,
    ) -> Union[ApplyResult, Dict[str, Any]]:
        """
        Run the full Lever apply flow for a given job URL.

        Args:
            job_url : Full jobs.lever.co URL for the specific job.
            profile : User profile dict from user_profile.json / config loader.

        Returns:
            Result dict: status, platform, fields_filled,
            dry_run_stopped, proof_screenshot_path, error.
        """
        job_url = job_url or self.job_meta.get("url", self.job_meta.get("job_url", ""))
        profile = profile or self.user_profile
        dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
        fields_filled = 0
        proof_path = ""

        try:
            async with async_playwright() as pw:
                browser: Browser = await pw.chromium.launch(
                    headless=os.getenv("PLAYWRIGHT_HEADLESS", "false").lower() == "true",
                    slow_mo=int(os.getenv("PLAYWRIGHT_SLOW_MO", "200")),
                    args=["--disable-blink-features=AutomationControlled"],
                )
                context = await browser.new_context(
                    viewport={"width": 1280, "height": 900},
                    user_agent=(
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/122.0.0.0 Safari/537.36"
                    ),
                    locale="en-US",
                )
                page = await context.new_page()

                logger.info("[lever] Navigating to: %s", job_url[:80])
                await page.goto(job_url, timeout=30_000)
                await page.wait_for_load_state("networkidle", timeout=20_000)
                proof_path = await _screenshot(page, "lever_01_landing")

                # ── Detect ATS redirect ───────────────────────────────────────
                current_url = page.url.lower()
                for keyword in SEL["ats_redirect_keywords"]:
                    if keyword in current_url and "lever" not in current_url:
                        logger.warning("[lever] Redirected to external ATS: %s", keyword)
                        return self._result("redirected_to_ats", 0, False, proof_path,
                                             f"Redirected to {keyword}")

                # ── Click Apply button ────────────────────────────────────────
                clicked = await self._click_apply(page)
                if not clicked:
                    proof_path = await _screenshot(page, "lever_02_no_apply_btn")
                    return self._result("error", 0, False, proof_path,
                                         "Apply button not found on page")

                # Wait for form to appear
                try:
                    await page.wait_for_selector(
                        "input[name='name'], input[name='email'], "
                        "form.application-form, .lever-job-app",
                        timeout=10_000,
                    )
                except PlaywrightTimeout:
                    logger.warning("[lever] Form did not appear after clicking Apply")

                await asyncio.sleep(0.5)
                proof_path = await _screenshot(page, "lever_02_form_open")

                # ── Check for redirect after apply click ──────────────────────
                post_click_url = page.url.lower()
                for keyword in SEL["ats_redirect_keywords"]:
                    if keyword in post_click_url and "lever" not in post_click_url:
                        logger.warning("[lever] Post-click redirect to: %s", keyword)
                        return self._result("redirected_to_ats", 0, False, proof_path,
                                             f"Redirected to {keyword}")

                # ── Fill core fields ──────────────────────────────────────────
                fields_filled += await self._fill_core_fields(page, profile)
                proof_path = await _screenshot(page, "lever_03_core_filled")

                # ── Upload resume ─────────────────────────────────────────────
                fields_filled += await self._upload_resume(page, profile)

                # ── Fill cover letter ─────────────────────────────────────────
                fields_filled += await self._fill_cover_letter(page, profile)

                # ── Handle custom questions ───────────────────────────────────
                fields_filled += await _handle_custom_questions(page, profile)

                # ── Handle EEOC dropdowns ─────────────────────────────────────
                fields_filled += await _fill_eeoc_selects(page)

                proof_path = await _screenshot(page, "lever_04_fully_filled")
                logger.info("[lever] Total fields filled: %d", fields_filled)

                # ── DRY_RUN GATE ──────────────────────────────────────────────
                if dry_run:
                    logger.info("[lever] DRY_RUN=true — stopping before submit")
                    return self._result("dry_run", fields_filled, True, proof_path, None)

                # ── Submit ────────────────────────────────────────────────────
                logger.info("[lever] Submitting application...")
                submitted = await self._click_submit(page)
                if not submitted:
                    return self._result("error", fields_filled, False, proof_path,
                                         "Submit button not found")

                await asyncio.sleep(3)
                proof_path = await _screenshot(page, "lever_05_submitted")

                # Verify confirmation
                confirmed = await self._check_confirmation(page)
                status = "applied" if confirmed else "submitted_unconfirmed"
                logger.info("[lever] Application result: %s", status)
                return self._result(status, fields_filled, False, proof_path, None)

        except Exception as exc:  # noqa: BLE001
            logger.error("[lever] Fatal error: %s", exc, exc_info=True)
            try:
                proof_path = await _screenshot(page, "lever_error")  # type: ignore[name-defined]
            except Exception:
                pass
            return self._result("error", fields_filled, False, proof_path, str(exc))

    # ── Click Apply ───────────────────────────────────────────────────────────

    async def _click_apply(self, page: Page) -> bool:
        """Find and click the Apply button on the Lever job page."""
        # Lever's main apply button is typically an anchor tag
        selectors = [
            "a[href*='apply']:has-text('Apply')",
            "a.apply-btn",
            "a:has-text('Apply for this job')",
            "button:has-text('Apply for this job')",
            "a:has-text('Apply now')",
            "button:has-text('Apply now')",
            "a:has-text('Apply')",
        ]
        for sel in selectors:
            try:
                el = page.locator(sel).first
                if await el.count() > 0 and await el.is_visible():
                    await el.scroll_into_view_if_needed()
                    await el.click()
                    await asyncio.sleep(1)
                    logger.info("[lever] Apply button clicked: %s", sel[:50])
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.debug("[lever] Apply btn [%s]: %s", sel[:40], exc)
        return False

    # ── Core fields ───────────────────────────────────────────────────────────

    async def _fill_core_fields(self, page: Page,
                                  profile: dict[str, Any]) -> int:
        """Fill the standard Lever fields present on every Lever form."""
        filled = 0

        # Full name — Lever uses a single "name" field (not first+last)
        full_name = (
            profile.get("full_name")
            or f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip()
        )

        field_map = [
            (SEL["full_name"],      full_name,                            "full_name"),
            (SEL["email"],          profile.get("email", ""),             "email"),
            (SEL["phone"],          profile.get("phone", ""),             "phone"),
            (SEL["org"],            profile.get("current_company", ""),   "org"),
            (SEL["urls_linkedin"],  profile.get("linkedin_url", ""),      "linkedin"),
            (SEL["urls_github"],    profile.get("github_url", ""),        "github"),
            (SEL["urls_portfolio"], profile.get("portfolio_url", ""),     "portfolio"),
        ]

        for selector, value, label in field_map:
            if not value:
                continue
            if await _safe_fill(page, selector, value, label=label):
                filled += 1

        logger.info("[lever] Core fields filled: %d", filled)
        return filled

    # ── Resume upload ─────────────────────────────────────────────────────────

    async def _upload_resume(self, page: Page,
                              profile: dict[str, Any]) -> int:
        """Upload resume file if a file input exists on the form."""
        resume_path = profile.get("resume_path", os.getenv("RESUME_PATH", ""))
        if not resume_path or not Path(resume_path).exists():
            logger.warning("[lever] Resume path missing or not found: %s", resume_path)
            return 0

        for sel in [s.strip() for s in SEL["resume_upload"].split(",")]:
            for attempt in range(1, 4):
                try:
                    el = page.locator(sel).first
                    if await el.count() > 0:
                        await el.set_input_files(resume_path)
                        await asyncio.sleep(1)  # allow upload to register
                        logger.info("[lever] Resume uploaded: %s", resume_path)
                        return 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[lever] Resume upload attempt %d: %s", attempt, exc)
                    if attempt < 3:
                        await asyncio.sleep(2 ** attempt)

        logger.warning("[lever] Resume upload: no file input found")
        return 0

    # ── Cover letter ──────────────────────────────────────────────────────────

    async def _fill_cover_letter(self, page: Page,
                                   profile: dict[str, Any]) -> int:
        """Fill the cover letter / comments textarea if present."""
        cover_text = profile.get("cover_letter_short", "")
        if not cover_text:
            return 0

        for sel in [s.strip() for s in SEL["cover_letter"].split(",")]:
            try:
                el = page.locator(sel).first
                if await el.count() > 0 and await el.is_visible():
                    current = await el.input_value()
                    if not current:
                        await el.fill(cover_text)
                        logger.info("[lever] Cover letter filled (%d chars)", len(cover_text))
                        return 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("[lever] Cover letter [%s]: %s", sel[:40], exc)
        return 0

    # ── Submit ────────────────────────────────────────────────────────────────

    async def _click_submit(self, page: Page) -> bool:
        """Click the submit / Send Application button."""
        if self.dry_run:
            self.logger.info(
                "[%s] DRY_RUN=True — submit BLOCKED",
                self.__class__.__name__,
            )
            return {
                "dry_run_stopped": True,
                "status": "dry_run_blocked",
                "platform": self.__class__.__name__,
                "fields_filled": 0,
                "proof_screenshot_path": "",
                "error": "",
            }
        for sel in [s.strip() for s in SEL["submit_btn"].split(",")]:
            for attempt in range(1, 4):
                try:
                    el = page.locator(sel).first
                    if await el.count() > 0 and await el.is_visible():
                        await el.scroll_into_view_if_needed()
                        await el.click()
                        logger.info("[lever] Submit clicked: %s", sel[:50])
                        return True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[lever] Submit attempt %d [%s]: %s", attempt, sel[:40], exc)
                    if attempt < 3:
                        await asyncio.sleep(2 ** attempt)
        return False

    # ── Confirmation check ────────────────────────────────────────────────────

    async def _check_confirmation(self, page: Page) -> bool:
        """Check if a confirmation/thank-you message appeared after submit."""
        try:
            await page.wait_for_selector(SEL["confirm_text"], timeout=8_000)
            logger.info("[lever] Confirmation message detected")
            return True
        except PlaywrightTimeout:
            logger.warning("[lever] No confirmation message found — may have submitted anyway")
            return False

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
            job_url = getattr(self, "job_url", "") or getattr(self, "url", "")
            result = await self.apply(job_url, profile)
            return ApplyResult(
                status=result.get("status", "error"),
                platform=result.get("platform", self.platform_name),
                error=result.get("error"),
                proof_screenshot_path=result.get("proof_screenshot_path"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[lever] _apply_compat error: %s", exc)
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
            "platform"             : "lever",
            "fields_filled"        : fields_filled,
            "dry_run_stopped"      : dry_run_stopped,
            "proof_screenshot_path": proof_screenshot_path,
            "error"                : error,
        }

LeverApply = LeverPlatform
