"""
auto_apply/platforms/workday.py
Production-grade Workday multi-tenant job apply automation.
[data-automation-id] selectors are stable across ALL Workday tenants.

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
from typing import Any, Optional, Dict, Union

from playwright.async_api import (
    Page,
    Browser,
    async_playwright,
    TimeoutError as PlaywrightTimeout,
)

from auto_apply.platforms.base_platform import BasePlatformApply, ApplyResult

__all__ = ["WorkdayPlatform"]

logger = logging.getLogger(__name__)




# ─────────────────────────────────────────────────────────────────────────────
# STABLE SELECTORS  (data-automation-id — universal across all Workday tenants)
# Source: amgenene/workday_auto, ubangura/Workday-Application-Automator,
#         raghuboosetty/workday, live DOM inspection 2024-2026
# ─────────────────────────────────────────────────────────────────────────────
SEL = {
    # ── Auth ──────────────────────────────────────────────────────────────────
    "sign_in_btn"        : '[data-automation-id="signIn"]',
    "email_input"        : '[data-automation-id="email"]',
    "password_input"     : '[data-automation-id="password"]',
    "submit_sign_in"     : '[data-automation-id="click_filter"]',
    "create_acct_btn"    : '[data-automation-id="createAccount"]',
    "new_acct_email"     : '[data-automation-id="createAccountEmail"]',
    "new_acct_verify"    : '[data-automation-id="createAccountVerifyEmail"]',
    "new_acct_password"  : '[data-automation-id="createAccountPassword"]',
    "new_acct_submit"    : '[data-automation-id="createAccountSubmitButton"]',

    # ── Apply entry ───────────────────────────────────────────────────────────
    "apply_btn"          : '[data-automation-id="applyButton"]',
    "apply_manually_btn" : '[data-automation-id="applyManuallyButton"]',
    "apply_with_resume"  : "button:has-text('Apply')",

    # ── Page 1: My Information ────────────────────────────────────────────────
    "first_name"         : '[data-automation-id="legalNameSection_firstName"]',
    "last_name"          : '[data-automation-id="legalNameSection_lastName"]',
    "phone_number"       : '[data-automation-id="phone-number"]',
    "phone_device_type"  : '[data-automation-id="phone-device-type"]',
    "country_dropdown"   : '[data-automation-id="addressSection_country"]',
    "address_line1"      : '[data-automation-id="addressSection_addressLine1"]',
    "city"               : '[data-automation-id="addressSection_city"]',
    "state"              : '[data-automation-id="addressSection_stateProvince"]',
    "postal_code"        : '[data-automation-id="addressSection_postalCode"]',
    "how_hear_dropdown"  : '[data-automation-id="howDidYouHearAboutUs"]',
    "prev_worked_no"     : 'input[type="radio"][value="No"], label:has-text("No") input[type="radio"]',
    "prev_worked_yes"    : 'input[type="radio"][value="Yes"]',

    # ── Page 2: My Experience ─────────────────────────────────────────────────
    "resume_upload"      : '[data-automation-id="file-upload-input-ref"]',
    "resume_btn"         : 'button[data-automation-id="fileSectionAddButton"]',
    "school_name"        : '[data-automation-id="school"]',
    "degree_dropdown"    : '[data-automation-id="degree"]',
    "field_of_study"     : '[data-automation-id="fieldOfStudy"]',
    "gpa"                : '[data-automation-id="gpa"]',
    "edu_from_date"      : '[data-automation-id="startDate"]',
    "edu_to_date"        : '[data-automation-id="endDate"]',
    "job_title"          : '[data-automation-id="jobTitle"]',
    "company_name"       : '[data-automation-id="company"]',
    "work_from_date"     : '[data-automation-id="workExperienceFromDate"]',
    "work_to_date"       : '[data-automation-id="workExperienceToDate"]',
    "description"        : '[data-automation-id="description"]',
    "linkedin_url"       : '[data-automation-id="linkedinUrl"]',
    "website_url"        : '[data-automation-id="websiteUrl"]',

    # ── Page 3-5: Questions / EEO / Self Identify ─────────────────────────────
    "text_area_generic"  : "textarea",
    "text_input_generic" : "input[type='text']:visible",
    "radio_generic"      : "input[type='radio']",
    "select_generic"     : "select:visible",
    "dropdown_generic"   : '[data-automation-id="selectWidget"]',

    # EEO / Voluntary Disclosure — always "Decline"
    "decline_options"    : [
        "Decline to Identify",
        "Decline to Self Identify",
        "I don't wish to answer",
        "Prefer not to say",
        "Choose not to disclose",
        "No Response",
        "I do not wish to disclose",
    ],

    # ── Navigation ────────────────────────────────────────────────────────────
    "next_btn"           : '[data-automation-id="bottom-navigation-next-button"]',
    "save_continue_btn"  : '[data-automation-id="bottom-navigation-next-button"]',
    "submit_btn"         : '[data-automation-id="bottom-navigation-next-button"]',  # same on final page
    "review_submit_btn"  : 'button:has-text("Submit"), [data-automation-id="bottom-navigation-next-button"]',

    # ── Page step indicator ───────────────────────────────────────────────────
    "page_title"         : "h2, h3, [data-automation-id='pageHeaderTitle']",
    "stepper_active"     : ".gwt-InlineHTML, [class*='progressStep']:not([class*='inactive'])",
}

SCREENSHOT_DIR = Path("logs")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

async def _screenshot(page: Page, name: str) -> str:
    """Save screenshot to logs/ and return the path string."""
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
    """Fill a field with retry+backoff. Returns True on success."""
    for attempt in range(1, retries + 1):
        try:
            el = page.locator(selector).first
            await el.wait_for(state="visible", timeout=5_000)
            await el.scroll_into_view_if_needed()
            await el.fill(value)
            logger.debug("Filled [%s] attempt %d/%d", label or selector[:40], attempt, retries)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fill attempt %d/%d failed [%s]: %s", attempt, retries, label, exc)
            if attempt < retries:
                await asyncio.sleep(2 ** attempt)
    return False


async def _safe_select_text(page: Page, selector: str, text: str,
                             retries: int = 3, label: str = "") -> bool:
    """Select dropdown option by visible text with retry."""
    for attempt in range(1, retries + 1):
        try:
            el = page.locator(selector).first
            await el.wait_for(state="visible", timeout=5_000)
            await el.select_option(label=text)
            logger.debug("Selected [%s]='%s' attempt %d", label or selector[:40], text, attempt)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Select attempt %d/%d failed [%s]: %s", attempt, retries, label, exc)
            if attempt < retries:
                await asyncio.sleep(2 ** attempt)
    return False


async def _click_next(page: Page, label: str = "Next") -> bool:
    """Click the navigation next/save-and-continue button."""
    for attempt in range(1, 4):
        try:
            btn = page.locator(SEL["next_btn"]).first
            await btn.wait_for(state="visible", timeout=8_000)
            await btn.scroll_into_view_if_needed()
            await btn.click()
            await page.wait_for_load_state("networkidle", timeout=15_000)
            logger.info("Clicked '%s' — page advanced", label)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Click '%s' attempt %d/3: %s", label, attempt, exc)
            await asyncio.sleep(2 ** attempt)
    return False


async def _select_decline_for_eeo(page: Page) -> int:
    """
    Find all radio/select groups on EEO/demographic pages and select
    the "Decline to Identify" / "Prefer not to say" option in each.
    Returns count of fields handled.
    """
    count = 0
    decline_phrases = SEL["decline_options"]

    # Approach 1 — radio buttons whose labels contain decline phrases
    labels = await page.locator("label").all()
    for lbl in labels:
        text = (await lbl.inner_text()).strip()
        if any(phrase.lower() in text.lower() for phrase in decline_phrases):
            try:
                radio_id = await lbl.get_attribute("for")
                if radio_id:
                    await page.locator(f"#{radio_id}").click()
                    count += 1
                    logger.debug("EEO: selected decline via label '%s'", text)
            except Exception as exc:  # noqa: BLE001
                logger.debug("EEO radio click failed: %s", exc)

    # Approach 2 — select dropdowns: choose first matching "decline" option
    selects = await page.locator("select:visible").all()
    for sel_el in selects:
        opts = await sel_el.locator("option").all()
        for opt in opts:
            opt_text = (await opt.inner_text()).strip()
            if any(phrase.lower() in opt_text.lower() for phrase in decline_phrases):
                try:
                    await sel_el.select_option(label=opt_text)
                    count += 1
                    logger.debug("EEO: selected decline option '%s'", opt_text)
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.debug("EEO select failed: %s", exc)

    return count


async def _handle_application_questions(page: Page, profile: dict[str, Any]) -> int:
    """
    Generic handler for Workday Application Questions page (Page 3).
    Fills text inputs and textareas with profile-matched answers.
    Returns number of fields filled.
    """
    filled = 0
    keyword_map: dict[str, str] = {
        "sponsor"         : "No",
        "visa"            : profile.get("visa_sponsorship_needed", "No"),
        "authorized"      : "Yes",
        "relocat"         : profile.get("willing_to_relocate", "No"),
        "remote"          : "Yes",
        "salary"          : str(profile.get("expected_salary_usd", "")),
        "linkedin"        : profile.get("linkedin_url", ""),
        "github"          : profile.get("github_url", ""),
        "portfolio"       : profile.get("portfolio_url", ""),
        "experience"      : profile.get("years_of_experience", ""),
        "year"            : profile.get("years_of_experience", ""),
        "cover"           : profile.get("cover_letter_short", ""),
        "why"             : profile.get("why_join_statement", profile.get("cover_letter_short", "")),
        "tell us about"   : profile.get("cover_letter_short", ""),
        "describe"        : profile.get("cover_letter_short", ""),
    }

    # Text inputs
    text_inputs = await page.locator("input[type='text']:visible").all()
    for inp in text_inputs:
        try:
            placeholder = (await inp.get_attribute("placeholder") or "").lower()
            aria_label  = (await inp.get_attribute("aria-label") or "").lower()
            hint = placeholder + " " + aria_label
            for kw, val in keyword_map.items():
                if kw in hint and val:
                    await inp.fill(val)
                    filled += 1
                    logger.debug("AppQ: filled text input [%s]", hint[:50])
                    break
        except Exception as exc:  # noqa: BLE001
            logger.debug("AppQ text input skip: %s", exc)

    # Textareas
    textareas = await page.locator("textarea:visible").all()
    for ta in textareas:
        try:
            val = await ta.input_value()
            if val:
                continue  # already filled
            placeholder = (await ta.get_attribute("placeholder") or "").lower()
            aria_label  = (await ta.get_attribute("aria-label") or "").lower()
            hint = placeholder + " " + aria_label
            text = profile.get("cover_letter_short", "")
            for kw, mapped_val in keyword_map.items():
                if kw in hint and mapped_val:
                    text = mapped_val
                    break
            if text:
                await ta.fill(text)
                filled += 1
                logger.debug("AppQ: filled textarea [%s]", hint[:50])
        except Exception as exc:  # noqa: BLE001
            logger.debug("AppQ textarea skip: %s", exc)

    # Yes/No radio buttons — safe defaults
    radio_groups = await page.locator("[role='radio']:visible, input[type='radio']:visible").all()
    for radio in radio_groups:
        try:
            val = (await radio.get_attribute("value") or "").lower()
            parent_text = await radio.evaluate(
                "el => el.closest('div,fieldset')?.innerText || ''")
            parent_text_lower = parent_text.lower()
            # Sponsor/visa → always No; authorized/remote → always Yes
            should_check = False
            if ("sponsor" in parent_text_lower or "visa" in parent_text_lower) and val == "no":
                should_check = True
            elif ("authoriz" in parent_text_lower or "remote" in parent_text_lower or
                  "eligible" in parent_text_lower) and val == "yes":
                should_check = True
            if should_check:
                await radio.check()
                filled += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("AppQ radio skip: %s", exc)

    return filled


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PLATFORM CLASS
# ─────────────────────────────────────────────────────────────────────────────

class WorkdayPlatform(BasePlatformApply):
    """
    Workday multi-tenant job application automator.

    Uses stable [data-automation-id] selectors that work across ALL Workday
    tenants (NVIDIA, Microsoft, Amazon, Stripe, Databricks, etc.).

    Handles:
      - Account creation / sign-in
      - 6-page application flow (My Information → Review)
      - Resume file upload
      - Generic application question answering
      - EEO / demographic fields → always "Decline to Identify"
      - DRY_RUN gate (fill but never submit)
    """

    platform_name: str = "workday"

    # ── Public entry point ────────────────────────────────────────────────────

    async def apply(
        self,
        job_url: Optional[str] = None,
        profile: Optional[Dict[str, Any]] = None,
    ) -> Union[ApplyResult, Dict[str, Any]]:
        """
        Run the full Workday apply flow for a given job URL.

        Args:
            job_url : Full Workday job URL (any tenant subdomain).
            profile : User profile dict from user_profile.json / config loader.

        Returns:
            Result dict with keys: status, platform, fields_filled,
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
                "platform": "workday",
                "fields_filled": 0,
                "proof_screenshot_path": "",
                "error": "",
            }
        dry_run = os.getenv("DRY_RUN").lower()
        fields_filled = 0
        proof_path = ""
        browser: Optional[Browser] = None

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(
                    headless=os.getenv("PLAYWRIGHT_HEADLESS", "false").lower() == "true",
                    slow_mo=int(os.getenv("PLAYWRIGHT_SLOW_MO", "300")),
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
                await self._inject_workday_cookies(context, job_url)
                page = await context.new_page()

                logger.info("[workday] Navigating to: %s", job_url[:80])
                await page.goto(job_url, timeout=30_000)
                await page.wait_for_load_state("networkidle", timeout=20_000)
                proof_path = await _screenshot(page, "workday_01_landing")

                # ── 1. Sign in or create account ──────────────────────────────
                auth_result = await self._handle_auth(page, profile)
                if auth_result == "login_required":
                    return self._result("login_required", 0, False, proof_path,
                                        "Login required — WORKDAY_EMAIL/PASSWORD not set")

                proof_path = await _screenshot(page, "workday_02_post_auth")

                # ── 2. Click Apply button ─────────────────────────────────────
                clicked = await self._click_apply(page)
                if not clicked:
                    proof_path = await _screenshot(page, "workday_03_no_apply_btn")
                    return self._result("error", 0, False, proof_path,
                                        "Apply button not found")

                await page.wait_for_load_state("networkidle", timeout=15_000)
                await asyncio.sleep(1)
                proof_path = await _screenshot(page, "workday_03_apply_clicked")

                # ── 3. Page 1 — My Information ────────────────────────────────
                logger.info("[workday] Page 1: My Information")
                filled = await self._fill_my_information(page, profile)
                fields_filled += filled
                proof_path = await _screenshot(page, "workday_04_my_information")
                await _click_next(page, "Save and Continue (P1)")
                await asyncio.sleep(1)

                # ── 4. Page 2 — My Experience ─────────────────────────────────
                logger.info("[workday] Page 2: My Experience")
                filled = await self._fill_my_experience(page, profile)
                fields_filled += filled
                proof_path = await _screenshot(page, "workday_05_my_experience")
                await _click_next(page, "Save and Continue (P2)")
                await asyncio.sleep(1)

                # ── 5. Page 3 — Application Questions ────────────────────────
                logger.info("[workday] Page 3: Application Questions")
                filled = await _handle_application_questions(page, profile)
                fields_filled += filled
                proof_path = await _screenshot(page, "workday_06_app_questions")
                await _click_next(page, "Save and Continue (P3)")
                await asyncio.sleep(1)

                # ── 6. Page 4 — Voluntary Disclosures ────────────────────────
                logger.info("[workday] Page 4: Voluntary Disclosures")
                filled = await _select_decline_for_eeo(page)
                fields_filled += filled
                proof_path = await _screenshot(page, "workday_07_voluntary")
                await _click_next(page, "Save and Continue (P4)")
                await asyncio.sleep(1)

                # ── 7. Page 5 — Self Identify ─────────────────────────────────
                logger.info("[workday] Page 5: Self Identify")
                filled = await _select_decline_for_eeo(page)
                fields_filled += filled
                proof_path = await _screenshot(page, "workday_08_self_identify")
                await _click_next(page, "Save and Continue (P5)")
                await asyncio.sleep(1)

                # ── 8. Page 6 — Review ────────────────────────────────────────
                logger.info("[workday] Page 6: Review")
                proof_path = await _screenshot(page, "workday_09_review")

                # ── DRY_RUN GATE ──────────────────────────────────────────────
                if dry_run:
                    logger.info("[workday] DRY_RUN=true — stopping before submit")
                    return self._result("dry_run", fields_filled, True, proof_path, None)

                # ── 9. Submit ─────────────────────────────────────────────────
                logger.info("[workday] Submitting application...")
                submitted = await self._click_submit(page)
                if not submitted:
                    return self._result("error", fields_filled, False, proof_path,
                                        "Submit button not found on review page")

                await asyncio.sleep(3)
                proof_path = await _screenshot(page, "workday_10_submitted")
                logger.info("[workday] Application submitted successfully")
                return self._result("applied", fields_filled, False, proof_path, None)

        except Exception as exc:  # noqa: BLE001
            logger.error("[workday] Fatal error: %s", exc, exc_info=True)
            try:
                proof_path = await _screenshot(page, "workday_error")  # type: ignore[name-defined]
            except Exception:
                pass
            return self._result("error", fields_filled, False, proof_path, str(exc))

    # ── Auth handler ──────────────────────────────────────────────────────────

    async def _inject_workday_cookies(self, context, job_url: str) -> None:
        """Inject session cookies to bypass Workday login walls (Issue 7)."""
        session_id = os.getenv("WORKDAY_SESSION_ID")
        if session_id:
            import urllib.parse
            domain = urllib.parse.urlparse(job_url).netloc
            await context.add_cookies([
                {"name": "PLAY_SESSION", "value": session_id, "domain": domain, "path": "/"}
            ])
            self.logger.info("[workday] Injected WORKDAY_SESSION_ID cookie for %s", domain)

    async def _handle_auth(self, page: Page, profile: dict[str, Any]) -> str:
        """
        Detect login wall and sign in or create account.

        Returns "ok" on success, "login_required" if credentials missing.
        """
        email    = os.getenv("WORKDAY_EMAIL", profile.get("email", ""))
        password = os.getenv("WORKDAY_PASSWORD", "")

        # Check if login wall is present
        try:
            await page.wait_for_selector(
                f"{SEL['sign_in_btn']}, {SEL['create_acct_btn']}, {SEL['apply_btn']}",
                timeout=8_000,
            )
        except PlaywrightTimeout:
            return "ok"  # No auth wall detected

        has_sign_in = await page.locator(SEL["sign_in_btn"]).count() > 0
        has_apply   = await page.locator(SEL["apply_btn"]).count() > 0

        if has_apply:
            return "ok"  # Job page loaded, no auth wall

        if not email or not password:
            logger.error("[workday] Auth wall detected but WORKDAY_EMAIL/PASSWORD not set")
            return "login_required"

        # Try to sign in first
        if has_sign_in:
            logger.info("[workday] Signing in with existing account")
            try:
                await page.locator(SEL["sign_in_btn"]).click()
                await asyncio.sleep(1)
                await _safe_fill(page, SEL["email_input"], email, label="email")
                await _safe_fill(page, SEL["password_input"], password, label="password")
                await page.locator(SEL["submit_sign_in"]).click()
                await page.wait_for_load_state("networkidle", timeout=15_000)
                logger.info("[workday] Sign-in complete")
                return "ok"
            except Exception as exc:  # noqa: BLE001
                logger.warning("[workday] Sign-in failed, trying create account: %s", exc)

        # Fall back to account creation
        logger.info("[workday] Creating new Workday account")
        try:
            await page.locator(SEL["create_acct_btn"]).click()
            await asyncio.sleep(1)
            await _safe_fill(page, SEL["new_acct_email"], email, label="new_acct_email")
            await _safe_fill(page, SEL["new_acct_verify"], email, label="new_acct_verify")
            await _safe_fill(page, SEL["new_acct_password"], password, label="new_acct_password")
            await page.locator(SEL["new_acct_submit"]).click()
            await page.wait_for_load_state("networkidle", timeout=15_000)
            logger.info("[workday] Account created")
            return "ok"
        except Exception as exc:  # noqa: BLE001
            logger.error("[workday] Account creation failed: %s", exc)
            return "login_required"

    # ── Click Apply ───────────────────────────────────────────────────────────

    async def _click_apply(self, page: Page) -> bool:
        """Find and click the Apply button on the job description page."""
        selectors = [
            SEL["apply_btn"],
            SEL["apply_manually_btn"],
            "button:has-text('Apply Now')",
            "button:has-text('Apply')",
            "a:has-text('Apply')",
        ]
        for sel in selectors:
            try:
                el = page.locator(sel).first
                if await el.count() > 0:
                    await el.scroll_into_view_if_needed()
                    await el.click()
                    logger.info("[workday] Clicked apply button: %s", sel[:50])
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.debug("[workday] Apply btn attempt failed [%s]: %s", sel[:40], exc)
        return False

    # ── Page 1: My Information ────────────────────────────────────────────────

    async def _fill_my_information(self, page: Page,
                                    profile: dict[str, Any]) -> int:
        """Fill Page 1 — My Information fields."""
        filled = 0

        field_map = [
            (SEL["first_name"],   profile.get("first_name", ""),      "first_name"),
            (SEL["last_name"],    profile.get("last_name", ""),        "last_name"),
            (SEL["phone_number"], profile.get("phone", ""),            "phone"),
            (SEL["address_line1"],profile.get("address_line1", ""),    "address_line1"),
            (SEL["city"],         profile.get("city", ""),             "city"),
            (SEL["postal_code"],  profile.get("postal_code", ""),      "postal_code"),
        ]
        for selector, value, label in field_map:
            if value and await _safe_fill(page, selector, value, label=label):
                filled += 1

        # Country dropdown
        country = profile.get("country", "India")
        if await _safe_select_text(page, SEL["country_dropdown"], country, label="country"):
            filled += 1

        # "How did you hear" — pick first available option (not blank)
        try:
            el = page.locator(SEL["how_hear_dropdown"]).first
            if await el.count() > 0:
                await el.select_option(index=1)  # index 0 is usually blank
                filled += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("[workday] how_hear dropdown: %s", exc)

        # "Previously worked" → always No
        try:
            no_radio = page.locator(SEL["prev_worked_no"]).first
            if await no_radio.count() > 0:
                await no_radio.check()
                filled += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("[workday] prev_worked radio: %s", exc)

        logger.info("[workday] P1 filled %d fields", filled)
        return filled

    # ── Page 2: My Experience ─────────────────────────────────────────────────

    async def _fill_my_experience(self, page: Page,
                                   profile: dict[str, Any]) -> int:
        """Fill Page 2 — My Experience (resume upload + education + work)."""
        filled = 0

        # Resume upload
        resume_path = profile.get("resume_path", os.getenv("RESUME_PATH", ""))
        if resume_path and Path(resume_path).exists():
            try:
                # Click "Add Resume" button to reveal file input if hidden
                add_btn = page.locator(SEL["resume_btn"]).first
                if await add_btn.count() > 0:
                    await add_btn.click()
                    await asyncio.sleep(1)

                upload_input = page.locator(SEL["resume_upload"]).first
                if await upload_input.count() > 0:
                    await upload_input.set_input_files(resume_path)
                    await asyncio.sleep(2)  # wait for upload progress
                    filled += 1
                    logger.info("[workday] Resume uploaded: %s", resume_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[workday] Resume upload failed: %s", exc)
        else:
            logger.warning("[workday] Resume path missing/not found: %s", resume_path)

        # Education
        edu = profile.get("education", {})
        if edu:
            edu_fields = [
                (SEL["school_name"],   edu.get("school", ""),        "school"),
                (SEL["field_of_study"],edu.get("field_of_study", ""),"field_of_study"),
                (SEL["gpa"],           edu.get("gpa", ""),           "gpa"),
            ]
            for selector, value, label in edu_fields:
                if value and await _safe_fill(page, selector, value, label=label):
                    filled += 1
            if edu.get("degree"):
                if await _safe_select_text(page, SEL["degree_dropdown"],
                                            edu["degree"], label="degree"):
                    filled += 1

        # LinkedIn / portfolio
        for sel_key, profile_key in [
            ("linkedin_url", "linkedin_url"),
            ("website_url",  "portfolio_url"),
        ]:
            val = profile.get(profile_key, "")
            if val and await _safe_fill(page, SEL[sel_key], val, label=sel_key):
                filled += 1

        logger.info("[workday] P2 filled %d fields", filled)
        return filled

    # ── Submit ────────────────────────────────────────────────────────────────

    async def _click_submit(self, page: Page) -> bool:
        """Click the final submit button on the Review page."""
        for sel in [SEL["review_submit_btn"], SEL["submit_btn"], "button[type='submit']"]:
            try:
                btn = page.locator(sel).first
                if await btn.count() > 0:
                    await btn.scroll_into_view_if_needed()
                    await btn.click()
                    logger.info("[workday] Submit button clicked: %s", sel[:50])
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.debug("[workday] Submit attempt failed [%s]: %s", sel[:40], exc)
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
            logger.error("[workday] _apply_compat error: %s", exc)
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
            "platform"             : "workday",
            "fields_filled"        : fields_filled,
            "dry_run_stopped"      : dry_run_stopped,
            "proof_screenshot_path": proof_screenshot_path,
            "error"                : error,
        }

WorkdayApply = WorkdayPlatform
