"""Native/generic HTML form applier — universal fallback for unknown ATS.

Used when ``ATSDetector`` cannot identify the ATS platform. Makes a
best-effort attempt to fill standard HTML form fields using keyword
matching against user profile data. Conservative by design — only fills
fields with high-confidence label matches. Routes to manual on any
submit uncertainty.

Handles niche ATS platforms including: Jobvite, BambooHR, ICIMS,
SmartRecruiters, Taleo (basic), JazzHR, Pinpoint, and raw HTML
company career pages.

User profile keys expected (split by caller before passing):
    ``first_name``, ``last_name``, ``email``, ``phone``,
    ``linkedin_url``, ``portfolio_url``, ``location``,
    ``years_experience``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any, List, Tuple

from playwright.async_api import (
    ElementHandle,
    TimeoutError as PlaywrightTimeoutError,
)

from auto_apply.platforms.base_platform import (
    BasePlatformApply,
    ApplyResult,
)

logger = logging.getLogger(__name__)

__all__ = ["NativeFormApply"]


# ---------------------------------------------------------------------------
# Keyword → Profile Mapping
# ---------------------------------------------------------------------------

# Ordered keyword mapping — first match wins per field
_NATIVE_KEYWORD_MAP: List[Tuple[List[str], str]] = [
    (["first name", "firstname", "given name"], "first_name"),
    (["last name", "lastname", "surname", "family name"], "last_name"),
    (["full name", "your name", "candidate", "name"], "full_name"),
    (["email", "e-mail"], "email"),
    (["phone", "mobile", "tel", "telephone", "cell"], "phone"),
    (["linkedin"], "linkedin_url"),
    (["portfolio", "github", "website", "personal url"], "portfolio_url"),
    (["location", "city", "country", "where"], "location"),
    (["years", "experience"], "years_experience"),
    (["salary", "ctc", "compensation"], "_salary"),
    (["sponsor", "visa"], "_sponsor"),
    (["available", "start date", "notice", "join"], "_available"),
    (["how did you hear", "referral", "source"], "_referral"),
]

# Static answers for keys prefixed with underscore
_STATIC_ANSWERS: Dict[str, str] = {
    "_salary": "Open to discussion",
    "_sponsor": "No",
    "_available": "Immediately",
    "_referral": "Online job board",
}

# ATS URL fingerprints for secondary detection (logging only)
_ATS_URL_MAP: Dict[str, str] = {
    "jobvite.com": "jobvite",
    "bamboohr.com": "bamboohr",
    "icims.com": "icims",
    "smartrecruiters.com": "smartrecruiters",
    "taleo.net": "taleo",
    "jazz.co": "jazzhr",
    "pinpoint.com": "pinpoint",
}

# ATS DOM fingerprints for secondary detection (logging only)
_ATS_DOM_MAP: Dict[str, str] = {
    "div.jv-content": "jobvite",
    "div#bambooHRApply": "bamboohr",
    "div[class*='iCIMS']": "icims",
    "div.application": "smartrecruiters",
    "div[id*='Taleo']": "taleo",
    "div.jazz-apply": "jazzhr",
}


# ---------------------------------------------------------------------------
# Native Form Apply Module
# ---------------------------------------------------------------------------


class NativeFormApply(BasePlatformApply):
    """Universal fallback form applier for unknown ATS platforms.

    Detects form fields via DOM scanning, maps them to user profile
    values using keyword matching, fills all matched fields, uploads
    resume if a file input is found, then evaluates fill confidence
    before deciding to submit or reroute to manual.

    Steps:
        1. Navigate + detect form presence + CAPTCHA check.
        2. Scan fields + fill + evaluate confidence + submit/reroute.

    The ``CONFIDENCE_THRESHOLD`` controls whether the module attempts
    auto-submit or reroutes to manual review.
    """

    PLATFORM_NAME: str = "native_form"
    STEPS_TOTAL: int = 2

    CONFIDENCE_THRESHOLD: float = 0.60

    async def apply(self) -> ApplyResult:
        """Best-effort native HTML form fill and submit.

        Detects form fields, maps to user profile via keyword matching,
        fills all matched fields, uploads resume if file input found,
        then evaluates fill confidence before deciding to submit or
        reroute.

        Returns:
            ApplyResult. Routes to manual if confidence is below
            ``CONFIDENCE_THRESHOLD`` or if submit confirmation cannot
            be captured.
        """
        self.steps_completed = 0

        # ── Step 1: Navigate + Detect form ──
        step1_result: ApplyResult | None = (
            await self._step_navigate_and_detect()
        )
        if step1_result is not None:
            return step1_result

        # ── Step 2: Fill + Evaluate + Submit ──
        return await self._step_fill_and_submit()

    # ------------------------------------------------------------------
    # Step 1: Navigate + Detect
    # ------------------------------------------------------------------

    async def _step_navigate_and_detect(self) -> ApplyResult | None:
        """Navigate and verify that fillable form fields exist.

        Returns:
            ApplyResult on failure, None on success to proceed.
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
            await asyncio.sleep(1)

            # Detect secondary ATS fingerprint for logging
            ats_secondary: str = await self._detect_secondary_ats()
            self.logger.info(
                "NativeFormApply: secondary ATS detected as '%s' for %s",
                ats_secondary,
                job_url,
            )

            # Check for forms or input fields
            forms = await self.page.query_selector_all("form")
            if not forms:
                input_count: int = len(
                    await self.page.query_selector_all(
                        "input[type='text'], input[type='email'], textarea"
                    )
                )
                if input_count < 2:
                    return self._build_result(
                        success=False,
                        error_code="UNKNOWN_ATS",
                        reroute_to_manual=True,
                        reroute_reason=(
                            "No form or input fields found on page — "
                            "cannot attempt native form fill"
                        ),
                    )

            if await self._detect_captcha():
                return self._build_result(
                    success=False,
                    error_code="CAPTCHA",
                    reroute_to_manual=True,
                    reroute_reason=(
                        "CAPTCHA detected on native form page"
                    ),
                )

            self.steps_completed = 1
            return None

        except PlaywrightTimeoutError:
            return self._build_result(
                success=False,
                error_code="TIMEOUT",
                reroute_to_manual=True,
                reroute_reason="Native form page load timeout",
            )
        except Exception as e:
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=str(e),
            )

    # ------------------------------------------------------------------
    # Step 2: Fill + Evaluate + Submit
    # ------------------------------------------------------------------

    async def _step_fill_and_submit(self) -> ApplyResult:
        """Scan fields, fill, evaluate confidence, and submit or reroute.

        Returns:
            ApplyResult with outcome.
        """
        try:
            # Scan all visible input fields
            detected: List[Dict[str, Any]] = await self._scan_all_fields()
            total_fields: int = len(detected)

            if total_fields == 0:
                return self._build_result(
                    success=False,
                    error_code="NAV_FAIL",
                    reroute_to_manual=True,
                    reroute_reason=(
                        "No fillable fields detected on native form"
                    ),
                )

            # Fill all matched fields
            filled_count: int
            file_input_found: bool
            filled_count, file_input_found = (
                await self._fill_detected_fields(detected)
            )

            # Resume upload if file input found
            if file_input_found:
                upload_ok: bool = await self._upload_resume(
                    "input[type='file']"
                )
                if upload_ok:
                    filled_count += 1

            # Calculate fill confidence
            confidence: float = (
                filled_count / total_fields if total_fields > 0 else 0.0
            )
            self.logger.info(
                "NativeFormApply: filled %d/%d fields, "
                "confidence=%.2f for %s",
                filled_count,
                total_fields,
                confidence,
                self.job_meta.get("job_url", ""),
            )

            # Below confidence threshold — reroute to manual
            if confidence < self.CONFIDENCE_THRESHOLD:
                return self._build_result(
                    success=False,
                    error_code=None,
                    reroute_to_manual=True,
                    reroute_reason=(
                        f"Fill confidence {confidence:.0%} below "
                        f"threshold {self.CONFIDENCE_THRESHOLD:.0%} — "
                        f"filled {filled_count}/{total_fields} fields. "
                        "Manual review recommended."
                    ),
                )

            # Dry-run check
            if self.dry_run:
                self.logger.info(
                    "[DRY_RUN] Would submit native form for %s "
                    "(confidence=%.2f)",
                    self.job_meta.get("job_url", ""),
                    confidence,
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
                    reroute_reason=(
                        "CAPTCHA appeared before native form submit"
                    ),
                )

            # Click submit
            submit_clicked: bool = await self._click_submit_button()
            if not submit_clicked:
                return self._build_result(
                    success=False,
                    error_code="NAV_FAIL",
                    reroute_to_manual=True,
                    reroute_reason=(
                        "Native form: no submit button found or "
                        "confidence insufficient to auto-submit"
                    ),
                )

            # Capture proof
            await asyncio.sleep(2)
            proof_result: ApplyResult = (
                await self._capture_submission_proof()
            )
            self.steps_completed = 2
            return proof_result

        except PlaywrightTimeoutError:
            return self._build_result(
                success=False,
                error_code="TIMEOUT",
                reroute_to_manual=True,
                reroute_reason="Native form fill/submit timeout",
            )
        except Exception as e:
            return self._build_result(
                success=False,
                error_code="NAV_FAIL",
                reroute_to_manual=True,
                reroute_reason=f"Native form error: {str(e)}",
            )

    # ==================================================================
    # Secondary ATS Detection (logging only)
    # ==================================================================

    async def _detect_secondary_ats(self) -> str:
        """Identify the specific ATS from URL and DOM for logging.

        This is for **logging and metadata only** — it does not change
        the fill logic. Routing is ``ATSDetector``'s responsibility.

        Returns:
            ATS name string: ``"jobvite"`` | ``"bamboohr"`` |
            ``"icims"`` | ``"smartrecruiters"`` | ``"taleo"`` |
            ``"jazzhr"`` | ``"pinpoint"`` | ``"unknown"``.
        """
        url: str = self.page.url.lower()
        for pattern, name in _ATS_URL_MAP.items():
            if pattern in url:
                return name

        # DOM fingerprint fallback
        for selector, name in _ATS_DOM_MAP.items():
            try:
                el = await self.page.query_selector(selector)
                if el:
                    return name
            except Exception:
                continue

        return "unknown"

    # ==================================================================
    # Field Scanning
    # ==================================================================

    async def _scan_all_fields(self) -> List[Dict[str, Any]]:
        """Scan all visible, fillable input fields on the current page.

        Returns a list of field descriptor dicts. Only includes:
        ``input[type=text/email/tel/url/number]``, ``textarea``,
        ``select``, ``input[type=file]``. Excludes hidden, disabled,
        readonly, submit, and button inputs.

        Returns:
            List of field descriptor dicts with keys: ``selector``,
            ``field_type``, ``combined_text``, ``label_text``,
            ``element_handle``, ``index``.
        """
        field_css: str = (
            "input[type='text']:not([disabled]):not([readonly]), "
            "input[type='email']:not([disabled]), "
            "input[type='tel']:not([disabled]), "
            "input[type='url']:not([disabled]), "
            "input[type='number']:not([disabled]), "
            "input:not([type]):not([disabled]):not([readonly]), "
            "textarea:not([disabled]), "
            "select:not([disabled]), "
            "input[type='file']"
        )

        elements: list[ElementHandle] = (
            await self.page.query_selector_all(field_css)
        )
        fields: List[Dict[str, Any]] = []

        for idx, el in enumerate(elements):
            try:
                # Skip invisible elements
                if not await el.is_visible():
                    continue

                # Skip hidden type inputs that slipped through
                el_type: str = (
                    await el.get_attribute("type") or "text"
                )
                if el_type == "hidden":
                    continue

                # Get identifying attributes
                el_id: str = await el.get_attribute("id") or ""
                el_name: str = await el.get_attribute("name") or ""
                el_placeholder: str = (
                    await el.get_attribute("placeholder") or ""
                )

                # Get label text — multiple strategies
                label_text: str = await self._resolve_label(el, el_id)

                combined_text: str = (
                    f"{label_text} {el_placeholder} "
                    f"{el_name} {el_id}"
                ).lower().strip()

                # Build selector — prefer id, then name
                selector: str | None = None
                if el_id:
                    selector = f"#{el_id}"
                elif el_name:
                    selector = f"[name='{el_name}']"

                fields.append(
                    {
                        "selector": selector,
                        "field_type": el_type,
                        "combined_text": combined_text,
                        "label_text": label_text.strip(),
                        "element_handle": el,
                        "index": idx,
                    }
                )
            except Exception as e:
                self.logger.debug(
                    "Field scan error at index %d: %s", idx, str(e)
                )
                continue

        return fields

    async def _resolve_label(
        self, el: ElementHandle, el_id: str
    ) -> str:
        """Resolve the label text for a form field element.

        Tries: ``<label for="id">``, ``aria-label`` attribute,
        parent ``<label>`` walk-up.

        Args:
            el: Playwright ElementHandle for the field.
            el_id: The element's id attribute.

        Returns:
            Label text string, or ``""`` if unresolvable.
        """
        label_text: str = ""

        # Strategy 1: label[for=id]
        if el_id:
            try:
                label_el = await self.page.query_selector(
                    f"label[for='{el_id}']"
                )
                if label_el:
                    label_text = await label_el.inner_text()
                    if label_text.strip():
                        return label_text.strip()
            except Exception:
                pass

        # Strategy 2: aria-label
        try:
            aria: str = await el.get_attribute("aria-label") or ""
            if aria.strip():
                return aria.strip()
        except Exception:
            pass

        # Strategy 3: parent <label> walk-up
        try:
            label_text = await el.evaluate(
                """
                el => {
                    const parent = el.closest('label');
                    return parent ? parent.innerText.trim() : '';
                }
            """
            )
            if label_text.strip():
                return label_text.strip()
        except Exception:
            pass

        return ""

    # ==================================================================
    # Field Filling
    # ==================================================================

    async def _fill_detected_fields(
        self, fields: List[Dict[str, Any]]
    ) -> Tuple[int, bool]:
        """Fill all detected fields using keyword-to-profile mapping.

        Uses ``_NATIVE_KEYWORD_MAP`` for matching and
        ``_STATIC_ANSWERS`` for fixed-value responses. First keyword
        match wins per field — no field is filled twice.

        Args:
            fields: List of field descriptor dicts from
                ``_scan_all_fields``.

        Returns:
            Tuple of ``(filled_count, file_input_found)``.
        """
        profile: Dict[str, Any] = self.user_profile
        full_name: str = (
            f"{profile.get('first_name', '')} "
            f"{profile.get('last_name', '')}"
        ).strip()

        # Build runtime value map
        value_map: Dict[str, str] = {
            "first_name": str(profile.get("first_name", "")),
            "last_name": str(profile.get("last_name", "")),
            "full_name": full_name,
            "email": str(profile.get("email", "")),
            "phone": str(profile.get("phone", "")),
            "linkedin_url": str(profile.get("linkedin_url", "")),
            "portfolio_url": str(profile.get("portfolio_url", "")),
            "location": str(profile.get("location", "")),
            "years_experience": str(
                profile.get("years_experience", "")
            ),
        }
        # Merge static answers
        value_map.update(_STATIC_ANSWERS)

        filled_count: int = 0
        file_input_found: bool = False

        for field in fields:
            # File input — flag for resume upload, skip keyword fill
            if field["field_type"] == "file":
                file_input_found = True
                continue

            # Skip fields with no usable selector
            if not field["selector"]:
                continue

            combined: str = field["combined_text"]
            target_key: str | None = None

            # Match keywords — first match wins
            for keywords, profile_key in _NATIVE_KEYWORD_MAP:
                for kw in keywords:
                    if kw in combined:
                        target_key = profile_key
                        break
                if target_key is not None:
                    break

            if target_key is None:
                continue  # No mapping — skip field

            value: str = value_map.get(target_key, "")
            if not value:
                continue  # Empty profile value — skip

            # Fill the field
            el_handle: ElementHandle = field["element_handle"]
            try:
                tag: str = await el_handle.evaluate(
                    "el => el.tagName.toLowerCase()"
                )
                if tag == "select":
                    try:
                        await el_handle.select_option(label=value)
                        filled_count += 1
                    except Exception:
                        try:
                            await el_handle.select_option(value=value)
                            filled_count += 1
                        except Exception:
                            pass
                else:
                    # Try React fill first, fall back to standard fill
                    selector: str = field["selector"]
                    react_ok: bool = await self._fill_react_input(
                        selector, value
                    )
                    if react_ok:
                        filled_count += 1
                    else:
                        std_ok: bool = await self._fill_text_field(
                            selector, value
                        )
                        if std_ok:
                            filled_count += 1
            except Exception as e:
                self.logger.debug(
                    "Field fill error for '%s': %s",
                    field["label_text"],
                    str(e),
                )
                continue

        return filled_count, file_input_found

    # ==================================================================
    # Submit Button
    # ==================================================================

    async def _click_submit_button(self) -> bool:
        """Find and click the form submit button.

        Uses multiple selector strategies. Prefers buttons with
        "submit", "apply", "send" text. Actively avoids "cancel",
        "back", "save draft" buttons via negative pattern check.

        Returns:
            True if a submit button was found and clicked.
        """
        submit_selectors: list[str] = [
            "button[type='submit']",
            "input[type='submit']",
            "button:has-text('Submit Application')",
            "button:has-text('Submit')",
            "button:has-text('Apply Now')",
            "button:has-text('Apply')",
            "button:has-text('Send Application')",
            "button:has-text('Send')",
            "a:has-text('Submit Application')",
            "[data-qa='submit']",
            "[data-testid='submit']",
            "#submit-app",
            ".submit-btn",
            ".apply-btn",
        ]

        negative_patterns: list[str] = [
            "cancel",
            "back",
            "previous",
            "save draft",
            "save for later",
            "close",
            "delete",
        ]

        for selector in submit_selectors:
            try:
                el = await self.page.query_selector(selector)
                if not el:
                    continue
                if not await el.is_visible():
                    continue

                # Verify button text is not a negative action
                btn_text: str = (await el.inner_text()).lower().strip()
                if any(neg in btn_text for neg in negative_patterns):
                    continue

                # Scroll into view and click
                await el.evaluate(
                    "el => el.scrollIntoView("
                    "{behavior: 'smooth', block: 'center'})"
                )
                await asyncio.sleep(0.3)
                await el.click()

                try:
                    await self.page.wait_for_load_state(
                        "networkidle", timeout=10000
                    )
                except PlaywrightTimeoutError:
                    pass

                self.logger.info(
                    "Native form submit clicked via: %s", selector
                )
                return True
            except Exception:
                continue

        return False

    # ==================================================================
    # Proof Capture
    # ==================================================================

    async def _capture_submission_proof(self) -> ApplyResult:
        """Capture proof of submission from the post-submit page.

        Tries multiple proof strategies in priority order. Routes to
        manual if no proof can be found.

        Returns:
            ApplyResult with proof data or ``reroute_to_manual=True``.
        """
        current_url: str = self.page.url

        # Strategy 1: explicit success / thank-you text
        success_selectors: list[str] = [
            "h1:has-text('Thank you')",
            "h2:has-text('Thank you')",
            "h1:has-text('Application submitted')",
            "h2:has-text('Application submitted')",
            "div:has-text('successfully submitted')",
            "p:has-text('successfully submitted')",
            "div:has-text('application has been received')",
            "div[class*='success']",
            "div[class*='confirmation']",
            "div[class*='thank']",
            ".success-message",
            ".confirmation-message",
            "#confirmation",
            "#success",
        ]

        for selector in success_selectors:
            try:
                el = await self.page.query_selector(selector)
                if el and await el.is_visible():
                    text: str = await el.inner_text()
                    return self._build_result(
                        success=True,
                        proof_type="form_disappearance",
                        proof_value=text.strip()[:200],
                        proof_confidence=0.88,
                    )
            except Exception:
                continue

        # Strategy 2: URL-based confirmation
        url_patterns: list[str] = [
            "confirmation",
            "thank",
            "thanks",
            "success",
            "submitted",
            "complete",
            "done",
            "applied",
        ]
        if any(p in current_url.lower() for p in url_patterns):
            return self._build_result(
                success=True,
                proof_type="success_url",
                proof_value=current_url,
                proof_confidence=0.85,
            )

        # Strategy 3: form + inputs disappeared from page
        form_els = await self.page.query_selector_all("form")
        input_els = await self.page.query_selector_all(
            "input[type='text'], input[type='email']"
        )
        if len(form_els) == 0 and len(input_els) == 0:
            return self._build_result(
                success=True,
                proof_type="form_disappearance",
                proof_value=current_url,
                proof_confidence=0.65,
            )

        # No proof found — reroute to manual
        return self._build_result(
            success=False,
            error_code="PROOF_FAIL",
            reroute_to_manual=True,
            reroute_reason=(
                "Native form: submit clicked but no confirmation "
                f"proof found. Final URL: {current_url}"
            ),
        )
