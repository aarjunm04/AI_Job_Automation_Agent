"""
ATS platform detector for AI Job Application Agent.

Production-grade, 3-layer detection engine that identifies the Applicant
Tracking System powering any job application page:

  **Layer 1 — URL pattern** (instant, zero cost):
    Pure string matching against known ATS domains.

  **Layer 2 — DOM fingerprint** (most accurate):
    Playwright ``query_selector`` probes for platform-specific CSS
    selectors, data-attributes, and element IDs.

  **Layer 3 — LLM classification** (budget-aware fallback):
    ``litellm.completion`` call only when Layers 1+2 both fail.

Outputs an ``ATSProfile`` dataclass consumed by ``FormFiller`` and
``tools/apply_tools.py`` to choose the correct selectors and strategy
for each ATS.

Supported ATS platforms (as of 2026): Greenhouse, Lever, Workday,
Ashby, iCIMS, SmartRecruiters, BambooHR, LinkedIn Easy Apply, and
generic Native / Direct career pages.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import agentops
from agentops.sdk.decorators import agent, operation
from litellm import completion
from playwright.async_api import Page

from config.settings import api_config, run_config
from integrations.llm_interface import LLMInterface

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

__all__ = ["ATSDetector", "ATSType", "ATSProfile", "DetectionMethod"]


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════


class ATSType(Enum):
    """Enumeration of all recognised ATS platforms."""

    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    WORKDAY = "workday"
    ASHBY = "ashby"
    ICIMS = "icims"
    SMARTRECRUITERS = "smartrecruiters"
    BAMBOOHR = "bamboohr"
    LINKEDIN_EASY_APPLY = "linkedin_easy_apply"
    NATIVE = "native"
    UNKNOWN = "unknown"


class DetectionMethod(Enum):
    """How the ATS was identified."""

    URL_PATTERN = "url_pattern"
    DOM_FINGERPRINT = "dom_fingerprint"
    LLM_CLASSIFICATION = "llm_classification"
    FALLBACK = "fallback"


# ═══════════════════════════════════════════════════════════════════════════
# ATSProfile dataclass
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ATSProfile:
    """Structured profile describing an ATS platform's selectors and strategy.

    Consumed by ``FormFiller.fill_all_fields()`` and
    ``tools/apply_tools._run_apply()`` to drive the correct per-platform
    form-fill behaviour.

    Attributes:
        ats_type: Detected ATS platform.
        detection_method: Which detection layer identified the platform.
        confidence: Detection confidence score (0.0–1.0).
        submit_selector: CSS selector(s) for the submit / apply button.
        next_button_selector: CSS selector for the "Next" / "Continue"
            button in multi-step flows.
        first_name_selector: CSS selector for the first-name input.
        last_name_selector: CSS selector for the last-name input.
        email_selector: CSS selector for the email input.
        phone_selector: CSS selector for the phone / tel input.
        resume_upload_selector: CSS selector for the resume file input.
        linkedin_selector: CSS selector for the LinkedIn URL input.
        is_multi_step: ``True`` if the form spans multiple pages.
        requires_login: ``True`` if the ATS requires an active session.
        captcha_risk: Risk level string (``"low"``/``"medium"``/``"high"``).
        apply_strategy: Strategy key for ``FormFiller`` (e.g.
            ``"standard"``, ``"multi_step"``, ``"linkedin_easy_apply"``).
        notes: Free-text notes about the ATS behaviour.
    """

    ats_type: ATSType = ATSType.UNKNOWN
    detection_method: DetectionMethod = DetectionMethod.FALLBACK
    confidence: float = 0.0
    submit_selector: str = ""
    next_button_selector: str = ""
    first_name_selector: str = ""
    last_name_selector: str = ""
    email_selector: str = ""
    phone_selector: str = ""
    resume_upload_selector: str = ""
    linkedin_selector: str = ""
    is_multi_step: bool = False
    requires_login: bool = False
    captcha_risk: str = "low"
    apply_strategy: str = "standard"
    notes: str = ""

    def to_dict(self) -> dict:
        """Serialise the profile to a plain dict for JSON transport.

        Returns:
            Dictionary with all profile fields; enum values are converted
            to their string representation.
        """
        return {
            "ats_type": self.ats_type.value,
            "detection_method": self.detection_method.value,
            "confidence": self.confidence,
            "submit_selector": self.submit_selector,
            "next_button_selector": self.next_button_selector,
            "first_name_selector": self.first_name_selector,
            "last_name_selector": self.last_name_selector,
            "email_selector": self.email_selector,
            "phone_selector": self.phone_selector,
            "resume_upload_selector": self.resume_upload_selector,
            "linkedin_selector": self.linkedin_selector,
            "is_multi_step": self.is_multi_step,
            "requires_login": self.requires_login,
            "captcha_risk": self.captcha_risk,
            "apply_strategy": self.apply_strategy,
            "notes": self.notes,
        }


# ═══════════════════════════════════════════════════════════════════════════
# ATS Selector Registry — single source of truth
# ═══════════════════════════════════════════════════════════════════════════

ATS_SELECTOR_REGISTRY: dict[ATSType, ATSProfile] = {
    # ------------------------------------------------------------------
    # Greenhouse — boards.greenhouse.io
    # ------------------------------------------------------------------
    ATSType.GREENHOUSE: ATSProfile(
        ats_type=ATSType.GREENHOUSE,
        submit_selector="input#submit_app, button[type=submit]",
        next_button_selector="button.next-btn",
        first_name_selector=(
            "input#first_name, input[name='application[first_name]']"
        ),
        last_name_selector=(
            "input#last_name, input[name='application[last_name]']"
        ),
        email_selector="input#email, input[name='application[email]']",
        phone_selector="input#phone, input[name='application[phone]']",
        resume_upload_selector=(
            "input#resume, input[name='application[resume]'], input[type=file]"
        ),
        linkedin_selector=(
            "input[name='application[linkedin_profile]'], "
            "input[id*='linkedin']"
        ),
        is_multi_step=False,
        requires_login=False,
        captcha_risk="low",
        apply_strategy="standard",
        notes=(
            "Single page form. Confirm iframe injection via grnhse_app id."
        ),
    ),
    # ------------------------------------------------------------------
    # Lever — jobs.lever.co
    # ------------------------------------------------------------------
    ATSType.LEVER: ATSProfile(
        ats_type=ATSType.LEVER,
        submit_selector="button[type=submit], button.template-btn-submit",
        next_button_selector="",
        first_name_selector="input[name='name']",
        last_name_selector="",
        email_selector="input[name='email']",
        phone_selector="input[name='phone']",
        resume_upload_selector="input[type=file]",
        linkedin_selector=(
            "input[name='urls[LinkedIn]'], input[placeholder*='LinkedIn']"
        ),
        is_multi_step=False,
        requires_login=False,
        captcha_risk="low",
        apply_strategy="standard",
        notes=(
            "Full name in single name field. No last name field. "
            "Straightforward single page."
        ),
    ),
    # ------------------------------------------------------------------
    # Workday — myworkdayjobs.com
    # ------------------------------------------------------------------
    ATSType.WORKDAY: ATSProfile(
        ats_type=ATSType.WORKDAY,
        submit_selector=(
            "button[data-automation-id='bottom-navigation-next-button'], "
            "button[data-automation-id='click_done']"
        ),
        next_button_selector=(
            "button[data-automation-id='bottom-navigation-next-button']"
        ),
        first_name_selector=(
            "input[data-automation-id='legalNameSection_firstName'], "
            "input[data-automation-id='firstName']"
        ),
        last_name_selector=(
            "input[data-automation-id='legalNameSection_lastName'], "
            "input[data-automation-id='lastName']"
        ),
        email_selector=(
            "input[data-automation-id='email'], input[type='email']"
        ),
        phone_selector=(
            "input[data-automation-id='phone-number'], "
            "input[data-automation-id='phoneNumber']"
        ),
        resume_upload_selector=(
            "input[data-automation-id='file-upload-input-ref'], "
            "input[type=file]"
        ),
        linkedin_selector=(
            "input[data-automation-id='linkedInUrl'], "
            "input[placeholder*='LinkedIn']"
        ),
        is_multi_step=True,
        requires_login=True,
        captcha_risk="medium",
        apply_strategy="multi_step",
        notes=(
            "3-7 step form. data-automation-id is the key. Login often "
            "required. High CAPTCHA risk on enterprise instances."
        ),
    ),
    # ------------------------------------------------------------------
    # Ashby — ashbyhq.com
    # ------------------------------------------------------------------
    ATSType.ASHBY: ATSProfile(
        ats_type=ATSType.ASHBY,
        submit_selector=(
            "button[type=submit], button[data-testid='submit-application']"
        ),
        next_button_selector="button[data-testid='next-button']",
        first_name_selector=(
            "input[name='firstName'], input[data-testid='firstName']"
        ),
        last_name_selector=(
            "input[name='lastName'], input[data-testid='lastName']"
        ),
        email_selector="input[name='email'], input[type='email']",
        phone_selector="input[name='phone'], input[type='tel']",
        resume_upload_selector=(
            "input[type=file], input[data-testid='resume-upload']"
        ),
        linkedin_selector=(
            "input[name='linkedInUrl'], input[placeholder*='LinkedIn']"
        ),
        is_multi_step=True,
        requires_login=False,
        captcha_risk="low",
        apply_strategy="multi_step",
        notes=(
            "React-controlled inputs require dispatch_event strategy. "
            "data-testid reliable."
        ),
    ),
    # ------------------------------------------------------------------
    # iCIMS — icims.com
    # ------------------------------------------------------------------
    ATSType.ICIMS: ATSProfile(
        ats_type=ATSType.ICIMS,
        submit_selector=(
            "button#Submit, input[value='Submit Application']"
        ),
        next_button_selector="button.iCIMS_Button_Next, a[title='Next']",
        first_name_selector=(
            "input[data-field-id='firstname'], input[name='firstname']"
        ),
        last_name_selector=(
            "input[data-field-id='lastname'], input[name='lastname']"
        ),
        email_selector=(
            "input[data-field-id='email'], input[name='email']"
        ),
        phone_selector=(
            "input[data-field-id='phone'], input[name='phone']"
        ),
        resume_upload_selector=(
            "input[type=file], div.iCIMS_AttachButton input"
        ),
        linkedin_selector=(
            "input[data-field-id='linkedin'], input[placeholder*='LinkedIn']"
        ),
        is_multi_step=True,
        requires_login=False,
        captcha_risk="low",
        apply_strategy="multi_step",
        notes=(
            "Enterprise ATS, heavy DOM. data-field-id is the key selector "
            "pattern."
        ),
    ),
    # ------------------------------------------------------------------
    # SmartRecruiters — smartrecruiters.com
    # ------------------------------------------------------------------
    ATSType.SMARTRECRUITERS: ATSProfile(
        ats_type=ATSType.SMARTRECRUITERS,
        submit_selector=(
            "button[data-qa='btn-submit-application'], button[type=submit]"
        ),
        next_button_selector="button[data-qa='btn-next']",
        first_name_selector=(
            "input[name='firstName'], input[data-qa='firstName']"
        ),
        last_name_selector=(
            "input[name='lastName'], input[data-qa='lastName']"
        ),
        email_selector="input[name='email'], input[data-qa='email']",
        phone_selector="input[name='phone'], input[data-qa='phone']",
        resume_upload_selector=(
            "input[type=file], div[data-qa='resume-upload'] input"
        ),
        linkedin_selector=(
            "input[name='web[linkedin]'], input[placeholder*='LinkedIn']"
        ),
        is_multi_step=True,
        requires_login=False,
        captcha_risk="low",
        apply_strategy="multi_step",
        notes=(
            "data-qa attributes are reliable. React-controlled — "
            "use dispatch_event strategy."
        ),
    ),
    # ------------------------------------------------------------------
    # BambooHR — bamboohr.com
    # ------------------------------------------------------------------
    ATSType.BAMBOOHR: ATSProfile(
        ats_type=ATSType.BAMBOOHR,
        submit_selector="button#submit, button[type=submit]",
        next_button_selector="",
        first_name_selector="input[name='firstName'], input#firstName",
        last_name_selector="input[name='lastName'], input#lastName",
        email_selector="input[name='email'], input#email",
        phone_selector="input[name='phone'], input#phone",
        resume_upload_selector="input[type=file], input#resume",
        linkedin_selector=(
            "input[name='linkedIn'], input[placeholder*='LinkedIn']"
        ),
        is_multi_step=False,
        requires_login=False,
        captcha_risk="low",
        apply_strategy="standard",
        notes=(
            "Single page. Standard HTML form. Most reliable ATS to automate."
        ),
    ),
    # ------------------------------------------------------------------
    # LinkedIn Easy Apply — linkedin.com
    # ------------------------------------------------------------------
    ATSType.LINKEDIN_EASY_APPLY: ATSProfile(
        ats_type=ATSType.LINKEDIN_EASY_APPLY,
        submit_selector=(
            "button[aria-label='Submit application'], "
            "button.jobs-apply-button"
        ),
        next_button_selector=(
            "button[aria-label='Continue to next step'], "
            "button[aria-label='Review your application']"
        ),
        first_name_selector=(
            "input[id*='firstName'], input[name*='firstName']"
        ),
        last_name_selector=(
            "input[id*='lastName'], input[name*='lastName']"
        ),
        email_selector="input[id*='email'], input[type='email']",
        phone_selector="input[id*='phoneNumber'], input[type='tel']",
        resume_upload_selector="input[type=file]",
        linkedin_selector="",
        is_multi_step=True,
        requires_login=True,
        captcha_risk="high",
        apply_strategy="linkedin_easy_apply",
        notes=(
            "Requires active LinkedIn session cookie li_at. Modal-based "
            "multi-step. High bot detection risk."
        ),
    ),
    # ------------------------------------------------------------------
    # Native / Direct — generic fallback
    # ------------------------------------------------------------------
    ATSType.NATIVE: ATSProfile(
        ats_type=ATSType.NATIVE,
        submit_selector=(
            "button[type=submit], input[type=submit], "
            "button:has-text('Apply'), button:has-text('Submit')"
        ),
        next_button_selector=(
            "button:has-text('Next'), button:has-text('Continue')"
        ),
        first_name_selector=(
            "input[name*='first'], input[id*='first'], "
            "input[placeholder*='First']"
        ),
        last_name_selector=(
            "input[name*='last'], input[id*='last'], "
            "input[placeholder*='Last']"
        ),
        email_selector=(
            "input[type='email'], input[name*='email'], input[id*='email']"
        ),
        phone_selector=(
            "input[type='tel'], input[name*='phone'], input[id*='phone']"
        ),
        resume_upload_selector="input[type=file]",
        linkedin_selector=(
            "input[placeholder*='LinkedIn'], input[name*='linkedin']"
        ),
        is_multi_step=False,
        requires_login=False,
        captcha_risk="low",
        apply_strategy="standard",
        notes="Generic fallback selectors. Best-effort fill.",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# URL pattern table
# ═══════════════════════════════════════════════════════════════════════════

URL_PATTERNS: dict[ATSType, list[str]] = {
    ATSType.GREENHOUSE: ["greenhouse.io", "boards.greenhouse.io"],
    ATSType.LEVER: ["jobs.lever.co", "lever.co/"],
    ATSType.WORKDAY: ["workday.com", "myworkdayjobs.com"],
    ATSType.ASHBY: ["ashbyhq.com", "jobs.ashbyhq.com"],
    ATSType.ICIMS: ["icims.com"],
    ATSType.SMARTRECRUITERS: ["smartrecruiters.com"],
    ATSType.BAMBOOHR: ["bamboohr.com"],
    ATSType.LINKEDIN_EASY_APPLY: ["linkedin.com/jobs"],
}


# ═══════════════════════════════════════════════════════════════════════════
# DOM fingerprint table
# ═══════════════════════════════════════════════════════════════════════════

DOM_FINGERPRINTS: dict[ATSType, list[str]] = {
    ATSType.GREENHOUSE: [
        "#grnhse_app",
        "form#application_form",
        "input[name*='application[first_name]']",
    ],
    ATSType.LEVER: [
        "div.lever-application",
        "div.application-form",
        "input[name='lever-application']",
    ],
    ATSType.WORKDAY: [
        "div[data-automation-id='applicationWidget']",
        "button[data-automation-id='bottom-navigation-next-button']",
    ],
    ATSType.ASHBY: [
        "div._application",
        "form[data-testid='application-form']",
        "div.ashby-application-form",
    ],
    ATSType.ICIMS: [
        "div#icims_content_form",
        "form.iCIMS_Form",
        "input[data-field-id]",
    ],
    ATSType.SMARTRECRUITERS: [
        "div[data-qa='application-form']",
        "form[action*='smartrecruiters']",
    ],
    ATSType.BAMBOOHR: [
        "div#applicationForm",
        "form.BambooHR-ATS-board",
    ],
    ATSType.LINKEDIN_EASY_APPLY: [
        "div.jobs-easy-apply-modal",
        "div[data-test-modal]",
        "form.jobs-easy-apply-content",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# ATSDetector
# ═══════════════════════════════════════════════════════════════════════════


@agent
class ATSDetector:
    """Three-layer ATS platform detector.

    Detection proceeds URL → DOM → LLM with increasing cost and
    decreasing speed.  The first layer to produce a match short-circuits
    the remaining layers, conserving LLM budget.

    Usage::

        detector = ATSDetector()
        profile  = await detector.detect(page, job_url)
        # profile.ats_type, profile.submit_selector, etc.
    """

    def __init__(self) -> None:
        self.llm_model_string: str = self._resolve_model_string()
        self.logger: logging.Logger = logging.getLogger(
            self.__class__.__name__
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model_string() -> str:
        """Obtain the APPLY_AGENT model string from ``LLMInterface``.

        Returns:
            Model identifier string for ``litellm.completion``.  Falls
            back to ``"xai/grok-4-1-fast-reasoning"`` on any error.
        """
        try:
            llm = LLMInterface().get_llm("APPLY_AGENT")
            model_attr = getattr(llm, "model", None)
            if model_attr:
                return str(model_attr)
        except Exception:  # noqa: BLE001
            pass
        return "xai/grok-4-1-fast-reasoning"

    def _detect_by_url(self, url: str) -> Optional[ATSType]:
        """Layer 1 — detect ATS via URL pattern matching.

        Pure string comparison against ``URL_PATTERNS``.  Zero network
        calls, O(n) over the pattern table — always completes instantly.

        Args:
            url: Full URL of the job application page.

        Returns:
            Matching ``ATSType``, or ``None`` if no pattern matches.
        """
        try:
            url_lower: str = url.lower()
            for ats_type, patterns in URL_PATTERNS.items():
                for pattern in patterns:
                    if pattern in url_lower:
                        return ats_type
            return None
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("_detect_by_url: error: %s", exc)
            return None

    async def _detect_by_dom(self, page: Page) -> Optional[ATSType]:
        """Layer 2 — detect ATS via DOM fingerprint probing.

        Iterates ``DOM_FINGERPRINTS`` and uses Playwright
        ``query_selector`` to test for platform-specific elements.
        First match wins — no further probes are executed.

        Args:
            page: Active Playwright page to inspect.

        Returns:
            Matching ``ATSType``, or ``None`` if no fingerprints match.
        """
        try:
            for ats_type, selectors in DOM_FINGERPRINTS.items():
                for selector in selectors:
                    try:
                        element = await page.query_selector(selector)
                        if element is not None:
                            return ats_type
                    except Exception:  # noqa: BLE001
                        # Individual selector may be invalid for some page
                        # contexts — continue probing.
                        continue
            return None
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("_detect_by_dom: error: %s", exc)
            return None

    async def _detect_by_llm(
        self, page_url: str, page_html: str
    ) -> ATSType:
        """Layer 3 — classify ATS platform via LLM reasoning.

        Called **only** when both URL and DOM detection fail.  Sends a
        truncated HTML snippet (first 3 000 characters) to conserve
        tokens.

        Args:
            page_url: URL of the application page.
            page_html: Full page HTML (will be truncated internally).

        Returns:
            Best-guess ``ATSType``.  Returns ``ATSType.UNKNOWN`` on any
            failure.
        """
        try:
            html_snippet: str = page_html[:3000]

            system_msg: str = (
                "You are an ATS platform classifier. Given this job "
                "application page URL and HTML snippet, identify which ATS "
                "is being used. Reply with EXACTLY one word from this list: "
                "greenhouse, lever, workday, ashby, icims, smartrecruiters, "
                "bamboohr, linkedin_easy_apply, native, unknown. "
                "No explanation."
            )
            user_msg: str = (
                f"URL: {page_url}\n\nHTML snippet:\n{html_snippet}"
            )

            xai_api_key: str = api_config.xai_api_key
            completion_kwargs: dict = {
                "model": self.llm_model_string,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": 10,
                "temperature": 0.0,
            }
            if xai_api_key:
                completion_kwargs["api_key"] = xai_api_key
                completion_kwargs["api_base"] = "https://api.x.ai/v1"

            response = completion(**completion_kwargs)
            raw: str = (
                (response.choices[0].message.content or "")
                .strip()
                .lower()
                .strip(".")
            )

            # Map the raw string to an ATSType enum value
            ats_type_map: dict[str, ATSType] = {
                member.value: member for member in ATSType
            }
            result: ATSType = ats_type_map.get(raw, ATSType.UNKNOWN)

            self.logger.info("LLM ATS classification: %s", result.value)
            return result

        except Exception as exc:  # noqa: BLE001
            self.logger.warning("_detect_by_llm: LLM call failed: %s", exc)
            return ATSType.UNKNOWN

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @operation
    async def detect(self, page: Page, job_url: str) -> ATSProfile:
        """Run the full 3-layer ATS detection pipeline.

        Processing order:
          1. **URL pattern** — instant, free.
          2. **DOM fingerprint** — accurate, one Playwright call per probe.
          3. **LLM classification** — budget-aware last resort.

        The first layer to match short-circuits the rest.

        Args:
            page: Active Playwright page (already navigated to the
                application URL).
            job_url: Full job application URL string.

        Returns:
            Populated ``ATSProfile`` with platform-specific selectors,
            detection metadata, and apply strategy.  Never raises.
        """

        # -- LAYER 1: URL pattern (instant) --------------------------------
        try:
            url_match: Optional[ATSType] = self._detect_by_url(job_url)
            if url_match is not None:
                profile: ATSProfile = copy.copy(
                    ATS_SELECTOR_REGISTRY.get(
                        url_match, ATS_SELECTOR_REGISTRY[ATSType.NATIVE]
                    )
                )
                profile.ats_type = url_match
                profile.detection_method = DetectionMethod.URL_PATTERN
                profile.confidence = 0.90
                self.logger.info(
                    "ATS detected by URL: %s (confidence=%.2f)",
                    url_match.value,
                    profile.confidence,
                )
                return profile
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("detect: URL layer error: %s", exc)

        # -- LAYER 2: DOM fingerprint (accurate) ---------------------------
        try:
            dom_match: Optional[ATSType] = await self._detect_by_dom(page)
            if dom_match is not None:
                profile = copy.copy(
                    ATS_SELECTOR_REGISTRY.get(
                        dom_match, ATS_SELECTOR_REGISTRY[ATSType.NATIVE]
                    )
                )
                profile.ats_type = dom_match
                profile.detection_method = DetectionMethod.DOM_FINGERPRINT
                profile.confidence = 0.95
                self.logger.info(
                    "ATS detected by DOM: %s (confidence=%.2f)",
                    dom_match.value,
                    profile.confidence,
                )
                return profile
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("detect: DOM layer error: %s", exc)

        # -- LAYER 3: LLM classification (fallback) ------------------------
        try:
            html: str = await page.content()
            llm_match: ATSType = await self._detect_by_llm(job_url, html)
            profile = copy.copy(
                ATS_SELECTOR_REGISTRY.get(
                    llm_match, ATS_SELECTOR_REGISTRY[ATSType.NATIVE]
                )
            )
            profile.ats_type = llm_match
            profile.detection_method = DetectionMethod.LLM_CLASSIFICATION
            profile.confidence = 0.65
            self.logger.info(
                "ATS detected by LLM: %s (confidence=%.2f)",
                llm_match.value,
                profile.confidence,
            )
            return profile
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "All ATS detection layers failed: %s — defaulting to NATIVE",
                exc,
            )
            profile = copy.copy(ATS_SELECTOR_REGISTRY[ATSType.NATIVE])
            profile.detection_method = DetectionMethod.FALLBACK
            profile.confidence = 0.30
            return profile

    def get_profile_for_ats(self, ats_type_string: str) -> ATSProfile:
        """Return the ``ATSProfile`` for a known ATS type string.

        Used by ``tools/apply_tools.py`` when the ATS type was already
        resolved by a prior ``detect()`` call and only the selector
        profile is needed.

        Args:
            ats_type_string: ATS type value string (e.g. ``"greenhouse"``).

        Returns:
            Matching ``ATSProfile`` from the registry.  Falls back to
            ``ATSType.NATIVE`` if the string is not recognised.
        """
        try:
            ats_type: ATSType = ATSType(ats_type_string.lower().strip())
        except ValueError:
            self.logger.warning(
                "get_profile_for_ats: unrecognised ATS type '%s' — "
                "falling back to NATIVE",
                ats_type_string,
            )
            ats_type = ATSType.NATIVE

        return copy.copy(
            ATS_SELECTOR_REGISTRY.get(
                ats_type, ATS_SELECTOR_REGISTRY[ATSType.NATIVE]
            )
        )

    @staticmethod
    def get_all_supported_ats() -> list[str]:
        """Return all supported ATS platform type strings.

        Excludes ``ATSType.UNKNOWN`` as it is not a real platform.

        Returns:
            Sorted list of ATS type value strings.
        """
        return sorted(
            member.value
            for member in ATSType
            if member != ATSType.UNKNOWN
        )
