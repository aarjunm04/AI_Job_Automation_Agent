"""
ATS-agnostic form filler for AI Job Application Agent.

Production-grade Playwright form filler that handles every field type found
across Greenhouse, Lever, Workday, Ashby, iCIMS, SmartRecruiters, BambooHR,
and native platform application forms.

The ``FormFiller`` class is instantiated per-application and called by
``tools/apply_tools.py``.  It uses LLM reasoning (via ``litellm``) for
complex / custom questions and deterministic label-matching for standard
profile fields.  Every interaction is fully **DRY_RUN** aware — when
``DRY_RUN=True`` the class scans, detects, maps and reasons but never
actually fills, selects, clicks or uploads.

Inspired by battle-tested patterns from:
- **LinkedIn_AIHawk** — LLM-powered custom Q&A, React input handling,
  multi-step navigation, resume upload strategy.
- **EasyApplyJobsBot** — field detection, dropdown selection, checkbox
  handling, time-delay humanisation.

All user profile data is sourced exclusively from environment variables
loaded via ``~/narad.env``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import agentops
from agentops.sdk.decorators import agent, operation
from litellm import completion
from playwright.async_api import ElementHandle, Locator, Page

from config.settings import api_config, run_config
from integrations.llm_interface import LLMInterface

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants derived from config singletons
# ---------------------------------------------------------------------------
DRY_RUN: bool = run_config.dry_run
RESUME_DIR: Path = Path(run_config.resume_dir)

__all__ = ["FormFiller", "FieldType", "FormField", "FillResult"]


# ═══════════════════════════════════════════════════════════════════════════
# Enums and dataclasses
# ═══════════════════════════════════════════════════════════════════════════


class FieldType(Enum):
    """Enumeration of all supported form field types."""

    TEXT = "text"
    EMAIL = "email"
    PHONE = "phone"
    TEXTAREA = "textarea"
    SELECT = "select"
    RADIO = "radio"
    CHECKBOX = "checkbox"
    FILE_UPLOAD = "file_upload"
    DATE = "date"
    NUMBER = "number"
    CUSTOM_QUESTION = "custom_question"
    UNKNOWN = "unknown"


@dataclass
class FormField:
    """Representation of a single interactive form field on the page."""

    selector: str
    field_type: FieldType
    label: str = ""
    placeholder: str = ""
    required: bool = False
    options: list[str] = field(default_factory=list)
    value_to_fill: str = ""


@dataclass
class FillResult:
    """Aggregate outcome of an entire form-fill pass.

    Attributes:
        total_fields: Number of interactive fields detected.
        filled: Number of fields successfully filled.
        skipped: Number of fields intentionally skipped.
        failed: Number of fields where fill was attempted but failed.
        llm_calls: Count of LLM completion calls made for custom questions.
        custom_questions: List of ``{question, answer}`` dicts from LLM.
        errors: Human-readable error strings for debugging.
        success: ``True`` when the form is considered adequately filled.
    """

    total_fields: int = 0
    filled: int = 0
    skipped: int = 0
    failed: int = 0
    llm_calls: int = 0
    custom_questions: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    success: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# UserProfile — loaded once at FormFiller init from env vars
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class UserProfile:
    """Candidate profile sourced exclusively from environment variables.

    All values originate from ``~/narad.env``.  No values are ever
    hard-coded.
    """

    first_name: str = ""
    last_name: str = ""
    full_name: str = ""
    email: str = ""
    phone: str = ""
    linkedin_url: str = ""
    portfolio_url: str = ""
    location: str = ""
    years_experience: str = ""
    accepted_job_types: str = ""
    accepted_locations: str = ""

    @classmethod
    def from_env(cls) -> "UserProfile":
        """Build a ``UserProfile`` by reading ``os.getenv()`` values.

        Returns:
            Populated ``UserProfile`` instance.
        """
        name: str = os.getenv("USERNAME", "")
        parts: list[str] = name.split()
        return cls(
            first_name=parts[0] if parts else "",
            last_name=parts[-1] if len(parts) > 1 else "",
            full_name=name,
            email=os.getenv("USER_EMAIL", ""),
            phone=os.getenv("USER_PHONE", ""),
            linkedin_url=os.getenv("USER_LINKEDIN_URL", ""),
            portfolio_url=os.getenv("USER_PORTFOLIO_URL", ""),
            location=os.getenv("USER_LOCATION", ""),
            years_experience=os.getenv("USER_YEARS_EXPERIENCE", "0"),
            accepted_job_types=os.getenv("USER_ACCEPTED_JOB_TYPES", "full-time"),
            accepted_locations=os.getenv("USER_ACCEPTED_LOCATIONS", "Remote"),
        )


# ═══════════════════════════════════════════════════════════════════════════
# FormFiller
# ═══════════════════════════════════════════════════════════════════════════


@agent
class FormFiller:
    """ATS-agnostic, LLM-powered Playwright form filler.

    Instantiated once per application by ``tools/apply_tools.py``.  Handles
    standard profile fields deterministically and delegates custom /
    ambiguous questions to an LLM via ``litellm``.

    When ``DRY_RUN=True`` the class performs full detection and reasoning
    but **never** writes to the DOM, uploads files, or clicks any buttons.
    """

    def __init__(
        self,
        page: Page,
        job_title: str,
        job_description: str,
        company: str,
        resume_filename: str,
        ats_type: str = "unknown",
    ) -> None:
        self.page: Page = page
        self.job_title: str = job_title
        self.job_description: str = job_description
        self.company: str = company
        self.resume_path: Path = RESUME_DIR / resume_filename
        self.ats_type: str = ats_type

        self.profile: UserProfile = UserProfile.from_env()
        self.llm = LLMInterface().get_llm("APPLY_AGENT")
        self.llm_model_string: str = self._get_model_string()
        self.result: FillResult = FillResult()
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model_string(self) -> str:
        """Extract the model identifier from the LLM object.

        Returns:
            The model string for ``litellm.completion`` calls, falling back
            to ``"xai/grok-4-1-fast-reasoning"`` if the attribute is absent.
        """
        try:
            model_attr = getattr(self.llm, "model", None)
            if model_attr:
                return str(model_attr)
        except Exception:  # noqa: BLE001
            pass
        return "xai/grok-4-1-fast-reasoning"

    async def _human_delay(
        self, min_ms: int = 80, max_ms: int = 300
    ) -> None:
        """Sleep for a random duration to humanise interaction timing.

        Inspired by the AIHawk delay pattern — prevents bot-detection
        heuristics from flagging robotic instant fills.

        Args:
            min_ms: Minimum delay in milliseconds.
            max_ms: Maximum delay in milliseconds.
        """
        delay_seconds: float = random.randint(min_ms, max_ms) / 1000.0
        await asyncio.sleep(delay_seconds)

    async def _safe_fill(self, selector: str, value: str) -> bool:
        """Fill a text-like input using three escalating strategies.

        Strategy order (AIHawk-inspired):
          1. ``page.fill()`` — standard Playwright fill.
          2. ``page.evaluate()`` — direct DOM value assignment + React
             ``input`` / ``change`` event dispatch.
          3. ``page.click()`` + ``keyboard.type()`` — physical keyboard
             simulation.

        Args:
            selector: CSS selector for the target input.
            value: Value string to fill.

        Returns:
            ``True`` if any strategy succeeded, ``False`` if all failed.
        """
        if DRY_RUN:
            self.logger.debug("dry_run: would fill %s → '%s'", selector, value)
            return True

        # Strategy 1 — standard fill
        try:
            await self.page.fill(selector, value)
            await self._human_delay()
            return True
        except Exception:  # noqa: BLE001
            pass

        # Strategy 2 — React-controlled input via JS evaluation
        try:
            escaped_value = value.replace("'", "\\'")
            await self.page.evaluate(
                f"document.querySelector('{selector}').value = '{escaped_value}'"
            )
            await self.page.dispatch_event(selector, "input")
            await self.page.dispatch_event(selector, "change")
            await self._human_delay()
            return True
        except Exception:  # noqa: BLE001
            pass

        # Strategy 3 — keyboard simulation
        try:
            await self.page.click(selector)
            await self.page.keyboard.type(value, delay=50)
            await self._human_delay()
            return True
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "_safe_fill: all 3 strategies failed for %s: %s", selector, exc
            )
            return False

    async def _safe_select(
        self,
        selector: str,
        options: list[str],
        preferred_value: str,
    ) -> bool:
        """Select an option from a ``<select>`` dropdown.

        Tries exact label match first, then case-insensitive partial match,
        then falls back to the first non-empty option.

        Args:
            selector: CSS selector for the ``<select>`` element.
            options: Known option labels from the dropdown.
            preferred_value: Desired option text.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        if DRY_RUN:
            self.logger.debug(
                "dry_run: would select '%s' on %s", preferred_value, selector
            )
            return True

        try:
            # Step 1 — exact label match
            try:
                await self.page.select_option(selector, label=preferred_value)
                await self._human_delay()
                return True
            except Exception:  # noqa: BLE001
                pass

            # Step 2 — case-insensitive partial match
            preferred_lower: str = preferred_value.lower()
            for option in options:
                if preferred_lower in option.lower() or option.lower() in preferred_lower:
                    try:
                        await self.page.select_option(selector, label=option)
                        await self._human_delay()
                        return True
                    except Exception:  # noqa: BLE001
                        continue

            # Step 3 — fallback to first non-empty option
            non_empty: list[str] = [o for o in options if o.strip()]
            if non_empty:
                await self.page.select_option(selector, label=non_empty[0])
                await self._human_delay()
                return True

            return False  # pragma: no cover
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "_safe_select: failed for %s: %s", selector, exc
            )
            return False

    async def _handle_radio_group(
        self,
        group_selector: str,
        label: str,
        preferred_answer: str,
    ) -> bool:
        """Click the appropriate radio button in a group.

        Falls back to contextual heuristics (sponsorship → No, authorised
        → Yes) when the preferred answer does not match any radio label.

        Args:
            group_selector: CSS selector matching all radios in the group.
            label: Human-readable question / label text.
            preferred_answer: Desired radio button label text.

        Returns:
            ``True`` on click, ``False`` if nothing found.
        """
        if DRY_RUN:
            self.logger.debug(
                "dry_run: would select radio '%s' for '%s'",
                preferred_answer,
                label,
            )
            return True

        try:
            radios: list[ElementHandle] = await self.page.query_selector_all(
                group_selector
            )
            if not radios:
                return False

            # Build label → element map
            radio_labels: list[tuple[str, ElementHandle]] = []
            for radio in radios:
                r_label: str = (
                    await radio.get_attribute("aria-label")
                    or await radio.get_attribute("value")
                    or ""
                )
                # Also try the parent/sibling text
                if not r_label:
                    parent = await radio.evaluate_handle(
                        "el => el.parentElement"
                    )
                    r_label = (
                        (await parent.inner_text())
                        if parent
                        else ""
                    )
                radio_labels.append((r_label.strip(), radio))

            # Try exact preferred match
            preferred_lower: str = preferred_answer.lower()
            for r_label, element in radio_labels:
                if r_label.lower() == preferred_lower:
                    await element.click()
                    await self._human_delay()
                    return True

            # Contextual fallbacks
            label_lower: str = label.lower()
            target: Optional[str] = None

            if "sponsorship" in label_lower or "visa" in label_lower:
                target = "no"
            elif "authorized" in label_lower or "legally" in label_lower:
                target = "yes"

            if target:
                for r_label, element in radio_labels:
                    if r_label.lower().startswith(target):
                        await element.click()
                        await self._human_delay()
                        return True

            # Last resort — click the first radio
            if radio_labels:
                await radio_labels[0][1].click()
                await self._human_delay()
                return True

            return False
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "_handle_radio_group: failed for '%s': %s", label, exc
            )
            return False

    async def _handle_checkbox(
        self,
        selector: str,
        label: str,
        should_check: bool = True,
    ) -> bool:
        """Check or uncheck a checkbox.

        Args:
            selector: CSS selector for the checkbox input.
            label: Human-readable label (for logging).
            should_check: ``True`` to check, ``False`` to uncheck.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        if DRY_RUN:
            action: str = "check" if should_check else "uncheck"
            self.logger.debug("dry_run: would %s %s", action, selector)
            return True

        try:
            if should_check:
                await self.page.check(selector)
            else:
                await self.page.uncheck(selector)
            await self._human_delay()
            return True
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "_handle_checkbox: failed for %s: %s", selector, exc
            )
            return False

    async def _upload_resume(self, selector: str) -> bool:
        """Upload the resume PDF to a file input.

        When ``DRY_RUN=True``, logs intent and returns ``True`` without
        touching the page.

        Args:
            selector: CSS selector for the ``<input type=file>`` element.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        if DRY_RUN:
            self.logger.debug("dry_run: skipping resume upload to %s", selector)
            return True

        try:
            resume_path: Path = self.resume_path
            if not resume_path.exists():
                # Fallback to default resume
                fallback_path: Path = RESUME_DIR / run_config.default_resume
                if fallback_path.exists():
                    resume_path = fallback_path
                    self.logger.info(
                        "Primary resume missing — falling back to %s",
                        fallback_path.name,
                    )
                else:
                    self.logger.error(
                        "_upload_resume: neither %s nor %s found",
                        self.resume_path,
                        fallback_path,
                    )
                    return False

            await self.page.set_input_files(selector, str(resume_path))
            self.logger.info("Resume uploaded: %s", resume_path.name)
            return True
        except Exception as exc:  # noqa: BLE001
            self.logger.error("_upload_resume: failed for %s: %s", selector, exc)
            return False

    async def _answer_custom_question(
        self,
        question_text: str,
        field_type: FieldType,
        options: list[str] | None = None,
    ) -> str:
        """Answer a custom / ambiguous question via LLM reasoning.

        This is the LLM reasoning core inspired by LinkedIn_AIHawk's
        question-answering module.  Deterministic shortcircuits handle
        common patterns (salary, visa, experience, gender/ethnicity)
        before falling back to a full ``litellm.completion`` call.

        Args:
            question_text: The question text / label from the form.
            field_type: Detected field type for answer formatting.
            options: Available options (for SELECT / RADIO fields).

        Returns:
            Answer string.  Empty string on LLM failure — never raises.
        """
        if options is None:
            options = []
        question_lower: str = question_text.lower()

        # -- Deterministic shortcuts ------------------------------------------

        if "years of experience" in question_lower and field_type == FieldType.NUMBER:
            answer = self.profile.years_experience
            self.result.custom_questions.append(
                {"question": question_text, "answer": answer}
            )
            return answer

        if "visa" in question_lower or "sponsorship" in question_lower:
            answer = "No I do not require visa sponsorship"
            if options:
                answer = self._pick_option(options, ["no", "not required"])
            self.result.custom_questions.append(
                {"question": question_text, "answer": answer}
            )
            return answer

        if "authorized" in question_lower or "eligible" in question_lower:
            answer = "Yes"
            if options:
                answer = self._pick_option(options, ["yes"])
            self.result.custom_questions.append(
                {"question": question_text, "answer": answer}
            )
            return answer

        if "notice period" in question_lower:
            answer = "2 weeks"
            self.result.custom_questions.append(
                {"question": question_text, "answer": answer}
            )
            return answer

        if any(
            kw in question_lower
            for kw in ("gender", "ethnicity", "race", "veteran", "disability")
        ):
            answer = "Prefer not to disclose"
            if options:
                answer = self._pick_option(
                    options,
                    ["prefer not", "decline", "not to disclose", "choose not"],
                )
            self.result.custom_questions.append(
                {"question": question_text, "answer": answer}
            )
            return answer

        # -- LLM call ----------------------------------------------------------

        try:
            system_msg: str = (
                f"You are filling out a job application for {self.job_title} at "
                f"{self.company}. Answer questions concisely and professionally. "
                f"Never make up credentials. Use only factual information provided."
            )

            profile_summary: str = (
                f"Candidate: {self.profile.full_name}, "
                f"{self.profile.years_experience} years experience, "
                f"Location: {self.profile.location}, "
                f"Accepted job types: {self.profile.accepted_job_types}."
            )

            user_prompt_parts: list[str] = [
                profile_summary,
                f"\nQuestion: {question_text}",
            ]

            if field_type in (FieldType.SELECT, FieldType.RADIO) and options:
                user_prompt_parts.append(
                    f"\nChoose EXACTLY one from these options: {options}"
                )
            elif field_type in (FieldType.TEXT, FieldType.TEXTAREA):
                user_prompt_parts.append(
                    "\nAnswer in under 150 words. Be specific and honest."
                )

            if "salary" in question_lower or "compensation" in question_lower:
                user_prompt_parts.append(
                    f"\nAnswer with: {self.profile.years_experience} years "
                    f"experience, targeting remote roles"
                )

            user_msg: str = " ".join(user_prompt_parts)

            xai_api_key: str = api_config.xai_api_key
            completion_kwargs: dict = {
                "model": self.llm_model_string,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": 100,
                "temperature": 0.1,
            }
            if xai_api_key:
                completion_kwargs["api_key"] = xai_api_key
                completion_kwargs["api_base"] = "https://api.x.ai/v1"

            response = completion(**completion_kwargs)
            self.result.llm_calls += 1

            raw_answer: str = (
                response.choices[0].message.content or ""
            ).strip().strip('"').strip("'")

            # If the LLM returned an option from a list, try to match it
            if options and raw_answer:
                raw_lower = raw_answer.lower()
                for opt in options:
                    if opt.lower() in raw_lower or raw_lower in opt.lower():
                        raw_answer = opt
                        break

            self.result.custom_questions.append(
                {"question": question_text, "answer": raw_answer}
            )
            return raw_answer

        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "_answer_custom_question: LLM failed for '%s': %s",
                question_text,
                exc,
            )
            self.result.custom_questions.append(
                {"question": question_text, "answer": ""}
            )
            return ""

    @staticmethod
    def _pick_option(options: list[str], preferred_keywords: list[str]) -> str:
        """Pick the first option whose text contains a preferred keyword.

        Args:
            options: Available option texts.
            preferred_keywords: Lower-case substrings to search for.

        Returns:
            Matching option text, or the first non-empty option as fallback.
        """
        for opt in options:
            opt_lower = opt.lower()
            for kw in preferred_keywords:
                if kw in opt_lower:
                    return opt
        non_empty = [o for o in options if o.strip()]
        return non_empty[0] if non_empty else ""

    async def _detect_field_type(self, element: ElementHandle) -> FieldType:
        """Determine the ``FieldType`` for a given DOM element.

        Args:
            element: Playwright ``ElementHandle`` to inspect.

        Returns:
            Detected ``FieldType`` enum value.
        """
        try:
            tag_name: str = (
                await element.evaluate("el => el.tagName.toLowerCase()")
            )
            if tag_name == "select":
                return FieldType.SELECT
            if tag_name == "textarea":
                return FieldType.TEXTAREA

            input_type: str = (
                (await element.get_attribute("type")) or "text"
            ).lower()

            type_map: dict[str, FieldType] = {
                "text": FieldType.TEXT,
                "email": FieldType.EMAIL,
                "tel": FieldType.PHONE,
                "number": FieldType.NUMBER,
                "date": FieldType.DATE,
                "file": FieldType.FILE_UPLOAD,
                "radio": FieldType.RADIO,
                "checkbox": FieldType.CHECKBOX,
            }
            return type_map.get(input_type, FieldType.UNKNOWN)
        except Exception:  # noqa: BLE001
            return FieldType.UNKNOWN

    async def _detect_field_label(
        self, element: ElementHandle, page: Page
    ) -> str:
        """Extract a human-readable label for a form element.

        Tries multiple strategies in priority order:
          1. ``aria-label`` attribute
          2. ``placeholder`` attribute
          3. Associated ``<label>`` via ``for`` attribute
          4. Parent element text content (trimmed to 100 chars)
          5. ``name`` attribute

        Args:
            element: The form element to label.
            page: Current Playwright page (for querying related elements).

        Returns:
            First non-empty label found, or ``""``.
        """
        try:
            # 1 — aria-label
            aria_label: Optional[str] = await element.get_attribute("aria-label")
            if aria_label and aria_label.strip():
                return aria_label.strip()

            # 2 — placeholder
            placeholder: Optional[str] = await element.get_attribute("placeholder")
            if placeholder and placeholder.strip():
                return placeholder.strip()

            # 3 — associated <label> via id
            elem_id: Optional[str] = await element.get_attribute("id")
            if elem_id:
                label_el = await page.query_selector(f"label[for='{elem_id}']")
                if label_el:
                    label_text: str = (await label_el.inner_text()).strip()
                    if label_text:
                        return label_text

            # 4 — parent text content
            try:
                parent_text: str = await element.evaluate(
                    "el => (el.parentElement ? el.parentElement.textContent : '').trim().slice(0, 100)"
                )
                if parent_text:
                    return parent_text
            except Exception:  # noqa: BLE001
                pass

            # 5 — name attribute
            name: Optional[str] = await element.get_attribute("name")
            if name and name.strip():
                return name.strip()

            return ""
        except Exception:  # noqa: BLE001
            return ""

    def _map_label_to_profile_value(self, label: str) -> Optional[str]:
        """Deterministically map a field label to a ``UserProfile`` value.

        Uses case-insensitive substring matching against common label
        patterns found across ATS platforms.

        Args:
            label: Human-readable label text.

        Returns:
            Matching profile value, or ``None`` if no match is found
            (triggers the LLM custom-question path).
        """
        label_lower: str = label.lower()

        mapping: list[tuple[list[str], str]] = [
            (["first name"], self.profile.first_name),
            (["last name", "surname", "family name"], self.profile.last_name),
            (
                ["full name", "your name", "candidate name", "applicant name"],
                self.profile.full_name,
            ),
            (["email"], self.profile.email),
            (
                ["phone", "mobile", "contact number", "telephone"],
                self.profile.phone,
            ),
            (["linkedin"], self.profile.linkedin_url),
            (
                ["portfolio", "website", "github", "personal url", "personal site"],
                self.profile.portfolio_url,
            ),
            (
                [
                    "location",
                    "city",
                    "where are you based",
                    "current location",
                    "address",
                ],
                self.profile.location,
            ),
            (
                ["years of experience", "experience", "years exp"],
                self.profile.years_experience,
            ),
        ]

        for keywords, value in mapping:
            for kw in keywords:
                if kw in label_lower:
                    return value

        return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def scan_form_fields(self) -> list[FormField]:
        """Scan the current page and return all interactive form fields.

        Detects every visible input (excluding hidden / submit / button),
        select, and textarea element.  For ``<select>`` elements the
        available options are also captured.

        Returns:
            List of ``FormField`` dataclass instances.
        """
        try:
            selectors: str = (
                "input:not([type=hidden]):not([type=submit]):not([type=button]), "
                "select, textarea"
            )
            elements: list[ElementHandle] = (
                await self.page.query_selector_all(selectors)
            )

            fields: list[FormField] = []
            for element in elements:
                try:
                    field_type: FieldType = await self._detect_field_type(element)
                    label: str = await self._detect_field_label(element, self.page)
                    options: list[str] = []

                    if field_type == FieldType.SELECT:
                        try:
                            options = await self.page.evaluate(
                                """(el) => {
                                    return Array.from(el.options).map(o => o.text.trim())
                                       .filter(t => t.length > 0);
                                }""",
                                element,
                            )
                        except Exception:  # noqa: BLE001
                            pass

                    required: bool = False
                    try:
                        required = (
                            await element.get_attribute("required") is not None
                            or await element.get_attribute("aria-required") == "true"
                        )
                    except Exception:  # noqa: BLE001
                        pass

                    placeholder: str = ""
                    try:
                        placeholder = (
                            await element.get_attribute("placeholder") or ""
                        )
                    except Exception:  # noqa: BLE001
                        pass

                    # Build a robust CSS selector for this element
                    elem_selector: str = ""
                    try:
                        elem_id = await element.get_attribute("id")
                        if elem_id:
                            elem_selector = f"#{elem_id}"
                        else:
                            elem_name = await element.get_attribute("name")
                            tag = await element.evaluate(
                                "el => el.tagName.toLowerCase()"
                            )
                            if elem_name:
                                elem_selector = f"{tag}[name='{elem_name}']"
                            else:
                                # Fallback: generate nth-child selector
                                elem_selector = await element.evaluate(
                                    """(el) => {
                                        const tag = el.tagName.toLowerCase();
                                        const parent = el.parentElement;
                                        if (!parent) return tag;
                                        const siblings = Array.from(
                                            parent.querySelectorAll(':scope > ' + tag)
                                        );
                                        const idx = siblings.indexOf(el) + 1;
                                        return tag + ':nth-child(' + idx + ')';
                                    }"""
                                )
                    except Exception:  # noqa: BLE001
                        elem_selector = "unknown"

                    fields.append(
                        FormField(
                            selector=elem_selector,
                            field_type=field_type,
                            label=label,
                            placeholder=placeholder,
                            required=required,
                            options=options,
                        )
                    )
                except Exception as inner_exc:  # noqa: BLE001
                    self.logger.warning(
                        "scan_form_fields: skipping element: %s", inner_exc
                    )
                    continue

            self.logger.info(
                "scan_form_fields: detected %d fields (ats=%s)",
                len(fields),
                self.ats_type,
            )
            return fields
        except Exception as exc:  # noqa: BLE001
            self.logger.error("scan_form_fields: fatal error: %s", exc)
            return []

    @operation
    async def fill_all_fields(self) -> FillResult:
        """Fill all detected form fields on the current page.

        Processing order:
          1. File upload fields (some ATS reveal subsequent fields only
             after resume upload).
          2. All remaining fields — profile-mapped first, then LLM-answered
             custom questions.

        Returns:
            Populated ``FillResult`` with counts and any errors.
        """
        fields: list[FormField] = await self.scan_form_fields()
        self.result.total_fields = len(fields)

        if not fields:
            self.logger.info("fill_all_fields: no fields detected — nothing to fill")
            self.result.success = True
            return self.result

        # -- Step 1: Process file upload fields first --------------------------
        for f in fields:
            if f.field_type == FieldType.FILE_UPLOAD:
                try:
                    success: bool = await self._upload_resume(f.selector)
                    if success:
                        self.result.filled += 1
                    else:
                        self.result.failed += 1
                        self.result.errors.append(f"upload failed: {f.selector}")
                except Exception as exc:  # noqa: BLE001
                    self.result.failed += 1
                    self.result.errors.append(f"upload error: {exc}")

        # -- Step 2: Process all non-file fields -------------------------------
        for f in fields:
            if f.field_type == FieldType.FILE_UPLOAD:
                continue  # already handled

            try:
                # a. Deterministic profile mapping
                mapped_value: Optional[str] = self._map_label_to_profile_value(
                    f.label
                )

                if mapped_value is not None and mapped_value:
                    # Fill based on field type
                    if f.field_type == FieldType.SELECT:
                        ok = await self._safe_select(
                            f.selector, f.options, mapped_value
                        )
                    elif f.field_type == FieldType.RADIO:
                        ok = await self._handle_radio_group(
                            f.selector, f.label, mapped_value
                        )
                    elif f.field_type == FieldType.CHECKBOX:
                        ok = await self._handle_checkbox(
                            f.selector, f.label, should_check=True
                        )
                    else:
                        ok = await self._safe_fill(f.selector, mapped_value)

                    if ok:
                        self.result.filled += 1
                    else:
                        self.result.failed += 1
                        self.result.errors.append(
                            f"fill failed (mapped): {f.selector} label='{f.label}'"
                        )

                elif f.field_type == FieldType.CHECKBOX:
                    # c. Checkboxes — default to checked unless label is negative
                    label_lower = f.label.lower()
                    should_check = not any(
                        neg in label_lower
                        for neg in ("opt out", "do not", "don't", "unsubscribe")
                    )
                    ok = await self._handle_checkbox(
                        f.selector, f.label, should_check=should_check
                    )
                    if ok:
                        self.result.filled += 1
                    else:
                        self.result.failed += 1

                elif f.field_type in (
                    FieldType.TEXT,
                    FieldType.TEXTAREA,
                    FieldType.SELECT,
                    FieldType.RADIO,
                    FieldType.NUMBER,
                    FieldType.EMAIL,
                    FieldType.PHONE,
                    FieldType.DATE,
                ):
                    # b. No mapped value — ask LLM for a custom answer
                    answer: str = await self._answer_custom_question(
                        f.label, f.field_type, f.options
                    )
                    if answer:
                        if f.field_type == FieldType.SELECT:
                            ok = await self._safe_select(
                                f.selector, f.options, answer
                            )
                        elif f.field_type == FieldType.RADIO:
                            ok = await self._handle_radio_group(
                                f.selector, f.label, answer
                            )
                        else:
                            ok = await self._safe_fill(f.selector, answer)

                        if ok:
                            self.result.filled += 1
                        else:
                            self.result.failed += 1
                            self.result.errors.append(
                                f"fill failed (llm): {f.selector} label='{f.label}'"
                            )
                    else:
                        self.result.skipped += 1
                else:
                    self.result.skipped += 1

                # d. Human delay between every field
                await self._human_delay()

            except Exception as exc:  # noqa: BLE001
                self.result.failed += 1
                self.result.errors.append(f"field error: {f.selector}: {exc}")
                self.logger.warning(
                    "fill_all_fields: field failed %s: %s", f.selector, exc
                )
                continue

        # -- Step 3: Determine overall success ---------------------------------
        total: int = max(self.result.total_fields, 1)
        self.result.success = (
            self.result.failed == 0
            or (self.result.filled / total) >= 0.7
        )

        self.logger.info(
            "FormFiller complete | filled=%d | skipped=%d | failed=%d | "
            "llm_calls=%d | total=%d | success=%s",
            self.result.filled,
            self.result.skipped,
            self.result.failed,
            self.result.llm_calls,
            self.result.total_fields,
            self.result.success,
        )
        return self.result

    async def handle_multi_step_form(
        self, max_steps: int = 8
    ) -> FillResult:
        """Navigate and fill a multi-step / paginated application form.

        Handles Workday (3–7 pages), Lever multi-step, and similar
        paginated ATS forms.  Loops up to ``max_steps`` times, filling
        each page and clicking the "Next" / "Continue" button.

        Args:
            max_steps: Maximum number of form pages to process.

        Returns:
            Accumulated ``FillResult`` across all pages.
        """
        next_button_selectors: list[str] = [
            "button:has-text('Next')",
            "button:has-text('Continue')",
            "button:has-text('Save and Continue')",
            "button:has-text('Save & Continue')",
            "[data-automation-id='bottom-navigation-next-button']",  # Workday
        ]

        confirmation_keywords: list[str] = [
            "review your application",
            "confirm",
            "summary",
            "review and submit",
            "please review",
        ]

        for step in range(max_steps):
            self.logger.info(
                "handle_multi_step_form: step %d/%d", step + 1, max_steps
            )

            # Step 1 — fill the current page
            await self.fill_all_fields()

            # Step 2 — look for a "Next" / "Continue" button
            next_button: Optional[Locator] = None
            for sel in next_button_selectors:
                try:
                    locator: Locator = self.page.locator(sel).first
                    if await locator.count() > 0:
                        next_button = locator
                        break
                except Exception:  # noqa: BLE001
                    continue

            if next_button is None:
                self.logger.info(
                    "handle_multi_step_form: no next button found — "
                    "assuming final page at step %d",
                    step + 1,
                )
                break

            # Step 3 — click next (guarded by DRY_RUN)
            if not DRY_RUN:
                try:
                    await next_button.click()
                    await self.page.wait_for_load_state(
                        "networkidle", timeout=10000
                    )
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning(
                        "handle_multi_step_form: next-click failed at step %d: %s",
                        step + 1,
                        exc,
                    )
                    break
            else:
                self.logger.debug(
                    "dry_run: would click next button at step %d", step + 1
                )

            # Step 4 — check for confirmation / review page
            try:
                page_html: str = await self.page.content()
                html_lower: str = page_html.lower()
                if any(kw in html_lower for kw in confirmation_keywords):
                    self.logger.info(
                        "handle_multi_step_form: review/confirmation page "
                        "detected at step %d — stopping",
                        step + 1,
                    )
                    break
            except Exception:  # noqa: BLE001
                pass

        return self.result
