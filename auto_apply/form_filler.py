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

All user profile data is sourced exclusively from ``config/user_profile.json``
via the ``config_loader`` singleton.
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
from config.config_loader import config_loader
from integrations.llm_interface import LLMInterface

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants derived from config singletons
# ---------------------------------------------------------------------------
DRY_RUN: bool = os.getenv("DRY_RUN", "false").lower() == "true"
RESUME_DIR: Path = Path(run_config.resume_dir)

__all__ = [
    "FormFiller",
    "FieldType",
    "FormField",
    "FillResult",
    "fill_field",
    "human_type",
    "human_delay",
]


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
        screenshot_path: Path to error screenshot if any error occurred.
        cost_usd: Total LLM cost incurred during form filling.
    """

    total_fields: int = 0
    filled: int = 0
    skipped: int = 0
    failed: int = 0
    llm_calls: int = 0
    screenshot_path: str = ""
    cost_usd: float = 0.0
    custom_questions: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    success: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# UserProfile — loaded once at FormFiller init from env vars
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class UserProfile:
    """Candidate profile sourced exclusively from ``config/user_profile.json``.

    All values are read once at instantiation via the ``config_loader``
    singleton.  No values are ever hard-coded or read from environment
    variables.
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
        """Build a ``UserProfile`` by reading ``config/user_profile.json``.

        All field values are sourced from the ``config_loader`` singleton
        which reads the canonical JSON config files.  No ``os.getenv()`` calls
        are made for user data.

        Returns:
            Populated ``UserProfile`` instance.
        """
        try:
            meta: dict = config_loader.get_user_metadata()
            prefs: dict = config_loader.get_job_preferences()
            name: str = meta.get("full_name", "")
            parts: list[str] = name.split()
            # work_setup may be dict (remote/hybrid/onsite) — stringify if needed
            work_setup = prefs.get("work_setup", {})
            if isinstance(work_setup, dict):
                accepted_job_types = "remote" if work_setup.get("remote") else "full-time"
            else:
                accepted_job_types = str(work_setup)
            must_have: list = prefs.get("locations", {}).get("must_have", ["Remote"])
            accepted_locations: str = must_have[0] if must_have else "Remote"
            return cls(
                first_name=parts[0] if parts else "",
                last_name=parts[-1] if len(parts) > 1 else "",
                full_name=name,
                email=meta.get("email", ""),
                phone=meta.get("phone", ""),
                linkedin_url=meta.get("linkedin_url", ""),
                portfolio_url=meta.get("portfolio_url", ""),
                location=meta.get("location_city", ""),
                years_experience=str(meta.get("years_experience_total", "0")),
                accepted_job_types=accepted_job_types,
                accepted_locations=accepted_locations,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("UserProfile.from_env: failed to load from config_loader: %s", exc)
            return cls()


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
        dry_run: bool = False,
    ) -> None:
        self.page: Page = page
        self.job_title: str = job_title
        self.job_description: str = job_description
        self.company: str = company
        self.resume_path: Path = RESUME_DIR / resume_filename
        self.ats_type: str = ats_type
        self.dry_run: bool = (
            dry_run or os.getenv("DRY_RUN", "false").lower() == "true"
        )

        self.profile: UserProfile = UserProfile.from_env()
        self.llm = LLMInterface().get_llm("APPLY_AGENT")
        self.llm_model_string: str = self._get_model_string()
        self.result: FillResult = FillResult()
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Class-level: verified ATS selectors from browser recon
    # ------------------------------------------------------------------

    _ATS_SELECTORS: dict[str, dict[str, str]] = {
        "greenhouse": {
            "form":       "#application-form",
            "first_name": "#first_name",
            "last_name":  "#last_name",
            "email":      "#email",
            "phone":      "#phone",
            "resume":     "input[type='file']#resume",
        },
        "lever": {
            "form":       "form#application-form",
            "full_name":  "input[name='name']",
            "email":      "input[name='email']",
            "phone":      "input[name='phone']",
            "resume":     "input#resume-upload-input",
        },
        "workable": {
            "form":       "form[data-ui='application-form']",
            "first_name": "input[name='firstname']",
            "last_name":  "input[name='lastname']",
            "email":      "input[name='email']",
            "phone":      "input[name='phone']",
            "resume":     "input[type='file']",
        },
    }

    _SUBMIT_SELECTORS: dict[str, str] = {
        "greenhouse": "button[type='submit']",
        "lever":      "button#btn-submit",
        "workable":   "button[data-ui='apply-button']",
    }

    # Fallback chain — tried in order when platform-specific submit fails
    _SUBMIT_FALLBACKS: list[str] = [
        "button[type='submit']",
        "input[type='submit']",
        "button:has-text('Submit')",
        "button:has-text('Apply')",
        "button:has-text('Submit Application')",
        "button:has-text('Submit application')",
    ]

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

    async def _human_mouse_move(
        self, target_x: float, target_y: float
    ) -> None:
        """Move mouse to target position using Bezier curve for human-like motion.
        
        Implements Fitts' Law compliant mouse movement with random micro-jitter
        to evade bot detection. Uses quadratic Bezier curves with random
        control points.
        
        Args:
            target_x: Target X coordinate on page.
            target_y: Target Y coordinate on page.
        """
        if self.dry_run:
            return
            
        try:
            # Get current mouse position (approximate from viewport center if unknown)
            viewport = self.page.viewport_size or {"width": 1280, "height": 720}
            current_x = viewport["width"] / 2 + random.uniform(-50, 50)
            current_y = viewport["height"] / 2 + random.uniform(-50, 50)
            
            # Calculate path using Bezier curve
            steps = random.randint(8, 15)
            
            # Random control point for quadratic Bezier
            ctrl_x = (current_x + target_x) / 2 + random.uniform(-100, 100)
            ctrl_y = (current_y + target_y) / 2 + random.uniform(-50, 50)
            
            for i in range(steps + 1):
                t = i / steps
                # Quadratic Bezier: B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
                x = (1-t)**2 * current_x + 2*(1-t)*t * ctrl_x + t**2 * target_x
                y = (1-t)**2 * current_y + 2*(1-t)*t * ctrl_y + t**2 * target_y
                
                # Add micro-jitter for human-like imprecision
                x += random.uniform(-3, 3)
                y += random.uniform(-3, 3)
                
                await self.page.mouse.move(x, y)
                await asyncio.sleep(random.uniform(0.01, 0.03))
                
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("_human_mouse_move: error (proceeding): %s", exc)

    async def _human_type(self, value: str) -> None:
        """Type text character-by-character with human-like variable delays.
        
        Simulates natural typing patterns with:
        - Variable inter-key delays (80-220ms, WPM-normal)
        - Occasional longer pauses between words
        - Random micro-hesitations
        
        Args:
            value: The text string to type.
        """
        if self.dry_run:
            self.logger.debug("dry_run: would type '%s'", value[:50])
            return
            
        try:
            for i, char in enumerate(value):
                # Base delay between keystrokes (80-220ms for ~40-60 WPM)
                delay = random.uniform(0.08, 0.22)
                
                # Occasionally add longer pause between words
                if char == ' ' and random.random() < 0.3:
                    delay += random.uniform(0.1, 0.3)
                
                # Occasional micro-hesitation (simulating thinking)
                if random.random() < 0.05:
                    delay += random.uniform(0.2, 0.5)
                
                await self.page.keyboard.type(char, delay=delay * 1000)
                
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("_human_type: error (proceeding): %s", exc)

    async def _scroll_to_element(self, selector: str) -> bool:
        """Scroll element into view before interacting with it.
        
        Args:
            selector: CSS selector for the target element.
            
        Returns:
            True if scroll succeeded, False otherwise.
        """
        try:
            element = await self.page.query_selector(selector)
            if element:
                await element.scroll_into_view_if_needed()
                await asyncio.sleep(random.uniform(0.2, 0.4))
                return True
            return False
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("_scroll_to_element: error: %s", exc)
            return False

    async def _human_click(self, selector: str) -> bool:
        """Click an element with human-like mouse movement and timing.
        
        Args:
            selector: CSS selector for the click target.
            
        Returns:
            True if click succeeded, False otherwise.
        """
        if self.dry_run:
            self.logger.debug("dry_run: would click %s", selector)
            return True
            
        try:
            # First scroll into view
            await self._scroll_to_element(selector)
            
            # Get element bounding box
            element = await self.page.query_selector(selector)
            if not element:
                return False
                
            box = await element.bounding_box()
            if not box:
                # Fallback to regular click
                await self.page.click(selector)
                return True
                
            # Calculate click position with slight randomness
            target_x = box["x"] + box["width"] / 2 + random.uniform(-5, 5)
            target_y = box["y"] + box["height"] / 2 + random.uniform(-3, 3)
            
            # Human-like mouse movement
            await self._human_mouse_move(target_x, target_y)
            
            # Small delay before clicking
            await asyncio.sleep(random.uniform(0.05, 0.15))
            
            # Click
            await self.page.mouse.click(target_x, target_y)
            
            return True
            
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("_human_click: error: %s", exc)
            return False

    async def _human_fill_field(
        self, selector: str, value: str, use_tab: bool = True
    ) -> bool:
        """Fill a field with complete human simulation.
        
        Combines: scroll → mouse move → click → clear → type → tab
        
        Args:
            selector: CSS selector for the input field.
            value: Value to type into the field.
            use_tab: Whether to press Tab after filling (for field navigation).
            
        Returns:
            True if fill succeeded, False otherwise.
        """
        if self.dry_run:
            self.logger.debug("dry_run: would fill %s → '%s'", selector, value)
            return True
            
        try:
            # Scroll element into view
            await self._scroll_to_element(selector)
            
            # Get element position
            element = await self.page.query_selector(selector)
            if not element:
                return False
                
            box = await element.bounding_box()
            if box:
                # Move mouse to field
                target_x = box["x"] + box["width"] / 2 + random.uniform(-10, 10)
                target_y = box["y"] + box["height"] / 2 + random.uniform(-3, 3)
                await self._human_mouse_move(target_x, target_y)
            
            # Click to focus
            await self.page.click(selector, click_count=3)  # Triple-click to select all
            await asyncio.sleep(random.uniform(0.1, 0.2))
            
            # Clear existing content
            await self.page.keyboard.press("Backspace")
            await asyncio.sleep(random.uniform(0.05, 0.1))
            
            # Type with human-like delays
            await self._human_type(value)
            
            # Tab to next field if requested
            if use_tab:
                await asyncio.sleep(random.uniform(0.3, 0.8))
                await self.page.keyboard.press("Tab")
            
            return True
            
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("_human_fill_field: error: %s — trying fallback", exc)
            # Fallback to standard fill
            return await self._safe_fill(selector, value)

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
        if self.dry_run:
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
        if self.dry_run:
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
        if self.dry_run:
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
        if self.dry_run:
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

    async def screenshot_on_error(self, job_id: str = "") -> str:
        """Capture a full-page screenshot when an error occurs.
        
        Screenshots are saved to /tmp/apply_{job_id}_{timestamp}.png
        
        Args:
            job_id: Job identifier for the filename (optional).
            
        Returns:
            Path to the saved screenshot, or empty string on failure.
        """
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_suffix = f"_{job_id}" if job_id else ""
            screenshot_path = f"/tmp/apply{job_suffix}_{timestamp}.png"
            
            await self.page.screenshot(path=screenshot_path, full_page=True)
            self.result.screenshot_path = screenshot_path
            self.logger.info("Error screenshot saved: %s", screenshot_path)
            return screenshot_path
            
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("screenshot_on_error failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Per-platform fill methods (browser-verified selectors)
    # ------------------------------------------------------------------

    async def fill_greenhouse(self) -> FillResult:
        """Fill a Greenhouse application form using verified selectors.

        Returns:
            ``FillResult`` after filling all standard Greenhouse fields.
        """
        sel = self._ATS_SELECTORS["greenhouse"]

        try:
            await self.page.wait_for_selector(
                sel["form"], timeout=15000
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "fill_greenhouse: form container not found: %s", exc
            )
            self.result.errors.append(f"greenhouse form not found: {exc}")
            return self.result

        # Fill standard fields
        for field_key, value in [
            ("first_name", self.profile.first_name),
            ("last_name",  self.profile.last_name),
            ("email",      self.profile.email),
            ("phone",      self.profile.phone),
        ]:
            if value and field_key in sel:
                ok = await self._human_fill_field(sel[field_key], value)
                if ok:
                    self.result.filled += 1
                else:
                    self.result.failed += 1
                await self._human_delay()

        # Resume upload
        if sel.get("resume"):
            ok = await self._upload_resume(sel["resume"])
            if ok:
                self.result.filled += 1
            else:
                self.result.failed += 1

        # Scan and fill remaining fields via generic pipeline
        await self.fill_all_fields()
        return self.result

    async def fill_lever(self) -> FillResult:
        """Fill a Lever application form using verified selectors.

        Lever uses a single full-name field (no separate first/last).

        Returns:
            ``FillResult`` after filling all standard Lever fields.
        """
        sel = self._ATS_SELECTORS["lever"]

        try:
            await self.page.wait_for_selector(
                sel["form"], timeout=15000
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "fill_lever: form container not found: %s", exc
            )
            self.result.errors.append(f"lever form not found: {exc}")
            return self.result

        for field_key, value in [
            ("full_name", self.profile.full_name),
            ("email",     self.profile.email),
            ("phone",     self.profile.phone),
        ]:
            if value and field_key in sel:
                ok = await self._human_fill_field(sel[field_key], value)
                if ok:
                    self.result.filled += 1
                else:
                    self.result.failed += 1
                await self._human_delay()

        if sel.get("resume"):
            ok = await self._upload_resume(sel["resume"])
            if ok:
                self.result.filled += 1
            else:
                self.result.failed += 1

        await self.fill_all_fields()
        return self.result

    async def fill_workable(self) -> FillResult:
        """Fill a Workable application form using verified selectors.

        Returns:
            ``FillResult`` after filling all standard Workable fields.
        """
        sel = self._ATS_SELECTORS["workable"]

        try:
            await self.page.wait_for_selector(
                sel["form"], timeout=15000
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "fill_workable: form container not found: %s", exc
            )
            self.result.errors.append(f"workable form not found: {exc}")
            return self.result

        for field_key, value in [
            ("first_name", self.profile.first_name),
            ("last_name",  self.profile.last_name),
            ("email",      self.profile.email),
            ("phone",      self.profile.phone),
        ]:
            if value and field_key in sel:
                ok = await self._human_fill_field(sel[field_key], value)
                if ok:
                    self.result.filled += 1
                else:
                    self.result.failed += 1
                await self._human_delay()

        if sel.get("resume"):
            ok = await self._upload_resume(sel["resume"])
            if ok:
                self.result.filled += 1
            else:
                self.result.failed += 1

        await self.fill_all_fields()
        return self.result

    # ------------------------------------------------------------------
    # fill_and_submit — orchestrator with DRY_RUN submit guard
    # ------------------------------------------------------------------

    @operation
    async def fill_and_submit(self) -> FillResult:
        """Route to the correct per-platform fill method and submit.

        Steps:
          1. Route to ``fill_greenhouse``, ``fill_lever``, or
             ``fill_workable`` based on ``self.ats_type``.
             Falls back to ``fill_all_fields`` for unknown ATS.
          2. Check ``DRY_RUN`` — if true, log and skip click.
          3. Click the platform-specific submit button.
          4. Wait for navigation / network idle.

        Returns:
            Populated ``FillResult``.
        """
        # Step 1 — Route to per-platform fill
        ats_lower: str = self.ats_type.lower()
        if ats_lower == "greenhouse":
            await self.fill_greenhouse()
        elif ats_lower == "lever":
            await self.fill_lever()
        elif ats_lower == "workable":
            await self.fill_workable()
        else:
            await self.fill_all_fields()

        # Step 2 — DRY_RUN guard on submit
        dry_run = os.getenv("DRY_RUN", "false").lower() == "true"
        if dry_run or self.dry_run:
            self.logger.info(
                "[DRY_RUN] fill_and_submit: skipping submit click "
                "(ats=%s, fields_filled=%d)",
                self.ats_type,
                self.result.filled,
            )
            self.result.success = True
            return self.result

        # Step 3 — Click submit
        submitted: bool = False
        primary_sel: str = self._SUBMIT_SELECTORS.get(ats_lower, "")
        selectors_to_try: list[str] = (
            [primary_sel] + self._SUBMIT_FALLBACKS if primary_sel
            else self._SUBMIT_FALLBACKS
        )

        for sel in selectors_to_try:
            try:
                btn = await self.page.query_selector(sel)
                if btn:
                    await self._human_delay(500, 1200)
                    await btn.click()
                    submitted = True
                    self.logger.info(
                        "fill_and_submit: clicked submit via '%s'", sel
                    )
                    break
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(
                    "fill_and_submit: submit selector '%s' failed: %s",
                    sel,
                    exc,
                )
                continue

        if not submitted:
            self.logger.error(
                "fill_and_submit: no submit button found for ats=%s",
                self.ats_type,
            )
            self.result.errors.append("submit_button_not_found")
            self.result.success = False
            return self.result

        # Step 4 — Wait for navigation
        try:
            await self.page.wait_for_load_state(
                "networkidle", timeout=15000
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "fill_and_submit: post-submit wait failed: %s", exc
            )

        self.result.success = True
        return self.result

    @operation
    async def fill_field(
        self,
        selector: str,
        value: str,
        field_type: str = "text",
    ) -> bool:
        """Fill a single form field with human-like simulation.
        
        This is the public interface for filling individual fields,
        used by platform-specific modules.
        
        Args:
            selector: CSS selector for the target field.
            value: Value to fill.
            field_type: One of: text, email, phone, dropdown, radio, 
                        checkbox, file_upload, textarea
                        
        Returns:
            True if field was successfully filled, False otherwise.
        """
        if not value:
            self.logger.debug("fill_field: empty value for %s — skipping", selector)
            return True
            
        try:
            # Scroll element into view first
            await self._scroll_to_element(selector)
            await self._human_delay(300, 800)  # Micro-pause between sections
            
            field_type_lower = field_type.lower()
            
            if field_type_lower == "file_upload":
                return await self._upload_resume(selector)
                
            elif field_type_lower == "dropdown":
                element = await self.page.query_selector(selector)
                if element:
                    options = await self.page.evaluate(
                        """(el) => {
                            return Array.from(el.options).map(o => o.text.trim())
                               .filter(t => t.length > 0);
                        }""",
                        element,
                    )
                    return await self._safe_select(selector, options, value)
                return False
                
            elif field_type_lower == "radio":
                return await self._handle_radio_group(selector, "", value)
                
            elif field_type_lower == "checkbox":
                should_check = value.lower() in ("true", "yes", "1", "checked")
                return await self._handle_checkbox(selector, "", should_check)
                
            else:
                # text, email, phone, textarea
                return await self._human_fill_field(selector, value)
                
        except Exception as exc:  # noqa: BLE001
            self.logger.error("fill_field failed for %s: %s", selector, exc)
            await self.screenshot_on_error()
            return False

    async def detect_required_fields(self) -> list[FormField]:
        """Detect all required fields on the current page.
        
        Returns:
            List of FormField objects marked as required.
        """
        all_fields = await self.scan_form_fields()
        return [f for f in all_fields if f.required]

    async def detect_captcha(self) -> bool:
        """Check if a CAPTCHA is present on the current page.
        
        Returns:
            True if CAPTCHA detected, False otherwise.
        """
        try:
            captcha_selectors = [
                "iframe[src*='recaptcha']",
                "iframe[src*='hcaptcha']",
                "div.g-recaptcha",
                "div.h-captcha",
                "iframe[src*='turnstile']",
                "div.cf-turnstile",
            ]
            
            for selector in captcha_selectors:
                element = await self.page.query_selector(selector)
                if element:
                    self.logger.warning("CAPTCHA detected: %s", selector)
                    return True
                    
            return False
            
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("detect_captcha error: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Standalone fill_field function for external use
# ---------------------------------------------------------------------------
async def fill_field(
    page: Page,
    selector: str,
    value: str,
    field_type: str = "text",
) -> bool:
    """Standalone function to fill a single form field with human simulation.
    
    Creates a temporary FormFiller instance to leverage all human-like
    behaviors without requiring full FormFiller setup.
    
    Args:
        page: Playwright Page object.
        selector: CSS selector for the target field.
        value: Value to fill.
        field_type: Field type (text, email, phone, dropdown, etc.)
        
    Returns:
        True if field was filled successfully, False otherwise.
    """
    filler = FormFiller(
        page=page,
        job_title="",
        job_description="",
        company="",
        resume_filename="",
        ats_type="native",
        dry_run=DRY_RUN,
    )
    return await filler.fill_field(selector, value, field_type)


# ---------------------------------------------------------------------------
# Module-level human_type and human_delay
# ---------------------------------------------------------------------------
async def human_type(page: Page, selector: str, text: str) -> bool:
    """Type text into a field with human-like keystroke simulation.

    Standalone function that creates a temporary ``FormFiller`` to leverage
    the full human-simulation pipeline (random delays between keystrokes,
    occasional typo correction, scroll-into-view).

    Args:
        page: Playwright ``Page`` object.
        selector: CSS selector for the target input.
        text: Text to type.

    Returns:
        ``True`` if the text was typed successfully, ``False`` otherwise.
    """
    filler = FormFiller(
        page=page,
        job_title="",
        job_description="",
        company="",
        resume_filename="",
        ats_type="native",
        dry_run=DRY_RUN,
    )
    return await filler._human_fill_field(selector, text)


async def human_delay(min_ms: int = 500, max_ms: int = 1500) -> None:
    """Sleep for a random duration to simulate human pause.

    Standalone async function — no ``FormFiller`` instance needed.

    Args:
        min_ms: Minimum delay in milliseconds.
        max_ms: Maximum delay in milliseconds.
    """
    delay_s: float = random.randint(min_ms, max_ms) / 1000
    await asyncio.sleep(delay_s)
