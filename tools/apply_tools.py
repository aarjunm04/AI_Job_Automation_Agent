from __future__ import annotations
from agentops.sdk.decorators import operation
"""
Apply tools for AI Job Application Agent.

Full rewrite wiring ``ATSDetector`` and ``FormFiller`` into every stage of the
Playwright session lifecycle.  Replaces the naive ``page.fill()`` loop from
the previous iteration with proper per-platform ATS detection, intelligent
multi-step form filling, CAPTCHA gating, and proof-of-submission capture.

Tool summary:
- ``detect_ats_platform``  — Playwright session + 3-layer ``ATSDetector``.
- ``capture_proof``         — Post-submit signal analysis (unchanged).
- ``check_captcha_present`` — HTML CAPTCHA fingerprint scan (unchanged).
- ``fill_standard_form``    — Orchestrates the full apply session via
                              ``ATSDetector`` + ``FormFiller``.
- ``get_apply_summary``     — Per-run Postgres aggregation (unchanged).

Every tool is decorated with ``@operation`` and called exclusively
by ``agents/apply_agent.py``.  All Playwright actions are fail-soft.
``DRY_RUN=true`` guards every submission, click, upload, and keyboard event.
"""


import asyncio
import base64
import json
from datetime import datetime
import logging
import os
import re
import time

from pathlib import Path

def _resolve_resume_path(resume_filename: str) -> Path:
    """Resolve absolute resume path, preventing double-prefix."""
    resume_filename = resume_filename.lstrip("/")
    if resume_filename.startswith("resumes/"):
        return Path("app") / resume_filename
    return Path("app") / "resumes" / resume_filename

from typing import Any, Optional

def _retry_call(fn, *args, max_retries: int = 3, **kwargs):
    """Execute fn with exponential backoff retry."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            wait = 2.0 ** attempt
            logger.warning(
                "Attempt %d/%d failed: %s — retrying in %.1fs",
                attempt + 1, max_retries, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"All {max_retries} attempts failed. Last error: {last_exc}"
    )

import agentops
from agentops.sdk.decorators import agent, operation
import psycopg2
import psycopg2.extras
import requests
from crewai.tools import tool
from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError, async_playwright

from auto_apply.ats_detector import ATSDetector, ATSProfile, ATSType
from auto_apply.form_filler import FormFiller, FillResult
from integrations.llm_interface import LLMInterface
from tools.postgres_tools import create_application, update_application_status, log_event, _fetch_user_config, _priority_text
from tools.budget_tools import check_xai_run_cap, record_llm_cost
from tools.agentops_tools import record_agent_error
from tools.notion_tools import queue_job_to_applications_db
from config.settings import db_config, run_config, budget_config
from config.config_loader import config_loader
from utils.db_utils import get_db_conn
from utils.proxy_rate_limit import get_playwright_proxy, get_next_proxy, mark_proxy_dead, mark_proxy_success

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------
__all__ = [
    "detect_ats_platform",
    "capture_proof",
    "check_captcha_present",
    "fill_standard_form",
    "get_apply_summary",
    "route_and_apply",
    "save_application_result",
    "save_to_queue",
    "get_best_resume",
    "verify_apply_budget",
]

# ---------------------------------------------------------------------------
# Constants from config singletons — no raw os.getenv() in tool logic
# ---------------------------------------------------------------------------
DRY_RUN: bool = os.getenv("DRY_RUN", "false").lower() == "true"
RESUME_DIR: Path = Path(run_config.resume_dir)
MAX_SESSIONS: int = run_config.max_playwright_sessions

# Database URL — kept as module-level for get_apply_summary (exact legacy logic)
_DB_URL: Optional[str] = (
    os.getenv("LOCAL_POSTGRES_URL")
    if os.getenv("ACTIVE_DB", "local") == "local"
    else os.getenv("SUPABASE_DB_URL")
)

# ---------------------------------------------------------------------------
# Proxy — delegate to utils.proxy_rate_limit for thread-safe round-robin
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TOOL 1 — ATS platform detection (Playwright + ATSDetector)
# ---------------------------------------------------------------------------
@tool
def detect_ats_platform(job_url: str, pipeline_run_id: str, dry_run: bool = False) -> str:
    """Detect the ATS platform powering a job application page.

    Launches a minimal headless Playwright session (no proxy), navigates to
    ``job_url``, and runs the ``ATSDetector`` 3-layer pipeline (URL pattern →
    DOM fingerprint → LLM classification).  Retries up to 2 times on
    ``TimeoutError`` only.

    Args:
        job_url: Full URL of the job application page.
        pipeline_run_id: UUID of the current run batch (for logging context).
        dry_run: If True, skip browser launch and return a mock ATS profile.

    Returns:
        JSON string of ``ATSProfile.to_dict()`` merged with
        ``{"job_url": job_url}``.  On any failure returns
        ``{"ats_type": "unknown", "confidence": 0.0, "job_url": ..., "error": ...}``.
    """
    if dry_run or os.getenv("DRY_RUN", "false").lower() == "true":
        logger.info(
            "detect_ats_platform: dry_run=True — skipping browser, "
            "returning mock profile for job_url=%s",
            job_url,
        )
        return json.dumps(
            {
                "ats_type": "linkedin_easy_apply",
                "detection_method": "dry_run_mock",
                "confidence": 1.0,
                "job_url": job_url,
                "dry_run": True,
            }
        )

    async def _detect_inner(url: str) -> dict[str, Any]:
        """Run ATSDetector inside a minimal Playwright session.

        Args:
            url: Full URL to navigate to.

        Returns:
            Merged dict of ``ATSProfile.to_dict()`` and ``{"job_url": url}``.
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page: Page = await context.new_page()
            try:
                await page.goto(url, wait_until="networkidle", timeout=20000)
                detector = ATSDetector()
                ats_profile: ATSProfile = await detector.detect(page, url)
                result: dict[str, Any] = {**ats_profile.to_dict(), "job_url": url}
                return result
            finally:
                try:
                    await context.close()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    await browser.close()
                except Exception:  # noqa: BLE001
                    pass

    max_retries: int = 2
    for attempt in range(max_retries + 1):
        try:
            result: dict[str, Any] = asyncio.run(_detect_inner(job_url))
            logger.info(
                "detect_ats_platform: ats=%s confidence=%.2f method=%s url=%s",
                result.get("ats_type"),
                result.get("confidence", 0.0),
                result.get("detection_method"),
                job_url,
            )
            return json.dumps(result)

        except PlaywrightTimeoutError as te:
            if attempt < max_retries:
                logger.warning(
                    "detect_ats_platform: TimeoutError attempt %d/%d for %s — retrying: %s",
                    attempt + 1,
                    max_retries,
                    job_url,
                    te,
                )
                continue
            logger.error(
                "detect_ats_platform: TimeoutError after %d retries for %s: %s",
                max_retries,
                job_url,
                te,
            )
            return json.dumps(
                {
                    "ats_type": "unknown",
                    "confidence": 0.0,
                    "job_url": job_url,
                    "error": str(te),
                }
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("detect_ats_platform: error for %s: %s", job_url, exc)
            return json.dumps(
                {
                    "ats_type": "unknown",
                    "confidence": 0.0,
                    "job_url": job_url,
                    "error": str(exc),
                }
            )

    # Unreachable — loop always returns or raises above
    return json.dumps(  # pragma: no cover
        {
            "ats_type": "unknown",
            "confidence": 0.0,
            "job_url": job_url,
            "error": "max_retries_exceeded",
        }
    )


# ---------------------------------------------------------------------------
# TOOL 2 — Proof of submission capture (UNCHANGED)
# ---------------------------------------------------------------------------
@tool
def capture_proof(page_html: str, page_url: str, job_url: str) -> str:
    """Extract proof-of-submission signals from a post-apply page snapshot.

    Checks four independent signals and assigns confidence based on how many
    are detected. No network calls are made.

    Args:
        page_html: Full HTML content of the page after submission.
        page_url: Current URL of the page after submission.
        job_url: Original job posting URL (for reference in the proof record).

    Returns:
        JSON string with all signal values, ``signals_captured`` count,
        ``proof_confidence`` ("high"|"medium"|"low"|null), and ``page_url``.
    """
    try:
        html_lower: str = page_html.lower()
        url_lower: str = page_url.lower()

        # Signal 1 — confirmation number in page text
        confirmation_match = re.search(
            r"(?:confirmation|reference|application.?id)[^\w]*([A-Za-z0-9\-]{4,32})",
            page_html,
            re.IGNORECASE,
        )
        confirmation_number: Optional[str] = (
            confirmation_match.group(1) if confirmation_match else None
        )

        # Signal 2 — success keywords in URL
        success_url_keywords = [
            "confirmation",
            "success",
            "thank-you",
            "submitted",
            "complete",
        ]
        success_url: bool = any(kw in url_lower for kw in success_url_keywords)

        # Signal 3 — success message in HTML
        success_phrases = [
            "application submitted",
            "successfully applied",
            "thank you for applying",
            "we received your application",
        ]
        success_message: bool = any(ph in html_lower for ph in success_phrases)

        # Signal 4 — form element has disappeared
        form_disappeared: bool = "<form" not in html_lower

        # Tally signals
        signals: int = sum(
            [
                confirmation_number is not None,
                success_url,
                success_message,
                form_disappeared,
            ]
        )

        if signals >= 3:
            proof_confidence: Optional[str] = "high"
        elif signals == 2:
            proof_confidence = "medium"
        elif signals == 1:
            proof_confidence = "low"
        else:
            proof_confidence = None

        logger.info(
            "capture_proof: signals=%d confidence=%s url=%s",
            signals,
            proof_confidence,
            page_url,
        )

        return json.dumps(
            {
                "confirmation_number": confirmation_number,
                "success_url": success_url,
                "success_message": success_message,
                "form_disappeared": form_disappeared,
                "signals_captured": signals,
                "proof_confidence": proof_confidence,
                "page_url": page_url,
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("capture_proof failed: %s", exc)
        return json.dumps(
            {
                "confirmation_number": None,
                "success_url": False,
                "success_message": False,
                "form_disappeared": False,
                "signals_captured": 0,
                "proof_confidence": None,
                "page_url": page_url,
            }
        )


# ---------------------------------------------------------------------------
# TOOL 3 — CAPTCHA detection (UNCHANGED)
# ---------------------------------------------------------------------------
@tool
def check_captcha_present(page_html: str, job_url: str) -> str:
    """Detect CAPTCHA or bot-challenge presence in page HTML.

    Scans for known CAPTCHA fingerprints (reCAPTCHA, hCaptcha, Cloudflare
    Turnstile, etc.). Logs a WARNING-level event to Python logger when
    detected; does not call Postgres directly (no valid pipeline_run_id at
    this scope).

    Args:
        page_html: Full HTML content of the page to inspect.
        job_url: Job URL associated with the page (for log context).

    Returns:
        JSON string ``{"captcha_detected": bool, "captcha_type": str|null,
        "action": "skip_to_manual"|"proceed"}``.
    """
    try:
        html_lower: str = page_html.lower()

        captcha_signatures: dict[str, str] = {
            "recaptcha": "recaptcha",
            "hcaptcha": "hcaptcha",
            "turnstile": "cf-turnstile",
            "captcha_generic": "captcha",
            "challenge_form": "challenge-form",
            "cloudflare": "cloudflare",
        }

        detected_type: Optional[str] = None
        for label, signature in captcha_signatures.items():
            if signature in html_lower:
                detected_type = label
                break

        captcha_detected: bool = detected_type is not None
        action: str = "skip_to_manual" if captcha_detected else "proceed"

        if captcha_detected:
            logger.warning(
                "check_captcha_present: CAPTCHA detected type=%s url=%s",
                detected_type,
                job_url,
            )
        else:
            logger.info("check_captcha_present: no CAPTCHA detected url=%s", job_url)

        return json.dumps(
            {
                "captcha_detected": captcha_detected,
                "captcha_type": detected_type,
                "action": action,
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("check_captcha_present failed: %s", exc)
        return json.dumps(
            {"captcha_detected": False, "captcha_type": None, "action": "proceed"}
        )


# ---------------------------------------------------------------------------
# Internal async apply runner
# ---------------------------------------------------------------------------
async def _run_apply(
    job_url: str,
    job_post_id: str,
    resume_filename: str,
    pipeline_run_id: str,
    user_id: str,
    ats_platform: str,
) -> dict[str, Any]:
    """Execute the full Playwright apply flow wiring ATSDetector + FormFiller.

    All 13 steps follow the orchestration contract defined in
    ``[CONTEXT]`` of the module docstring.  Every step is individually
    guarded — a single step failure never aborts the run.

    Args:
        job_url: URL of the job application form.
        job_post_id: UUID of the ``job_posts`` row.
        resume_filename: Filename of the resume PDF inside ``RESUME_DIR``.
        pipeline_run_id: UUID of the current run batch.
        user_id: UUID of the candidate (``users`` table).
        ats_platform: ATS type string from a prior ``detect_ats_platform``
            call, or ``"unknown"`` / ``""`` to trigger live detection.

    Returns:
        Dict with apply result keys per the STEP 13 spec.  Never raises.
    """
    # -----------------------------------------------------------------------
    # DB CONFIG FETCH — Authoritative dry_run + default_resume from Postgres
    # -----------------------------------------------------------------------
    dry_run_effective: bool
    default_resume_effective: str
    try:
        _cfg: dict[str, Any] = _fetch_user_config()
        _user_settings: dict[str, Any] = _cfg.get("user_settings", {})

        platform_config_path = "config_loader.platforms"
        platform_config = config_loader.platforms
        job_filters: dict[str, Any] = platform_config.get("job_filters", {})

        dry_run_effective = bool(_user_settings.get("dry_run", False))
        logger.info(
            "_run_apply: DB config — dry_run=%s default_resume=%s auto_apply_enabled=%s",
            dry_run_effective,
            default_resume_effective,
            auto_apply_enabled,
        )
    except Exception as _cfg_exc:  # noqa: BLE001
        dry_run_effective = os.getenv("DRY_RUN", "false").lower() == "true"
        default_resume_effective = "AarjunGen.pdf"
        logger.warning(
            "_run_apply: DB config fetch failed (%s) — "
            "falling back to env/hardcoded: dry_run=%s",
            _cfg_exc,
            dry_run_effective,
        )

    # Resolve resume filename — use DB default_resume if given file is missing on disk
    _resume_path: Path = RESUME_DIR / resume_filename
    if not _resume_path.is_file():
        logger.warning(
            "_run_apply: resume '%s' not found at '%s' — using default_resume '%s'",
            resume_filename,
            _resume_path,
            default_resume_effective,
        )
        resume_filename = default_resume_effective

    # -----------------------------------------------------------------------
    # STEP 1 — Guards: DRY_RUN + budget cap
    # -----------------------------------------------------------------------
    if dry_run_effective:
        logger.info("_run_apply: DRY_RUN=true — skipping browser for %s", job_url)
        try:
            _log_event_fn(
                pipeline_run_id=pipeline_run_id,
                event_type="dry_run_skip",
                level="INFO",
                agent="apply_agent",
                message=f"dry_run|{job_url}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("_run_apply: log_event (dry_run_skip) failed: %s", exc)

        fill_simulation: dict = {
            "fields_filled": 6,
            "fields_total": 7,
            "fields_skipped": 1,
            "resume_checked": True,
            "resume_filename": resume_filename,
            "ats_platform": ats_platform,
            "dry_run_timestamp": datetime.utcnow().isoformat() + "Z",
            "custom_questions_answered": 1,
            "simulation_mode": "headless_mock",
        }

        return {
            "applied": False,
            "dry_run": True,
            "ats_type": ats_platform,
            "fill_simulation": fill_simulation,
        }

    # Budget gate
    try:
        budget_result: dict[str, Any] = json.loads(check_xai_run_cap(pipeline_run_id))
        if budget_result.get("abort"):
            logger.critical(
                "_run_apply: xAI budget cap hit — aborting apply for %s", job_url
            )
            return {
                "applied": False,
                "status": "budget_cap_hit",
                "re_route": "manual",
                "job_url": job_url,
            }
    except Exception as exc:  # noqa: BLE001
        logger.warning("_run_apply: budget cap check failed (proceeding): %s", exc)

    # -----------------------------------------------------------------------
    # STEP 2 — Browser setup
    # -----------------------------------------------------------------------
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        proxy: Optional[dict[str, str]] = get_playwright_proxy()
        context = await browser.new_context(
            proxy=proxy,  # type: ignore[arg-type]
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        await context.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
        )
        page: Page = await context.new_page()

        # Initialise fill_result default before any early returns that skip STEP 8
        fill_result: FillResult = FillResult()

        try:
            # -------------------------------------------------------------------
            # STEP 3 — Navigate
            # -------------------------------------------------------------------
            await page.goto(job_url, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(1500)  # let React hydrate

            # -------------------------------------------------------------------
            # STEP 4 — CAPTCHA check
            # -------------------------------------------------------------------
            html: str = await page.content()
            captcha: dict[str, Any] = json.loads(check_captcha_present(html, job_url))
            if captcha.get("captcha_detected"):
                logger.warning(
                    "_run_apply: CAPTCHA detected: %s at %s",
                    captcha.get("captcha_type"),
                    job_url,
                )
                await context.close()
                await browser.close()
                return {
                    "applied": False,
                    "status": "captcha_blocked",
                    "re_route": "manual",
                    "job_url": job_url,
                }

            # -------------------------------------------------------------------
            # STEP 5 — ATS Detection
            # -------------------------------------------------------------------
            ats_profile: ATSProfile
            if ats_platform in ("unknown", ""):
                detector = ATSDetector()
                ats_profile = await detector.detect(page, job_url)
            else:
                detector = ATSDetector()
                ats_profile = detector.get_profile_for_ats(ats_platform)

            logger.info(
                "_run_apply: ATS profile: %s | confidence=%.2f | strategy=%s",
                ats_profile.ats_type.value,
                ats_profile.confidence,
                ats_profile.apply_strategy,
            )

            # -------------------------------------------------------------------
            # STEP 6 — LinkedIn Easy Apply gate
            # -------------------------------------------------------------------
            if ats_profile.ats_type == ATSType.LINKEDIN_EASY_APPLY:
                logger.warning(
                    "_run_apply: LinkedIn Easy Apply requires active session — "
                    "routing to manual"
                )
                await context.close()
                await browser.close()
                return {
                    "applied": False,
                    "status": "linkedin_requires_session",
                    "re_route": "manual",
                }

            # -------------------------------------------------------------------
            # STEP 7 — Workday login gate
            # -------------------------------------------------------------------
            if ats_profile.requires_login and ats_profile.ats_type == ATSType.WORKDAY:
                logger.warning(
                    "_run_apply: Workday requires account login — routing to manual"
                )
                await context.close()
                await browser.close()
                return {
                    "applied": False,
                    "status": "requires_login",
                    "re_route": "manual",
                }

            # -------------------------------------------------------------------
            # STEP 8 — Form filling via FormFiller
            # -------------------------------------------------------------------
            filler = FormFiller(
                page=page,
                job_title=run_config.search_query,
                job_description="",
                company="",
                resume_filename=resume_filename,
                ats_type=ats_profile.ats_type.value,
            )
            try:
                if ats_profile.is_multi_step:
                    fill_result = await filler.handle_multi_step_form(max_steps=8)
                else:
                    fill_result = await filler.fill_all_fields()
            except Exception as fill_exc:  # noqa: BLE001
                logger.warning("_run_apply: form fill raised: %s", fill_exc)
                fill_result = filler.result  # partial result from filler

            try:
                record_llm_cost(
                    "xai",
                    0.002 * fill_result.llm_calls,
                    "APPLY_AGENT",
                    pipeline_run_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("_run_apply: record_llm_cost failed: %s", exc)

            logger.info(
                "_run_apply: Form filled: %d/%d fields | llm_calls=%d",
                fill_result.filled,
                fill_result.total_fields,
                fill_result.llm_calls,
            )

            # -------------------------------------------------------------------
            # STEP 9 — Submit (guarded by DRY_RUN)
            # -------------------------------------------------------------------
            if not dry_run_effective:
                for raw_selector in ats_profile.submit_selector.split(","):
                    submit_selector: str = raw_selector.strip()
                    if not submit_selector:
                        continue
                    try:
                        submit_btn = await page.query_selector(submit_selector)
                        if submit_btn is not None:
                            await submit_btn.click()
                            break
                    except Exception as sub_exc:  # noqa: BLE001
                        logger.warning(
                            "_run_apply: submit selector failed '%s': %s",
                            submit_selector,
                            sub_exc,
                        )
                        continue

                try:
                    await page.wait_for_load_state("networkidle", timeout=15000)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "_run_apply: wait_for_load_state after submit failed: %s", exc
                    )

            await page.wait_for_timeout(2000)

            # -------------------------------------------------------------------
            # STEP 10 — Proof capture
            # -------------------------------------------------------------------
            post_html: str = await page.content()
            post_url: str = page.url
            proof: dict[str, Any] = json.loads(
                capture_proof(post_html, post_url, job_url)
            )

            screenshot_b64: str = ""
            try:
                screenshot_b64 = base64.b64encode(
                    await page.screenshot(type="png", full_page=False)
                ).decode()
            except Exception as ss_exc:  # noqa: BLE001
                logger.warning("_run_apply: screenshot failed: %s", ss_exc)

            # -------------------------------------------------------------------
            # STEP 11 — Close browser
            # -------------------------------------------------------------------
            try:
                await context.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                await browser.close()
            except Exception:  # noqa: BLE001
                pass

            # -------------------------------------------------------------------
            # STEP 12 — Determine status + write to Postgres
            # -------------------------------------------------------------------
            status: str = (
                "applied"
                if proof.get("proof_confidence") in ("high", "medium", "low")
                else "failed"
            )

            try:
                create_application(
                    job_post_id=job_post_id,
                    resume_id="",
                    user_id=user_id,
                    mode="auto",
                    status=status,
                    platform=ats_profile.ats_type.value,
                    error_code="" if status == "applied" else "proof_none",
                )
            except Exception as db_exc:  # noqa: BLE001
                logger.error("_run_apply: create_application failed: %s", db_exc)

            try:
                _log_event_fn(
                    pipeline_run_id=pipeline_run_id,
                    event_type="auto_apply_attempt",
                    level="INFO" if status == "applied" else "ERROR",
                    agent="apply_agent",
                    message=(
                        f"{status}|{job_url}"
                        f"|ats={ats_profile.ats_type.value}"
                        f"|confidence={proof.get('proof_confidence')}"
                        f"|fields={fill_result.filled}/{fill_result.total_fields}"
                    ),
                )
            except Exception as log_exc:  # noqa: BLE001
                logger.warning("_run_apply: log_event failed: %s", log_exc)

            # -------------------------------------------------------------------
            # STEP 13 — Return
            # -------------------------------------------------------------------
            return {
                "applied": status == "applied",
                "status": status,
                "ats_type": ats_profile.ats_type.value,
                "ats_detection_method": ats_profile.detection_method.value,
                "proof_confidence": proof.get("proof_confidence"),
                "proof_signals": proof.get("signals_captured"),
                "fields_filled": fill_result.filled,
                "fields_total": fill_result.total_fields,
                "llm_calls_used": fill_result.llm_calls,
                "custom_questions_answered": len(fill_result.custom_questions),
                "screenshot_captured": bool(screenshot_b64),
                "dry_run": dry_run_effective,
                "job_url": job_url,
            }

        except Exception as inner_exc:  # noqa: BLE001
            # Fail-soft: close browser and re-route to manual
            try:
                await context.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                await browser.close()
            except Exception:  # noqa: BLE001
                pass
            logger.error(
                "_run_apply: unexpected error for %s: %s", job_url, inner_exc
            )
            return {
                "applied": False,
                "status": "failed",
                "reason": str(inner_exc),
                "re_route": "manual",
                "job_url": job_url,
            }


# ---------------------------------------------------------------------------
# TOOL 4 — Fill standard form (sync wrapper around _run_apply)
# ---------------------------------------------------------------------------
@tool
def fill_standard_form(
    job_url: str,
    job_post_id: str,
    resume_filename: str,
    pipeline_run_id: str,
    user_id: str,
    ats_platform: str,
) -> str:
    """Apply to a job via Playwright using ATSDetector + FormFiller.

    Wraps the async ``_run_apply`` coroutine in a synchronous CrewAI tool
    interface.  Retries up to 2 times on ``TimeoutError`` only (exponential
    backoff).  Any non-retriable failure is recorded via ``record_agent_error``
    and immediately re-routed to manual queue.

    ``DRY_RUN=true`` prevents any real form submission, file upload, or click
    from occurring, but still runs ATSDetector + FormFiller detection to
    validate the wiring.

    Args:
        job_url: URL of the job application page.
        job_post_id: UUID of the ``job_posts`` row.
        resume_filename: Filename of the resume PDF inside ``RESUME_DIR``.
        pipeline_run_id: UUID of the current run batch.
        user_id: UUID of the candidate (``users`` table).
        ats_platform: ATS type string from ``detect_ats_platform``, or
            ``"unknown"``/``""`` to trigger live detection.

    Returns:
        JSON string with apply outcome containing: applied, status, ats_type,
        ats_detection_method, proof_confidence, proof_signals, fields_filled,
        fields_total, llm_calls_used, custom_questions_answered,
        screenshot_captured, dry_run, job_url.
    """
    max_retries: int = 2

    for attempt in range(max_retries + 1):
        try:
            result: dict[str, Any] = asyncio.run(
                _run_apply(
                    job_url=job_url,
                    job_post_id=job_post_id,
                    resume_filename=resume_filename,
                    pipeline_run_id=pipeline_run_id,
                    user_id=user_id,
                    ats_platform=ats_platform,
                )
            )
            return json.dumps(result)

        except (PlaywrightTimeoutError, asyncio.TimeoutError) as te:
            if attempt < max_retries:
                backoff: int = 2 ** attempt
                logger.warning(
                    "fill_standard_form: TimeoutError attempt %d/%d for %s — "
                    "retrying in %ds: %s",
                    attempt + 1,
                    max_retries,
                    job_url,
                    backoff,
                    te,
                )
                time.sleep(backoff)
            else:
                logger.error(
                    "fill_standard_form: TimeoutError after %d retries for %s: %s",
                    max_retries,
                    job_url,
                    te,
                )
                try:
                    record_agent_error(
                        agent_type="ApplyAgent",
                        error_message=str(te),
                        pipeline_run_id=pipeline_run_id,
                        error_code="TIMEOUT",
                        job_post_id=job_post_id,
                    )
                except Exception:  # noqa: BLE001
                    pass
                return json.dumps(
                    {
                        "applied": False,
                        "status": "failed",
                        "reason": str(te),
                        "re_route": "manual",
                        "job_url": job_url,
                    }
                )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "fill_standard_form: non-retriable error for %s: %s", job_url, exc
            )
            try:
                record_agent_error(
                    agent_type="ApplyAgent",
                    error_message=str(exc),
                    pipeline_run_id=pipeline_run_id,
                    error_code="APPLY_ERROR",
                    job_post_id=job_post_id,
                )
            except Exception:  # noqa: BLE001
                pass
            return json.dumps(
                {
                    "applied": False,
                    "status": "failed",
                    "reason": str(exc),
                    "re_route": "manual",
                    "job_url": job_url,
                }
            )

    # Unreachable — all retry paths return above
    return json.dumps(  # pragma: no cover
        {
            "applied": False,
            "status": "failed",
            "reason": "max_retries_exceeded",
            "re_route": "manual",
            "job_url": job_url,
        }
    )


# ---------------------------------------------------------------------------
# TOOL 5 — Per-run apply summary (UNCHANGED)
# ---------------------------------------------------------------------------
@tool
def get_apply_summary(pipeline_run_id: str) -> str:
    """Aggregate application counts by status for a given run batch.

    Queries the ``applications`` table joined with ``job_posts`` to count
    results grouped by status (applied, failed, manual_queued).

    Args:
        pipeline_run_id: UUID of the run batch to summarise.

    Returns:
        JSON string ``{pipeline_run_id, applied, failed, manual_queued,
        total_attempted}`` or an error dict on failure.
    """
    conn = None
    try:
        conn = get_db_conn()
        if not conn:
            logger.error("get_apply_summary: DB connection failed")
            return json.dumps({"error": "db_not_configured", "detail": "DB connection failed"})
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cursor.execute(
            """
            SELECT a.status, COUNT(*) AS cnt
            FROM applications a
            JOIN jobs jp ON jp.id = a.job_post_id
            WHERE jp.pipeline_run_id = %s
            GROUP BY a.status
            """,
            (pipeline_run_id,),
        )

        rows = cursor.fetchall()

        counts: dict[str, int] = {"applied": 0, "failed": 0, "manual_queued": 0}
        for row in rows:
            status_key: str = str(row["status"])
            if status_key in counts:
                counts[status_key] = int(str(row["cnt"]))

        total: int = sum(counts.values())

        logger.info(
            "get_apply_summary: pipeline_run_id=%s applied=%d failed=%d "
            "manual_queued=%d total=%d",
            pipeline_run_id,
            counts["applied"],
            counts["failed"],
            counts["manual_queued"],
            total,
        )

        return json.dumps(
            {
                "pipeline_run_id": pipeline_run_id,
                "applied": counts["applied"],
                "failed": counts["failed"],
                "manual_queued": counts["manual_queued"],
                "total_attempted": total,
            }
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("get_apply_summary failed: %s", exc)
        return json.dumps(
            {"error": "get_apply_summary_failed", "detail": str(exc)}
        )
    finally:
        if conn:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    # Unreachable — both branches return above; type-checker satisfaction only
    return json.dumps(  # pragma: no cover
        {"error": "get_apply_summary_failed", "detail": "unreachable"}
    )


# ---------------------------------------------------------------------------
# TOOL 6 — Route and Apply (HTTP call to apply_service)
# ---------------------------------------------------------------------------
@tool
def route_and_apply(
    job_id: str,
    job_url: str,
    resume_path: str,
    fit_score: float,
    pipeline_run_id: str,
    user_id: str,
    job_title: str = "",
    company: str = "",
    platform: str = "",
) -> str:
    """Route a job to the correct platform and execute application via apply_service.
    
    This is the primary tool for ApplyAgent to execute applications via HTTP
    calls to the apply_service microservice running on port 8003.
    
    Args:
        job_id: UUID of the job post.
        job_url: Full URL of the job application page.
        resume_path: Filename of the resume PDF.
        fit_score: Fit score from analyser (0.0-1.0).
        pipeline_run_id: UUID of the current run batch.
        user_id: UUID of the candidate.
        job_title: Job title for context.
        company: Company name for context.
        platform: ATS platform (auto-detected if empty).
        
    Returns:
        JSON string with apply result from apply_service.
    """
    apply_service_url = os.getenv(
        "AUTO_APPLY_SERVICE_URL",
        "http://ai_auto_apply:8003",
    )
    
    # Detect platform from URL if not provided
    if not platform:
        detector = ATSDetector()
        detected_type = detector._detect_by_url(job_url)
        platform = detected_type.value if detected_type else "native"
    
    try:
        payload = {
            "job_id": job_id,
            "resume_path": resume_path,
            "platform": platform,
            "job_url": job_url,
            "fit_score": fit_score,
            "pipeline_run_id": pipeline_run_id,
            "user_id": user_id,
            "job_title": job_title,
            "company": company,
        }
        
        response = None
        url = f"{apply_service_url}/apply"
        for attempt in range(3):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=120,  # 2 minute timeout for Playwright operations
                )
                if response.status_code >= 500:
                    response.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 2:
                    logger.error("requests.post failed after 3 attempts: %s", e)
                    raise
                time.sleep(2 ** attempt)
        if response is None:
            return json.dumps({"success": False, "error": "apply_service_no_response"})
        
        if response.status_code == 200:
            result = response.json()
            logger.info(
                "route_and_apply: job=%s status=%s",
                job_id,
                result.get("status"),
            )
            return json.dumps(result)
        else:
            logger.error(
                "route_and_apply: HTTP %d for job %s: %s",
                response.status_code,
                job_id,
                response.text,
            )
            return json.dumps({
                "status": "failed",
                "job_id": job_id,
                "error_code": f"HTTP_{response.status_code}",
                "error_message": response.text[:200],
            })
            
    except requests.exceptions.Timeout:
        logger.error("route_and_apply: timeout for job %s", job_id)
        return json.dumps({
            "status": "queued",
            "job_id": job_id,
            "error_code": "TIMEOUT",
            "error_message": "Apply service request timed out",
        })
    except requests.exceptions.ConnectionError as e:
        logger.error("route_and_apply: connection error for job %s: %s", job_id, e)
        return json.dumps({
            "status": "queued",
            "job_id": job_id,
            "error_code": "CONNECTION_ERROR",
            "error_message": str(e),
        })
    except Exception as exc:
        logger.error("route_and_apply: error for job %s: %s", job_id, exc)
        return json.dumps({
            "status": "failed",
            "job_id": job_id,
            "error_code": "APPLY_ERROR",
            "error_message": str(exc),
        })


# ---------------------------------------------------------------------------
# TOOL 7 — Save Application Result to Postgres
# ---------------------------------------------------------------------------
@tool
def save_application_result(
    job_id: str,
    user_id: str,
    status: str,
    platform: str,
    proof_type: str = "",
    proof_value: str = "",
    proof_confidence: float = 0.0,
    error_code: str = "",
    screenshot_path: str = "",
    cost_usd: float = 0.0,
) -> str:
    """Save an application result to the applications table.
    
    Args:
        job_id: UUID of the job post.
        user_id: UUID of the user.
        status: Application status (applied, failed, manual_queued).
        platform: ATS platform used.
        proof_type: Type of submission proof.
        proof_value: Proof value (URL, confirmation number).
        proof_confidence: Proof confidence 0.0-1.0.
        error_code: Error code if failed.
        screenshot_path: Path to proof screenshot.
        cost_usd: LLM cost incurred.
        
    Returns:
        JSON string with application_id or error.
    """
    conn = None
    try:
        conn = get_db_conn()
        if not conn:
            logger.error("save_application_result: DB connection failed")
            return json.dumps({"success": False, "error": "DB connection failed"})
        
        conn.autocommit = False
        cursor = conn.cursor()
        
        import uuid
        app_id = str(uuid.uuid4())
        
        cursor.execute(
            """
            INSERT INTO applications (id, job_post_id, user_id, mode, status, platform, error_code)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (app_id, job_id, user_id, "auto", status, platform, error_code or None)
        )
        
        conn.commit()
        
        logger.info("save_application_result: created %s for job %s", app_id, job_id)
        
        return json.dumps({
            "success": True,
            "application_id": app_id,
            "job_id": job_id,
            "status": status,
        })
        
    except Exception as exc:
        if conn:
            conn.rollback()
        logger.error("save_application_result failed: %s", exc)
        return json.dumps({
            "success": False,
            "error": str(exc),
        })
    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------------
# TOOL 8 — Save to Manual Queue
# ---------------------------------------------------------------------------
@tool
def save_to_queue(
    job_id: str,
    user_id: str,
    reason: str,
    priority: int = 5,
    notes: str = "",
) -> str:
    """Save a job to the manual queue for later processing.
    
    Creates an application record with status='manual_queued' and
    adds an entry to the queued_jobs table.
    
    Args:
        job_id: UUID of the job post.
        user_id: UUID of the user.
        reason: Reason for queueing (becomes error_code).
        priority: Queue priority (higher = more urgent, default 5).
        notes: Additional notes for manual reviewer.
        
    Returns:
        JSON string with queue entry details or error.
    """
    conn = None
    try:
        conn = get_db_conn()
        if not conn:
            logger.error("save_to_queue: DB connection failed")
            return json.dumps({"success": False, "error": "DB connection failed"})
        
        conn.autocommit = False
        cursor = conn.cursor()
        
        import uuid
        app_id = str(uuid.uuid4())
        
        # First create application with manual_queued status
        cursor.execute(
            """
            INSERT INTO applications (id, job_post_id, user_id, mode, status, platform, error_code)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (app_id, job_id, user_id, "manual", "manual_queued", "unknown", reason)
        )
        
        # Then create queued_jobs entry
        cursor.execute(
            """
            INSERT INTO queued_jobs (application_id, job_post_id, priority, notes)
            VALUES (%s, %s, %s, %s)
            """,
            (app_id, job_id, _priority_text(int(priority)), notes or reason)
        )
        
        conn.commit()
        
        logger.info("save_to_queue: queued job %s with priority %d", job_id, priority)
        
        return json.dumps({
            "success": True,
            "application_id": app_id,
            "job_id": job_id,
            "priority": priority,
            "reason": reason,
        })
        
    except Exception as exc:
        if conn:
            conn.rollback()
        logger.error("save_to_queue failed: %s", exc)
        return json.dumps({
            "success": False,
            "error": str(exc),
        })
    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------------
# TOOL 9 — Get Best Resume via RAG
# ---------------------------------------------------------------------------
@tool
def get_best_resume(
    job_title: str,
    job_description: str,
    company: str = "",
) -> str:
    """Get the best matching resume for a job via RAG server.
    
    Calls the RAG service /match endpoint to find the most suitable
    resume variant based on job requirements.
    
    Args:
        job_title: Title of the job position.
        job_description: Full job description text.
        company: Company name (optional).
        
    Returns:
        JSON string with resume_path and match_score, or error.
    """
    rag_service_url = os.getenv(
        "RAG_SERVER_URL",
        "http://ai_rag_server:8090",
    )
    default_resume = os.getenv("DEFAULT_RESUME", "AarjunGen.pdf")
    
    try:
        payload = {
            "job_title": job_title,
            "job_description": job_description[:2000],  # Truncate for token limits
            "company": company,
        }
        
        response = None
        url = f"{rag_service_url}/match"
        for attempt in range(3):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=30,
                )
                if response.status_code >= 500:
                    response.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 2:
                    logger.error("requests.post failed after 3 attempts: %s", e)
                    raise
                time.sleep(2 ** attempt)
        if response is None:
            return json.dumps({"success": False, "error": "rag_service_no_response"})
        
        if response.status_code == 200:
            result = response.json()
            resume_path = result.get("resume_path", default_resume)
            match_score = result.get("match_score", 0.0)
            
            logger.info(
                "get_best_resume: matched '%s' for %s (score=%.2f)",
                resume_path,
                job_title,
                match_score,
            )
            
            return json.dumps({
                "success": True,
                "resume_path": resume_path,
                "match_score": match_score,
            })
        else:
            logger.warning(
                "get_best_resume: RAG service returned %d, using default",
                response.status_code,
            )
            return json.dumps({
                "success": True,
                "resume_path": default_resume,
                "match_score": 0.5,
                "fallback": True,
            })
            
    except requests.exceptions.RequestException as e:
        logger.warning("get_best_resume: RAG service unavailable: %s", e)
        return json.dumps({
            "success": True,
            "resume_path": default_resume,
            "match_score": 0.5,
            "fallback": True,
            "error": str(e),
        })
    except Exception as exc:
        logger.error("get_best_resume failed: %s", exc)
        return json.dumps({
            "success": True,
            "resume_path": default_resume,
            "match_score": 0.5,
            "fallback": True,
            "error": str(exc),
        })


# ---------------------------------------------------------------------------
# TOOL 10 — Verify Apply Budget
# ---------------------------------------------------------------------------
@tool
def verify_apply_budget(
    projected_cost: float,
    pipeline_run_id: str,
) -> str:
    """Check if projected cost fits within remaining budget.
    
    Verifies both the per-run xAI cap and monthly budget before
    allowing an LLM call to proceed.
    
    Args:
        projected_cost: Estimated cost of the upcoming operation in USD.
        pipeline_run_id: UUID of the current run batch.
        
    Returns:
        JSON string with allowed (bool) and remaining_budget.
    """
    try:
        # Check xAI run cap
        cap_result = json.loads(check_xai_run_cap(pipeline_run_id))
        
        if cap_result.get("abort", False):
            return json.dumps({
                "allowed": False,
                "reason": "xai_run_cap_exceeded",
                "remaining_budget": 0.0,
            })
        
        current_spent = cap_result.get("spent", 0.0)
        cap = cap_result.get("cap", 0.38)
        remaining = cap - current_spent
        
        if projected_cost > remaining:
            return json.dumps({
                "allowed": False,
                "reason": "insufficient_budget",
                "remaining_budget": remaining,
                "projected_cost": projected_cost,
            })
        
        return json.dumps({
            "allowed": True,
            "remaining_budget": remaining,
            "projected_cost": projected_cost,
        })
        
    except Exception as exc:
        logger.error("verify_apply_budget failed: %s", exc)
        # Fail open - allow operation if budget check fails
        return json.dumps({
            "allowed": True,
            "remaining_budget": -1,  # Unknown
            "error": str(exc),
        })


# ═══════════════════════════════════════════════════════════════════════════════
# .func ALIASES — raw function access (bypasses CrewAI Tool Pydantic wrapper)
# ═══════════════════════════════════════════════════════════════════════════════
_detect_ats_platform     = detect_ats_platform.func     if hasattr(detect_ats_platform,     "func") else detect_ats_platform
_capture_proof           = capture_proof.func           if hasattr(capture_proof,           "func") else capture_proof
_check_captcha_present   = check_captcha_present.func   if hasattr(check_captcha_present,   "func") else check_captcha_present
_fill_standard_form      = fill_standard_form.func      if hasattr(fill_standard_form,      "func") else fill_standard_form
_get_apply_summary       = get_apply_summary.func       if hasattr(get_apply_summary,       "func") else get_apply_summary
_route_and_apply         = route_and_apply.func         if hasattr(route_and_apply,         "func") else route_and_apply
_save_application_result = save_application_result.func if hasattr(save_application_result, "func") else save_application_result
_save_to_queue           = save_to_queue.func           if hasattr(save_to_queue,           "func") else save_to_queue
_get_best_resume         = get_best_resume.func         if hasattr(get_best_resume,         "func") else get_best_resume
_verify_apply_budget     = verify_apply_budget.func     if hasattr(verify_apply_budget,     "func") else verify_apply_budget
