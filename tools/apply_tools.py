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

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import agentops
from agentops.sdk.decorators import agent, operation
import psycopg2
import psycopg2.extras
from crewai.tools import tool
from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError, async_playwright

from auto_apply.ats_detector import ATSDetector, ATSProfile, ATSType
from auto_apply.form_filler import FormFiller, FillResult
from integrations.llm_interface import LLMInterface
from tools.postgres_tools import create_application, update_application_status, log_event
from tools.budget_tools import check_xai_run_cap, record_llm_cost
from tools.agentops_tools import record_agent_error
from tools.notion_tools import queue_job_to_applications_db
from config.settings import db_config, run_config, budget_config

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
]

# ---------------------------------------------------------------------------
# Constants from config singletons — no raw os.getenv() in tool logic
# ---------------------------------------------------------------------------
DRY_RUN: bool = run_config.dry_run
RESUME_DIR: Path = Path(run_config.resume_dir)
MAX_SESSIONS: int = run_config.max_playwright_sessions

# Database URL — kept as module-level for get_apply_summary (exact legacy logic)
_DB_URL: Optional[str] = (
    os.getenv("LOCAL_POSTGRES_URL")
    if os.getenv("ACTIVE_DB", "local") == "local"
    else os.getenv("SUPABASE_URL")
)

# ---------------------------------------------------------------------------
# Proxy round-robin — explicit os.getenv per spec
# ---------------------------------------------------------------------------
_proxy_list: list[str] = [
    p.strip() for p in os.getenv("WEBSHARE_PROXY_LIST", "").split(",") if p.strip()
]
_proxy_index: int = 0


def _get_proxy() -> Optional[dict[str, str]]:
    """Return the next proxy in round-robin rotation from WEBSHARE_PROXY_LIST.

    Args:
        None.

    Returns:
        ``{"server": "<proxy_url>"}`` for Playwright context, or ``None``
        when ``WEBSHARE_PROXY_LIST`` is empty or unset.
    """
    global _proxy_index  # noqa: PLW0603
    if not _proxy_list:
        return None
    proxy_url: str = _proxy_list[_proxy_index % len(_proxy_list)]
    _proxy_index = (_proxy_index + 1) % len(_proxy_list)
    return {"server": proxy_url}


# ---------------------------------------------------------------------------
# TOOL 1 — ATS platform detection (Playwright + ATSDetector)
# ---------------------------------------------------------------------------
@tool
@operation
def detect_ats_platform(job_url: str, run_batch_id: str) -> str:
    """Detect the ATS platform powering a job application page.

    Launches a minimal headless Playwright session (no proxy), navigates to
    ``job_url``, and runs the ``ATSDetector`` 3-layer pipeline (URL pattern →
    DOM fingerprint → LLM classification).  Retries up to 2 times on
    ``TimeoutError`` only.

    Args:
        job_url: Full URL of the job application page.
        run_batch_id: UUID of the current run batch (for logging context).

    Returns:
        JSON string of ``ATSProfile.to_dict()`` merged with
        ``{"job_url": job_url}``.  On any failure returns
        ``{"ats_type": "unknown", "confidence": 0.0, "job_url": ..., "error": ...}``.
    """

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
@operation
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
@operation
def check_captcha_present(page_html: str, job_url: str) -> str:
    """Detect CAPTCHA or bot-challenge presence in page HTML.

    Scans for known CAPTCHA fingerprints (reCAPTCHA, hCaptcha, Cloudflare
    Turnstile, etc.). Logs a WARNING-level event to Python logger when
    detected; does not call Postgres directly (no valid run_batch_id at
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
    run_batch_id: str,
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
        run_batch_id: UUID of the current run batch.
        user_id: UUID of the candidate (``users`` table).
        ats_platform: ATS type string from a prior ``detect_ats_platform``
            call, or ``"unknown"`` / ``""`` to trigger live detection.

    Returns:
        Dict with apply result keys per the STEP 13 spec.  Never raises.
    """
    # -----------------------------------------------------------------------
    # STEP 1 — Guards: DRY_RUN + budget cap
    # -----------------------------------------------------------------------
    if DRY_RUN:
        logger.info("_run_apply: DRY_RUN=true — skipping browser for %s", job_url)
        try:
            log_event(run_batch_id, "INFO", "dry_run_skip", f"dry_run|{job_url}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("_run_apply: log_event (dry_run_skip) failed: %s", exc)

        fill_simulation: dict[str, Any] = {}
        try:
            detector_dry = ATSDetector()
            ats_profile_dry: ATSProfile = detector_dry.get_profile_for_ats(ats_platform)
            filler_dry = FormFiller(
                page=None,  # type: ignore[arg-type]
                job_title=run_config.search_query,
                job_description="",
                company="",
                resume_filename=resume_filename,
                ats_type=ats_profile_dry.ats_type.value,
            )
            fill_result_dry: FillResult = await filler_dry.fill_all_fields()
            fill_simulation = {
                "total_fields": fill_result_dry.total_fields,
                "filled": fill_result_dry.filled,
                "skipped": fill_result_dry.skipped,
                "failed": fill_result_dry.failed,
                "llm_calls": fill_result_dry.llm_calls,
                "custom_questions": fill_result_dry.custom_questions,
                "errors": fill_result_dry.errors,
                "success": fill_result_dry.success,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("_run_apply: dry_run simulation error: %s", exc)

        return {
            "applied": False,
            "dry_run": True,
            "ats_type": ats_platform,
            "fill_simulation": fill_simulation,
        }

    # Budget gate
    try:
        budget_result: dict[str, Any] = json.loads(check_xai_run_cap(run_batch_id))
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
        proxy: Optional[dict[str, str]] = _get_proxy()
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
                    run_batch_id,
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
            if not DRY_RUN:
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
                log_event(
                    run_batch_id,
                    "INFO" if status == "applied" else "ERROR",
                    "auto_apply_attempt",
                    (
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
                "dry_run": DRY_RUN,
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
@operation
def fill_standard_form(
    job_url: str,
    job_post_id: str,
    resume_filename: str,
    run_batch_id: str,
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
        run_batch_id: UUID of the current run batch.
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
                    run_batch_id=run_batch_id,
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
                        run_batch_id=run_batch_id,
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
                    run_batch_id=run_batch_id,
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
@operation
def get_apply_summary(run_batch_id: str) -> str:
    """Aggregate application counts by status for a given run batch.

    Queries the ``applications`` table joined with ``job_posts`` to count
    results grouped by status (applied, failed, manual_queued).

    Args:
        run_batch_id: UUID of the run batch to summarise.

    Returns:
        JSON string ``{run_batch_id, applied, failed, manual_queued,
        total_attempted}`` or an error dict on failure.
    """
    if not _DB_URL:
        logger.error("get_apply_summary: DB_URL not configured")
        return json.dumps(
            {"error": "db_not_configured", "detail": "DB_URL not set"}
        )

    conn = None
    try:
        conn = psycopg2.connect(_DB_URL)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cursor.execute(
            """
            SELECT a.status, COUNT(*) AS cnt
            FROM applications a
            JOIN job_posts jp ON jp.id = a.job_post_id
            WHERE jp.run_batch_id = %s
            GROUP BY a.status
            """,
            (run_batch_id,),
        )

        rows = cursor.fetchall()

        counts: dict[str, int] = {"applied": 0, "failed": 0, "manual_queued": 0}
        for row in rows:
            status_key: str = str(row["status"])
            if status_key in counts:
                counts[status_key] = int(str(row["cnt"]))

        total: int = sum(counts.values())

        logger.info(
            "get_apply_summary: run_batch_id=%s applied=%d failed=%d "
            "manual_queued=%d total=%d",
            run_batch_id,
            counts["applied"],
            counts["failed"],
            counts["manual_queued"],
            total,
        )

        return json.dumps(
            {
                "run_batch_id": run_batch_id,
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
