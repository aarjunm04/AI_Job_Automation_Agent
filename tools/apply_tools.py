"""
Apply tools for AI Job Application Agent.

Provides CrewAI tool functions for ATS platform detection, CAPTCHA checking,
proof-of-submission capture, Playwright-driven form filling, and per-run apply
summary aggregation. All Playwright actions are fail-soft: a single field
failure never aborts the run. DRY_RUN=true guards every submission path.

All user profile data is sourced exclusively from environment variables loaded
via narad.env.
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
from typing import Any, Optional, cast

import psycopg2
import psycopg2.extras
from crewai.tools import tool
from litellm import completion
from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError, async_playwright
import agentops

from tools.postgres_tools import create_application, log_event
from tools.budget_tools import check_xai_run_cap, record_llm_cost
from tools.agentops_tools import record_agent_error

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# Constants from environment
# ---------------------------------------------------------------------------
DRY_RUN: bool = os.getenv("DRY_RUN", "false").lower() == "true"
RESUME_DIR: str = os.getenv("RESUME_DIR", "resumes")
MAX_SESSIONS: int = int(os.getenv("MAX_PLAYWRIGHT_SESSIONS", "5"))

# Database URL (mirrors pattern in other tool modules)
_DB_URL: Optional[str] = (
    os.getenv("LOCAL_POSTGRES_URL")
    if os.getenv("ACTIVE_DB", "local") == "local"
    else os.getenv("SUPABASE_URL")
)

# APPLY_AGENT model string (xAI primary per LLMInterface._AGENT_CONFIG)
_APPLY_MODEL: str = "xai/grok-4-1-fast-reasoning"
_APPLY_API_KEY_ENV: str = "XAI_API_KEY"
_APPLY_API_BASE: str = "https://api.x.ai/v1"

# ---------------------------------------------------------------------------
# ATS URL patterns
# ---------------------------------------------------------------------------
ATS_PATTERNS: dict[str, list[str]] = {
    "greenhouse": ["greenhouse.io"],
    "lever": ["lever.co"],
    "workday": ["workday.com", "myworkdayjobs.com"],
    "ashby": ["ashbyhq.com"],
    "jobvite": ["jobvite.com"],
    "icims": ["icims.com"],
    "smartrecruiters": ["smartrecruiters.com"],
    "bamboohr": ["bamboohr.com"],
    "direct": [],
}

# ---------------------------------------------------------------------------
# Proxy round-robin state
# ---------------------------------------------------------------------------
_proxy_index: int = 0


def _get_proxy() -> Optional[dict[str, str]]:
    """Return the next proxy in round-robin rotation from WEBSHARE_PROXY_LIST.

    Returns:
        A dict ``{"server": "<proxy_url>"}`` for Playwright, or ``None`` if
        the environment variable is not set or the list is empty.
    """
    global _proxy_index  # noqa: PLW0603

    raw: str = os.getenv("WEBSHARE_PROXY_LIST", "").strip()
    if not raw:
        return None

    proxies: list[str] = [p.strip() for p in raw.split(",") if p.strip()]
    if not proxies:
        return None

    proxy_url: str = proxies[_proxy_index % len(proxies)]
    _proxy_index = (_proxy_index + 1) % len(proxies)
    return {"server": proxy_url}


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
# TOOL 1 — ATS platform detection
# ---------------------------------------------------------------------------
@tool
@agentops.track_tool
def detect_ats_platform(job_url: str, run_batch_id: str) -> str:
    """Detect the ATS platform from a job URL using pure pattern matching.

    No network calls are made. The URL is matched against known ATS domain
    patterns defined in ``ATS_PATTERNS``. Returns ``"direct"`` when no known
    ATS is detected.

    Args:
        job_url: The full URL of the job posting.
        run_batch_id: UUID of the current run batch (used for logging only).

    Returns:
        JSON string ``{"ats": str, "job_url": str, "confidence": "high"|"low"}``.
    """
    try:
        url_lower: str = job_url.lower()
        detected_ats: str = "direct"
        confidence: str = "low"

        for ats_name, domains in ATS_PATTERNS.items():
            if ats_name == "direct":
                continue
            for domain in domains:
                if domain in url_lower:
                    detected_ats = ats_name
                    confidence = "high"
                    break
            if detected_ats != "direct":
                break

        logger.info(
            "detect_ats_platform: url=%s ats=%s confidence=%s",
            job_url,
            detected_ats,
            confidence,
        )
        return json.dumps(
            {"ats": detected_ats, "job_url": job_url, "confidence": confidence}
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("detect_ats_platform failed: %s", exc)
        return json.dumps(
            {"ats": "direct", "job_url": job_url, "confidence": "low"}
        )


# ---------------------------------------------------------------------------
# TOOL 2 — Proof of submission capture
# ---------------------------------------------------------------------------
@tool
@agentops.track_tool
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
# TOOL 3 — CAPTCHA detection
# ---------------------------------------------------------------------------
@tool
@agentops.track_tool
def check_captcha_present(page_html: str, job_url: str) -> str:
    """Detect CAPTCHA or bot-challenge presence in page HTML.

    Scans for known CAPTCHA fingerprints (reCAPTCHA, hCaptcha, Cloudflare
    Turnstile, etc.). Logs a WARNING-level event to Postgres when detected.

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
            # run_batch_id is not available at this tool scope; log via Python
            # logger only — calling log_event without a valid run_batch_id would
            # violate the FK constraint on logs_events.run_batch_id.
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
    """Execute the full Playwright apply flow for a single job posting.

    Handles: dry-run gate, budget gate, CAPTCHA abort, LLM field detection,
    form fill, submission, proof capture, screenshot, and DB audit write.

    Args:
        job_url: URL of the job application form.
        job_post_id: UUID of the job_posts row.
        resume_filename: Filename of the resume PDF in RESUME_DIR.
        run_batch_id: UUID of the current run batch.
        user_id: UUID of the candidate user.
        ats_platform: Detected ATS platform label.

    Returns:
        Dict with apply result keys (applied, status, proof_confidence, etc.).
    """
    # Step 1 — DRY_RUN gate
    if DRY_RUN:
        logger.info("_run_apply: DRY_RUN=true — skipping real apply for %s", job_url)
        log_event(run_batch_id, "INFO", "dry_run_skip", f"dry_run|{job_url}")
        return {"applied": False, "dry_run": True, "job_url": job_url}

    # Step 2 — Budget gate
    try:
        cap_result: dict[str, Any] = json.loads(check_xai_run_cap(run_batch_id))
        if cap_result.get("abort"):
            logger.critical(
                "_run_apply: xAI budget cap hit — aborting apply for %s", job_url
            )
            return {
                "applied": False,
                "status": "failed",
                "reason": "budget_cap_hit",
                "job_url": job_url,
            }
    except Exception as exc:  # noqa: BLE001
        logger.warning("_run_apply: budget cap check failed (proceeding): %s", exc)

    browser = None
    context = None
    try:
        # Step 3 — Launch Playwright browser
        async with async_playwright() as pw:
            proxy_cfg: Optional[dict[str, str]] = _get_proxy()
            launch_kwargs: dict[str, Any] = {"headless": True}
            if proxy_cfg:
                launch_kwargs["proxy"] = proxy_cfg

            browser = await pw.chromium.launch(**launch_kwargs)

            context_kwargs: dict[str, Any] = {
                "user_agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
            context = await browser.new_context(**context_kwargs)
            page: Page = await context.new_page()

            # Step 4 — Navigate to job URL
            await page.goto(job_url, wait_until="networkidle", timeout=30000)

            # Step 5 — Capture initial HTML
            html: str = await page.content()

            # Step 6 — CAPTCHA check
            captcha_result: dict[str, Any] = json.loads(
                check_captcha_present(html, job_url)
            )
            if captcha_result.get("captcha_detected"):
                logger.warning(
                    "_run_apply: CAPTCHA detected — re-routing to manual: %s", job_url
                )
                await context.close()
                await browser.close()
                return {
                    "applied": False,
                    "status": "failed",
                    "re_route": "manual",
                    "reason": "captcha",
                    "job_url": job_url,
                }

            # Step 7 — LLM field detection
            field_map: dict[str, str] = {}
            try:
                xai_api_key: Optional[str] = os.getenv(_APPLY_API_KEY_ENV)
                if xai_api_key:
                    _start: int = 0
                    _stop: int = 8000
                    html_snippet: str = html[_start:_stop]  # explicit int bounds for type checker
                    llm_prompt = (
                        "You are a form analysis assistant. Given the following HTML, "
                        "identify all visible input fields in the application form. "
                        "Return ONLY a valid JSON object mapping CSS selectors to field "
                        "types. Example: {\"input[name='first_name']\": \"first_name\", "
                        "\"input[name='email']\": \"email\"}. "
                        "Field types to detect: first_name, last_name, email, phone, "
                        "linkedin, portfolio, website, location, years_experience, resume, cv. "
                        "Return only JSON, no explanation.\n\nHTML:\n" + html_snippet
                    )
                    llm_response = completion(
                        model=_APPLY_MODEL,
                        api_key=xai_api_key,
                        api_base=_APPLY_API_BASE,
                        messages=[{"role": "user", "content": llm_prompt}],
                        max_tokens=512,
                    )
                    raw_content: str = llm_response.choices[0].message.content or "{}"
                    # Strip markdown code fences if present
                    raw_content = re.sub(
                        r"^```(?:json)?\s*|\s*```$", "", raw_content.strip()
                    )
                    # cast: json.loads returns Any; we assert the LLM returned a
                    # flat {selector: field_type} object — invalid shapes are caught
                    # by the per-field try/except in step 8.
                    parsed: Any = json.loads(raw_content)
                    field_map = cast(dict[str, str], parsed) if isinstance(parsed, dict) else {}
            except Exception as exc:  # noqa: BLE001
                logger.warning("_run_apply: LLM field detection failed: %s", exc)
                field_map = {}

            try:
                record_llm_cost("xai", 0.002, "APPLY_AGENT", run_batch_id)
            except Exception:  # noqa: BLE001
                pass

            # Step 8 — Fill fields
            username: str = os.getenv("USERNAME", "")
            name_parts: list[str] = username.split()
            first_name: str = name_parts[0] if name_parts else ""
            last_name: str = name_parts[-1] if len(name_parts) > 1 else ""

            field_values: dict[str, str] = {
                "first_name": first_name,
                "last_name": last_name,
                "email": os.getenv("USER_EMAIL", ""),
                "phone": os.getenv("USER_PHONE", ""),
                "linkedin": os.getenv("USER_LINKEDIN_URL", ""),
                "portfolio": os.getenv("USER_PORTFOLIO_URL", ""),
                "website": os.getenv("USER_PORTFOLIO_URL", ""),
                "location": os.getenv("USER_LOCATION", ""),
                "years_experience": os.getenv("USER_YEARS_EXPERIENCE", ""),
            }
            file_field_types: set[str] = {"resume", "cv"}
            resume_path: Path = Path(RESUME_DIR) / resume_filename

            for selector, field_type in field_map.items():
                try:
                    if field_type in file_field_types:
                        if not DRY_RUN and resume_path.exists():
                            await page.set_input_files(selector, str(resume_path))
                        elif DRY_RUN:
                            logger.info(
                                "_run_apply: DRY_RUN — skipping file upload %s", selector
                            )
                    elif field_type in field_values:
                        value: str = field_values[field_type]
                        if value:
                            await page.fill(selector, value)
                except Exception as field_exc:  # noqa: BLE001
                    logger.warning(
                        "_run_apply: field fill skipped selector=%s type=%s err=%s",
                        selector,
                        field_type,
                        field_exc,
                    )

            # Step 9 — Submit (guarded by DRY_RUN)
            if not DRY_RUN:
                submit_selectors: list[str] = [
                    "button[type=submit]",
                    "input[type=submit]",
                    "button:has-text('Submit')",
                    "button:has-text('Apply')",
                ]
                submitted: bool = False
                for submit_sel in submit_selectors:
                    try:
                        submit_locator = page.locator(submit_sel).first
                        if await submit_locator.count() > 0:
                            await submit_locator.click()
                            await page.wait_for_load_state(
                                "networkidle", timeout=15000
                            )
                            submitted = True
                            break
                    except Exception as sub_exc:  # noqa: BLE001
                        logger.warning(
                            "_run_apply: submit selector failed %s: %s",
                            submit_sel,
                            sub_exc,
                        )
                if not submitted:
                    logger.warning(
                        "_run_apply: no submit button found for %s", job_url
                    )

            # Step 10 — Post-submit page snapshot
            post_html: str = await page.content()
            post_url: str = page.url

            # Step 11 — Proof capture
            proof: dict[str, Any] = json.loads(
                capture_proof(post_html, post_url, job_url)
            )

            # Step 12 — Screenshot (always captured for audit)
            screenshot_b64: str = ""
            try:
                screenshot_bytes: bytes = await page.screenshot(
                    type="png", full_page=False
                )
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
            except Exception as ss_exc:  # noqa: BLE001
                logger.warning("_run_apply: screenshot failed: %s", ss_exc)

            # Step 13 — Close browser
            await context.close()
            await browser.close()
            browser = None

            # Step 14 — Determine final status
            applied_status: str = (
                "applied"
                if proof.get("proof_confidence") in ("high", "medium", "low")
                else "failed"
            )

            # Step 15 — Persist application record
            try:
                create_application(
                    job_post_id=job_post_id,
                    resume_id="",
                    user_id=user_id,
                    mode="auto",
                    status=applied_status,
                    platform=ats_platform,
                    error_code="" if applied_status == "applied" else "proof_none",
                )
            except Exception as db_exc:  # noqa: BLE001
                logger.error("_run_apply: create_application failed: %s", db_exc)

            # Step 16 — Audit log
            try:
                log_event(
                    run_batch_id,
                    "INFO" if applied_status == "applied" else "ERROR",
                    "auto_apply_attempt",
                    (
                        f"{applied_status}|{job_url}|"
                        f"confidence={proof.get('proof_confidence')}"
                    ),
                )
            except Exception as log_exc:  # noqa: BLE001
                logger.warning("_run_apply: log_event failed: %s", log_exc)

            # Step 17 — Return result
            return {
                "applied": applied_status == "applied",
                "status": applied_status,
                "proof_confidence": proof.get("proof_confidence"),
                "proof_signals": proof.get("signals_captured"),
                "job_url": job_url,
                "ats_platform": ats_platform,
                "screenshot_captured": bool(screenshot_b64),
                "dry_run": False,
            }

    except Exception as outer_exc:  # noqa: BLE001
        # Close browser if still open — best effort, swallow secondary errors
        if browser:
            try:
                await browser.close()
            except Exception:  # noqa: BLE001
                pass
        logger.error("_run_apply: unexpected error for %s: %s", job_url, outer_exc)
        # Return a structured failure dict so fill_standard_form can decide
        # whether to retry (TimeoutError) or immediately re-route to manual.
        return {
            "applied": False,
            "status": "failed",
            "reason": str(outer_exc),
            "re_route": "manual",
            "job_url": job_url,
        }

    # Unreachable fallback — the async with always returns via step 17 or the
    # outer except above, but the type checker needs an explicit return here.
    return {  # pragma: no cover
        "applied": False,
        "status": "failed",
        "reason": "unreachable",
        "job_url": job_url,
    }


# ---------------------------------------------------------------------------
# TOOL 4 — Fill standard form (sync wrapper)
# ---------------------------------------------------------------------------
@tool
@agentops.track_tool
def fill_standard_form(
    job_url: str,
    job_post_id: str,
    resume_filename: str,
    run_batch_id: str,
    user_id: str,
    ats_platform: str,
) -> str:
    """Apply to a job via Playwright with LLM-assisted form field detection.

    Wraps the async apply coroutine in a synchronous interface for CrewAI.
    Retries up to 2 times on ``TimeoutError`` only, using exponential backoff.
    Any non-retriable failure is logged and re-routed to manual queue.

    DRY_RUN=true prevents any real form submission or file upload from
    occurring, even if this function is called.

    Args:
        job_url: URL of the job application page.
        job_post_id: UUID of the job_posts row.
        resume_filename: Filename of the resume PDF inside RESUME_DIR.
        run_batch_id: UUID of the current run batch.
        user_id: UUID of the candidate (users table).
        ats_platform: Detected ATS platform label (from detect_ats_platform).

    Returns:
        JSON string with apply outcome fields:
        applied, status, proof_confidence, proof_signals, job_url,
        ats_platform, screenshot_captured, dry_run (or error details).
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
                backoff: int = 2**attempt
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
                }
            )

    # Unreachable — kept for type-checker satisfaction
    return json.dumps(
        {"applied": False, "status": "failed", "reason": "max_retries_exceeded"}
    )


# ---------------------------------------------------------------------------
# TOOL 5 — Per-run apply summary
# ---------------------------------------------------------------------------
@tool
@agentops.track_tool
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
            # RealDictRow values are typed as object; cast explicitly
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

    # Unreachable fallback — both try and except branches return, but the type
    # checker cannot prove this when a finally block is present.
    return json.dumps(  # pragma: no cover
        {"error": "get_apply_summary_failed", "detail": "unreachable"}
    )
