"""
auto_apply/platforms/__init__.py
Fault-tolerant platform registry. Each import is individually guarded.
One broken platform NEVER cascades to kill all others.
"""
from __future__ import annotations
import logging
import importlib
from typing import Optional

logger = logging.getLogger(__name__)


def _safe_import(module: str, classname: str) -> Optional[type]:
    """Import a platform class safely. Returns None on failure."""
    try:
        mod = importlib.import_module(module)
        cls = getattr(mod, classname, None)
        if cls is None:
            logger.warning("Class %s not found in %s", classname, module)
        return cls
    except Exception as exc:
        logger.warning("Import failed [%s.%s]: %s", module, classname, exc)
        return None


GreenhousePlatform        = _safe_import("auto_apply.platforms.greenhouse",           "GreenhousePlatform")
LeverPlatform             = _safe_import("auto_apply.platforms.lever",                "LeverPlatform")
WorkdayPlatform           = _safe_import("auto_apply.platforms.workday",              "WorkdayPlatform")
WellfoundPlatform         = _safe_import("auto_apply.platforms.wellfound",            "WellfoundPlatform")
LinkedInEasyApplyPlatform = _safe_import("auto_apply.platforms.linkedin_easy_apply",  "LinkedInEasyApplyPlatform")
IndeedEasyApplyPlatform   = _safe_import("auto_apply.platforms.indeed_easy_apply",    "IndeedEasyApplyPlatform")
NativeFormPlatform        = _safe_import("auto_apply.platforms.native_form",          "NativeFormPlatform")

# Legacy aliases — keeps apply_service.py and ATSDetector working
GreenhouseApply   = GreenhousePlatform
LeverApply        = LeverPlatform
WorkdayApply      = WorkdayPlatform
WellfoundApply    = WellfoundPlatform
LinkedInEasyApply = LinkedInEasyApplyPlatform
IndeedEasyApply   = IndeedEasyApplyPlatform
NativeFormApply   = NativeFormPlatform

PLATFORM_MAP: dict[str, Optional[type]] = {
    "greenhouse"          : GreenhousePlatform,
    "lever"               : LeverPlatform,
    "workday"             : WorkdayPlatform,
    "wellfound"           : WellfoundPlatform,
    "linkedin_easy_apply" : LinkedInEasyApplyPlatform,
    "indeed_easy_apply"   : IndeedEasyApplyPlatform,
    "native_form"         : NativeFormPlatform,
}

_loaded = [k for k, v in PLATFORM_MAP.items() if v is not None]
_failed = [k for k, v in PLATFORM_MAP.items() if v is None]
if _loaded:
    logger.info("Platforms loaded OK: %s", _loaded)
if _failed:
    logger.warning("Platforms FAILED to load: %s", _failed)

__all__ = [
    "GreenhousePlatform",         "GreenhouseApply",
    "LeverPlatform",              "LeverApply",
    "WorkdayPlatform",            "WorkdayApply",
    "WellfoundPlatform",          "WellfoundApply",
    "LinkedInEasyApplyPlatform",  "LinkedInEasyApply",
    "IndeedEasyApplyPlatform",    "IndeedEasyApply",
    "NativeFormPlatform",         "NativeFormApply",
    "PLATFORM_MAP",
]
