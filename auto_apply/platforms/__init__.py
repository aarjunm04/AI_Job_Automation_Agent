"""Platform-specific Playwright apply modules for ATS auto-apply."""

from auto_apply.platforms.base_platform import (
    BasePlatformApply,
    ApplyResult,
    ApplyError,
)
from auto_apply.platforms.greenhouse import GreenhouseApply
from auto_apply.platforms.lever import LeverApply
from auto_apply.platforms.workday import WorkdayApply
from auto_apply.platforms.linkedin_easy_apply import LinkedInEasyApply
from auto_apply.platforms.native_form import NativeFormApply
from auto_apply.platforms.indeed import IndeedApply
from auto_apply.platforms.wellfound import WellfoundApply
from auto_apply.platforms.arc_dev import ArcDevApply

__all__ = [
    "BasePlatformApply",
    "ApplyResult",
    "ApplyError",
    "GreenhouseApply",
    "LeverApply",
    "WorkdayApply",
    "LinkedInEasyApply",
    "NativeFormApply",
    "IndeedApply",
    "WellfoundApply",
    "ArcDevApply",
]

