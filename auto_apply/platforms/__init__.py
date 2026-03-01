"""Platform-specific Playwright apply modules for ATS auto-apply."""

from auto_apply.platforms.base_platform import (
    BasePlatformApply,
    ApplyResult,
    ApplyError,
)
from auto_apply.platforms.greenhouse import GreenhouseApply
from auto_apply.platforms.lever import LeverApply
from auto_apply.platforms.workday import WorkdayApply

__all__ = [
    "BasePlatformApply",
    "ApplyResult",
    "ApplyError",
    "GreenhouseApply",
    "LeverApply",
    "WorkdayApply",
]
