"""
config/config_loader.py

Centralised read-only loader for the three canonical JSON config files:
  - config/user_profile.json      — user data, form answers, job preferences
  - config/platform_config.json   — static platform data (URLs, auth, tiers)
  - config/platform_settings.json — runtime settings (thresholds, budgets, filters)

RULE: All user data and runtime settings come from these files, never from
os.getenv(). Only secrets (API keys, DB URLs, passwords, tokens) live in java.env.

All methods are fail-safe: any missing key or I/O error returns an empty
dict / list and logs a warning — the application NEVER crashes on a bad config key.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["config_loader", "ConfigLoader"]

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
_CONFIG_DIR: Path = Path(__file__).parent


def _load_json(filename: str) -> dict[str, Any]:
    """Load a JSON file relative to the config directory.

    Args:
        filename: Bare filename inside the config/ directory.

    Returns:
        Parsed dict, or empty dict on any failure.
    """
    path = _CONFIG_DIR / filename
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return dict(json.load(fh))
    except FileNotFoundError:
        logger.error("ConfigLoader: file not found — %s", path)
    except json.JSONDecodeError as exc:
        logger.error("ConfigLoader: JSON decode error in %s — %s", path, exc)
    except Exception as exc:  # noqa: BLE001
        logger.error("ConfigLoader: unexpected error loading %s — %s", path, exc)
    return {}


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------


class ConfigLoader:
    """Singleton-style config loader for the three canonical JSON files.

    Loads all three JSON files once at class instantiation and caches them as
    instance attributes.  Every helper method is wrapped in ``try/except`` so
    a missing key never crashes the caller.

    Attributes:
        user: Parsed ``user_profile.json``.
        platforms: Parsed ``platform_config.json``.
        settings: Parsed ``platform_settings.json``.
    """

    def __init__(self) -> None:
        """Load all three config files and cache them."""
        self.user: dict[str, Any] = _load_json("user_profile.json")
        self.platforms: dict[str, Any] = _load_json("platform_config.json")
        self.settings: dict[str, Any] = _load_json("platform_settings.json")
        logger.info(
            "ConfigLoader: loaded — user_profile=%s platform_config=%s platform_settings=%s",
            bool(self.user),
            bool(self.platforms),
            bool(self.settings),
        )

    # ------------------------------------------------------------------
    # user_profile.json helpers
    # ------------------------------------------------------------------

    def get_user_metadata(self) -> dict[str, Any]:
        """Return the ``metadata`` block from ``user_profile.json``.

        Returns:
            Dict of user metadata fields (name, email, phone, etc.),
            or empty dict on failure.
        """
        try:
            return dict(self.user.get("metadata", {}))
        except Exception as exc:  # noqa: BLE001
            logger.error("ConfigLoader.get_user_metadata: %s", exc)
            return {}

    def get_form_answers(self) -> dict[str, Any]:
        """Return the ``application_qa.form_answers`` block from ``user_profile.json``.

        Returns:
            Dict of pre-written form answers keyed by field name,
            or empty dict on failure.
        """
        try:
            return dict(
                self.user.get("application_qa", {}).get("form_answers", {})
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("ConfigLoader.get_form_answers: %s", exc)
            return {}

    def get_job_preferences(self) -> dict[str, Any]:
        """Return the ``job_preferences`` block from ``user_profile.json``.

        Returns:
            Dict of job preference settings, or empty dict on failure.
        """
        try:
            return dict(self.user.get("job_preferences", {}))
        except Exception as exc:  # noqa: BLE001
            logger.error("ConfigLoader.get_job_preferences: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # platform_config.json helpers
    # ------------------------------------------------------------------

    def get_platform(self, name: str) -> dict[str, Any]:
        """Return the config block for a single platform from ``platform_config.json``.

        Args:
            name: Platform key, e.g. ``"linkedin"``, ``"remoteok"``.

        Returns:
            Platform config dict, or empty dict on failure.

        Raises:
            KeyError: Only raised when the platform is genuinely absent and no
                      fallback is appropriate (callers should catch this).
        """
        try:
            result = self.platforms.get(name)
            if result is None:
                raise KeyError(f"Platform '{name}' not found in platform_config.json")
            return dict(result)
        except KeyError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("ConfigLoader.get_platform(%s): %s", name, exc)
            return {}

    # ------------------------------------------------------------------
    # platform_settings.json helpers
    # ------------------------------------------------------------------

    def get_active_platforms(self) -> list[str]:
        """Return names of platforms whose ``active`` flag is ``True``.

        Reads from ``platform_settings.platform_settings.platforms``.

        Returns:
            List of active platform name strings, or empty list on failure.
        """
        try:
            platforms_block: dict[str, Any] = (
                self.settings
                .get("platform_settings", {})
                .get("platforms", {})
            )
            return [
                name
                for name, cfg in platforms_block.items()
                if cfg.get("active") is True
            ]
        except Exception as exc:  # noqa: BLE001
            logger.error("ConfigLoader.get_active_platforms: %s", exc)
            return []

    def get_playwright_platforms(self) -> list[str]:
        """Return the list of Playwright-based platform names.

        Reads from ``platform_settings.platform_settings.playwright_platforms``.

        Returns:
            List of platform name strings, or empty list on failure.
        """
        try:
            return list(
                self.settings
                .get("platform_settings", {})
                .get("playwright_platforms", [])
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("ConfigLoader.get_playwright_platforms: %s", exc)
            return []

    def get_run_config(self) -> dict[str, Any]:
        """Return the ``run_config`` block from ``platform_settings.json``.

        Returns:
            Dict with run configuration values (e.g. ``jobs_per_run_target``),
            or empty dict on failure.
        """
        try:
            return dict(self.settings.get("run_config", {}))
        except Exception as exc:  # noqa: BLE001
            logger.error("ConfigLoader.get_run_config: %s", exc)
            return {}

    def get_scoring_thresholds(self) -> dict[str, Any]:
        """Return the ``scoring_thresholds`` block from ``platform_settings.json``.

        Returns:
            Dict of scoring threshold values, or empty dict on failure.
        """
        try:
            return dict(self.settings.get("scoring_thresholds", {}))
        except Exception as exc:  # noqa: BLE001
            logger.error("ConfigLoader.get_scoring_thresholds: %s", exc)
            return {}

    def get_apply_settings(self) -> dict[str, Any]:
        """Return the ``apply_settings`` block from ``platform_settings.json``.

        Returns:
            Dict of apply settings (dry_run, auto_apply_enabled, delays, etc.),
            or empty dict on failure.
        """
        try:
            return dict(self.settings.get("apply_settings", {}))
        except Exception as exc:  # noqa: BLE001
            logger.error("ConfigLoader.get_apply_settings: %s", exc)
            return {}

    def get_budget_settings(self) -> dict[str, Any]:
        """Return the ``budget_settings`` block from ``platform_settings.json``.

        Returns:
            Dict of budget cap values, or empty dict on failure.
        """
        try:
            return dict(self.settings.get("budget_settings", {}))
        except Exception as exc:  # noqa: BLE001
            logger.error("ConfigLoader.get_budget_settings: %s", exc)
            return {}

    def get_job_filters(self) -> dict[str, Any]:
        """Return the ``job_filters`` block from ``platform_settings.json``.

        Returns:
            Dict of job filter rules (include/exclude keywords, seniority, etc.),
            or empty dict on failure.
        """
        try:
            return dict(self.settings.get("job_filters", {}))
        except Exception as exc:  # noqa: BLE001
            logger.error("ConfigLoader.get_job_filters: %s", exc)
            return {}

    def get_platform_runtime(self, name: str) -> dict[str, Any]:
        """Return the runtime settings for a single platform.

        Reads from ``platform_settings.platform_settings.platforms[name]``.

        Args:
            name: Platform key, e.g. ``"linkedin"``.

        Returns:
            Runtime settings dict (active, max_jobs_per_scrape, scoring_threshold,
            budget_cap_usd, rag_settings), or empty dict on failure.
        """
        try:
            platforms_block: dict[str, Any] = (
                self.settings
                .get("platform_settings", {})
                .get("platforms", {})
            )
            result = platforms_block.get(name)
            if result is None:
                logger.warning(
                    "ConfigLoader.get_platform_runtime: platform '%s' not found", name
                )
                return {}
            return dict(result)
        except Exception as exc:  # noqa: BLE001
            logger.error("ConfigLoader.get_platform_runtime(%s): %s", name, exc)
            return {}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

config_loader: ConfigLoader = ConfigLoader()
