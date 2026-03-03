# integrations/llm_interface.py
"""
LLM Interface Module — Spec-Compliant Agent LLM Configuration

Centralized interface for managing CrewAI LLM objects for each agent type.
Provides primary and fallback LLM chains per IDE_README.md AGENT SYSTEM table.
All API keys via os.getenv(); no Gemini or OpenRouter logic (embedding-only providers).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from crewai import LLM
from dotenv import load_dotenv

load_dotenv()

__all__ = ["LLMInterface"]

# Logging configuration
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Agent type constants
VALID_AGENT_TYPES = frozenset({
    "MASTER_AGENT",
    "SCRAPER_AGENT",
    "ANALYSER_AGENT",
    "APPLY_AGENT",
    "TRACKER_AGENT",
    "DEVELOPER_AGENT",
})

# Provider config: (model, api_key_env, base_url or None)
_AGENT_CONFIG: dict[str, dict[str, Any]] = {
    "MASTER_AGENT": {
        "primary": ("groq/llama-3.3-70b-versatile", "GROQ_API_KEY", None),
        "fallback_1": ("cerebras/llama3.3-70b", "CEREBRAS_API_KEY", None),
        "fallback_2": None,
    },
    "SCRAPER_AGENT": {
        "primary": ("perplexity/sonar", "PERPLEXITY_API_KEY", None),
        "fallback_1": None,
        "fallback_2": None,
    },
    "ANALYSER_AGENT": {
        "primary": ("xai/grok-4-fast-reasoning", "XAI_API_KEY", "https://api.x.ai/v1"),
        "fallback_1": ("sambanova/Meta-Llama-3.1-70B-Instruct", "SAMBANOVA_API_KEY", None),
        "fallback_2": ("cerebras/llama3.3-70b", "CEREBRAS_API_KEY", None),
    },
    "APPLY_AGENT": {
        "primary": ("xai/grok-4-1-fast-reasoning", "XAI_API_KEY", "https://api.x.ai/v1"),
        "fallback_1": ("sambanova/Meta-Llama-3.1-70B-Instruct", "SAMBANOVA_API_KEY", None),
        "fallback_2": ("cerebras/llama3.3-70b", "CEREBRAS_API_KEY", None),
    },
    "TRACKER_AGENT": {
        "primary": ("groq/llama-3.3-70b-versatile", "GROQ_API_KEY", None),
        "fallback_1": ("cerebras/llama3.3-70b", "CEREBRAS_API_KEY", None),
        "fallback_2": None,
    },
    "DEVELOPER_AGENT": {
        "primary": ("xai/grok-3-mini-latest", "XAI_API_KEY", "https://api.x.ai/v1"),
        "fallback_1": ("perplexity/sonar", "PERPLEXITY_API_KEY", None),
        "fallback_2": None,
    },
}


def _create_llm(model: str, api_key: str, base_url: Optional[str] = None) -> LLM:
    """Create a CrewAI LLM instance with optional base_url for xAI providers."""
    kwargs: dict[str, Any] = {"model": model, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return LLM(**kwargs)


class LLMInterface:
    """
    Centralized LLM configuration for all CrewAI agents.

    Returns configured CrewAI LLM objects for each agent type per the
    IDE_README.md AGENT SYSTEM table. Supports primary and fallback chains
    with per-provider retry, exponential backoff, and unavailability tracking.
    """

    # Auth-error substrings that should NOT be retried
    _AUTH_SIGNALS: tuple[str, ...] = (
        "401", "403", "unauthorized", "forbidden",
        "invalid api key", "api key",
    )

    def __init__(self) -> None:
        self._unavailable: set[str] = set()
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Private: construction helper with retry + backoff
    # ------------------------------------------------------------------

    def _build_llm(
        self,
        provider_name: str,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
    ) -> Optional[LLM]:
        """Attempt to construct a CrewAI LLM with up to 3 retries.

        Auth errors are detected immediately and the provider is marked
        unavailable without retrying. Transient errors are retried with
        2s / 4s exponential backoff.

        Args:
            provider_name: Human-readable name used for logging and
                the ``_unavailable`` set (e.g. ``"groq"``).
            model: CrewAI model string (e.g. ``"groq/llama-3.3-70b-versatile"``).
            api_key: Resolved API key string.
            base_url: Optional base URL override for the provider.

        Returns:
            Constructed ``LLM`` on success, ``None`` if all attempts failed.
        """
        llm: Optional[LLM] = None
        for attempt in range(1, 4):  # max 3 attempts
            try:
                llm = _create_llm(model, api_key, base_url)
                break  # success — exit retry loop
            except Exception as e:
                err_str = str(e).lower()
                # Auth errors — do not retry, mark unavailable immediately
                if any(sig in err_str for sig in self._AUTH_SIGNALS):
                    self.logger.error(
                        "LLM provider '%s' auth error — marking unavailable: %s",
                        provider_name, str(e),
                    )
                    self._unavailable.add(provider_name)
                    return None
                # Transient error — retry with backoff unless exhausted
                if attempt < 3:
                    wait = 2 ** attempt  # 2s, then 4s
                    self.logger.warning(
                        "LLM provider '%s' attempt %d/3 failed: %s "
                        "— retrying in %ds",
                        provider_name, attempt, str(e), wait,
                    )
                    time.sleep(wait)
                else:
                    self.logger.error(
                        "LLM provider '%s' failed after 3 attempts: %s "
                        "— marking unavailable",
                        provider_name, str(e),
                    )
                    self._unavailable.add(provider_name)
                    return None
        return llm

    # ------------------------------------------------------------------
    # Private: build ordered fallback chain for an agent type
    # ------------------------------------------------------------------

    def _chain_for(self, agent_type: str) -> list[tuple[str, str, str, Optional[str]]]:
        """Return ordered list of (provider_name, model, api_key, base_url).

        Skips entries where the API key env var is not set. The list
        preserves primary → fallback_1 → fallback_2 order.
        """
        config = _AGENT_CONFIG[agent_type]
        chain: list[tuple[str, str, str, Optional[str]]] = []
        for slot in ("primary", "fallback_1", "fallback_2"):
            entry = config.get(slot)
            if entry is None:
                continue
            model, key_env, base_url = entry
            api_key = os.getenv(key_env, "").strip()
            provider_name = key_env.replace("_API_KEY", "").lower()
            if not api_key:
                self.logger.warning(
                    "API key %s not set for %s %s — skipping",
                    key_env, agent_type, slot,
                )
                continue
            chain.append((provider_name, model, api_key, base_url))
        return chain

    # ------------------------------------------------------------------
    # Public: get_llm — primary + full fallback walk
    # ------------------------------------------------------------------

    def get_llm(self, agent_type: str) -> LLM:
        """Return the best available LLM for ``agent_type``.

        Walks primary → fallback_1 → fallback_2 in order. Skips any
        provider already in ``_unavailable``. Constructs and returns
        the first successfully built LLM.

        Args:
            agent_type: One of MASTER_AGENT, SCRAPER_AGENT, ANALYSER_AGENT,
                APPLY_AGENT, TRACKER_AGENT, DEVELOPER_AGENT.

        Returns:
            Configured CrewAI LLM instance.

        Raises:
            ValueError: If ``agent_type`` is unrecognised.
            RuntimeError: If all providers in the chain are unavailable.
        """
        agent_type = agent_type.strip().upper()
        if agent_type not in VALID_AGENT_TYPES:
            raise ValueError(
                f"Unrecognised agent_type: '{agent_type}'. "
                f"Valid types: {sorted(VALID_AGENT_TYPES)}"
            )

        chain = self._chain_for(agent_type)
        for provider_name, model, api_key, base_url in chain:
            if provider_name in self._unavailable:
                self.logger.debug(
                    "Skipping unavailable provider '%s' for %s",
                    provider_name, agent_type,
                )
                continue
            llm = self._build_llm(provider_name, model, api_key, base_url)
            if llm is not None:
                return llm

        raise RuntimeError(
            f"All LLM providers unavailable for agent '{agent_type}'. "
            "Check API keys in java.env."
        )

    def get_fallback_llm(self, agent_type: str, level: int = 1) -> Optional[LLM]:
        """Return fallback_1 LLM when level=1, fallback_2 when level=2.

        Args:
            agent_type: One of the valid agent type strings.
            level: 1 for fallback_1, 2 for fallback_2.

        Returns:
            Configured CrewAI LLM for the fallback at that level, or None
            if no fallback exists at that level or the provider is unavailable.

        Raises:
            ValueError: If ``agent_type`` is unrecognised.
        """
        agent_type = agent_type.strip().upper()
        if agent_type not in VALID_AGENT_TYPES:
            raise ValueError(
                f"Unrecognised agent_type: '{agent_type}'. "
                f"Valid types: {sorted(VALID_AGENT_TYPES)}"
            )

        key = f"fallback_{level}"
        fallback = _AGENT_CONFIG[agent_type].get(key)
        if fallback is None:
            return None

        model, key_env, base_url = fallback
        api_key = os.getenv(key_env, "").strip()
        provider_name = key_env.replace("_API_KEY", "").lower()

        if not api_key:
            self.logger.warning(
                "API key %s not set for %s %s — skipping",
                key_env, agent_type, key,
            )
            return None

        if provider_name in self._unavailable:
            self.logger.debug(
                "Skipping unavailable provider '%s' for %s fallback_%d",
                provider_name, agent_type, level,
            )
            return None

        return self._build_llm(provider_name, model, api_key, base_url)

    def get_llm_with_fallback(self, agent_type: str) -> tuple[LLM, list[LLM]]:
        """Return (primary_llm, [fallback_llms in order]) for the agent type.

        Convenience method for agents to receive their full fallback chain
        at once. Skips unavailable providers transparently.

        Args:
            agent_type: One of the valid agent type strings.

        Returns:
            Tuple of (primary LLM, list of fallback LLMs in order).
            The list may be empty if the agent has no fallbacks.

        Raises:
            ValueError: If ``agent_type`` is unrecognised.
            RuntimeError: If the primary provider and all fallbacks are
                unavailable.
        """
        primary = self.get_llm(agent_type)
        fallbacks: list[LLM] = []
        for level in (1, 2):
            fb = self.get_fallback_llm(agent_type, level=level)
            if fb is not None:
                fallbacks.append(fb)
        return (primary, fallbacks)

    def test_connection(self, agent_type: str) -> dict[str, Any]:
        """
        Test primary LLM with a minimal single-token ping call.

        Does not raise; always returns a status dict.

        Args:
            agent_type: One of the valid agent type strings.

        Returns:
            Dict with keys: agent, provider, model, reachable, latency_ms, error.
            error is None on success.
        """
        agent_type = agent_type.strip().upper()
        provider = "unknown"
        model = "unknown"
        start = time.perf_counter()

        if agent_type not in VALID_AGENT_TYPES:
            return {
                "agent": agent_type,
                "provider": provider,
                "model": model,
                "reachable": False,
                "latency_ms": (time.perf_counter() - start) * 1000,
                "error": f"Unrecognised agent_type: '{agent_type}'",
            }

        config = _AGENT_CONFIG[agent_type]
        model, key_env, base_url = config["primary"]
        api_key = os.getenv(key_env)
        if not api_key or not str(api_key).strip():
            return {
                "agent": agent_type,
                "provider": key_env.replace("_API_KEY", "").lower(),
                "model": model,
                "reachable": False,
                "latency_ms": (time.perf_counter() - start) * 1000,
                "error": f"Missing or empty {key_env}",
            }

        provider = key_env.replace("_API_KEY", "").lower()
        max_retries = 3
        last_error: Optional[str] = None

        for attempt in range(max_retries):
            try:
                import litellm

                completion_kwargs: dict[str, Any] = {
                    "model": model,
                    "api_key": api_key.strip(),
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1,
                }
                if base_url:
                    completion_kwargs["api_base"] = base_url
                response = litellm.completion(**completion_kwargs)
                if response and response.choices:
                    latency_ms = (time.perf_counter() - start) * 1000
                    return {
                        "agent": agent_type,
                        "provider": provider,
                        "model": model,
                        "reachable": True,
                        "latency_ms": round(latency_ms, 2),
                        "error": None,
                    }
            except ImportError as e:
                last_error = f"litellm not available: {e}"
                break
            except Exception as e:  # noqa: BLE001
                last_error = str(e)
                logger.warning(
                    "test_connection %s attempt %d failed: %s",
                    agent_type,
                    attempt + 1,
                    e,
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)

        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "agent": agent_type,
            "provider": provider,
            "model": model,
            "reachable": False,
            "latency_ms": round(latency_ms, 2),
            "error": last_error,
        }

    def test_all_connections(self) -> list[dict[str, Any]]:
        """
        Run test_connection for all 6 agent types.

        Logs a summary of how many providers are reachable vs failed.

        Returns:
            List of 6 status dicts, one per agent type.
        """
        results: list[dict[str, Any]] = []
        for agent_type in sorted(VALID_AGENT_TYPES):
            results.append(self.test_connection(agent_type))

        reachable = sum(1 for r in results if r.get("reachable") is True)
        failed = len(results) - reachable
        logger.info(
            "test_all_connections: %d reachable, %d failed out of %d agents",
            reachable,
            failed,
            len(results),
        )
        return results
