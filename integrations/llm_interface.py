# integrations/llm_interface.py
"""
LLM Interface Module — Spec-Compliant Agent LLM Configuration

Centralized interface for managing CrewAI LLM objects for each agent type.
Provides primary and fallback LLM chains per IDE_README.md AGENT SYSTEM table.
Includes cost projection and budget-aware call gating.
All API keys via os.getenv(); no Gemini or OpenRouter logic (embedding-only providers).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from crewai import LLM
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from java.env in home directory or project root
env_path = Path.home() / "java.env" if (Path.home() / "java.env").exists() else Path("java.env")
load_dotenv(dotenv_path=env_path)

# --- CRITICAL LITELLM ENVIRONMENT OVERRIDES ---
# LiteLLM aggressively prioritizes environment variables over kwargs.
# If these are malformed in the local environment, it causes Errno 8 and 404s.
# We forcefully correct them here before any LLM object is instantiated.

if os.getenv("XAI_BASE_URL") == "xai":
    os.environ["XAI_BASE_URL"] = "https://api.x.ai/v1"

# Force Cerebras to use the correct model ID and base URL
# Cerebras model IDs use no hyphen: llama3.3-70b not llama-3.3-70b
os.environ["CEREBRAS_MODEL"] = os.getenv("CEREBRAS_MODEL")

# ----------------------------------------------


def _resolve_fallback_1_model() -> str:
    """Resolve fallback_1 model string based on available API keys."""
    if os.getenv("CEREBRAS_API_KEY"):
        return "cerebras/llama3.3-70b"
    if os.getenv("SAMBANOVA_API_KEY"):
        return "sambanova/Meta-Llama-3.3-70B-Instruct"
    return "groq/llama-3.3-70b-versatile"

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
        "primary": (f"groq/{os.getenv('GROQ_MODEL')}", "GROQ_API_KEY", os.getenv("GROQ_BASE_URL")),
        "fallback_1": (_resolve_fallback_1_model(), "CEREBRAS_API_KEY", os.getenv("CEREBRAS_BASE_URL")),
        "fallback_2": ("groq/llama-3.3-70b-versatile", "GROQ_API_KEY", os.getenv("GROQ_BASE_URL")),
    },
    "SCRAPER_AGENT": {
        "primary": (f"groq/{os.getenv('GROQ_MODEL')}", "GROQ_API_KEY", os.getenv("GROQ_BASE_URL")),
        "fallback_1": (_resolve_fallback_1_model(), "CEREBRAS_API_KEY", os.getenv("CEREBRAS_BASE_URL")),
        "fallback_2": ("groq/llama-3.3-70b-versatile", "GROQ_API_KEY", os.getenv("GROQ_BASE_URL")),
    },
    "ANALYSER_AGENT": {
        "primary": ("xai/grok-4-fast-reasoning", "XAI_API_KEY", os.getenv("XAI_BASE_URL")),
        "fallback_1": (_resolve_fallback_1_model(), "CEREBRAS_API_KEY", os.getenv("CEREBRAS_BASE_URL")),
        "fallback_2": ("groq/llama-3.3-70b-versatile", "GROQ_API_KEY", os.getenv("GROQ_BASE_URL")),
    },
    "APPLY_AGENT": {
        "primary": (
            "xai/grok-4-1-fast-reasoning",
            "XAI_API_KEY",
            os.getenv("XAI_BASE_URL"),
        ),
        "fallback_1": (_resolve_fallback_1_model(), "CEREBRAS_API_KEY", os.getenv("CEREBRAS_BASE_URL")),
        "fallback_2": ("groq/llama-3.3-70b-versatile", "GROQ_API_KEY", os.getenv("GROQ_BASE_URL")),
    },
    "TRACKER_AGENT": {
        "primary": (
            f"groq/{os.getenv('GROQ_MODEL')}",
            "GROQ_API_KEY",
            os.getenv("GROQ_BASE_URL"),
        ),
        "fallback_1": (_resolve_fallback_1_model(), "CEREBRAS_API_KEY", os.getenv("CEREBRAS_BASE_URL")),
        "fallback_2": ("groq/llama-3.3-70b-versatile", "GROQ_API_KEY", os.getenv("GROQ_BASE_URL")),
    },
    "DEVELOPER_AGENT": {
        "primary": (
            "xai/grok-4-1-fast-reasoning",
            "XAI_API_KEY",
            os.getenv("XAI_BASE_URL"),
        ),
        "fallback_1": (
            f"perplexity/{os.getenv('PERPLEXITY_DEFAULT_MODEL')}",
            "PERPLEXITY_API_KEY",
            os.getenv("PERPLEXITY_BASE_URL"),
        ),
        "fallback_2": ("groq/llama-3.3-70b-versatile", "GROQ_API_KEY", os.getenv("GROQ_BASE_URL")),
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

    Current Agent System (Primary LLM):
    - Master    → Groq llama-3.3-70b-versatile
    - Scraper   → Groq llama-3.3-70b-versatile (CrewAI orchestration) | Perplexity sonar reserved for async complete() serpapi calls only
    - Analyser  → xAI grok-4-fast-reasoning
    - Apply     → xAI grok-4-1-fast-reasoning
    - Tracker   → Groq llama-3.3-70b-versatile
    - Developer → xAI grok-3-mini-latest
"""

    # Auth-error substrings that should NOT be retried
    _AUTH_SIGNALS: tuple[str, ...] = (
        "401", "403", "unauthorized", "forbidden",
        "invalid api key", "api key",
    )

    # Cost-per-1K-token estimates (input+output blended) for budget projection
    _COST_PER_1K_TOKENS: dict[str, float] = {
        "xai": 0.005,           # xAI Grok models (paid)
        "perplexity": 0.0005,   # Perplexity Sonar (cheap)
        "groq": 0.0,            # Groq free tier
        "cerebras": 0.0,        # Cerebras free tier
        "sambanova": 0.0,       # SambaNova free tier
    }

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
                # Safety net for xAI base URL if not loaded from environment
                if provider_name == "xai" and not base_url:
                    base_url = "https://api.x.ai/v1"
                    self.logger.info("xAI base_url not found in environment, using default: %s", base_url)

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

    # ------------------------------------------------------------------
    # Public: cost projection and budget gating
    # ------------------------------------------------------------------

    def project_cost(self, agent_type: str, estimated_tokens: int = 2000) -> dict[str, Any]:
        """Project the cost of an LLM call before making it.

        Uses the active provider for ``agent_type`` to estimate cost
        based on per-1K-token rates.

        Args:
            agent_type: One of the valid agent type strings.
            estimated_tokens: Approximate total tokens (input + output).
                Default 2000 (typical for a scoring call).

        Returns:
            Dict with ``provider``, ``model``, ``estimated_tokens``,
            ``projected_cost_usd``, and ``cost_per_1k_tokens``.
        """
        agent_type = agent_type.strip().upper()
        chain = self._chain_for(agent_type)
        if not chain:
            return {
                "provider": "none",
                "model": "none",
                "estimated_tokens": estimated_tokens,
                "projected_cost_usd": 0.0,
                "cost_per_1k_tokens": 0.0,
            }

        # Use first available provider
        for provider_name, model, _api_key, _base_url in chain:
            if provider_name not in self._unavailable:
                cost_1k = self._COST_PER_1K_TOKENS.get(provider_name, 0.001)
                projected = (estimated_tokens / 1000.0) * cost_1k
                return {
                    "provider": provider_name,
                    "model": model,
                    "estimated_tokens": estimated_tokens,
                    "projected_cost_usd": round(projected, 6),
                    "cost_per_1k_tokens": cost_1k,
                }

        return {
            "provider": "all_unavailable",
            "model": "none",
            "estimated_tokens": estimated_tokens,
            "projected_cost_usd": 0.0,
            "cost_per_1k_tokens": 0.0,
        }

    def check_budget_before_call(
        self,
        agent_type: str,
        estimated_tokens: int = 2000,
        budget_remaining: float = 0.10,
    ) -> dict[str, Any]:
        """Check if a projected LLM call fits within remaining budget.

        Args:
            agent_type: One of the valid agent type strings.
            estimated_tokens: Approximate total tokens (input + output).
            budget_remaining: Remaining budget in USD for the current run.

        Returns:
            Dict with ``proceed`` (bool), ``projected_cost_usd``,
            ``budget_remaining``, and ``provider``.
        """
        projection = self.project_cost(agent_type, estimated_tokens)
        proceed = projection["projected_cost_usd"] <= budget_remaining
        if not proceed:
            self.logger.warning(
                "Budget gate: projected $%.6f > remaining $%.4f for %s — BLOCKED",
                projection["projected_cost_usd"],
                budget_remaining,
                agent_type,
            )
        return {
            "proceed": proceed,
            "projected_cost_usd": projection["projected_cost_usd"],
            "budget_remaining": budget_remaining,
            "provider": projection["provider"],
            "model": projection["model"],
        }

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

    # =========================================================================
    # ASYNC COMPLETE — Spec-Compliant Fallback Chain (caller-driven)
    # =========================================================================
    # If caller == "scraper":
    #   0) Perplexity → 1) Groq → 2) Cerebras
    # Else (default: analyser/apply):
    #   0) xAI → 1) SambaNova → 2) Cerebras
    # =========================================================================

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        purpose: str = "general",
        budget_remaining: Optional[float] = None,
        run_batch_id: Optional[str] = None,
    ) -> str:
        """
        Execute an LLM completion with a spec-compliant fallback chain.

        Args:
            prompt: The prompt to send to the LLM.
            max_tokens: Maximum tokens for the response.
            purpose: Description of what this completion is for (for logging).
            budget_remaining: Optional remaining budget in USD.

        Returns:
            The LLM response text.

        Raises:
            LLMExhaustedError: If all 5 providers fail.
        """
        try:
            from tools.agentops_tools import _record_fallback_event as record_fallback_event
        except Exception:  # pragma: no cover
            record_fallback_event = None  # type: ignore[assignment]

        try:
            from tools.postgres_tools import _log_event as log_event
        except Exception:  # pragma: no cover
            log_event = None  # type: ignore[assignment]

        caller = (purpose or "").strip().lower()
        if caller != "scraper":
            caller = caller or "analyser"

        # Fallback chain configuration (provider, model, api_key_env, base_url, is_paid)
        analyser_apply_chain: list[tuple[str, str, str, str, bool]] = [
            (
                "xai",
                os.getenv("XAI_DEFAULT_MODEL"),
                "XAI_API_KEY",
                os.getenv("XAI_BASE_URL"),
                True,
            ),
            (
                "cerebras",
                os.getenv("CEREBRAS_MODEL"),
                "CEREBRAS_API_KEY",
                os.getenv("CEREBRAS_BASE_URL"),
                False,
            ),
        ]

        scraper_chain: list[tuple[str, str, str, str, bool]] = [
            (
                "perplexity",
                os.getenv("PERPLEXITY_MODEL"),
                "PERPLEXITY_API_KEY",
                os.getenv("PERPLEXITY_BASE_URL"),
                True,
            ),
            (
                "groq",
                os.getenv("GROQ_MODEL"),
                "GROQ_API_KEY",
                os.getenv("GROQ_BASE_URL"),
                False,
            ),
            (
                "cerebras",
                os.getenv("CEREBRAS_MODEL"),
                "CEREBRAS_API_KEY",
                os.getenv("CEREBRAS_BASE_URL"),
                False,
            ),
        ]

        chain = scraper_chain if caller == "scraper" else analyser_apply_chain

        # Budget gate: Level 0 only (no retries/levels attempted if blocked)
        provider0 = chain[0][0]
        projected_tokens_in = max(1, len(prompt) // 4)
        projected_tokens_out = max_tokens
        projected_cost_usd = (
            (projected_tokens_in + projected_tokens_out) / 1000.0
        ) * self._COST_PER_1K_TOKENS.get(provider0, 0.0)
        if budget_remaining is not None and projected_cost_usd > budget_remaining:
            raise BudgetExceededError(
                f"BudgetExceededError: projected_cost={projected_cost_usd:.6f} > remaining={budget_remaining:.6f}"
            )

        last_error: Optional[Exception] = None

        for level, (provider_name, model, api_key_env, base_url, is_paid) in enumerate(chain):
            # Check if provider is marked unavailable
            if provider_name in self._unavailable:
                self.logger.debug("Skipping unavailable provider %s", provider_name)
                continue

            api_key = os.getenv(api_key_env, "").strip()
            if not api_key:
                last_error = RuntimeError(f"Missing or empty {api_key_env}")
            else:
                for attempt in range(3):
                    try:
                        response_text = await self._call_provider(
                            provider_name=provider_name,
                            model=model,
                            api_key=api_key,
                            base_url=base_url,
                            prompt=prompt,
                            max_tokens=max_tokens,
                        )

                        if is_paid:
                            await self._track_cost_async(
                                provider_name, projected_tokens_in, max_tokens, run_batch_id
                            )

                        self.logger.info(
                            "complete() success: caller=%s provider=%s purpose=%s tokens~%d",
                            caller,
                            provider_name,
                            purpose,
                            max_tokens,
                        )
                        return response_text
                    except Exception as e:
                        last_error = e
                        error_str = str(e).lower()

                        # Auth errors — mark unavailable
                        if any(s in error_str for s in self._AUTH_SIGNALS):
                            self.logger.error("Provider %s auth error: %s", provider_name, e)
                            self._unavailable.add(provider_name)
                            break

                        if attempt == 2:
                            break

                        delay = 2 ** attempt  # 1s, 2s, 4s
                        self.logger.warning(
                            "Provider %s attempt %d/3 failed: %s — retrying in %ds",
                            provider_name,
                            attempt + 1,
                            e,
                            delay,
                        )
                        await asyncio.sleep(delay)

            # Level failed after 3 retries (or missing key): log and fall through to next level
            if last_error is not None:
                self.logger.warning(
                    "Provider %s failed after 3 attempts: %s", provider_name, last_error
                )
                if log_event is not None:
                    try:
                        log_event(
                            run_batch_id=run_batch_id or "async_call",
                            level="ERROR",
                            event_type="llm_provider_failed",
                            message=f"caller={caller} provider={provider_name} error={last_error}",
                        )
                    except Exception:  # noqa: BLE001
                        pass

            # Record fallback event when advancing to the next level
            if level < len(chain) - 1 and record_fallback_event is not None:
                next_provider = chain[level + 1][0]
                try:
                    record_fallback_event(
                        agent_type="LLMInterface",
                        from_provider=provider_name,
                        to_provider=next_provider,
                        run_batch_id=run_batch_id or "async_call",
                        fallback_level=level + 1,
                        reason=str(last_error) if last_error else "",
                    )
                except Exception:  # noqa: BLE001
                    pass

        raise RuntimeError(f"LLM fallback chain exhausted for caller={caller}")

    async def _call_provider(
        self,
        provider_name: str,
        model: str,
        api_key: str,
        base_url: str,
        prompt: str,
        max_tokens: int,
    ) -> str:
        """Make an async HTTP call to an LLM provider."""
        import httpx

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Build the request payload (OpenAI-compatible format)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }

        # Adjust for Perplexity's different API
        if provider_name == "perplexity":
            payload["model"] = f"llama-3.1-sonar-large-128k-online"

        endpoint = f"{base_url}/chat/completions"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(endpoint, headers=headers, json=payload)

            if response.status_code == 429:
                raise Exception(f"429 rate limit exceeded for {provider_name}")

            if response.status_code != 200:
                raise Exception(
                    f"Provider {provider_name} returned {response.status_code}: {response.text}"
                )

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise Exception(f"No choices in response from {provider_name}")

            return choices[0].get("message", {}).get("content", "")

    async def _track_cost_async(
        self,
        provider: str,
        tokens_in: int,
        tokens_out: int,
        run_batch_id: Optional[str] = None,
    ) -> None:
        """Track LLM cost asynchronously via budget_tools."""
        try:
            from tools.budget_tools import record_llm_cost

            cost_per_1k = self._COST_PER_1K_TOKENS.get(provider, 0.0)
            total_tokens = tokens_in + tokens_out
            cost_usd = (total_tokens / 1000.0) * cost_per_1k

            if cost_usd > 0:
                record_llm_cost(
                    provider=provider,
                    cost_usd=cost_usd,
                    agent_type="ASYNC_COMPLETE",
                    run_batch_id=run_batch_id or "async_call",
                )
        except Exception as e:
            self.logger.warning("Failed to track cost: %s", e)


# =========================================================================
# CUSTOM EXCEPTIONS
# =========================================================================

class LLMExhaustedError(Exception):
    """Raised when all LLM providers in the fallback chain are exhausted."""
    pass


class BudgetExceededError(Exception):
    """Raised when a projected LLM call exceeds the remaining budget."""
    pass


# Import asyncio for the async methods
import asyncio
