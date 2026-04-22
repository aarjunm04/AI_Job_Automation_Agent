"""Absolute Kill Switch for AgentOps."""

import logging

logger = logging.getLogger(__name__)

def safe_track_agent(name: str = ""):
    def decorator(cls):
        return cls
    return decorator

def safe_track_tool(fn):
    return fn


def safe_end_agentops_session() -> None:
    """Safely end the AgentOps session without crashing on
    NonRecordingSpan version mismatches.

    Wraps agentops.end_session() in a broad try/except so that
    AttributeError on span.name or span.context never propagates
    to the caller. Logs a debug message if ending fails.
    """
    try:
        import agentops
        agentops.end_session(end_state="Success")
    except AttributeError as e:
        logger.debug(
            "AgentOps session end suppressed (NonRecordingSpan "
            "version mismatch): %s", e
        )
    except Exception as e:
        logger.debug("AgentOps session end failed (non-fatal): %s", e)
