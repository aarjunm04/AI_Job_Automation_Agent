"""utils/shutdown.py

Thread-safe graceful shutdown flag for the entire pipeline.

The SIGTERM handler in main.py calls request_shutdown(). Every agent
and long-running loop polls is_shutdown_requested() between operations.
This prevents Docker force-kill (SIGKILL) from corrupting Postgres
run_sessions rows by giving the pipeline time to write closed_at.

Single source of truth for shutdown state — no global variables
scattered across agent files.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

__all__ = [
    "is_shutdown_requested",
    "request_shutdown",
    "clear_shutdown",
    "get_shutdown_reason",
    "ShutdownGuard",
]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# INTERNAL STATE — module-level, thread-safe
# ---------------------------------------------------------------------------

_shutdown_event: threading.Event = threading.Event()
_shutdown_reason: Optional[str] = None
_shutdown_lock: threading.Lock = threading.Lock()

# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def is_shutdown_requested() -> bool:
    """Check if a graceful shutdown has been requested.

    Called by agents and tools between operations to detect SIGTERM.
    Non-blocking — returns immediately.

    Returns:
        True if shutdown has been requested, False otherwise.

    Example::

        while jobs:
            if is_shutdown_requested():
                logger.warning("Shutdown requested — stopping job loop")
                break
            process(jobs.pop())
    """
    return _shutdown_event.is_set()


def request_shutdown(reason: str = "SIGTERM") -> None:
    """Request a graceful shutdown of the pipeline.

    Sets the shared shutdown event. All polling agents will detect
    this on their next loop iteration and exit cleanly.

    Called by:
        - ``main.py`` SIGTERM/SIGINT signal handler
        - ``MasterAgent`` on critical unrecoverable errors
        - GitHub Actions timeout watchdog (if implemented)

    Args:
        reason: Human-readable reason for the shutdown request.
                Logged and stored for diagnostic purposes.

    Example::

        def _sigterm_handler(signum, frame):
            request_shutdown(reason="SIGTERM received from Docker")
    """
    global _shutdown_reason

    with _shutdown_lock:
        if _shutdown_event.is_set():
            logger.debug(
                "shutdown.request_shutdown: already set (reason=%s) — ignoring duplicate",
                reason,
            )
            return
        _shutdown_reason = reason
        _shutdown_event.set()

    logger.warning(
        "SHUTDOWN REQUESTED — reason=%s | "
        "all agents will exit after current operation completes",
        reason,
    )


def clear_shutdown() -> None:
    """Clear the shutdown flag and reset state.

    WARNING: Only call this in test environments or after a clean
    pipeline restart. Never call during a live run.

    Clears both the threading.Event and the stored reason string.
    """
    global _shutdown_reason

    with _shutdown_lock:
        _shutdown_event.clear()
        _shutdown_reason = None

    logger.info("shutdown.clear_shutdown: shutdown flag cleared")


def get_shutdown_reason() -> Optional[str]:
    """Return the reason string passed to request_shutdown().

    Returns:
        The reason string if shutdown was requested, None otherwise.

    Example::

        if is_shutdown_requested():
            reason = get_shutdown_reason()
            log_event(run_batch_id, "WARNING", "shutdown", reason or "unknown")
    """
    with _shutdown_lock:
        return _shutdown_reason


def wait_for_shutdown(timeout: Optional[float] = None) -> bool:
    """Block until shutdown is requested or timeout elapses.

    Useful for background watchdog threads that need to wake up
    only when the pipeline is stopping.

    Args:
        timeout: Seconds to wait. None = block indefinitely.

    Returns:
        True if shutdown was requested, False if timeout elapsed.
    """
    return _shutdown_event.wait(timeout=timeout)


# ---------------------------------------------------------------------------
# CONTEXT MANAGER — for guarded operations
# ---------------------------------------------------------------------------

class ShutdownGuard:
    """Context manager that raises RuntimeError if shutdown is requested.

    Wrap any multi-step critical section (e.g. a Postgres write sequence)
    to ensure it does not begin after a shutdown signal arrives.

    Example::

        with ShutdownGuard("postgres_write"):
            conn = get_db_conn()
            cur.execute(INSERT_SQL, values)
            conn.commit()

    Raises:
        RuntimeError: If shutdown has already been requested when entering.
    """

    def __init__(self, operation_name: str = "unnamed") -> None:
        """Initialise the guard.

        Args:
            operation_name: Label shown in log and error message.
        """
        self.operation_name = operation_name

    def __enter__(self) -> "ShutdownGuard":
        """Check shutdown flag on entry.

        Returns:
            Self.

        Raises:
            RuntimeError: If shutdown has been requested.
        """
        if is_shutdown_requested():
            reason = get_shutdown_reason() or "unknown"
            raise RuntimeError(
                f"ShutdownGuard blocked '{self.operation_name}' — "
                f"shutdown already requested (reason={reason})"
            )
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> bool:
        """Exit the guarded block.

        Does not suppress exceptions — always returns False.

        Returns:
            False (never suppresses exceptions).
        """
        return False
