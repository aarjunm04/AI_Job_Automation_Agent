"""Centralised Postgres connection helper for the AI Job Application Agent.

All agent and tool modules that need a psycopg2 connection should import
``get_db_conn`` from here instead of maintaining their own copies.
"""

from __future__ import annotations

import logging
import os

import psycopg2
from psycopg2.extensions import connection as PgConnection

__all__ = ["get_db_conn"]

logger = logging.getLogger(__name__)


def get_db_conn() -> PgConnection:
    """Open and return a psycopg2 connection to the active Postgres instance.

    Selects the connection URL based on the ``ACTIVE_DB`` environment variable
    (``"local"`` → ``LOCAL_POSTGRES_URL``, anything else → ``SUPABASE_URL``).
    The returned connection has ``autocommit=False``.

    Returns:
        Database connection with ``autocommit=False``.

    Raises:
        RuntimeError: If the connection URL is not configured or if the
            connection attempt fails.
    """
    active_db = os.getenv("ACTIVE_DB", "local")
    db_url: str | None = (
        os.getenv("LOCAL_POSTGRES_URL")
        if active_db == "local"
        else os.getenv("SUPABASE_URL")
    )

    if not db_url:
        raise RuntimeError(
            "Database URL is not configured.  "
            "Set LOCAL_POSTGRES_URL or SUPABASE_URL in java.env and "
            "ACTIVE_DB=local|supabase."
        )
    try:
        conn = psycopg2.connect(db_url)
        conn.autocommit = False
        return conn
    except Exception as exc:
        raise RuntimeError(f"Postgres connection failed: {exc}") from exc
