"""Database package for the AI Job Application Agent.

Exports a get_db_client() factory that returns the correct database client
based on the ACTIVE_DB environment variable. Import only this function —
never instantiate client classes directly in application code.

Usage::

    from database import get_db_client

    db = get_db_client()
    rows = db.execute_query("SELECT * FROM jobs LIMIT 10;")
"""

from __future__ import annotations

__all__ = ["get_db_client", "LocalPostgresClient", "SupabasePostgresClient"]

import logging
import os
from typing import Optional, Union

from database.local_postgres_client import LocalPostgresClient
from database.supabase_client import SupabasePostgresClient

_db_client_instance: Optional[Union[LocalPostgresClient, SupabasePostgresClient]] = None


def get_db_client() -> Union[LocalPostgresClient, SupabasePostgresClient]:
    """Return the active database client singleton based on ACTIVE_DB env var.

    Reads ACTIVE_DB from environment (default: "local"). Returns a cached
    singleton — the connection pool is created once and reused across all calls.

    Returns:
        LocalPostgresClient if ACTIVE_DB=local.
        SupabasePostgresClient if ACTIVE_DB=supabase.

    Raises:
        RuntimeError: If ACTIVE_DB is set to an unrecognised value.
    """
    global _db_client_instance
    if _db_client_instance is not None:
        return _db_client_instance

    active_db: str = os.getenv("ACTIVE_DB", "local").strip().lower()
    _logger = logging.getLogger(__name__)

    if active_db == "local":
        _logger.info("DB mode: local (Docker Postgres)")
        _db_client_instance = LocalPostgresClient()
    elif active_db == "supabase":
        _logger.info("DB mode: supabase (cloud Postgres)")
        _db_client_instance = SupabasePostgresClient()
    else:
        raise RuntimeError(
            f"Unknown ACTIVE_DB value: '{active_db}'. Must be 'local' or 'supabase'."
        )

    return _db_client_instance
