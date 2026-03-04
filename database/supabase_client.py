"""Supabase PostgreSQL client using psycopg2 ThreadedConnectionPool.

Used when ACTIVE_DB=supabase. Connects directly to the Supabase project's
Postgres wire-protocol endpoint via psycopg2 with sslmode=require — no
supabase-py SDK dependency. Exposes an interface identical to
LocalPostgresClient so callers require zero changes when switching modes.
"""

from __future__ import annotations

__all__ = ["SupabasePostgresClient"]

import logging
import os
import time
from typing import Any, Optional

import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class SupabasePostgresClient:
    """Production-grade psycopg2 ThreadedConnectionPool client for Supabase Postgres.

    Connects to the Supabase cloud Postgres instance using the project's direct
    database connection credentials (not the supabase-py REST/RPC SDK). SSL is
    mandatory (``sslmode=require``) for all Supabase connections.

    This class exposes an interface that is **identical** to
    :class:`~database.local_postgres_client.LocalPostgresClient`, ensuring
    that the rest of the codebase can switch between local and cloud Postgres
    with zero code changes — only the ``ACTIVE_DB`` environment variable needs
    to differ.

    Attributes:
        _pool: The underlying psycopg2 ThreadedConnectionPool instance.
        _pool_min: Minimum number of connections kept open.
        _pool_max: Maximum number of connections in the pool.
    """

    def __init__(self) -> None:
        """Initialise the ThreadedConnectionPool for Supabase Postgres.

        Builds the DSN from individual environment variables and enforces
        ``sslmode=require``. Applies three-attempt retry with exponential
        back-off (2 s / 4 s / 8 s).

        Environment variables consumed:
            SUPABASE_DB_HOST: Supabase project database host (required).
            SUPABASE_DB_PORT: Database port (default "5432").
            SUPABASE_DB_NAME: Database name (default "postgres").
            SUPABASE_DB_USER: Database user (default "postgres").
            SUPABASE_DB_PASSWORD: Database password / service-role key (required).
            PG_POOL_MIN: Minimum pool size (default 2).
            PG_POOL_MAX: Maximum pool size (default 10).

        Raises:
            RuntimeError: If required env vars are missing, or if the pool
                cannot be created after 3 retry attempts.
        """
        host: Optional[str] = os.getenv("SUPABASE_DB_HOST")
        password: Optional[str] = os.getenv("SUPABASE_DB_PASSWORD")

        if host is None or password is None:
            logger.critical(
                "SUPABASE_DB_HOST and SUPABASE_DB_PASSWORD are required "
                "when ACTIVE_DB=supabase."
            )
            raise RuntimeError(
                "SupabasePostgresClient requires SUPABASE_DB_HOST and "
                "SUPABASE_DB_PASSWORD to be set in the environment."
            )

        port: str = os.getenv("SUPABASE_DB_PORT", "5432")
        dbname: str = os.getenv("SUPABASE_DB_NAME", "postgres")
        user: str = os.getenv("SUPABASE_DB_USER", "postgres")

        dsn: str = (
            f"postgresql://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"
        )

        self._pool_min: int = int(os.getenv("PG_POOL_MIN", "2"))
        self._pool_max: int = int(os.getenv("PG_POOL_MAX", "10"))

        delays = [2, 4, 8]
        last_exc: Exception = RuntimeError("Pool initialisation never attempted.")
        for attempt, delay in enumerate(delays, start=1):
            try:
                self._pool: psycopg2.pool.ThreadedConnectionPool = (
                    psycopg2.pool.ThreadedConnectionPool(
                        self._pool_min,
                        self._pool_max,
                        dsn,
                    )
                )
                logger.info(
                    "SupabasePostgresClient pool created (min=%d, max=%d).",
                    self._pool_min,
                    self._pool_max,
                )
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "Supabase pool init attempt %d/3 failed: %s. Retrying in %ds…",
                    attempt,
                    exc,
                    delay,
                )
                time.sleep(delay)

        logger.critical(
            "SupabasePostgresClient could not initialise pool after 3 attempts: %s",
            last_exc,
        )
        raise RuntimeError(
            f"SupabasePostgresClient pool initialisation failed: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Pool access helpers
    # ------------------------------------------------------------------

    def get_connection(self) -> psycopg2.extensions.connection:
        """Borrow a connection from the pool.

        Returns:
            An open psycopg2 connection object.

        Raises:
            psycopg2.pool.PoolError: If no connection is available.
        """
        try:
            conn = self._pool.getconn()
            return conn
        except Exception as exc:
            logger.error("Failed to get connection from Supabase pool: %s", exc)
            raise

    def release_connection(self, conn: psycopg2.extensions.connection) -> None:
        """Return a connection to the pool.

        Fails soft — logs a warning if the return fails but never raises.

        Args:
            conn: The psycopg2 connection to return.
        """
        try:
            self._pool.putconn(conn)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to release Supabase connection back to pool: %s", exc
            )

    # ------------------------------------------------------------------
    # CRUD helpers (all fail soft)
    # ------------------------------------------------------------------

    def execute_query(
        self, query: str, params: Optional[tuple] = None
    ) -> list[dict[str, Any]]:
        """Execute a SELECT query and return all rows as a list of dicts.

        Acquires a connection from the pool, executes the query, fetches all
        results, and releases the connection. On any error the connection is
        rolled back, released, and an empty list is returned — never raises.

        Args:
            query: SQL SELECT statement to execute.
            params: Optional tuple of query parameters for parameterised queries.

        Returns:
            A list of dicts keyed by column name, or an empty list on error.
        """
        conn: Optional[psycopg2.extensions.connection] = None
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                return [dict(row) for row in rows]
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "execute_query failed [%.100s]: %s", query, exc
            )
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:  # noqa: BLE001
                    pass
            return []
        finally:
            if conn is not None:
                self.release_connection(conn)

    def execute_write(
        self, query: str, params: Optional[tuple] = None
    ) -> bool:
        """Execute a single INSERT/UPDATE/DELETE statement.

        Acquires a connection, executes the statement, commits, and releases.
        On any error: rolls back, releases, and returns False — never raises.

        Args:
            query: SQL write statement to execute.
            params: Optional tuple of query parameters.

        Returns:
            True on success, False on failure.
        """
        conn: Optional[psycopg2.extensions.connection] = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                cur.execute(query, params)
            conn.commit()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "execute_write failed [%.100s]: %s", query, exc
            )
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:  # noqa: BLE001
                    pass
            return False
        finally:
            if conn is not None:
                self.release_connection(conn)

    def execute_write_many(
        self, query: str, params_list: list[tuple]
    ) -> bool:
        """Execute a batch write using executemany.

        Useful for bulk-inserting job listings or audit log rows. On any error:
        rolls back, releases, and returns False — never raises.

        Args:
            query: Parameterised SQL write statement.
            params_list: List of parameter tuples, one per row.

        Returns:
            True on success, False on failure.
        """
        conn: Optional[psycopg2.extensions.connection] = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                cur.executemany(query, params_list)
            conn.commit()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "execute_write_many failed [%.100s]: %s", query, exc
            )
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:  # noqa: BLE001
                    pass
            return False
        finally:
            if conn is not None:
                self.release_connection(conn)

    def execute_transaction(
        self, queries: list[tuple[str, Optional[tuple]]]
    ) -> bool:
        """Execute multiple statements in a single atomic transaction.

        All statements are executed inside one transaction. If any statement
        raises an exception the whole transaction is rolled back and False is
        returned. Used for multi-table atomic writes such as inserting a job
        row and its corresponding audit_log row simultaneously.

        Args:
            queries: Sequence of (sql, params) tuples to execute in order.

        Returns:
            True if all statements committed successfully, False otherwise.
        """
        conn: Optional[psycopg2.extensions.connection] = None
        try:
            conn = self.get_connection()
            conn.autocommit = False
            with conn.cursor() as cur:
                for idx, (query, params) in enumerate(queries):
                    try:
                        cur.execute(query, params)
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "execute_transaction failed at query index %d "
                            "[%.100s]: %s",
                            idx,
                            query,
                            exc,
                        )
                        conn.rollback()
                        return False
            conn.commit()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("execute_transaction outer error: %s", exc)
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:  # noqa: BLE001
                    pass
            return False
        finally:
            if conn is not None:
                conn.autocommit = True
                self.release_connection(conn)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Run a lightweight connectivity check against Supabase Postgres.

        Executes ``SELECT version(), current_database(), pg_is_in_recovery()``
        and returns a structured dict. Never raises — returns status "unhealthy"
        with an error message if anything goes wrong. Includes ``mode: "supabase"``
        to distinguish from the local client health response.

        Returns:
            A dict with keys:
                status (str): "healthy" or "unhealthy".
                mode (str): Always "supabase".
                version (str | None): PostgreSQL server version string.
                database (str | None): Current database name.
                is_replica (bool | None): True if the server is a hot-standby.
                pool_min (int): Configured minimum pool size.
                pool_max (int): Configured maximum pool size.
                error (str | None): Error message if status is "unhealthy".
        """
        try:
            rows = self.execute_query(
                "SELECT version(), current_database(), pg_is_in_recovery();"
            )
            if not rows:
                return {
                    "status": "unhealthy",
                    "mode": "supabase",
                    "version": None,
                    "database": None,
                    "is_replica": None,
                    "pool_min": self._pool_min,
                    "pool_max": self._pool_max,
                    "error": "Health query returned no rows.",
                }
            row = rows[0]
            return {
                "status": "healthy",
                "mode": "supabase",
                "version": row.get("version"),
                "database": row.get("current_database"),
                "is_replica": row.get("pg_is_in_recovery"),
                "pool_min": self._pool_min,
                "pool_max": self._pool_max,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "unhealthy",
                "mode": "supabase",
                "version": None,
                "database": None,
                "is_replica": None,
                "pool_min": self._pool_min,
                "pool_max": self._pool_max,
                "error": str(exc),
            }

    def close(self) -> None:
        """Close all connections in the pool.

        Should be called during application shutdown to release database
        resources cleanly.
        """
        self._pool.closeall()
        logger.info("SupabasePostgresClient pool closed.")
