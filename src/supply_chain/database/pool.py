from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
import structlog

from supply_chain.config import get_settings

logger = structlog.get_logger(__name__)


class Database:
    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create the connection pool. Called once at app startup."""
        settings = get_settings()
        self._pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
            # Fail fast on bad connections rather than silently returning broken ones
            command_timeout=30,
        )
        logger.info(
            "db_pool_created",
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
        )

    async def disconnect(self) -> None:
        """Drain and close the pool. Called once at app shutdown."""
        if self._pool:
            await self._pool.close()
            logger.info("db_pool_closed")

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Borrow a connection from the pool for the duration of the block."""
        if self._pool is None:
            raise RuntimeError("Database.connect() has not been called yet.")
        async with self._pool.acquire() as conn:
            yield conn

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Borrow a connection and wrap the block in a transaction."""
        async with self.acquire() as conn:
            async with conn.transaction():
                yield conn


# Module-level singleton — import this everywhere
db = Database()
