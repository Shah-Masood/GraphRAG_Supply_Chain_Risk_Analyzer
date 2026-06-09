from typing import Any

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver

from supply_chain.config import get_settings

logger = structlog.get_logger(__name__)


class GraphDB:
    def __init__(self) -> None:
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        settings = get_settings()
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        # Verify connectivity
        await self._driver.verify_connectivity()
        logger.info("neo4j_connected", uri=settings.neo4j_uri)

    async def disconnect(self) -> None:
        if self._driver:
            await self._driver.close()
            logger.info("neo4j_disconnected")

    def _ensure_ready(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("GraphDB.connect() has not been called yet.")
        return self._driver

    async def query(
        self, cypher: str, **params: Any
    ) -> list[dict[str, Any]]:
        """Run a read query, return list of record dicts."""
        driver = self._ensure_ready()
        async with driver.session() as session:
            result = await session.run(cypher, **params)
            records = await result.data()
            return records

    async def write(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Run a write query inside a transaction, return result records."""
        driver = self._ensure_ready()
        async with driver.session() as session:
            result = await session.run(cypher, **params)
            records = await result.data()
            return records

    async def write_batch(self, cypher: str, batch: list[dict[str, Any]]) -> None:
        """
        Run a write query for each item in batch inside a single transaction.
        More efficient than calling write() in a loop.
        Uses UNWIND for bulk upserts.
        """
        driver = self._ensure_ready()
        async with driver.session() as session:
            await session.run(cypher, batch=batch)

    async def setup_schema(self) -> None:
        """
        Create uniqueness constraints and indexes.
        Idempotent — safe to run on every startup.
        """
        constraints = [
            "CREATE CONSTRAINT company_name IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT country_name IF NOT EXISTS FOR (c:Country) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT port_name IF NOT EXISTS FOR (p:Port) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT product_name IF NOT EXISTS FOR (p:Product) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT risk_event_id IF NOT EXISTS FOR (r:RiskEvent) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT regulation_id IF NOT EXISTS FOR (r:Regulation) REQUIRE r.id IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX company_country IF NOT EXISTS FOR (c:Company) ON (c.country)",
            "CREATE INDEX company_industry IF NOT EXISTS FOR (c:Company) ON (c.industry)",
            "CREATE INDEX risk_event_category IF NOT EXISTS FOR (r:RiskEvent) ON (r.category)",
            "CREATE INDEX risk_event_level IF NOT EXISTS FOR (r:RiskEvent) ON (r.level)",
        ]
        for stmt in constraints + indexes:
            try:
                await self.write(stmt)
            except Exception as e:
                # Constraint already exists — fine
                if "already exists" not in str(e).lower():
                    logger.warning("schema_setup_warning", stmt=stmt[:60], error=str(e))

        logger.info("neo4j_schema_ready")


# Module-level singleton
graph_db = GraphDB()
