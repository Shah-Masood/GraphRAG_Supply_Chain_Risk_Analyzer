import asyncio

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from supply_chain.ingestion.news import fetch_and_ingest_default_queries, fetch_and_ingest_news

logger = structlog.get_logger(__name__)


async def _refresh_default_queries() -> None:
    """Ingest the last 24h of news for all default risk queries."""
    logger.info("scheduler_job_start", job="refresh_default_queries")
    try:
        results = await fetch_and_ingest_default_queries(days_back=1)
        total = sum(len(ids) for ids in results.values())
        logger.info("scheduler_job_done", job="refresh_default_queries", total_ingested=total)
    except Exception as e:
        logger.error("scheduler_job_failed", job="refresh_default_queries", error=str(e))


async def _refresh_supplier_news() -> None:
    """
    Fetch news for every supplier in the database.
    Runs once daily — pulls 3 days back so weekend gaps are covered.
    """
    from supply_chain.database.pool import db

    logger.info("scheduler_job_start", job="refresh_supplier_news")
    try:
        async with db.acquire() as conn:
            suppliers = await conn.fetch(
                "SELECT id, name, country FROM suppliers ORDER BY name"
            )

        if not suppliers:
            logger.info("scheduler_job_skip", job="refresh_supplier_news", reason="no suppliers in db")
            return

        # Run supplier news fetches with a semaphore to avoid hammering APIs
        semaphore = asyncio.Semaphore(3)

        async def _fetch_one(supplier: dict) -> None:
            async with semaphore:
                query = f"{supplier['name']} supply chain risk"
                try:
                    doc_ids = await fetch_and_ingest_news(
                        query=query,
                        max_articles=5,
                        days_back=3,
                        supplier_id=str(supplier["id"]),
                        country=supplier["country"],
                    )
                    if doc_ids:
                        logger.info(
                            "supplier_news_refreshed",
                            supplier=supplier["name"],
                            new_docs=len(doc_ids),
                        )
                except Exception as e:
                    logger.warning(
                        "supplier_news_refresh_failed",
                        supplier=supplier["name"],
                        error=str(e),
                    )

        await asyncio.gather(*[_fetch_one(s) for s in suppliers])
        logger.info("scheduler_job_done", job="refresh_supplier_news", supplier_count=len(suppliers))

    except Exception as e:
        logger.error("scheduler_job_failed", job="refresh_supplier_news", error=str(e))


def start_scheduler() -> AsyncIOScheduler:
    """
    Create, configure, and start the scheduler.
    Returns the scheduler instance so lifespan can shut it down cleanly.
    """
    scheduler = AsyncIOScheduler()

    # Default queries every 6 hours
    scheduler.add_job(
        _refresh_default_queries,
        trigger=IntervalTrigger(hours=6),
        id="refresh_default_queries",
        name="Refresh default supply chain risk queries",
        replace_existing=True,
        misfire_grace_time=300,  # allow 5 min late start
    )

    # Supplier-specific news once daily at 6am
    scheduler.add_job(
        _refresh_supplier_news,
        trigger=CronTrigger(hour=6, minute=0),
        id="refresh_supplier_news",
        name="Refresh per-supplier news",
        replace_existing=True,
        misfire_grace_time=600,
    )

    scheduler.start()
    logger.info(
        "scheduler_started",
        jobs=[j.id for j in scheduler.get_jobs()],
    )
    return scheduler


def stop_scheduler(scheduler: AsyncIOScheduler) -> None:
    scheduler.shutdown(wait=False)
    logger.info("scheduler_stopped")
