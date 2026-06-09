"""
FastAPI application entry point.

Startup sequence (lifespan):
    1. Init structlog
    2. Connect asyncpg pool
    3. Connect ChromaDB async client
    4. Connect Neo4j + setup schema
    5. Set up PostgresSaver checkpointer + compile LangGraph agent
    6. Start APScheduler for background news refresh
    7. Seed suppliers to Neo4j graph (if empty)

Run locally:
    uvicorn supply_chain.main:app --reload --port 8000

Open the UI: http://localhost:8000/app
Neo4j Browser: http://localhost:7474
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from supply_chain.config import get_settings
from supply_chain.database.pool import db
from supply_chain.vector_store.chroma import vector_store
from supply_chain.graph.neo4j_client import graph_db


def _configure_logging() -> None:
    import logging
    settings = get_settings()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if not settings.is_production
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


logger = structlog.get_logger(__name__)
FRONTEND_DIR = Path(__file__).parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    _configure_logging()
    settings = get_settings()
    logger.info("app_starting", env=settings.app_env)

    # Infrastructure
    await db.connect()
    await vector_store.connect()
    await graph_db.connect()
    await graph_db.setup_schema()

    # Seed suppliers into Neo4j if the graph is empty
    try:
        count_result = await graph_db.query("MATCH (n:Company) RETURN count(n) AS cnt")
        if count_result and count_result[0]["cnt"] == 0:
            from supply_chain.graph.pipeline import seed_suppliers_to_graph
            await seed_suppliers_to_graph()
    except Exception as e:
        logger.warning("graph_seed_failed", error=str(e))

    # Agent
    from supply_chain.agent.graph import build_graph, get_checkpointer
    checkpointer, cp_pool = await get_checkpointer()
    app.state.agent_graph = build_graph(checkpointer)
    app.state.checkpointer = checkpointer
    app.state.cp_pool = cp_pool

    # Scheduler
    from supply_chain.scheduler import start_scheduler, stop_scheduler
    scheduler = start_scheduler()
    app.state.scheduler = scheduler

    logger.info("app_ready")
    yield

    logger.info("app_shutting_down")
    stop_scheduler(app.state.scheduler)
    await vector_store.disconnect()
    await graph_db.disconnect()
    if hasattr(app.state, "cp_pool"):
        await app.state.cp_pool.close()
    await db.disconnect()
    logger.info("app_stopped")


app = FastAPI(title="Supply Chain Risk Analyzer", version="0.5.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Thread-ID"],
)

from supply_chain.ingestion.router import router as ingest_router  # noqa: E402
from supply_chain.agent.router import router as agent_router        # noqa: E402

app.include_router(ingest_router)
app.include_router(agent_router)


@app.get("/app", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health", tags=["meta"])
async def health() -> JSONResponse:
    db_ok = chroma_ok = neo4j_ok = False
    agent_ok = hasattr(app.state, "agent_graph")
    scheduler_ok = hasattr(app.state, "scheduler") and app.state.scheduler.running
    count = -1
    graph_nodes = -1

    try:
        async with db.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_ok = True
    except Exception:
        pass

    try:
        count = await vector_store.count()
        chroma_ok = True
    except Exception:
        pass

    try:
        result = await graph_db.query("MATCH (n) RETURN count(n) AS cnt")
        graph_nodes = result[0]["cnt"] if result else 0
        neo4j_ok = True
    except Exception:
        pass

    all_ok = db_ok and chroma_ok and neo4j_ok and agent_ok
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={
            "status": "ok" if all_ok else "degraded",
            "postgres": "ok" if db_ok else "error",
            "chromadb": "ok" if chroma_ok else "error",
            "chroma_chunk_count": count if chroma_ok else None,
            "neo4j": "ok" if neo4j_ok else "error",
            "neo4j_node_count": graph_nodes if neo4j_ok else None,
            "agent": "ok" if agent_ok else "not_ready",
            "scheduler": "ok" if scheduler_ok else "not_ready",
        },
    )
