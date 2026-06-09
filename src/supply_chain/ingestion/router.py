"""
Ingestion API router.

Endpoints:
    POST /ingest/news       — fetch + ingest news articles for a query
    POST /ingest/news/bulk  — run all default supply chain risk queries
    POST /ingest/document   — ingest an uploaded PDF or text file
    GET  /ingest/stats      — document + chunk counts
"""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field

import structlog
import tempfile
from pathlib import Path

from supply_chain.ingestion.news import fetch_and_ingest_news, fetch_and_ingest_default_queries
from supply_chain.ingestion.pipeline import ingest_document
from supply_chain.database.pool import db
from supply_chain.vector_store.chroma import vector_store

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])


# ── Request / Response models ──────────────────────────────────────────────────

class NewsIngestRequest(BaseModel):
    query: str = Field(..., description="Search query, e.g. 'port strike Taiwan'")
    max_articles: int = Field(default=20, ge=1, le=100)
    days_back: int = Field(default=7, ge=1, le=30)
    supplier_id: str | None = Field(default=None, description="Tag results to a supplier UUID")
    country: str | None = Field(default=None, description="Tag results with a country")


class IngestResponse(BaseModel):
    doc_ids: list[str]
    ingested: int
    message: str


class IngestStatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    documents_by_type: dict[str, int]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/news", response_model=IngestResponse)
async def ingest_news(body: NewsIngestRequest) -> IngestResponse:
    """
    Fetch news articles from NewsAPI for the given query and ingest them.
    Duplicate articles (by content hash) are silently skipped.
    """
    try:
        doc_ids = await fetch_and_ingest_news(
            query=body.query,
            max_articles=body.max_articles,
            days_back=body.days_back,
            supplier_id=body.supplier_id,
            country=body.country,
        )
    except Exception as e:
        logger.error("news_ingest_endpoint_error", error=str(e))
        raise HTTPException(status_code=502, detail=f"News fetch failed: {e}")

    return IngestResponse(
        doc_ids=doc_ids,
        ingested=len(doc_ids),
        message=f"Ingested {len(doc_ids)} new articles for query '{body.query}'.",
    )


@router.post("/news/bulk", response_model=dict[str, IngestResponse])
async def ingest_news_bulk(
    background_tasks: BackgroundTasks,
    days_back: Annotated[int, Query(ge=1, le=7)] = 3,
) -> dict[str, IngestResponse]:
    """
    Run all default supply chain risk queries (port strikes, shortages, etc.).
    Runs in the background — returns immediately with a job confirmation,
    then fires off all queries concurrently.

    Since this can take 30-60s, it's kicked off as a background task.
    """
    async def _run_bulk() -> None:
        results = await fetch_and_ingest_default_queries(days_back=days_back)
        total = sum(len(ids) for ids in results.values())
        logger.info("bulk_ingest_complete", total_ingested=total, queries=len(results))

    background_tasks.add_task(_run_bulk)
    return {
        "status": IngestResponse(
            doc_ids=[],
            ingested=0,
            message=f"Bulk ingestion started in background (last {days_back} days). Check /ingest/stats for progress.",
        )
    }


@router.post("/document", response_model=IngestResponse)
async def ingest_uploaded_document(
    file: Annotated[UploadFile, File(description="PDF or .txt file")],
    supplier_id: str | None = Query(default=None),
    country: str | None = Query(default=None),
    title: str | None = Query(default=None),
) -> IngestResponse:
    """
    Upload and ingest a PDF or plain text document.
    The file is written to a temp path, ingested, then cleaned up.
    """
    if file.content_type not in ("application/pdf", "text/plain"):
        raise HTTPException(
            status_code=415,
            detail="Only PDF and plain text files are supported.",
        )

    doc_type = "pdf" if file.content_type == "application/pdf" else "text"
    suffix = ".pdf" if doc_type == "pdf" else ".txt"

    # Write to temp file — pipeline expects a Path
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        doc_id = await ingest_document(
            source=tmp_path,
            doc_type=doc_type,
            title=title or file.filename,
            supplier_id=supplier_id,
            country=country,
            source_url=file.filename,
        )
    except Exception as e:
        logger.error("document_ingest_error", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        doc_ids=[doc_id],
        ingested=1,
        message=f"Document '{file.filename}' ingested successfully.",
    )


@router.get("/stats", response_model=IngestStatsResponse)
async def ingest_stats() -> IngestStatsResponse:
    """
    Returns total document count (Postgres) and chunk count (ChromaDB).
    """
    async with db.acquire() as conn:
        total_docs = await conn.fetchval("SELECT COUNT(*) FROM documents")
        rows = await conn.fetch(
            "SELECT doc_type, COUNT(*) AS cnt FROM documents GROUP BY doc_type"
        )

    by_type = {row["doc_type"]: row["cnt"] for row in rows}
    total_chunks = await vector_store.count()

    return IngestStatsResponse(
        total_documents=total_docs,
        total_chunks=total_chunks,
        documents_by_type=by_type,
    )
