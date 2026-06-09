import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from supply_chain.config import get_settings
from supply_chain.database.pool import db
from supply_chain.ingestion.pipeline import _chunk_text, _embed_batch, _sha256
from supply_chain.vector_store.chroma import Chunk, vector_store
import uuid

logger = structlog.get_logger(__name__)

# ── Default search queries for supply chain risk domains ──────────────────────
DEFAULT_QUERIES = [
    "supply chain disruption",
    "port congestion shipping delay",
    "semiconductor shortage",
    "factory shutdown strike",
    "sanctions trade restriction",
    "logistics freight crisis",
    "supplier bankruptcy insolvency",
    "natural disaster factory flood earthquake",
]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _fetch_articles(
    client: httpx.AsyncClient,
    query: str,
    from_date: str,
    page_size: int = 20,
) -> list[dict[str, Any]]:
    """
    Call NewsAPI /v2/everything and return a list of article dicts.
    Retries up to 3 times with exponential backoff on transient failures.
    """
    settings = get_settings()
    response = await client.get(
        f"{settings.news_api_base_url}/everything",
        params={
            "q": query,
            "from": from_date,
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": page_size,
            "apiKey": settings.news_api_key,
        },
        timeout=15.0,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "ok":
        raise ValueError(f"NewsAPI error: {data.get('message', 'unknown')}")

    return data.get("articles", [])


def _article_to_text(article: dict[str, Any]) -> str:
    """
    Combine title + description + content into a single text block.
    NewsAPI truncates `content` at ~200 chars on free tier — description
    often has more signal, so we include both.
    """
    parts = [
        article.get("title") or "",
        article.get("description") or "",
        article.get("content") or "",
    ]
    return "\n\n".join(p.strip() for p in parts if p.strip())


async def _ingest_article(
    article: dict[str, Any],
    openai_client: Any,
    extra_metadata: dict[str, Any],
) -> str | None:
    """
    Ingest a single news article. Returns doc_id or None if skipped (duplicate).
    """
    from openai import AsyncOpenAI  # local import to avoid circular

    text = _article_to_text(article)
    if not text or len(text) < 50:
        return None  # skip near-empty articles

    content_hash = _sha256(text)

    # Dedup check
    async with db.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM documents WHERE raw_text_hash = $1", content_hash
        )
    if existing:
        return None

    # Chunk + embed
    raw_chunks = _chunk_text(text)
    embeddings = await _embed_batch(raw_chunks, openai_client)

    doc_id = str(uuid.uuid4())
    published_at = article.get("publishedAt", "")
    source_name = (article.get("source") or {}).get("name", "unknown")
    source_url = article.get("url", "")
    title = article.get("title", "Untitled")

    chunk_metadata: dict[str, Any] = {
        "doc_id": doc_id,
        "doc_type": "news",
        "source_url": source_url,
        "source_name": source_name,
        "published_at": published_at,
        **extra_metadata,
    }

    chunks = [
        Chunk(
            id=f"{doc_id}_{i}",
            text=chunk_text,
            embedding=emb,
            metadata={**chunk_metadata, "chunk_index": i},
        )
        for i, (chunk_text, emb) in enumerate(zip(raw_chunks, embeddings))
    ]
    await vector_store.upsert(chunks)

    # Record in Postgres
    async with db.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO documents (id, source_url, title, doc_type,
                                   supplier_id, country, raw_text_hash, chunk_count)
            VALUES ($1, $2, $3, 'news', $4::uuid, $5, $6, $7)
            """,
            doc_id,
            source_url,
            title,
            extra_metadata.get("supplier_id"),
            extra_metadata.get("country"),
            content_hash,
            len(chunks),
        )

    logger.info("news_article_ingested", doc_id=doc_id, title=title, source=source_name)
    return doc_id


async def fetch_and_ingest_news(
    query: str,
    max_articles: int = 20,
    days_back: int = 7,
    supplier_id: str | None = None,
    country: str | None = None,
) -> list[str]:
    """
    Fetch news articles for a query and ingest them.
    Returns list of newly created doc_ids (duplicates are skipped).
    """
    from openai import AsyncOpenAI
    settings = get_settings()
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

    extra_metadata: dict[str, Any] = {}
    if supplier_id:
        extra_metadata["supplier_id"] = supplier_id
    if country:
        extra_metadata["country"] = country

    async with httpx.AsyncClient() as client:
        articles = await _fetch_articles(client, query, from_date, page_size=max_articles)

    logger.info("news_fetched", query=query, article_count=len(articles))

    # Ingest concurrently — cap at 5 parallel to avoid hammering OpenAI
    semaphore = asyncio.Semaphore(5)

    async def _guarded_ingest(article: dict[str, Any]) -> str | None:
        async with semaphore:
            try:
                return await _ingest_article(article, openai_client, extra_metadata)
            except Exception as e:
                logger.warning("news_ingest_failed", title=article.get("title"), error=str(e))
                return None

    results = await asyncio.gather(*[_guarded_ingest(a) for a in articles])
    doc_ids = [r for r in results if r is not None]

    logger.info("news_ingest_complete", query=query, ingested=len(doc_ids), skipped=len(articles) - len(doc_ids))
    return doc_ids


async def fetch_and_ingest_default_queries(days_back: int = 3) -> dict[str, list[str]]:
    """
    Run all DEFAULT_QUERIES concurrently. Useful for a scheduled daily refresh.
    Returns a dict of {query: [doc_ids]}.
    """
    async def _run(query: str) -> tuple[str, list[str]]:
        doc_ids = await fetch_and_ingest_news(query, max_articles=10, days_back=days_back)
        return query, doc_ids

    results = await asyncio.gather(*[_run(q) for q in DEFAULT_QUERIES])
    return dict(results)
