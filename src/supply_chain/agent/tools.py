"""
All tools available to the supply chain risk agent.

Tools:
    retrieve_supply_chain_docs      — semantic search over ingested docs (RAG)
    traverse_supply_chain_graph     — graph traversal: dependencies, impact, paths (GraphRAG)
    fetch_news_for_supplier         — live NewsAPI fetch for a specific supplier/topic
    query_supplier_db               — structured supplier + risk data from Postgres
    calculate_risk_score            — compute + persist a risk score for a supplier

The agent imports ALL_TOOLS and registers them on the graph.
"""

from typing import Any

import structlog
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from supply_chain.config import get_settings
from supply_chain.database.pool import db
from supply_chain.ingestion.news import fetch_and_ingest_news
from supply_chain.vector_store.retrieval import retrieve_supply_chain_docs  # RAG tool
from supply_chain.graph.retrieval import traverse_supply_chain_graph        # GraphRAG tool

logger = structlog.get_logger(__name__)


# ── Tool: fetch_news_for_supplier ──────────────────────────────────────────────

class FetchNewsInput(BaseModel):
    topic: str = Field(..., description="Search topic, e.g. 'TSMC factory shutdown Taiwan'")
    supplier_id: str | None = Field(default=None, description="Supplier UUID to tag ingested articles")
    country: str | None = Field(default=None, description="Country to filter/tag news")
    max_articles: int = Field(default=10, ge=1, le=50)
    days_back: int = Field(default=7, ge=1, le=30)


@tool("fetch_news_for_supplier", args_schema=FetchNewsInput)
async def fetch_news_for_supplier(
    topic: str,
    supplier_id: str | None = None,
    country: str | None = None,
    max_articles: int = 10,
    days_back: int = 7,
) -> str:
    """
    Fetch and ingest the latest news articles about a supplier or supply chain topic.
    Use this when you need fresh news that may not be in the knowledge base yet.
    After fetching, use retrieve_supply_chain_docs to search the newly ingested content.
    """
    doc_ids = await fetch_and_ingest_news(
        query=topic,
        max_articles=max_articles,
        days_back=days_back,
        supplier_id=supplier_id,
        country=country,
    )
    if not doc_ids:
        return f"No new articles found for '{topic}' (may already be in knowledge base)."
    return f"Fetched and ingested {len(doc_ids)} new articles about '{topic}'. Now use retrieve_supply_chain_docs to search this content."


# ── Tool: query_supplier_db ────────────────────────────────────────────────────

class SupplierQueryInput(BaseModel):
    supplier_name: str | None = Field(default=None, description="Partial supplier name to search")
    country: str | None = Field(default=None, description="Filter by country")
    tier: int | None = Field(default=None, description="Filter by tier (1=direct, 2=sub-supplier)")
    include_risk_scores: bool = Field(default=True, description="Include latest risk scores")


@tool("query_supplier_db", args_schema=SupplierQueryInput)
async def query_supplier_db(
    supplier_name: str | None = None,
    country: str | None = None,
    tier: int | None = None,
    include_risk_scores: bool = True,
) -> str:
    """
    Query the supplier database for structured information about suppliers.
    Returns supplier profiles, locations, tiers, and their latest risk scores.
    Use this to get an overview of which suppliers exist and their current risk status.
    """
    conditions: list[str] = []
    params: list[Any] = []
    i = 1

    if supplier_name:
        conditions.append(f"s.name ILIKE ${i}")
        params.append(f"%{supplier_name}%")
        i += 1
    if country:
        conditions.append(f"s.country ILIKE ${i}")
        params.append(f"%{country}%")
        i += 1
    if tier:
        conditions.append(f"s.tier = ${i}")
        params.append(tier)
        i += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    if include_risk_scores:
        query = f"""
            SELECT s.id, s.name, s.country, s.region, s.tier, s.industry,
                   lrs.overall_score, lrs.scored_at
            FROM suppliers s
            LEFT JOIN latest_risk_scores lrs ON lrs.supplier_id = s.id
            {where}
            ORDER BY lrs.overall_score DESC NULLS LAST
            LIMIT 20
        """
    else:
        query = f"""
            SELECT id, name, country, region, tier, industry
            FROM suppliers s
            {where}
            ORDER BY name
            LIMIT 20
        """

    async with db.acquire() as conn:
        rows = await conn.fetch(query, *params)

    if not rows:
        return "No suppliers found matching those criteria."

    lines = ["Suppliers found:\n"]
    for r in rows:
        score_str = f" | Risk score: {r['overall_score']:.1f}/100" if include_risk_scores and r.get("overall_score") else " | Risk score: not assessed"
        lines.append(f"- {r['name']} ({r['country']}, Tier {r['tier']}){score_str}")
    return "\n".join(lines)


# ── Tool: calculate_risk_score ─────────────────────────────────────────────────

class RiskScoreInput(BaseModel):
    supplier_id: str = Field(..., description="UUID of the supplier to score")
    reasoning: str = Field(..., description="Your analysis and reasoning for the scores")
    overall_score: float = Field(..., ge=0, le=100, description="Overall risk score 0-100 (100=highest risk)")
    geopolitical: float | None = Field(default=None, ge=0, le=100)
    financial: float | None = Field(default=None, ge=0, le=100)
    logistics: float | None = Field(default=None, ge=0, le=100)
    environmental: float | None = Field(default=None, ge=0, le=100)
    regulatory: float | None = Field(default=None, ge=0, le=100)
    supplier_health: float | None = Field(default=None, ge=0, le=100)
    cyber: float | None = Field(default=None, ge=0, le=100)


@tool("calculate_risk_score", args_schema=RiskScoreInput)
async def calculate_risk_score(
    supplier_id: str,
    reasoning: str,
    overall_score: float,
    geopolitical: float | None = None,
    financial: float | None = None,
    logistics: float | None = None,
    environmental: float | None = None,
    regulatory: float | None = None,
    supplier_health: float | None = None,
    cyber: float | None = None,
) -> str:
    """
    Compute and persist a risk score for a supplier based on your analysis.
    Call this after you've retrieved relevant documents and traversed the graph.
    Scores are 0-100 where 100 = maximum risk.

    Guidelines: 0-25 Low | 26-50 Medium | 51-75 High | 76-100 Critical
    """
    settings = get_settings()
    async with db.transaction() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO risk_scores (
                supplier_id, overall_score, geopolitical, financial,
                logistics, environmental, regulatory, supplier_health,
                cyber, reasoning, model_version
            )
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING id, scored_at
            """,
            supplier_id, overall_score, geopolitical, financial,
            logistics, environmental, regulatory, supplier_health,
            cyber, reasoning, settings.openai_chat_model,
        )

    level = (
        "LOW" if overall_score <= 25 else
        "MEDIUM" if overall_score <= 50 else
        "HIGH" if overall_score <= 75 else
        "CRITICAL"
    )

    logger.info("risk_score_saved", supplier_id=supplier_id, score=overall_score, level=level)
    return (
        f"Risk score saved (id: {row['id']}).\n"
        f"Supplier {supplier_id}: {level} risk ({overall_score:.1f}/100)\n"
        f"Scored at: {row['scored_at'].isoformat()}"
    )


# ── Tool registry ──────────────────────────────────────────────────────────────
ALL_TOOLS = [
    retrieve_supply_chain_docs,     # Vector RAG
    traverse_supply_chain_graph,    # Graph RAG  ← NEW
    fetch_news_for_supplier,
    query_supplier_db,
    calculate_risk_score,
]
