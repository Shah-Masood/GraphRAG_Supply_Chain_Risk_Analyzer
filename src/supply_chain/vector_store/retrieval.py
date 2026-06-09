"""
Retrieval tool — the bridge between the agent and ChromaDB.

This module exposes two things:
  1. `retrieve()` — the raw async function (used directly in tests / scripts)
  2. `retrieve_supply_chain_docs` — a LangChain @tool wrapper the agent calls

The agent in Stage 3 imports `retrieve_supply_chain_docs` and adds it to its
tool list. It will call it like:
    retrieve_supply_chain_docs.invoke({"query": "port delays Taiwan", "n_results": 5})
"""

from typing import Any

import structlog
from langchain_core.tools import tool
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from supply_chain.config import get_settings
from supply_chain.vector_store.chroma import RetrievedChunk, vector_store

logger = structlog.get_logger(__name__)


# ── Input schema (used by LangGraph for structured tool calling) ───────────────

class RetrievalInput(BaseModel):
    query: str = Field(..., description="Natural language query about supply chain risks")
    n_results: int = Field(default=5, ge=1, le=20, description="Number of chunks to return")
    supplier_id: str | None = Field(
        default=None,
        description="Filter results to a specific supplier UUID",
    )
    country: str | None = Field(
        default=None,
        description="Filter results to a specific country, e.g. 'Taiwan'",
    )
    doc_type: str | None = Field(
        default=None,
        description="Filter by document type: 'news', 'pdf', 'report'",
    )


class RetrievalResult(BaseModel):
    chunk_id: str
    text: str
    source_url: str | None
    source_name: str | None
    doc_type: str
    supplier_id: str | None
    country: str | None
    published_at: str | None
    similarity_score: float  # 1 - distance (higher = more relevant)


async def _embed_query(query: str) -> list[float]:
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.embeddings.create(
        input=[query],
        model=settings.openai_embedding_model,
    )
    return response.data[0].embedding


def _build_where_filter(
    supplier_id: str | None,
    country: str | None,
    doc_type: str | None,
) -> dict[str, Any] | None:
    """
    Build a ChromaDB metadata filter. Returns None if no filters set.
    Combines multiple filters with $and.
    """
    conditions: list[dict[str, Any]] = []

    if supplier_id:
        conditions.append({"supplier_id": supplier_id})
    if country:
        conditions.append({"country": country})
    if doc_type:
        conditions.append({"doc_type": doc_type})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


async def retrieve(
    query: str,
    n_results: int = 5,
    supplier_id: str | None = None,
    country: str | None = None,
    doc_type: str | None = None,
) -> list[RetrievalResult]:
    """
    Core retrieval function. Embeds the query, searches ChromaDB,
    and returns ranked results with metadata.
    """
    query_embedding = await _embed_query(query)
    where = _build_where_filter(supplier_id, country, doc_type)

    raw_chunks: list[RetrievedChunk] = await vector_store.query(
        query_embedding=query_embedding,
        n_results=n_results,
        where=where,
    )

    results = [
        RetrievalResult(
            chunk_id=chunk.id,
            text=chunk.text,
            source_url=chunk.metadata.get("source_url"),
            source_name=chunk.metadata.get("source_name"),
            doc_type=chunk.metadata.get("doc_type", "unknown"),
            supplier_id=chunk.metadata.get("supplier_id"),
            country=chunk.metadata.get("country"),
            published_at=chunk.metadata.get("published_at"),
            # cosine distance: 0 = identical, 2 = opposite → convert to similarity
            similarity_score=round(1 - chunk.distance, 4),
        )
        for chunk in raw_chunks
    ]

    logger.info(
        "retrieval_complete",
        query=query[:80],
        n_results=len(results),
        top_score=results[0].similarity_score if results else None,
    )
    return results


def _format_results_for_agent(results: list[RetrievalResult]) -> str:
    """
    Format retrieval results as a readable string the LLM can reason over.
    Each chunk includes its source and similarity score.
    """
    if not results:
        return "No relevant documents found."

    parts: list[str] = []
    for i, r in enumerate(results, 1):
        source = r.source_name or r.source_url or "Unknown source"
        date = f" ({r.published_at[:10]})" if r.published_at else ""
        country = f" | {r.country}" if r.country else ""
        score = f"relevance: {r.similarity_score:.2f}"

        parts.append(
            f"[{i}] {source}{date}{country} — {score}\n{r.text}"
        )

    return "\n\n---\n\n".join(parts)


# ── LangChain tool wrapper ─────────────────────────────────────────────────────
# Stage 3 agent imports this directly.

@tool("retrieve_supply_chain_docs", args_schema=RetrievalInput)
async def retrieve_supply_chain_docs(
    query: str,
    n_results: int = 5,
    supplier_id: str | None = None,
    country: str | None = None,
    doc_type: str | None = None,
) -> str:
    """
    Search the supply chain knowledge base for relevant risk information.

    Use this tool when you need to find:
    - News about supply chain disruptions, port delays, factory shutdowns
    - Risk events affecting specific suppliers or countries
    - Historical reports and analysis on supply chain vulnerabilities

    Returns the most relevant document chunks with their sources.
    """
    results = await retrieve(
        query=query,
        n_results=n_results,
        supplier_id=supplier_id,
        country=country,
        doc_type=doc_type,
    )
    return _format_results_for_agent(results)
