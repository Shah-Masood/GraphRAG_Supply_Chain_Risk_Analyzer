import hashlib
import json
from typing import Any

import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from supply_chain.config import get_settings

logger = structlog.get_logger(__name__)

# ── Pydantic models for structured output ─────────────────────────────────────

class ExtractedNode(BaseModel):
    label: str = Field(..., description="One of: Company, Country, Port, Product, RiskEvent, Regulation")
    name: str = Field(..., description="Canonical name of the entity")
    properties: dict[str, Any] = Field(default_factory=dict, description="Extra properties: country, industry, level, category, etc.")


class ExtractedRelationship(BaseModel):
    source: str = Field(..., description="Name of the source entity")
    source_label: str = Field(..., description="Label of the source entity")
    relation: str = Field(..., description="Relationship type: SUPPLIES, DEPENDS_ON, PRODUCES, LOCATED_IN, SHIPS_THROUGH, AFFECTED_BY, AFFECTS, RESTRICTS, TARGETS, HAS_PORT")
    target: str = Field(..., description="Name of the target entity")
    target_label: str = Field(..., description="Label of the target entity")
    properties: dict[str, Any] = Field(default_factory=dict, description="Extra properties: since, weight, notes")


class ExtractionResult(BaseModel):
    nodes: list[ExtractedNode] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


EXTRACTION_SYSTEM_PROMPT = """You are an expert supply chain knowledge graph builder.

Extract entities and relationships from supply chain text for a Neo4j knowledge graph.

NODES to extract:
- Company: any business, manufacturer, supplier, logistics provider, tech firm
- Country: nations, territories (use full name: "United States" not "US")
- Port: seaports, airports, land crossings (e.g. "Port of Shanghai", "Kaohsiung Port")
- Product: semiconductors, raw materials, components, commodities (e.g. "DRAM", "rare earth oxides", "EUV machines")
- RiskEvent: disruptions, strikes, sanctions, natural disasters, shortages, cyber attacks
- Regulation: laws, bans, export controls, tariffs, trade restrictions

RELATIONSHIPS to extract:
- SUPPLIES: Company → Company (supplier relationship)
- DEPENDS_ON: Company → Product (company needs this product)
- PRODUCES: Company → Product (company makes this)
- LOCATED_IN: Company → Country (headquarters or major operations)
- SHIPS_THROUGH: Company → Port (uses this port)
- AFFECTED_BY: Company → RiskEvent (company impacted)
- AFFECTS: RiskEvent → Country or Company (who is hit)
- RESTRICTS: Regulation → Product (what is banned/limited)
- TARGETS: Regulation → Country (which country it applies to)
- HAS_PORT: Country → Port

Rules:
- Only extract what is explicitly stated or strongly implied
- Use canonical names (full company names, full country names)
- Be specific: "TSMC" not "the company", "Taiwan" not "the island"
- For RiskEvent names, be concise: "Pentagon rare earth ban 2027", "Taiwan Strait tension"
- Skip vague/generic entities
- Return empty lists if nothing is clearly extractable
"""


async def extract_from_chunk(text: str, doc_metadata: dict[str, Any]) -> ExtractionResult:
    """
    Extract entities and relationships from a single text chunk.
    Uses GPT-4o with JSON structured output.
    """
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    # Add doc context to help the LLM
    context = ""
    if doc_metadata.get("source_name"):
        context += f"Source: {doc_metadata['source_name']}\n"
    if doc_metadata.get("country"):
        context += f"Geographic context: {doc_metadata['country']}\n"

    prompt = f"{context}\nText to analyze:\n{text}"

    try:
        response = await client.beta.chat.completions.parse(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format=ExtractionResult,
            temperature=0,
            max_tokens=2000,
        )
        result = response.choices[0].message.parsed
        if result is None:
            return ExtractionResult()

        logger.debug(
            "extraction_complete",
            nodes=len(result.nodes),
            relationships=len(result.relationships),
            text_preview=text[:60],
        )
        return result

    except Exception as e:
        logger.warning("extraction_failed", error=str(e), text_preview=text[:60])
        return ExtractionResult()


async def extract_from_chunks(
    chunks: list[str],
    doc_metadata: dict[str, Any],
    max_chunks: int = 10,
) -> ExtractionResult:
    """
    Extract from multiple chunks, merge results, deduplicate nodes.
    We cap at max_chunks to control cost — first N chunks usually have the most entities.
    """
    import asyncio

    # Only process first max_chunks — later chunks are usually less entity-dense
    selected = chunks[:max_chunks]

    results = await asyncio.gather(
        *[extract_from_chunk(chunk, doc_metadata) for chunk in selected],
        return_exceptions=True,
    )

    # Merge all results, deduplicate by (label, name)
    seen_nodes: set[tuple[str, str]] = set()
    seen_rels: set[tuple[str, str, str]] = set()
    merged = ExtractionResult()

    for result in results:
        if isinstance(result, Exception):
            continue
        for node in result.nodes:
            key = (node.label, node.name.lower())
            if key not in seen_nodes:
                seen_nodes.add(key)
                merged.nodes.append(node)
        for rel in result.relationships:
            key = (rel.source.lower(), rel.relation, rel.target.lower())
            if key not in seen_rels:
                seen_rels.add(key)
                merged.relationships.append(rel)

    logger.info(
        "extraction_merged",
        chunks_processed=len(selected),
        total_nodes=len(merged.nodes),
        total_relationships=len(merged.relationships),
    )
    return merged
