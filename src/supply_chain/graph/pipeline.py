"""
Graph ingestion pipeline.

Takes an already-ingested document (chunks in ChromaDB, metadata in Postgres)
and builds the knowledge graph in Neo4j: extract entities + relationships from
chunks, then MERGE them into the graph.

Entry point:
    from supply_chain.graph.pipeline import build_graph_from_doc

    await build_graph_from_doc(doc_id, chunks, metadata)

This is called automatically at the end of the standard ingestion pipeline.
"""

import uuid
from typing import Any

import structlog

from supply_chain.graph.extractor import ExtractionResult, extract_from_chunks
from supply_chain.graph.neo4j_client import graph_db

logger = structlog.get_logger(__name__)

# Valid node labels and relationship types — guards against LLM hallucinating bad types
VALID_LABELS = {"Company", "Country", "Port", "Product", "RiskEvent", "Regulation"}
VALID_RELATIONS = {
    "SUPPLIES", "DEPENDS_ON", "PRODUCES", "LOCATED_IN", "SHIPS_THROUGH",
    "AFFECTED_BY", "AFFECTS", "RESTRICTS", "TARGETS", "HAS_PORT",
}


def _sanitize(result: ExtractionResult) -> ExtractionResult:
    """Drop any nodes/rels with invalid labels or relation types."""
    result.nodes = [n for n in result.nodes if n.label in VALID_LABELS and n.name.strip()]
    result.relationships = [
        r for r in result.relationships
        if r.relation in VALID_RELATIONS
        and r.source.strip()
        and r.target.strip()
        and r.source_label in VALID_LABELS
        and r.target_label in VALID_LABELS
    ]
    return result


async def _upsert_nodes(nodes: list, doc_id: str) -> None:
    """MERGE nodes into Neo4j, grouped by label for efficiency."""
    by_label: dict[str, list[dict]] = {}
    for node in nodes:
        by_label.setdefault(node.label, []).append({
            "name": node.name,
            "doc_id": doc_id,
            **{k: str(v) for k, v in node.properties.items()},
        })

    for label, batch in by_label.items():
        # UNWIND for bulk MERGE — one transaction per label
        cypher = f"""
        UNWIND $batch AS props
        MERGE (n:{label} {{name: props.name}})
        SET n += props
        """
        await graph_db.write_batch(cypher, batch)


async def _upsert_relationships(rels: list, doc_id: str) -> None:
    """MERGE relationships. Groups by relation type for bulk upsert."""
    by_type: dict[str, list[dict]] = {}
    for rel in rels:
        key = f"{rel.source_label}__{rel.relation}__{rel.target_label}"
        by_type.setdefault(key, []).append({
            "source": rel.source,
            "target": rel.target,
            "doc_id": doc_id,
            **{k: str(v) for k, v in rel.properties.items()},
        })

    for type_key, batch in by_type.items():
        src_label, relation, tgt_label = type_key.split("__")
        cypher = f"""
        UNWIND $batch AS props
        MATCH (src:{src_label} {{name: props.source}})
        MATCH (tgt:{tgt_label} {{name: props.target}})
        MERGE (src)-[r:{relation}]->(tgt)
        SET r += props
        """
        try:
            await graph_db.write_batch(cypher, batch)
        except Exception as e:
            # Relationship upsert can fail if nodes weren't created (LLM hallucinated names)
            logger.warning("rel_upsert_failed", relation=relation, error=str(e)[:100])


async def build_graph_from_doc(
    doc_id: str,
    chunks: list[str],
    metadata: dict[str, Any],
) -> tuple[int, int]:
    """
    Extract entities/relationships from a document's chunks and store in Neo4j.

    Returns (nodes_created, relationships_created).
    """
    logger.info("graph_pipeline_start", doc_id=doc_id, chunk_count=len(chunks))

    result = await extract_from_chunks(chunks, metadata)
    result = _sanitize(result)

    if not result.nodes:
        logger.info("graph_pipeline_skip", doc_id=doc_id, reason="no entities extracted")
        return 0, 0

    await _upsert_nodes(result.nodes, doc_id)
    await _upsert_relationships(result.relationships, doc_id)

    # Mark document as graph-processed in Postgres
    try:
        from supply_chain.database.pool import db
        async with db.acquire() as conn:
            await conn.execute(
                "UPDATE documents SET metadata = metadata || $1::jsonb WHERE id = $2::uuid",
                '{"graph_processed": true}',
                doc_id,
            )
    except Exception as e:
        logger.warning("graph_processed_flag_failed", doc_id=doc_id, error=str(e))

    logger.info(
        "graph_pipeline_done",
        doc_id=doc_id,
        nodes=len(result.nodes),
        relationships=len(result.relationships),
    )
    return len(result.nodes), len(result.relationships)


async def seed_suppliers_to_graph() -> None:
    """
    Seed all suppliers from Postgres into Neo4j as Company nodes.
    Useful for initial graph population — run once after seed_suppliers.py.
    """
    from supply_chain.database.pool import db

    async with db.acquire() as conn:
        rows = await conn.fetch("SELECT id, name, country, region, tier, industry FROM suppliers")

    if not rows:
        logger.info("graph_seed_skip", reason="no suppliers in postgres")
        return

    batch = [
        {
            "name": r["name"],
            "country": r["country"],
            "region": r["region"] or "",
            "tier": str(r["tier"]),
            "industry": r["industry"] or "",
            "doc_id": str(r["id"]),
        }
        for r in rows
    ]

    # Create Company nodes
    await graph_db.write_batch(
        """
        UNWIND $batch AS props
        MERGE (c:Company {name: props.name})
        SET c += props
        """,
        batch,
    )

    # Create Country nodes and LOCATED_IN relationships
    await graph_db.write_batch(
        """
        UNWIND $batch AS props
        MERGE (country:Country {name: props.country})
        WITH country, props
        MATCH (c:Company {name: props.name})
        MERGE (c)-[:LOCATED_IN]->(country)
        """,
        batch,
    )

    logger.info("graph_suppliers_seeded", count=len(batch))
