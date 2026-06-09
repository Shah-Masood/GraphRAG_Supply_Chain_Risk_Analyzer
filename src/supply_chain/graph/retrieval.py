"""
Graph traversal tool for the supply chain risk agent.

Exposes four traversal modes the agent can call:

  supplier_dependencies   — what does company X depend on? (products, countries, ports)
  impact_analysis         — if X (country/port/event) goes down, who is affected?
  supply_path             — shortest supply chain path between two entities
  risk_cluster            — all entities within N hops of a given entity

These are things vector search fundamentally cannot do — they require
traversing relationships, not measuring text similarity.
"""

from typing import Any, Literal

import structlog
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from supply_chain.graph.neo4j_client import graph_db

logger = structlog.get_logger(__name__)


# ── Input schema ──────────────────────────────────────────────────────────────

class GraphTraversalInput(BaseModel):
    mode: Literal["supplier_dependencies", "impact_analysis", "supply_path", "risk_cluster"] = Field(
        ...,
        description=(
            "Traversal mode:\n"
            "  supplier_dependencies — what does a company depend on?\n"
            "  impact_analysis       — who is affected if a country/port/event fails?\n"
            "  supply_path           — shortest path between two entities\n"
            "  risk_cluster          — all entities within 2 hops of a given entity"
        ),
    )
    entity: str = Field(
        ...,
        description="Primary entity name (company, country, port, or risk event)",
    )
    target_entity: str | None = Field(
        default=None,
        description="Target entity name — required for supply_path mode only",
    )
    max_hops: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Maximum relationship hops for risk_cluster (default 2)",
    )


# ── Traversal functions ────────────────────────────────────────────────────────

async def _supplier_dependencies(entity: str) -> str:
    """What does this company depend on? Products, countries, upstream suppliers."""
    results = await graph_db.query(
        """
        MATCH (c:Company)
        WHERE toLower(c.name) CONTAINS toLower($name)
        OPTIONAL MATCH (c)-[:DEPENDS_ON]->(p:Product)
        OPTIONAL MATCH (c)-[:LOCATED_IN]->(country:Country)
        OPTIONAL MATCH (c)-[:SHIPS_THROUGH]->(port:Port)
        OPTIONAL MATCH (upstream:Company)-[:SUPPLIES]->(c)
        RETURN
            c.name AS company,
            c.country AS hq_country,
            c.industry AS industry,
            collect(DISTINCT p.name) AS depends_on_products,
            collect(DISTINCT country.name) AS located_in,
            collect(DISTINCT port.name) AS ships_through,
            collect(DISTINCT upstream.name) AS supplied_by
        LIMIT 5
        """,
        name=entity,
    )

    if not results:
        return f"No company found matching '{entity}' in the knowledge graph."

    lines = []
    for r in results:
        lines.append(f"Company: {r['company']} ({r.get('industry', 'N/A')})")
        lines.append(f"  Headquarters: {r.get('hq_country', 'Unknown')}")
        if r["depends_on_products"]:
            lines.append(f"  Depends on products: {', '.join(r['depends_on_products'])}")
        if r["located_in"]:
            lines.append(f"  Operations in: {', '.join(r['located_in'])}")
        if r["ships_through"]:
            lines.append(f"  Ships through: {', '.join(r['ships_through'])}")
        if r["supplied_by"]:
            lines.append(f"  Supplied by: {', '.join(r['supplied_by'])}")
        lines.append("")

    return "\n".join(lines) or "No dependency data found."


async def _impact_analysis(entity: str) -> str:
    """If this country/port/event is disrupted, which companies are affected?"""
    results = await graph_db.query(
        """
        MATCH (e)
        WHERE (e:Country OR e:Port OR e:RiskEvent)
          AND toLower(e.name) CONTAINS toLower($name)
        CALL {
            WITH e
            MATCH (c:Company)-[:LOCATED_IN|SHIPS_THROUGH|AFFECTED_BY]->(e)
            RETURN c.name AS affected_company, c.country AS company_country,
                   c.industry AS industry, type(last(relationships(shortestPath((c)-[*1..2]->(e))))) AS via
            UNION
            WITH e
            MATCH (e)-[:AFFECTS]->(c:Company)
            RETURN c.name AS affected_company, c.country AS company_country,
                   c.industry AS industry, 'DIRECT' AS via
        }
        RETURN e.name AS disrupted_entity, labels(e)[0] AS entity_type,
               collect(DISTINCT {company: affected_company, country: company_country,
                                  industry: industry, via: via}) AS affected
        LIMIT 3
        """,
        name=entity,
    )

    if not results or not results[0]["affected"]:
        # Fallback: simpler query
        results = await graph_db.query(
            """
            MATCH (e {name: $name})
            MATCH (c:Company)-[r]->(e)
            RETURN e.name AS disrupted_entity, labels(e)[0] AS entity_type,
                   collect({company: c.name, country: c.country, via: type(r)}) AS affected
            """,
            name=entity,
        )

    if not results:
        return f"No entity matching '{entity}' found in the knowledge graph."

    lines = []
    for r in results:
        lines.append(f"Disruption target: {r['disrupted_entity']} ({r.get('entity_type', '?')})")
        affected = r.get("affected", [])
        if not affected:
            lines.append("  No directly linked companies found in graph.")
        else:
            lines.append(f"  Affected companies ({len(affected)}):")
            for a in affected[:15]:
                lines.append(f"    - {a['company']} ({a.get('country', '?')}, {a.get('industry', '?')}) via {a.get('via', '?')}")
        lines.append("")

    return "\n".join(lines)


async def _supply_path(entity: str, target: str) -> str:
    """Find the shortest supply chain path between two entities."""
    results = await graph_db.query(
        """
        MATCH (start), (end)
        WHERE toLower(start.name) CONTAINS toLower($source)
          AND toLower(end.name) CONTAINS toLower($target)
        MATCH path = shortestPath((start)-[*1..6]-(end))
        RETURN
            [node IN nodes(path) | node.name + ' (' + labels(node)[0] + ')'] AS path_nodes,
            [rel IN relationships(path) | type(rel)] AS path_rels,
            length(path) AS hops
        ORDER BY hops
        LIMIT 3
        """,
        source=entity,
        target=target,
    )

    if not results:
        return f"No supply chain path found between '{entity}' and '{target}' within 6 hops."

    lines = [f"Supply chain paths from '{entity}' to '{target}':\n"]
    for i, r in enumerate(results, 1):
        nodes = r["path_nodes"]
        rels = r["path_rels"]
        hops = r["hops"]
        # Interleave nodes and relationships
        path_str = ""
        for j, node in enumerate(nodes):
            path_str += node
            if j < len(rels):
                path_str += f" --[{rels[j]}]--> "
        lines.append(f"Path {i} ({hops} hops): {path_str}")

    return "\n".join(lines)


async def _risk_cluster(entity: str, max_hops: int) -> str:
    """All entities within N hops — reveals the full risk neighborhood."""
    # Neo4j does not allow parameters in variable-length patterns — interpolate directly
    results = await graph_db.query(
        f"""
        MATCH (center)
        WHERE toLower(center.name) CONTAINS toLower($name)
        MATCH (center)-[*1..{max_hops}]-(neighbor)
        WITH center, neighbor, labels(neighbor)[0] AS label
        WHERE neighbor <> center
        RETURN center.name AS center,
               label,
               collect(DISTINCT neighbor.name) AS entities
        ORDER BY label
        """,
        name=entity,
    )

    if not results:
        return f"No entities found within {max_hops} hops of '{entity}'."

    center = results[0]["center"]
    lines = [f"Risk cluster around '{center}' (within {max_hops} hops):\n"]
    for r in results:
        label = r["label"]
        entities = r["entities"]
        lines.append(f"  {label}s ({len(entities)}): {', '.join(entities[:10])}")
        if len(entities) > 10:
            lines.append(f"    ... and {len(entities) - 10} more")

    return "\n".join(lines)


# ── LangChain tool wrapper ─────────────────────────────────────────────────────

@tool("traverse_supply_chain_graph", args_schema=GraphTraversalInput)
async def traverse_supply_chain_graph(
    mode: str,
    entity: str,
    target_entity: str | None = None,
    max_hops: int = 2,
) -> str:
    """
    Traverse the supply chain knowledge graph to answer relationship questions.

    Use this tool for questions that require following connections, NOT just text search:
    - "What does TSMC depend on?" → mode: supplier_dependencies, entity: TSMC
    - "If Taiwan is disrupted, who is affected?" → mode: impact_analysis, entity: Taiwan
    - "What's the supply chain path from REalloys to Apple?" → mode: supply_path
    - "Show me the full risk network around Foxconn" → mode: risk_cluster

    This complements retrieve_supply_chain_docs — use both for comprehensive analysis.
    """
    logger.info("graph_traversal", mode=mode, entity=entity)

    if mode == "supplier_dependencies":
        return await _supplier_dependencies(entity)
    elif mode == "impact_analysis":
        return await _impact_analysis(entity)
    elif mode == "supply_path":
        if not target_entity:
            return "supply_path mode requires a target_entity."
        return await _supply_path(entity, target_entity)
    elif mode == "risk_cluster":
        return await _risk_cluster(entity, max_hops)
    else:
        return f"Unknown mode '{mode}'. Use: supplier_dependencies, impact_analysis, supply_path, risk_cluster"
