"""
Seed real-world supply chain suppliers into the database.

Covers the major players across semiconductors, electronics manufacturing,
logistics, raw materials, and automotive supply chains.

Usage:
    python seed_suppliers.py

Idempotent — safe to run multiple times (uses INSERT ... ON CONFLICT DO NOTHING).
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

import asyncpg

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres@localhost:5432/supply_chain_db")

# (name, country, region, tier, industry, metadata)
SUPPLIERS = [
    # ── Semiconductors ──────────────────────────────────────────────────────────
    ("TSMC", "Taiwan", "East Asia", 1, "Semiconductors",
     {"products": ["logic chips", "advanced nodes"], "customers": ["Apple", "NVIDIA", "AMD"], "employees": 73000}),
    ("Samsung Electronics", "South Korea", "East Asia", 1, "Semiconductors",
     {"products": ["memory", "logic chips", "displays"], "employees": 270000}),
    ("ASML", "Netherlands", "Europe", 1, "Semiconductor Equipment",
     {"products": ["EUV lithography machines"], "note": "sole supplier of EUV globally", "employees": 42000}),
    ("SK Hynix", "South Korea", "East Asia", 1, "Semiconductors",
     {"products": ["DRAM", "NAND flash"], "employees": 30000}),
    ("Micron Technology", "USA", "North America", 1, "Semiconductors",
     {"products": ["DRAM", "NAND", "NOR flash"], "employees": 48000}),
    ("Intel Foundry Services", "USA", "North America", 1, "Semiconductors",
     {"products": ["x86 CPUs", "foundry services"], "employees": 120000}),
    ("GlobalFoundries", "USA", "North America", 1, "Semiconductors",
     {"products": ["mature node chips", "RF semiconductors"], "employees": 12000}),
    ("UMC", "Taiwan", "East Asia", 1, "Semiconductors",
     {"products": ["specialty semiconductors", "mature nodes"], "employees": 20000}),

    # ── Electronics Manufacturing ───────────────────────────────────────────────
    ("Foxconn", "Taiwan", "East Asia", 1, "Electronics Manufacturing",
     {"products": ["iPhone assembly", "servers", "EVs"], "major_plants": ["Zhengzhou", "Shenzhen"], "employees": 800000}),
    ("Pegatron", "Taiwan", "East Asia", 1, "Electronics Manufacturing",
     {"products": ["iPhone assembly", "consumer electronics"], "employees": 100000}),
    ("Flex Ltd", "Singapore", "Southeast Asia", 1, "Electronics Manufacturing",
     {"products": ["PCBs", "medical devices", "auto electronics"], "employees": 170000}),
    ("Jabil", "USA", "North America", 1, "Electronics Manufacturing",
     {"products": ["PCBs", "packaging", "healthcare devices"], "employees": 260000}),
    ("Wistron", "Taiwan", "East Asia", 1, "Electronics Manufacturing",
     {"products": ["laptops", "servers", "smartphones"], "employees": 100000}),

    # ── Raw Materials ───────────────────────────────────────────────────────────
    ("MP Materials", "USA", "North America", 2, "Rare Earth Materials",
     {"products": ["rare earth oxides", "NdFeB magnets"], "note": "largest US rare earth producer"}),
    ("Ucore Rare Metals", "Canada", "North America", 2, "Rare Earth Materials",
     {"products": ["rare earth separation", "critical minerals"]}),
    ("Albemarle Corporation", "USA", "North America", 1, "Lithium",
     {"products": ["lithium compounds", "battery-grade lithium"], "employees": 7000}),
    ("SQM", "Chile", "South America", 1, "Lithium",
     {"products": ["lithium carbonate", "lithium hydroxide"], "note": "Atacama brine operations"}),
    ("BASF", "Germany", "Europe", 2, "Chemicals",
     {"products": ["specialty chemicals", "battery materials", "catalysts"], "employees": 111000}),

    # ── Logistics & Shipping ────────────────────────────────────────────────────
    ("Maersk", "Denmark", "Europe", 1, "Logistics",
     {"products": ["container shipping", "port operations", "logistics"], "fleet_size": 700, "employees": 100000}),
    ("MSC Mediterranean Shipping", "Switzerland", "Europe", 1, "Logistics",
     {"products": ["container shipping"], "note": "world's largest container line by capacity"}),
    ("FedEx", "USA", "North America", 1, "Logistics",
     {"products": ["express freight", "ground shipping", "supply chain services"], "employees": 500000}),
    ("DHL Supply Chain", "Germany", "Europe", 1, "Logistics",
     {"products": ["warehousing", "fulfillment", "last-mile delivery"], "employees": 600000}),
    ("Evergreen Marine", "Taiwan", "East Asia", 1, "Logistics",
     {"products": ["container shipping"], "note": "Ever Given operator"}),

    # ── Automotive Supply Chain ─────────────────────────────────────────────────
    ("Bosch", "Germany", "Europe", 1, "Automotive Components",
     {"products": ["automotive ECUs", "sensors", "fuel systems"], "employees": 429000}),
    ("Continental AG", "Germany", "Europe", 1, "Automotive Components",
     {"products": ["tires", "brake systems", "ADAS sensors"], "employees": 200000}),
    ("Denso", "Japan", "East Asia", 1, "Automotive Components",
     {"products": ["thermal systems", "powertrain", "electronics"], "employees": 168000}),
    ("Aptiv", "Ireland", "Europe", 1, "Automotive Components",
     {"products": ["vehicle architecture", "autonomous driving"], "employees": 200000}),

    # ── Cloud & Data Center ─────────────────────────────────────────────────────
    ("Quanta Computer", "Taiwan", "East Asia", 1, "Data Center Hardware",
     {"products": ["servers", "cloud infrastructure", "laptops"], "customers": ["AWS", "Google", "Meta"]}),
    ("Super Micro Computer", "USA", "North America", 1, "Data Center Hardware",
     {"products": ["AI servers", "rack systems", "storage"], "note": "major NVIDIA GPU server integrator"}),
]


async def seed() -> None:
    print(f"\nConnecting to {DATABASE_URL[:40]}...\n")
    conn = await asyncpg.connect(DATABASE_URL)

    try:
        inserted = 0
        skipped = 0
        for name, country, region, tier, industry, metadata in SUPPLIERS:
            import json
            result = await conn.execute(
                """
                INSERT INTO suppliers (name, country, region, tier, industry, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                ON CONFLICT DO NOTHING
                """,
                name, country, region, tier, industry, json.dumps(metadata),
            )
            if result == "INSERT 0 1":
                inserted += 1
                print(f"  ✓ {name} ({country})")
            else:
                skipped += 1

        print(f"\nDone! Inserted {inserted} suppliers, skipped {skipped} duplicates.")
        print(f"Total suppliers in DB: {await conn.fetchval('SELECT COUNT(*) FROM suppliers')}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(seed())
