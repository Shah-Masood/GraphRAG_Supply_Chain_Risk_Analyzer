import asyncio
import hashlib
import uuid
from pathlib import Path
from typing import Any
 
import structlog
from openai import AsyncOpenAI
from pypdf import PdfReader
 
from supply_chain.config import get_settings
from supply_chain.database.pool import db
from supply_chain.vector_store.chroma import Chunk, vector_store
 
logger = structlog.get_logger(__name__)
 
# ── Chunking config ────────────────────────────────────────────────────────────
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
CHARS_PER_TOKEN = 4
 
 
def _chunk_text(text: str) -> list[str]:
    char_size = CHUNK_SIZE * CHARS_PER_TOKEN
    char_overlap = CHUNK_OVERLAP * CHARS_PER_TOKEN
    step = char_size - char_overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + char_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks
 
 
def _extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)
 
 
def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
 
 
async def _embed_batch(texts: list[str], client: AsyncOpenAI) -> list[list[float]]:
    settings = get_settings()
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        response = await client.embeddings.create(
            input=batch,
            model=settings.openai_embedding_model,
        )
        embeddings.extend([item.embedding for item in response.data])
    return embeddings
 
 
async def ingest_document(
    source: str | Path,
    doc_type: str,
    title: str | None = None,
    supplier_id: str | None = None,
    country: str | None = None,
    source_url: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    run_graph_extraction: bool = True,
) -> str:
    """
    Full ingestion pipeline for a single document.
    Returns the document UUID.
 
    Steps:
        1. Extract raw text
        2. Dedup check via SHA-256
        3. Chunk text
        4. Embed chunks (OpenAI)
        5. Upsert to ChromaDB
        6. Record in Postgres
        7. Extract entities/relationships → upsert to Neo4j  (GraphRAG)
    """
    settings = get_settings()
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    source = Path(source) if isinstance(source, str) else source
 
    # ── 1. Extract text ────────────────────────────────────────────────────────
    if doc_type == "pdf":
        raw_text = _extract_text_from_pdf(source)
    else:
        raw_text = source.read_text(encoding="utf-8")
 
    if not raw_text.strip():
        raise ValueError(f"No text extracted from {source}")
 
    # ── 2. Dedup ───────────────────────────────────────────────────────────────
    content_hash = _sha256(raw_text)
    async with db.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM documents WHERE raw_text_hash = $1", content_hash
        )
    if existing:
        logger.info("ingest_skipped_duplicate", doc_id=str(existing["id"]))
        return str(existing["id"])
 
    # ── 3. Chunk ───────────────────────────────────────────────────────────────
    raw_chunks = _chunk_text(raw_text)
    logger.info("ingest_chunked", source=str(source), chunk_count=len(raw_chunks))
 
    # ── 4. Embed ───────────────────────────────────────────────────────────────
    embeddings = await _embed_batch(raw_chunks, openai_client)
 
    # ── 5. Upsert to ChromaDB ──────────────────────────────────────────────────
    doc_id = str(uuid.uuid4())
    chunk_metadata: dict[str, Any] = {"doc_id": doc_id, "doc_type": doc_type}
    if supplier_id:
        chunk_metadata["supplier_id"] = supplier_id
    if country:
        chunk_metadata["country"] = country
    if source_url:
        chunk_metadata["source_url"] = source_url
    if extra_metadata:
        chunk_metadata.update(extra_metadata)
 
    chunks = [
        Chunk(
            id=f"{doc_id}_{i}",
            text=text,
            embedding=emb,
            metadata={**chunk_metadata, "chunk_index": i},
        )
        for i, (text, emb) in enumerate(zip(raw_chunks, embeddings))
    ]
    await vector_store.upsert(chunks)
 
    # ── 6. Record in Postgres ──────────────────────────────────────────────────
    async with db.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO documents (id, source_url, title, doc_type, supplier_id,
                                   country, raw_text_hash, chunk_count)
            VALUES ($1, $2, $3, $4, $5::uuid, $6, $7, $8)
            """,
            doc_id,
            source_url or str(source),
            title or source.stem,
            doc_type,
            supplier_id,
            country,
            content_hash,
            len(chunks),
        )
 
    # ── 7. Graph extraction (async, non-blocking) ──────────────────────────────
    if run_graph_extraction:
        try:
            from supply_chain.graph.pipeline import build_graph_from_doc
            # Fire and forget — don't block ingestion on graph extraction
            asyncio.create_task(
                build_graph_from_doc(
                    doc_id=doc_id,
                    chunks=raw_chunks,
                    metadata={
                        "source_url": source_url,
                        "country": country,
                        "supplier_id": supplier_id,
                        **(extra_metadata or {}),
                    },
                )
            )
        except Exception as e:
            logger.warning("graph_extraction_skipped", doc_id=doc_id, error=str(e))
 
    logger.info("ingest_complete", doc_id=doc_id, chunk_count=len(chunks))
    return doc_id
 
 
async def ingest_directory(
    directory: str | Path,
    glob: str = "**/*.pdf",
    **kwargs: Any,
) -> list[str]:
    """Ingest all matching files in a directory concurrently."""
    directory = Path(directory)
    paths = list(directory.glob(glob))
    logger.info("ingest_directory_start", path=str(directory), file_count=len(paths))
    tasks = [ingest_document(p, doc_type="pdf", **kwargs) for p in paths]
    doc_ids = await asyncio.gather(*tasks, return_exceptions=True)
    successes = [d for d in doc_ids if isinstance(d, str)]
    failures = [e for e in doc_ids if isinstance(e, Exception)]
    if failures:
        logger.warning("ingest_directory_partial_failure", failures=len(failures))
    return successes