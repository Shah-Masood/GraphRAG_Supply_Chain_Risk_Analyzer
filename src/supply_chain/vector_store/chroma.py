from dataclasses import dataclass, field
from typing import Any

import chromadb
import structlog

from supply_chain.config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class Chunk:
    """A single embedded chunk ready for upsert."""
    id: str                         # stable ID: f"{doc_id}_{chunk_index}"
    text: str                       # raw chunk text
    embedding: list[float]          # pre-computed embedding vector
    metadata: dict[str, Any] = field(default_factory=dict)
    # Recommended metadata keys:
    #   doc_id, doc_type, supplier_id, country, source_url, chunk_index


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float                 # lower = more similar


class VectorStore:
    def __init__(self) -> None:
        self._client: chromadb.AsyncHttpClient | None = None
        self._collection: chromadb.Collection | None = None

    async def connect(self) -> None:
        """Initialise client and get-or-create the collection. Called at startup."""
        settings = get_settings()
        self._client = await chromadb.AsyncHttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        # get_or_create is idempotent — safe to call on every startup
        self._collection = await self._client.get_or_create_collection(
            name=settings.chroma_collection,
            # cosine is better than L2 for semantic similarity on normalised embeddings
            metadata={"hnsw:space": "cosine"},
        )
        count = await self._collection.count()
        logger.info(
            "chroma_connected",
            collection=settings.chroma_collection,
            existing_chunks=count,
        )

    async def disconnect(self) -> None:
        """No explicit teardown needed for HTTP client, but kept for symmetry."""
        self._client = None
        self._collection = None
        logger.info("chroma_disconnected")

    def _ensure_ready(self) -> chromadb.Collection:
        if self._collection is None:
            raise RuntimeError("VectorStore.connect() has not been called yet.")
        return self._collection

    async def upsert(self, chunks: list[Chunk]) -> None:
        """
        Insert or update chunks. Idempotent — safe to re-run on the same doc.
        Batches in groups of 100 to stay within Chroma's recommended payload size.
        """
        collection = self._ensure_ready()
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            await collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=[c.embedding for c in batch],
                metadatas=[c.metadata for c in batch],
            )
        logger.info("chroma_upsert", chunk_count=len(chunks))

    async def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Semantic search. Returns up to n_results chunks sorted by similarity.

        `where` lets you filter by metadata, e.g.:
            {"supplier_id": "abc-123"}
            {"country": {"$in": ["Taiwan", "China"]}}
        """
        collection = self._ensure_ready()
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = await collection.query(**kwargs)

        chunks: list[RetrievedChunk] = []
        ids       = results["ids"][0]
        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        for cid, doc, meta, dist in zip(ids, docs, metas, distances):
            chunks.append(RetrievedChunk(
                id=cid,
                text=doc,
                metadata=meta or {},
                distance=dist,
            ))

        return chunks

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document (e.g. for re-ingestion)."""
        collection = self._ensure_ready()
        await collection.delete(where={"doc_id": doc_id})
        logger.info("chroma_delete", doc_id=doc_id)

    async def count(self) -> int:
        collection = self._ensure_ready()
        return await collection.count()


# Module-level singleton
vector_store = VectorStore()
