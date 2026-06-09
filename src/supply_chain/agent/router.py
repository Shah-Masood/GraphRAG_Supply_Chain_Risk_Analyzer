"""
Agent API router.

Endpoints:
    POST /agent/chat            — send a message, get a streaming SSE response
    POST /agent/sessions        — create a new named session
    GET  /agent/sessions/{id}   — get session info + message history
    GET  /agent/sessions        — list all sessions for a user
"""

import json
import uuid
from typing import AsyncGenerator

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from supply_chain.database.pool import db

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])


# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to the agent")
    thread_id: str | None = Field(
        default=None,
        description="Session thread ID. Omit to start a new session.",
    )
    user_id: str | None = Field(default=None, description="Optional user identifier")
    supplier_id: str | None = Field(default=None, description="Focus analysis on a specific supplier")


class SessionCreateRequest(BaseModel):
    user_id: str | None = None
    supplier_id: str | None = None
    topic: str | None = None


class SessionResponse(BaseModel):
    thread_id: str
    user_id: str | None
    supplier_id: str | None
    topic: str | None
    created_at: str
    last_active_at: str


# ── SSE streaming helper ───────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    """Format a server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_agent_response(
    request: Request,
    message: str,
    thread_id: str,
) -> AsyncGenerator[str, None]:
    """
    Stream the agent's response as SSE events.

    Event types:
        token       — a single streamed token from the LLM
        tool_start  — agent is calling a tool
        tool_end    — tool returned a result
        done        — agent finished, includes full final message
        error       — something went wrong
    """
    # Access the compiled graph from app state (set in lifespan)
    graph = request.app.state.agent_graph
    config = {"configurable": {"thread_id": thread_id}}

    try:
        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            version="v2",
        ):
            kind = event["event"]
            name = event.get("name", "")

            # Stream LLM tokens
            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield _sse("token", {"content": chunk.content})

            # Tool call started
            elif kind == "on_tool_start":
                yield _sse("tool_start", {
                    "tool": name,
                    "input": str(event["data"].get("input", ""))[:200],
                })

            # Tool call finished
            elif kind == "on_tool_end":
                output = event["data"].get("output", "")
                yield _sse("tool_end", {
                    "tool": name,
                    "output_preview": str(output)[:300],
                })

            # Check for client disconnect
            if await request.is_disconnected():
                logger.info("client_disconnected", thread_id=thread_id)
                break

        yield _sse("done", {"thread_id": thread_id})

    except Exception as e:
        logger.error("agent_stream_error", thread_id=thread_id, error=str(e))
        yield _sse("error", {"message": str(e)})


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    """
    Send a message to the agent and stream the response as SSE.

    If no thread_id is provided, a new session is created automatically.
    Pass the returned thread_id in subsequent requests to continue the conversation.

    Connect with:
        EventSource('/agent/chat') or fetch() with ReadableStream
    """
    thread_id = body.thread_id or str(uuid.uuid4())

    # Upsert session record
    async with db.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO agent_sessions (thread_id, user_id, supplier_id, topic, last_active_at)
            VALUES ($1, $2, $3::uuid, $4, now())
            ON CONFLICT (thread_id) DO UPDATE
                SET last_active_at = now(),
                    user_id = EXCLUDED.user_id
            """,
            thread_id,
            body.user_id,
            body.supplier_id,
            body.message[:100],  # use first message as topic if none set
        )

    logger.info("agent_chat_start", thread_id=thread_id, user_id=body.user_id)

    return StreamingResponse(
        _stream_agent_response(request, body.message, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Thread-ID": thread_id,  # client can grab this from headers
        },
    )


@router.post("/sessions", response_model=SessionResponse)
async def create_session(body: SessionCreateRequest) -> SessionResponse:
    """Create a new agent session and return its thread_id."""
    thread_id = str(uuid.uuid4())
    async with db.transaction() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO agent_sessions (thread_id, user_id, supplier_id, topic)
            VALUES ($1, $2, $3::uuid, $4)
            RETURNING *
            """,
            thread_id, body.user_id, body.supplier_id, body.topic,
        )
    return SessionResponse(
        thread_id=row["thread_id"],
        user_id=row["user_id"],
        supplier_id=str(row["supplier_id"]) if row["supplier_id"] else None,
        topic=row["topic"],
        created_at=row["created_at"].isoformat(),
        last_active_at=row["last_active_at"].isoformat(),
    )


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(user_id: str | None = None, limit: int = 20) -> list[SessionResponse]:
    """List agent sessions, optionally filtered by user_id."""
    async with db.acquire() as conn:
        if user_id:
            rows = await conn.fetch(
                "SELECT * FROM agent_sessions WHERE user_id = $1 ORDER BY last_active_at DESC LIMIT $2",
                user_id, limit,
            )
        else:
            rows = await conn.fetch(
                "SELECT * FROM agent_sessions ORDER BY last_active_at DESC LIMIT $1", limit
            )
    return [
        SessionResponse(
            thread_id=r["thread_id"],
            user_id=r["user_id"],
            supplier_id=str(r["supplier_id"]) if r["supplier_id"] else None,
            topic=r["topic"],
            created_at=r["created_at"].isoformat(),
            last_active_at=r["last_active_at"].isoformat(),
        )
        for r in rows
    ]


@router.get("/sessions/{thread_id}", response_model=SessionResponse)
async def get_session(thread_id: str) -> SessionResponse:
    """Get a specific session by thread_id."""
    async with db.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM agent_sessions WHERE thread_id = $1", thread_id
        )
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(
        thread_id=row["thread_id"],
        user_id=row["user_id"],
        supplier_id=str(row["supplier_id"]) if row["supplier_id"] else None,
        topic=row["topic"],
        created_at=row["created_at"].isoformat(),
        last_active_at=row["last_active_at"].isoformat(),
    )
