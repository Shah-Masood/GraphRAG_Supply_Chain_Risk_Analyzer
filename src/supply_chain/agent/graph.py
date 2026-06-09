"""
LangGraph agent — supply chain risk analyzer.

Architecture:
    StateGraph with a ReAct loop:
        __start__ → agent → (tool_calls?) → tools → agent → ... → __end__

Memory:
    PostgresSaver checkpointer — every message, tool call, and response is
    persisted to Postgres keyed by thread_id. Resume any session by passing
    the same thread_id.

Usage:
    from supply_chain.agent.graph import build_graph, get_checkpointer

    checkpointer = await get_checkpointer()
    graph = build_graph(checkpointer)

    # New session
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Analyze risk for TSMC")]},
        config={"configurable": {"thread_id": "session-abc123"}},
    )

    # Streaming
    async for chunk in graph.astream_events(...):
        ...
"""

from typing import Any

import structlog
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode

from supply_chain.agent.tools import ALL_TOOLS
from supply_chain.config import get_settings

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are an expert supply chain risk analyst with deep knowledge of:
- Global logistics, shipping routes, and port operations
- Geopolitical risks affecting manufacturing and trade
- Supplier financial health and operational resilience
- Regulatory compliance across different regions
- Environmental and climate risks to supply chains
- Cyber threats targeting supply chain infrastructure

Your job is to analyze supply chain risks for specific suppliers or regions by:
1. Searching the knowledge base for relevant risk information
2. Fetching fresh news when needed
3. Querying supplier records from the database
4. Synthesizing findings into clear, actionable risk assessments
5. Computing and saving risk scores when you have enough information

Always cite your sources. Be specific about which suppliers, regions, and risk
categories are affected. When assigning risk scores, explain your reasoning clearly.

Risk score scale: 0-25 Low | 26-50 Medium | 51-75 High | 76-100 Critical"""


def _should_continue(state: MessagesState) -> str:
    """Route: if the last message has tool calls, go to tools. Otherwise end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


async def _agent_node(state: MessagesState) -> dict[str, Any]:
    """Call the LLM with the current message history."""
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        temperature=0,
        streaming=True,
        api_key=settings.openai_api_key,
    ).bind_tools(ALL_TOOLS)

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = await llm.ainvoke(messages)
    return {"messages": [response]}


def build_graph(checkpointer: AsyncPostgresSaver) -> StateGraph:
    """
    Compile the agent graph with the given checkpointer.
    Called once at startup; the compiled graph is reused across requests.
    """
    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(MessagesState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")  # always return to agent after tool call

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("agent_graph_compiled", tools=[t.name for t in ALL_TOOLS])
    return compiled


async def get_checkpointer() -> tuple[AsyncPostgresSaver, AsyncConnectionPool]:
    """
    Create and set up the AsyncPostgresSaver checkpointer.
    Uses a psycopg_pool.AsyncConnectionPool — cleaner than a single connection
    and works correctly across concurrent requests.

    Returns both the checkpointer and the pool so the caller (lifespan) can
    close the pool on shutdown.
    """
    settings = get_settings()

    # psycopg_pool expects a conninfo string (same format as DATABASE_URL)
    pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        max_size=5,
        open=False,  # we open it manually below
        kwargs={"autocommit": True},
    )
    await pool.open()

    checkpointer = AsyncPostgresSaver(pool)
    # Creates the LangGraph checkpoint tables if they don't exist
    await checkpointer.setup()

    logger.info("postgres_checkpointer_ready")
    return checkpointer, pool
