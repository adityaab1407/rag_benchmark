"""
RAG Benchmark MCP Server
========================
Exposes your RAG benchmark system as tools Claude Desktop can call.

HOW TO REGISTER WITH CLAUDE DESKTOP:
-------------------------------------
1. Find Claude Desktop config file:
   Windows:  %APPDATA%\Claude\claude_desktop_config.json
   Mac:      ~/Library/Application Support/Claude/claude_desktop_config.json

2. Add this to the config:
   {
     "mcpServers": {
       "rag-benchmark": {
         "command": "python",
         "args": ["C:/path/to/your/project/app/mcp/server.py"],
         "env": {
           "RAG_API_BASE": "http://localhost:8000/api/v1"
         }
       }
     }
   }

3. Make sure FastAPI is running:
   uvicorn app.main:app --reload

4. Restart Claude Desktop

AVAILABLE TOOLS:
----------------
  rag_query       Ask a single strategy a question
  rag_benchmark   Run all 4 strategies in parallel and compare
  rag_health      Check if the RAG system is online
"""

import sys
import json
import os
import asyncio
import httpx
from typing import Any

# MCP SDK
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("MCP SDK not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)


API_BASE = os.environ.get("RAG_API_BASE", "http://localhost:8000/api/v1")

server = Server("rag-benchmark")


# =============================================================================
# TOOLS DEFINITION
# =============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="rag_benchmark",
            description=(
                "Run a question through all 4 RAG strategies simultaneously "
                "(naive, hybrid, HyDE, reranked) and compare their answers. "
                "Returns the best answer, confidence scores, latency per strategy, "
                "and a summary of which strategy performed best. "
                "Use this when comparing RAG strategies or when you want the "
                "most reliable answer from the knowledge base."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask the RAG system"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve per strategy (default 5)",
                        "default": 5
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="rag_query",
            description=(
                "Ask a single RAG strategy a question. "
                "Use naive for fastest baseline, hybrid for balanced performance, "
                "hyde for complex conceptual questions, reranked for highest quality. "
                "Use rag_benchmark instead if you want to compare all strategies."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask"
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["naive", "hybrid", "hyde", "reranked"],
                        "description": "Which RAG strategy to use",
                        "default": "reranked"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve (default 5)",
                        "default": 5
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="rag_health",
            description="Check if the RAG benchmark system is online and all components are loaded.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


# =============================================================================
# TOOL HANDLERS
# =============================================================================

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:

    async with httpx.AsyncClient(timeout=60.0) as client:

        if name == "rag_health":
            try:
                r = await client.get(f"{API_BASE}/health")
                data = r.json()
                lines = ["RAG System Health\n"]
                for component, status in data.items():
                    icon = "✓" if status else "✗"
                    lines.append(f"  {icon} {component}: {'online' if status else 'offline'}")
                return [TextContent(type="text", text="\n".join(lines))]
            except Exception as e:
                return [TextContent(type="text", text=f"Health check failed: {e}\nMake sure uvicorn is running at {API_BASE}")]

        elif name == "rag_query":
            question = arguments["question"]
            strategy = arguments.get("strategy", "reranked")
            top_k    = arguments.get("top_k", 5)
            try:
                r = await client.post(
                    f"{API_BASE}/query",
                    json={"question": question, "strategy": strategy, "top_k": top_k}
                )
                data = r.json()
                if "error" in data:
                    return [TextContent(type="text", text=f"Error: {data['error']}")]

                answerable = data.get("is_answerable", False)
                confidence = data.get("confidence", 0)
                answer     = data.get("answer", "")
                latency    = data.get("latency", {}).get("total", 0)
                reasoning  = data.get("reasoning", "")

                result = (
                    f"Strategy: {strategy}\n"
                    f"Answerable: {'yes' if answerable else 'no'}\n"
                    f"Confidence: {confidence:.2f}\n"
                    f"Latency: {latency:.2f}s\n\n"
                    f"Answer:\n{answer}\n\n"
                    f"Reasoning: {reasoning}"
                )
                return [TextContent(type="text", text=result)]

            except Exception as e:
                return [TextContent(type="text", text=f"Query failed: {e}")]

        elif name == "rag_benchmark":
            question = arguments["question"]
            top_k    = arguments.get("top_k", 5)
            try:
                r = await client.post(
                    f"{API_BASE}/benchmark",
                    json={"question": question, "top_k": top_k}
                )
                data = r.json()
                if "error" in data:
                    return [TextContent(type="text", text=f"Error: {data['error']}")]

                lines = [
                    f"Question: {question}\n",
                    f"Best strategy:    {data.get('best_strategy', '—')}",
                    f"Fastest strategy: {data.get('fastest_strategy', '—')}",
                    f"Wall time:        {data.get('total_time', 0):.2f}s",
                    f"(All 4 ran in parallel)\n",
                ]

                for s in data.get("strategies", []):
                    strategy   = s.get("strategy", "")
                    answer     = s.get("answer", "")
                    confidence = s.get("confidence", 0)
                    answerable = s.get("is_answerable", False)
                    latency    = s.get("latency", {}).get("total", 0)
                    is_best    = strategy == data.get("best_strategy")

                    lines.append(f"{'★ ' if is_best else '  '}{strategy.upper()}")
                    lines.append(f"  answerable: {'yes' if answerable else 'no'} | confidence: {confidence:.2f} | latency: {latency:.2f}s")
                    if answerable and not answer.startswith("ERROR"):
                        lines.append(f"  {answer[:300]}")
                    elif answer.startswith("ERROR"):
                        lines.append(f"  [rate limited or error]")
                    else:
                        lines.append(f"  [abstained — question not answerable from corpus]")
                    lines.append("")

                return [TextContent(type="text", text="\n".join(lines))]

            except Exception as e:
                return [TextContent(type="text", text=f"Benchmark failed: {e}")]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]


# =============================================================================
# ENTRY POINT
# =============================================================================

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())