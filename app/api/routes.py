"""
app/api/routes.py — UPDATED VERSION
=====================================
Changes from original:
  + Imports log_observability from database
  + Imports get_request_id from middleware
  + /query endpoint logs to observability_log
  + /benchmark endpoint logs all 4 strategies + total cost
  Everything else unchanged.
"""

import time
import asyncio
from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from app.models import (
    QueryRequest, BenchmarkRequest,
    RAGResult, StrategyComparison, BenchmarkResult,
)
from app.database import log_query, log_observability
from app.middleware import get_request_id

router = APIRouter()


# =============================================================================
# HEALTH
# =============================================================================

@router.get("/health")
async def health(request: Request):
    app_state = request.app.state
    return {
        "naive_rag":    hasattr(app_state, "naive_rag"),
        "hybrid_rag":   hasattr(app_state, "hybrid_rag"),
        "hyde_rag":     hasattr(app_state, "hyde_rag"),
        "reranked_rag": hasattr(app_state, "reranked_rag"),
        "embedder":     hasattr(app_state, "embedder"),
        "vector_store": hasattr(app_state, "vector_store"),
        "bm25_store":   hasattr(app_state, "bm25_store"),
    }


@router.get("/strategies")
async def strategies(request: Request):
    return {
        "strategies": [
            {"name": "naive",    "status": "available", "description": "Vector search baseline"},
            {"name": "hybrid",   "status": "available", "description": "BM25 + vector + RRF"},
            {"name": "hyde",     "status": "available", "description": "Hypothetical document embeddings"},
            {"name": "reranked", "status": "available", "description": "Cross-encoder reranking"},
        ]
    }


# =============================================================================
# SINGLE QUERY
# =============================================================================

@router.post("/query")
async def query(request: Request, body: QueryRequest):
    app_state  = request.app.state
    req_id     = get_request_id()
    strategy   = body.strategy
    top_k      = body.top_k or 5

    strategy_map = {
        "naive":    app_state.naive_rag,
        "hybrid":   app_state.hybrid_rag,
        "hyde":     app_state.hyde_rag,
        "reranked": app_state.reranked_rag,
    }

    if strategy not in strategy_map:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy}")

    rag = strategy_map[strategy]

    try:
        result: RAGResult = await rag.run_async(body.question, top_k)

        # Log to original query_logs table
        top_score = result.retrieved_chunks[0].get("score", 0.0) if result.retrieved_chunks else 0.0
        await log_query(
            strategy=strategy,
            query=body.question,
            answer=result.answer,
            confidence=result.confidence,
            is_answerable=result.is_answerable,
            retrieve_latency=result.latency.get("retrieve", 0),
            generate_latency=result.latency.get("generate", 0),
            total_latency=result.latency.get("total", 0),
            top_chunk_score=top_score,
        )

        # NEW — log to observability table
        token_usage = result.token_usage
        await log_observability(
            request_id=req_id,
            endpoint="/query",
            strategy=strategy,
            query=body.question,
            prompt_tokens=token_usage.prompt_tokens if token_usage else 0,
            completion_tokens=token_usage.completion_tokens if token_usage else 0,
            total_tokens=token_usage.total_tokens if token_usage else 0,
            cost_usd=token_usage.cost_usd if token_usage else 0.0,
            retrieve_ms=result.latency.get("retrieve", 0) * 1000,
            generate_ms=result.latency.get("generate", 0) * 1000,
            total_ms=result.latency.get("total", 0) * 1000,
            confidence=result.confidence,
            is_answerable=result.is_answerable,
            model=token_usage.model if token_usage else "",
        )

        logger.info(
            "[{}] /query {} — tokens={} cost=${:.6f}",
            req_id,
            strategy,
            token_usage.total_tokens if token_usage else 0,
            token_usage.cost_usd if token_usage else 0,
        )

        return result

    except Exception as e:
        await log_observability(
            request_id=req_id,
            endpoint="/query",
            strategy=strategy,
            query=body.question,
            error=str(e)[:300],
        )
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# BENCHMARK — ALL 4 STRATEGIES IN PARALLEL
# =============================================================================

@router.post("/benchmark")
async def benchmark(request: Request, body: BenchmarkRequest):
    app_state = request.app.state
    req_id    = get_request_id()
    top_k     = body.top_k or 5

    t_start = time.time()

    # Run all 4 in parallel
    results = await asyncio.gather(
        app_state.naive_rag.run_async(body.question, top_k),
        app_state.hybrid_rag.run_async(body.question, top_k),
        app_state.hyde_rag.run_async(body.question, top_k),
        app_state.reranked_rag.run_async(body.question, top_k),
        return_exceptions=True,
    )

    total_time   = round(time.time() - t_start, 3)
    comparisons  = []
    total_cost   = 0.0

    strategy_names = ["naive", "hybrid", "hyde", "reranked"]

    for strategy_name, result in zip(strategy_names, results):
        if isinstance(result, Exception):
            comparisons.append(StrategyComparison(
                strategy=strategy_name,
                answer=f"ERROR: {str(result)[:200]}",
                confidence=0.0,
                is_answerable=False,
                latency={"retrieve": 0, "generate": 0, "total": 0},
                top_chunk_score=0.0,
                retrieval_method=strategy_name,
            ))

            # Log error
            await log_observability(
                request_id=req_id,
                endpoint="/benchmark",
                strategy=strategy_name,
                query=body.question,
                error=str(result)[:300],
            )
            continue

        token_usage = result.token_usage
        cost        = token_usage.cost_usd if token_usage else 0.0
        total_cost += cost

        top_score = result.retrieved_chunks[0].get("score", 0.0) if result.retrieved_chunks else 0.0

        comparisons.append(StrategyComparison(
            strategy=strategy_name,
            answer=result.answer,
            confidence=result.confidence,
            is_answerable=result.is_answerable,
            latency=result.latency,
            top_chunk_score=top_score,
            retrieval_method=result.strategy,
            token_usage=token_usage,
            cost_usd=cost,
        ))

        # Log each strategy to observability
        await log_observability(
            request_id=req_id,
            endpoint="/benchmark",
            strategy=strategy_name,
            query=body.question,
            prompt_tokens=token_usage.prompt_tokens if token_usage else 0,
            completion_tokens=token_usage.completion_tokens if token_usage else 0,
            total_tokens=token_usage.total_tokens if token_usage else 0,
            cost_usd=cost,
            retrieve_ms=result.latency.get("retrieve", 0) * 1000,
            generate_ms=result.latency.get("generate", 0) * 1000,
            total_ms=result.latency.get("total", 0) * 1000,
            confidence=result.confidence,
            is_answerable=result.is_answerable,
            model=token_usage.model if token_usage else "",
        )

    # Find best and fastest
    answered = [c for c in comparisons if c.is_answerable and not c.answer.startswith("ERROR")]
    best_strategy    = max(answered, key=lambda x: x.confidence).strategy if answered else "none"
    fastest_strategy = min(
        [c for c in comparisons if not c.answer.startswith("ERROR")],
        key=lambda x: x.latency.get("total", 999)
    ).strategy if comparisons else "none"

    summary_parts = []
    for c in comparisons:
        if c.answer.startswith("ERROR"):
            summary_parts.append(f"{c.strategy}: FAILED")
        else:
            summary_parts.append(
                f"{c.strategy}: confidence={c.confidence} latency={c.latency.get('total',0):.3f}s cost=${c.cost_usd:.6f}"
            )

    logger.info(
        "[{}] /benchmark — {}s total — ${:.6f} total cost",
        req_id, total_time, total_cost,
    )

    return BenchmarkResult(
        query=body.question,
        strategies=comparisons,
        best_strategy=best_strategy,
        fastest_strategy=fastest_strategy,
        total_time=total_time,
        summary=" | ".join(summary_parts),
        total_cost_usd=round(total_cost, 8),
        request_id=req_id,
    )