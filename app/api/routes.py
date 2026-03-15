import time
import asyncio
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from loguru import logger

from app.models import (
    QueryRequest, BenchmarkRequest,
    RAGResult, StrategyComparison, BenchmarkResult,
)
from app.database import log_query, log_observability, save_feedback
from app.middleware import get_request_id

router = APIRouter()


# =============================================================================
# FEEDBACK MODEL
# =============================================================================

class FeedbackRequest(BaseModel):
    request_id: str = ""
    strategy:   str
    question:   str
    answer:     str
    rating:     int   # +1 thumbs up, -1 thumbs down
    comment:    str = ""


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
    app_state = request.app.state
    req_id    = get_request_id()
    strategy  = body.strategy
    top_k     = body.top_k or 5

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
            req_id, strategy,
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
# BENCHMARK — ALL 4 IN PARALLEL
# =============================================================================

@router.post("/benchmark")
async def benchmark(request: Request, body: BenchmarkRequest):
    app_state = request.app.state
    req_id    = get_request_id()
    top_k     = body.top_k or 5

    t_start = time.time()

    results = await asyncio.gather(
        app_state.naive_rag.run_async(body.question, top_k),
        app_state.hybrid_rag.run_async(body.question, top_k),
        app_state.hyde_rag.run_async(body.question, top_k),
        app_state.reranked_rag.run_async(body.question, top_k),
        return_exceptions=True,
    )

    total_time  = round(time.time() - t_start, 3)
    comparisons = []
    total_cost  = 0.0

    for strategy_name, result in zip(["naive", "hybrid", "hyde", "reranked"], results):
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
        top_score   = result.retrieved_chunks[0].get("score", 0.0) if result.retrieved_chunks else 0.0

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

    answered         = [c for c in comparisons if c.is_answerable and not c.answer.startswith("ERROR")]
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
                f"{c.strategy}: confidence={c.confidence} "
                f"latency={c.latency.get('total',0):.3f}s "
                f"cost=${c.cost_usd:.6f}"
            )

    logger.info("[{}] /benchmark — {}s — ${:.6f}", req_id, total_time, total_cost)

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


# =============================================================================
# FEEDBACK
# =============================================================================

@router.post("/feedback")
async def feedback(request: Request, body: FeedbackRequest):
    if body.rating not in [1, -1]:
        raise HTTPException(status_code=400, detail="rating must be 1 or -1")

    req_id = get_request_id()

    await save_feedback(
        request_id=body.request_id or req_id,
        strategy=body.strategy,
        question=body.question,
        answer=body.answer,
        rating=body.rating,
        comment=body.comment,
    )

    logger.info(
        "[{}] Feedback: {} {} — {}",
        req_id, body.strategy,
        "👍" if body.rating == 1 else "👎",
        body.question[:40],
    )

    return {
        "status":   "saved",
        "strategy": body.strategy,
        "rating":   body.rating,
    }