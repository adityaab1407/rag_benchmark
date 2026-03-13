import time
import asyncio
from fastapi import APIRouter, HTTPException
from loguru import logger
from app.models import QueryRequest, RAGResult, BenchmarkRequest, BenchmarkResult, StrategyComparison
from app.main import app_state

router = APIRouter(prefix="/api/v1")

AVAILABLE_STRATEGIES = {
    "naive":    "naive_rag",
    "hybrid":   "hybrid_rag",
    "hyde":     "hyde_rag",
    "reranked": "reranked_rag",
}


@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "components": {
            "embedder":     "embedder" in app_state,
            "vector_store": "vector_store" in app_state,
            "bm25_store":   "bm25_store" in app_state,
            "naive_rag":    "naive_rag" in app_state,
            "hybrid_rag":   "hybrid_rag" in app_state,
            "hyde_rag":     "hyde_rag" in app_state,
            "reranked_rag": "reranked_rag" in app_state,
        }
    }


@router.get("/strategies")
async def list_strategies():
    return {
        "strategies": [
            {
                "name": "naive",
                "description": "Fixed-size chunks + vector search. Baseline strategy.",
                "status": "available"
            },
            {
                "name": "hybrid",
                "description": "BM25 + vector search with RRF fusion.",
                "status": "available"
            },
            {
                "name": "hyde",
                "description": "Hypothetical document embeddings for better complex query retrieval.",
                "status": "available"
            },
            {
                "name": "reranked",
                "description": "Hybrid search + cross-encoder reranking. Best quality.",
                "status": "available"
            }
        ]
    }


@router.post("/query", response_model=RAGResult)
async def query(request: QueryRequest):
    logger.info(
        "Query received: strategy={} question={}",
        request.strategy,
        request.question[:60]
    )

    if request.strategy not in AVAILABLE_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail="Unknown strategy '{}'. Available: {}".format(
                request.strategy,
                list(AVAILABLE_STRATEGIES.keys())
            )
        )

    state_key = AVAILABLE_STRATEGIES[request.strategy]

    if state_key not in app_state:
        raise HTTPException(
            status_code=503,
            detail="{} not initialised - check server logs".format(request.strategy)
        )

    result = app_state[state_key].run(request.question, request.top_k or 5)

    logger.info(
        "Query complete: strategy={} confidence={} latency={}s",
        request.strategy,
        result.confidence,
        result.latency["total"]
    )

    return result


@router.post("/benchmark", response_model=BenchmarkResult)
async def benchmark(request: BenchmarkRequest):
    # THE async endpoint - runs all 4 strategies simultaneously
    # asyncio.gather() is the key - it fires all 4 at once
    # and waits for all to complete
    # Total time = slowest strategy, not sum of all 4
    logger.info("Benchmark received: question={}", request.question[:60])

    # Check all strategies are loaded
    for state_key in AVAILABLE_STRATEGIES.values():
        if state_key not in app_state:
            raise HTTPException(
                status_code=503,
                detail="{} not initialised - check server logs".format(state_key)
            )

    top_k = request.top_k or 5
    wall_clock_start = time.time()

    # THIS is asyncio.gather() in action
    # All 4 run_async() calls start simultaneously
    # Each one is running in its own thread (via run_in_executor)
    # Python doesn't wait for one to finish before starting the next
    # Returns a list of 4 results in the same order as the calls
    logger.info("Launching all 4 strategies in parallel...")
    results = await asyncio.gather(
        app_state["naive_rag"].run_async(request.question, top_k),
        app_state["hybrid_rag"].run_async(request.question, top_k),
        app_state["hyde_rag"].run_async(request.question, top_k),
        app_state["reranked_rag"].run_async(request.question, top_k),
        return_exceptions=True  # if one fails, others still complete
    )

    total_wall_time = round(time.time() - wall_clock_start, 3)
    logger.info("All 4 strategies completed in {}s", total_wall_time)

    # Process results - handle any exceptions gracefully
    comparisons = []
    for i, (strategy_name, result) in enumerate(
        zip(AVAILABLE_STRATEGIES.keys(), results)
    ):
        if isinstance(result, Exception):
            # One strategy failed - still return others
            logger.error("Strategy {} failed: {}", strategy_name, result)
            comparisons.append(StrategyComparison(
                strategy=strategy_name,
                answer="ERROR: {}".format(str(result)[:100]),
                confidence=0.0,
                is_answerable=False,
                latency={"retrieve": 0.0, "generate": 0.0, "total": 0.0},
                top_chunk_score=0.0,
                retrieval_method="error"
            ))
        else:
            # Get top chunk score safely
            top_chunk_score = 0.0
            if result.retrieved_chunks:
                top_chunk_score = result.retrieved_chunks[0].get("score", 0.0)

            # Get retrieval method from first chunk
            retrieval_method = "vector"
            if result.retrieved_chunks:
                retrieval_method = result.retrieved_chunks[0].get(
                    "retrieval_method", strategy_name
                )

            comparisons.append(StrategyComparison(
                strategy=strategy_name,
                answer=result.answer,
                confidence=result.confidence,
                is_answerable=result.is_answerable,
                latency=result.latency,
                top_chunk_score=round(top_chunk_score, 4),
                retrieval_method=retrieval_method
            ))

            logger.info(
                "  {} -> confidence={} latency={}s",
                strategy_name,
                result.confidence,
                result.latency["total"]
            )

    # Find best and fastest from successful results
    successful = [c for c in comparisons if c.confidence > 0]

    best_strategy = max(
        successful,
        key=lambda x: x.confidence
    ).strategy if successful else "none"

    fastest_strategy = min(
        successful,
        key=lambda x: x.latency["total"]
    ).strategy if successful else "none"

    # Build human readable summary
    summary_lines = []
    for c in comparisons:
        if c.confidence > 0:
            summary_lines.append(
                "{}: confidence={} latency={}s".format(
                    c.strategy, c.confidence, c.latency["total"]
                )
            )
        else:
            summary_lines.append("{}: FAILED".format(c.strategy))

    summary = " | ".join(summary_lines)

    logger.info(
        "Benchmark complete: best={} fastest={} wall_time={}s",
        best_strategy,
        fastest_strategy,
        total_wall_time
    )

    return BenchmarkResult(
        query=request.question,
        strategies=comparisons,
        best_strategy=best_strategy,
        fastest_strategy=fastest_strategy,
        total_time=total_wall_time,
        summary=summary
    )