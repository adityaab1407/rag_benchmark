from fastapi import APIRouter, HTTPException
from loguru import logger
from app.models import QueryRequest, RAGResult
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