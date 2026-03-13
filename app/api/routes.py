from fastapi import APIRouter, HTTPException
from loguru import logger
from app.models import QueryRequest, RAGResult
from app.main import app_state

router = APIRouter(prefix="/api/v1")


@router.get("/health")
async def health():
    # Quick check that server and components are alive
    return {
        "status": "healthy",
        "components": {
            "embedder": "embedder" in app_state,
            "vector_store": "vector_store" in app_state,
            "bm25_store": "bm25_store" in app_state,
            "naive_rag": "naive_rag" in app_state,
        }
    }


@router.get("/strategies")
async def list_strategies():
    # Lists all available RAG strategies
    # More will be added as Days 3-5 complete
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
                "status": "coming soon"
            },
            {
                "name": "hyde",
                "description": "Hypothetical document embeddings.",
                "status": "coming soon"
            },
            {
                "name": "reranked",
                "description": "Hybrid search + cross-encoder reranking.",
                "status": "coming soon"
            }
        ]
    }


@router.post("/query", response_model=RAGResult)
async def query(request: QueryRequest):
    # Main endpoint - runs RAG pipeline and returns structured result
    logger.info("Query received: strategy={} question={}", request.strategy, request.question[:60])

    if request.strategy == "naive":
        if "naive_rag" not in app_state:
            raise HTTPException(status_code=503, detail="Naive RAG not initialised")
        result = app_state["naive_rag"].run(request.question, request.top_k or 5)

    else:
        raise HTTPException(
            status_code=400,
            detail="Strategy '{}' not available yet. Use: naive".format(request.strategy)
        )

    logger.info(
        "Query complete: confidence={} latency={}s",
        result.confidence,
        result.latency["total"]
    )
    return result