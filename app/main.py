from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_store import BM25Store
from app.retrieval.strategies.naive import NaiveRAG
from app.retrieval.strategies.hybrid import HybridRAG
from app.retrieval.strategies.hyde import HyDeRAG
from app.retrieval.strategies.reranked import RerankedRAG
from app.database import init_db
from app.middleware import ObservabilityMiddleware   # NEW


app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up RAG Benchmark API...")

    await init_db()
    logger.info("Database ready")

    embedder = Embedder()
    app.state.embedder = embedder
    app_state["embedder"] = embedder
    logger.info("Embedder ready")

    vector_store = VectorStore()
    app.state.vector_store = vector_store
    app_state["vector_store"] = vector_store
    logger.info("Vector store ready")

    bm25_store = BM25Store()
    loaded = bm25_store.load_index()
    if not loaded:
        logger.warning("BM25 index not found — run ingest.py first")
    app.state.bm25_store = bm25_store
    app_state["bm25_store"] = bm25_store
    logger.info("BM25 store ready")

    app.state.naive_rag    = NaiveRAG(embedder, vector_store)
    app.state.hybrid_rag   = HybridRAG(embedder, vector_store, bm25_store)
    app.state.hyde_rag     = HyDeRAG(embedder, vector_store)
    app.state.reranked_rag = RerankedRAG(embedder, vector_store, bm25_store)

    app_state["naive_rag"]    = app.state.naive_rag
    app_state["hybrid_rag"]   = app.state.hybrid_rag
    app_state["hyde_rag"]     = app.state.hyde_rag
    app_state["reranked_rag"] = app.state.reranked_rag

    logger.info("All 4 strategies ready")
    logger.info("Startup complete — API is ready")

    yield

    logger.info("Shutting down...")
    app_state.clear()


app = FastAPI(
    title="RAG Benchmark API",
    description="Multi-strategy RAG comparison engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(ObservabilityMiddleware)          # NEW

from app.api.routes import router
app.include_router(router, prefix="/api/v1")