from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_store import BM25Store
from app.retrieval.strategies.naive import NaiveRAG
from app.database import init_db

# Shared state dictionary
# Components loaded once at startup, reused on every request
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP - runs once when server starts
    logger.info("Starting up RAG Benchmark API...")

    # Init database
    await init_db()
    logger.info("Database ready")

    # Load embedding model - slow, do it once
    embedder = Embedder()
    app_state["embedder"] = embedder
    logger.info("Embedder ready")

    # Connect to Qdrant
    vector_store = VectorStore()
    app_state["vector_store"] = vector_store
    logger.info("Vector store ready")

    # Load BM25 index
    bm25_store = BM25Store()
    loaded = bm25_store.load_index()
    if not loaded:
        logger.warning("BM25 index not found - run ingest.py first")
    app_state["bm25_store"] = bm25_store
    logger.info("BM25 store ready")

    # Initialise RAG strategies
    app_state["naive_rag"] = NaiveRAG(embedder, vector_store)
    logger.info("All strategies ready")

    logger.info("Startup complete - API is ready")

    yield  # server runs here

    # SHUTDOWN - runs when server stops
    logger.info("Shutting down...")
    app_state.clear()


app = FastAPI(
    title="RAG Benchmark API",
    description="Multi-strategy RAG comparison engine",
    version="1.0.0",
    lifespan=lifespan
)

# Register routes
from app.api.routes import router
app.include_router(router)