import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from collections import Counter
from app.ingestion.loader import load_all_documents
from app.ingestion.chunker import chunk_document_all_strategies
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_store import BM25Store


def main():
    logger.info("=" * 60)
    logger.info("RAG BENCHMARK - INGESTION PIPELINE")
    logger.info("=" * 60)

    # Step 1 - Load documents
    logger.info("STEP 1/5 - Loading documents from data/raw/")
    docs = load_all_documents("data/raw")
    if not docs:
        logger.error("No documents found - check data/raw/ folder")
        return
    type_counts = Counter(d.metadata["doc_type"] for d in docs)
    for doc_type, count in type_counts.items():
        logger.info("  {}: {} files", doc_type, count)
    logger.info("  TOTAL: {} documents", len(docs))

    # Step 2 - Chunk documents
    logger.info("STEP 2/5 - Chunking (3 strategies per document)")
    all_chunks = []
    for i, doc in enumerate(docs):
        chunks = chunk_document_all_strategies(doc)
        all_chunks.extend(chunks)
        logger.info("[{}/{}] {} -> {} chunks", i + 1, len(docs), doc.metadata["filename"], len(chunks))

    strategy_counts = Counter(c.strategy for c in all_chunks)
    for strategy, count in strategy_counts.items():
        logger.info("  {}: {}", strategy, count)
    logger.info("  TOTAL: {}", len(all_chunks))

    # Step 3 - Embed
    logger.info("STEP 3/5 - Embedding chunks (runs locally, no API needed)")
    embedder = Embedder()
    embedder.verify()
    texts = [c.content for c in all_chunks]
    embeddings = embedder.embed_texts(texts)
    logger.info("Generated {} embeddings", len(embeddings))

    # Step 4 - Store in Qdrant
    logger.info("STEP 4/5 - Storing in Qdrant")
    store = VectorStore()
    store.create_collection()
    store.store_chunks(all_chunks, embeddings)
    total = store.count()
    logger.info("Qdrant now contains {} chunks", total)

    # Step 5 - Build BM25
    logger.info("STEP 5/5 - Building BM25 keyword index")
    fixed_chunks = [c for c in all_chunks if c.strategy == "fixed_size"]
    bm25_store = BM25Store()
    bm25_store.build_index(fixed_chunks)

    # Summary
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("  Documents:  {}", len(docs))
    logger.info("  Chunks:     {}", len(all_chunks))
    logger.info("  In Qdrant:  {}", total)
    logger.info("  BM25:       {} fixed-size chunks", len(fixed_chunks))
    logger.info("  Qdrant folder:  ./qdrant_storage/")
    logger.info("  BM25 cache:     ./bm25_index.pkl")
    logger.info("=" * 60)
    logger.info("Next step: python scripts/verify_setup.py")


if __name__ == "__main__":
    main()