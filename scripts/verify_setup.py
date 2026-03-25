import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from loguru import logger
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_store import BM25Store


def verify():
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            logger.info("  PASS - {} {}", name, detail)
            passed += 1
        else:
            logger.error("  FAIL - {} {}", name, detail)
            failed += 1

    logger.info("Running setup verification...")

    check("Qdrant storage folder", Path("./qdrant_storage").exists(), "(./qdrant_storage/)")

    store = None
    try:
        store = VectorStore()
        count = store.count()
        check("Qdrant chunk count", count > 1000, "({} chunks)".format(count))
    except Exception as e:
        check("Qdrant connection", False, str(e))

    check("BM25 index file", Path("./bm25_index.pkl").exists(), "(./bm25_index.pkl)")

    try:
        bm25 = BM25Store()
        loaded = bm25.load_index()
        check("BM25 index loads", loaded, "({} chunks)".format(len(bm25.chunks)))
    except Exception as e:
        check("BM25 index loads", False, str(e))

    golden_path = Path("data/golden_dataset.json")
    check("Golden dataset file", golden_path.exists())

    if golden_path.exists():
        golden = json.loads(golden_path.read_text())
        questions = golden.get("questions", golden if isinstance(golden, list) else [])
        check("Golden dataset questions", len(questions) >= 10, "({} questions)".format(len(questions)))

    raw_files = list(Path("data/raw").rglob("*.*"))
    check("Raw documents", len(raw_files) > 0, "({} files)".format(len(raw_files)))

    try:
        bm25 = BM25Store()
        bm25.load_index()
        results = bm25.search("LoRA fine-tuning", top_k=3)
        check("BM25 search works", len(results) > 0, "({} results)".format(len(results)))
    except Exception as e:
        check("BM25 search works", False, str(e))

    try:
        from app.ingestion.embedder import Embedder
        embedder = Embedder()
        embedding = embedder.embed_single("what is LoRA")
        if store is None:
            store = VectorStore()
        results = store.search(embedding, top_k=3)
        check("Qdrant search works", len(results) > 0, "({} results)".format(len(results)))
    except Exception as e:
        check("Qdrant search works", False, str(e))

    logger.info("")
    logger.info("Results: {} passed, {} failed", passed, failed)

    if failed == 0:
        logger.info("All checks passed! Ready to run.")
    else:
        logger.warning("{} checks failed - fix before continuing", failed)


if __name__ == "__main__":
    verify()