import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from typing import List
from loguru import logger

BM25_CACHE_PATH = "./bm25_index.pkl"


class BM25Store:

    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def build_index(self, chunks):
        self.chunks = chunks
        tokenised = [chunk.content.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenised)
        self._save_to_disk()
        logger.info("BM25 index built - {} chunks indexed", len(chunks))

    def load_index(self) -> bool:
        cache = Path(BM25_CACHE_PATH)
        if not cache.exists():
            logger.warning("No BM25 cache found - run ingest.py first")
            return False
        with open(BM25_CACHE_PATH, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.chunks = data["chunks"]
        logger.info("BM25 index loaded - {} chunks", len(self.chunks))
        return True

    def _save_to_disk(self):
        with open(BM25_CACHE_PATH, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
        logger.info("BM25 index saved to {}", BM25_CACHE_PATH)

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        if not self.bm25:
            raise ValueError("BM25 index not loaded - call load_index() first")
        tokenised_query = query.lower().split()
        scores = self.bm25.get_scores(tokenised_query)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        return [
            {
                "content": self.chunks[i].content,
                "score": float(scores[i]),
                "metadata": self.chunks[i].metadata,
                "retrieval_method": "bm25"
            }
            for i in top_indices
            if scores[i] > 0
        ]