from sentence_transformers import SentenceTransformer
from typing import List
from loguru import logger
from app.config import settings


class Embedder:
    # Converts text to vectors (lists of 384 numbers)
    # Runs LOCALLY on your CPU - no API key, no cost, no rate limits
    #
    # Model: all-MiniLM-L6-v2
    # Size: ~90MB downloaded once, cached forever
    # Output: 384 numbers per text
    # Speed: ~1000 chunks/minute on CPU
    #
    # normalize_embeddings=True means every vector has length 1.0
    # This makes cosine similarity = dot product = faster in Qdrant

    def __init__(self):
        logger.info("Loading embedding model: {}", settings.embedding_model)
        self.model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model ready")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # batch_size=32 = process 32 chunks at a time, efficient on CPU
        # show_progress_bar=True = shows progress for 5000+ chunks
        logger.info("Embedding {} texts...", len(texts))
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        logger.info("Embedding complete - shape: {}", embeddings.shape)
        return embeddings.tolist()

    def embed_single(self, text: str) -> List[float]:
        # Used at query time - no progress bar needed for one text
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True
        )
        return embedding[0].tolist()

    def verify(self):
        # Sanity check - call after init to confirm model works
        test = self.embed_single("test sentence")
        dim = len(test)
        magnitude = sum(x**2 for x in test) ** 0.5

        assert dim == settings.embedding_dimension, "Expected {} dims, got {}".format(
            settings.embedding_dimension, dim
        )
        assert abs(magnitude - 1.0) < 0.001, "Vector not normalised - magnitude: {}".format(magnitude)

        logger.info("Embedder verified: {} dims, magnitude={:.4f}", dim, magnitude)
        return True
