from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Optional
from loguru import logger
from app.config import settings
import uuid


class VectorStore:

    def __init__(self):
        if settings.qdrant_host:
            self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
            logger.info("Qdrant connected - server {}:{}", settings.qdrant_host, settings.qdrant_port)
        else:
            self.client = QdrantClient(path=settings.qdrant_path)
            logger.info("Qdrant connected - local path: {}", settings.qdrant_path)
        self.collection_name = settings.collection_name

    def create_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing:
            logger.info("Collection {} already exists - skipping", self.collection_name)
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dimension,
                distance=Distance.COSINE
            )
        )
        logger.info("Created collection: {}", self.collection_name)

    def store_chunks(self, chunks, embeddings: List[List[float]]):
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "content": chunk.content,
                    "strategy": chunk.strategy,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata
                }
            ))
        batch_size = 100
        total_batches = (len(points) // batch_size) + 1
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            batch_num = (i // batch_size) + 1
            logger.info("Stored batch {}/{} ({} chunks)", batch_num, total_batches, len(batch))
        logger.info("Total stored: {} chunks", len(points))

    def search(self, query_embedding: List[float], top_k: int = 5, strategy_filter: Optional[str] = None):
        query_filter = None
        if strategy_filter:
            query_filter = Filter(
                must=[FieldCondition(
                    key="strategy",
                    match=MatchValue(value=strategy_filter)
                )]
            )
        results = self.client.query_points(
    collection_name=self.collection_name,
    query=query_embedding,
    limit=top_k,
    query_filter=query_filter,
    with_payload=True
).points
        return [
            {
                "content": r.payload["content"],
                "score": r.score,
                "metadata": {k: v for k, v in r.payload.items() if k != "content"},
                "id": r.id
            }
            for r in results
        ]

    def count(self) -> int:
        result = self.client.count(collection_name=self.collection_name)
        return result.count