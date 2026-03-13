import time
from typing import List
from loguru import logger
from groq import Groq
import instructor
from sentence_transformers import CrossEncoder
from app.config import settings
from app.models import RAGResponse, RAGResult
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_store import BM25Store


class RerankedRAG:
    # Hybrid search + cross-encoder reranking
    # The most accurate strategy - best quality, slightly slower
    #
    # WHY RERANKING WORKS:
    # Vector search: compares query embedding vs chunk embedding
    #   Fast but indirect - compares 384 numbers vs 384 numbers
    #   Loses nuance in compression to 384 dims
    #
    # Cross-encoder: reads FULL query + FULL chunk together
    #   "Is this specific chunk relevant to this specific query?"
    #   Much more accurate - no compression loss
    #   But slow - can't precompute, must run at query time
    #
    # SOLUTION - two stage:
    # Stage 1: hybrid search gets top 20 candidates (fast)
    # Stage 2: cross-encoder scores all 20 against query (accurate)
    # Return top 5 by cross-encoder score
    #
    # Best of both worlds: speed of vector + accuracy of cross-encoder

    def __init__(self, embedder: Embedder, vector_store: VectorStore, bm25_store: BM25Store):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_store = bm25_store

        # Cross-encoder runs locally - no API key needed
        # Downloads ~80MB once, cached forever
        logger.info("Loading cross-encoder model...")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Cross-encoder ready")

        self.client = instructor.from_groq(
            Groq(api_key=settings.groq_api_key),
            mode=instructor.Mode.JSON
        )
        logger.info("RerankedRAG initialised")

    def reciprocal_rank_fusion(self, vector_results: List[dict], bm25_results: List[dict], k: int = 60) -> List[dict]:
        scores = {}
        contents = {}

        for rank, result in enumerate(vector_results):
            doc_id = result["content"][:100]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + k)
            contents[doc_id] = result

        for rank, result in enumerate(bm25_results):
            doc_id = result["content"][:100]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + k)
            if doc_id not in contents:
                contents[doc_id] = result

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        merged = []
        for doc_id in sorted_ids:
            result = contents[doc_id].copy()
            result["rrf_score"] = round(scores[doc_id], 6)
            merged.append(result)

        return merged

    def rerank(self, query: str, candidates: List[dict], top_k: int = 5) -> List[dict]:
        # Cross-encoder scores each candidate against the query
        # Input: list of [query, chunk_text] pairs
        # Output: relevance score for each pair
        pairs = [[query, c["content"]] for c in candidates]
        scores = self.cross_encoder.predict(pairs)

        # Attach cross-encoder score to each chunk
        for i, chunk in enumerate(candidates):
            chunk["cross_encoder_score"] = float(scores[i])
            chunk["retrieval_method"] = "reranked"

        # Sort by cross-encoder score - highest first
        reranked = sorted(candidates, key=lambda x: x["cross_encoder_score"], reverse=True)

        logger.info(
            "Reranked {} candidates -> top score: {:.3f}, bottom score: {:.3f}",
            len(candidates),
            reranked[0]["cross_encoder_score"],
            reranked[-1]["cross_encoder_score"]
        )

        return reranked[:top_k]

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        # Fetch more candidates than needed for reranking
        fetch_k = top_k * 4

        # Stage 1 - hybrid search for candidates
        query_embedding = self.embedder.embed_single(query)
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            strategy_filter="fixed_size"
        )
        bm25_results = self.bm25_store.search(query, top_k=fetch_k)

        logger.info("Vector: {} results, BM25: {} results", len(vector_results), len(bm25_results))

        candidates = self.reciprocal_rank_fusion(vector_results, bm25_results)
        logger.info("RRF merged: {} candidates", len(candidates))

        # Stage 2 - cross-encoder reranking
        final = self.rerank(query, candidates, top_k)
        return final

    def generate(self, query: str, chunks: List[dict]) -> RAGResponse:
        context = "\n\n---\n\n".join([c["content"] for c in chunks])

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_model=RAGResponse,
            max_retries=3,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise question answering system. "
                        "Answer ONLY using the provided context. "
                        "If the context does not contain enough information "
                        "to answer confidently, set is_answerable to false "
                        "and confidence below 0.3. "
                        "Never make up information not present in the context."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Context:\n{}\n\n"
                        "Question: {}\n\n"
                        "Answer the question using only the context above."
                    ).format(context, query)
                }
            ]
        )
        return response

    def run(self, query: str, top_k: int = 5) -> RAGResult:
        logger.info("RerankedRAG running query: {}", query[:60])

        t0 = time.time()
        chunks = self.retrieve(query, top_k)
        retrieve_latency = round(time.time() - t0, 3)
        logger.info("Retrieved and reranked {} chunks in {}s", len(chunks), retrieve_latency)

        t1 = time.time()
        response = self.generate(query, chunks)
        generate_latency = round(time.time() - t1, 3)
        logger.info("Generated answer in {}s", generate_latency)

        total_latency = round(retrieve_latency + generate_latency, 3)

        return RAGResult(
            strategy="reranked",
            query=query,
            answer=response.answer,
            confidence=response.confidence,
            is_answerable=response.is_answerable,
            reasoning=response.reasoning,
            retrieved_chunks=chunks,
            latency={
                "retrieve": retrieve_latency,
                "generate": generate_latency,
                "total": total_latency
            }
        )

    async def run_async(self, query: str, top_k: int = 5) -> RAGResult:
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.run,
            query,
            top_k
        )
        return result

if __name__ == "__main__":
    from app.ingestion.embedder import Embedder
    from app.retrieval.vector_store import VectorStore
    from app.retrieval.bm25_store import BM25Store

    print("Testing RerankedRAG standalone...")
    print()

    embedder = Embedder()
    vector_store = VectorStore()
    bm25_store = BM25Store()
    bm25_store.load_index()
    rag = RerankedRAG(embedder, vector_store, bm25_store)

    # Tier 2 - complex question reranker should excel at
    print("=== TIER 2 - Complex question ===")
    result = rag.run("How does attention mechanism scale with sequence length and what are the memory implications?")
    print("Answer:    ", result.answer[:200])
    print("Confidence:", result.confidence)
    print("Answerable:", result.is_answerable)
    print("Latency:   ", result.latency)
    print()

    # Tier 3 - adversarial
    print("=== TIER 3 - Adversarial ===")
    result2 = rag.run("What did Elon Musk say about RAG systems in 2024?")
    print("Answer:    ", result2.answer[:200])
    print("Confidence:", result2.confidence)
    print("Answerable:", result2.is_answerable)