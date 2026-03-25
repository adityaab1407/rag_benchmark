import time
from typing import List
from loguru import logger
from groq import Groq
import instructor
from sentence_transformers import CrossEncoder
from app.config import settings
from app.models import RAGResponse, RAGResult, TokenUsage, calculate_cost
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_store import BM25Store


class RerankedRAG:
    # Hybrid retrieval + cross-encoder reranking
    # Step 1: Hybrid search fetches top 20 candidates
    # Step 2: Cross-encoder scores each candidate against the query
    # Step 3: Re-sort by cross-encoder score, keep top 5
    # Most expensive but highest quality

    def __init__(self, embedder: Embedder, vector_store: VectorStore, bm25_store: BM25Store):
        self.embedder     = embedder
        self.vector_store = vector_store
        self.bm25_store   = bm25_store
        self.model        = "qwen/qwen3-32b"

        logger.info("Loading cross-encoder model...")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Cross-encoder ready")

        self.client = instructor.from_groq(
            Groq(api_key=settings.groq_api_key),
            mode=instructor.Mode.JSON
        )
        logger.info("RerankedRAG initialised")

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        fetch_k = top_k * 4  # Fetch more candidates for reranking

        # Hybrid retrieval — same as HybridRAG
        query_embedding = self.embedder.embed_single(query)
        vector_results  = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=fetch_k,
        )
        bm25_results = self.bm25_store.search(query, top_k=fetch_k)

        # RRF fusion
        scores = {}
        k = 60
        for rank, chunk in enumerate(vector_results, start=1):
            cid = chunk.get("id", chunk.get("content", "")[:50])
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
        for rank, chunk in enumerate(bm25_results, start=1):
            cid = chunk.get("id", chunk.get("content", "")[:50])
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank)

        all_chunks = {
            c.get("id", c.get("content", "")[:50]): c
            for c in vector_results + bm25_results
        }

        sorted_ids     = sorted(scores, key=lambda x: scores[x], reverse=True)
        candidates     = [all_chunks[cid] for cid in sorted_ids[:fetch_k] if cid in all_chunks]

        # Cross-encoder reranking
        pairs          = [(query, c.get("content", "")[:512]) for c in candidates]
        rerank_scores  = self.cross_encoder.predict(pairs)

        for chunk, score in zip(candidates, rerank_scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return reranked[:top_k]

    def generate(self, query: str, chunks: List[dict]) -> tuple:
        context = "\n\n---\n\n".join([c["content"][:800] for c in chunks])

        response, raw = self.client.chat.completions.create_with_completion(
            model=self.model,
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
                        f"<context>\n{context}\n</context>\n\n"
                        f"Question: {query}\n\n"
                        "Respond with only the JSON answer object."
                    )
                }
            ]
        )

        usage             = raw.usage if hasattr(raw, "usage") and raw.usage else None
        prompt_tokens     = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        cost              = calculate_cost(self.model, prompt_tokens, completion_tokens)

        token_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost,
            model=self.model,
        )

        return response, token_usage

    def run(self, query: str, top_k: int = 5) -> RAGResult:
        logger.info("RerankedRAG running query: {}", query[:60])

        t0               = time.time()
        chunks           = self.retrieve(query, top_k)
        retrieve_latency = round(time.time() - t0, 3)
        logger.info("Retrieved+reranked {} chunks in {}s", len(chunks), retrieve_latency)

        t1                    = time.time()
        response, token_usage = self.generate(query, chunks)
        generate_latency      = round(time.time() - t1, 3)
        logger.info(
            "Generated answer in {}s — {} tokens — ${:.6f}",
            generate_latency, token_usage.total_tokens, token_usage.cost_usd,
        )

        total_latency = round(retrieve_latency + generate_latency, 3)

        top_score = chunks[0].get("rerank_score", 0.0) if chunks else 0.0

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
                "total":    total_latency,
            },
            token_usage=token_usage,
            cost_usd=token_usage.cost_usd,
        )

    async def run_async(self, query: str, top_k: int = 5) -> RAGResult:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, query, top_k)