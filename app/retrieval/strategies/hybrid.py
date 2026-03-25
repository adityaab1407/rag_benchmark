import time
from typing import List
from loguru import logger
from groq import Groq
import instructor
from app.config import settings
from app.models import RAGResponse, RAGResult, TokenUsage, calculate_cost
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_store import BM25Store


class HybridRAG:
    # BM25 + vector search combined with Reciprocal Rank Fusion
    # Beats naive on recall because it catches both semantic and keyword matches

    def __init__(self, embedder: Embedder, vector_store: VectorStore, bm25_store: BM25Store):
        self.embedder     = embedder
        self.vector_store = vector_store
        self.bm25_store   = bm25_store
        self.model        = "qwen/qwen3-32b"
        self.client = instructor.from_groq(
            Groq(api_key=settings.groq_api_key),
            mode=instructor.Mode.JSON
        )
        logger.info("HybridRAG initialised")

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        fetch_k = top_k * 2

        # Vector search
        query_embedding = self.embedder.embed_single(query)
        vector_results  = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=fetch_k,
        )

        # BM25 search
        bm25_results = self.bm25_store.search(query, top_k=fetch_k)

        # Reciprocal Rank Fusion
        scores = {}
        k = 60  # RRF constant

        for rank, chunk in enumerate(vector_results, start=1):
            cid = chunk.get("id", chunk.get("content", "")[:50])
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank)

        for rank, chunk in enumerate(bm25_results, start=1):
            cid = chunk.get("id", chunk.get("content", "")[:50])
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank)

        # Merge all chunks into one lookup
        all_chunks = {
            c.get("id", c.get("content", "")[:50]): c
            for c in vector_results + bm25_results
        }

        # Sort by RRF score and return top_k
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        results = []
        for cid in sorted_ids[:top_k]:
            chunk = all_chunks.get(cid, {})
            chunk["rrf_score"] = round(scores[cid], 6)
            results.append(chunk)

        return results

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
        total_tokens      = getattr(usage, "total_tokens", 0) or 0
        cost              = calculate_cost(self.model, prompt_tokens, completion_tokens)

        token_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            model=self.model,
        )

        return response, token_usage

    def run(self, query: str, top_k: int = 5) -> RAGResult:
        logger.info("HybridRAG running query: {}", query[:60])

        t0               = time.time()
        chunks           = self.retrieve(query, top_k)
        retrieve_latency = round(time.time() - t0, 3)
        logger.info("Retrieved {} chunks in {}s", len(chunks), retrieve_latency)

        t1                    = time.time()
        response, token_usage = self.generate(query, chunks)
        generate_latency      = round(time.time() - t1, 3)
        logger.info(
            "Generated answer in {}s — {} tokens — ${:.6f}",
            generate_latency, token_usage.total_tokens, token_usage.cost_usd,
        )

        total_latency = round(retrieve_latency + generate_latency, 3)

        return RAGResult(
            strategy="hybrid",
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