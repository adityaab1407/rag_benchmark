import time
from typing import List
from loguru import logger
from groq import Groq
import instructor
from app.config import settings
from app.models import RAGResponse, RAGResult, TokenUsage, calculate_cost
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore


class HyDeRAG:
    # Hypothetical Document Embeddings
    # Step 1: Ask LLM to generate a hypothetical answer to the question
    # Step 2: Embed that hypothetical answer (not the original query)
    # Step 3: Use that embedding to search — finds chunks similar to a good answer
    # Makes 2 Groq calls per query — both tracked for cost

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder     = embedder
        self.vector_store = vector_store
        self.model        = "qwen/qwen3-32b"
        self.client = instructor.from_groq(
            Groq(api_key=settings.groq_api_key),
            mode=instructor.Mode.JSON
        )
        logger.info("HyDeRAG initialised")

    def generate_hypothesis(self, query: str) -> tuple:
        """
        Call 1: Generate a hypothetical answer.
        Returns (hypothesis_text, TokenUsage)
        """
        # Raw client needed to access usage stats from response
        from groq import Groq as RawGroq
        raw_client = RawGroq(api_key=settings.groq_api_key)

        raw = raw_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Generate a short, factual, "
                        "hypothetical answer to the question as if you had perfect "
                        "knowledge. This will be used to improve document retrieval. "
                        "Be specific and technical. 2-3 sentences max."
                    )
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nGenerate a hypothetical answer:"
                }
            ],
            max_tokens=200,
        )

        hypothesis = raw.choices[0].message.content or ""

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

        logger.info("Hypothesis generated: {}", hypothesis[:80])
        return hypothesis, token_usage

    def retrieve(self, hypothesis: str, top_k: int = 5) -> List[dict]:
        """Embed the hypothesis and search for similar chunks."""
        hypothesis_embedding = self.embedder.embed_single(hypothesis)
        results = self.vector_store.search(
            query_embedding=hypothesis_embedding,
            top_k=top_k,
        )
        return results

    def generate(self, query: str, chunks: List[dict]) -> tuple:
        """
        Call 2: Generate final answer from retrieved chunks.
        Returns (RAGResponse, TokenUsage)
        """
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
        logger.info("HyDeRAG running query: {}", query[:60])

        t0 = time.time()

        # Call 1 — generate hypothesis
        hypothesis, hypothesis_tokens = self.generate_hypothesis(query)

        # Retrieve using hypothesis embedding
        chunks = self.retrieve(hypothesis, top_k)
        retrieve_latency = round(time.time() - t0, 3)
        logger.info(
            "Retrieved {} chunks in {}s (includes hypothesis generation)",
            len(chunks), retrieve_latency
        )

        # Call 2 — generate final answer
        t1 = time.time()
        response, generation_tokens = self.generate(query, chunks)
        generate_latency = round(time.time() - t1, 3)

        # Sum token usage from BOTH calls
        total_token_usage = TokenUsage(
            prompt_tokens=hypothesis_tokens.prompt_tokens + generation_tokens.prompt_tokens,
            completion_tokens=hypothesis_tokens.completion_tokens + generation_tokens.completion_tokens,
            total_tokens=hypothesis_tokens.total_tokens + generation_tokens.total_tokens,
            cost_usd=round(hypothesis_tokens.cost_usd + generation_tokens.cost_usd, 8),
            model=self.model,
        )

        logger.info(
            "Generated answer in {}s — {} tokens total (2 calls) — ${:.6f}",
            generate_latency,
            total_token_usage.total_tokens,
            total_token_usage.cost_usd,
        )

        total_latency = round(retrieve_latency + generate_latency, 3)

        return RAGResult(
            strategy="hyde",
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
            token_usage=total_token_usage,
            cost_usd=total_token_usage.cost_usd,
        )

    async def run_async(self, query: str, top_k: int = 5) -> RAGResult:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, query, top_k)