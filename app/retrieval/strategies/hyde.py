import time
from typing import List
from loguru import logger
from groq import Groq
import instructor
from app.config import settings
from app.models import RAGResponse, RAGResult
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore


class HyDeRAG:
    # Hypothetical Document Embeddings
    #
    # THE PROBLEM WITH NAIVE RAG:
    # Query: "What is LoRA?"
    # Query embedding lives in "question space"
    # Document chunks live in "answer space"
    # These spaces don't perfectly overlap
    # So vector similarity is imperfect
    #
    # THE HYDE INSIGHT:
    # Instead of embedding the question, generate a
    # hypothetical answer first, then embed THAT
    # "LoRA is a parameter efficient fine-tuning method..."
    # This hypothetical answer lives in "answer space"
    # Much closer to real document chunks
    # Better retrieval on complex questions
    #
    # TWO Groq calls per query:
    # Call 1: generate hypothetical answer (no retrieval)
    # Call 2: generate real answer from retrieved chunks

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store
        self.client = instructor.from_groq(
            Groq(api_key=settings.groq_api_key),
            mode=instructor.Mode.JSON
        )
        logger.info("HyDeRAG initialised")

    def generate_hypothesis(self, query: str) -> str:
        # Call 1 - generate a hypothetical answer
        # No retrieval yet - just ask the LLM to imagine an answer
        # We embed THIS instead of the original query
        from pydantic import BaseModel

        class Hypothesis(BaseModel):
            hypothetical_answer: str

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                response_model=Hypothesis,
                max_retries=3,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Generate a hypothetical passage that would answer "
                            "the question if it existed in a technical document. "
                            "Write 2-3 sentences as if from a research paper or documentation. "
                            "Be specific and technical."
                        )
                    },
                    {
                        "role": "user",
                        "content": "Question: {}".format(query)
                    }
                ]
            )
            hypothesis = response.hypothetical_answer
            logger.info("Hypothesis generated: {}...", hypothesis[:80])
            return hypothesis

        except Exception as e:
            logger.warning("Hypothesis generation failed: {} - falling back to query", e)
            return query

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        # Generate hypothesis then embed it
        hypothesis = self.generate_hypothesis(query)

        # Embed the hypothesis not the original query
        hypothesis_embedding = self.embedder.embed_single(hypothesis)

        results = self.vector_store.search(
            query_embedding=hypothesis_embedding,
            top_k=top_k,
            strategy_filter="fixed_size"
        )

        # Tag chunks with retrieval method
        for r in results:
            r["retrieval_method"] = "hyde"

        return results

    def generate(self, query: str, chunks: List[dict]) -> RAGResponse:
        # Call 2 - generate real answer from retrieved chunks
        # Same as naive from here - only retrieval step was different
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
        logger.info("HyDeRAG running query: {}", query[:60])

        t0 = time.time()
        chunks = self.retrieve(query, top_k)
        retrieve_latency = round(time.time() - t0, 3)
        logger.info("Retrieved {} chunks in {}s (includes hypothesis generation)", len(chunks), retrieve_latency)

        t1 = time.time()
        response = self.generate(query, chunks)
        generate_latency = round(time.time() - t1, 3)
        logger.info("Generated answer in {}s", generate_latency)

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

    print("Testing HyDeRAG standalone...")
    print()

    embedder = Embedder()
    vector_store = VectorStore()
    rag = HyDeRAG(embedder, vector_store)

    # Tier 2 - complex question where HyDE should shine
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