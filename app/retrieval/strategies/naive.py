import time
from typing import List
from loguru import logger
from groq import Groq
import instructor
from app.config import settings
from app.models import RAGResponse, RAGResult
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore


class NaiveRAG:
    # Simplest possible RAG pipeline
    # Fixed-size chunks + vector search + Groq generation
    # This is the BASELINE - everything else is measured against this

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store
        self.client = instructor.from_groq(
            Groq(api_key=settings.groq_api_key),
            mode=instructor.Mode.JSON
        )
        logger.info("NaiveRAG initialised")

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        # Step 1 - embed the query (same model as chunks)
        # Step 2 - find top_k most similar chunks in Qdrant
        # strategy_filter="fixed_size" = only search naive RAG chunks
        query_embedding = self.embedder.embed_single(query)
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            strategy_filter="fixed_size"
        )
        return results

    def generate(self, query: str, chunks: List[dict]) -> RAGResponse:
        # Build context from retrieved chunks
        context = "\n\n---\n\n".join([c["content"][:800] for c in chunks])

        # Instructor forces Groq to return a RAGResponse object
        # If Groq returns plain text, Instructor retries automatically
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
        # Full pipeline with timing at each step
        logger.info("NaiveRAG running query: {}", query[:60])

        # Time the retrieve step
        t0 = time.time()
        chunks = self.retrieve(query, top_k)
        retrieve_latency = round(time.time() - t0, 3)
        logger.info("Retrieved {} chunks in {}s", len(chunks), retrieve_latency)

        # Time the generate step
        t1 = time.time()
        response = self.generate(query, chunks)
        generate_latency = round(time.time() - t1, 3)
        logger.info("Generated answer in {}s", generate_latency)

        total_latency = round(retrieve_latency + generate_latency, 3)

        return RAGResult(
            strategy="naive",
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
        # Wraps the sync run() method for async/concurrent execution
        # run_in_executor runs sync code in a thread pool
        # This frees the FastAPI event loop to handle other tasks
        # while waiting for Groq to respond
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,       # None = use default thread pool
            self.run,   # sync function to run
            query,      # arg 1
            top_k       # arg 2
        )
        return result

if __name__ == "__main__":
    from app.ingestion.embedder import Embedder
    from app.retrieval.vector_store import VectorStore

    print("Testing NaiveRAG standalone...")
    print()

    embedder = Embedder()
    vector_store = VectorStore()
    rag = NaiveRAG(embedder, vector_store)

    # Tier 1 - should answer confidently
    print("=== TIER 1 - Easy question ===")
    result = rag.run("What is LoRA and how does it reduce trainable parameters?")
    print("Answer:      ", result.answer[:200])
    print("Confidence:  ", result.confidence)
    print("Answerable:  ", result.is_answerable)
    print("Latency:     ", result.latency)
    print()

    # Tier 3 - should abstain
    print("=== TIER 3 - Adversarial question ===")
    result2 = rag.run("What did Elon Musk say about RAG systems in 2024?")
    print("Answer:      ", result2.answer[:200])
    print("Confidence:  ", result2.confidence)
    print("Answerable:  ", result2.is_answerable)
    print("Reasoning:   ", result2.reasoning[:200])