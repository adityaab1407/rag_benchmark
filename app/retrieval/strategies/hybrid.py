import time
from typing import List
from loguru import logger
from groq import Groq
import instructor
from app.config import settings
from app.models import RAGResponse, RAGResult
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_store import BM25Store


class HybridRAG:
    # Combines BM25 keyword search + vector semantic search
    # Uses Reciprocal Rank Fusion (RRF) to merge results
    #
    # WHY HYBRID BEATS NAIVE:
    # Naive (vector only): finds semantically similar chunks
    #   "LoRA rank 16" -> finds chunks about fine-tuning in general
    # BM25 (keyword only): finds exact term matches
    #   "LoRA rank 16" -> finds chunks containing "LoRA", "rank", "16"
    # Hybrid: gets both - semantic AND exact match
    #
    # RRF formula: score = 1 / (rank + 60)
    # A chunk ranked #1 in both lists scores highest
    # A chunk only in one list scores lower
    # 60 is a constant that prevents top ranks from dominating

    def __init__(self, embedder: Embedder, vector_store: VectorStore, bm25_store: BM25Store):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.client = instructor.from_groq(
            Groq(api_key=settings.groq_api_key),
            mode=instructor.Mode.JSON
        )
        logger.info("HybridRAG initialised")

    def reciprocal_rank_fusion(self, vector_results: List[dict], bm25_results: List[dict], k: int = 60) -> List[dict]:
        # Merge two ranked lists into one using RRF
        # k=60 is standard RRF constant from the original paper
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
            result["retrieval_method"] = "hybrid_rrf"
            merged.append(result)

        return merged

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        # Get more candidates than needed from each retriever
        # Then RRF merges and reranks them
        fetch_k = top_k * 2

        # Vector search - semantic similarity
        query_embedding = self.embedder.embed_single(query)
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            strategy_filter="fixed_size"
        )

        # BM25 search - keyword matching
        bm25_results = self.bm25_store.search(query, top_k=fetch_k)

        logger.info("Vector: {} results, BM25: {} results", len(vector_results), len(bm25_results))

        # Merge with RRF
        merged = self.reciprocal_rank_fusion(vector_results, bm25_results)

        # Return top_k after fusion
        return merged[:top_k]

    def generate(self, query: str, chunks: List[dict]) -> RAGResponse:
        context = "\n\n---\n\n".join([c["content"][:800] for c in chunks])

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
        logger.info("HybridRAG running query: {}", query[:60])

        t0 = time.time()
        chunks = self.retrieve(query, top_k)
        retrieve_latency = round(time.time() - t0, 3)
        logger.info("Retrieved {} chunks in {}s", len(chunks), retrieve_latency)

        t1 = time.time()
        response = self.generate(query, chunks)
        generate_latency = round(time.time() - t1, 3)
        logger.info("Generated answer in {}s", generate_latency)

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

    print("Testing HybridRAG standalone...")
    print()

    embedder = Embedder()
    vector_store = VectorStore()
    bm25_store = BM25Store()
    bm25_store.load_index()
    rag = HybridRAG(embedder, vector_store, bm25_store)

    # Tier 1 - technical term query where BM25 helps
    print("=== TIER 1 - Technical term query ===")
    result = rag.run("What is the rank r in LoRA and how do you choose it?")
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